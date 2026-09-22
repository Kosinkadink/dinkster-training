"""Training nodes execute through an isolated worker over one SQLite ledger."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import types
from importlib.metadata import distribution
from pathlib import Path

import pytest
from dinkster.compose import PackSpec, ServingComposer, training_pack_specs
from dinkster_engine import ExecutionError
from dinkster_graph import Graph, GraphNode
from dinkster_protocol import (
    NodeError,
    TrainingEventName,
    TrainingJournalEvent,
    TrainingSessionHandle,
)
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_worker.backend import create_training_service
from dinkster_training_worker.fake import FakeTrainer
from test_training_pack import (
    CONFIG,
    event_names,
    fold_training_graph,
    names_committed,
    while_training_graph,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_MANIFEST = ROOT / "packages/dinkster-nodes-training/dinkster-pack.toml"
WORKER_MANIFEST = ROOT / "packages/dinkster-training-worker/dinkster-pack.toml"
FOUNDATION_MANIFEST = Path(
    str(
        distribution("dinkster-nodes-foundation").locate_file(
            "dinkster_nodes_foundation_pack/dinkster-pack.toml"
        )
    )
)


def _worker_spec(journal: Path, *, first_step_delay: float = 0.0) -> PackSpec:
    env = {
        "DINKSTER_TRAINING_BACKEND": "fake",
        "DINKSTER_TRAINING_JOURNAL": str(journal),
    }
    if first_step_delay:
        env["DINKSTER_TRAINING_FAKE_FIRST_STEP_DELAY"] = str(first_step_delay)
    return PackSpec(
        WORKER_MANIFEST,
        env=env,
        trust_reserved=True,
    )


async def _compose(journal: Path, *, first_step_delay: float = 0.0) -> ServingComposer:
    composer = ServingComposer()
    owner = await composer.add_pack(PackSpec(SCHEMA_MANIFEST, in_process=True))
    assert owner.schemas == {}
    assert all(
        not arms
        for node_type, arms in composer._topology.items()
        if node_type.startswith("training.")
    )
    executor = await composer.add_pack(_worker_spec(journal, first_step_delay=first_step_delay))
    assert set(executor.schemas) == {
        "training.dry_run",
        "training.create_session",
        "training.advance",
        "training.complete_session",
        "training.export_lora",
    }
    composer.validate_complete_generation()
    return composer


async def _resolved_run(
    composer: ServingComposer, graph: Graph, demands: list[str]
) -> dict[str, dict[str, object]]:
    result = await composer.composition.make_engine(lambda _event: None).run(graph, demands)
    return {
        node_id: {output_id: value.resolve() for output_id, value in outputs.items()}
        for node_id, outputs in result.outputs.items()
    }


def _store(path: Path) -> TrainingSessionStore:
    return TrainingSessionStore(JournalStore(path))


def test_backend_selection_keeps_fake_torch_free_and_loads_lora_backends_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path / "training.sqlite")
    try:
        assert "torch" not in sys.modules
        assert isinstance(create_training_service("fake", store), FakeTrainer)
        assert "torch" not in sys.modules

        selected: dict[str, object] = {}

        class TorchService:
            def __init__(
                self,
                selected_store: TrainingSessionStore,
                checkpoint_root: Path,
                *,
                cancelled: object,
            ) -> None:
                selected.update(
                    store=selected_store,
                    checkpoint_root=checkpoint_root,
                    cancelled=cancelled,
                )

        class SD15TorchService(TorchService):
            pass

        class SDXLTorchService(TorchService):
            pass

        class MiniMaxH3TorchService(TorchService):
            pass

        class MiniMaxMusic3TorchService(TorchService):
            pass

        class WanTorchService(TorchService):
            pass

        class FluxTorchService(TorchService):
            pass

        class Flux2TorchService(TorchService):
            pass

        class QwenImageTorchService(TorchService):
            pass

        class Ideogram4TorchService(TorchService):
            pass

        module = types.ModuleType("dinkster_training_torch")
        module.__dict__["SD15LoRATrainingService"] = SD15TorchService
        module.__dict__["SDXLLoRATrainingService"] = SDXLTorchService
        module.__dict__["MiniMaxH3LoRATrainingService"] = MiniMaxH3TorchService
        module.__dict__["MiniMaxMusic3LoRATrainingService"] = MiniMaxMusic3TorchService
        module.__dict__["WanLoRATrainingService"] = WanTorchService
        module.__dict__["FluxLoRATrainingService"] = FluxTorchService
        module.__dict__["Flux2LoRATrainingService"] = Flux2TorchService
        module.__dict__["QwenImageLoRATrainingService"] = QwenImageTorchService
        module.__dict__["Ideogram4LoRATrainingService"] = Ideogram4TorchService
        monkeypatch.setitem(sys.modules, "dinkster_training_torch", module)
        for backend, service_type in (
            ("sd15-lora", SD15TorchService),
            ("sdxl-lora", SDXLTorchService),
            ("minimax-h3-lora", MiniMaxH3TorchService),
            ("minimax-music3-lora", MiniMaxMusic3TorchService),
            ("wan-lora", WanTorchService),
            ("flux-lora", FluxTorchService),
            ("flux2-lora", Flux2TorchService),
            ("qwen-image-lora", QwenImageTorchService),
            ("ideogram4-lora", Ideogram4TorchService),
        ):
            selected.clear()
            service = create_training_service(
                backend,
                store,
                environment={"DINKSTER_TRAINING_CHECKPOINT_ROOT": str(tmp_path / "checkpoints")},
            )
            assert isinstance(service, service_type)
            assert selected["store"] is store
            assert selected["checkpoint_root"] == tmp_path / "checkpoints"
            assert callable(selected["cancelled"])
    finally:
        store.close()


def _write_policy_pack(root: Path) -> Path:
    root.mkdir()
    (root / "training_worker_policy.py").write_text(
        "from dinkster_api.v1 import CORE_BOOLEAN, CORE_INT, InputSpec, Node, NodeSchema, "
        "OutputSpec, TypeExpr\n"
        "INT = TypeExpr.concrete(CORE_INT)\n"
        "BOOL = TypeExpr.concrete(CORE_BOOLEAN)\n"
        "class StepBelow(Node):\n"
        "    @classmethod\n"
        "    def define_schema(cls):\n"
        "        return NodeSchema(\n"
        "            node_type='test.step_below',\n"
        "            inputs=(InputSpec('step_cursor', INT), InputSpec('target', INT)),\n"
        "            outputs=(OutputSpec('below', BOOL),),\n"
        "        )\n"
        "    @classmethod\n"
        "    def execute(cls, step_cursor, target):\n"
        "        return cls.outputs(below=step_cursor < target)\n"
        "NODES = (StepBelow,)\n",
        encoding="utf-8",
    )
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname = "training-worker-policy"\nnamespaces = ["test"]\n\n'
        '[pack.entry]\nnodes = "training_worker_policy:NODES"\n',
        encoding="utf-8",
    )
    return manifest


def test_fold_and_while_loops_commit_across_process_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        journal = tmp_path / "training.sqlite"
        store = _store(journal)
        composer = await _compose(journal)
        try:
            worker = composer._records["dinkster-training-worker"].worker
            assert worker._proc is not None
            assert worker._proc.pid != os.getpid()

            outputs = await _resolved_run(
                composer,
                fold_training_graph("worker-fold", [1, 2, 3]),
                ["loop"],
            )
            handle = outputs["loop"]["handle"]
            assert isinstance(handle, TrainingSessionHandle)
            assert handle.step_cursor == 6
            checkpoints = store.list_checkpoints(handle.session_id)
            assert [checkpoint.step_cursor for checkpoint in checkpoints] == [0, 1, 3, 6]
            assert handle.checkpoint_manifest_digest == checkpoints[-1].manifest_digest
            assert names_committed(store, handle.session_id) == 3

            await composer.add_pack(
                PackSpec(_write_policy_pack(tmp_path / "policy"), in_process=True)
            )
            while_outputs = await _resolved_run(
                composer,
                while_training_graph("worker-while", target=6, steps_per_advance=2),
                ["loop"],
            )
            while_handle = while_outputs["loop"]["handle"]
            assert isinstance(while_handle, TrainingSessionHandle)
            assert while_handle.step_cursor == 6
            assert names_committed(store, while_handle.session_id) == 3
        finally:
            await composer.close()
            store.close()

    asyncio.run(scenario())


def test_dry_run_capabilities_cross_process_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        journal = tmp_path / "training.sqlite"
        store = _store(journal)
        composer = await _compose(journal)
        try:
            graph = Graph(nodes={"dry": GraphNode("training.dry_run", {"config": CONFIG})})
            outputs = await _resolved_run(composer, graph, ["dry"])
            report = json.loads(str(outputs["dry"]["report"]))
            assert report["trainer"] == "fake-trainer/1"
            assert report["capabilities"] == {
                "autograd": False,
                "checkpointResume": True,
                "families": [],
                "safePointCancellation": True,
            }
            assert store.get_session(FakeTrainer._session_id("unused")) is None
        finally:
            await composer.close()
            store.close()

    asyncio.run(scenario())


def test_committed_graph_replays_without_new_steps(tmp_path: Path) -> None:
    async def scenario() -> None:
        journal = tmp_path / "training.sqlite"
        store = _store(journal)
        composer = await _compose(journal)
        try:
            graph = fold_training_graph("worker-replay", [1, 2, 3])
            first = (await _resolved_run(composer, graph, ["loop"]))["loop"]["handle"]
            assert isinstance(first, TrainingSessionHandle)
            events_after_first = event_names(store, first.session_id)

            second = (await _resolved_run(composer, graph, ["loop"]))["loop"]["handle"]
            assert second == first
            assert event_names(store, first.session_id) == events_after_first
            assert names_committed(store, first.session_id) == 3
        finally:
            await composer.close()
            store.close()

    asyncio.run(scenario())


def test_killed_worker_restarts_from_recovery_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        journal = tmp_path / "training.sqlite"
        store = _store(journal)
        composer = await _compose(journal, first_step_delay=5.0)
        target = 50
        session_key = "worker-recovery"
        session_id = FakeTrainer._session_id(session_key)
        graph = fold_training_graph(session_key, [target])
        try:
            worker = composer._records["dinkster-training-worker"].worker
            process = worker._proc
            assert process is not None
            running = asyncio.create_task(
                composer.composition.make_engine(lambda _event: None).run(graph, ["loop"])
            )

            recovery = []
            interrupted = False
            for _ in range(2_000):
                recovery = [
                    record
                    for record in store.read_events(session_id, after=0, limit=1_000).records
                    if record.name == TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED
                ]
                session = store.get_session(session_id)
                if recovery and session is not None and session.committed_step_cursor == 0:
                    interrupted = True
                    break
                await asyncio.sleep(0.001)
            assert interrupted, "worker did not pause after publishing a recovery checkpoint"

            process.kill()
            await process.wait()
            with pytest.raises(ExecutionError) as caught:
                await running
            assert isinstance(caught.value.error, NodeError)
            assert "dinkster-training-worker" in caught.value.error.message

            operation_id = str(recovery[-1].payload["advanceId"])
            operation = store.get_operation(session_id, operation_id)
            assert operation is not None
            assert operation.status == "started"
            assert operation.recovery_checkpoint_digest

            await composer.reload_pack("dinkster-training-worker")
            outputs = await _resolved_run(composer, graph, ["loop"])
            handle = outputs["loop"]["handle"]
            assert isinstance(handle, TrainingSessionHandle)
            assert handle.step_cursor == target

            recovery_events = [
                TrainingJournalEvent.from_wire(record.payload)
                for record in store.read_events(session_id, after=0, limit=1_000).records
                if record.name == TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED
            ]
            expected = [
                FakeTrainer._manifest_digest(
                    session_id,
                    FakeTrainer._config_digest(CONFIG),
                    step,
                )
                for step in range(1, target + 1)
            ]
            assert [event.data["manifestDigest"] for event in recovery_events] == expected
            assert [
                checkpoint.step_cursor for checkpoint in store.list_checkpoints(session_id)
            ] == [
                0,
                target,
            ]
            assert names_committed(store, session_id) == 1
        finally:
            await composer.close()
            store.close()

    asyncio.run(scenario())


def test_training_packs_compose_beside_foundation(tmp_path: Path) -> None:
    """The training packs must compose in one composition with the serve
    default set. The worker claims no namespace (a defaulted pack-name
    claim would nest under foundation's reserved "dinkster" root), so
    composing the packs alone cannot prove the served startup path - this
    is the composition dinkster-serve actually builds."""

    async def scenario() -> None:
        composer = ServingComposer()
        try:
            foundation = await composer.add_pack(
                PackSpec(FOUNDATION_MANIFEST, in_process=True, trust_reserved=True)
            )
            assert foundation.schemas
            owner = await composer.add_pack(PackSpec(SCHEMA_MANIFEST, in_process=True))
            assert owner.schemas == {}
            executor = await composer.add_pack(_worker_spec(tmp_path / "training.sqlite"))
            assert set(executor.schemas) == {
                "training.dry_run",
                "training.create_session",
                "training.advance",
                "training.complete_session",
                "training.export_lora",
            }
            composer.validate_complete_generation()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_serve_specs_share_one_training_journal(tmp_path: Path) -> None:
    journal = tmp_path / "training.sqlite"
    owner, executor = training_pack_specs(journal)
    assert owner.in_process
    assert not executor.in_process
    assert executor.env == {
        "DINKSTER_TRAINING_BACKEND": "fake",
        "DINKSTER_TRAINING_JOURNAL": str(journal.resolve()),
    }
