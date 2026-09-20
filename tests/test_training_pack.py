"""The training pack end to end: a fold/while region carries the session
handle through ``training.advance`` while the durable supervisor below the
graph keeps advances exactly-once (training-design.md 3.1-3.2, 9.1, 9.3).

The FakeTrainer (dinkster_training_worker.fake) is deterministic and torch-free;
everything durable it drives - claim/commit, recovery checkpoints, pause
acknowledgement, fence takeover, committed replay - is the real supervisor.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, ExecutionError
from dinkster_graph import PORTS_NODE_ID, Graph, GraphNode, Link, RegionNode, RegionOutput
from dinkster_nodes_training import (
    HANDLE,
    PACK_NODES,
    bind_training_service,
    register_training_types,
)
from dinkster_protocol import TrainingEventName, TrainingSessionHandle
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_worker.fake import FakeTrainer
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import InProcessWorker

INT = TypeExpr.concrete("core.int")
BOOL = TypeExpr.concrete("core.boolean")
STRING = TypeExpr.concrete("core.string")

CONFIG = '{"family": "fake", "lr": 0.001}'


class StepBelow(Node):
    """While-loop continuation policy: keep advancing below a target step."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.step_below",
            inputs=(InputSpec("step_cursor", INT), InputSpec("target", INT)),
            outputs=(OutputSpec("below", BOOL),),
        )

    @classmethod
    def execute(cls, step_cursor: int, target: int) -> Mapping[str, object]:
        return cls.outputs(below=step_cursor < target)


NODES: tuple[type[Node], ...] = (*PACK_NODES, StepBelow)


def make_store(tmp_path: Path) -> TrainingSessionStore:
    return TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))


def make_engine() -> Engine:
    registry = TypeRegistry()
    register_core_types(registry)
    register_training_types(registry)
    return Engine(
        schemas=build_schemas(NODES),
        registry=registry,
        worker=InProcessWorker(build_node_types(NODES), registry),
        cache=MemoryLRUCache(),
    )


def port(port_id: str) -> Link:
    return Link(PORTS_NODE_ID, port_id)


def fold_training_graph(session_key: str, intervals: list[int]) -> Graph:
    """Create a session, then fold the handle through one advance per
    interval element."""
    return Graph(
        nodes={
            "create": GraphNode(
                "training.create_session",
                {"session_key": session_key, "config": CONFIG},
            ),
            "loop": RegionNode(
                kind="fold",
                body=Graph(
                    nodes={
                        "adv": GraphNode(
                            "training.advance",
                            {"handle": port("handle"), "steps": port("interval")},
                        )
                    }
                ),
                ports={"interval": INT, "handle": HANDLE},
                inputs={"interval": intervals, "handle": Link("create", "handle")},
                element_ports=("interval",),
                state_ports=("handle",),
                outputs={"handle": RegionOutput(Link("adv", "handle"), mode="state")},
            ),
        }
    )


def while_training_graph(session_key: str, *, target: int, steps_per_advance: int) -> Graph:
    """Create a session, then advance until the step cursor reaches the
    target."""
    return Graph(
        nodes={
            "create": GraphNode(
                "training.create_session",
                {"session_key": session_key, "config": CONFIG},
            ),
            "loop": RegionNode(
                kind="while",
                body=Graph(
                    nodes={
                        "adv": GraphNode(
                            "training.advance",
                            {"handle": port("handle"), "steps": steps_per_advance},
                        ),
                        "below": GraphNode(
                            "test.step_below",
                            {"step_cursor": Link("adv", "step_cursor"), "target": port("target")},
                        ),
                    }
                ),
                ports={"handle": HANDLE, "target": INT},
                inputs={"handle": Link("create", "handle"), "target": target},
                state_ports=("handle",),
                outputs={"handle": RegionOutput(Link("adv", "handle"), mode="state")},
                continue_source=Link("below", "below"),
                max_iterations=50,
            ),
        }
    )


def run_graph(
    trainer: FakeTrainer, graph: Graph, demands: list[str]
) -> Mapping[str, Mapping[str, object]]:
    async def scenario() -> Mapping[str, Mapping[str, object]]:
        engine = make_engine()
        result = await engine.run(graph, demands)
        return {
            node_id: {out_id: value.resolve() for out_id, value in outs.items()}
            for node_id, outs in result.outputs.items()
        }

    with bind_training_service(trainer):
        return asyncio.run(scenario())


def event_names(store: TrainingSessionStore, session_id: str) -> list[str]:
    page = store.read_events(session_id, after=0, limit=1000)
    return [record.name for record in page.records]


# -- the fold/while handle loop ----------------------------------------------


def test_fold_loop_advances_and_commits(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    outputs = run_graph(trainer, fold_training_graph("fold-run", [1, 2, 3]), ["loop"])
    handle = outputs["loop"]["handle"]
    assert isinstance(handle, TrainingSessionHandle)
    assert handle.step_cursor == 6
    assert trainer.steps_run == 6
    checkpoints = store.list_checkpoints(handle.session_id)
    assert [c.step_cursor for c in checkpoints] == [0, 1, 3, 6]
    assert handle.checkpoint_manifest_digest == checkpoints[-1].manifest_digest
    names = event_names(store, handle.session_id)
    assert names.count(TrainingEventName.ADVANCE_COMMITTED) == 3
    assert names.count(TrainingEventName.CHECKPOINT_PUBLISHED) == 3
    store.close()


def test_while_loop_advances_until_target(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    outputs = run_graph(
        trainer, while_training_graph("while-run", target=6, steps_per_advance=2), ["loop"]
    )
    handle = outputs["loop"]["handle"]
    assert isinstance(handle, TrainingSessionHandle)
    assert handle.step_cursor == 6
    assert trainer.steps_run == 6
    assert names_committed(store, handle.session_id) == 3
    store.close()


def names_committed(store: TrainingSessionStore, session_id: str) -> int:
    return event_names(store, session_id).count(TrainingEventName.ADVANCE_COMMITTED)


# -- retry safety: committed replay, duplicate refusal ------------------------


def test_rerun_replays_committed_operations_without_stepping(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    graph = fold_training_graph("replay-run", [1, 2, 3])
    first = run_graph(trainer, graph, ["loop"])["loop"]["handle"]
    assert trainer.steps_run == 6
    # Same trainer re-runs the whole graph: every advance replays its
    # committed outcome; zero optimizer steps happen.
    second = run_graph(trainer, graph, ["loop"])["loop"]["handle"]
    assert second == first
    assert trainer.steps_run == 6
    # A replacement trainer (new fence) replays too: committed operations
    # return their recorded outcome under any fence.
    replacement = FakeTrainer(store)
    third = run_graph(replacement, graph, ["loop"])["loop"]["handle"]
    assert third == first
    assert replacement.steps_run == 0
    assert isinstance(first, TrainingSessionHandle)
    assert names_committed(store, first.session_id) == 3
    store.close()


def test_divergent_advance_from_stale_handle_is_refused(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    initial, _ = trainer.create("diverge-run", CONFIG)
    run_graph(trainer, fold_training_graph("diverge-run", [1, 2]), ["loop"])
    # A DIFFERENT interval from the already-superseded initial checkpoint is
    # a new operation id whose input is not the committed head: refused, so
    # one session can never fork into two optimizer trajectories.
    divergent = Graph(
        nodes={"adv": GraphNode("training.advance", {"handle": initial.to_wire(), "steps": 5})}
    )
    with bind_training_service(trainer), pytest.raises(ExecutionError, match="stale or unknown"):
        asyncio.run(make_engine().run(divergent, ["adv"]))
    store.close()


# -- safe-point cancellation, resume, takeover --------------------------------


def test_cancellation_pauses_at_safe_point_and_same_trainer_resumes(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    armed = {"cancel": False}
    trainer = FakeTrainer(store, cancelled=lambda: armed["cancel"] and trainer.steps_run >= 1)
    handle, _ = trainer.create("pause-run", CONFIG)
    graph = Graph(
        nodes={"adv": GraphNode("training.advance", {"handle": handle.to_wire(), "steps": 3})}
    )
    armed["cancel"] = True
    with bind_training_service(trainer), pytest.raises(ExecutionError, match="safe point"):
        asyncio.run(make_engine().run(graph, ["adv"]))
    assert trainer.steps_run == 1
    names = event_names(store, handle.session_id)
    assert TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED in names
    assert TrainingEventName.ADVANCE_PAUSED in names
    assert TrainingEventName.ADVANCE_COMMITTED not in names
    # Resume is re-running the same graph: the trainer continues its own
    # paused claim from the recovery checkpoint - only the remaining steps.
    armed["cancel"] = False
    outputs = run_graph(trainer, graph, ["adv"])
    assert trainer.steps_run == 3
    assert outputs["adv"]["step_cursor"] == 3
    names = event_names(store, handle.session_id)
    assert TrainingEventName.ADVANCE_COMMITTED in names
    assert names.count(TrainingEventName.ADVANCE_STARTED) == 1
    store.close()


def test_paused_claim_is_taken_over_by_replacement_trainer(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    armed = {"cancel": False}
    trainer = FakeTrainer(store, cancelled=lambda: armed["cancel"] and trainer.steps_run >= 1)
    handle, _ = trainer.create("takeover-run", CONFIG)
    graph = Graph(
        nodes={"adv": GraphNode("training.advance", {"handle": handle.to_wire(), "steps": 3})}
    )
    armed["cancel"] = True
    with bind_training_service(trainer), pytest.raises(ExecutionError, match="safe point"):
        asyncio.run(make_engine().run(graph, ["adv"]))
    assert trainer.steps_run == 1
    # Cold recovery: a replacement trainer acquires a new fence, takes over
    # the paused claim, and runs only the remaining steps from the recovery
    # checkpoint.
    replacement = FakeTrainer(store)
    outputs = run_graph(replacement, graph, ["adv"])
    assert outputs["adv"]["step_cursor"] == 3
    assert replacement.steps_run == 2
    names = event_names(store, handle.session_id)
    assert TrainingEventName.ADVANCE_RESUMED in names
    assert names.count(TrainingEventName.ADVANCE_STARTED) == 1
    store.close()


# -- session lifecycle nodes ---------------------------------------------------


def test_dry_run_reports_capabilities_without_creating_state(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    graph = Graph(nodes={"dry": GraphNode("training.dry_run", {"config": CONFIG})})
    outputs = run_graph(trainer, graph, ["dry"])
    report = json.loads(str(outputs["dry"]["report"]))
    assert report["trainer"] == "fake-trainer/1"
    assert report["capabilities"]["autograd"] is False
    assert report["capabilities"]["safePointCancellation"] is True
    assert store.get_session(FakeTrainer._session_id("dry")) is None
    assert trainer.steps_run == 0
    store.close()


def test_create_rejoins_by_key_and_refuses_a_different_config(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    graph = fold_training_graph("rejoin-run", [2])
    first = run_graph(trainer, graph, ["create", "loop"])
    second = run_graph(trainer, graph, ["create"])
    assert second["create"]["handle"] == first["create"]["handle"]
    conflicting = Graph(
        nodes={
            "create": GraphNode(
                "training.create_session",
                {"session_key": "rejoin-run", "config": '{"different": true}'},
            )
        }
    )
    with (
        bind_training_service(trainer),
        pytest.raises(ExecutionError, match="different identity facts"),
    ):
        asyncio.run(make_engine().run(conflicting, ["create"]))
    store.close()


def test_handle_literal_resumes_a_session_across_graphs(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    run_graph(trainer, fold_training_graph("cross-run", [2]), ["loop"])
    session = store.get_session(FakeTrainer._session_id("cross-run"))
    assert session is not None
    # A second graph job resumes from the committed handle as a plain wire
    # literal: the session outlives any one graph run.
    resumed = Graph(
        nodes={
            "adv": GraphNode("training.advance", {"handle": session.handle().to_wire(), "steps": 1})
        }
    )
    outputs = run_graph(trainer, resumed, ["adv"])
    assert outputs["adv"]["step_cursor"] == 3
    assert trainer.steps_run == 3
    store.close()


def test_forged_handle_watermark_is_refused(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    handle, _ = trainer.create("forge-run", CONFIG)
    # The digest is genuine but the journal watermark is not the ledger's:
    # the handle no longer matches any committed checkpoint row.
    forged = replace(handle, journal_seq=handle.journal_seq + 7)
    graph = Graph(
        nodes={"adv": GraphNode("training.advance", {"handle": forged.to_wire(), "steps": 1})}
    )
    with (
        bind_training_service(trainer),
        pytest.raises(ExecutionError, match="committed checkpoint"),
    ):
        asyncio.run(make_engine().run(graph, ["adv"]))
    store.close()


def test_complete_requires_the_committed_head(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    initial, _ = trainer.create("head-run", CONFIG)
    run_graph(trainer, fold_training_graph("head-run", [2]), ["loop"])
    stale = Graph(
        nodes={"done": GraphNode("training.complete_session", {"handle": initial.to_wire()})}
    )
    with bind_training_service(trainer), pytest.raises(ExecutionError, match="committed head"):
        asyncio.run(make_engine().run(stale, ["done"]))
    store.close()


def test_complete_session_refuses_further_advances(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    outputs = run_graph(trainer, fold_training_graph("finish-run", [2]), ["loop"])
    handle = outputs["loop"]["handle"]
    assert isinstance(handle, TrainingSessionHandle)
    finish = Graph(
        nodes={"done": GraphNode("training.complete_session", {"handle": handle.to_wire()})}
    )
    done = run_graph(trainer, finish, ["done"])
    assert done["done"]["step_cursor"] == 2
    assert TrainingEventName.SESSION_COMPLETED in event_names(store, handle.session_id)
    more = Graph(
        nodes={"adv": GraphNode("training.advance", {"handle": handle.to_wire(), "steps": 1})}
    )
    with bind_training_service(trainer), pytest.raises(ExecutionError, match="completed"):
        asyncio.run(make_engine().run(more, ["adv"]))
    store.close()


def test_advance_refuses_nonpositive_steps(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    trainer = FakeTrainer(store)
    handle, _ = trainer.create("zero-run", CONFIG)
    graph = Graph(
        nodes={"adv": GraphNode("training.advance", {"handle": handle.to_wire(), "steps": 0})}
    )
    with (
        bind_training_service(trainer),
        pytest.raises(ExecutionError, match="at least one optimizer step"),
    ):
        asyncio.run(make_engine().run(graph, ["adv"]))
    store.close()


def test_unbound_service_fails_loudly(tmp_path: Path) -> None:
    graph = Graph(nodes={"dry": GraphNode("training.dry_run", {"config": CONFIG})})
    with pytest.raises(ExecutionError, match="no training service is bound"):
        asyncio.run(make_engine().run(graph, ["dry"]))
