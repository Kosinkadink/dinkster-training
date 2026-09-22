"""End-to-end CPU proofs for the multi-rank durable training service."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import dinkster_training_torch.rank_worker as rank_worker
import dinkster_training_torch.service as training_service
import pytest
import torch
from dinkster_inference_torch import UNetModel
from dinkster_protocol import TrainingEventName, TrainingJournalEvent
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    CheckpointState,
    ContentAddressedCheckpointStore,
    ModelFactory,
    PreparedBatch,
    QwenImageLoRATrainer,
    QwenImageTrainingConfig,
    SD15LoRATrainer,
    SD15LoRATrainingService,
    TrainingAdvancePaused,
    TrainingConfig,
    TrainingConfigError,
)
from dinkster_training_torch.distributed import PROCESS_GROUP_TIMEOUT, TorchDistributedRankContext
from dinkster_training_torch.rank_worker import adapter_sync_digest, config_for_rank
from PIL import Image
from test_qwen_image_lora import config_mapping as qwen_image_config_mapping

_UNET: dict[str, object] = {
    "inChannels": 4,
    "outChannels": 4,
    "modelChannels": 32,
    "numResBlocks": [1],
    "channelMult": [1],
    "transformerDepth": [1],
    "transformerDepthOutput": [1, 1],
    "transformerDepthMiddle": 1,
    "contextDim": 8,
    "useLinearInTransformer": False,
    "numHeads": 8,
}


def _config_mapping(
    rendezvous_path: Path,
    *,
    world_size: int | None = 2,
    device: str = "cpu",
    rng_policy: str = "sequential",
    checkpoint_interval: int = 0,
    sync_digest_interval: int = 0,
) -> dict[str, object]:
    mapping: dict[str, object] = {
        "schemaVersion": 1,
        "family": "sd15",
        "unet": _UNET,
        "device": device,
        "baseDtype": "float32",
        "rank": 2,
        "alpha": 2.0,
        "learningRate": 0.0005,
        "weightDecay": 0.01,
        "optimizer": "adamw",
        "gradientAccumulationSteps": 1,
        "gradientCheckpointing": False,
        "seed": 1234,
        "latentShape": [1, 4, 2, 2],
        "contextShape": [1, 2, 8],
    }
    if world_size is not None:
        mapping["distributed"] = {
            "worldSize": world_size,
            "backend": "gloo",
            "rendezvous": {
                "method": "file",
                "path": str(rendezvous_path.resolve()),
            },
        }
    if rng_policy != "sequential":
        mapping["rngPolicy"] = rng_policy
    if checkpoint_interval:
        mapping["checkpointInterval"] = checkpoint_interval
    if sync_digest_interval:
        mapping["syncDigestInterval"] = sync_digest_interval
    return mapping


def _serialized_config(
    rendezvous_path: Path,
    *,
    world_size: int | None = 2,
    device: str = "cpu",
    rng_policy: str = "sequential",
    checkpoint_interval: int = 0,
    sync_digest_interval: int = 0,
) -> str:
    return json.dumps(
        _config_mapping(
            rendezvous_path,
            world_size=world_size,
            device=device,
            rng_policy=rng_policy,
            checkpoint_interval=checkpoint_interval,
            sync_digest_interval=sync_digest_interval,
        ),
        sort_keys=True,
    )


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = torch.nn.Conv2d(4, 4, 1, bias=False)
        with torch.no_grad():
            values = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1)
            self.proj_in.weight.copy_(values / 32.0 - 0.25)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        condition = timesteps.reshape(-1, 1, 1, 1).float() / 1000.0
        condition = condition + context.float().mean(dim=(1, 2)).reshape(-1, 1, 1, 1) * 0.01
        return self.proj_in(latents) + condition.to(latents.dtype)


class _Batches:
    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        del generator, device
        latents = torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2) / 16.0
        context = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 16.0
        return PreparedBatch(latents + cursor * 0.01, context - cursor * 0.02)


def _model_factory(config: TrainingConfig) -> torch.nn.Module:
    del config
    return _TinyModel()


def _unet_model_factory(config: TrainingConfig) -> torch.nn.Module:
    model = UNetModel(config.unet)
    state: dict[str, torch.Tensor] = {}
    for index, (name, value) in enumerate(model.state_dict().items(), 1):
        values = torch.arange(value.numel(), dtype=torch.float32).remainder_(31)
        state[name] = ((values + index) / 1000.0).reshape(value.shape).to(value.dtype)
    model.load_state_dict(state, strict=True)
    return model


def _data_source_factory(config: TrainingConfig) -> _Batches:
    del config
    return _Batches()


def _make_store(path: Path) -> TrainingSessionStore:
    return TrainingSessionStore(JournalStore(path))


def _checkpoint(root: Path, digest: str) -> CheckpointState:
    return ContentAddressedCheckpointStore(root).load(digest)


def test_adapter_sync_digest_matches_per_parameter_byte_stream() -> None:
    parameters = (
        torch.nn.Parameter(torch.tensor([[1.25, -2.5], [3.75, 4.0]], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([5.5, -6.25, 7.0], dtype=torch.float32)),
        torch.nn.Parameter(
            torch.tensor([[0.5, 1.5, -2.0], [8.25, -9.0, 10.5]], dtype=torch.float32).t()
        ),
    )
    assert not parameters[2].is_contiguous()
    expected = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().to(device="cpu").contiguous()
        expected.update(value.view(torch.uint8).numpy().tobytes())
    trainer = SimpleNamespace(attachment=SimpleNamespace(parameters=lambda: parameters))

    actual = adapter_sync_digest(cast("Any", trainer))

    assert actual == "sha256:" + expected.hexdigest()


def _recovery_losses(
    root: Path,
    store: TrainingSessionStore,
    session_id: str,
) -> tuple[float, ...]:
    checkpoints = ContentAddressedCheckpointStore(root)
    losses: list[float] = []
    for record in store.read_events(session_id, after=0, limit=100).records:
        if record.name != TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED:
            continue
        event = TrainingJournalEvent.from_wire(record.payload)
        digest = event.data["manifestDigest"]
        assert isinstance(digest, str)
        loss = checkpoints.load(digest).loss
        assert loss is not None
        losses.append(loss)
    return tuple(losses)


def _assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        left_mapping = cast("dict[object, object]", left)
        right_mapping = cast("dict[object, object]", right)
        assert left_mapping.keys() == right_mapping.keys()
        for key in left_mapping:
            _assert_tree_equal(left_mapping[key], right_mapping[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        right_values = cast("list[object] | tuple[object, ...]", right)
        assert len(left) == len(right_values)
        for left_value, right_value in zip(left, right_values, strict=True):
            _assert_tree_equal(left_value, right_value)
        return
    assert left == right


class _GradientAverager:
    def __init__(self, world_size: int) -> None:
        self._world_size = world_size
        self._barrier = threading.Barrier(world_size)
        self._participants: dict[
            int,
            tuple[
                tuple[torch.nn.Parameter, ...],
                tuple[torch.Tensor | None, ...],
            ],
        ] = {}

    def average(self, rank: int, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        gradients = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in parameters
        )
        self._participants[rank] = (parameters, gradients)
        self._barrier.wait()
        if rank == 0:
            for index in range(len(parameters)):
                values = [
                    participant_gradients[index]
                    for _, participant_gradients in self._participants.values()
                ]
                if all(value is None for value in values):
                    continue
                assert all(value is not None for value in values)
                tensors = cast("list[torch.Tensor]", values)
                mean = torch.stack(tensors).sum(dim=0).div_(self._world_size)
                for participant_parameters, _ in self._participants.values():
                    gradient = participant_parameters[index].grad
                    assert gradient is not None
                    gradient.copy_(mean)
        self._barrier.wait()
        if rank == 0:
            self._participants.clear()
        self._barrier.wait()


class _FakeRankContext:
    def __init__(self, rank: int, group: _GradientAverager) -> None:
        self.rank = rank
        self.world_size = 2
        self._group = group

    def all_reduce_gradients(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        self._group.average(self.rank, parameters)

    def broadcast_parameters(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        del parameters

    def barrier(self) -> None:
        pass


def _manual_run(
    config: TrainingConfig,
    steps: int,
    model_factory: ModelFactory = _model_factory,
) -> tuple[tuple[float, ...], SD15LoRATrainer]:
    group = _GradientAverager(2)
    trainers = tuple(
        SD15LoRATrainer(
            config,
            model_factory(config),
            _data_source_factory(config),
            rank_context=_FakeRankContext(rank, group),
        )
        for rank in range(2)
    )
    rank_zero_losses: list[float] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for _ in range(steps):
            futures = tuple(executor.submit(trainer.train_step) for trainer in trainers)
            losses = tuple(future.result() for future in futures)
            rank_zero_losses.append(losses[0])
    _assert_tree_equal(
        trainers[0].attachment.state_dict(),
        trainers[1].attachment.state_dict(),
    )
    return tuple(rank_zero_losses), trainers[0]


def test_world_two_service_matches_manual_run_exports_replays_and_completes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    store = _make_store(tmp_path / "training.sqlite")
    service = SD15LoRATrainingService(
        store,
        root,
        model_factory=_unet_model_factory,
        data_source_factory=_data_source_factory,
    )
    try:
        # checkpointInterval=1 keeps the per-step recovery stream this test compares.
        serialized = _serialized_config(tmp_path / "service-rendezvous", checkpoint_interval=1)
        initial, report = service.create("world-two", serialized)
        assert (
            report["distributed"] == _config_mapping(tmp_path / "service-rendezvous")["distributed"]
        )
        initial_state = _checkpoint(root, initial.checkpoint_manifest_digest)
        assert tuple(initial_state.rng) == (
            "rank0:data",
            "rank0:timestep-noise",
            "rank1:data",
            "rank1:timestep-noise",
        )

        first = service.advance(initial, 2, "first two steps")
        steps_before_replay = service.steps_run
        replay = service.advance(initial, 2, "first two steps")
        assert replay.replayed
        assert replay.handle == first.handle
        assert service.steps_run == steps_before_replay

        second = service.advance(first.handle, 2, "second two steps")
        losses = _recovery_losses(root, store, initial.session_id)
        assert len(losses) == 4
        assert first.loss == losses[1]
        assert second.loss == losses[3]

        export_path, export_digest = service.export_lora(
            second.handle,
            json.dumps({"path": "world-two.safetensors"}, sort_keys=True),
        )
        assert Path(export_path).is_file()
        assert export_digest.startswith("blake3:")

        state = _checkpoint(root, second.handle.checkpoint_manifest_digest)
        config = TrainingConfig.from_mapping(state.config)
        manual_losses, manual = _manual_run(config, 4, _unet_model_factory)
        assert losses == manual_losses
        _assert_tree_equal(state.adapter, manual.attachment.state_dict())
        _assert_tree_equal(state.optimizer, manual.optimizer_state_dict())

        completed = service.complete(second.handle)
        assert completed == second.handle
        assert initial.session_id not in service._hot  # pyright: ignore[reportPrivateUsage]
    finally:
        service._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service.session_id("world-two")
        )
        store.close()


def test_world_two_service_persists_one_counter_policy_marker(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    store = _make_store(tmp_path / "training.sqlite")
    service = SD15LoRATrainingService(
        store,
        root,
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )
    try:
        initial, _ = service.create(
            "counter-policy",
            _serialized_config(
                tmp_path / "counter-policy-rendezvous",
                rng_policy="counter",
            ),
        )
        initial_state = _checkpoint(root, initial.checkpoint_manifest_digest)
        assert tuple(initial_state.rng) == ("__dinkster_counter_rng_policy__",)
        assert torch.equal(
            initial_state.rng["__dinkster_counter_rng_policy__"],
            torch.tensor([1], dtype=torch.uint8),
        )

        advanced = service.advance(initial, 1, "one counter-policy step")
        state = _checkpoint(root, advanced.handle.checkpoint_manifest_digest)
        assert tuple(state.rng) == ("__dinkster_counter_rng_policy__",)
    finally:
        service._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service.session_id("counter-policy")
        )
        store.close()


def test_world_two_resume_from_committed_handle_is_exact(tmp_path: Path) -> None:
    serialized_a = _serialized_config(tmp_path / "uninterrupted-rendezvous")
    root_a = tmp_path / "uninterrupted-checkpoints"
    store_a = _make_store(tmp_path / "uninterrupted.sqlite")
    service_a = SD15LoRATrainingService(
        store_a,
        root_a,
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )
    try:
        initial_a, _ = service_a.create("uninterrupted", serialized_a)
        middle_a = service_a.advance(initial_a, 2, "prefix").handle
        final_a = service_a.advance(middle_a, 2, "suffix").handle
        losses_a = _recovery_losses(root_a, store_a, initial_a.session_id)
        state_a = _checkpoint(root_a, final_a.checkpoint_manifest_digest)
        service_a.complete(final_a)
    finally:
        service_a._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service_a.session_id("uninterrupted")
        )
        store_a.close()

    serialized_b = _serialized_config(tmp_path / "resumed-rendezvous")
    root_b = tmp_path / "resumed-checkpoints"
    store_b = _make_store(tmp_path / "resumed.sqlite")
    service_b = SD15LoRATrainingService(
        store_b,
        root_b,
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )
    successor: SD15LoRATrainingService | None = None
    try:
        initial_b, _ = service_b.create("resumed", serialized_b)
        middle_b = service_b.advance(initial_b, 2, "prefix").handle
        service_b._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            initial_b.session_id
        )
        successor = SD15LoRATrainingService(
            store_b,
            root_b,
            model_factory=_model_factory,
            data_source_factory=_data_source_factory,
        )
        final_b = successor.advance(middle_b, 2, "suffix").handle
        losses_b = _recovery_losses(root_b, store_b, initial_b.session_id)
        state_b = _checkpoint(root_b, final_b.checkpoint_manifest_digest)

        assert losses_b == losses_a
        assert state_b.step_cursor == state_a.step_cursor == 4
        assert state_b.data_cursor == state_a.data_cursor == 8
        assert state_b.loss == state_a.loss
        _assert_tree_equal(state_b.adapter, state_a.adapter)
        _assert_tree_equal(state_b.optimizer, state_a.optimizer)
        _assert_tree_equal(state_b.rng, state_a.rng)
        successor.complete(final_b)
    finally:
        service_b._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service_b.session_id("resumed")
        )
        if successor is not None:
            successor._evict_runtime(  # pyright: ignore[reportPrivateUsage]
                successor.session_id("resumed")
            )
        store_b.close()


def test_world_two_recovery_checkpoint_resumes_after_safe_point_pause(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    store = _make_store(tmp_path / "training.sqlite")
    service: SD15LoRATrainingService

    def cancelled() -> bool:
        return service.steps_run >= 1

    service = SD15LoRATrainingService(
        store,
        root,
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
        cancelled=cancelled,
    )
    successor: SD15LoRATrainingService | None = None
    try:
        # checkpointInterval=1 keeps the per-step recovery stream this test compares.
        initial, _ = service.create(
            "safe-point",
            _serialized_config(tmp_path / "safe-point-rendezvous", checkpoint_interval=1),
        )
        with pytest.raises(TrainingAdvancePaused, match="step 1 of 3"):
            service.advance(initial, 3, "recover this operation")
        assert service.steps_run == 1
        paused_losses = _recovery_losses(root, store, initial.session_id)
        assert len(paused_losses) == 1

        service._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            initial.session_id
        )
        successor = SD15LoRATrainingService(
            store,
            root,
            model_factory=_model_factory,
            data_source_factory=_data_source_factory,
        )
        final = successor.advance(initial, 3, "recover this operation")
        assert not final.replayed
        assert final.handle.step_cursor == 3
        assert successor.steps_run == 2
        losses = _recovery_losses(root, store, initial.session_id)
        assert len(losses) == 3
        assert losses[:1] == paused_losses
        successor.complete(final.handle)
    finally:
        service._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service.session_id("safe-point")
        )
        if successor is not None:
            successor._evict_runtime(  # pyright: ignore[reportPrivateUsage]
                successor.session_id("safe-point")
            )
        store.close()


def test_world_two_detects_adapter_divergence_and_tears_down(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    store = _make_store(tmp_path / "training.sqlite")
    service = SD15LoRATrainingService(
        store,
        root,
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )
    try:
        initial, _ = service.create(
            "divergence",
            _serialized_config(tmp_path / "divergence-rendezvous"),
        )
        committed = service.advance(initial, 1, "before tamper").handle
        runtime = service._hot[initial.session_id]  # pyright: ignore[reportPrivateUsage]
        with torch.no_grad():
            runtime.trainer.attachment.parameters()[0].add_(1.0)

        with pytest.raises(RuntimeError, match="distributed adapter parameters diverged"):
            service.advance(committed, 1, "detect tamper")
        assert initial.session_id not in service._hot  # pyright: ignore[reportPrivateUsage]
    finally:
        service._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service.session_id("divergence")
        )
        store.close()


def test_world_two_checks_sync_digests_at_interval_checkpoint_and_final_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkpoints"
    store = _make_store(tmp_path / "training.sqlite")
    rank_zero_steps: list[int] = []
    worker_checks = 0
    original_digest = training_service.adapter_sync_digest  # pyright: ignore[reportPrivateImportUsage]
    original_sync_digests = rank_worker.RankWorkerGroup.sync_digests

    def record_rank_zero_digest(trainer: Any) -> str:
        rank_zero_steps.append(trainer.step_cursor)
        return original_digest(trainer)

    def record_worker_digests(group: rank_worker.RankWorkerGroup) -> tuple[str, ...]:
        nonlocal worker_checks
        worker_checks += 1
        return original_sync_digests(group)

    monkeypatch.setattr(training_service, "adapter_sync_digest", record_rank_zero_digest)
    monkeypatch.setattr(rank_worker.RankWorkerGroup, "sync_digests", record_worker_digests)
    service = SD15LoRATrainingService(
        store,
        root,
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )
    try:
        initial, _ = service.create(
            "digest-cadence",
            _serialized_config(
                tmp_path / "digest-cadence-rendezvous",
                checkpoint_interval=3,
                sync_digest_interval=2,
            ),
        )

        service.advance(initial, 6, "exercise digest cadence")

        assert rank_zero_steps == [2, 3, 4, 6]
        assert worker_checks == 4
    finally:
        service._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service.session_id("digest-cadence")
        )
        store.close()


def test_world_two_recovers_after_worker_death_leaves_stale_file_rendezvous(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    rendezvous = tmp_path / "worker-death-rendezvous"
    store = _make_store(tmp_path / "training.sqlite")
    service = SD15LoRATrainingService(
        store,
        root,
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )
    successor: SD15LoRATrainingService | None = None
    try:
        initial, _ = service.create(
            "worker-death",
            _serialized_config(rendezvous),
        )
        committed = service.advance(initial, 1, "prefix").handle
        runtime = service._hot[initial.session_id]  # pyright: ignore[reportPrivateUsage]
        assert runtime.workers is not None
        worker = runtime.workers._workers[0]  # pyright: ignore[reportPrivateUsage]
        worker.process.kill()
        worker.process.join(timeout=5.0)
        assert not worker.process.is_alive()

        with pytest.raises(RuntimeError, match="rank 1 exited before 'step'"):
            service.advance(committed, 1, "recover after worker death")
        assert rendezvous.is_file()

        successor = SD15LoRATrainingService(
            store,
            root,
            model_factory=_model_factory,
            data_source_factory=_data_source_factory,
        )
        resumed = successor.advance(committed, 1, "recover after worker death")
        assert resumed.handle.step_cursor == 2
        assert service.steps_run == successor.steps_run == 1
        successor.complete(resumed.handle)
    finally:
        service._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service.session_id("worker-death")
        )
        if successor is not None:
            successor._evict_runtime(  # pyright: ignore[reportPrivateUsage]
                successor.session_id("worker-death")
            )
        store.close()


def test_rank_worker_sets_only_cuda_as_the_current_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_devices: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "set_device", selected_devices.append)

    cuda_config = TrainingConfig.from_mapping(
        _config_mapping(tmp_path / "cuda-rendezvous", device="cuda")
    )
    rank_worker._set_rank_local_cuda_device(  # pyright: ignore[reportPrivateUsage]
        config_for_rank(cuda_config, 1)
    )
    assert selected_devices == [torch.device("cuda:1")]

    selected_devices.clear()
    cpu_config = TrainingConfig.from_mapping(_config_mapping(tmp_path / "cpu-rendezvous"))
    rank_worker._set_rank_local_cuda_device(  # pyright: ignore[reportPrivateUsage]
        config_for_rank(cpu_config, 1)
    )
    assert selected_devices == []


def test_rank_worker_selects_cuda_device_before_distributed_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    trainer = object()
    rank_context = object()

    def select_device(config: object) -> None:
        parsed = cast("TrainingConfig", config)
        events.append(f"device:{parsed.device}")

    def bootstrap(_settings: object, rank: int) -> object:
        events.append(f"bootstrap:{rank}")
        return rank_context

    def build(_config: object, **_kwargs: object) -> object:
        events.append("build")
        return trainer

    monkeypatch.setattr(rank_worker, "_set_rank_local_cuda_device", select_device)
    monkeypatch.setattr(rank_worker, "bootstrap_rank_context", bootstrap)
    monkeypatch.setattr(rank_worker, "build_trainer", build)

    result = rank_worker._load_worker_trainer(  # pyright: ignore[reportPrivateUsage]
        1,
        (
            _config_mapping(tmp_path / "cuda-rendezvous", device="cuda"),
            None,
            str(tmp_path / "checkpoints"),
        ),
        expected_family="sd15",
        trainer_type=SD15LoRATrainer,
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )

    assert result == (trainer, rank_context)
    assert events == ["device:cuda:1", "bootstrap:1", "build"]


def test_qwen_image_rank_worker_parses_and_preserves_construction_order(tmp_path: Path) -> None:
    events: list[str] = []

    class Trainer:
        def __init__(
            self,
            config: QwenImageTrainingConfig,
            model: object,
            data_source: object,
            *,
            rank_context: object | None = None,
        ) -> None:
            events.append("trainer")
            self.values = (config, model, data_source, rank_context)

    def model_factory(_config: QwenImageTrainingConfig) -> object:
        events.append("model")
        return object()

    def data_factory(_config: QwenImageTrainingConfig) -> object:
        events.append("data")
        return object()

    injected_mapping = qwen_image_config_mapping(tmp_path / "injected")
    injected = rank_worker._config_from_mapping(  # pyright: ignore[reportPrivateUsage]
        injected_mapping, "qwen-image"
    )
    assert isinstance(injected, QwenImageTrainingConfig)
    trainer = rank_worker.build_trainer(
        injected,
        trainer_type=cast("type[QwenImageLoRATrainer]", Trainer),
        model_factory=cast("Any", model_factory),
        data_source_factory=cast("Any", data_factory),
    )
    assert isinstance(cast("object", trainer), Trainer)
    assert events == ["model", "data", "trainer"]

    events.clear()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (16, 16)).save(dataset / "item.png")
    (dataset / "item.txt").write_text("caption", encoding="utf-8")
    dataset_mapping = qwen_image_config_mapping(tmp_path / "configured")
    dataset_mapping.pop("datasetIdentity")
    dataset_mapping["baseDtype"] = "float32"
    dataset_mapping["gradientCheckpointing"] = False
    dataset_mapping["latentShape"] = [1, 16, 1, 2, 2]
    dataset_mapping["dataset"] = {
        "type": "qwen-image-image-caption-folder",
        "root": str(dataset.resolve()),
        "resolution": [16, 16],
        "contextTokens": 512,
    }
    dataset_mapping["distributed"] = {
        "worldSize": 2,
        "backend": "gloo",
        "rendezvous": {
            "method": "file",
            "path": str((tmp_path / "qwen-rendezvous").resolve()),
        },
    }
    configured = rank_worker._config_from_mapping(  # pyright: ignore[reportPrivateUsage]
        dataset_mapping, "qwen-image"
    )
    assert isinstance(configured, QwenImageTrainingConfig)

    class RankContext:
        rank = 1
        world_size = 2

    ranked = rank_worker.build_trainer(
        configured,
        trainer_type=cast("type[QwenImageLoRATrainer]", Trainer),
        model_factory=cast("Any", model_factory),
        data_source_factory=cast("Any", data_factory),
        rank_context=cast("Any", RankContext()),
    )
    assert isinstance(cast("object", ranked), Trainer)
    values = cast("Trainer", ranked).values
    assert values[0].device == "cpu"
    assert isinstance(values[3], RankContext)
    assert events == ["data", "model", "trainer"]


def test_multi_rank_device_satisfiability_is_validated_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(tmp_path / "training.sqlite")
    service = SD15LoRATrainingService(
        store,
        tmp_path / "checkpoints",
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    try:
        cuda_config = _serialized_config(
            tmp_path / "cuda-rendezvous",
            device="cuda",
        )
        with pytest.raises(TrainingConfigError, match="requires at least 2 CUDA devices"):
            service.dry_run(cuda_config)
        with pytest.raises(TrainingConfigError, match="requires at least 2 CUDA devices"):
            service.create("insufficient-cuda", cuda_config)

        unsupported = _serialized_config(
            tmp_path / "unsupported-rendezvous",
            device="cuda:1",
        )
        with pytest.raises(TrainingConfigError, match="multi-rank training supports device"):
            service.dry_run(unsupported)
        with pytest.raises(TrainingConfigError, match="multi-rank training supports device"):
            service.create("unsupported-device", unsupported)

        parsed = TrainingConfig.from_mapping(
            _config_mapping(tmp_path / "mapping-rendezvous", device="cuda")
        )
        assert config_for_rank(parsed, 0).device == "cuda:0"
        assert config_for_rank(parsed, 1).device == "cuda:1"
        assert PROCESS_GROUP_TIMEOUT.total_seconds() == 120
    finally:
        store.close()


def test_world_one_service_never_spawns_a_rank_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_attempts: list[int] = []
    bootstrap_attempts: list[int] = []

    def unexpected_group(world_size: int, **_kwargs: object) -> object:
        spawn_attempts.append(world_size)
        raise AssertionError("world-one service spawned a rank worker")

    def unexpected_bootstrap(_settings: object, rank: int) -> object:
        bootstrap_attempts.append(rank)
        raise AssertionError("world-one service bootstrapped a process group")

    monkeypatch.setattr(training_service, "RankWorkerGroup", unexpected_group)
    monkeypatch.setattr(training_service, "bootstrap_rank_context", unexpected_bootstrap)
    store = _make_store(tmp_path / "training.sqlite")
    service = SD15LoRATrainingService(
        store,
        tmp_path / "checkpoints",
        model_factory=_model_factory,
        data_source_factory=_data_source_factory,
    )
    try:
        initial, _ = service.create(
            "world-one",
            _serialized_config(
                tmp_path / "unused-rendezvous",
                world_size=1,
            ),
        )
        final = service.advance(initial, 1, "one step").handle
        service.complete(final)
        assert spawn_attempts == []
        assert bootstrap_attempts == []
    finally:
        service._evict_runtime(  # pyright: ignore[reportPrivateUsage]
            service.session_id("world-one")
        )
        store.close()


def test_hot_runtime_signals_teardown_before_rank_zero_close_and_worker_reap(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class Workers:
        def signal_teardown(self) -> bool:
            events.append("signal")
            return True

        def close(self) -> None:
            events.append("reap")

    class RankContext:
        def close(self) -> None:
            events.append("rank-zero-close")

        def abort(self) -> None:
            raise AssertionError("healthy teardown aborted rank zero")

    config = TrainingConfig.from_mapping(_config_mapping(tmp_path / "rendezvous"))
    runtime = training_service._HotRuntime(  # pyright: ignore[reportPrivateUsage]
        "checkpoint",
        cast("SD15LoRATrainer", object()),
        config,
        cast("rank_worker.RankWorkerGroup", Workers()),
        cast("TorchDistributedRankContext", RankContext()),
    )

    runtime.close()
    runtime.close()

    assert events == ["signal", "rank-zero-close", "reap"]


def test_hot_runtime_teardown_does_not_wait_for_unsignaled_workers(tmp_path: Path) -> None:
    teardown_signaled = threading.Event()
    rank_zero_closed = threading.Event()
    workers_reaped = threading.Event()
    close_finished = threading.Event()

    class Workers:
        def signal_teardown(self) -> bool:
            teardown_signaled.set()
            return True

        def close(self) -> None:
            workers_reaped.set()

    class RankContext:
        def close(self) -> None:
            teardown_signaled.wait()
            rank_zero_closed.set()

        def abort(self) -> None:
            raise AssertionError("healthy teardown aborted rank zero")

    config = TrainingConfig.from_mapping(_config_mapping(tmp_path / "rendezvous"))
    runtime = training_service._HotRuntime(  # pyright: ignore[reportPrivateUsage]
        "checkpoint",
        cast("SD15LoRATrainer", object()),
        config,
        cast("rank_worker.RankWorkerGroup", Workers()),
        cast("TorchDistributedRankContext", RankContext()),
    )

    def close_runtime() -> None:
        runtime.close()
        close_finished.set()

    watchdog = threading.Thread(target=close_runtime, daemon=True)
    watchdog.start()

    assert close_finished.wait(timeout=1.0)
    watchdog.join(timeout=0.1)
    assert not watchdog.is_alive()
    assert rank_zero_closed.is_set()
    assert workers_reaped.is_set()


def test_hot_runtime_aborts_nccl_rank_zero_when_a_worker_is_dead(tmp_path: Path) -> None:
    events: list[str] = []

    class Workers:
        def signal_teardown(self) -> bool:
            events.append("signal-dead-worker")
            return False

        def close(self) -> None:
            events.append("reap")

    class RankContext:
        def close(self) -> None:
            raise AssertionError("dead NCCL peer used graceful rank-zero close")

        def abort(self) -> None:
            events.append("rank-zero-abort")

    mapping = _config_mapping(tmp_path / "rendezvous")
    distributed = cast("dict[str, object]", mapping["distributed"])
    distributed["backend"] = "nccl"
    config = TrainingConfig.from_mapping(mapping)
    runtime = training_service._HotRuntime(  # pyright: ignore[reportPrivateUsage]
        "checkpoint",
        cast("SD15LoRATrainer", object()),
        config,
        cast("rank_worker.RankWorkerGroup", Workers()),
        cast("TorchDistributedRankContext", RankContext()),
    )

    runtime.close()

    assert events == ["signal-dead-worker", "rank-zero-abort", "reap"]
