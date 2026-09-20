"""CPU proofs for distributed training configuration and bootstrap."""

from __future__ import annotations

import builtins
import copy
import importlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import dinkster_training_torch.service as training_service
import pytest
import torch
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    DistributedSettings,
    FileRendezvousSettings,
    MiniMaxH3TrainingConfig,
    PreparedBatch,
    RankContext,
    SD15LoRATrainingService,
    TcpRendezvousSettings,
    TrainingConfig,
    TrainingConfigError,
    bootstrap_rank_context,
)
from dinkster_training_torch.distributed import TorchDistributedRankContext
from torch.multiprocessing.spawn import spawn

_ConfigMappingFactory = Callable[[], dict[str, object]]

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


def _sd_mapping() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "family": "sd15",
        "unet": _UNET,
        "device": "cpu",
        "baseDtype": "float32",
        "rank": 2,
        "alpha": 2.0,
        "learningRate": 0.0005,
        "weightDecay": 0.01,
        "optimizer": "adamw",
        "gradientAccumulationSteps": 2,
        "gradientCheckpointing": False,
        "seed": 1234,
        "latentShape": [1, 4, 4, 4],
        "contextShape": [1, 2, 8],
    }


def _h3_mapping() -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "family": "minimax-h3",
        "ditRole": "fl2va-dit",
        "ditIdentity": "native:dinkster.minimax_h3:" + "1" * 64,
        "conditionerIdentity": "native:dinkster.minimax_h3:" + "2" * 64,
        "device": "cpu",
        "baseDtype": "float32",
        "rank": 2,
        "alpha": 2.0,
        "learningRate": 0.0005,
        "weightDecay": 0.01,
        "optimizer": "adamw",
        "gradientAccumulationSteps": 2,
        "gradientCheckpointing": False,
        "seed": 1234,
        "videoLatentShape": [1, 24, 1, 2, 2],
        "audioLatentShape": [1, 32, 2, 1],
        "conditionerShape": [1, 1, 5120],
    }


def _parse(
    factory: _ConfigMappingFactory, mapping: dict[str, object]
) -> TrainingConfig | MiniMaxH3TrainingConfig:
    if factory is _sd_mapping:
        return TrainingConfig.from_mapping(mapping)
    return MiniMaxH3TrainingConfig.from_mapping(mapping)


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
@pytest.mark.parametrize(
    ("distributed", "error"),
    [
        (
            {
                "worldSize": 0,
                "backend": "gloo",
                "rendezvous": {"method": "file", "path": "/tmp/group"},
            },
            "worldSize",
        ),
        (
            {
                "worldSize": True,
                "backend": "gloo",
                "rendezvous": {"method": "file", "path": "/tmp/group"},
            },
            "worldSize",
        ),
        (
            {
                "worldSize": 1,
                "backend": "mpi",
                "rendezvous": {"method": "file", "path": "/tmp/group"},
            },
            "backend",
        ),
        ({"worldSize": 1, "backend": "gloo", "rendezvous": "file:///tmp/group"}, "object"),
        ({"worldSize": 1, "backend": "gloo", "rendezvous": {"method": "env"}}, "method"),
        (
            {
                "worldSize": 1,
                "backend": "gloo",
                "rendezvous": {"method": "tcp", "host": "", "port": 29500},
            },
            "host",
        ),
        (
            {
                "worldSize": 1,
                "backend": "gloo",
                "rendezvous": {"method": "tcp", "host": "localhost", "port": 0},
            },
            "port",
        ),
        ({"worldSize": 1, "backend": "gloo", "rendezvous": {"method": "file", "path": ""}}, "path"),
        (
            {
                "worldSize": 1,
                "backend": "gloo",
                "rendezvous": {"method": "file", "path": "/tmp/group", "extra": True},
            },
            "unknown fields",
        ),
        (
            {
                "worldSize": 1,
                "backend": "gloo",
                "rendezvous": {"method": "file", "path": "/tmp/group"},
                "extra": True,
            },
            "unknown fields",
        ),
        ({"worldSize": 1, "backend": "gloo"}, "missing fields"),
    ],
)
def test_distributed_config_rejects_invalid_sections(
    factory: _ConfigMappingFactory,
    distributed: object,
    error: str,
) -> None:
    mapping = factory()
    mapping["distributed"] = distributed

    with pytest.raises(TrainingConfigError, match=error):
        _parse(factory, mapping)


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
def test_distributed_config_parses_tcp_and_file_rendezvous(
    factory: _ConfigMappingFactory,
    tmp_path: Path,
) -> None:
    mappings = []
    for rendezvous in (
        {"method": "tcp", "host": "127.0.0.1", "port": 29500},
        {"method": "file", "path": str((tmp_path / "group").resolve())},
    ):
        mapping = factory()
        mapping["distributed"] = {
            "worldSize": 2,
            "backend": "gloo",
            "rendezvous": rendezvous,
        }
        mappings.append(mapping)

    configs = [_parse(factory, mapping) for mapping in mappings]
    assert all(config.distributed is not None for config in configs)
    assert configs[0].to_mapping()["distributed"] == mappings[0]["distributed"]
    assert configs[1].to_mapping()["distributed"] == mappings[1]["distributed"]
    assert TcpRendezvousSettings("::1", 29500).url() == "tcp://[::1]:29500"


@pytest.mark.parametrize(
    ("factory", "expected_digest"),
    [
        (_sd_mapping, "blake3:77370f07dcf9535b1f7431053bc01572cfd56b0a39eff8f467b3159ff8ee8089"),
        (_h3_mapping, "blake3:687d24ccc03e54f5f1b2cf82c61dd0eca78e785378b1a6c4867d18835317f095"),
    ],
    ids=["sd", "h3"],
)
def test_absent_distributed_section_preserves_pinned_config_digest(
    factory: _ConfigMappingFactory,
    expected_digest: str,
) -> None:
    config = _parse(factory, factory())

    assert config.distributed is None
    assert "distributed" not in config.to_mapping()
    assert (
        training_service._config_digest(config)  # pyright: ignore[reportPrivateUsage]
        == expected_digest
    )


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
def test_rng_policy_defaults_to_sequential_and_round_trips_counter(
    factory: _ConfigMappingFactory,
) -> None:
    default = _parse(factory, factory())
    assert default.rng_policy == "sequential"
    assert "rngPolicy" not in default.to_mapping()

    mapping = factory()
    mapping["rngPolicy"] = "counter"
    counter = _parse(factory, mapping)
    assert counter.rng_policy == "counter"
    assert counter.to_mapping()["rngPolicy"] == "counter"


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
@pytest.mark.parametrize("value", ["rank-strided", "", None, 1])
def test_rng_policy_rejects_unknown_values(
    factory: _ConfigMappingFactory,
    value: object,
) -> None:
    mapping = factory()
    mapping["rngPolicy"] = value

    with pytest.raises(TrainingConfigError, match="rngPolicy must be 'sequential' or 'counter'"):
        _parse(factory, mapping)


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
def test_checkpoint_interval_defaults_to_zero_and_round_trips(
    factory: _ConfigMappingFactory,
) -> None:
    default = _parse(factory, factory())
    assert default.checkpoint_interval == 0
    assert "checkpointInterval" not in default.to_mapping()

    mapping = factory()
    mapping["checkpointInterval"] = 3
    periodic = _parse(factory, mapping)
    assert periodic.checkpoint_interval == 3
    assert periodic.to_mapping()["checkpointInterval"] == 3


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
@pytest.mark.parametrize("value", [-1, -100, "2", 1.5, None])
def test_checkpoint_interval_rejects_invalid_values(
    factory: _ConfigMappingFactory,
    value: object,
) -> None:
    mapping = factory()
    mapping["checkpointInterval"] = value

    with pytest.raises(TrainingConfigError, match="checkpointInterval must be an integer >= 0"):
        _parse(factory, mapping)


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
def test_checkpoint_interval_is_runtime_only_for_config_identity(
    factory: _ConfigMappingFactory,
) -> None:
    absent_digest = training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
        _parse(factory, factory())
    )
    mapping = factory()
    mapping["checkpointInterval"] = 5
    periodic = _parse(factory, mapping)

    assert "checkpointInterval" not in periodic.identity_mapping()
    assert (
        training_service._config_digest(periodic)  # pyright: ignore[reportPrivateUsage]
        == absent_digest
    )


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
def test_lora_export_interval_defaults_to_zero_and_round_trips(
    factory: _ConfigMappingFactory,
) -> None:
    default = _parse(factory, factory())
    assert default.lora_export_interval == 0
    assert "loraExportInterval" not in default.to_mapping()

    mapping = factory()
    mapping["loraExportInterval"] = 3
    periodic = _parse(factory, mapping)
    assert periodic.lora_export_interval == 3
    assert periodic.to_mapping()["loraExportInterval"] == 3


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
@pytest.mark.parametrize("value", [-1, -100, "2", 1.5, None])
def test_lora_export_interval_rejects_invalid_values(
    factory: _ConfigMappingFactory,
    value: object,
) -> None:
    mapping = factory()
    mapping["loraExportInterval"] = value

    with pytest.raises(TrainingConfigError, match="loraExportInterval must be an integer >= 0"):
        _parse(factory, mapping)


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
def test_lora_export_interval_is_runtime_only_for_config_identity(
    factory: _ConfigMappingFactory,
) -> None:
    absent_digest = training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
        _parse(factory, factory())
    )
    mapping = factory()
    mapping["loraExportInterval"] = 5
    periodic = _parse(factory, mapping)

    assert "loraExportInterval" not in periodic.identity_mapping()
    assert (
        training_service._config_digest(periodic)  # pyright: ignore[reportPrivateUsage]
        == absent_digest
    )


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
def test_sync_digest_interval_defaults_to_zero_and_round_trips(
    factory: _ConfigMappingFactory,
) -> None:
    default = _parse(factory, factory())
    assert default.sync_digest_interval == 0
    assert "syncDigestInterval" not in default.to_mapping()

    mapping = factory()
    mapping["syncDigestInterval"] = 3
    periodic = _parse(factory, mapping)
    assert periodic.sync_digest_interval == 3
    assert periodic.to_mapping()["syncDigestInterval"] == 3
    assert "syncDigestInterval" not in periodic.identity_mapping()


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
@pytest.mark.parametrize("value", [-1, -100, "2", 1.5, None])
def test_sync_digest_interval_rejects_invalid_values(
    factory: _ConfigMappingFactory,
    value: object,
) -> None:
    mapping = factory()
    mapping["syncDigestInterval"] = value

    with pytest.raises(TrainingConfigError, match="syncDigestInterval must be an integer >= 0"):
        _parse(factory, mapping)


@pytest.mark.parametrize("factory", [_sd_mapping, _h3_mapping], ids=["sd", "h3"])
def test_distributed_identity_keeps_only_world_size(factory: _ConfigMappingFactory) -> None:
    base = factory()
    variants = []
    for world_size, backend, rendezvous in (
        (1, "gloo", {"method": "file", "path": "/tmp/first-group"}),
        (1, "nccl", {"method": "tcp", "host": "localhost", "port": 29500}),
        (2, "gloo", {"method": "file", "path": "/tmp/second-group"}),
    ):
        mapping = copy.deepcopy(base)
        mapping["distributed"] = {
            "worldSize": world_size,
            "backend": backend,
            "rendezvous": rendezvous,
        }
        variants.append(_parse(factory, mapping))

    digests = [
        training_service._config_digest(config)  # pyright: ignore[reportPrivateUsage]
        for config in variants
    ]
    absent_digest = training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
        _parse(factory, base)
    )
    assert digests[0] == digests[1]
    assert digests[2] != digests[0]
    assert digests[0] != absent_digest
    assert variants[0].identity_mapping()["distributed"] == {"worldSize": 1}
    assert variants[0].to_mapping()["distributed"] != variants[1].to_mapping()["distributed"]


def _gloo_worker(rank: int, rendezvous_path: str) -> None:
    settings = DistributedSettings(
        world_size=2,
        backend="gloo",
        rendezvous=FileRendezvousSettings(rendezvous_path),
    )
    with bootstrap_rank_context(settings, rank) as context:
        context_as_protocol: RankContext = context
        assert context_as_protocol is context
        assert context.rank == rank
        assert context.world_size == 2

        broadcast = torch.nn.Parameter(torch.tensor([float(rank + 1)]))
        context.broadcast_parameters((broadcast,))
        assert torch.equal(broadcast, torch.tensor([1.0]))
        context.barrier()

        equal = torch.nn.Parameter(torch.zeros(1))
        equal.grad = torch.tensor([3.0])
        differing = torch.nn.Parameter(torch.zeros(2))
        differing.grad = torch.tensor([1.0, 3.0]) + rank * 2.0
        equal_gradient = equal.grad
        differing_gradient = differing.grad

        context.all_reduce_gradients((equal, differing))

        assert equal.grad is equal_gradient
        assert differing.grad is differing_gradient
        assert torch.equal(equal.grad, torch.tensor([3.0]))
        assert torch.equal(differing.grad, torch.tensor([2.0, 4.0]))


def test_two_process_gloo_rank_context_averages_gradients_in_place(tmp_path: Path) -> None:
    spawn(_gloo_worker, args=(str(tmp_path / "gloo-rendezvous"),), nprocs=2, join=True)


class _RecordingDistributed:
    def __init__(self, peer_gradients: torch.Tensor) -> None:
        self.peer_gradients = peer_gradients
        self.all_reduce_calls = 0

    def get_world_size(self) -> int:
        return 2

    def get_rank(self) -> int:
        return 0

    def all_reduce(self, tensor: torch.Tensor) -> None:
        self.all_reduce_calls += 1
        tensor.add_(self.peer_gradients)


def test_rank_context_flattens_present_gradients_into_one_all_reduce() -> None:
    first = torch.nn.Parameter(torch.zeros(2))
    first.grad = torch.tensor([1.0, 3.0])
    missing = torch.nn.Parameter(torch.zeros(1))
    last = torch.nn.Parameter(torch.zeros(2, 2))
    last.grad = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
    first_gradient = first.grad
    last_gradient = last.grad
    distributed = _RecordingDistributed(torch.tensor([5.0, 7.0, 4.0, 8.0, 10.0, 12.0]))
    context = TorchDistributedRankContext(distributed)  # type: ignore[arg-type]

    context.all_reduce_gradients((first, missing, last))

    assert distributed.all_reduce_calls == 1
    assert first.grad is first_gradient
    assert last.grad is last_gradient
    assert torch.equal(first.grad, torch.tensor([3.0, 5.0]))
    assert torch.equal(last.grad, torch.tensor([[3.0, 6.0], [8.0, 10.0]]))


def test_distributed_module_import_does_not_import_torch_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "dinkster_training_torch.distributed"
    existing = sys.modules.pop(module_name)
    imported: list[str] = []
    original_import = builtins.__import__

    def tracked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "torch.distributed":
            imported.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracked_import)
    try:
        importlib.import_module(module_name)
    finally:
        sys.modules[module_name] = existing
    assert imported == []


def test_bootstrap_imports_torch_distributed_only_after_rank_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = DistributedSettings(
        world_size=1,
        backend="gloo",
        rendezvous=FileRendezvousSettings("/unused"),
    )
    imported: list[str] = []
    original_import = builtins.__import__

    def tracked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "torch.distributed":
            imported.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", tracked_import)
    with pytest.raises(ValueError, match="rank must be"):
        bootstrap_rank_context(settings, -1)
    assert imported == []


class _TinySDModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = torch.nn.Conv2d(4, 4, 1, bias=False)

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        del timesteps, context
        return self.proj_in(latents)


class _Batches:
    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        del cursor, generator, device
        return PreparedBatch(torch.zeros(1, 4, 4, 4), torch.zeros(1, 2, 8))


def test_training_service_accepts_multi_rank_dry_run_and_explicit_single_rank(
    tmp_path: Path,
) -> None:
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    service = SD15LoRATrainingService(
        store,
        tmp_path / "checkpoints",
        model_factory=lambda config: _TinySDModel(),
        data_source_factory=lambda config: _Batches(),
    )
    mapping = _sd_mapping()
    mapping["distributed"] = {
        "worldSize": 2,
        "backend": "gloo",
        "rendezvous": {
            "method": "file",
            "path": str((tmp_path / "multi-rank").resolve()),
        },
    }
    serialized = json.dumps(mapping)
    try:
        report = service.dry_run(serialized)
        assert report["distributed"] == mapping["distributed"]

        distributed = copy.deepcopy(mapping["distributed"])
        assert isinstance(distributed, dict)
        distributed["worldSize"] = 1
        mapping["distributed"] = distributed
        serialized = json.dumps(mapping)

        report = service.dry_run(serialized)
        handle, created_report = service.create("single-rank", serialized)
        assert report["distributed"] == distributed
        assert created_report["distributed"] == distributed
        assert handle.step_cursor == 0
    finally:
        store.close()
