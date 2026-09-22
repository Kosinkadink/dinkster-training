"""Checkpoint schema proofs for single-rank and per-rank RNG streams."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from dinkster_training_torch import (
    ContentAddressedCheckpointStore,
    CounterRandomnessPolicy,
    PreparedBatch,
    SD15LoRATrainer,
    TrainingConfig,
    merge_rank_rng_state_dicts,
)

# Canonical golden platform: Linux x86-64 CPU torch. Windows CPU float
# differences produce a different manifest; regenerate only on Linux.
_WORLD_ONE_MANIFEST_DIGEST = (
    "blake3:0271f4e951123a2b614a8c6b0b6bfcc3034088a68696a4d8e0e1f1435efb3258"
)


def _config(
    world_size: int | None = None,
    *,
    rng_policy: str = "sequential",
) -> TrainingConfig:
    mapping: dict[str, object] = {
        "schemaVersion": 1,
        "family": "sd15",
        "unet": {
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
        },
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
    if world_size is not None:
        mapping["distributed"] = {
            "worldSize": world_size,
            "backend": "gloo",
            "rendezvous": {"method": "file", "path": "/tmp/dinkster-rank-rng-checkpoint"},
        }
    if rng_policy != "sequential":
        mapping["rngPolicy"] = rng_policy
    return TrainingConfig.from_mapping(mapping)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = torch.nn.Conv2d(4, 4, 1, bias=False)
        with torch.no_grad():
            values = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1)
            self.proj_in.weight.copy_(values / 16.0)

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


class _FakeRankContext:
    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size

    def all_reduce_gradients(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        del parameters

    def broadcast_parameters(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        del parameters

    def barrier(self) -> None:
        pass


def _trainer(
    *,
    world_size: int | None = None,
    rank: int = 0,
    rng_policy: str = "sequential",
) -> SD15LoRATrainer:
    context = None if world_size is None else _FakeRankContext(rank, world_size)
    return SD15LoRATrainer(
        _config(world_size, rng_policy=rng_policy),
        _TinyModel(),
        _Batches(),
        rank_context=context,
    )


def _draws(trainer: SD15LoRATrainer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = torch.rand(
        (5,),
        generator=trainer.randomness_policy.data_generator(0),
        device=trainer.device,
    )
    timesteps, noise = trainer.randomness_policy.draw_sd_timestep_noise(
        0,
        timestep_count=1000,
        latent_shape=(1, 4, 2, 2),
    )
    return data, timesteps, noise


def _world_two_states() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rank_zero = _trainer(world_size=2, rank=0)
    rank_one = _trainer(world_size=2, rank=1)
    return rank_zero.rng_state_dict(), rank_one.rng_state_dict()


def _write_checkpoint(
    root: Path,
    trainer: SD15LoRATrainer,
    rng: dict[str, torch.Tensor],
) -> tuple[ContentAddressedCheckpointStore, str]:
    store = ContentAddressedCheckpointStore(root)
    digest = store.write(
        session_id="a" * 64,
        config_digest="blake3:" + "b" * 64,
        extension_snapshot_digest="blake3:" + "c" * 64,
        parent_manifest_digest="",
        step_cursor=trainer.step_cursor,
        config=trainer.config.to_mapping(),
        adapter=trainer.attachment.state_dict(),
        optimizer=trainer.optimizer_state_dict(),
        rng=rng,
        data_cursor=trainer.data_cursor,
        loss=trainer.last_loss,
    )
    return store, digest


def test_world_one_checkpoint_manifest_digest_matches_origin_main(tmp_path: Path) -> None:
    trainer = _trainer()
    rng = trainer.rng_state_dict()
    assert tuple(rng) == ("data", "timestep-noise")

    store = ContentAddressedCheckpointStore(tmp_path / "checkpoints")
    digest = store.write(
        session_id="a" * 64,
        config_digest="blake3:" + "b" * 64,
        extension_snapshot_digest="blake3:" + "c" * 64,
        parent_manifest_digest="",
        step_cursor=trainer.step_cursor,
        config=trainer.config.to_mapping(),
        adapter=trainer.attachment.state_dict(),
        optimizer=trainer.optimizer_state_dict(),
        rng=rng,
        data_cursor=trainer.data_cursor,
        loss=trainer.last_loss,
    )

    assert digest == _WORLD_ONE_MANIFEST_DIGEST
    persisted = store.load(digest).rng
    assert tuple(persisted) == ("data", "timestep-noise")
    assert all(torch.equal(persisted[name], rng[name]) for name in rng)


def test_world_two_full_matrix_round_trip_preserves_exact_rank_draws() -> None:
    uninterrupted = (
        _trainer(world_size=2, rank=0),
        _trainer(world_size=2, rank=1),
    )
    for trainer in uninterrupted:
        _draws(trainer)

    matrix = merge_rank_rng_state_dicts(
        tuple(trainer.rng_state_dict() for trainer in uninterrupted),
        world_size=2,
    )
    assert tuple(matrix) == (
        "rank0:data",
        "rank0:timestep-noise",
        "rank1:data",
        "rank1:timestep-noise",
    )
    expected = tuple(_draws(trainer) for trainer in uninterrupted)

    resumed = (
        _trainer(world_size=2, rank=0),
        _trainer(world_size=2, rank=1),
    )
    for rank, trainer in enumerate(resumed):
        trainer.load_rng_state_dict(matrix)
        actual = _draws(trainer)
        assert all(
            torch.equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected[rank], strict=True)
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("incomplete", "RNG checkpoint rank matrix differs"),
        ("mixed", "RNG checkpoint mixes bare and rank-prefixed stream keys"),
        ("out-of-range", "RNG checkpoint rank 2 is outside configured world size 2"),
        ("bare", "RNG checkpoint for world size 2 requires rank-prefixed stream keys"),
    ],
)
def test_world_two_rejects_invalid_rng_checkpoint_matrices(case: str, message: str) -> None:
    matrix = merge_rank_rng_state_dicts(_world_two_states(), world_size=2)
    state = dict(matrix)
    if case == "incomplete":
        state.pop("rank1:timestep-noise")
    elif case == "mixed":
        state["data"] = state["rank0:data"]
    elif case == "out-of-range":
        state["rank2:data"] = state["rank0:data"]
    else:
        state = _trainer().rng_state_dict()

    trainer = _trainer(world_size=2, rank=0)
    with pytest.raises(ValueError, match=message):
        trainer.load_rng_state_dict(state)


def test_world_one_rejects_rank_prefixed_rng_checkpoint() -> None:
    matrix = merge_rank_rng_state_dicts(_world_two_states(), world_size=2)

    with pytest.raises(ValueError, match="single-rank RNG checkpoint requires bare stream keys"):
        _trainer().load_rng_state_dict(matrix)


def test_rank_rng_merge_requires_one_complete_state_per_rank() -> None:
    rank_zero, rank_one = _world_two_states()
    with pytest.raises(ValueError, match="expected one RNG state per rank for world size 2, got 1"):
        merge_rank_rng_state_dicts((rank_zero,), world_size=2)

    incomplete_rank_one = dict(rank_one)
    incomplete_rank_one.pop("rank1:timestep-noise")
    with pytest.raises(ValueError, match="RNG state for rank 1 differs"):
        merge_rank_rng_state_dicts((rank_zero, incomplete_rank_one), world_size=2)


@pytest.mark.parametrize(
    ("source_world_size", "target_world_size"),
    [(2, None), (None, 2)],
    ids=["world-two-to-one", "world-one-to-two"],
)
def test_counter_rng_checkpoint_restores_across_world_sizes(
    tmp_path: Path,
    source_world_size: int | None,
    target_world_size: int | None,
) -> None:
    source = _trainer(world_size=source_world_size, rng_policy="counter")
    source.train_step()
    rng = source.rng_state_dict()
    if source_world_size == 2:
        rank_one = _trainer(world_size=2, rank=1, rng_policy="counter")
        rng = merge_rank_rng_state_dicts((rng, rank_one.rng_state_dict()), world_size=2)

    store, digest = _write_checkpoint(tmp_path / "checkpoints", source, rng)
    state = store.load(digest)
    assert tuple(state.rng) == ("__dinkster_counter_rng_policy__",)
    assert torch.equal(
        state.rng["__dinkster_counter_rng_policy__"],
        torch.tensor([1], dtype=torch.uint8),
    )

    world_size = 1 if target_world_size is None else target_world_size
    for rank in range(world_size):
        resumed = _trainer(
            world_size=target_world_size,
            rank=rank,
            rng_policy="counter",
        )
        resumed.restore(
            adapter=state.adapter,
            optimizer=state.optimizer,
            rng=state.rng,
            step_cursor=state.step_cursor,
            data_cursor=state.data_cursor,
            loss=state.loss,
        )
        assert resumed.step_cursor == state.step_cursor
        assert resumed.data_cursor == state.data_cursor
        resumed.train_step()
        assert resumed.step_cursor == state.step_cursor + 1
        assert resumed.data_cursor == state.data_cursor + 2 * world_size
        assert tuple(resumed.rng_state_dict()) == ("__dinkster_counter_rng_policy__",)


def test_counter_rng_checkpoint_rejects_a_nonzero_cursor_at_step_zero() -> None:
    trainer = _trainer(rng_policy="counter")
    with pytest.raises(
        ValueError,
        match="counter RNG checkpoint data cursor 2 is invalid for optimizer step 0",
    ):
        trainer.restore(
            adapter=trainer.attachment.state_dict(),
            optimizer=trainer.optimizer_state_dict(),
            rng=trainer.rng_state_dict(),
            step_cursor=0,
            data_cursor=2,
            loss=None,
        )


@pytest.mark.parametrize("state_shape", ["bare", "rank-matrix"])
def test_counter_rng_policy_rejects_stateful_checkpoints(state_shape: str) -> None:
    if state_shape == "bare":
        state = _trainer().rng_state_dict()
    else:
        state = merge_rank_rng_state_dicts(_world_two_states(), world_size=2)

    counter = _trainer(world_size=2, rng_policy="counter")
    with pytest.raises(
        ValueError, match="counter RNG checkpoint must contain only the policy marker"
    ):
        counter.load_rng_state_dict(state)


@pytest.mark.parametrize("world_size", [None, 2], ids=["sequential", "rank-strided"])
def test_stateful_rng_policies_reject_counter_checkpoint(world_size: int | None) -> None:
    marker = CounterRandomnessPolicy(1234, torch.device("cpu")).rng_state_dict()
    trainer = _trainer(world_size=world_size)

    with pytest.raises(ValueError, match="counter RNG checkpoint requires the counter RNG policy"):
        trainer.load_rng_state_dict(marker)


@pytest.mark.parametrize(
    "marker",
    [
        torch.tensor([2], dtype=torch.uint8),
        torch.tensor(1, dtype=torch.uint8),
        torch.tensor([1], dtype=torch.int64),
    ],
    ids=["version", "shape", "dtype"],
)
def test_counter_rng_policy_strictly_validates_its_marker(marker: torch.Tensor) -> None:
    policy = CounterRandomnessPolicy(1234, torch.device("cpu"))
    with pytest.raises(ValueError, match="policy marker must be a CPU uint8 tensor containing"):
        policy.load_rng_state_dict({"__dinkster_counter_rng_policy__": marker})
