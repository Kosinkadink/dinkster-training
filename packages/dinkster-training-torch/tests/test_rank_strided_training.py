"""CPU proofs for rank-strided trainer data and randomness streams."""

from __future__ import annotations

import hashlib

import pytest
import torch
from dinkster_inference import LatentStream, MultiStreamLatent
from dinkster_inference_torch.minimax_h3_dit import MiniMaxH3DiTConditioning
from dinkster_training_torch import (
    CounterRandomnessPolicy,
    MiniMaxH3LoRATrainer,
    MiniMaxH3PreparedBatch,
    MiniMaxH3TrainingConfig,
    PreparedBatch,
    RandomnessPolicy,
    RankStridedRandomnessPolicy,
    SD15LoRATrainer,
    SequentialRandomnessPolicy,
    TrainingConfig,
    merge_rank_rng_state_dicts,
)

_CounterDraws = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    torch.Tensor,
    torch.Tensor,
]


def _sd_config(
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
            "rendezvous": {"method": "file", "path": "/tmp/dinkster-rank-strided-test"},
        }
    if rng_policy != "sequential":
        mapping["rngPolicy"] = rng_policy
    return TrainingConfig.from_mapping(mapping)


class _TinySDModel(torch.nn.Module):
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


class _RecordingBatches:
    def __init__(self) -> None:
        self.cursors: list[int] = []

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        del generator, device
        self.cursors.append(cursor)
        latents = torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4) / 64.0
        context = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 16.0
        return PreparedBatch(latents + cursor * 0.01, context - cursor * 0.02)


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
    config: TrainingConfig,
    context: _FakeRankContext | None = None,
    randomness_policy: RandomnessPolicy | None = None,
) -> tuple[SD15LoRATrainer, _RecordingBatches]:
    batches = _RecordingBatches()
    trainer = SD15LoRATrainer(
        config,
        _TinySDModel(),
        batches,
        rank_context=context,
        randomness_policy=randomness_policy,
    )
    return trainer, batches


def _h3_config() -> MiniMaxH3TrainingConfig:
    return MiniMaxH3TrainingConfig.from_mapping(
        {
            "schemaVersion": 1,
            "family": "minimax-h3",
            "ditRole": "fl2va-dit",
            "ditIdentity": "native:dinkster.minimax_h3:" + "1" * 64,
            "conditionerIdentity": "native:dinkster.minimax_h3:" + "2" * 64,
            "distributed": {
                "worldSize": 2,
                "backend": "gloo",
                "rendezvous": {"method": "file", "path": "/tmp/dinkster-rank-strided-test"},
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
            "videoLatentShape": [1, 24, 1, 2, 2],
            "audioLatentShape": [1, 32, 2, 1],
            "conditionerShape": [1, 1, 5120],
        }
    )


class _TinyH3Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_projection = torch.nn.Linear(2, 2, bias=False)
        self.audio_projection = torch.nn.Linear(1, 1, bias=False)
        self.context_projection = torch.nn.Linear(5120, 1, bias=False)

    def forward(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        sigma: float,
        context: torch.Tensor,
        *,
        conditioning: MiniMaxH3DiTConditioning,
        sigmas: object,
    ) -> MultiStreamLatent[torch.Tensor]:
        del sigma, conditioning, sigmas
        context_value = self.context_projection(context).mean() * 0.001
        return MultiStreamLatent(
            (
                LatentStream(
                    "video", self.video_projection(latent.by_role("video")) + context_value
                ),
                LatentStream(
                    "audio", self.audio_projection(latent.by_role("audio")) + context_value
                ),
            )
        )


class _RecordingH3Batches:
    def __init__(self) -> None:
        self.cursors: list[int] = []

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxH3PreparedBatch:
        del generator, device
        self.cursors.append(cursor)
        return MiniMaxH3PreparedBatch(
            torch.full((1, 24, 1, 2, 2), cursor * 0.01),
            torch.full((1, 32, 2, 1), cursor * -0.01),
            torch.zeros(1, 1, 5120),
            MiniMaxH3DiTConditioning(frame_count=1, seed=99),
        )


class _RecordingCounterRandomnessPolicy(CounterRandomnessPolicy):
    def __init__(self, seed: int) -> None:
        super().__init__(seed, torch.device("cpu"))
        self.sd_draws: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def draw_sd_timestep_noise(
        self,
        data_cursor: int,
        *,
        timestep_count: int,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        draw = super().draw_sd_timestep_noise(
            data_cursor,
            timestep_count=timestep_count,
            latent_shape=latent_shape,
        )
        self.sd_draws[data_cursor] = (draw[0].clone(), draw[1].clone())
        return draw


def _expected_generator_state(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator.get_state()


def _assert_rng_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert tuple(left) == tuple(right)
    assert all(torch.equal(left[name], right[name]) for name in left)


def _counter_draws(
    policy: RandomnessPolicy,
    data_cursor: int,
) -> _CounterDraws:
    data = torch.rand((5,), generator=policy.data_generator(data_cursor))
    timesteps, noise = policy.draw_sd_timestep_noise(
        data_cursor,
        timestep_count=1000,
        latent_shape=(1, 4, 2, 2),
    )
    sigma_index, video_noise, audio_noise = policy.draw_minimax_h3_timestep_noise(
        data_cursor,
        sigma_count=1000,
        video_shape=(1, 4, 2, 2),
        audio_shape=(1, 3, 2),
    )
    return data, timesteps, noise, sigma_index, video_noise, audio_noise


def _assert_draws_equal(
    left: _CounterDraws,
    right: _CounterDraws,
) -> None:
    for left_value, right_value in zip(left, right, strict=True):
        if isinstance(left_value, torch.Tensor):
            assert isinstance(right_value, torch.Tensor)
            assert torch.equal(left_value, right_value)
        else:
            assert left_value == right_value


def test_rank_stream_seeds_match_independent_sha256_derivation() -> None:
    policies = [
        RankStridedRandomnessPolicy(1234, torch.device("cpu"), rank, 2) for rank in range(2)
    ]

    for rank, policy in enumerate(policies):
        states = policy.rng_state_dict()
        assert tuple(states) == ("data", "timestep-noise")
        for stream, state in states.items():
            digest = hashlib.sha256(f"1234:rng:{stream}:{rank}".encode()).digest()
            expected_seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
            assert torch.equal(state, _expected_generator_state(expected_seed))

    legacy = SequentialRandomnessPolicy(1234, torch.device("cpu")).rng_state_dict()
    rank_zero = policies[0].rng_state_dict()
    assert all(not torch.equal(rank_zero[name], legacy[name]) for name in rank_zero)


def test_counter_streams_are_cursor_keyed_and_rank_count_invariant() -> None:
    world_one, _ = _trainer(_sd_config(rng_policy="counter"))
    world_two = tuple(
        _trainer(
            _sd_config(2, rng_policy="counter"),
            _FakeRankContext(rank, 2),
        )[0]
        for rank in range(2)
    )
    trainers = (world_one, *world_two)
    assert all(type(trainer.randomness_policy) is CounterRandomnessPolicy for trainer in trainers)

    cursor = 17
    expected = _counter_draws(world_one.randomness_policy, cursor)
    for trainer in trainers:
        _assert_draws_equal(_counter_draws(trainer.randomness_policy, cursor), expected)

    digest = hashlib.sha256(f"1234:rng:data:cursor:{cursor}".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
    assert torch.equal(expected[0], torch.rand((5,), generator=generator))
    assert expected[3] == int(expected[1].item())
    assert torch.equal(expected[4], expected[2])

    different = _counter_draws(world_one.randomness_policy, cursor + 1)
    assert not torch.equal(expected[0], different[0])
    assert not torch.equal(expected[2], different[2])
    assert not torch.equal(expected[4], different[4])
    assert not torch.equal(expected[5], different[5])


def test_counter_trainers_draw_identical_inputs_for_the_same_sample_cursors() -> None:
    world_one_policy = _RecordingCounterRandomnessPolicy(1234)
    world_one, world_one_batches = _trainer(
        _sd_config(rng_policy="counter"),
        randomness_policy=world_one_policy,
    )
    world_one.train_step()
    world_one.train_step()

    rank_policies = tuple(_RecordingCounterRandomnessPolicy(1234) for _ in range(2))
    rank_batches: list[_RecordingBatches] = []
    for rank, policy in enumerate(rank_policies):
        trainer, batches = _trainer(
            _sd_config(2, rng_policy="counter"),
            _FakeRankContext(rank, 2),
            randomness_policy=policy,
        )
        trainer.train_step()
        rank_batches.append(batches)

    assert world_one_batches.cursors == [0, 1, 2, 3]
    assert [batches.cursors for batches in rank_batches] == [[0, 2], [1, 3]]
    distributed_draws = rank_policies[0].sd_draws | rank_policies[1].sd_draws
    assert set(world_one_policy.sd_draws) == set(distributed_draws) == set(range(4))
    for cursor, expected in world_one_policy.sd_draws.items():
        actual = distributed_draws[cursor]
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


def test_world_two_trainers_read_disjoint_rank_strided_cursors() -> None:
    rank_cursors: list[list[int]] = []
    for rank in range(2):
        trainer, batches = _trainer(_sd_config(2), _FakeRankContext(rank, 2))
        assert type(trainer.randomness_policy) is RankStridedRandomnessPolicy

        trainer.train_step()
        trainer.train_step()

        assert trainer.step_cursor == 2
        assert trainer.data_cursor == 8
        rank_cursors.append(batches.cursors)

    assert rank_cursors == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert set(rank_cursors[0]).isdisjoint(rank_cursors[1])
    assert sorted(rank_cursors[0] + rank_cursors[1]) == list(range(8))


def test_minimax_h3_trainer_uses_the_same_rank_strided_mapping() -> None:
    rank_cursors: list[list[int]] = []
    for rank in range(2):
        batches = _RecordingH3Batches()
        trainer = MiniMaxH3LoRATrainer(
            _h3_config(),
            _TinyH3Model(),
            batches,
            rank_context=_FakeRankContext(rank, 2),
        )
        assert type(trainer.randomness_policy) is RankStridedRandomnessPolicy

        trainer.train_step()
        trainer.train_step()

        assert trainer.data_cursor == 8
        rank_cursors.append(batches.cursors)

    assert rank_cursors == [[0, 2, 4, 6], [1, 3, 5, 7]]


@pytest.mark.parametrize("world_size", [None, 1], ids=["absent", "explicit-one"])
def test_world_one_defaults_to_byte_identical_sequential_rng_state(
    world_size: int | None,
) -> None:
    trainer, _ = _trainer(_sd_config(world_size))
    expected = SequentialRandomnessPolicy(1234, torch.device("cpu"))

    assert type(trainer.randomness_policy) is SequentialRandomnessPolicy
    _assert_rng_equal(trainer.rng_state_dict(), expected.rng_state_dict())


def test_explicit_randomness_policy_wins_for_multi_rank_trainer() -> None:
    policy = SequentialRandomnessPolicy(99, torch.device("cpu"))
    trainer, _ = _trainer(
        _sd_config(2, rng_policy="counter"),
        _FakeRankContext(1, 2),
        randomness_policy=policy,
    )

    assert trainer.randomness_policy is policy


@pytest.mark.parametrize(
    ("config_world_size", "rank", "context_world_size", "message"),
    [
        (2, 0, 3, "rank context world size 3 does not match configured world size 2"),
        (2, 2, 2, r"rank must be in \[0, 2\), got 2"),
        (None, 0, 2, "rank context world size 2 does not match configured world size 1"),
        (1, 0, 2, "rank context world size 2 does not match configured world size 1"),
    ],
    ids=["mismatch", "rank-out-of-range", "absent-is-one", "explicit-one"],
)
def test_trainer_rejects_rank_context_mismatches(
    config_world_size: int | None,
    rank: int,
    context_world_size: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _trainer(
            _sd_config(config_world_size),
            _FakeRankContext(rank, context_world_size),
        )


def test_world_two_checkpoint_restore_continues_exact_rank_streams() -> None:
    config = _sd_config(2)
    context = _FakeRankContext(1, 2)
    uninterrupted, _ = _trainer(config, context)
    uninterrupted_losses = tuple(uninterrupted.train_step() for _ in range(4))

    paused, paused_batches = _trainer(config, _FakeRankContext(1, 2))
    prefix = tuple(paused.train_step() for _ in range(2))
    rank_zero, _ = _trainer(config, _FakeRankContext(0, 2))
    for _ in range(2):
        rank_zero.train_step()
    checkpoint_rng = merge_rank_rng_state_dicts(
        (rank_zero.rng_state_dict(), paused.rng_state_dict()),
        world_size=2,
    )
    assert paused_batches.cursors == [1, 3, 5, 7]
    assert paused.data_cursor == 8

    resumed, resumed_batches = _trainer(config, _FakeRankContext(1, 2))
    resumed.restore(
        adapter=paused.attachment.state_dict(),
        optimizer=paused.optimizer_state_dict(),
        rng=checkpoint_rng,
        step_cursor=paused.step_cursor,
        data_cursor=paused.data_cursor,
        loss=paused.last_loss,
    )
    suffix = tuple(resumed.train_step() for _ in range(2))

    assert prefix + suffix == uninterrupted_losses
    assert resumed_batches.cursors == [9, 11, 13, 15]
    assert resumed.data_cursor == 16
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            resumed.attachment.state_dict().values(),
            uninterrupted.attachment.state_dict().values(),
            strict=True,
        )
    )
    _assert_rng_equal(resumed.rng_state_dict(), uninterrupted.rng_state_dict())

    invalid, _ = _trainer(config, _FakeRankContext(1, 2))
    with pytest.raises(
        ValueError,
        match="checkpoint data cursor 7 does not match optimizer step 2",
    ):
        invalid.restore(
            adapter=paused.attachment.state_dict(),
            optimizer=paused.optimizer_state_dict(),
            rng=checkpoint_rng,
            step_cursor=2,
            data_cursor=7,
            loss=paused.last_loss,
        )
