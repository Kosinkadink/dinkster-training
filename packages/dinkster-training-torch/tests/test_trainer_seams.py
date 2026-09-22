"""Behavior locks for trainer rank, cursor, and randomness seams."""

from __future__ import annotations

import math

import pytest
import torch
from dinkster_inference import LatentStream, MultiStreamLatent
from dinkster_inference_torch.minimax_h3_dit import MiniMaxH3DiTConditioning
from dinkster_training_torch import (
    MiniMaxH3LoRATrainer,
    MiniMaxH3PreparedBatch,
    MiniMaxH3TrainingConfig,
    PreparedBatch,
    RandomnessPolicy,
    RankContext,
    SD15LoRATrainer,
    SequentialRandomnessPolicy,
    SingleRankContext,
    TrainingConfig,
)

_PINNED_SD_LOSSES = (
    0.9912053942680359,
    1.717738151550293,
    1.7654621601104736,
    1.3517751693725586,
)
_PINNED_COUNTER_SD_LOSSES = (
    1.4804391860961914,
    2.140136957168579,
    1.5129215717315674,
    2.1759910583496094,
)


def _sd_config() -> TrainingConfig:
    return TrainingConfig.from_mapping(
        {
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
    )


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


class _SDBatches:
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


def _sd_trainer(
    *,
    rank_context: RankContext | None = None,
    randomness_policy: RandomnessPolicy | None = None,
) -> tuple[SD15LoRATrainer, _SDBatches]:
    batches = _SDBatches()
    trainer = SD15LoRATrainer(
        _sd_config(),
        _TinySDModel(),
        batches,
        rank_context=rank_context,
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
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters(), 1):
                values = torch.arange(parameter.numel(), dtype=torch.float32)
                parameter.copy_((values.reshape(parameter.shape) + index) / parameter.numel())

    def forward(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        sigma: float,
        context: torch.Tensor,
        *,
        conditioning: MiniMaxH3DiTConditioning,
        sigmas: object,
    ) -> MultiStreamLatent[torch.Tensor]:
        del conditioning, sigmas
        context_value = self.context_projection(context).mean() * 0.001 + sigma * 0.001
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


class _H3Batches:
    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxH3PreparedBatch:
        del generator, device
        config = _h3_config()
        video = torch.arange(math.prod(config.video_latent_shape), dtype=torch.float32)
        audio = torch.arange(math.prod(config.audio_latent_shape), dtype=torch.float32)
        context = torch.arange(math.prod(config.conditioner_shape), dtype=torch.float32)
        return MiniMaxH3PreparedBatch(
            video.reshape(config.video_latent_shape) / 100.0 + cursor * 0.01,
            audio.reshape(config.audio_latent_shape) / 100.0 - cursor * 0.01,
            context.reshape(config.conditioner_shape) / 5120.0,
            MiniMaxH3DiTConditioning(frame_count=1, seed=99),
        )


class _RecordingRankContext:
    world_size = 1
    rank = 0

    def __init__(self) -> None:
        self.gradient_reductions = 0

    def all_reduce_gradients(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        assert parameters and all(parameter.grad is not None for parameter in parameters)
        self.gradient_reductions += 1

    def broadcast_parameters(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        del parameters

    def barrier(self) -> None:
        pass


class _RecordingRandomnessPolicy:
    def __init__(self, seed: int) -> None:
        self.inner = SequentialRandomnessPolicy(seed, torch.device("cpu"))
        self.data_cursors: list[int] = []
        self.sd_cursors: list[int] = []
        self.h3_cursors: list[int] = []

    def data_generator(self, data_cursor: int) -> torch.Generator:
        self.data_cursors.append(data_cursor)
        return self.inner.data_generator(data_cursor)

    def draw_sd_timestep_noise(
        self,
        data_cursor: int,
        *,
        timestep_count: int,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.sd_cursors.append(data_cursor)
        return self.inner.draw_sd_timestep_noise(
            data_cursor,
            timestep_count=timestep_count,
            latent_shape=latent_shape,
        )

    def draw_minimax_h3_timestep_noise(
        self,
        data_cursor: int,
        *,
        sigma_count: int,
        video_shape: tuple[int, ...],
        audio_shape: tuple[int, ...],
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        self.h3_cursors.append(data_cursor)
        return self.inner.draw_minimax_h3_timestep_noise(
            data_cursor,
            sigma_count=sigma_count,
            video_shape=video_shape,
            audio_shape=audio_shape,
        )

    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        return self.inner.rng_state_dict()

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.inner.load_rng_state_dict(state)


class _OffsetCursorTrainer(SD15LoRATrainer):
    def data_cursor_for_micro_batch(self, micro_batch_index: int) -> int:
        return micro_batch_index + 10


def _assert_rng_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    assert all(torch.equal(left[name], right[name]) for name in left)


def test_default_single_rank_trainer_preserves_the_pinned_cpu_trajectory() -> None:
    trainer, batches = _sd_trainer()

    assert type(trainer.rank_context) is SingleRankContext
    assert trainer.rank_context.world_size == 1
    assert trainer.rank_context.rank == 0
    assert tuple(trainer.train_step() for _ in range(4)) == _PINNED_SD_LOSSES
    assert batches.cursors == list(range(8))

    parameters = trainer.attachment.parameters()
    trainer.rank_context.broadcast_parameters(parameters)
    trainer.rank_context.barrier()


def test_counter_policy_has_a_pinned_cpu_trajectory() -> None:
    mapping = _sd_config().to_mapping()
    mapping["rngPolicy"] = "counter"
    trainer = SD15LoRATrainer(
        TrainingConfig.from_mapping(mapping),
        _TinySDModel(),
        _SDBatches(),
    )

    assert tuple(trainer.train_step() for _ in range(4)) == _PINNED_COUNTER_SD_LOSSES


def test_rng_policy_state_round_trip_preserves_exact_training_continuation() -> None:
    uninterrupted, _ = _sd_trainer()
    uninterrupted_losses = tuple(uninterrupted.train_step() for _ in range(4))

    paused, _ = _sd_trainer()
    prefix = tuple(paused.train_step() for _ in range(2))
    checkpoint_rng = paused.rng_state_dict()
    assert tuple(checkpoint_rng) == ("data", "timestep-noise")

    resumed, _ = _sd_trainer()
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
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            resumed.attachment.state_dict().values(),
            uninterrupted.attachment.state_dict().values(),
            strict=True,
        )
    )
    _assert_rng_equal(resumed.rng_state_dict(), uninterrupted.rng_state_dict())

    with pytest.raises(
        ValueError,
        match=r"RNG checkpoint streams differ: expected \['data', 'timestep-noise'\],"
        r" got \['data'\]",
    ):
        resumed.load_rng_state_dict({"data": checkpoint_rng["data"]})


def test_rank_cursor_and_randomness_seams_receive_each_micro_batch() -> None:
    rank_context = _RecordingRankContext()
    sd_policy = _RecordingRandomnessPolicy(1234)
    sd_batches = _SDBatches()
    sd_trainer = _OffsetCursorTrainer(
        _sd_config(),
        _TinySDModel(),
        sd_batches,
        rank_context=rank_context,
        randomness_policy=sd_policy,
    )

    assert math.isfinite(sd_trainer.train_step())
    assert sd_trainer.data_cursor == 2
    assert sd_batches.cursors == [10, 11]
    assert sd_policy.data_cursors == [10, 11]
    assert sd_policy.sd_cursors == [10, 11]
    assert sd_policy.h3_cursors == []
    assert rank_context.gradient_reductions == 1

    h3_policy = _RecordingRandomnessPolicy(1234)
    h3_trainer = MiniMaxH3LoRATrainer(
        _h3_config(),
        _TinyH3Model(),
        _H3Batches(),
        randomness_policy=h3_policy,
    )

    assert math.isfinite(h3_trainer.train_step())
    assert h3_policy.data_cursors == [0, 1]
    assert h3_policy.sd_cursors == []
    assert h3_policy.h3_cursors == [0, 1]
