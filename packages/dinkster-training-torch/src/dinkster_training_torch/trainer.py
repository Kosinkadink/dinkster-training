"""Native SD epsilon-prediction and Wan/H3 flow-matching LoRA trainers."""

from __future__ import annotations

import hashlib
import math
import weakref
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Generic, Protocol, TypeVar, cast

import torch
import torch.nn.functional as functional
from dinkster_inference import (
    BFLOAT16,
    CLIP_L_TEXT_CONFIG,
    FLOAT16,
    FLOAT32,
    FLUX2_DEV,
    FLUX2_DEV_CONFIG,
    FLUX2_KLEIN_4B,
    FLUX2_KLEIN_4B_CONFIG,
    FLUX2_KLEIN_9B,
    FLUX2_KLEIN_9B_CONFIG,
    FLUX_DEV,
    FLUX_DEV_CONFIG,
    FLUX_SCHNELL,
    FLUX_SCHNELL_CONFIG,
    KLEIN_QWEN3_4B_CONFIG,
    KLEIN_QWEN3_8B_CONFIG,
    MINIMAX_H3,
    MINIMAX_H3_SIGMAS,
    MINIMAX_MUSIC3,
    MISTRAL3_24B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
    QWEN_IMAGE,
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_TEXT_CONFIG,
    T5_XXL_CONFIG,
    WAN21_SIGMAS,
    WAN21_T2V_1_3B,
    WAN21_T2V_14B,
    WAN21_VAE_CONFIG,
    WAN22_I2V_14B,
    WAN22_TI2V_5B,
    ComponentPlan,
    Flux2AssemblyPlan,
    Flux2PlannedComponent,
    FluxAssemblyPlan,
    FluxFlowSigmas,
    LatentStream,
    MiniMaxH3ModelAssemblyPlan,
    MiniMaxH3TimeEmbeddingKind,
    ModelFamily,
    MultiStreamLatent,
    QwenImageAssemblyPlan,
    SafetensorsSource,
    Wan21AssemblyPlan,
    Wan21VAEConfig,
    Wan22VAEConfig,
    flux2_component_runtime_identity,
    ideogram4_component_runtime_identity,
    ideogram4_component_uses_fp8_matmul,
    linear_beta_sigmas,
    load_safetensors_header,
    plan_flux2_assembly,
    plan_flux2_split_component,
    plan_flux_assembly,
    plan_ideogram4_split_component,
    plan_qwen_image_assembly,
    plan_sd_assembly,
    plan_wan21_assembly,
    plan_wan22_assembly,
)
from dinkster_inference_torch import (
    AttentionGuidanceContext,
    AttentionRole,
    AutoencoderKL,
    CastOperations,
    ClipTextModel,
    Flux,
    Fp8Linear,
    Ideogram4Block,
    Ideogram4DiT,
    Ideogram4TextEncoder,
    Int8Linear,
    MiniMaxH3DiTConditioning,
    MiniMaxH3KeyframeLatent,
    MiniMaxH3ReferenceLatents,
    MiniMaxMusic3DiT,
    MiniMaxMusic3TextModel,
    QwenImage,
    QwenImageLanguageModel,
    QwenImageTextModel,
    QwenImageTransformerBlock,
    QwenTextModel,
    SD15AttentionExecutionContext,
    T5TextModel,
    UNetModel,
    Wan21Model,
    Wan21MultiTalkExecution,
    Wan21TextRuntime,
    Wan22VAE,
    WanAttentionBlock,
    WanVAE,
    bind_fp8_matmul_layer,
    encode_sdxl_adm,
    load_flux2_component,
    load_ideogram4_component,
    load_minimax_h3_model,
    load_minimax_music3_component,
    load_tensors,
    plan_minimax_h3_model_assembly,
    simple_schedule,
)
from dinkster_inference_torch import (
    WanVAEConfig as TorchWanVAEConfig,
)
from tokenizers import Tokenizer
from torch.utils.checkpoint import checkpoint

from .attachment import TrainableAttachment
from .config import (
    Flux2TrainingConfig,
    Flux2VariantName,
    FluxTrainingConfig,
    Ideogram4TrainingConfig,
    MiniMaxH3Int8BaseForward,
    MiniMaxH3TrainingConfig,
    MiniMaxMusic3TrainingConfig,
    NativeTrainingArtifactSource,
    QwenImageTrainingConfig,
    TrainingArtifactSource,
    TrainingConfig,
    WanTrainingConfig,
)
from .data import (
    QWEN_IMAGE_TEXT_ENCODING_DTYPE,
    WAN21_ENCODING_DTYPE,
    FilesystemMiniMaxH3PreparedBatchSource,
    FilesystemPreparedBatchSource,
    Flux2DatasetSource,
    Flux2PreparedBatch,
    Flux2PreparedBatchSource,
    FluxDatasetSource,
    FluxPreparedBatch,
    FluxPreparedBatchSource,
    Ideogram4DatasetSource,
    Ideogram4PreparedBatch,
    Ideogram4PreparedBatchSource,
    MiniMaxH3PreparedBatch,
    MiniMaxH3PreparedBatchSource,
    PreparedBatch,
    PreparedBatchSource,
    QwenImageDatasetSource,
    QwenImagePreparedBatch,
    QwenImagePreparedBatchSource,
    WanDatasetSource,
    WanPreparedBatch,
    WanPreparedBatchSource,
    default_image_caption_source,
    default_minimax_h3_dataset_source,
    fixed_asset_ref,
    load_planned_component,
    validate_sdxl_training_plan,
)
from .minimax_music3_training import (
    MiniMaxMusic3DatasetSource,
    MiniMaxMusic3PreparedBatch,
    MiniMaxMusic3PreparedBatchSource,
    load_minimax_music3_dav_encoder,
    load_minimax_music3_rvq_encoder,
)
from .optimizer import FactoredAdamW
from .paging import FrozenLayerPager

_RNG_STREAMS = ("data", "timestep-noise")
_COUNTER_RNG_POLICY_KEY = "__dinkster_counter_rng_policy__"
_COUNTER_RNG_POLICY_VERSION = 1


def _wan_expert_sigma_indices(
    table: tuple[float, ...],
    multiplier: float,
    expert: str,
    timestep_range: tuple[int, int],
) -> tuple[int, ...]:
    upper, lower = timestep_range
    if expert == "high-noise":
        return tuple(
            index for index, sigma in enumerate(table) if lower <= sigma * multiplier <= upper
        )
    return tuple(index for index, sigma in enumerate(table) if lower <= sigma * multiplier < upper)


def _checkpoint_sd_unet_blocks(model: UNetModel) -> None:
    def checkpointed(
        original_forward: Callable[..., torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        def forward(
            _block: torch.nn.Module,
            x: torch.Tensor,
            emb: torch.Tensor,
            context: torch.Tensor,
            output_shape: tuple[int, ...] | None = None,
            attention_guidance: AttentionGuidanceContext | None = None,
            ipadapter: SD15AttentionExecutionContext | None = None,
        ) -> torch.Tensor:
            del _block
            # Reentrant block boundaries change BF16 gradient accumulation
            # order relative to one unpartitioned autograd graph.
            return checkpoint(
                original_forward,
                x,
                emb,
                context,
                output_shape,
                attention_guidance,
                ipadapter,
                use_reentrant=True,
                preserve_rng_state=False,
            )

        return forward

    blocks = (*model.input_blocks, model.middle_block, *model.output_blocks)
    for block in blocks:
        block.forward = MethodType(  # pyright: ignore[reportAttributeAccessIssue]
            checkpointed(block.forward), block
        )


def _checkpoint_wan_blocks(model: Wan21Model) -> None:
    def checkpointed(
        original_forward: Callable[..., torch.Tensor],
        block: WanAttentionBlock,
    ) -> Callable[..., torch.Tensor]:
        def forward(
            x: torch.Tensor,
            time: torch.Tensor,
            freqs: torch.Tensor,
            context: torch.Tensor,
            image_rows: int | None,
            *,
            multitalk: Wan21MultiTalkExecution | None = None,
            multitalk_block_index: int = 0,
            grid_shape: tuple[int, int, int] | None = None,
        ) -> torch.Tensor:
            def evaluate(
                x_value: torch.Tensor,
                time_value: torch.Tensor,
                freqs_value: torch.Tensor,
                context_value: torch.Tensor,
            ) -> torch.Tensor:
                return original_forward(
                    block,
                    x_value,
                    time_value,
                    freqs_value,
                    context_value,
                    image_rows,
                    multitalk=multitalk,
                    multitalk_block_index=multitalk_block_index,
                    grid_shape=grid_shape,
                )

            return checkpoint(
                evaluate,
                x,
                time,
                freqs,
                context,
                use_reentrant=False,
                preserve_rng_state=True,
            )

        return forward

    for block in model.blocks:
        block.forward = checkpointed(  # pyright: ignore[reportAttributeAccessIssue]
            type(block).forward,
            cast("WanAttentionBlock", weakref.proxy(block)),
        )


def _checkpoint_flux_blocks(model: Flux) -> None:
    def double_forward(
        original: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        def forward(
            img: torch.Tensor, txt: torch.Tensor, vec: object, pe: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return checkpoint(
                original,
                img,
                txt,
                vec,
                pe,
                use_reentrant=False,
                preserve_rng_state=True,
            )

        return forward

    def single_forward(original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def forward(x: torch.Tensor, vec: object, pe: torch.Tensor) -> torch.Tensor:
            return checkpoint(
                original,
                x,
                vec,
                pe,
                use_reentrant=False,
                preserve_rng_state=True,
            )

        return forward

    for block in model.double_blocks:
        block.forward = double_forward(block.forward)  # pyright: ignore[reportAttributeAccessIssue]
    for block in model.single_blocks:
        block.forward = single_forward(block.forward)  # pyright: ignore[reportAttributeAccessIssue]


def _checkpoint_qwen_image_blocks(model: QwenImage) -> None:
    def checkpointed(
        original: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        def forward(
            image: torch.Tensor,
            text: torch.Tensor,
            temb: torch.Tensor,
            frequencies: torch.Tensor,
            mask: torch.Tensor | None,
            timestep_zero_index: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return checkpoint(
                original,
                image,
                text,
                temb,
                frequencies,
                mask,
                timestep_zero_index,
                use_reentrant=False,
                preserve_rng_state=True,
            )

        return forward

    for block in model.transformer_blocks:
        typed = cast("QwenImageTransformerBlock", block)
        typed.forward = checkpointed(typed.forward)  # pyright: ignore[reportAttributeAccessIssue]


def _checkpoint_ideogram4_blocks(model: Ideogram4DiT) -> None:
    def checkpointed(original: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        def forward(
            hidden: torch.Tensor,
            mask: torch.Tensor | None,
            rope: torch.Tensor,
            adaln: torch.Tensor,
        ) -> torch.Tensor:
            def evaluate(
                hidden_value: torch.Tensor,
                rope_value: torch.Tensor,
                adaln_value: torch.Tensor,
            ) -> torch.Tensor:
                return original(hidden_value, mask, rope_value, adaln_value)

            return checkpoint(
                evaluate,
                hidden,
                rope,
                adaln,
                use_reentrant=False,
                preserve_rng_state=True,
            )

        return forward

    for block in model.layers:
        typed = cast("Ideogram4Block", block)
        typed.forward = checkpointed(typed.forward)  # pyright: ignore[reportAttributeAccessIssue]


class ModelFactory(Protocol):
    def __call__(self, config: TrainingConfig) -> torch.nn.Module: ...


class DataSourceFactory(Protocol):
    def __call__(self, config: TrainingConfig) -> PreparedBatchSource: ...


class MiniMaxH3ModelFactory(Protocol):
    def __call__(self, config: MiniMaxH3TrainingConfig) -> torch.nn.Module: ...


class MiniMaxH3DataSourceFactory(Protocol):
    def __call__(self, config: MiniMaxH3TrainingConfig) -> MiniMaxH3PreparedBatchSource: ...


class MiniMaxMusic3ModelFactory(Protocol):
    def __call__(self, config: MiniMaxMusic3TrainingConfig) -> MiniMaxMusic3DiT: ...


class MiniMaxMusic3DataSourceFactory(Protocol):
    def __call__(self, config: MiniMaxMusic3TrainingConfig) -> MiniMaxMusic3PreparedBatchSource: ...


class WanModelFactory(Protocol):
    def __call__(self, config: WanTrainingConfig) -> Wan21Model: ...


class WanDataSourceFactory(Protocol):
    def __call__(self, config: WanTrainingConfig) -> WanPreparedBatchSource: ...


class FluxModelFactory(Protocol):
    def __call__(self, config: FluxTrainingConfig) -> Flux: ...


class FluxDataSourceFactory(Protocol):
    def __call__(self, config: FluxTrainingConfig) -> FluxPreparedBatchSource: ...


class Flux2ModelFactory(Protocol):
    def __call__(self, config: Flux2TrainingConfig) -> Flux: ...


class Flux2DataSourceFactory(Protocol):
    def __call__(self, config: Flux2TrainingConfig) -> Flux2PreparedBatchSource: ...


class QwenImageModelFactory(Protocol):
    def __call__(self, config: QwenImageTrainingConfig) -> QwenImage: ...


class QwenImageDataSourceFactory(Protocol):
    def __call__(self, config: QwenImageTrainingConfig) -> QwenImagePreparedBatchSource: ...


class Ideogram4ModelFactory(Protocol):
    def __call__(self, config: Ideogram4TrainingConfig) -> Ideogram4DiT: ...


class Ideogram4DataSourceFactory(Protocol):
    def __call__(self, config: Ideogram4TrainingConfig) -> Ideogram4PreparedBatchSource: ...


class RankContext(Protocol):
    @property
    def world_size(self) -> int: ...

    @property
    def rank(self) -> int: ...

    def all_reduce_gradients(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        """Average gradients across ranks in place."""
        ...

    def broadcast_parameters(self, parameters: tuple[torch.nn.Parameter, ...]) -> None: ...

    def barrier(self) -> None: ...


class SingleRankContext:
    """No-op collectives for one local training process."""

    world_size = 1
    rank = 0

    def all_reduce_gradients(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        del parameters

    def broadcast_parameters(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        del parameters

    def barrier(self) -> None:
        pass


class RandomnessPolicy(Protocol):
    def data_generator(self, data_cursor: int) -> torch.Generator: ...

    def draw_sd_timestep_noise(
        self,
        data_cursor: int,
        *,
        timestep_count: int,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def draw_minimax_h3_timestep_noise(
        self,
        data_cursor: int,
        *,
        sigma_count: int,
        video_shape: tuple[int, ...],
        audio_shape: tuple[int, ...],
    ) -> tuple[int, torch.Tensor, torch.Tensor]: ...

    def rng_state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None: ...


class MiniMaxMusic3RandomnessPolicy(RandomnessPolicy, Protocol):
    def draw_minimax_music3_timestep_noise(
        self,
        data_cursor: int,
        *,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class Ideogram4RandomnessPolicy(RandomnessPolicy, Protocol):
    def draw_ideogram4_timestep_noise(
        self,
        data_cursor: int,
        *,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class _GeneratorRandomnessPolicy:
    def __init__(self, device: torch.device, stream_seeds: dict[str, int]) -> None:
        self._device = device
        self._generators: dict[str, torch.Generator] = {}
        for stream, seed in stream_seeds.items():
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            self._generators[stream] = generator

    def data_generator(self, data_cursor: int) -> torch.Generator:
        del data_cursor
        return self._generators["data"]

    def draw_sd_timestep_noise(
        self,
        data_cursor: int,
        *,
        timestep_count: int,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del data_cursor
        generator = self._generators["timestep-noise"]
        timesteps = torch.randint(
            0,
            timestep_count,
            (latent_shape[0],),
            generator=generator,
            device=self._device,
        )
        noise = torch.randn(
            latent_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        return timesteps, noise

    def draw_minimax_h3_timestep_noise(
        self,
        data_cursor: int,
        *,
        sigma_count: int,
        video_shape: tuple[int, ...],
        audio_shape: tuple[int, ...],
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        del data_cursor
        generator = self._generators["timestep-noise"]
        index = int(
            torch.randint(
                0,
                sigma_count,
                (1,),
                generator=generator,
                device=self._device,
            ).item()
        )
        video_noise = torch.randn(
            video_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        audio_noise = torch.randn(
            audio_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        return index, video_noise, audio_noise

    def draw_minimax_music3_timestep_noise(
        self,
        data_cursor: int,
        *,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del data_cursor
        generator = self._generators["timestep-noise"]
        timesteps = torch.randn(
            (latent_shape[0],),
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        ).sigmoid_()
        noise = torch.randn(
            latent_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        return timesteps, noise

    def draw_ideogram4_timestep_noise(
        self,
        data_cursor: int,
        *,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del data_cursor
        generator = self._generators["timestep-noise"]
        uniform = torch.rand(
            (latent_shape[0],),
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        noise = torch.randn(
            latent_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        return uniform, noise

    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: generator.get_state().clone() for name, generator in self._generators.items()}

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if set(state) != set(self._generators):
            raise ValueError(
                f"RNG checkpoint streams differ: expected {sorted(self._generators)},"
                f" got {sorted(state)}"
            )
        for name, generator in self._generators.items():
            generator.set_state(state[name].to(device="cpu"))


class SequentialRandomnessPolicy(_GeneratorRandomnessPolicy):
    """Checkpointed sequential generator streams on one training device."""

    _STREAMS = _RNG_STREAMS

    def __init__(
        self,
        seed: int,
        device: torch.device,
        *,
        streams: tuple[str, ...] = _STREAMS,
    ) -> None:
        super().__init__(device, {stream: self._seed(seed, stream) for stream in streams})

    @staticmethod
    def _seed(seed: int, stream: str) -> int:
        digest = hashlib.sha256(f"{seed}:rng:{stream}".encode()).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _validate_counter_rng_state_dict(state: dict[str, torch.Tensor]) -> None:
    if set(state) != {_COUNTER_RNG_POLICY_KEY}:
        raise ValueError(
            "counter RNG checkpoint must contain only the policy marker"
            f" {_COUNTER_RNG_POLICY_KEY!r}; got {sorted(state)}"
        )
    marker = state[_COUNTER_RNG_POLICY_KEY]
    if (
        marker.device.type != "cpu"
        or marker.dtype != torch.uint8
        or tuple(marker.shape) != (1,)
        or marker.item() != _COUNTER_RNG_POLICY_VERSION
    ):
        raise ValueError(
            "counter RNG checkpoint policy marker must be a CPU uint8 tensor"
            f" containing [{_COUNTER_RNG_POLICY_VERSION}]"
        )


class CounterRandomnessPolicy:
    """Stateless generator streams derived from each absolute data cursor."""

    def __init__(self, seed: int, device: torch.device) -> None:
        self._seed_value = seed
        self._device = device

    def _generator(self, stream: str, data_cursor: int) -> torch.Generator:
        if data_cursor < 0:
            raise ValueError(f"data cursor must be non-negative, got {data_cursor}")
        digest = hashlib.sha256(
            f"{self._seed_value}:rng:{stream}:cursor:{data_cursor}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device=self._device)
        generator.manual_seed(seed)
        return generator

    def data_generator(self, data_cursor: int) -> torch.Generator:
        return self._generator("data", data_cursor)

    def draw_sd_timestep_noise(
        self,
        data_cursor: int,
        *,
        timestep_count: int,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = self._generator("timestep-noise", data_cursor)
        timesteps = torch.randint(
            0,
            timestep_count,
            (latent_shape[0],),
            generator=generator,
            device=self._device,
        )
        noise = torch.randn(
            latent_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        return timesteps, noise

    def draw_minimax_h3_timestep_noise(
        self,
        data_cursor: int,
        *,
        sigma_count: int,
        video_shape: tuple[int, ...],
        audio_shape: tuple[int, ...],
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        generator = self._generator("timestep-noise", data_cursor)
        index = int(
            torch.randint(
                0,
                sigma_count,
                (1,),
                generator=generator,
                device=self._device,
            ).item()
        )
        video_noise = torch.randn(
            video_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        audio_noise = torch.randn(
            audio_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        return index, video_noise, audio_noise

    def draw_minimax_music3_timestep_noise(
        self,
        data_cursor: int,
        *,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = self._generator("timestep-noise", data_cursor)
        timesteps = torch.randn(
            (latent_shape[0],),
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        ).sigmoid_()
        noise = torch.randn(
            latent_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        return timesteps, noise

    def draw_ideogram4_timestep_noise(
        self,
        data_cursor: int,
        *,
        latent_shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = self._generator("timestep-noise", data_cursor)
        uniform = torch.rand(
            (latent_shape[0],),
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        noise = torch.randn(
            latent_shape,
            generator=generator,
            device=self._device,
            dtype=torch.float32,
        )
        return uniform, noise

    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            _COUNTER_RNG_POLICY_KEY: torch.tensor([_COUNTER_RNG_POLICY_VERSION], dtype=torch.uint8)
        }

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        _validate_counter_rng_state_dict(state)


class RankStridedRandomnessPolicy(_GeneratorRandomnessPolicy):
    """Checkpointed generator streams derived independently for one rank."""

    _STREAMS = _RNG_STREAMS

    def __init__(
        self,
        seed: int,
        device: torch.device,
        rank: int,
        world_size: int,
    ) -> None:
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        super().__init__(
            device,
            {stream: self._seed(seed, stream, rank) for stream in self._STREAMS},
        )

    @staticmethod
    def _seed(seed: int, stream: str, rank: int) -> int:
        digest = hashlib.sha256(f"{seed}:rng:{stream}:{rank}".encode()).digest()
        return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _validate_rank_rng_state_dict(
    state: dict[str, torch.Tensor],
    *,
    world_size: int,
) -> None:
    rank_keys = {key for key in state if key.startswith("rank")}
    bare_keys = set(state) - rank_keys
    if rank_keys and bare_keys:
        raise ValueError("RNG checkpoint mixes bare and rank-prefixed stream keys")
    if not rank_keys:
        raise ValueError(
            f"RNG checkpoint for world size {world_size} requires rank-prefixed stream keys"
        )

    for key in rank_keys:
        prefix, separator, stream = key.partition(":")
        rank_text = prefix.removeprefix("rank")
        if (
            not separator
            or not rank_text.isdecimal()
            or str(int(rank_text)) != rank_text
            or not stream
        ):
            raise ValueError(f"RNG checkpoint has malformed rank-prefixed key {key!r}")
        rank = int(rank_text)
        if rank >= world_size:
            raise ValueError(
                f"RNG checkpoint rank {rank} is outside configured world size {world_size}"
            )

    expected = {f"rank{rank}:{stream}" for rank in range(world_size) for stream in _RNG_STREAMS}
    if set(state) != expected:
        raise ValueError(
            f"RNG checkpoint rank matrix differs: expected {sorted(expected)}, got {sorted(state)}"
        )


def _extract_rank_rng_state_dict(
    state: dict[str, torch.Tensor],
    *,
    rank: int,
    world_size: int,
) -> dict[str, torch.Tensor]:
    _validate_rank_rng_state_dict(state, world_size=world_size)
    return {stream: state[f"rank{rank}:{stream}"] for stream in _RNG_STREAMS}


def merge_rank_rng_state_dicts(
    rank_states: tuple[dict[str, torch.Tensor], ...],
    *,
    world_size: int,
) -> dict[str, torch.Tensor]:
    """Merge one rank-prefixed RNG state per rank into a complete checkpoint matrix."""
    if world_size <= 1:
        raise ValueError("rank RNG checkpoint merge requires world size greater than one")
    if len(rank_states) != world_size:
        raise ValueError(
            f"expected one RNG state per rank for world size {world_size}, got {len(rank_states)}"
        )

    if any(_COUNTER_RNG_POLICY_KEY in state for state in rank_states):
        for state in rank_states:
            _validate_counter_rng_state_dict(state)
        return {_COUNTER_RNG_POLICY_KEY: rank_states[0][_COUNTER_RNG_POLICY_KEY].clone()}

    merged: dict[str, torch.Tensor] = {}
    for rank, state in enumerate(rank_states):
        expected = {f"rank{rank}:{stream}" for stream in _RNG_STREAMS}
        if set(state) != expected:
            raise ValueError(
                f"RNG state for rank {rank} differs: expected {sorted(expected)},"
                f" got {sorted(state)}"
            )
        merged.update(state)
    _validate_rank_rng_state_dict(merged, world_size=world_size)
    return {
        f"rank{rank}:{stream}": merged[f"rank{rank}:{stream}"]
        for rank in range(world_size)
        for stream in _RNG_STREAMS
    }


def _configure_minimax_h3_base(
    model: torch.nn.Module,
    *,
    quantized: bool,
    int8_base_forward: MiniMaxH3Int8BaseForward,
) -> torch.nn.Module:
    int8_layers = tuple(module for module in model.modules() if isinstance(module, Int8Linear))
    if bool(int8_layers) != quantized:
        actual = "INT8-quantized" if int8_layers else "unquantized"
        expected = "INT8-quantized" if quantized else "unquantized"
        raise ValueError(f"MiniMax H3 model is {actual}, but config selects an {expected} base")
    for layer in int8_layers:
        if int8_base_forward == "fused":
            layer.bind_fused_training(True)
        else:
            layer.bind_fused_training(False)
            layer.full_precision_matmul = True
    return model


def default_model_factory(config: TrainingConfig) -> torch.nn.Module:
    """Load an exact native UNet state from a safetensors source."""
    if config.base_state_path is None or config.base_state_digest is None:
        raise ValueError("baseState is required unless the host injects a model factory")
    path = Path(config.base_state_path)
    path = fixed_asset_ref(path, config.base_state_digest, path.stat().st_size).local_path()
    if config.family == "sdxl":
        plan = validate_sdxl_training_plan(
            plan_sd_assembly(checkpoint=load_safetensors_header(path))
        )
        if plan.diffusion.config != config.unet:
            raise ValueError("baseState SDXL UNet geometry does not match the training config")
        return load_planned_component(plan.diffusion, UNetModel)
    with torch.device("meta"):
        model = UNetModel(config.unet)
    state_keys = tuple(model.state_dict())
    source_keys = tuple(config.base_state_prefix + key for key in state_keys)
    source = load_tensors(path, source_keys)
    state = {key: source[config.base_state_prefix + key] for key in state_keys}
    model.load_state_dict(state, strict=True, assign=True)
    return model


def default_data_source_factory(config: TrainingConfig) -> PreparedBatchSource:
    if config.dataset is not None:
        return default_image_caption_source(
            config.dataset,
            batch_size=config.latent_shape[0],
            device=torch.device(config.device),
        )
    if config.prepared_batch_root is None:
        raise ValueError(
            "dataset or preparedBatchRoot is required unless the host injects a data source"
        )
    return FilesystemPreparedBatchSource(Path(config.prepared_batch_root))


def minimax_h3_model_assembly_plan(
    config: MiniMaxH3TrainingConfig,
) -> MiniMaxH3ModelAssemblyPlan:
    """Verify and plan the H3 DiT artifact selected by a training config."""
    if (
        config.dit_state_path is None
        or config.dit_state_digest is None
        or config.dit_state_size is None
    ):
        raise ValueError("ditState is required unless the host injects an H3 model factory")
    path = Path(config.dit_state_path)
    source = load_safetensors_header(
        path,
        asset_digest=config.dit_state_digest,
        asset_size=config.dit_state_size,
    )
    return plan_minimax_h3_model_assembly(source, role=config.dit_role, path=path)


def minimax_h3_time_embedding_kind(
    config: MiniMaxH3TrainingConfig,
) -> MiniMaxH3TimeEmbeddingKind:
    """Resolve the DiT layout from the pinned artifact when one is present."""
    if config.dit_state_path is None:
        return "curve"
    return minimax_h3_model_assembly_plan(config).diffusion.config.time_embedding_kind


def _registered_attention_backend(family: ModelFamily, component_role: str) -> AttentionRole:
    backend = family.engine.attention_backend(component_role)
    if backend is None:
        raise RuntimeError(
            f"{family.id} has no registered attention backend for {component_role!r}"
        )
    return cast("AttentionRole", backend)


def default_minimax_h3_model_factory(config: MiniMaxH3TrainingConfig) -> torch.nn.Module:
    """Verify, plan, identity-check, and strict-load one H3 DiT component."""
    plan = minimax_h3_model_assembly_plan(config)
    assert (
        config.dit_state_path is not None
        and config.dit_state_digest is not None
        and config.dit_state_size is not None
    )
    path = Path(config.dit_state_path)
    source_is_quantized = bool(plan.diffusion.quant)
    if source_is_quantized != config.quantized_base:
        actual = "quantized" if source_is_quantized else "unquantized"
        selected = "quantized" if config.quantized_base else "unquantized"
        raise ValueError(f"ditState is {actual}, but quantizedBase selects {selected}")
    if source_is_quantized and any(
        quant.format != "int8_tensorwise" for quant in plan.diffusion.quant.values()
    ):
        raise ValueError("quantized MiniMax H3 training supports only INT8 tensorwise layers")
    dtype = torch.float32 if config.base_dtype == "float32" else torch.bfloat16
    loaded = load_minimax_h3_model(
        path,
        asset=fixed_asset_ref(path, config.dit_state_digest, config.dit_state_size),
        role=config.dit_role,
        expected_identity=config.dit_identity,
        diffusion_dtype=dtype,
        attention_backend=_registered_attention_backend(MINIMAX_H3, "diffusion"),
    )
    if loaded.model_role != config.dit_role:
        raise ValueError(
            f"loaded MiniMax H3 role mismatch: expected {config.dit_role!r},"
            f" got {loaded.model_role!r}"
        )
    return _configure_minimax_h3_base(
        loaded.assembled.diffusion,
        quantized=config.quantized_base,
        int8_base_forward=config.int8_base_forward,
    )


def _artifact_header(source: TrainingArtifactSource, family: str) -> SafetensorsSource:
    path = Path(source.path)
    if path.stat().st_size != source.size:
        raise ValueError(f"{family} artifact {path} byte size differs from its training config")
    path = fixed_asset_ref(path, source.digest, source.size).local_path()
    return load_safetensors_header(
        path,
        asset_digest=source.digest,
        asset_size=source.size,
    )


def _wan_artifact_header(source: TrainingArtifactSource) -> SafetensorsSource:
    return _artifact_header(source, "Wan")


def _flux_artifact_header(source: TrainingArtifactSource) -> SafetensorsSource:
    return _artifact_header(source, "Flux")


def flux_model_assembly_plan(config: FluxTrainingConfig) -> FluxAssemblyPlan:
    """Verify and plan the complete component set selected for Flux training."""
    plan = plan_flux_assembly(
        diffusion=_flux_artifact_header(config.dit_state),
        clip_l=_flux_artifact_header(config.clip_l_state),
        t5xxl=_flux_artifact_header(config.t5xxl_state),
        vae=_flux_artifact_header(config.vae_state),
    )
    expected_config, expected_family = {
        "flux1-dev": (FLUX_DEV_CONFIG, FLUX_DEV),
        "flux1-schnell": (FLUX_SCHNELL_CONFIG, FLUX_SCHNELL),
    }[config.variant]
    if plan.diffusion.config != expected_config or plan.family.id != expected_family.id:
        raise ValueError(f"DiT does not match Flux training variant {config.variant!r}")
    if plan.qwen3_2b is not None or plan.clip_l is None or plan.t5xxl is None:
        raise ValueError("classic Flux training requires CLIP-L and T5-XXL")
    if plan.clip_l.config != CLIP_L_TEXT_CONFIG:
        raise ValueError("Flux training requires the CLIP-L text encoder")
    if plan.t5xxl.config != T5_XXL_CONFIG:
        raise ValueError("Flux training requires the classic T5-XXL text encoder")
    for component in (plan.diffusion, plan.clip_l, plan.t5xxl, plan.vae):
        if component.quant:
            raise ValueError(
                f"{component.component}: quantized Flux training components are not supported"
            )
    unsupported_storage = sorted(
        {dtype.name for dtype in plan.diffusion.dtypes.values() if dtype not in (BFLOAT16, FLOAT16)}
    )
    if unsupported_storage:
        raise ValueError(
            "Flux training DiT storage must be float16 or bfloat16, got "
            + ", ".join(unsupported_storage)
        )
    return plan


def default_flux_model_factory(config: FluxTrainingConfig) -> Flux:
    """Digest-verify, validate, and strict-load one trainable classic Flux DiT."""
    plan = flux_model_assembly_plan(config)
    model = load_planned_component(plan.diffusion, Flux)
    dtype = torch.float16 if config.base_dtype == "float16" else torch.bfloat16
    model.to(dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def flux2_model_assembly_plan(config: Flux2TrainingConfig) -> Flux2AssemblyPlan:
    """Verify and plan the complete component set selected for Flux2 training."""
    plan = plan_flux2_assembly(
        diffusion=_artifact_header(config.dit_state, "Flux2"),
        text_encoder=_artifact_header(config.text_encoder_state, "Flux2"),
        vae=_artifact_header(config.vae_state, "Flux2"),
    )
    expected_config, expected_family, text_architectures = {
        "flux2-dev": (
            FLUX2_DEV_CONFIG,
            FLUX2_DEV,
            (MISTRAL3_24B_CONFIG.architecture, MISTRAL3_24B_PRUNED_CONFIG.architecture),
        ),
        "flux2-klein-9b": (
            FLUX2_KLEIN_9B_CONFIG,
            FLUX2_KLEIN_9B,
            (KLEIN_QWEN3_8B_CONFIG.architecture,),
        ),
        "flux2-klein-4b": (
            FLUX2_KLEIN_4B_CONFIG,
            FLUX2_KLEIN_4B,
            (KLEIN_QWEN3_4B_CONFIG.architecture,),
        ),
    }[config.variant]
    if plan.diffusion.config is not expected_config or plan.family is not expected_family:
        raise ValueError(f"DiT does not match Flux2 training variant {config.variant!r}")
    if plan.text_encoder.config.architecture not in text_architectures:
        raise ValueError(f"text encoder does not match Flux2 training variant {config.variant!r}")
    if not plan.vae.config.batch_norm_latent or plan.vae.config.latent_channels != 128:
        raise ValueError("Flux2 training requires the 128-channel packed batch-norm KL VAE")
    for component in (plan.diffusion, plan.text_encoder, plan.vae):
        if component.quant:
            raise ValueError(
                f"{component.component}: quantized Flux2 training components are not supported"
            )
    unsupported_storage = sorted(
        {dtype.name for dtype in plan.diffusion.dtypes.values() if dtype not in (BFLOAT16, FLOAT16)}
    )
    if unsupported_storage:
        raise ValueError(
            "Flux2 training DiT storage must be float16 or bfloat16, got "
            + ", ".join(unsupported_storage)
        )
    return plan


def default_flux2_model_factory(config: Flux2TrainingConfig) -> Flux:
    """Digest-verify, validate, and strict-load one trainable Flux2 DiT."""
    plan = flux2_model_assembly_plan(config)
    model = load_planned_component(plan.diffusion, Flux)
    dtype = torch.float16 if config.base_dtype == "float16" else torch.bfloat16
    model.to(dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def qwen_image_model_assembly_plan(config: QwenImageTrainingConfig) -> QwenImageAssemblyPlan:
    """Verify and plan the complete component set selected for Qwen-Image training."""
    plan = plan_qwen_image_assembly(
        diffusion=_artifact_header(config.dit_state, "Qwen-Image"),
        qwen2_5_vl_7b=_artifact_header(config.text_encoder_state, "Qwen-Image"),
        vae=_artifact_header(config.vae_state, "Qwen-Image"),
    )
    if plan.family is not QWEN_IMAGE or plan.diffusion.config is not QWEN_IMAGE_CONFIG:
        raise ValueError("DiT does not match the base Qwen-Image training variant")
    if plan.qwen2_5_vl_7b.config != QWEN_IMAGE_TEXT_CONFIG:
        raise ValueError("Qwen-Image training requires the Qwen2.5-VL-7B text encoder")
    if plan.vae.config != WAN21_VAE_CONFIG:
        raise ValueError("Qwen-Image training requires the Wan 2.1 VAE")
    for component in (plan.diffusion, plan.qwen2_5_vl_7b, plan.vae):
        if component.quant:
            raise ValueError(
                f"{component.component}: quantized Qwen-Image training components are not supported"
            )
    unsupported_storage = sorted(
        {dtype.name for dtype in plan.diffusion.dtypes.values() if dtype not in (BFLOAT16, FLOAT32)}
    )
    if unsupported_storage:
        raise ValueError(
            "Qwen-Image training DiT storage must be bfloat16 or float32, got "
            + ", ".join(unsupported_storage)
        )
    return plan


def default_qwen_image_model_factory(config: QwenImageTrainingConfig) -> QwenImage:
    """Digest-verify, validate, and strict-load one trainable Qwen-Image DiT."""
    plan = qwen_image_model_assembly_plan(config)
    model = load_planned_component(plan.diffusion, QwenImage)
    dtype = torch.bfloat16 if config.base_dtype == "bfloat16" else torch.float32
    model.to(dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def default_qwen_image_data_source_factory(
    config: QwenImageTrainingConfig,
) -> QwenImagePreparedBatchSource:
    """Digest-verify and encode the configured Qwen-Image dataset."""
    if config.dataset is None:
        raise ValueError("dataset is required unless the host injects a Qwen-Image data source")
    plan = qwen_image_model_assembly_plan(config)
    device = torch.device(config.device)

    def vae_factory() -> WanVAE:
        def build(vae: Wan21VAEConfig) -> WanVAE:
            return WanVAE(
                TorchWanVAEConfig(
                    dim=vae.dim,
                    z_dim=vae.z_dim,
                    dim_mult=vae.dim_mult,
                    num_res_blocks=vae.num_res_blocks,
                    attn_scales=vae.attn_scales,
                    temporal_downsample=vae.temporal_downsample,
                    image_channels=vae.image_channels,
                    conv_out_channels=vae.conv_out_channels,
                    dropout=vae.dropout,
                ),
                operations=CastOperations(WAN21_ENCODING_DTYPE),
            )

        return load_planned_component(plan.vae, build)

    def text_factory() -> QwenImageTextModel:
        return load_planned_component(
            plan.qwen2_5_vl_7b,
            lambda _config: QwenImageTextModel(
                operations=CastOperations(QWEN_IMAGE_TEXT_ENCODING_DTYPE)
            ),
        )

    return QwenImageDatasetSource(
        config.dataset,
        vae_factory,
        text_factory,
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        attention_mask_shape=config.attention_mask_shape,
        vae_identity={"digest": config.vae_state.digest, "size": config.vae_state.size},
        text_encoder_identity={
            "digest": config.text_encoder_state.digest,
            "size": config.text_encoder_state.size,
        },
        device=device,
    )


@dataclass(frozen=True)
class Ideogram4ComponentPlans:
    diffusion: ComponentPlan[object]
    text_encoder: ComponentPlan[object] | None
    vae: Flux2PlannedComponent


def _native_artifact_header(source: NativeTrainingArtifactSource, family: str) -> SafetensorsSource:
    return _artifact_header(cast("TrainingArtifactSource", source), family)


def ideogram4_component_plans(config: Ideogram4TrainingConfig) -> Ideogram4ComponentPlans:
    """Verify all role-bound native components before constructing a model."""
    diffusion_path = Path(config.diffusion_state.path)
    diffusion = plan_ideogram4_split_component(
        _native_artifact_header(config.diffusion_state, "Ideogram 4"),
        role="diffusion",
        path=diffusion_path,
    )
    diffusion_identity = ideogram4_component_runtime_identity(diffusion, "diffusion", BFLOAT16)
    if diffusion_identity != config.diffusion_state.identity:
        raise ValueError("Ideogram 4 diffusion runtime identity differs from its training config")
    if config.base_storage == "fp8":
        if not ideogram4_component_uses_fp8_matmul("diffusion", diffusion):
            raise ValueError("Ideogram 4 diffusion artifact is not the selected FP8 base")
    elif not diffusion.quant or any(
        quant.format != "int8_tensorwise"
        or quant.parameters.get("convrot") is not True
        or type(quant.parameters.get("convrot_groupsize")) is not int
        for quant in diffusion.quant.values()
    ):
        raise ValueError("Ideogram 4 diffusion artifact is not the selected INT8 ConvRot base")

    text_encoder = None
    if config.text_encoder_state is not None:
        text_path = Path(config.text_encoder_state.path)
        text_encoder = plan_ideogram4_split_component(
            _native_artifact_header(config.text_encoder_state, "Ideogram 4"),
            role="qwen3vl_8b",
            path=text_path,
        )
        text_identity = ideogram4_component_runtime_identity(text_encoder, "qwen3vl_8b", BFLOAT16)
        if text_identity != config.text_encoder_state.identity:
            raise ValueError("Ideogram 4 text runtime identity differs from its training config")

    vae_path = Path(config.vae_state.path)
    vae = plan_flux2_split_component(
        _native_artifact_header(config.vae_state, "Flux2"),
        role="vae",
        path=vae_path,
    )
    if vae.family_id != "dinkster.flux2":
        raise ValueError("Ideogram 4 training requires the shared Flux2 VAE")
    vae_identity = flux2_component_runtime_identity(vae, FLOAT32)
    if vae_identity != config.vae_state.identity:
        raise ValueError("Flux2 VAE runtime identity differs from its training config")
    if vae.plan.quant:
        raise ValueError("quantized Flux2 VAEs are not supported for Ideogram 4 training")
    return Ideogram4ComponentPlans(diffusion, text_encoder, vae)


def _configure_ideogram4_base(model: Ideogram4DiT, config: Ideogram4TrainingConfig) -> Ideogram4DiT:
    int8_layers = tuple(module for module in model.modules() if isinstance(module, Int8Linear))
    fp8_layers = tuple(
        module
        for module in model.modules()
        if isinstance(module, Fp8Linear)
        or (
            isinstance(module, torch.nn.Linear)
            and module.weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        )
    )
    if config.base_storage == "fp8":
        if int8_layers or not fp8_layers:
            raise ValueError("loaded Ideogram 4 model does not use the selected FP8 base storage")
        for layer in fp8_layers:
            if isinstance(layer, Fp8Linear):
                layer.bind_fp8_matmul(False)
            else:
                bind_fp8_matmul_layer(layer, False)
    else:
        if fp8_layers or not int8_layers:
            raise ValueError(
                "loaded Ideogram 4 model does not use the selected INT8 ConvRot base storage"
            )
        assert config.int8_base_forward is not None
        for layer in int8_layers:
            if not layer.convrot:
                raise ValueError("Ideogram 4 INT8 training requires ConvRot layers")
            if config.int8_base_forward == "fused":
                layer.bind_fused_training(True)
            else:
                layer.bind_fused_training(False)
                layer.full_precision_matmul = True
    return model


def default_ideogram4_model_factory(config: Ideogram4TrainingConfig) -> Ideogram4DiT:
    """Verify and strict-load one role-bound trainable Ideogram 4 DiT."""
    ideogram4_component_plans(config)
    source = config.diffusion_state
    path = Path(source.path)
    loaded = load_ideogram4_component(
        path,
        asset=fixed_asset_ref(path, source.digest, source.size),
        expected_role="diffusion",
        expected_identity=source.identity,
        compute_dtype=torch.bfloat16,
    )
    return _configure_ideogram4_base(cast("Ideogram4DiT", loaded.module), config)


def default_ideogram4_data_source_factory(
    config: Ideogram4TrainingConfig,
) -> Ideogram4PreparedBatchSource:
    """Verify and encode the configured role-bound Ideogram 4 dataset."""
    if config.dataset is None:
        raise ValueError("dataset is required unless the host injects an Ideogram 4 data source")
    ideogram4_component_plans(config)
    device = torch.device(config.device)

    def vae_factory() -> AutoencoderKL:
        source = config.vae_state
        path = Path(source.path)
        loaded = load_flux2_component(
            path,
            asset=fixed_asset_ref(path, source.digest, source.size),
            expected_role="vae",
            expected_identity=source.identity,
            compute_dtype=torch.float32,
        )
        return cast("AutoencoderKL", loaded.module)

    text_factory: Callable[[], Ideogram4TextEncoder] | None = None
    if config.text_encoder_state is not None:

        def build_text() -> Ideogram4TextEncoder:
            source = config.text_encoder_state
            assert source is not None
            path = Path(source.path)
            loaded = load_ideogram4_component(
                path,
                asset=fixed_asset_ref(path, source.digest, source.size),
                expected_role="qwen3vl_8b",
                expected_identity=source.identity,
                compute_dtype=torch.bfloat16,
            )
            model = cast("QwenImageLanguageModel", loaded.module)
            model.requires_grad_(False).eval().to(device=device)
            return Ideogram4TextEncoder(model)

        text_factory = build_text

    return Ideogram4DatasetSource(
        config.dataset,
        vae_factory,
        text_factory,
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        attention_mask_shape=config.attention_mask_shape,
        vae_identity=config.vae_state.to_mapping(),
        text_encoder_identity=(
            None if config.text_encoder_state is None else config.text_encoder_state.to_mapping()
        ),
        device=device,
    )


def default_flux_data_source_factory(config: FluxTrainingConfig) -> FluxPreparedBatchSource:
    """Digest-verify and encode the configured classic Flux dataset."""
    if config.dataset is None:
        raise ValueError("dataset is required unless the host injects a Flux data source")
    plan = flux_model_assembly_plan(config)
    assert plan.clip_l is not None and plan.t5xxl is not None
    clip_plan = plan.clip_l
    t5_plan = plan.t5xxl
    operations = CastOperations(torch.float32)
    return FluxDatasetSource(
        config.dataset,
        lambda: load_planned_component(
            plan.vae, lambda value: AutoencoderKL(value, operations=operations)
        ),
        lambda: load_planned_component(
            clip_plan, lambda value: ClipTextModel(value, operations=operations)
        ),
        lambda: load_planned_component(
            t5_plan, lambda value: T5TextModel(value, operations=operations)
        ),
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        pooled_shape=config.pooled_shape,
        artifact_identities=(
            {"digest": config.vae_state.digest, "size": config.vae_state.size},
            {"digest": config.clip_l_state.digest, "size": config.clip_l_state.size},
            {"digest": config.t5xxl_state.digest, "size": config.t5xxl_state.size},
        ),
        device=torch.device(config.device),
    )


def default_flux2_data_source_factory(config: Flux2TrainingConfig) -> Flux2PreparedBatchSource:
    """Digest-verify and encode the configured Flux2 dataset."""
    if config.dataset is None:
        raise ValueError("dataset is required unless the host injects a Flux2 data source")
    plan = flux2_model_assembly_plan(config)
    vae_operations = CastOperations(torch.float32)
    text_operations = CastOperations(torch.bfloat16)
    return Flux2DatasetSource(
        config.dataset,
        lambda: load_planned_component(
            plan.vae, lambda value: AutoencoderKL(value, operations=vae_operations)
        ),
        lambda: load_planned_component(
            plan.text_encoder, lambda value: QwenTextModel(value, operations=text_operations)
        ),
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        artifact_identities=(
            {"digest": config.vae_state.digest, "size": config.vae_state.size},
            {
                "digest": config.text_encoder_state.digest,
                "size": config.text_encoder_state.size,
            },
        ),
        device=torch.device(config.device),
    )


def wan_model_assembly_plan(config: WanTrainingConfig) -> Wan21AssemblyPlan:
    """Verify and plan the complete component set selected for Wan training."""
    sources = {
        "diffusion": _wan_artifact_header(config.dit_state),
        "umt5xxl": _wan_artifact_header(config.umt5xxl_state),
        "vae": _wan_artifact_header(config.vae_state),
    }
    if config.variant == "wan22-ti2v-5b":
        plan = plan_wan22_assembly(**sources)
        expected_configs = (WAN22_TI2V_5B,)
    else:
        plan = plan_wan21_assembly(**sources)
        expected_configs = {
            "wan21-t2v": (WAN21_T2V_1_3B, WAN21_T2V_14B),
            "wan22-t2v-14b": (WAN21_T2V_14B,),
            "wan22-i2v-14b": (WAN22_I2V_14B,),
        }[config.variant]
    if plan.diffusion.config not in expected_configs:
        raise ValueError(f"DiT does not match Wan training variant {config.variant!r}")
    if plan.clip_vision is not None:
        raise ValueError("Wan training variants must not include CLIP vision")
    for component in (plan.diffusion, plan.umt5xxl, plan.vae):
        if component.quant:
            raise ValueError(
                f"{component.component}: quantized Wan training components are not supported"
            )
    unsupported_storage = sorted(
        {
            dtype.name
            for dtype in plan.diffusion.dtypes.values()
            if dtype not in (BFLOAT16, FLOAT16, FLOAT32)
        }
    )
    if unsupported_storage:
        raise ValueError(
            "Wan training DiT storage must be float16, bfloat16, or float32, got "
            + ", ".join(unsupported_storage)
        )
    return plan


def default_wan_model_factory(config: WanTrainingConfig) -> Wan21Model:
    """Digest-verify, validate, and strict-load one trainable Wan DiT."""
    plan = wan_model_assembly_plan(config)
    model = load_planned_component(plan.diffusion, Wan21Model)
    dtype = torch.float16 if config.base_dtype == "float16" else torch.bfloat16
    model.to(dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def default_wan_data_source_factory(config: WanTrainingConfig) -> WanPreparedBatchSource:
    """Digest-verify and encode the configured Wan video/caption dataset."""
    if config.dataset is None:
        raise ValueError("dataset is required unless the host injects a Wan data source")
    plan = wan_model_assembly_plan(config)
    device = torch.device(config.device)

    def vae_factory() -> WanVAE | Wan22VAE:
        def build(vae: Wan21VAEConfig | Wan22VAEConfig) -> WanVAE | Wan22VAE:
            if isinstance(vae, Wan22VAEConfig):
                return Wan22VAE(vae, operations=CastOperations(WAN21_ENCODING_DTYPE))
            return WanVAE(
                TorchWanVAEConfig(
                    dim=vae.dim,
                    z_dim=vae.z_dim,
                    dim_mult=vae.dim_mult,
                    num_res_blocks=vae.num_res_blocks,
                    attn_scales=vae.attn_scales,
                    temporal_downsample=vae.temporal_downsample,
                    image_channels=vae.image_channels,
                    conv_out_channels=vae.conv_out_channels,
                    dropout=vae.dropout,
                ),
                operations=CastOperations(WAN21_ENCODING_DTYPE),
            )

        return load_planned_component(plan.vae, build)

    def text_factory() -> Wan21TextRuntime:
        model = load_planned_component(
            plan.umt5xxl,
            lambda text: T5TextModel(text, operations=CastOperations(WAN21_ENCODING_DTYPE)),
        )
        tokenizer = load_tensors(plan.umt5xxl.path, (plan.tokenizer_source_key,))[
            plan.tokenizer_source_key
        ]
        if tokenizer.dtype != torch.uint8 or tokenizer.ndim != 1 or tokenizer.numel() == 0:
            raise ValueError("Wan UMT5 tokenizer payload must be nonempty rank-1 uint8")
        model.__dict__["_dinkster_wan21_spiece_model"] = tokenizer.contiguous().numpy().tobytes()
        model.requires_grad_(False).eval().to(device=device)
        return Wan21TextRuntime(model)

    return WanDatasetSource(
        config.dataset,
        vae_factory,
        text_factory,
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        vae_identity={"digest": config.vae_state.digest, "size": config.vae_state.size},
        umt5xxl_identity={
            "digest": config.umt5xxl_state.digest,
            "size": config.umt5xxl_state.size,
        },
        device=device,
    )


def default_minimax_h3_data_source_factory(
    config: MiniMaxH3TrainingConfig,
) -> MiniMaxH3PreparedBatchSource:
    if config.dataset is not None:
        return default_minimax_h3_dataset_source(
            config.dataset,
            video_latent_shape=config.video_latent_shape,
            audio_latent_shape=config.audio_latent_shape,
            conditioner_shape=config.conditioner_shape,
            device=torch.device(config.device),
        )
    if config.prepared_batch_root is None:
        raise ValueError(
            "dataset or preparedBatchRoot is required unless the host injects an H3 data source"
        )
    return FilesystemMiniMaxH3PreparedBatchSource(Path(config.prepared_batch_root))


def default_minimax_music3_model_factory(
    config: MiniMaxMusic3TrainingConfig,
) -> MiniMaxMusic3DiT:
    """Verify and strict-load one native Music 3 diffusion component."""
    pin = config.diffusion_state
    assert pin.identity is not None
    path = Path(pin.path)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.base_dtype]
    loaded = load_minimax_music3_component(
        path,
        asset=fixed_asset_ref(path, pin.digest, pin.size),
        expected_role="diffusion",
        expected_identity=pin.identity,
        compute_dtype=dtype,
        attention_backend=_registered_attention_backend(MINIMAX_MUSIC3, "diffusion"),
    )
    model = cast("MiniMaxMusic3DiT", loaded.module)
    int8_layers = tuple(module for module in model.modules() if isinstance(module, Int8Linear))
    if bool(int8_layers) != config.quantized_base:
        actual = "quantized" if int8_layers else "unquantized"
        selected = "quantized" if config.quantized_base else "unquantized"
        raise ValueError(f"diffusionState is {actual}, but quantizedBase selects {selected}")
    for layer in int8_layers:
        layer.bind_fused_training(False)
        layer.full_precision_matmul = True
    return model


def default_minimax_music3_data_source_factory(
    config: MiniMaxMusic3TrainingConfig,
) -> MiniMaxMusic3PreparedBatchSource:
    """Build pinned DAV/RVQ/official-text conditioning and encoded cache."""
    device = torch.device(config.device)

    def text_factory() -> tuple[MiniMaxMusic3TextModel, Tokenizer, torch.dtype]:
        pin = config.dataset.text_encoder_state
        assert pin.identity is not None
        path = Path(pin.path)
        dtype = torch.bfloat16 if config.text_dtype == "bfloat16" else torch.float32
        loaded = load_minimax_music3_component(
            path,
            asset=fixed_asset_ref(path, pin.digest, pin.size),
            expected_role="text",
            expected_identity=pin.identity,
            compute_dtype=dtype,
            attention_backend=_registered_attention_backend(MINIMAX_MUSIC3, "text"),
        )
        if loaded.tokenizer is None:
            raise ValueError("MiniMax Music 3 text artifact contains no tokenizer")
        return cast("MiniMaxMusic3TextModel", loaded.module), loaded.tokenizer, dtype

    return MiniMaxMusic3DatasetSource(
        config.dataset,
        lambda: load_minimax_music3_dav_encoder(config.dataset.dav_encoder_state),
        lambda: load_minimax_music3_rvq_encoder(config.dataset.rvq_encoder_state),
        text_factory,
        device=device,
    )


_TrainerConfig = TypeVar(
    "_TrainerConfig",
    TrainingConfig,
    MiniMaxH3TrainingConfig,
    MiniMaxMusic3TrainingConfig,
    WanTrainingConfig,
    FluxTrainingConfig,
    Flux2TrainingConfig,
    QwenImageTrainingConfig,
    Ideogram4TrainingConfig,
)
_TrainerDataSource = TypeVar(
    "_TrainerDataSource",
    PreparedBatchSource,
    MiniMaxH3PreparedBatchSource,
    MiniMaxMusic3PreparedBatchSource,
    WanPreparedBatchSource,
    FluxPreparedBatchSource,
    Flux2PreparedBatchSource,
    QwenImagePreparedBatchSource,
    Ideogram4PreparedBatchSource,
)


class _BaseLoRATrainer(Generic[_TrainerConfig, _TrainerDataSource]):
    """One hot model, attachment, optimizer, RNG set, and data cursor."""

    _STREAMS = _RNG_STREAMS

    def __init__(
        self,
        config: _TrainerConfig,
        model: torch.nn.Module,
        data_source: _TrainerDataSource,
        *,
        expected_family: str,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        if config.family != expected_family:
            raise ValueError(f"{expected_family} trainer received {config.family} configuration")
        self.rank_context = SingleRankContext() if rank_context is None else rank_context
        configured_world_size = 1 if config.distributed is None else config.distributed.world_size
        if self.rank_context.world_size != configured_world_size:
            raise ValueError(
                f"rank context world size {self.rank_context.world_size} does not match"
                f" configured world size {configured_world_size}"
            )
        if self.rank_context.rank < 0 or self.rank_context.rank >= self.rank_context.world_size:
            raise ValueError(
                f"rank must be in [0, {self.rank_context.world_size}), got {self.rank_context.rank}"
            )
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(
                f"the {config.family} LoRA trainer supports cpu and cuda devices,"
                f" not {self.device.type}"
            )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"CUDA device {self.device} was requested but CUDA is unavailable")
        base_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.base_dtype]
        self._model_dtype = base_dtype
        self.layer_pager: FrozenLayerPager | None = None
        if config.family == "minimax-h3":
            h3_config = cast("MiniMaxH3TrainingConfig", config)
            self.model = _configure_minimax_h3_base(
                model,
                quantized=h3_config.quantized_base,
                int8_base_forward=h3_config.int8_base_forward,
            )
            self.lora_export_model_state_keys = tuple(self.model.state_dict())
            self.attachment = TrainableAttachment.attach(
                self.model,
                rank=config.rank,
                alpha=config.alpha,
                seed=config.seed,
                family=config.family,
                device=self.device,
            )
            if h3_config.host_layer_paging_fraction:
                self.layer_pager = FrozenLayerPager(
                    self.model,
                    fraction=h3_config.host_layer_paging_fraction,
                    device=self.device,
                    checkpoint_all_layers=config.gradient_checkpointing,
                )
            self.model.to(device=self.device)
            self.model = _configure_minimax_h3_base(
                self.model,
                quantized=h3_config.quantized_base,
                int8_base_forward=h3_config.int8_base_forward,
            )
        elif config.family == "ideogram4":
            ideogram_config = cast("Ideogram4TrainingConfig", config)
            self.model = _configure_ideogram4_base(
                cast("Ideogram4DiT", model).to(device=self.device), ideogram_config
            )
            self.lora_export_model_state_keys = tuple(self.model.state_dict())
            self.attachment = TrainableAttachment.attach(
                self.model,
                rank=config.rank,
                alpha=config.alpha,
                seed=config.seed,
                family="ideogram4",
                device=self.device,
                role=ideogram_config.role,
            )
            if config.gradient_checkpointing and config.checkpointing_mode == "blockNonReentrant":
                _checkpoint_ideogram4_blocks(self.model)
        else:
            self.model = model.to(device=self.device, dtype=base_dtype)
            self.lora_export_model_state_keys = tuple(self.model.state_dict())
            self.attachment = TrainableAttachment.attach(
                self.model,
                rank=config.rank,
                alpha=config.alpha,
                seed=config.seed,
                family=config.family,
            )
            if config.gradient_checkpointing and config.checkpointing_mode == "blockReentrant":
                _checkpoint_sd_unet_blocks(cast("UNetModel", self.model))
            elif (
                config.family == "wan"
                and config.gradient_checkpointing
                and config.checkpointing_mode == "blockNonReentrant"
            ):
                _checkpoint_wan_blocks(cast("Wan21Model", self.model))
            elif (
                config.family in ("flux", "flux2")
                and config.gradient_checkpointing
                and config.checkpointing_mode == "blockNonReentrant"
            ):
                _checkpoint_flux_blocks(cast("Flux", self.model))
            elif (
                config.family == "qwen-image"
                and config.gradient_checkpointing
                and config.checkpointing_mode == "blockNonReentrant"
            ):
                _checkpoint_qwen_image_blocks(cast("QwenImage", self.model))
        self.model.train()
        self.data_source = data_source
        parameters = self.attachment.parameters()
        if config.optimizer == "adamw":
            self.optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = FactoredAdamW(
                parameters,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.epsilon,
                weight_decay=config.weight_decay,
            )
        if randomness_policy is not None:
            self.randomness_policy = randomness_policy
        elif config.rng_policy == "counter":
            self.randomness_policy = CounterRandomnessPolicy(config.seed, self.device)
        elif configured_world_size == 1:
            self.randomness_policy = SequentialRandomnessPolicy(
                config.seed, self.device, streams=self._STREAMS
            )
        else:
            self.randomness_policy = RankStridedRandomnessPolicy(
                config.seed,
                self.device,
                self.rank_context.rank,
                self.rank_context.world_size,
            )
        self.step_cursor = 0
        self.data_cursor = 0
        self.last_loss: float | None = None

    def _autocast(self) -> AbstractContextManager[object]:
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        state = self.randomness_policy.rng_state_dict()
        if _COUNTER_RNG_POLICY_KEY in state:
            _validate_counter_rng_state_dict(state)
            return state
        if self.rank_context.world_size == 1:
            return state
        if set(state) != set(self._STREAMS):
            raise ValueError(
                f"RNG checkpoint streams differ: expected {sorted(self._STREAMS)},"
                f" got {sorted(state)}"
            )
        return {f"rank{self.rank_context.rank}:{stream}": value for stream, value in state.items()}

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if isinstance(self.randomness_policy, CounterRandomnessPolicy):
            self.randomness_policy.load_rng_state_dict(state)
            return
        if _COUNTER_RNG_POLICY_KEY in state:
            _validate_counter_rng_state_dict(state)
            raise ValueError("counter RNG checkpoint requires the counter RNG policy")
        if self.rank_context.world_size == 1:
            if any(key.startswith("rank") for key in state):
                raise ValueError("single-rank RNG checkpoint requires bare stream keys")
            rank_state = state
        else:
            rank_state = _extract_rank_rng_state_dict(
                state,
                rank=self.rank_context.rank,
                world_size=self.rank_context.world_size,
            )
        self.randomness_policy.load_rng_state_dict(rank_state)

    def data_cursor_for_micro_batch(self, micro_batch_index: int) -> int:
        return micro_batch_index + self.rank_context.rank

    def optimizer_state_dict(self) -> dict[str, object]:
        return cast("dict[str, object]", self.optimizer.state_dict())

    def restore(
        self,
        *,
        adapter: dict[str, torch.Tensor],
        optimizer: dict[str, object],
        rng: dict[str, torch.Tensor],
        step_cursor: int,
        data_cursor: int,
        loss: float | None,
    ) -> None:
        if (step_cursor == 0) != (loss is None):
            raise ValueError("checkpoint loss must be absent exactly at optimizer step zero")
        if loss is not None and not math.isfinite(loss):
            raise ValueError("checkpoint loss must be finite")
        expected_data_cursor = (
            step_cursor * self.config.gradient_accumulation_steps * self.rank_context.world_size
        )
        if isinstance(self.randomness_policy, CounterRandomnessPolicy):
            minimum_data_cursor = step_cursor * self.config.gradient_accumulation_steps
            if (
                (step_cursor == 0 and data_cursor != 0)
                or data_cursor < minimum_data_cursor
                or data_cursor % self.config.gradient_accumulation_steps
            ):
                raise ValueError(
                    f"counter RNG checkpoint data cursor {data_cursor} is invalid"
                    f" for optimizer step {step_cursor}"
                )
        elif data_cursor != expected_data_cursor:
            raise ValueError(
                f"checkpoint data cursor {data_cursor} does not match optimizer step {step_cursor}"
            )
        self.attachment.load_state_dict(adapter)
        self.optimizer.load_state_dict(optimizer)
        self.load_rng_state_dict(rng)
        self.step_cursor = step_cursor
        self.data_cursor = data_cursor
        self.last_loss = loss


class _LoRATrainer(_BaseLoRATrainer[TrainingConfig, PreparedBatchSource]):
    """One native SD UNet training runtime."""

    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        data_source: PreparedBatchSource,
        *,
        expected_family: str,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family=expected_family,
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )
        sigmas = torch.tensor(linear_beta_sigmas(), dtype=torch.float64)
        self._alphas_cumprod = (1.0 / (1.0 + sigmas.square())).to(
            device=self.device, dtype=torch.float32
        )

    def _validate_batch(self, batch: PreparedBatch) -> None:
        if tuple(batch.latents.shape) != self.config.latent_shape:
            raise ValueError(
                f"batch latents have shape {tuple(batch.latents.shape)};"
                f" expected {self.config.latent_shape}"
            )
        if tuple(batch.text_embeddings.shape) != self.config.context_shape:
            raise ValueError(
                f"batch text embeddings have shape {tuple(batch.text_embeddings.shape)};"
                f" expected {self.config.context_shape}"
            )
        if not batch.latents.is_floating_point() or not batch.text_embeddings.is_floating_point():
            raise ValueError("prepared latents and text embeddings must be floating tensors")
        if self.config.family == "sd15":
            if batch.pooled_embeddings is not None or batch.time_ids is not None:
                raise ValueError("SD1.5 prepared batches must not carry SDXL conditioning")
        else:
            if batch.pooled_embeddings is None or batch.time_ids is None:
                raise ValueError("SDXL prepared batches require pooled_embeddings and time_ids")
            assert self.config.pooled_shape is not None
            if tuple(batch.pooled_embeddings.shape) != self.config.pooled_shape:
                raise ValueError(
                    f"batch pooled embeddings have shape {tuple(batch.pooled_embeddings.shape)};"
                    f" expected {self.config.pooled_shape}"
                )
            expected_time_shape = (self.config.latent_shape[0], 6)
            if tuple(batch.time_ids.shape) != expected_time_shape:
                raise ValueError(
                    f"batch time IDs have shape {tuple(batch.time_ids.shape)};"
                    f" expected {expected_time_shape}"
                )
            if not batch.pooled_embeddings.is_floating_point():
                raise ValueError("SDXL pooled embeddings must be floating tensors")
            height = self.config.latent_shape[2] * 8
            width = self.config.latent_shape[3] * 8
            expected_time_ids = torch.tensor(
                [height, width, 0, 0, height, width],
                device=batch.time_ids.device,
                dtype=batch.time_ids.dtype,
            ).expand(self.config.latent_shape[0], -1)
            if not torch.equal(batch.time_ids, expected_time_ids):
                raise ValueError(
                    "SDXL time_ids must use training size for original/target"
                    " and zero top-left crop"
                )
        if (batch.timesteps is None) != (batch.noise is None):
            raise ValueError("prepared timesteps and noise must either both be present or absent")

    def _noise_batch(
        self, batch: PreparedBatch, data_cursor: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch.latents.shape[0]
        if batch.timesteps is None:
            return self.randomness_policy.draw_sd_timestep_noise(
                data_cursor,
                timestep_count=len(self._alphas_cumprod),
                latent_shape=tuple(batch.latents.shape),
            )
        assert batch.noise is not None
        if batch.timesteps.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("prepared timesteps must have an integer dtype")
        timesteps = batch.timesteps.to(device=self.device, dtype=torch.int64)
        noise = batch.noise.to(device=self.device, dtype=torch.float32)
        if tuple(timesteps.shape) != (batch_size,):
            raise ValueError(
                f"timesteps have shape {tuple(timesteps.shape)}; expected {(batch_size,)}"
            )
        if tuple(noise.shape) != tuple(batch.latents.shape):
            raise ValueError(
                f"noise has shape {tuple(noise.shape)}; expected {tuple(batch.latents.shape)}"
            )
        if bool(torch.any((timesteps < 0) | (timesteps >= len(self._alphas_cumprod))).item()):
            raise ValueError("timesteps fall outside the 1000-step linear-beta DDPM schedule")
        return timesteps, noise

    def _forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
        adm: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self.config.gradient_checkpointing
            and self.config.checkpointing_mode == "blockReentrant"
        ):
            noisy_latents = noisy_latents.detach().requires_grad_(True)
        elif self.config.gradient_checkpointing:
            if adm is not None:
                return checkpoint(
                    self.model,
                    noisy_latents,
                    timesteps,
                    text_embeddings,
                    adm,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            return checkpoint(
                self.model,
                noisy_latents,
                timesteps,
                text_embeddings,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        if adm is None:
            return self.model(noisy_latents, timesteps, text_embeddings)
        return self.model(noisy_latents, timesteps, text_embeddings, adm)

    def _adm(self, batch: PreparedBatch) -> torch.Tensor | None:
        if self.config.family == "sd15":
            return None
        assert batch.pooled_embeddings is not None
        height = self.config.latent_shape[2] * 8
        width = self.config.latent_shape[3] * 8
        return encode_sdxl_adm(
            batch.pooled_embeddings.to(device=self.device),
            width=width,
            height=height,
            crop_w=0,
            crop_h=0,
            target_width=width,
            target_height=height,
        )

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for _ in range(self.config.gradient_accumulation_steps):
            data_cursor = self.data_cursor_for_micro_batch(self.data_cursor)
            batch = self.data_source.batch(
                data_cursor,
                generator=self.randomness_policy.data_generator(data_cursor),
                device=self.device,
            )
            self._validate_batch(batch)
            timesteps, noise = self._noise_batch(batch, data_cursor)
            latents = batch.latents.to(device=self.device, dtype=torch.float32)
            context = batch.text_embeddings.to(device=self.device)
            adm = self._adm(batch)
            alpha = self._alphas_cumprod[timesteps].reshape(
                (latents.shape[0],) + (1,) * (latents.ndim - 1)
            )
            noisy = alpha.sqrt() * latents + (1.0 - alpha).sqrt() * noise
            with self._autocast():
                prediction = self._forward(noisy, timesteps, context, adm)
                loss = functional.mse_loss(prediction.float(), noise, reduction="mean")
                scaled = loss / self.config.gradient_accumulation_steps
            scaled.backward()
            losses.append(loss.detach())
            self.data_cursor += self.rank_context.world_size
        self.rank_context.all_reduce_gradients(self.attachment.parameters())
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss


class MiniMaxH3LoRATrainer(_BaseLoRATrainer[MiniMaxH3TrainingConfig, MiniMaxH3PreparedBatchSource]):
    """Frozen-base H3 DiT trainer using the native paired flow objective."""

    _STREAMS = _RNG_STREAMS

    def __init__(
        self,
        config: MiniMaxH3TrainingConfig,
        model: torch.nn.Module,
        data_source: MiniMaxH3PreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="minimax-h3",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )
        table = MINIMAX_H3_SIGMAS.table
        assert table is not None
        self._sigma_table = table

    def _validate_batch(self, batch: MiniMaxH3PreparedBatch) -> None:
        expected = (
            ("video latents", batch.video_latents, self.config.video_latent_shape),
            ("audio latents", batch.audio_latents, self.config.audio_latent_shape),
            (
                "conditioner embeddings",
                batch.conditioner_embeddings,
                self.config.conditioner_shape,
            ),
        )
        for name, value, shape in expected:
            if tuple(value.shape) != shape:
                raise ValueError(f"batch {name} have shape {tuple(value.shape)}; expected {shape}")
            if not value.is_floating_point():
                raise ValueError(f"batch {name} must be floating tensors")
        conditioning = batch.conditioning
        if type(conditioning) is not MiniMaxH3DiTConditioning:
            raise TypeError("batch conditioning must be exact MiniMaxH3DiTConditioning")
        if self.config.dit_role == "ref2va-dit" and not conditioning.references:
            raise ValueError("ref2va-dit training batches require reference conditioning")
        if self.config.dit_role == "fl2va-dit" and conditioning.references:
            raise ValueError("fl2va-dit training batches cannot carry reference conditioning")
        supplied = (
            batch.sigma_indices is not None,
            batch.video_noise is not None,
            batch.audio_noise is not None,
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                "sigma indices and both prepared noise streams must be supplied together"
            )

    def _draw_noise(
        self, batch: MiniMaxH3PreparedBatch, data_cursor: int
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        if batch.sigma_indices is None:
            index, video_noise, audio_noise = self.randomness_policy.draw_minimax_h3_timestep_noise(
                data_cursor,
                sigma_count=len(self._sigma_table),
                video_shape=tuple(batch.video_latents.shape),
                audio_shape=tuple(batch.audio_latents.shape),
            )
            return self._sigma_table[index], video_noise, audio_noise
        assert batch.video_noise is not None and batch.audio_noise is not None
        indices = batch.sigma_indices
        if indices.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("prepared H3 sigma indices must have an integer dtype")
        if tuple(indices.shape) != (1,):
            raise ValueError("prepared H3 sigma indices must have shape (1,)")
        index = int(indices.item())
        if not 0 <= index < len(self._sigma_table):
            raise ValueError("prepared H3 sigma index falls outside the fixed training schedule")
        if tuple(batch.video_noise.shape) != self.config.video_latent_shape:
            raise ValueError("prepared H3 video noise has the wrong shape")
        if tuple(batch.audio_noise.shape) != self.config.audio_latent_shape:
            raise ValueError("prepared H3 audio noise has the wrong shape")
        if not batch.video_noise.is_floating_point() or not batch.audio_noise.is_floating_point():
            raise ValueError("prepared H3 noise streams must be floating tensors")
        return (
            self._sigma_table[index],
            batch.video_noise.to(device=self.device, dtype=torch.float32),
            batch.audio_noise.to(device=self.device, dtype=torch.float32),
        )

    def _conditioning(self, value: MiniMaxH3DiTConditioning) -> MiniMaxH3DiTConditioning:
        return replace(
            value,
            text_token_tags=(
                None if value.text_token_tags is None else value.text_token_tags.to(self.device)
            ),
            keyframes=tuple(
                MiniMaxH3KeyframeLatent(
                    keyframe.resolved_frame_index,
                    keyframe.video.to(device=self.device, dtype=self._model_dtype),
                )
                for keyframe in value.keyframes
            ),
            references=tuple(
                MiniMaxH3ReferenceLatents(
                    reference.kind,
                    None
                    if reference.video is None
                    else reference.video.to(device=self.device, dtype=self._model_dtype),
                    None
                    if reference.audio is None
                    else reference.audio.to(device=self.device, dtype=self._model_dtype),
                )
                for reference in value.references
            ),
        )

    @staticmethod
    def _latent(video: torch.Tensor, audio: torch.Tensor) -> MultiStreamLatent[torch.Tensor]:
        return MultiStreamLatent((LatentStream("video", video), LatentStream("audio", audio)))

    def _forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sigma: float,
        context: torch.Tensor,
        conditioning: MiniMaxH3DiTConditioning,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def evaluate(
            video_value: torch.Tensor,
            audio_value: torch.Tensor,
            context_value: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            output = self.model(
                self._latent(video_value, audio_value),
                sigma,
                context_value,
                conditioning=conditioning,
                sigmas=MINIMAX_H3_SIGMAS,
            )
            if type(output) is not MultiStreamLatent or output.roles != ("video", "audio"):
                raise TypeError("MiniMax H3 DiT must return exact ordered video/audio streams")
            return output.by_role("video"), output.by_role("audio")

        if self.config.gradient_checkpointing and self.layer_pager is None:
            return checkpoint(
                evaluate,
                video,
                audio,
                context,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return evaluate(video, audio, context)

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for _ in range(self.config.gradient_accumulation_steps):
            data_cursor = self.data_cursor_for_micro_batch(self.data_cursor)
            batch = self.data_source.batch(
                data_cursor,
                generator=self.randomness_policy.data_generator(data_cursor),
                device=self.device,
            )
            self._validate_batch(batch)
            sigma, video_noise, audio_noise = self._draw_noise(batch, data_cursor)
            video = batch.video_latents.to(device=self.device, dtype=torch.float32)
            # Sampling stores audio in the video-sigma coordinate, so scaling
            # precedes flow interpolation and velocity target construction.
            audio = batch.audio_latents.to(device=self.device, dtype=torch.float32)
            audio = audio * MINIMAX_H3_SIGMAS.audio_scale
            noisy_video = (1.0 - sigma) * video + sigma * video_noise
            noisy_audio = (1.0 - sigma) * audio + sigma * audio_noise
            target_video = video_noise - video
            target_audio = audio_noise - audio
            model_video = noisy_video.to(dtype=self._model_dtype)
            model_audio = noisy_audio.to(dtype=self._model_dtype)
            context = batch.conditioner_embeddings.to(device=self.device, dtype=self._model_dtype)
            conditioning = self._conditioning(batch.conditioning)
            predicted_video, predicted_audio = self._forward(
                model_video,
                model_audio,
                sigma,
                context,
                conditioning,
            )
            squared = (predicted_video.float() - target_video).square().sum()
            squared = squared + (predicted_audio.float() - target_audio).square().sum()
            loss = squared / (target_video.numel() + target_audio.numel())
            (loss / self.config.gradient_accumulation_steps).backward()
            if self.layer_pager is not None:
                self.layer_pager.assert_idle()
            losses.append(loss.detach())
            self.data_cursor += self.rank_context.world_size
        self.rank_context.all_reduce_gradients(self.attachment.parameters())
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss


class MiniMaxMusic3LoRATrainer(
    _BaseLoRATrainer[MiniMaxMusic3TrainingConfig, MiniMaxMusic3PreparedBatchSource]
):
    """Frozen-base Music 3 DiT trainer using its community-derived flow objective."""

    def __init__(
        self,
        config: MiniMaxMusic3TrainingConfig,
        model: MiniMaxMusic3DiT,
        data_source: MiniMaxMusic3PreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: MiniMaxMusic3RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="minimax-music3",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )

    def _validate_batch(self, batch: MiniMaxMusic3PreparedBatch) -> None:
        if tuple(batch.latents.shape) != self.config.latent_shape:
            raise ValueError(
                f"batch latents have shape {tuple(batch.latents.shape)};"
                f" expected {self.config.latent_shape}"
            )
        if tuple(batch.context.shape) != self.config.context_shape:
            raise ValueError(
                f"batch context has shape {tuple(batch.context.shape)};"
                f" expected {self.config.context_shape}"
            )
        if not batch.latents.is_floating_point() or not batch.context.is_floating_point():
            raise ValueError("Music 3 latents and context must be floating tensors")
        if (batch.timesteps is None) != (batch.noise is None):
            raise ValueError("prepared Music 3 timesteps and noise must be supplied together")

    def _draw_noise(
        self,
        batch: MiniMaxMusic3PreparedBatch,
        data_cursor: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.timesteps is None:
            policy = cast("MiniMaxMusic3RandomnessPolicy", self.randomness_policy)
            return policy.draw_minimax_music3_timestep_noise(
                data_cursor,
                latent_shape=self.config.latent_shape,
            )
        assert batch.noise is not None
        if tuple(batch.timesteps.shape) != (self.config.latent_shape[0],):
            raise ValueError("prepared Music 3 timesteps have the wrong shape")
        if not batch.timesteps.is_floating_point():
            raise ValueError("prepared Music 3 timesteps must be floating")
        timesteps = batch.timesteps.to(device=self.device, dtype=torch.float32)
        if not bool(torch.all((timesteps > 0.0) & (timesteps < 1.0)).item()):
            raise ValueError("prepared Music 3 timesteps must be strictly between zero and one")
        if (
            tuple(batch.noise.shape) != self.config.latent_shape
            or not batch.noise.is_floating_point()
        ):
            raise ValueError("prepared Music 3 noise has the wrong shape or dtype")
        return timesteps, batch.noise.to(device=self.device, dtype=torch.float32)

    def _forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        model = cast("MiniMaxMusic3DiT", self.model)
        scale = torch.ones(
            (latent.shape[0], 1, 1),
            device=self.device,
            dtype=self._model_dtype,
        )
        condition = model.prepare_condition(context, scale)
        if condition.shape[-1] != latent.shape[-1]:
            condition = functional.interpolate(condition, size=latent.shape[-1], mode="nearest")
        rotary = model.prepare_rotary(latent)
        if self.config.gradient_checkpointing:
            return checkpoint(
                model.forward_prepared,
                latent,
                timestep,
                condition,
                rotary,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return model.forward_prepared(latent, timestep, condition, rotary)

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for _ in range(self.config.gradient_accumulation_steps):
            data_cursor = self.data_cursor_for_micro_batch(self.data_cursor)
            batch = self.data_source.batch(
                data_cursor,
                generator=self.randomness_policy.data_generator(data_cursor),
                device=self.device,
            )
            self._validate_batch(batch)
            timestep, noise = self._draw_noise(batch, data_cursor)
            data = batch.latents.to(device=self.device, dtype=torch.float32)
            weight = timestep.reshape((data.shape[0],) + (1,) * (data.ndim - 1))
            noisy = weight * data + (1.0 - weight) * noise
            target = data - noise
            prediction = self._forward(
                noisy.to(dtype=self._model_dtype),
                timestep,
                batch.context.to(device=self.device, dtype=self._model_dtype),
            )
            loss = functional.mse_loss(prediction.float(), target, reduction="mean")
            (loss / self.config.gradient_accumulation_steps).backward()
            losses.append(loss.detach())
            self.data_cursor += self.rank_context.world_size
        self.rank_context.all_reduce_gradients(self.attachment.parameters())
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss


class FluxLoRATrainer(_BaseLoRATrainer[FluxTrainingConfig, FluxPreparedBatchSource]):
    """Frozen-base classic Flux trainer using its shifted flow objective."""

    def __init__(
        self,
        config: FluxTrainingConfig,
        model: Flux,
        data_source: FluxPreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="flux",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )
        if config.variant == "flux1-dev":
            timesteps = torch.arange(1, 10001, dtype=torch.float32) / 10000
            exponent = math.exp(1.15)
            self._sigma_table = exponent / (exponent + (1.0 / timesteps - 1.0))
        else:
            self._sigma_table = torch.arange(1, 1001, dtype=torch.float32) / 1000

    def _validate_batch(self, batch: FluxPreparedBatch) -> None:
        for name, value, shape in (
            ("latents", batch.latents, self.config.latent_shape),
            ("context", batch.context, self.config.context_shape),
            ("pooled", batch.pooled, self.config.pooled_shape),
        ):
            if tuple(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f"batch {name} must be floating with shape {shape}")
        if (batch.sigma_indices is None) != (batch.noise is None):
            raise ValueError("prepared Flux sigma indices and noise must be supplied together")

    def _draw_noise(
        self, batch: FluxPreparedBatch, data_cursor: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.sigma_indices is None:
            return self.randomness_policy.draw_sd_timestep_noise(
                data_cursor,
                timestep_count=len(self._sigma_table),
                latent_shape=tuple(batch.latents.shape),
            )
        assert batch.noise is not None
        indices = batch.sigma_indices
        if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("prepared Flux sigma indices must have an integer dtype")
        if tuple(indices.shape) != (batch.latents.shape[0],):
            raise ValueError("prepared Flux sigma indices must have one entry per batch item")
        if bool(torch.any((indices < 0) | (indices >= len(self._sigma_table))).item()):
            raise ValueError("prepared Flux sigma index falls outside the training schedule")
        if (
            tuple(batch.noise.shape) != tuple(batch.latents.shape)
            or not batch.noise.is_floating_point()
        ):
            raise ValueError("prepared Flux noise has the wrong shape or dtype")
        return indices.to(self.device, dtype=torch.int64), batch.noise.to(
            self.device, dtype=torch.float32
        )

    def _forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        pooled: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        def evaluate(
            noisy_value: torch.Tensor,
            timestep_value: torch.Tensor,
            context_value: torch.Tensor,
            pooled_value: torch.Tensor,
        ) -> torch.Tensor:
            return self.model(noisy_value, timestep_value, context_value, pooled_value, guidance)

        if self.config.gradient_checkpointing and self.config.checkpointing_mode == "wholeModel":
            return checkpoint(
                evaluate,
                noisy,
                timesteps,
                context,
                pooled,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return evaluate(noisy, timesteps, context, pooled)

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for _ in range(self.config.gradient_accumulation_steps):
            data_cursor = self.data_cursor_for_micro_batch(self.data_cursor)
            batch = self.data_source.batch(
                data_cursor,
                generator=self.randomness_policy.data_generator(data_cursor),
                device=self.device,
            )
            self._validate_batch(batch)
            indices, noise = self._draw_noise(batch, data_cursor)
            latents = batch.latents.to(self.device, dtype=torch.float32)
            sigmas = self._sigma_table[indices.cpu()].to(self.device)
            broadcast = sigmas.reshape((latents.shape[0],) + (1,) * (latents.ndim - 1))
            noisy = (1.0 - broadcast) * latents + broadcast * noise
            target = noise - latents
            guidance = (
                None
                if self.config.guidance is None
                else torch.full(
                    (latents.shape[0],),
                    self.config.guidance,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            prediction = self._forward(
                noisy.to(self._model_dtype),
                sigmas,
                batch.context.to(self.device, dtype=self._model_dtype),
                batch.pooled.to(self.device, dtype=self._model_dtype),
                guidance,
            )
            loss = functional.mse_loss(prediction.float(), target)
            (loss / self.config.gradient_accumulation_steps).backward()
            losses.append(loss.detach())
            self.data_cursor += self.rank_context.world_size
        self.rank_context.all_reduce_gradients(self.attachment.parameters())
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss


def flux2_training_sigma_table(variant: Flux2VariantName) -> torch.Tensor:
    """Materialize the registered Flux2 float32 reference schedule in ascending order."""
    shift, timesteps = {
        "flux2-dev": (2.02, 10000),
        "flux2-klein-9b": (2.02, 10000),
        "flux2-klein-4b": (2.02, 10000),
    }[variant]
    descending = simple_schedule(timesteps, FluxFlowSigmas(shift=shift, timesteps=timesteps))
    return torch.tensor(tuple(reversed(descending[:-1])), dtype=torch.float32)


class Flux2LoRATrainer(_BaseLoRATrainer[Flux2TrainingConfig, Flux2PreparedBatchSource]):
    """Frozen-base Flux2 trainer using its registered shifted flow objective."""

    def __init__(
        self,
        config: Flux2TrainingConfig,
        model: Flux,
        data_source: Flux2PreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="flux2",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )
        self._sigma_table = flux2_training_sigma_table(config.variant)

    def _validate_batch(self, batch: Flux2PreparedBatch) -> None:
        for name, value, shape in (
            ("latents", batch.latents, self.config.latent_shape),
            ("context", batch.context, self.config.context_shape),
        ):
            if tuple(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f"batch {name} must be floating with shape {shape}")
        if (batch.sigma_indices is None) != (batch.noise is None):
            raise ValueError("prepared Flux2 sigma indices and noise must be supplied together")

    def _draw_noise(
        self, batch: Flux2PreparedBatch, data_cursor: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.sigma_indices is None:
            return self.randomness_policy.draw_sd_timestep_noise(
                data_cursor,
                timestep_count=len(self._sigma_table),
                latent_shape=tuple(batch.latents.shape),
            )
        assert batch.noise is not None
        indices = batch.sigma_indices
        if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("prepared Flux2 sigma indices must have an integer dtype")
        if tuple(indices.shape) != (batch.latents.shape[0],):
            raise ValueError("prepared Flux2 sigma indices must have one entry per batch item")
        if bool(torch.any((indices < 0) | (indices >= len(self._sigma_table))).item()):
            raise ValueError("prepared Flux2 sigma index falls outside the training schedule")
        if (
            tuple(batch.noise.shape) != tuple(batch.latents.shape)
            or not batch.noise.is_floating_point()
        ):
            raise ValueError("prepared Flux2 noise has the wrong shape or dtype")
        return indices.to(self.device, dtype=torch.int64), batch.noise.to(
            self.device, dtype=torch.float32
        )

    def _forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        def evaluate(
            noisy_value: torch.Tensor,
            timestep_value: torch.Tensor,
            context_value: torch.Tensor,
        ) -> torch.Tensor:
            return self.model(noisy_value, timestep_value, context_value, None, guidance)

        if self.config.gradient_checkpointing and self.config.checkpointing_mode == "wholeModel":
            return checkpoint(
                evaluate,
                noisy,
                timesteps,
                context,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return evaluate(noisy, timesteps, context)

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for _ in range(self.config.gradient_accumulation_steps):
            data_cursor = self.data_cursor_for_micro_batch(self.data_cursor)
            batch = self.data_source.batch(
                data_cursor,
                generator=self.randomness_policy.data_generator(data_cursor),
                device=self.device,
            )
            self._validate_batch(batch)
            indices, noise = self._draw_noise(batch, data_cursor)
            latents = batch.latents.to(self.device, dtype=torch.float32)
            sigmas = self._sigma_table[indices.cpu()].to(self.device)
            broadcast = sigmas.reshape((latents.shape[0],) + (1,) * (latents.ndim - 1))
            noisy = (1.0 - broadcast) * latents + broadcast * noise
            target = noise - latents
            guidance = (
                None
                if self.config.guidance is None
                else torch.full(
                    (latents.shape[0],),
                    self.config.guidance,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            prediction = self._forward(
                noisy.to(self._model_dtype),
                sigmas,
                batch.context.to(self.device, dtype=self._model_dtype),
                guidance,
            )
            loss = functional.mse_loss(prediction.float(), target)
            (loss / self.config.gradient_accumulation_steps).backward()
            losses.append(loss.detach())
            self.data_cursor += self.rank_context.world_size
        self.rank_context.all_reduce_gradients(self.attachment.parameters())
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss


def qwen_image_training_sigma_table(variant: str) -> torch.Tensor:
    """Materialize the registered Qwen-Image flow schedule in ascending order."""
    if variant != "qwen-image":
        raise ValueError(f"unsupported Qwen-Image training variant {variant!r}")
    timesteps = 10000
    descending = simple_schedule(
        timesteps,
        FluxFlowSigmas(shift=QWEN_IMAGE_CONFIG.sampling_shift, timesteps=timesteps),
    )
    return torch.tensor(tuple(reversed(descending[:-1])), dtype=torch.float32)


class QwenImageLoRATrainer(_BaseLoRATrainer[QwenImageTrainingConfig, QwenImagePreparedBatchSource]):
    """Frozen-base Qwen-Image trainer using its registered flow objective."""

    def __init__(
        self,
        config: QwenImageTrainingConfig,
        model: QwenImage,
        data_source: QwenImagePreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="qwen-image",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )
        self._sigma_table = qwen_image_training_sigma_table(config.variant)

    def _validate_batch(self, batch: QwenImagePreparedBatch) -> None:
        for name, value, shape in (
            ("latents", batch.latents, self.config.latent_shape),
            ("context", batch.context, self.config.context_shape),
            ("attention mask", batch.attention_mask, self.config.attention_mask_shape),
        ):
            if tuple(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f"batch {name} must be floating with shape {shape}")
        if not bool(
            torch.all((batch.attention_mask == 0.0) | (batch.attention_mask == 1.0)).item()
        ):
            raise ValueError("batch Qwen-Image attention mask must be binary")
        if (batch.sigma_indices is None) != (batch.noise is None):
            raise ValueError(
                "prepared Qwen-Image sigma indices and noise must be supplied together"
            )

    def _draw_noise(
        self, batch: QwenImagePreparedBatch, data_cursor: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.sigma_indices is None:
            return self.randomness_policy.draw_sd_timestep_noise(
                data_cursor,
                timestep_count=len(self._sigma_table),
                latent_shape=tuple(batch.latents.shape),
            )
        assert batch.noise is not None
        indices = batch.sigma_indices
        if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("prepared Qwen-Image sigma indices must have an integer dtype")
        if tuple(indices.shape) != (batch.latents.shape[0],):
            raise ValueError("prepared Qwen-Image sigma indices need one entry per batch item")
        if bool(torch.any((indices < 0) | (indices >= len(self._sigma_table))).item()):
            raise ValueError("prepared Qwen-Image sigma index falls outside the training schedule")
        if (
            tuple(batch.noise.shape) != tuple(batch.latents.shape)
            or not batch.noise.is_floating_point()
        ):
            raise ValueError("prepared Qwen-Image noise has the wrong shape or dtype")
        return indices.to(self.device, dtype=torch.int64), batch.noise.to(
            self.device, dtype=torch.float32
        )

    def _forward(
        self,
        noisy: torch.Tensor,
        sigmas: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        def evaluate(
            noisy_value: torch.Tensor,
            sigma_value: torch.Tensor,
            context_value: torch.Tensor,
            mask_value: torch.Tensor,
        ) -> torch.Tensor:
            return self.model(noisy_value, sigma_value, context_value, mask_value)

        if self.config.gradient_checkpointing and self.config.checkpointing_mode == "wholeModel":
            return checkpoint(
                evaluate,
                noisy,
                sigmas,
                context,
                attention_mask,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return evaluate(noisy, sigmas, context, attention_mask)

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for _ in range(self.config.gradient_accumulation_steps):
            data_cursor = self.data_cursor_for_micro_batch(self.data_cursor)
            batch = self.data_source.batch(
                data_cursor,
                generator=self.randomness_policy.data_generator(data_cursor),
                device=self.device,
            )
            self._validate_batch(batch)
            indices, noise = self._draw_noise(batch, data_cursor)
            latents = batch.latents.to(self.device, dtype=torch.float32)
            sigmas = self._sigma_table[indices.cpu()].to(self.device)
            broadcast = sigmas.reshape((latents.shape[0],) + (1,) * (latents.ndim - 1))
            noisy = (1.0 - broadcast) * latents + broadcast * noise
            target = noise - latents
            prediction = self._forward(
                noisy.to(self._model_dtype),
                sigmas,
                batch.context.to(self.device, dtype=self._model_dtype),
                batch.attention_mask.to(self.device, dtype=torch.bool),
            )
            loss = functional.mse_loss(prediction.float(), target)
            (loss / self.config.gradient_accumulation_steps).backward()
            losses.append(loss.detach())
            self.data_cursor += self.rank_context.world_size
        self.rank_context.all_reduce_gradients(self.attachment.parameters())
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss


def ideogram4_training_sigmas(
    uniform_samples: torch.Tensor,
    latent_shape: tuple[int, int, int, int],
) -> torch.Tensor:
    """Map uniform samples to the native resolution-aware training sigmas."""
    if uniform_samples.ndim != 1 or uniform_samples.shape[0] != latent_shape[0]:
        raise ValueError("Ideogram 4 timestep samples need one value per batch item")
    if not uniform_samples.is_floating_point():
        raise ValueError("Ideogram 4 timestep samples must be floating")
    height, width = latent_shape[-2:]
    pixels = height * 16 * width * 16
    mean = 0.5 * math.log(pixels / (512 * 512))
    samples = uniform_samples.to(torch.float64).clamp(1e-7, 1.0 - 1e-7)
    sigmas = torch.special.expit(mean + 1.5 * torch.special.ndtri(samples))
    lower = 1.0 - 1.0 / (1.0 + math.exp(0.5 * -15.0))
    upper = 1.0 - 1.0 / (1.0 + math.exp(0.5 * 18.0))
    return sigmas.clamp(lower, upper).to(dtype=uniform_samples.dtype)


class Ideogram4LoRATrainer(_BaseLoRATrainer[Ideogram4TrainingConfig, Ideogram4PreparedBatchSource]):
    """Role-bound Ideogram 4 trainer using the native rectified-flow objective."""

    def __init__(
        self,
        config: Ideogram4TrainingConfig,
        model: Ideogram4DiT,
        data_source: Ideogram4PreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="ideogram4",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )

    def _validate_batch(self, batch: Ideogram4PreparedBatch) -> None:
        if (
            tuple(batch.latents.shape) != self.config.latent_shape
            or not batch.latents.is_floating_point()
        ):
            raise ValueError(
                f"batch Ideogram 4 latents must be floating with shape {self.config.latent_shape}"
            )
        if self.config.role == "conditional":
            assert self.config.context_shape is not None
            assert self.config.attention_mask_shape is not None
            if (
                batch.context is None
                or tuple(batch.context.shape) != self.config.context_shape
                or not batch.context.is_floating_point()
            ):
                raise ValueError(
                    "conditional Ideogram 4 context must be floating with shape"
                    f" {self.config.context_shape}"
                )
            if (
                batch.attention_mask is None
                or tuple(batch.attention_mask.shape) != self.config.attention_mask_shape
                or not batch.attention_mask.is_floating_point()
                or not bool(
                    torch.all((batch.attention_mask == 0.0) | (batch.attention_mask == 1.0)).item()
                )
            ):
                raise ValueError("conditional Ideogram 4 attention mask must be binary")
        elif batch.context is not None or batch.attention_mask is not None:
            raise ValueError("unconditional Ideogram 4 batches cannot carry text conditioning")
        if (batch.sigmas is None) != (batch.noise is None):
            raise ValueError("prepared Ideogram 4 sigmas and noise must be supplied together")

    def _draw_noise(
        self, batch: Ideogram4PreparedBatch, data_cursor: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.sigmas is None:
            policy = cast("Ideogram4RandomnessPolicy", self.randomness_policy)
            uniform, noise = policy.draw_ideogram4_timestep_noise(
                data_cursor,
                latent_shape=tuple(batch.latents.shape),
            )
            return ideogram4_training_sigmas(uniform, self.config.latent_shape), noise
        assert batch.noise is not None
        sigmas = batch.sigmas
        if not sigmas.is_floating_point():
            raise ValueError("prepared Ideogram 4 sigmas must be floating")
        if tuple(sigmas.shape) != (batch.latents.shape[0],):
            raise ValueError("prepared Ideogram 4 sigmas need one entry per batch item")
        if bool(torch.any((sigmas <= 0.0) | (sigmas >= 1.0)).item()):
            raise ValueError("prepared Ideogram 4 sigma falls outside the training space")
        if (
            tuple(batch.noise.shape) != tuple(batch.latents.shape)
            or not batch.noise.is_floating_point()
        ):
            raise ValueError("prepared Ideogram 4 noise has the wrong shape or dtype")
        return sigmas.to(self.device, dtype=torch.float32), batch.noise.to(
            self.device, dtype=torch.float32
        )

    def _forward(
        self,
        noisy: torch.Tensor,
        sigmas: torch.Tensor,
        context: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        model = cast("Ideogram4DiT", self.model)

        def evaluate(
            noisy_value: torch.Tensor,
            sigma_value: torch.Tensor,
            *conditioning: torch.Tensor,
        ) -> torch.Tensor:
            if conditioning:
                return model(noisy_value, sigma_value, conditioning[0], conditioning[1])
            return model(noisy_value, sigma_value)

        if self.config.gradient_checkpointing and self.config.checkpointing_mode == "wholeModel":
            conditioning = (
                () if context is None else (context, cast("torch.Tensor", attention_mask))
            )
            return checkpoint(
                evaluate,
                noisy,
                sigmas,
                *conditioning,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        if context is None:
            return evaluate(noisy, sigmas)
        assert attention_mask is not None
        return evaluate(noisy, sigmas, context, attention_mask)

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for _ in range(self.config.gradient_accumulation_steps):
            data_cursor = self.data_cursor_for_micro_batch(self.data_cursor)
            batch = self.data_source.batch(
                data_cursor,
                generator=self.randomness_policy.data_generator(data_cursor),
                device=self.device,
            )
            self._validate_batch(batch)
            sigmas, noise = self._draw_noise(batch, data_cursor)
            latents = batch.latents.to(self.device, dtype=torch.float32)
            broadcast = sigmas.reshape((latents.shape[0],) + (1,) * (latents.ndim - 1))
            noisy = (1.0 - broadcast) * latents + broadcast * noise
            # The inference model negates its native data-minus-noise output.
            target = noise - latents
            context = (
                None
                if batch.context is None
                else batch.context.to(self.device, dtype=self._model_dtype)
            )
            attention_mask = (
                None
                if batch.attention_mask is None
                else batch.attention_mask.to(self.device, dtype=torch.bool)
            )
            prediction = self._forward(
                noisy.to(dtype=self._model_dtype), sigmas, context, attention_mask
            )
            loss = functional.mse_loss(prediction.float(), target)
            (loss / self.config.gradient_accumulation_steps).backward()
            losses.append(loss.detach())
            self.data_cursor += self.rank_context.world_size
        self.rank_context.all_reduce_gradients(self.attachment.parameters())
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss


class WanLoRATrainer(_BaseLoRATrainer[WanTrainingConfig, WanPreparedBatchSource]):
    """Frozen-base Wan DiT trainer using its shifted flow objective."""

    def __init__(
        self,
        config: WanTrainingConfig,
        model: Wan21Model,
        data_source: WanPreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="wan",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )
        table = WAN21_SIGMAS.table
        assert table is not None
        self._sigma_table = table
        if config.timestep_range is None:
            self._sigma_index_start = 0
            self._sigma_index_stop = len(table)
        else:
            assert config.expert is not None
            allowed = _wan_expert_sigma_indices(
                table,
                WAN21_SIGMAS.multiplier,
                config.expert,
                config.timestep_range,
            )
            if not allowed or allowed != tuple(range(allowed[0], allowed[-1] + 1)):
                raise ValueError("Wan expert timestep range does not select a contiguous schedule")
            self._sigma_index_start = allowed[0]
            self._sigma_index_stop = allowed[-1] + 1

    def _validate_batch(self, batch: WanPreparedBatch) -> None:
        for name, value, shape in (
            ("latents", batch.latents, self.config.latent_shape),
            ("context", batch.context, self.config.context_shape),
        ):
            if tuple(value.shape) != shape:
                raise ValueError(f"batch {name} have shape {tuple(value.shape)}; expected {shape}")
            if not value.is_floating_point():
                raise ValueError(f"batch {name} must be a floating tensor")
        if (batch.sigma_indices is None) != (batch.noise is None):
            raise ValueError("prepared Wan sigma indices and noise must be supplied together")
        if self.config.variant == "wan22-i2v-14b":
            conditioning = batch.i2v_conditioning
            expected = (self.config.latent_shape[0], 20, *self.config.latent_shape[2:])
            if (
                conditioning is None
                or tuple(conditioning.shape) != expected
                or not conditioning.is_floating_point()
            ):
                raise ValueError(
                    f"batch i2v_conditioning must be a floating tensor with shape {expected}"
                )
        elif batch.i2v_conditioning is not None:
            raise ValueError("batch I2V conditioning applies only to Wan 2.2 I2V training")

    def _draw_noise(
        self, batch: WanPreparedBatch, data_cursor: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch.sigma_indices is None:
            indices, noise = self.randomness_policy.draw_sd_timestep_noise(
                data_cursor,
                timestep_count=self._sigma_index_stop - self._sigma_index_start,
                latent_shape=tuple(batch.latents.shape),
            )
            return indices + self._sigma_index_start, noise
        assert batch.noise is not None
        indices = batch.sigma_indices
        if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("prepared Wan sigma indices must have an integer dtype")
        if tuple(indices.shape) != (batch.latents.shape[0],):
            raise ValueError("prepared Wan sigma indices must have one entry per batch item")
        if bool(
            torch.any(
                (indices < self._sigma_index_start) | (indices >= self._sigma_index_stop)
            ).item()
        ):
            raise ValueError("prepared Wan sigma index falls outside the configured timestep range")
        if (
            tuple(batch.noise.shape) != tuple(batch.latents.shape)
            or not batch.noise.is_floating_point()
        ):
            raise ValueError("prepared Wan noise has the wrong shape or dtype")
        return (
            indices.to(device=self.device, dtype=torch.int64),
            batch.noise.to(device=self.device, dtype=torch.float32),
        )

    def _forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        def evaluate(
            noisy_value: torch.Tensor,
            timestep_value: torch.Tensor,
            context_value: torch.Tensor,
        ) -> torch.Tensor:
            return self.model(noisy_value, timestep_value, context_value)

        if self.config.gradient_checkpointing and self.config.checkpointing_mode == "wholeModel":
            return checkpoint(
                evaluate,
                noisy,
                timesteps,
                context,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        return evaluate(noisy, timesteps, context)

    def train_step(self) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for _ in range(self.config.gradient_accumulation_steps):
            data_cursor = self.data_cursor_for_micro_batch(self.data_cursor)
            batch = self.data_source.batch(
                data_cursor,
                generator=self.randomness_policy.data_generator(data_cursor),
                device=self.device,
            )
            self._validate_batch(batch)
            indices, noise = self._draw_noise(batch, data_cursor)
            latents = batch.latents.to(device=self.device, dtype=torch.float32)
            sigmas = torch.tensor(
                [self._sigma_table[int(index)] for index in indices.tolist()],
                device=self.device,
                dtype=torch.float32,
            )
            broadcast = sigmas.reshape((latents.shape[0],) + (1,) * (latents.ndim - 1))
            noisy = (1.0 - broadcast) * latents + broadcast * noise
            target = noise - latents
            timesteps = sigmas * WAN21_SIGMAS.multiplier
            prediction = self._forward(
                torch.cat(
                    (
                        noisy.to(dtype=self._model_dtype),
                        batch.i2v_conditioning.to(
                            device=self.device,
                            dtype=self._model_dtype,
                        ),
                    ),
                    dim=1,
                )
                if batch.i2v_conditioning is not None
                else noisy.to(dtype=self._model_dtype),
                timesteps,
                batch.context.to(device=self.device, dtype=self._model_dtype),
            )
            loss = functional.mse_loss(prediction.float(), target, reduction="mean")
            (loss / self.config.gradient_accumulation_steps).backward()
            losses.append(loss.detach())
            self.data_cursor += self.rank_context.world_size
        self.rank_context.all_reduce_gradients(self.attachment.parameters())
        self.optimizer.step()
        mean_loss = torch.stack(losses).mean().item()
        if not math.isfinite(mean_loss):
            raise FloatingPointError(f"training loss is not finite: {mean_loss}")
        self.step_cursor += 1
        self.last_loss = mean_loss
        return mean_loss


class SD15LoRATrainer(_LoRATrainer):
    """Frozen-base SD1.5 UNet trainer."""

    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        data_source: PreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="sd15",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )


class SDXLLoRATrainer(_LoRATrainer):
    """Frozen-base SDXL UNet trainer with frozen two-tower conditioning."""

    def __init__(
        self,
        config: TrainingConfig,
        model: torch.nn.Module,
        data_source: PreparedBatchSource,
        *,
        rank_context: RankContext | None = None,
        randomness_policy: RandomnessPolicy | None = None,
    ) -> None:
        super().__init__(
            config,
            model,
            data_source,
            expected_family="sdxl",
            rank_context=rank_context,
            randomness_policy=randomness_policy,
        )
