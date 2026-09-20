"""Prepared-batch boundary for the trainer-owned data cursor."""

from __future__ import annotations

import hashlib
import io
import wave
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeVar, cast

import torch
import torch.nn.functional as functional
from dinkster_api.v1 import AssetRef, digest_bytes
from dinkster_inference import (
    CLIP_G_PROFILE,
    CLIP_L_PROFILE,
    CLIP_TEXT_OPTIONAL_KEYS,
    IDEOGRAM4_TEXT_CONFIG,
    MINIMAX_H3_CONFIG,
    QWEN_IMAGE_TEXT_CONFIG,
    SD15,
    SDXL,
    T5_XXL_FLUX_PROFILE,
    ClipTextConfig,
    ComponentPlan,
    LatentStream,
    LinearToConv2D,
    MiniMaxH3AudioContent,
    MiniMaxH3CommonComponentRole,
    MiniMaxH3ConditionerConfig,
    MiniMaxH3T2VARequest,
    MultiStreamLatent,
    PromptTokenizer,
    RowChunk,
    SafetensorsSource,
    SDAssemblyPlan,
    TensorGeometry,
    Transpose2D,
    detect_clip_text_config,
    detect_kl_config,
    format_qwen_image_prompt,
    load_clip_bpe,
    load_flux2_tekken_bpe,
    load_qwen_bpe,
    load_safetensors_header,
    load_t5_spm,
    pack_spans,
    plan_minimax_h3_common_component,
    plan_sd_assembly,
    select_qwen_image_output,
    tokenize_flux2_dev_prompt,
    tokenize_flux2_klein_prompt,
    tokenize_ideogram4_prompt,
)
from dinkster_inference_torch import (
    SDXL_CLIP_POLICY,
    AutoencoderKL,
    ClipTextEncoder,
    ClipTextModel,
    Flux2DevTextEncoder,
    Flux2KleinTextEncoder,
    Ideogram4TextEncoder,
    MiniMaxH3AudioVAE,
    MiniMaxH3AudioVaeRuntime,
    MiniMaxH3ConditionerModel,
    MiniMaxH3ConditionerRuntime,
    MiniMaxH3DiTConditioning,
    MiniMaxH3KeyframeLatent,
    MiniMaxH3ReferenceKind,
    MiniMaxH3ReferenceLatents,
    MiniMaxH3VideoVAE,
    MiniMaxH3VideoVAEConfig,
    MiniMaxH3VideoVaeRuntime,
    QwenImageTextModel,
    QwenTextModel,
    T5TextEncoder,
    T5TextModel,
    Wan21TextRuntime,
    Wan22VAE,
    WanVAE,
    load_minimax_h3_component,
    load_tensors,
    minimax_h3_audio_vae_runtime_identity,
    minimax_h3_conditioner_runtime_identity,
    minimax_h3_video_vae_runtime_identity,
    process_input,
    worker_planning_context,
)
from PIL import Image, ImageOps

from .container import decode_audio_s16_stereo_32k, decode_video_rgb24
from .dataset import (
    QWEN_IMAGE_TEXT_ENCODING_DTYPE as QWEN_IMAGE_TEXT_ENCODING_DTYPE_NAME,
)
from .dataset import (
    EncoderStateSource,
    Flux2DatasetSettings,
    FluxDatasetSettings,
    Ideogram4DatasetSettings,
    ImageCaptionDatasetSettings,
    ImageCaptionItem,
    MiniMaxH3ComponentStateSource,
    MiniMaxH3DatasetItem,
    MiniMaxH3DatasetSettings,
    QwenImageDatasetSettings,
    WanDatasetItem,
    WanDatasetSettings,
    inspect_flux2_dataset,
    inspect_flux_dataset,
    inspect_ideogram4_dataset,
    inspect_image_caption_dataset,
    inspect_minimax_h3_dataset,
    inspect_qwen_image_dataset,
    inspect_wan_dataset,
    minimax_h3_audio_sample_count,
)
from .encoded_cache import (
    EncodedDatasetCache,
    EncodedDatasetTensors,
    EncodedFlux2DatasetTensors,
    EncodedFluxDatasetTensors,
    EncodedIdeogram4DatasetTensors,
    EncodedMiniMaxH3DatasetTensors,
    EncodedQwenImageDatasetTensors,
    EncodedWanDatasetTensors,
    Flux2EncodedDatasetCache,
    FluxEncodedDatasetCache,
    Ideogram4EncodedDatasetCache,
    MiniMaxH3EncodedDatasetCache,
    QwenImageEncodedDatasetCache,
    WanEncodedDatasetCache,
)

_CLIP_TEXT_INERT_KEYS = frozenset({"text_model.embeddings.position_ids", "embeddings.position_ids"})
WAN21_ENCODING_DTYPE = torch.float32
QWEN_IMAGE_TEXT_ENCODING_DTYPE = cast(
    "torch.dtype", getattr(torch, QWEN_IMAGE_TEXT_ENCODING_DTYPE_NAME)
)
_C = TypeVar("_C")
_M = TypeVar("_M", bound=torch.nn.Module)


@dataclass(frozen=True)
class PreparedBatch:
    """Prepared model inputs, optionally with fixed timestep/noise draws."""

    latents: torch.Tensor
    text_embeddings: torch.Tensor
    pooled_embeddings: torch.Tensor | None = None
    time_ids: torch.Tensor | None = None
    timesteps: torch.Tensor | None = None
    noise: torch.Tensor | None = None


class PreparedBatchSource(Protocol):
    """Produce one deterministic batch for an absolute data cursor."""

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch: ...


@dataclass(frozen=True)
class FluxPreparedBatch:
    latents: torch.Tensor
    context: torch.Tensor
    pooled: torch.Tensor
    sigma_indices: torch.Tensor | None = None
    noise: torch.Tensor | None = None


class FluxPreparedBatchSource(Protocol):
    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> FluxPreparedBatch: ...


@dataclass(frozen=True)
class Flux2PreparedBatch:
    latents: torch.Tensor
    context: torch.Tensor
    sigma_indices: torch.Tensor | None = None
    noise: torch.Tensor | None = None


class Flux2PreparedBatchSource(Protocol):
    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> Flux2PreparedBatch: ...


@dataclass(frozen=True)
class QwenImagePreparedBatch:
    latents: torch.Tensor
    context: torch.Tensor
    attention_mask: torch.Tensor
    sigma_indices: torch.Tensor | None = None
    noise: torch.Tensor | None = None


class QwenImagePreparedBatchSource(Protocol):
    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> QwenImagePreparedBatch: ...


@dataclass(frozen=True)
class Ideogram4PreparedBatch:
    latents: torch.Tensor
    context: torch.Tensor | None
    attention_mask: torch.Tensor | None
    sigmas: torch.Tensor | None = None
    noise: torch.Tensor | None = None


class Ideogram4PreparedBatchSource(Protocol):
    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> Ideogram4PreparedBatch: ...


class FilesystemPreparedBatchSource:
    """Read prepared tensor batches named by their absolute cursor.

    Files are ``000000000000.pt`` and contain ``latents`` and
    ``text_embeddings`` tensors. ``timesteps`` and ``noise`` must either both
    be present or both be absent. The dataset pipeline can replace this source
    through the protocol without changing trainer state.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        del generator
        path = self._root / f"{cursor:012d}.pt"
        try:
            value = torch.load(path, map_location=device, weights_only=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"no prepared training batch for cursor {cursor}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"prepared batch {path} must contain a tensor mapping")
        raw = cast("dict[object, object]", value)
        latents = raw.get("latents")
        text = raw.get("text_embeddings")
        pooled = raw.get("pooled_embeddings")
        time_ids = raw.get("time_ids")
        timesteps = raw.get("timesteps")
        noise = raw.get("noise")
        if not isinstance(latents, torch.Tensor) or not isinstance(text, torch.Tensor):
            raise ValueError(f"prepared batch {path} needs latents and text_embeddings tensors")
        if (pooled is None) != (time_ids is None):
            raise ValueError(
                f"prepared batch {path} must provide both pooled_embeddings and time_ids or neither"
            )
        if pooled is not None and (
            not isinstance(pooled, torch.Tensor) or not isinstance(time_ids, torch.Tensor)
        ):
            raise ValueError(f"prepared batch {path} SDXL conditioning values must be tensors")
        if (timesteps is None) != (noise is None):
            raise ValueError(
                f"prepared batch {path} must provide both timesteps and noise or neither"
            )
        if timesteps is not None and (
            not isinstance(timesteps, torch.Tensor) or not isinstance(noise, torch.Tensor)
        ):
            raise ValueError(f"prepared batch {path} timestep/noise values must be tensors")
        return PreparedBatch(
            latents=latents,
            text_embeddings=text,
            pooled_embeddings=pooled,
            time_ids=cast("torch.Tensor | None", time_ids),
            timesteps=timesteps,
            noise=cast("torch.Tensor | None", noise),
        )


@dataclass(frozen=True)
class MiniMaxH3PreparedBatch:
    """Prepared H3 streams, conditioner embeddings, and DiT carrier."""

    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    conditioner_embeddings: torch.Tensor
    conditioning: MiniMaxH3DiTConditioning
    sigma_indices: torch.Tensor | None = None
    video_noise: torch.Tensor | None = None
    audio_noise: torch.Tensor | None = None


class MiniMaxH3PreparedBatchSource(Protocol):
    """Produce one deterministic H3 batch for an absolute data cursor."""

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxH3PreparedBatch: ...


@dataclass(frozen=True)
class WanPreparedBatch:
    """Prepared Wan video latents and UMT5 context."""

    latents: torch.Tensor
    context: torch.Tensor
    sigma_indices: torch.Tensor | None = None
    noise: torch.Tensor | None = None
    i2v_conditioning: torch.Tensor | None = None


class WanPreparedBatchSource(Protocol):
    """Produce one deterministic Wan batch for an absolute data cursor."""

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> WanPreparedBatch: ...


def _h3_tensor(value: object, name: str, path: Path) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"prepared H3 batch {path} {name} must be a tensor")
    return value


def _h3_conditioning(value: object, path: Path) -> MiniMaxH3DiTConditioning:
    if value is None:
        return MiniMaxH3DiTConditioning()
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"prepared H3 batch {path} conditioning must be an object")
    raw = cast("dict[str, object]", value)
    allowed = {
        "text_token_tags",
        "keyframes",
        "references",
        "frame_count",
        "visual_noise_timestep",
        "audio_noise_timestep",
        "seed",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"prepared H3 batch {path} conditioning has unknown fields: {', '.join(unknown)}"
        )
    tags_value = raw.get("text_token_tags")
    tags = None if tags_value is None else _h3_tensor(tags_value, "text_token_tags", path)
    keyframes_value = raw.get("keyframes", [])
    if not isinstance(keyframes_value, list):
        raise ValueError(f"prepared H3 batch {path} keyframes must be an array")
    keyframes: list[MiniMaxH3KeyframeLatent] = []
    for index, item in enumerate(keyframes_value):
        if not isinstance(item, dict) or set(item) != {"resolved_frame_index", "video"}:
            raise ValueError(
                f"prepared H3 batch {path} keyframes[{index}] must contain index and video"
            )
        frame = cast("dict[str, object]", item)
        frame_index = frame["resolved_frame_index"]
        if type(frame_index) is not int:
            raise ValueError(
                f"prepared H3 batch {path} keyframes[{index}] index must be an integer"
            )
        keyframes.append(
            MiniMaxH3KeyframeLatent(
                frame_index,
                _h3_tensor(frame["video"], f"keyframes[{index}].video", path),
            )
        )
    references_value = raw.get("references", [])
    if not isinstance(references_value, list):
        raise ValueError(f"prepared H3 batch {path} references must be an array")
    references: list[MiniMaxH3ReferenceLatents] = []
    for index, item in enumerate(references_value):
        if not isinstance(item, dict) or not all(isinstance(key, str) for key in item):
            raise ValueError(f"prepared H3 batch {path} references[{index}] must be an object")
        reference = cast("dict[str, object]", item)
        if set(reference) - {"kind", "video", "audio"}:
            raise ValueError(f"prepared H3 batch {path} references[{index}] has unknown fields")
        try:
            kind = MiniMaxH3ReferenceKind(reference.get("kind"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"prepared H3 batch {path} references[{index}] has an invalid kind"
            ) from exc
        video_value = reference.get("video")
        audio_value = reference.get("audio")
        references.append(
            MiniMaxH3ReferenceLatents(
                kind,
                None
                if video_value is None
                else _h3_tensor(video_value, f"references[{index}].video", path),
                None
                if audio_value is None
                else _h3_tensor(audio_value, f"references[{index}].audio", path),
            )
        )
    frame_count = raw.get("frame_count")
    seed = raw.get("seed", 0)
    if frame_count is not None and type(frame_count) is not int:
        raise ValueError(f"prepared H3 batch {path} frame_count must be an integer")
    if type(seed) is not int:
        raise ValueError(f"prepared H3 batch {path} conditioning seed must be an integer")
    defaults = MiniMaxH3DiTConditioning()
    visual = raw.get("visual_noise_timestep", defaults.visual_noise_timestep)
    audio = raw.get("audio_noise_timestep", defaults.audio_noise_timestep)
    if type(visual) not in (int, float) or type(audio) not in (int, float):
        raise ValueError(f"prepared H3 batch {path} conditioning timesteps must be numbers")
    return MiniMaxH3DiTConditioning(
        text_token_tags=tags,
        keyframes=tuple(keyframes),
        references=tuple(references),
        frame_count=frame_count,
        visual_noise_timestep=float(cast("int | float", visual)),
        audio_noise_timestep=float(cast("int | float", audio)),
        seed=seed,
    )


class FilesystemMiniMaxH3PreparedBatchSource:
    """Read complete prepared H3 tensor batches named by absolute cursor."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxH3PreparedBatch:
        del generator
        path = self._root / f"{cursor:012d}.pt"
        try:
            value = torch.load(path, map_location=device, weights_only=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"no prepared H3 training batch for cursor {cursor}: {path}"
            ) from exc
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"prepared H3 batch {path} must contain a tensor mapping")
        raw = cast("dict[str, object]", value)
        allowed = {
            "video_latents",
            "audio_latents",
            "conditioner_embeddings",
            "conditioning",
            "sigma_indices",
            "video_noise",
            "audio_noise",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"prepared H3 batch {path} has unknown fields: {', '.join(unknown)}")
        sigma_value = raw.get("sigma_indices")
        video_noise_value = raw.get("video_noise")
        audio_noise_value = raw.get("audio_noise")
        supplied = tuple(
            item is not None for item in (sigma_value, video_noise_value, audio_noise_value)
        )
        if any(supplied) and not all(supplied):
            raise ValueError(
                f"prepared H3 batch {path} must provide sigma_indices and both noise streams"
            )
        return MiniMaxH3PreparedBatch(
            video_latents=_h3_tensor(raw.get("video_latents"), "video_latents", path),
            audio_latents=_h3_tensor(raw.get("audio_latents"), "audio_latents", path),
            conditioner_embeddings=_h3_tensor(
                raw.get("conditioner_embeddings"), "conditioner_embeddings", path
            ),
            conditioning=_h3_conditioning(raw.get("conditioning"), path),
            sigma_indices=(
                None if sigma_value is None else _h3_tensor(sigma_value, "sigma_indices", path)
            ),
            video_noise=(
                None
                if video_noise_value is None
                else _h3_tensor(video_noise_value, "video_noise", path)
            ),
            audio_noise=(
                None
                if audio_noise_value is None
                else _h3_tensor(audio_noise_value, "audio_noise", path)
            ),
        )


def _digest_bytes(data: bytes) -> str:
    return digest_bytes(data)


@dataclass(frozen=True)
class _FixedPathResolver:
    path: Path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def fixed_asset_ref(path: Path, digest: str, size: int) -> AssetRef:
    return AssetRef(digest, path.name, size, resolver=_FixedPathResolver(path))


def _verified_header(source: EncoderStateSource) -> SafetensorsSource:
    path = Path(source.path)
    path = fixed_asset_ref(path, source.digest, path.stat().st_size).local_path()
    return load_safetensors_header(path)


def validate_sdxl_training_plan(plan: SDAssemblyPlan) -> SDAssemblyPlan:
    """Require the complete unquantized SDXL base components used by training."""
    if plan.family is not SDXL or plan.sampling != SDXL.sampling:
        raise ValueError("the checkpoint must be a plain epsilon-prediction SDXL base")
    if plan.clip_l is None or plan.clip_g is None:
        raise ValueError("SDXL LoRA training requires checkpoint CLIP-L and CLIP-G")
    if not isinstance(plan.vae, ComponentPlan):
        raise ValueError("SDXL LoRA training requires the checkpoint's full KL VAE")
    for component in (plan.diffusion, plan.vae, plan.clip_l, plan.clip_g):
        if component.quant:
            raise ValueError(
                f"{component.component}: quantized training components are not supported"
            )
    return plan


def verified_sdxl_plan(source: EncoderStateSource) -> SDAssemblyPlan:
    """Plan one digest-pinned standard SDXL base checkpoint."""
    return validate_sdxl_training_plan(plan_sd_assembly(checkpoint=_verified_header(source)))


def load_planned_component(plan: ComponentPlan[_C], factory: Callable[[_C], _M]) -> _M:
    """Load one unquantized planned component without retaining its siblings."""
    if plan.quant:
        raise ValueError(f"{plan.component}: quantized training components are not supported")
    values = load_tensors(plan.path, set(plan.keys.values()))
    state: dict[str, torch.Tensor] = {}
    for model_key, source_key in plan.keys.items():
        value = values[source_key]
        transform = plan.transforms.get(model_key)
        if isinstance(transform, RowChunk):
            if value.ndim == 0:
                raise ValueError(f"{plan.component}: {model_key!r} row-chunk source is a scalar")
            rows, remainder = divmod(value.shape[0], transform.parts)
            if remainder:
                raise ValueError(
                    f"{plan.component}: {model_key!r} row-chunk source axis 0"
                    f" does not split into {transform.parts} equal chunks"
                )
            value = value[rows * transform.part : rows * (transform.part + 1)].contiguous()
        elif isinstance(transform, Transpose2D):
            if value.ndim != 2:
                raise ValueError(f"{plan.component}: {model_key!r} transpose source must be rank 2")
            value = value.transpose(0, 1).contiguous()
        elif isinstance(transform, LinearToConv2D):
            if value.ndim != 2:
                raise ValueError(
                    f"{plan.component}: {model_key!r} linear-to-conv source must be rank 2"
                )
            value = value.reshape(*value.shape, 1, 1)
        elif transform is not None:
            raise TypeError(
                f"{plan.component}: unknown tensor transform {type(transform).__name__}"
            )
        state[model_key] = value
    with torch.device("meta"):
        model = factory(plan.config)
    for key in plan.absent:
        if key != "text_projection.weight":
            raise ValueError(f"{plan.component}: no training default for absent key {key!r}")
        hidden = state["text_model.embeddings.token_embedding.weight"].shape[1]
        state[key] = torch.eye(hidden, dtype=torch.float32)
    model.load_state_dict(state, strict=True, assign=True)
    meta_buffers = tuple(
        name for name, buffer in model.named_buffers(remove_duplicate=False) if buffer.is_meta
    )
    if meta_buffers:
        skeleton = factory(plan.config)
        skeleton_buffers = dict(skeleton.named_buffers(remove_duplicate=False))
        for name in meta_buffers:
            replacement = skeleton_buffers.get(name)
            if replacement is None or replacement.is_meta:
                raise RuntimeError(
                    f"{plan.component}: CPU skeleton did not materialize buffer {name!r}"
                )
            owner_name, _, buffer_name = name.rpartition(".")
            owner = model.get_submodule(owner_name)
            setattr(owner, buffer_name, replacement.detach().clone())
        del skeleton
    remaining_meta = tuple(
        [
            f"parameter {name!r}"
            for name, parameter in model.named_parameters(remove_duplicate=False)
            if parameter.is_meta
        ]
        + [
            f"buffer {name!r}"
            for name, buffer in model.named_buffers(remove_duplicate=False)
            if buffer.is_meta
        ]
    )
    if remaining_meta:
        raise RuntimeError(
            f"{plan.component}: planned component load left meta tensors: "
            + ", ".join(remaining_meta)
        )
    return model


def inspect_encoder_memory(
    settings: ImageCaptionDatasetSettings,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Count transient float32 encoder storage without loading tensor payloads."""
    categories: dict[str, int] = {}
    errors: list[str] = []
    if settings.family == "sdxl":
        assert settings.checkpoint_state is not None
        try:
            header = _verified_header(settings.checkpoint_state)
            plan = validate_sdxl_training_plan(plan_sd_assembly(checkpoint=header))
            assert plan.clip_l is not None and plan.clip_g is not None
            assert isinstance(plan.vae, ComponentPlan)
            components = (
                ("vaeEncoderParametersTransientLowerBound", plan.vae),
                ("clipLTextEncoderParametersTransientLowerBound", plan.clip_l),
                ("clipGTextEncoderParametersTransientLowerBound", plan.clip_g),
            )
            for name, component in components:
                source_keys = set(component.keys.values())
                categories[name] = sum(header.entry(key).geometry.numel for key in source_keys) * 4
        except (OSError, ValueError) as exc:
            errors.append(f"sdxlEncoderParametersTransientLowerBound: {exc}")
        return categories, tuple(errors)
    assert settings.vae_state is not None and settings.text_encoder_state is not None
    for name, source in (
        ("vaeEncoderParametersTransientLowerBound", settings.vae_state),
        ("clipTextEncoderParametersTransientLowerBound", settings.text_encoder_state),
    ):
        try:
            header = _verified_header(source)
            scoped = [
                entry
                for key, entry in header.entries.items()
                if key.startswith(source.prefix)
                and (
                    name != "clipTextEncoderParametersTransientLowerBound"
                    or key[len(source.prefix) :] not in _CLIP_TEXT_INERT_KEYS
                )
            ]
            if not scoped:
                raise ValueError(f"no tensors use prefix {source.prefix!r}")
            categories[name] = sum(entry.geometry.numel for entry in scoped) * 4
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    return categories, tuple(errors)


def inspect_minimax_h3_encoder_memory(
    settings: MiniMaxH3DatasetSettings,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Validate H3 component pins and count transient source storage."""
    categories: dict[str, int] = {}
    errors: list[str] = []
    components = (
        (
            "videoVaeParametersTransientLowerBound",
            "video-vae",
            settings.video_vae_state,
            torch.float16,
        ),
        (
            "audioVaeParametersTransientLowerBound",
            "audio-vae",
            settings.audio_vae_state,
            torch.float16,
        ),
        (
            "conditionerParametersTransientLowerBound",
            "qwen3vl-32b-conditioner",
            settings.conditioner_state,
            torch.bfloat16,
        ),
    )
    for category, role, state, dtype in components:
        try:
            path = Path(state.path)
            if path.stat().st_size != state.size:
                raise ValueError(f"artifact size differs from pinned size {state.size}")
            header = load_safetensors_header(
                path,
                asset_digest=state.digest,
                asset_size=state.size,
            )
            plan = plan_minimax_h3_common_component(
                header,
                role=cast("MiniMaxH3CommonComponentRole", role),
                path=path,
                context=worker_planning_context(),
            )
            if role == "video-vae":
                identity = minimax_h3_video_vae_runtime_identity(
                    cast("ComponentPlan[MiniMaxH3VideoVAEConfig]", plan),
                    video_vae_dtype=dtype,
                )
            elif role == "audio-vae":
                identity = minimax_h3_audio_vae_runtime_identity(
                    cast("ComponentPlan[None]", plan),
                    audio_vae_dtype=dtype,
                )
            else:
                identity = minimax_h3_conditioner_runtime_identity(
                    cast("ComponentPlan[MiniMaxH3ConditionerConfig]", plan),
                    conditioner_dtype=dtype,
                )
            if identity != state.identity:
                raise ValueError(
                    f"expected component identity {state.identity!r}, constructed {identity!r}"
                )
            categories[category] = sum(entry.geometry.nbytes for entry in header.entries.values())
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"{category}: {exc}")
    return categories, tuple(errors)


def _scoped_geometries(source: SafetensorsSource, prefix: str) -> dict[str, TensorGeometry]:
    return {
        key[len(prefix) :]: entry.geometry
        for key, entry in source.entries.items()
        if key.startswith(prefix)
    }


def _load_vae(source: EncoderStateSource) -> AutoencoderKL:
    header = _verified_header(source)
    geometries = _scoped_geometries(header, source.prefix)
    config = detect_kl_config(geometries)
    with torch.device("meta"):
        model = AutoencoderKL(config)
    keys = tuple(model.state_dict())
    values = load_tensors(Path(source.path), (source.prefix + key for key in keys))
    model.load_state_dict(
        {key: values[source.prefix + key] for key in keys}, strict=True, assign=True
    )
    return model


def _detect_training_clip_text_config(
    geometries: dict[str, TensorGeometry],
) -> ClipTextConfig:
    return detect_clip_text_config(
        {key: geometry for key, geometry in geometries.items() if key not in _CLIP_TEXT_INERT_KEYS}
    )


def _load_text_encoder(source: EncoderStateSource) -> ClipTextModel:
    header = _verified_header(source)
    geometries = _scoped_geometries(header, source.prefix)
    config = _detect_training_clip_text_config(geometries)
    with torch.device("meta"):
        model = ClipTextModel(config)
    state_keys = tuple(model.state_dict())
    available = set(geometries)
    required = tuple(key for key in state_keys if key not in CLIP_TEXT_OPTIONAL_KEYS)
    values = load_tensors(Path(source.path), (source.prefix + key for key in required))
    state = {key: values[source.prefix + key] for key in required}
    for key in CLIP_TEXT_OPTIONAL_KEYS:
        if key in available:
            state[key] = load_tensors(Path(source.path), (source.prefix + key,))[
                source.prefix + key
            ]
        else:
            state[key] = torch.eye(model.config.hidden_size, dtype=torch.float32)
    model.load_state_dict(state, strict=True, assign=True)
    return model


def default_image_caption_source(
    settings: ImageCaptionDatasetSettings,
    *,
    batch_size: int,
    device: torch.device,
) -> ImageCaptionDatasetSource:
    """Precompute a folder dataset with the pinned native frozen encoders."""
    if settings.family == "sdxl":
        assert settings.checkpoint_state is not None
        checkpoint_state = settings.checkpoint_state
        plan: SDAssemblyPlan | None = None

        def planned() -> SDAssemblyPlan:
            nonlocal plan
            if plan is None:
                plan = verified_sdxl_plan(checkpoint_state)
            return plan

        def load_vae() -> AutoencoderKL:
            vae_plan = planned().vae
            assert isinstance(vae_plan, ComponentPlan)
            return load_planned_component(vae_plan, AutoencoderKL)

        def load_clip_l() -> ClipTextModel:
            clip_l_plan = planned().clip_l
            assert clip_l_plan is not None
            return load_planned_component(clip_l_plan, ClipTextModel)

        def load_clip_g() -> ClipTextModel:
            clip_g_plan = planned().clip_g
            assert clip_g_plan is not None
            return load_planned_component(clip_g_plan, ClipTextModel)

        return ImageCaptionDatasetSource(
            settings,
            load_vae,
            load_clip_l,
            load_clip_g,
            batch_size=batch_size,
            device=device,
        )
    assert settings.vae_state is not None and settings.text_encoder_state is not None
    vae_state = settings.vae_state
    text_encoder_state = settings.text_encoder_state
    return ImageCaptionDatasetSource(
        settings,
        lambda: _load_vae(vae_state),
        lambda: _load_text_encoder(text_encoder_state),
        batch_size=batch_size,
        device=device,
    )


class ImageCaptionDatasetSource:
    """Serve deterministic shuffled batches from a CPU encoded-item store."""

    def __init__(
        self,
        settings: ImageCaptionDatasetSettings,
        vae_factory: Callable[[], AutoencoderKL],
        text_model_factory: Callable[[], ClipTextModel],
        clip_g_model_factory: Callable[[], ClipTextModel] | None = None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        current = inspect_image_caption_dataset(
            Path(settings.root),
            settings.resolution,
            settings.vae_state,
            settings.text_encoder_state,
            family=settings.family,
            checkpoint_state=settings.checkpoint_state,
        )
        if current.digest != settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {settings.digest}, got {current.digest}"
            )
        if current.errors:
            raise ValueError("dataset validation failed: " + "; ".join(current.errors))
        if batch_size < 1:
            raise ValueError("image/caption batch size must be positive")
        self.settings = settings
        self._items = current.items
        self._batch_size = batch_size
        self._permutations: dict[int, tuple[int, ...]] = {}
        cache = (
            None
            if settings.encoded_cache_root is None
            else EncodedDatasetCache(settings, batch_size=batch_size, device=device)
        )
        cached = None if cache is None else cache.load()
        if cached is not None:
            self._use_cached(cached)
            return
        if cache is None:
            self._build(vae_factory, text_model_factory, clip_g_model_factory, device)
            return
        with cache.build_lock():
            cached = cache.load()
            if cached is not None:
                self._use_cached(cached)
                return
            self._build(vae_factory, text_model_factory, clip_g_model_factory, device)
            cache.write(self._cache_tensors())

    def _use_cached(self, cached: EncodedDatasetTensors) -> None:
        self._latents = tuple(cached.latents.unbind(0))
        self._text_embeddings = tuple(cached.text_embeddings.unbind(0))
        self._pooled_embeddings = (
            None if cached.pooled_embeddings is None else tuple(cached.pooled_embeddings.unbind(0))
        )
        self._time_ids = cached.time_ids

    def _build(
        self,
        vae_factory: Callable[[], AutoencoderKL],
        text_model_factory: Callable[[], ClipTextModel],
        clip_g_model_factory: Callable[[], ClipTextModel] | None,
        device: torch.device,
    ) -> None:
        self._tokenizer = PromptTokenizer(
            encode_word=load_clip_bpe().encode,
            disable_weights=True,
        )
        try:
            self._latents = self._precompute_latents(vae_factory, device)
        finally:
            self._release_cuda_cache(device)
        try:
            clip_l, _ = self._precompute_text_embeddings(
                text_model_factory,
                device,
                clip_g=False,
            )
        finally:
            self._release_cuda_cache(device)
        if self.settings.family == "sdxl":
            if clip_g_model_factory is None:
                raise ValueError("SDXL dataset precompute requires a CLIP-G factory")
            try:
                clip_g, pooled = self._precompute_text_embeddings(
                    clip_g_model_factory,
                    device,
                    clip_g=True,
                )
            finally:
                self._release_cuda_cache(device)
            if pooled is None:
                raise ValueError("SDXL CLIP-G produced no pooled conditioning")
            self._text_embeddings = tuple(
                torch.cat((left, right), dim=-1) for left, right in zip(clip_l, clip_g, strict=True)
            )
            self._pooled_embeddings = pooled
            height, width = self.settings.resolution
            self._time_ids = torch.tensor([height, width, 0, 0, height, width], dtype=torch.float32)
        else:
            if clip_g_model_factory is not None:
                raise ValueError("SD1.5 dataset precompute does not use CLIP-G")
            self._text_embeddings = clip_l
            self._pooled_embeddings = None
            self._time_ids = None

    def _cache_tensors(self) -> EncodedDatasetTensors:
        return EncodedDatasetTensors(
            latents=torch.stack(self._latents),
            text_embeddings=torch.stack(self._text_embeddings),
            pooled_embeddings=(
                None if self._pooled_embeddings is None else torch.stack(self._pooled_embeddings)
            ),
            time_ids=self._time_ids,
        )

    @staticmethod
    def _release_cuda_cache(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def _permutation(self, epoch: int) -> tuple[int, ...]:
        found = self._permutations.get(epoch)
        if found is not None:
            return found
        preimage = f"{self.settings.digest}:epoch:{epoch}".encode("ascii")
        seed = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        found = tuple(torch.randperm(len(self._items), generator=generator).tolist())
        self._permutations[epoch] = found
        return found

    def _item_index(self, absolute_sample: int) -> int:
        epoch, offset = divmod(absolute_sample, len(self._items))
        return self._permutation(epoch)[offset]

    @staticmethod
    def _read_image(item: ImageCaptionItem) -> bytes:
        image = item.image_path.read_bytes()
        if _digest_bytes(image) != item.image_digest:
            raise ValueError(f"dataset image changed after validation: {item.relative_image}")
        return image

    @staticmethod
    def _read_caption(item: ImageCaptionItem) -> str:
        caption_data = item.caption_path.read_bytes()
        if _digest_bytes(caption_data) != item.caption_digest:
            raise ValueError(f"dataset caption changed after validation: {item.relative_caption}")
        caption = caption_data.decode("utf-8").strip()
        if not caption:
            raise ValueError(f"dataset caption is empty: {item.relative_caption}")
        return caption

    def _pixels(self, data: bytes) -> torch.Tensor:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            target_height, target_width = self.settings.resolution
            if width * target_height > height * target_width:
                crop_width = height * target_width // target_height
                left = (width - crop_width) // 2
                image = image.crop((left, 0, left + crop_width, height))
            elif width * target_height < height * target_width:
                crop_height = width * target_height // target_width
                top = (height - crop_height) // 2
                image = image.crop((0, top, width, top + crop_height))
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            pixels = pixels.reshape(image.height, image.width, 3).permute(2, 0, 1)
        pixels = pixels.to(dtype=torch.float32).div_(255.0).unsqueeze(0)
        if tuple(pixels.shape[2:]) != self.settings.resolution:
            pixels = functional.interpolate(
                pixels,
                size=self.settings.resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return pixels.squeeze(0)

    def _precompute_latents(
        self,
        vae_factory: Callable[[], AutoencoderKL],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        vae = vae_factory().requires_grad_(False).eval()
        vae.to(device=device, dtype=torch.float32)
        latents: list[torch.Tensor] = []
        with torch.no_grad():
            for item in self._items:
                pixels = self._pixels(self._read_image(item)).unsqueeze(0)
                # VAE kernels can select different algorithms by batch size.
                pixels = pixels.expand(self._batch_size, -1, -1, -1).contiguous()
                pixels = pixels.to(device=device)
                encoded = vae.encode(process_input(pixels))
                family = SD15 if self.settings.family == "sd15" else SDXL
                encoded = encoded.float() * family.single_stream_latent().scale_factor
                latents.append(encoded[0].detach().cpu())
        return tuple(latents)

    def _precompute_text_embeddings(
        self,
        text_model_factory: Callable[[], ClipTextModel],
        device: torch.device,
        *,
        clip_g: bool,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
        text_model = text_model_factory().requires_grad_(False).eval()
        text_model.to(device=device, dtype=torch.float32)
        if self.settings.family == "sd15":
            text_encoder = ClipTextEncoder(text_model)
            profile = CLIP_L_PROFILE
        else:
            profile = CLIP_G_PROFILE if clip_g else CLIP_L_PROFILE
            text_encoder = ClipTextEncoder(
                text_model,
                profile=profile,
                policy=SDXL_CLIP_POLICY,
            )
        embeddings: list[torch.Tensor] = []
        pooled: list[torch.Tensor] = []
        with torch.no_grad():
            for item in self._items:
                spans = self._tokenizer.tokenize(self._read_caption(item))
                chunks = pack_spans(spans, profile)
                encoded = text_encoder.encode_chunks(chunks[:1])
                embeddings.append(encoded.embeddings.squeeze(0).detach().cpu())
                if clip_g:
                    if encoded.pooled is None:
                        raise ValueError("CLIP-G produced no pooled output")
                    pooled.append(encoded.pooled.squeeze(0).detach().cpu())
        return tuple(embeddings), tuple(pooled) if clip_g else None

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        del generator
        if cursor < 0:
            raise ValueError("data cursor must be non-negative")
        selected = [
            self._item_index(cursor * self._batch_size + offset)
            for offset in range(self._batch_size)
        ]
        pooled = (
            None
            if self._pooled_embeddings is None
            else torch.stack([self._pooled_embeddings[index] for index in selected]).to(
                device=device
            )
        )
        time_ids = (
            None
            if self._time_ids is None
            else self._time_ids.unsqueeze(0).expand(self._batch_size, -1).to(device=device)
        )
        return PreparedBatch(
            latents=torch.stack([self._latents[index] for index in selected]).to(device=device),
            text_embeddings=torch.stack([self._text_embeddings[index] for index in selected]).to(
                device=device
            ),
            pooled_embeddings=pooled,
            time_ids=time_ids,
        )


class FluxDatasetSource:
    """Serve deterministic classic Flux batches from encoded images and captions."""

    def __init__(
        self,
        settings: FluxDatasetSettings,
        vae_factory: Callable[[], AutoencoderKL],
        clip_factory: Callable[[], ClipTextModel],
        t5_factory: Callable[[], T5TextModel],
        *,
        latent_shape: tuple[int, int, int, int],
        context_shape: tuple[int, int, int],
        pooled_shape: tuple[int, int],
        artifact_identities: tuple[object, object, object],
        device: torch.device,
    ) -> None:
        current = inspect_flux_dataset(
            Path(settings.root),
            settings.resolution,
            settings.context_tokens,
            variant=settings.variant,
            vae_identity=artifact_identities[0],
            clip_l_identity=artifact_identities[1],
            t5xxl_identity=artifact_identities[2],
        )
        if current.digest != settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {settings.digest}, got {current.digest}"
            )
        if current.errors:
            raise ValueError("dataset validation failed: " + "; ".join(current.errors))
        self.settings = settings
        self._items = current.items
        self._shapes = (latent_shape, context_shape, pooled_shape)
        self._permutations: dict[int, tuple[int, ...]] = {}
        cache = (
            None
            if settings.encoded_cache_root is None
            else FluxEncodedDatasetCache(
                settings,
                latent_shape=latent_shape,
                context_shape=context_shape,
                pooled_shape=pooled_shape,
                device=device,
            )
        )
        cached = None if cache is None else cache.load()
        if cached is not None:
            self._use_cached(cached)
            return
        if cache is None:
            self._build(vae_factory, clip_factory, t5_factory, device)
            return
        with cache.build_lock():
            cached = cache.load()
            if cached is not None:
                self._use_cached(cached)
                return
            self._build(vae_factory, clip_factory, t5_factory, device)
            cache.write(self._cache_tensors())

    def _use_cached(self, tensors: EncodedFluxDatasetTensors) -> None:
        self._latents = tuple(tensors.latents.unbind())
        self._contexts = tuple(tensors.context.unbind())
        self._pooled = tuple(tensors.pooled.unbind())
        self._validate()

    @staticmethod
    def _release(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def _build(
        self,
        vae_factory: Callable[[], AutoencoderKL],
        clip_factory: Callable[[], ClipTextModel],
        t5_factory: Callable[[], T5TextModel],
        device: torch.device,
    ) -> None:
        try:
            vae = vae_factory().requires_grad_(False).eval().to(device=device, dtype=torch.float32)
            with torch.no_grad():
                self._latents = tuple(
                    (
                        vae.encode(process_input(self._pixels(item).unsqueeze(0).to(device)))
                        .sub(0.1159)
                        .mul(0.3611)
                        .float()
                        .squeeze(0)
                        .cpu()
                    )
                    for item in self._items
                )
            del vae
        finally:
            self._release(device)
        captions = tuple(self._caption(item) for item in self._items)
        try:
            model = (
                clip_factory().requires_grad_(False).eval().to(device=device, dtype=torch.float32)
            )
            encoder = ClipTextEncoder(model)
            tokenizer = PromptTokenizer(encode_word=load_clip_bpe().encode)
            pooled: list[torch.Tensor] = []
            with torch.no_grad():
                for caption in captions:
                    encoded = encoder.encode(tokenizer.tokenize(caption))
                    if encoded.pooled is None:
                        raise ValueError("Flux CLIP-L produced no pooled output")
                    pooled.append(encoded.pooled.float().squeeze(0).cpu())
            self._pooled = tuple(pooled)
            del encoder, model
        finally:
            self._release(device)
        try:
            model = t5_factory().requires_grad_(False).eval().to(device=device, dtype=torch.float32)
            encoder = T5TextEncoder(model)
            tokenizer = PromptTokenizer(encode_word=load_t5_spm().encode)
            profile = replace(T5_XXL_FLUX_PROFILE, min_length=self.settings.context_tokens)
            contexts: list[torch.Tensor] = []
            with torch.no_grad():
                for caption in captions:
                    token_count = self.settings.context_tokens
                    chunks = pack_spans(tokenizer.tokenize(caption), profile)
                    if sum(len(chunk) for chunk in chunks) > token_count:
                        raise ValueError(
                            f"Flux caption exceeds configured {token_count}-token context"
                        )
                    value = encoder.encode_chunks(chunks).embeddings.float().squeeze(0)
                    contexts.append(value[:token_count].cpu())
            self._contexts = tuple(contexts)
            del encoder, model
        finally:
            self._release(device)
        self._validate()

    def _pixels(self, item: ImageCaptionItem) -> torch.Tensor:
        data = item.image_path.read_bytes()
        if _digest_bytes(data) != item.image_digest:
            raise ValueError(f"dataset image changed after validation: {item.relative_image}")
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            target_height, target_width = self.settings.resolution
            if width * target_height > height * target_width:
                crop_width = height * target_width // target_height
                left = (width - crop_width) // 2
                image = image.crop((left, 0, left + crop_width, height))
            elif width * target_height < height * target_width:
                crop_height = width * target_height // target_width
                top = (height - crop_height) // 2
                image = image.crop((0, top, width, top + crop_height))
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            pixels = pixels.reshape(image.height, image.width, 3).permute(2, 0, 1)
        value = pixels.float().div_(255.0).unsqueeze(0)
        if tuple(value.shape[2:]) != self.settings.resolution:
            value = functional.interpolate(
                value,
                size=self.settings.resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return value.squeeze(0)

    @staticmethod
    def _caption(item: ImageCaptionItem) -> str:
        data = item.caption_path.read_bytes()
        if _digest_bytes(data) != item.caption_digest:
            raise ValueError(f"dataset caption changed after validation: {item.relative_caption}")
        value = data.decode("utf-8").strip()
        if not value:
            raise ValueError(f"dataset caption is empty: {item.relative_caption}")
        return value

    def _validate(self) -> None:
        count = len(self._items)
        stores = (self._latents, self._contexts, self._pooled)
        if any(len(store) != count for store in stores):
            raise ValueError("encoded Flux dataset item counts do not match")
        for store, shape in zip(stores, self._shapes, strict=True):
            if any(
                value.dtype != torch.float32 or tuple(value.shape) != shape[1:] for value in store
            ):
                raise ValueError("encoded Flux dataset tensor has the wrong shape or dtype")

    def _cache_tensors(self) -> EncodedFluxDatasetTensors:
        return EncodedFluxDatasetTensors(
            torch.stack(self._latents), torch.stack(self._contexts), torch.stack(self._pooled)
        )

    def _item_index(self, cursor: int) -> int:
        epoch, offset = divmod(cursor, len(self._items))
        found = self._permutations.get(epoch)
        if found is None:
            digest = digest_bytes(f"{self.settings.digest}:epoch:{epoch}".encode("ascii"))
            generator = torch.Generator(device="cpu").manual_seed(
                int(digest[7:23], 16) & ((1 << 63) - 1)
            )
            found = tuple(torch.randperm(len(self._items), generator=generator).tolist())
            self._permutations[epoch] = found
        return found[offset]

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> FluxPreparedBatch:
        del generator
        if cursor < 0:
            raise ValueError("data cursor must be non-negative")
        batch_size = self._shapes[0][0]
        selected = [self._item_index(cursor * batch_size + offset) for offset in range(batch_size)]
        return FluxPreparedBatch(
            torch.stack([self._latents[index] for index in selected]).to(device),
            torch.stack([self._contexts[index] for index in selected]).to(device),
            torch.stack([self._pooled[index] for index in selected]).to(device),
        )


class Flux2DatasetSource:
    """Serve deterministic Flux2 batches from encoded images and captions."""

    def __init__(
        self,
        settings: Flux2DatasetSettings,
        vae_factory: Callable[[], AutoencoderKL],
        text_encoder_factory: Callable[[], QwenTextModel],
        *,
        latent_shape: tuple[int, int, int, int],
        context_shape: tuple[int, int, int],
        artifact_identities: tuple[object, object],
        device: torch.device,
    ) -> None:
        current = inspect_flux2_dataset(
            Path(settings.root),
            settings.resolution,
            settings.context_tokens,
            variant=settings.variant,
            vae_identity=artifact_identities[0],
            text_encoder_identity=artifact_identities[1],
        )
        if current.digest != settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {settings.digest}, got {current.digest}"
            )
        if current.errors:
            raise ValueError("dataset validation failed: " + "; ".join(current.errors))
        self.settings = settings
        self._items = current.items
        self._shapes = (latent_shape, context_shape)
        self._permutations: dict[int, tuple[int, ...]] = {}
        cache = (
            None
            if settings.encoded_cache_root is None
            else Flux2EncodedDatasetCache(
                settings,
                latent_shape=latent_shape,
                context_shape=context_shape,
                device=device,
            )
        )
        cached = None if cache is None else cache.load()
        if cached is not None:
            self._use_cached(cached)
            return
        if cache is None:
            self._build(vae_factory, text_encoder_factory, device)
            return
        with cache.build_lock():
            cached = cache.load()
            if cached is not None:
                self._use_cached(cached)
                return
            self._build(vae_factory, text_encoder_factory, device)
            cache.write(self._cache_tensors())

    def _use_cached(self, tensors: EncodedFlux2DatasetTensors) -> None:
        self._latents = tuple(tensors.latents.unbind())
        self._contexts = tuple(tensors.context.unbind())
        self._validate()

    @staticmethod
    def _release(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def _build(
        self,
        vae_factory: Callable[[], AutoencoderKL],
        text_encoder_factory: Callable[[], QwenTextModel],
        device: torch.device,
    ) -> None:
        try:
            vae = vae_factory().requires_grad_(False).eval().to(device=device, dtype=torch.float32)
            with torch.no_grad():
                self._latents = tuple(
                    (
                        vae.encode(process_input(self._pixels(item).unsqueeze(0).to(device)))
                        .float()
                        .squeeze(0)
                        .cpu()
                    )
                    for item in self._items
                )
            del vae
        finally:
            self._release(device)
        captions = tuple(self._caption(item) for item in self._items)
        dev_tokenizer = load_flux2_tekken_bpe() if self.settings.variant == "flux2-dev" else None
        if dev_tokenizer is not None:
            prompt_tokens = tuple(
                tokenize_flux2_dev_prompt(caption, tokenizer=dev_tokenizer) for caption in captions
            )
        else:
            prompt_tokens = tuple(tokenize_flux2_klein_prompt(caption) for caption in captions)
        token_count = self.settings.context_tokens
        if any(len(tokens.ids) > token_count for tokens in prompt_tokens):
            raise ValueError(f"Flux2 caption exceeds configured {token_count}-token context")
        try:
            model = text_encoder_factory().requires_grad_(False).eval().to(device=device)
            encoder = (
                Flux2DevTextEncoder(model, dev_tokenizer)
                if dev_tokenizer is not None
                else Flux2KleinTextEncoder(model)
            )
            contexts: list[torch.Tensor] = []
            with torch.no_grad():
                for caption in captions:
                    value = encoder.encode(caption).embeddings.float().squeeze(0)
                    if value.shape[0] > token_count:
                        raise ValueError(
                            f"Flux2 caption exceeds configured {token_count}-token context"
                        )
                    missing = token_count - value.shape[0]
                    padding = (
                        (0, 0, missing, 0)
                        if self.settings.variant == "flux2-dev"
                        else (0, 0, 0, missing)
                    )
                    value = functional.pad(value, padding)
                    contexts.append(value.cpu())
            self._contexts = tuple(contexts)
            del encoder, model
        finally:
            self._release(device)
        self._validate()

    def _pixels(self, item: ImageCaptionItem) -> torch.Tensor:
        data = item.image_path.read_bytes()
        if _digest_bytes(data) != item.image_digest:
            raise ValueError(f"dataset image changed after validation: {item.relative_image}")
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            target_height, target_width = self.settings.resolution
            if width * target_height > height * target_width:
                crop_width = height * target_width // target_height
                left = (width - crop_width) // 2
                image = image.crop((left, 0, left + crop_width, height))
            elif width * target_height < height * target_width:
                crop_height = width * target_height // target_width
                top = (height - crop_height) // 2
                image = image.crop((0, top, width, top + crop_height))
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            pixels = pixels.reshape(image.height, image.width, 3).permute(2, 0, 1)
        value = pixels.float().div_(255.0).unsqueeze(0)
        if tuple(value.shape[2:]) != self.settings.resolution:
            value = functional.interpolate(
                value,
                size=self.settings.resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return value.squeeze(0)

    @staticmethod
    def _caption(item: ImageCaptionItem) -> str:
        data = item.caption_path.read_bytes()
        if _digest_bytes(data) != item.caption_digest:
            raise ValueError(f"dataset caption changed after validation: {item.relative_caption}")
        value = data.decode("utf-8").strip()
        if not value:
            raise ValueError(f"dataset caption is empty: {item.relative_caption}")
        return value

    def _validate(self) -> None:
        count = len(self._items)
        stores = (self._latents, self._contexts)
        if any(len(store) != count for store in stores):
            raise ValueError("encoded Flux2 dataset item counts do not match")
        for store, shape in zip(stores, self._shapes, strict=True):
            if any(
                value.dtype != torch.float32 or tuple(value.shape) != shape[1:] for value in store
            ):
                raise ValueError("encoded Flux2 dataset tensor has the wrong shape or dtype")

    def _cache_tensors(self) -> EncodedFlux2DatasetTensors:
        return EncodedFlux2DatasetTensors(torch.stack(self._latents), torch.stack(self._contexts))

    def _item_index(self, cursor: int) -> int:
        epoch, offset = divmod(cursor, len(self._items))
        found = self._permutations.get(epoch)
        if found is None:
            digest = digest_bytes(f"{self.settings.digest}:epoch:{epoch}".encode("ascii"))
            generator = torch.Generator(device="cpu").manual_seed(
                int(digest[7:23], 16) & ((1 << 63) - 1)
            )
            found = tuple(torch.randperm(len(self._items), generator=generator).tolist())
            self._permutations[epoch] = found
        return found[offset]

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> Flux2PreparedBatch:
        del generator
        if cursor < 0:
            raise ValueError("data cursor must be non-negative")
        batch_size = self._shapes[0][0]
        selected = [self._item_index(cursor * batch_size + offset) for offset in range(batch_size)]
        return Flux2PreparedBatch(
            torch.stack([self._latents[index] for index in selected]).to(device),
            torch.stack([self._contexts[index] for index in selected]).to(device),
        )


class QwenImageDatasetSource:
    """Serve Qwen-Image batches from sequentially precomputed encoder outputs."""

    def __init__(
        self,
        settings: QwenImageDatasetSettings,
        vae_factory: Callable[[], WanVAE],
        text_encoder_factory: Callable[[], QwenImageTextModel],
        *,
        latent_shape: tuple[int, int, int, int, int],
        context_shape: tuple[int, int, int],
        attention_mask_shape: tuple[int, int],
        vae_identity: object,
        text_encoder_identity: object,
        device: torch.device,
    ) -> None:
        current = inspect_qwen_image_dataset(
            Path(settings.root),
            settings.resolution,
            settings.context_tokens,
            vae_identity=vae_identity,
            text_encoder_identity=text_encoder_identity,
        )
        if current.digest != settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {settings.digest}, got {current.digest}"
            )
        if current.errors:
            raise ValueError("dataset validation failed: " + "; ".join(current.errors))
        self.settings = settings
        self._items = current.items
        self._shapes = (latent_shape, context_shape, attention_mask_shape)
        self._permutations: dict[int, tuple[int, ...]] = {}
        cache = (
            None
            if settings.encoded_cache_root is None
            else QwenImageEncodedDatasetCache(
                settings,
                latent_shape=latent_shape,
                context_shape=context_shape,
                attention_mask_shape=attention_mask_shape,
                device=device,
            )
        )
        cached = None if cache is None else cache.load()
        if cached is not None:
            self._use_cached(cached)
            return
        if cache is None:
            self._build(vae_factory, text_encoder_factory, device)
            return
        with cache.build_lock():
            cached = cache.load()
            if cached is not None:
                self._use_cached(cached)
                return
            self._build(vae_factory, text_encoder_factory, device)
            cache.write(self._cache_tensors())

    def _use_cached(self, tensors: EncodedQwenImageDatasetTensors) -> None:
        self._latents = tuple(tensors.latents.unbind())
        self._contexts = tuple(tensors.context.unbind())
        self._attention_masks = tuple(tensors.attention_mask.unbind())
        self._validate()

    @staticmethod
    def _release(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def _prompt_rows(self) -> tuple[tuple[int, ...], ...]:
        tokenizer = load_qwen_bpe()
        rows: list[tuple[int, ...]] = []
        for item in self._items:
            prompt = format_qwen_image_prompt(self._caption(item))
            row = tuple(tokenizer.encode(prompt.text))
            if len(row) > QWEN_IMAGE_TEXT_CONFIG.max_position_embeddings:
                raise ValueError(
                    "Qwen-Image formatted prompt exceeds"
                    f" {QWEN_IMAGE_TEXT_CONFIG.max_position_embeddings}-token position limit"
                )
            selected = select_qwen_image_output((row,), None)
            if len(selected.token_rows[0]) > self.settings.context_tokens:
                raise ValueError(
                    "Qwen-Image caption exceeds configured"
                    f" {self.settings.context_tokens}-token context"
                )
            rows.append(row)
        return tuple(rows)

    def _build(
        self,
        vae_factory: Callable[[], WanVAE],
        text_encoder_factory: Callable[[], QwenImageTextModel],
        device: torch.device,
    ) -> None:
        rows = self._prompt_rows()
        vae: WanVAE | None = None
        try:
            vae = (
                vae_factory()
                .requires_grad_(False)
                .eval()
                .to(device=device, dtype=WAN21_ENCODING_DTYPE)
            )
            latents: list[torch.Tensor] = []
            with torch.no_grad():
                for item in self._items:
                    content = self._pixels(item).unsqueeze(0).unsqueeze(2).to(device)
                    encoded = vae.process_in(vae.encode(content.mul(2.0).sub(1.0)))
                    latents.append(encoded.float().squeeze(0).cpu())
            self._latents = tuple(latents)
        finally:
            vae = None
            self._release(device)
        model: QwenImageTextModel | None = None
        try:
            model = text_encoder_factory().requires_grad_(False).eval().to(device=device)
            contexts: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            with torch.no_grad():
                for row in rows:
                    ids = torch.tensor((row,), dtype=torch.long, device=device)
                    value, model_mask = model(ids)
                    if model_mask is not None:
                        raise ValueError("unpadded Qwen-Image text encoding returned a mask")
                    selected = select_qwen_image_output((row,), None)
                    selected_tokens = len(selected.token_rows[0])
                    expected = (1, selected_tokens, self._shapes[1][2])
                    if value.shape != expected:
                        raise ValueError(
                            "Qwen-Image text encoder returned"
                            f" {tuple(value.shape)}; expected {expected}"
                        )
                    missing = self.settings.context_tokens - selected_tokens
                    contexts.append(
                        functional.pad(value.float().squeeze(0), (0, 0, 0, missing)).cpu()
                    )
                    masks.append(
                        functional.pad(
                            torch.ones(selected_tokens, dtype=torch.float32), (0, missing)
                        )
                    )
            self._contexts = tuple(contexts)
            self._attention_masks = tuple(masks)
        finally:
            model = None
            self._release(device)
        self._validate()

    def _pixels(self, item: ImageCaptionItem) -> torch.Tensor:
        data = item.image_path.read_bytes()
        if _digest_bytes(data) != item.image_digest:
            raise ValueError(f"dataset image changed after validation: {item.relative_image}")
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            target_height, target_width = self.settings.resolution
            if width * target_height > height * target_width:
                crop_width = height * target_width // target_height
                left = (width - crop_width) // 2
                image = image.crop((left, 0, left + crop_width, height))
            elif width * target_height < height * target_width:
                crop_height = width * target_height // target_width
                top = (height - crop_height) // 2
                image = image.crop((0, top, width, top + crop_height))
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            pixels = pixels.reshape(image.height, image.width, 3).permute(2, 0, 1)
        value = pixels.float().div_(255.0).unsqueeze(0)
        if tuple(value.shape[2:]) != self.settings.resolution:
            value = functional.interpolate(
                value,
                size=self.settings.resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return value.squeeze(0)

    @staticmethod
    def _caption(item: ImageCaptionItem) -> str:
        data = item.caption_path.read_bytes()
        if _digest_bytes(data) != item.caption_digest:
            raise ValueError(f"dataset caption changed after validation: {item.relative_caption}")
        value = data.decode("utf-8").strip()
        if not value:
            raise ValueError(f"dataset caption is empty: {item.relative_caption}")
        return value

    def _validate(self) -> None:
        count = len(self._items)
        stores = (self._latents, self._contexts, self._attention_masks)
        if any(len(store) != count for store in stores):
            raise ValueError("encoded Qwen-Image dataset item counts do not match")
        for store, shape in zip(stores, self._shapes, strict=True):
            if any(
                value.dtype != torch.float32 or tuple(value.shape) != shape[1:] for value in store
            ):
                raise ValueError("encoded Qwen-Image dataset tensor has the wrong shape or dtype")
        if any(
            not bool(torch.all((mask == 0.0) | (mask == 1.0)).item())
            for mask in self._attention_masks
        ):
            raise ValueError("encoded Qwen-Image attention mask must be binary")

    def _cache_tensors(self) -> EncodedQwenImageDatasetTensors:
        return EncodedQwenImageDatasetTensors(
            torch.stack(self._latents),
            torch.stack(self._contexts),
            torch.stack(self._attention_masks),
        )

    def _item_index(self, cursor: int) -> int:
        epoch, offset = divmod(cursor, len(self._items))
        found = self._permutations.get(epoch)
        if found is None:
            digest = digest_bytes(f"{self.settings.digest}:epoch:{epoch}".encode("ascii"))
            generator = torch.Generator(device="cpu").manual_seed(
                int(digest[7:23], 16) & ((1 << 63) - 1)
            )
            found = tuple(torch.randperm(len(self._items), generator=generator).tolist())
            self._permutations[epoch] = found
        return found[offset]

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> QwenImagePreparedBatch:
        del generator
        if cursor < 0:
            raise ValueError("data cursor must be non-negative")
        batch_size = self._shapes[0][0]
        selected = [self._item_index(cursor * batch_size + offset) for offset in range(batch_size)]
        return QwenImagePreparedBatch(
            torch.stack([self._latents[index] for index in selected]).to(device),
            torch.stack([self._contexts[index] for index in selected]).to(device),
            torch.stack([self._attention_masks[index] for index in selected]).to(device),
        )


class Ideogram4DatasetSource:
    """Serve role-bound Ideogram 4 batches from verified encoded data."""

    def __init__(
        self,
        settings: Ideogram4DatasetSettings,
        vae_factory: Callable[[], AutoencoderKL],
        text_encoder_factory: Callable[[], Ideogram4TextEncoder] | None,
        *,
        latent_shape: tuple[int, int, int, int],
        context_shape: tuple[int, int, int] | None,
        attention_mask_shape: tuple[int, int] | None,
        vae_identity: object,
        text_encoder_identity: object | None,
        device: torch.device,
    ) -> None:
        current = inspect_ideogram4_dataset(
            Path(settings.root),
            settings.resolution,
            settings.context_tokens,
            role=settings.role,
            vae_identity=vae_identity,
            text_encoder_identity=text_encoder_identity,
        )
        if current.digest != settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {settings.digest}, got {current.digest}"
            )
        if current.errors:
            raise ValueError("dataset validation failed: " + "; ".join(current.errors))
        if (text_encoder_factory is not None) != (settings.role == "conditional"):
            raise ValueError("Ideogram 4 text encoder factory does not match the model role")
        if (context_shape is None) != (attention_mask_shape is None):
            raise ValueError("Ideogram 4 context and mask geometry must be supplied together")
        if (context_shape is not None) != (settings.role == "conditional"):
            raise ValueError("Ideogram 4 conditioning geometry does not match the model role")
        self.settings = settings
        self._items = current.items
        self._latent_shape = latent_shape
        self._context_shape = context_shape
        self._attention_mask_shape = attention_mask_shape
        self._permutations: dict[int, tuple[int, ...]] = {}
        self._validate_prompts()
        cache = (
            None
            if settings.encoded_cache_root is None
            else Ideogram4EncodedDatasetCache(
                settings,
                latent_shape=latent_shape,
                context_shape=context_shape,
                attention_mask_shape=attention_mask_shape,
                device=device,
            )
        )
        cached = None if cache is None else cache.load()
        if cached is not None:
            self._use_cached(cached)
            return
        if cache is None:
            self._build(vae_factory, text_encoder_factory, device)
            return
        with cache.build_lock():
            cached = cache.load()
            if cached is not None:
                self._use_cached(cached)
                return
            self._build(vae_factory, text_encoder_factory, device)
            cache.write(self._cache_tensors())

    @staticmethod
    def _release(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def _validate_prompts(self) -> None:
        if self.settings.role == "unconditional":
            return
        for item in self._items:
            tokens = tokenize_ideogram4_prompt(self._caption(item))
            if len(tokens.ids) > IDEOGRAM4_TEXT_CONFIG.max_position_embeddings:
                raise ValueError(
                    "Ideogram 4 prompt exceeds"
                    f" {IDEOGRAM4_TEXT_CONFIG.max_position_embeddings}-token position limit"
                )
            if len(tokens.ids) > self.settings.context_tokens:
                raise ValueError(
                    "Ideogram 4 prompt exceeds configured"
                    f" {self.settings.context_tokens}-token context"
                )

    def _use_cached(self, tensors: EncodedIdeogram4DatasetTensors) -> None:
        self._latents = tuple(tensors.latents.unbind())
        self._contexts = None if tensors.context is None else tuple(tensors.context.unbind())
        self._attention_masks = (
            None if tensors.attention_mask is None else tuple(tensors.attention_mask.unbind())
        )
        self._validate()

    def _build(
        self,
        vae_factory: Callable[[], AutoencoderKL],
        text_encoder_factory: Callable[[], Ideogram4TextEncoder] | None,
        device: torch.device,
    ) -> None:
        vae: AutoencoderKL | None = None
        try:
            vae = vae_factory().requires_grad_(False).eval().to(device=device)
            with torch.no_grad():
                self._latents = tuple(
                    vae.encode(process_input(self._pixels(item).unsqueeze(0).to(device)))
                    .float()
                    .squeeze(0)
                    .cpu()
                    for item in self._items
                )
        finally:
            vae = None
            self._release(device)
        if text_encoder_factory is None:
            self._contexts = None
            self._attention_masks = None
            self._validate()
            return
        encoder: Ideogram4TextEncoder | None = None
        try:
            encoder = text_encoder_factory()
            contexts: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            with torch.no_grad():
                for item in self._items:
                    conditioning = encoder.encode(self._caption(item))
                    value = conditioning.embeddings.float().squeeze(0)
                    if value.shape[0] > self.settings.context_tokens:
                        raise ValueError(
                            "Ideogram 4 text encoder output exceeds configured context"
                        )
                    source_mask = conditioning.attention_mask
                    mask = (
                        torch.ones(value.shape[0], dtype=torch.float32, device=value.device)
                        if source_mask is None
                        else source_mask.float().squeeze(0)
                    )
                    missing = self.settings.context_tokens - value.shape[0]
                    contexts.append(functional.pad(value, (0, 0, 0, missing)).cpu())
                    masks.append(functional.pad(mask, (0, missing)).cpu())
            self._contexts = tuple(contexts)
            self._attention_masks = tuple(masks)
        finally:
            encoder = None
            self._release(device)
        self._validate()

    def _pixels(self, item: ImageCaptionItem) -> torch.Tensor:
        data = item.image_path.read_bytes()
        if _digest_bytes(data) != item.image_digest:
            raise ValueError(f"dataset image changed after validation: {item.relative_image}")
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            width, height = image.size
            target_height, target_width = self.settings.resolution
            if width * target_height > height * target_width:
                crop_width = height * target_width // target_height
                left = (width - crop_width) // 2
                image = image.crop((left, 0, left + crop_width, height))
            elif width * target_height < height * target_width:
                crop_height = width * target_height // target_width
                top = (height - crop_height) // 2
                image = image.crop((0, top, width, top + crop_height))
            pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
            pixels = pixels.reshape(image.height, image.width, 3).permute(2, 0, 1)
        value = pixels.float().div_(255.0).unsqueeze(0)
        if tuple(value.shape[2:]) != self.settings.resolution:
            value = functional.interpolate(
                value,
                size=self.settings.resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return value.squeeze(0)

    @staticmethod
    def _caption(item: ImageCaptionItem) -> str:
        data = item.caption_path.read_bytes()
        if _digest_bytes(data) != item.caption_digest:
            raise ValueError(f"dataset caption changed after validation: {item.relative_caption}")
        value = data.decode("utf-8").strip()
        if not value:
            raise ValueError(f"dataset caption is empty: {item.relative_caption}")
        return value

    def _validate(self) -> None:
        count = len(self._items)
        if len(self._latents) != count or any(
            value.dtype != torch.float32 or tuple(value.shape) != self._latent_shape[1:]
            for value in self._latents
        ):
            raise ValueError("encoded Ideogram 4 latents have the wrong shape or dtype")
        if self.settings.role == "unconditional":
            if self._contexts is not None or self._attention_masks is not None:
                raise ValueError("unconditional Ideogram 4 data cannot carry text conditioning")
            return
        assert self._context_shape is not None and self._attention_mask_shape is not None
        if self._contexts is None or self._attention_masks is None:
            raise ValueError("conditional Ideogram 4 data requires context and attention masks")
        if len(self._contexts) != count or len(self._attention_masks) != count:
            raise ValueError("encoded Ideogram 4 dataset item counts do not match")
        if any(
            value.dtype != torch.float32 or tuple(value.shape) != self._context_shape[1:]
            for value in self._contexts
        ) or any(
            value.dtype != torch.float32 or tuple(value.shape) != self._attention_mask_shape[1:]
            for value in self._attention_masks
        ):
            raise ValueError("encoded Ideogram 4 conditioning has the wrong shape or dtype")
        if any(
            not bool(torch.all((mask == 0.0) | (mask == 1.0)).item())
            for mask in self._attention_masks
        ):
            raise ValueError("encoded Ideogram 4 attention mask must be binary")

    def _cache_tensors(self) -> EncodedIdeogram4DatasetTensors:
        return EncodedIdeogram4DatasetTensors(
            torch.stack(self._latents),
            None if self._contexts is None else torch.stack(self._contexts),
            None if self._attention_masks is None else torch.stack(self._attention_masks),
        )

    def _item_index(self, cursor: int) -> int:
        epoch, offset = divmod(cursor, len(self._items))
        found = self._permutations.get(epoch)
        if found is None:
            digest = digest_bytes(f"{self.settings.digest}:epoch:{epoch}".encode("ascii"))
            generator = torch.Generator(device="cpu").manual_seed(
                int(digest[7:23], 16) & ((1 << 63) - 1)
            )
            found = tuple(torch.randperm(len(self._items), generator=generator).tolist())
            self._permutations[epoch] = found
        return found[offset]

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> Ideogram4PreparedBatch:
        del generator
        if cursor < 0:
            raise ValueError("data cursor must be non-negative")
        batch_size = self._latent_shape[0]
        selected = [self._item_index(cursor * batch_size + offset) for offset in range(batch_size)]
        return Ideogram4PreparedBatch(
            torch.stack([self._latents[index] for index in selected]).to(device),
            (
                None
                if self._contexts is None
                else torch.stack([self._contexts[index] for index in selected]).to(device)
            ),
            (
                None
                if self._attention_masks is None
                else torch.stack([self._attention_masks[index] for index in selected]).to(device)
            ),
        )


def _load_minimax_h3_video_vae(
    state: MiniMaxH3ComponentStateSource, device: torch.device
) -> MiniMaxH3VideoVaeRuntime:
    path = Path(state.path)
    loaded = load_minimax_h3_component(
        path,
        asset=fixed_asset_ref(path, state.digest, state.size),
        expected_role="video-vae",
        expected_identity=state.identity,
        compute_dtype=torch.float16,
    )
    if not isinstance(loaded.module, MiniMaxH3VideoVAE):
        raise TypeError("loaded MiniMax H3 video VAE has the wrong module type")
    module = loaded.module.requires_grad_(False).eval().to(device=device)
    return MiniMaxH3VideoVaeRuntime(
        module,
        runtime_identity=loaded.runtime_identity,
        compute_dtype=torch.float16,
    )


def _load_minimax_h3_audio_vae(
    state: MiniMaxH3ComponentStateSource, device: torch.device
) -> MiniMaxH3AudioVaeRuntime:
    path = Path(state.path)
    loaded = load_minimax_h3_component(
        path,
        asset=fixed_asset_ref(path, state.digest, state.size),
        expected_role="audio-vae",
        expected_identity=state.identity,
        compute_dtype=torch.float16,
    )
    if not isinstance(loaded.module, MiniMaxH3AudioVAE):
        raise TypeError("loaded MiniMax H3 audio VAE has the wrong module type")
    module = loaded.module.requires_grad_(False).eval().to(device=device)
    return MiniMaxH3AudioVaeRuntime(
        module,
        runtime_identity=loaded.runtime_identity,
        compute_dtype=torch.float16,
    )


def _load_minimax_h3_conditioner(
    state: MiniMaxH3ComponentStateSource, device: torch.device
) -> MiniMaxH3ConditionerRuntime:
    path = Path(state.path)
    loaded = load_minimax_h3_component(
        path,
        asset=fixed_asset_ref(path, state.digest, state.size),
        expected_role="qwen3vl-32b-conditioner",
        expected_identity=state.identity,
        compute_dtype=torch.bfloat16,
    )
    if not isinstance(loaded.module, MiniMaxH3ConditionerModel):
        raise TypeError("loaded MiniMax H3 conditioner has the wrong module type")
    module = loaded.module.requires_grad_(False).eval().to(device=device)
    return MiniMaxH3ConditionerRuntime(module, runtime_identity=loaded.runtime_identity)


def default_minimax_h3_dataset_source(
    settings: MiniMaxH3DatasetSettings,
    *,
    video_latent_shape: tuple[int, int, int, int, int],
    audio_latent_shape: tuple[int, int, int, int],
    conditioner_shape: tuple[int, int, int],
    device: torch.device,
) -> MiniMaxH3DatasetSource:
    """Precompute H3 media and captions with pinned native components."""
    return MiniMaxH3DatasetSource(
        settings,
        lambda: _load_minimax_h3_video_vae(settings.video_vae_state, device),
        lambda: _load_minimax_h3_audio_vae(settings.audio_vae_state, device),
        lambda: _load_minimax_h3_conditioner(settings.conditioner_state, device),
        video_latent_shape=video_latent_shape,
        audio_latent_shape=audio_latent_shape,
        conditioner_shape=conditioner_shape,
        device=device,
    )


class MiniMaxH3DatasetSource:
    """Serve deterministic H3 T2VA batches from validated media items."""

    def __init__(
        self,
        settings: MiniMaxH3DatasetSettings,
        video_vae_factory: Callable[[], MiniMaxH3VideoVaeRuntime],
        audio_vae_factory: Callable[[], MiniMaxH3AudioVaeRuntime],
        conditioner_factory: Callable[[], MiniMaxH3ConditionerRuntime],
        *,
        video_latent_shape: tuple[int, int, int, int, int],
        audio_latent_shape: tuple[int, int, int, int],
        conditioner_shape: tuple[int, int, int],
        device: torch.device,
    ) -> None:
        current = inspect_minimax_h3_dataset(
            Path(settings.root),
            settings.resolution,
            settings.frame_count,
            settings.video_vae_state,
            settings.audio_vae_state,
            settings.conditioner_state,
            validate_media=False,
        )
        if current.digest != settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {settings.digest}, got {current.digest}"
            )
        validation_errors = settings.inspection.errors or current.errors
        if validation_errors:
            raise ValueError("dataset validation failed: " + "; ".join(validation_errors))
        self.settings = settings
        self._items = current.items
        self._video_latent_shape = video_latent_shape
        self._audio_latent_shape = audio_latent_shape
        self._conditioner_shape = conditioner_shape
        self._permutations: dict[int, tuple[int, ...]] = {}
        cache = (
            None
            if settings.encoded_cache_root is None
            else MiniMaxH3EncodedDatasetCache(settings, device=device)
        )
        cached = None if cache is None else cache.load()
        if cached is not None:
            self._use_cached(cached)
            return
        if cache is None:
            self._build(video_vae_factory, audio_vae_factory, conditioner_factory, device)
            return
        with cache.build_lock():
            cached = cache.load()
            if cached is not None:
                self._use_cached(cached)
                return
            self._build(video_vae_factory, audio_vae_factory, conditioner_factory, device)
            cache.write(self._cache_tensors())

    def _use_cached(self, cached: EncodedMiniMaxH3DatasetTensors) -> None:
        self._video_latents = tuple(cached.video_latents.unbind(0))
        self._audio_latents = tuple(cached.audio_latents.unbind(0))
        self._conditioner_embeddings = tuple(cached.conditioner_embeddings.unbind(0))
        self._conditioning_text_token_tags = tuple(cached.conditioning_text_token_tags.unbind(0))
        self._validate_stores()

    def _build(
        self,
        video_vae_factory: Callable[[], MiniMaxH3VideoVaeRuntime],
        audio_vae_factory: Callable[[], MiniMaxH3AudioVaeRuntime],
        conditioner_factory: Callable[[], MiniMaxH3ConditionerRuntime],
        device: torch.device,
    ) -> None:
        current = inspect_minimax_h3_dataset(
            Path(self.settings.root),
            self.settings.resolution,
            self.settings.frame_count,
            self.settings.video_vae_state,
            self.settings.audio_vae_state,
            self.settings.conditioner_state,
        )
        if current.digest != self.settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {self.settings.digest}, got {current.digest}"
            )
        if current.errors:
            raise ValueError("dataset validation failed: " + "; ".join(current.errors))
        self._items = current.items
        try:
            self._video_latents = self._precompute_video(video_vae_factory, device)
        finally:
            self._release_cuda_cache(device)
        try:
            self._audio_latents = self._precompute_audio(audio_vae_factory, device)
        finally:
            self._release_cuda_cache(device)
        try:
            (
                self._conditioner_embeddings,
                self._conditioning_text_token_tags,
            ) = self._precompute_conditioning(conditioner_factory, device)
        finally:
            self._release_cuda_cache(device)
        self._validate_stores()

    def _cache_tensors(self) -> EncodedMiniMaxH3DatasetTensors:
        return EncodedMiniMaxH3DatasetTensors(
            video_latents=torch.stack(self._video_latents),
            audio_latents=torch.stack(self._audio_latents),
            conditioner_embeddings=torch.stack(self._conditioner_embeddings),
            conditioning_text_token_tags=torch.stack(self._conditioning_text_token_tags),
        )

    @staticmethod
    def _release_cuda_cache(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def _validate_stores(self) -> None:
        expected_count = len(self._items)
        stores = (
            self._video_latents,
            self._audio_latents,
            self._conditioner_embeddings,
            self._conditioning_text_token_tags,
        )
        if any(len(store) != expected_count for store in stores):
            raise ValueError("encoded H3 dataset item counts do not match")
        for video, audio, context, tags in zip(*stores, strict=True):
            if video.dtype != torch.float32 or tuple(video.shape) != self._video_latent_shape[1:]:
                raise ValueError("encoded H3 video latent has the wrong shape or dtype")
            if audio.dtype != torch.float32 or tuple(audio.shape) != self._audio_latent_shape[1:]:
                raise ValueError("encoded H3 audio latent has the wrong shape or dtype")
            if (
                context.dtype != torch.float32
                or tuple(context.shape) != self._conditioner_shape[1:]
            ):
                raise ValueError("encoded H3 conditioner embedding has the wrong shape or dtype")
            if tags.dtype != torch.int64 or tuple(tags.shape) != (self._conditioner_shape[1],):
                raise ValueError("encoded H3 conditioning tags have the wrong shape or dtype")

    def _permutation(self, epoch: int) -> tuple[int, ...]:
        found = self._permutations.get(epoch)
        if found is not None:
            return found
        preimage = f"{self.settings.digest}:epoch:{epoch}".encode("ascii")
        seed = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        found = tuple(torch.randperm(len(self._items), generator=generator).tolist())
        self._permutations[epoch] = found
        return found

    def _item_index(self, cursor: int) -> int:
        epoch, offset = divmod(cursor, len(self._items))
        return self._permutation(epoch)[offset]

    @staticmethod
    def _read_caption(item: MiniMaxH3DatasetItem) -> str:
        data = item.caption_path.read_bytes()
        if _digest_bytes(data) != item.caption_digest:
            raise ValueError(f"dataset caption changed after validation: {item.relative_caption}")
        caption = data.decode("utf-8").strip()
        if not caption:
            raise ValueError(f"dataset caption is empty: {item.relative_caption}")
        return caption

    def _read_video(self, item: MiniMaxH3DatasetItem) -> torch.Tensor:
        frames: list[torch.Tensor] = []
        if item.video_kind == "container":
            data = item.video_path.read_bytes()
            if _digest_bytes(data) != item.container_digest:
                raise ValueError(
                    f"dataset container changed after validation: {item.relative_container}"
                )
            payloads = decode_video_rgb24(
                data,
                resolution=self.settings.resolution,
                frame_count=self.settings.frame_count,
            )
            height, width = self.settings.resolution
            for payload in payloads:
                pixels = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                frames.append(pixels.reshape(height, width, 3).to(dtype=torch.float32).div_(255.0))
            return torch.stack(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous()

        root = Path(self.settings.root)
        for relative, expected_digest in zip(item.relative_frames, item.frame_digests, strict=True):
            data = (root / relative).read_bytes()
            if _digest_bytes(data) != expected_digest:
                raise ValueError(f"dataset frame changed after validation: {relative}")
            with Image.open(io.BytesIO(data)) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                pixels = pixels.reshape(image.height, image.width, 3)
            # [0, 1] comfy image range; encode_video owns the [-1, 1] transform.
            frames.append(pixels.to(dtype=torch.float32).div_(255.0))
        return torch.stack(frames).permute(3, 0, 1, 2).unsqueeze(0).contiguous()

    @staticmethod
    def _pcm_waveform(
        payload: bytes, *, sample_width: int, channels: int, samples: int
    ) -> torch.Tensor:
        """Convert little-endian PCM bytes to the H3 float32 waveform layout."""
        packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        packed = packed.reshape(-1, sample_width).to(dtype=torch.int64)
        values = torch.zeros((packed.shape[0],), dtype=torch.int64)
        for byte_index in range(sample_width):
            values.bitwise_or_(packed[:, byte_index] << (8 * byte_index))
        bits = sample_width * 8
        if sample_width == 1:
            values.sub_(1 << (bits - 1))
        else:
            sign = 1 << (bits - 1)
            values = values.bitwise_xor(sign).sub_(sign)
        waveform = values.to(dtype=torch.float32).div_(float(1 << (bits - 1)))
        return waveform.reshape(samples, channels).transpose(0, 1).unsqueeze(0).contiguous()

    def _read_audio(self, item: MiniMaxH3DatasetItem) -> torch.Tensor:
        data = item.audio_path.read_bytes()
        if _digest_bytes(data) != item.audio_digest:
            raise ValueError(f"dataset audio changed after validation: {item.relative_audio}")
        if item.audio_kind == "container":
            samples = minimax_h3_audio_sample_count(self.settings.frame_count)
            payload = decode_audio_s16_stereo_32k(data, sample_count=samples)
            if payload is None:
                raise ValueError(f"dataset container has no audio stream: {item.relative_audio}")
            return self._pcm_waveform(payload, sample_width=2, channels=2, samples=samples)
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            samples = audio.getnframes()
            payload = audio.readframes(samples)
        return self._pcm_waveform(
            payload,
            sample_width=sample_width,
            channels=channels,
            samples=samples,
        )

    def _precompute_video(
        self,
        factory: Callable[[], MiniMaxH3VideoVaeRuntime],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        runtime = factory()
        if runtime.runtime_identity != self.settings.video_vae_state.identity:
            raise ValueError("H3 video VAE runtime identity does not match its dataset pin")
        values: list[torch.Tensor] = []
        with torch.no_grad():
            for item in self._items:
                encoded = runtime.encode_video(self._read_video(item).to(device=device))
                values.append(encoded.detach().float().squeeze(0).cpu())
        del runtime
        return tuple(values)

    def _precompute_audio(
        self,
        factory: Callable[[], MiniMaxH3AudioVaeRuntime],
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        runtime = factory()
        if runtime.runtime_identity != self.settings.audio_vae_state.identity:
            raise ValueError("H3 audio VAE runtime identity does not match its dataset pin")
        values: list[torch.Tensor] = []
        with torch.no_grad():
            for item in self._items:
                waveform = self._read_audio(item).to(device=device)
                encoded = runtime.encode_audio(
                    MiniMaxH3AudioContent(
                        waveform,
                        MINIMAX_H3_CONFIG.audio_sample_rate_hz,
                    )
                )
                values.append(encoded.detach().float().squeeze(0).cpu())
        del runtime
        return tuple(values)

    def _precompute_conditioning(
        self,
        factory: Callable[[], MiniMaxH3ConditionerRuntime],
        device: torch.device,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        runtime = factory()
        if runtime.runtime_identity != self.settings.conditioner_state.identity:
            raise ValueError("H3 conditioner runtime identity does not match its dataset pin")
        contexts: list[torch.Tensor] = []
        tags: list[torch.Tensor] = []
        with torch.no_grad():
            for item, video, audio in zip(
                self._items, self._video_latents, self._audio_latents, strict=True
            ):
                target = MultiStreamLatent(
                    (
                        LatentStream("video", video.unsqueeze(0).to(device=device)),
                        LatentStream("audio", audio.unsqueeze(0).to(device=device)),
                    )
                )
                prepared = runtime.condition(
                    MiniMaxH3T2VARequest(self._read_caption(item)),
                    target=target,
                    frame_count=self.settings.frame_count,
                    payloads={},
                    cancelled=lambda: False,
                )
                token_tags = prepared.dit.text_token_tags
                if prepared.dit.keyframes or prepared.dit.references or token_tags is None:
                    raise ValueError("H3 T2VA conditioner returned an invalid training carrier")
                contexts.append(prepared.context.detach().float().squeeze(0).cpu())
                tags.append(token_tags.detach().to(dtype=torch.int64).squeeze(0).cpu())
        del runtime
        return tuple(contexts), tuple(tags)

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxH3PreparedBatch:
        del generator
        if cursor < 0:
            raise ValueError("data cursor must be non-negative")
        index = self._item_index(cursor)
        tags = self._conditioning_text_token_tags[index].unsqueeze(0).to(device=device)
        return MiniMaxH3PreparedBatch(
            video_latents=self._video_latents[index].unsqueeze(0).to(device=device),
            audio_latents=self._audio_latents[index].unsqueeze(0).to(device=device),
            conditioner_embeddings=self._conditioner_embeddings[index]
            .unsqueeze(0)
            .to(device=device),
            conditioning=MiniMaxH3DiTConditioning(text_token_tags=tags),
        )


class WanDatasetSource:
    """Serve deterministic Wan batches from encoded videos and captions."""

    def __init__(
        self,
        settings: WanDatasetSettings,
        vae_factory: Callable[[], WanVAE | Wan22VAE],
        text_factory: Callable[[], Wan21TextRuntime],
        *,
        latent_shape: tuple[int, int, int, int, int],
        context_shape: tuple[int, int, int],
        vae_identity: object,
        umt5xxl_identity: object,
        device: torch.device,
    ) -> None:
        current = inspect_wan_dataset(
            Path(settings.root),
            settings.resolution,
            settings.frame_count,
            vae_identity=vae_identity,
            umt5xxl_identity=umt5xxl_identity,
            vae_contract=settings.vae_contract,
            conditioning_contract=settings.conditioning_contract,
            validate_media=False,
        )
        if current.digest != settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {settings.digest}, got {current.digest}"
            )
        if current.errors:
            raise ValueError("dataset validation failed: " + "; ".join(current.errors))
        self.settings = settings
        self._items = current.items
        self._latent_shape = latent_shape
        self._context_shape = context_shape
        self._permutations: dict[int, tuple[int, ...]] = {}
        cache = (
            None
            if settings.encoded_cache_root is None
            else WanEncodedDatasetCache(
                settings,
                latent_shape=latent_shape,
                context_shape=context_shape,
                device=device,
            )
        )
        cached = None if cache is None else cache.load()
        if cached is not None:
            self._use_cached(cached)
            return
        if cache is None:
            self._build(vae_factory, text_factory, device)
            return
        with cache.build_lock():
            cached = cache.load()
            if cached is not None:
                self._use_cached(cached)
                return
            self._build(vae_factory, text_factory, device)
            cache.write(self._cache_tensors())

    def _use_cached(self, cached: EncodedWanDatasetTensors) -> None:
        self._latents = tuple(cached.latents.unbind(0))
        self._contexts = tuple(cached.context.unbind(0))
        self._i2v_conditioning = (
            None if cached.i2v_conditioning is None else tuple(cached.i2v_conditioning.unbind(0))
        )
        self._validate_stores()

    def _build(
        self,
        vae_factory: Callable[[], WanVAE | Wan22VAE],
        text_factory: Callable[[], Wan21TextRuntime],
        device: torch.device,
    ) -> None:
        try:
            vae = vae_factory().requires_grad_(False).eval().to(device=device)
            values: list[torch.Tensor] = []
            conditioning_values: list[torch.Tensor] = []
            with torch.no_grad():
                for item in self._items:
                    content = self._read_video(item).to(
                        device=device,
                        dtype=WAN21_ENCODING_DTYPE,
                    )
                    encoded = vae.process_in(vae.encode(content))
                    values.append(encoded.detach().float().squeeze(0).cpu())
                    if self.settings.conditioning_contract == "first-frame-i2v":
                        conditioning_video = torch.zeros_like(content)
                        conditioning_video[:, :, :1] = content[:, :, :1]
                        conditioning_latent = vae.process_in(vae.encode(conditioning_video))
                        mask = conditioning_latent.new_zeros(
                            conditioning_latent.shape[0],
                            4,
                            *conditioning_latent.shape[2:],
                        )
                        mask[:, :, :1] = 1.0
                        conditioning_values.append(
                            torch.cat((mask, conditioning_latent), dim=1)
                            .detach()
                            .float()
                            .squeeze(0)
                            .cpu()
                        )
            self._latents = tuple(values)
            self._i2v_conditioning = (
                tuple(conditioning_values)
                if self.settings.conditioning_contract == "first-frame-i2v"
                else None
            )
            del vae
        finally:
            self._release_cuda_cache(device)
        try:
            runtime = text_factory()
            contexts: list[torch.Tensor] = []
            with torch.no_grad():
                for item in self._items:
                    encoded = runtime.encode_text(self._read_caption(item)).embeddings
                    contexts.append(encoded.detach().float().squeeze(0).cpu())
            self._contexts = tuple(contexts)
            del runtime
        finally:
            self._release_cuda_cache(device)
        self._validate_stores()

    @staticmethod
    def _release_cuda_cache(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def _cache_tensors(self) -> EncodedWanDatasetTensors:
        return EncodedWanDatasetTensors(
            latents=torch.stack(self._latents),
            context=torch.stack(self._contexts),
            i2v_conditioning=(
                None if self._i2v_conditioning is None else torch.stack(self._i2v_conditioning)
            ),
        )

    def _validate_stores(self) -> None:
        if len(self._latents) != len(self._items) or len(self._contexts) != len(self._items):
            raise ValueError("encoded Wan dataset item counts do not match")
        if (self._i2v_conditioning is None) != (self.settings.conditioning_contract == "none"):
            raise ValueError("encoded Wan I2V conditioning does not match the dataset contract")
        if self._i2v_conditioning is not None and len(self._i2v_conditioning) != len(self._items):
            raise ValueError("encoded Wan I2V conditioning item count does not match")
        for latent, context in zip(self._latents, self._contexts, strict=True):
            if latent.dtype != torch.float32 or tuple(latent.shape) != self._latent_shape[1:]:
                raise ValueError("encoded Wan latent has the wrong shape or dtype")
            if context.dtype != torch.float32 or tuple(context.shape) != self._context_shape[1:]:
                raise ValueError("encoded Wan context has the wrong shape or dtype")
        if self._i2v_conditioning is not None:
            expected = (20, *self._latent_shape[2:])
            if any(
                value.dtype != torch.float32 or tuple(value.shape) != expected
                for value in self._i2v_conditioning
            ):
                raise ValueError("encoded Wan I2V conditioning has the wrong shape or dtype")

    @staticmethod
    def _read_caption(item: WanDatasetItem) -> str:
        data = item.caption_path.read_bytes()
        if _digest_bytes(data) != item.caption_digest:
            raise ValueError(f"dataset caption changed after validation: {item.relative_caption}")
        caption = data.decode("utf-8").strip()
        if not caption:
            raise ValueError(f"dataset caption is empty: {item.relative_caption}")
        return caption

    def _read_video(self, item: WanDatasetItem) -> torch.Tensor:
        data = item.video_path.read_bytes()
        if _digest_bytes(data) != item.video_digest:
            raise ValueError(f"dataset video changed after validation: {item.relative_video}")
        payloads = decode_video_rgb24(
            data,
            resolution=self.settings.resolution,
            frame_count=self.settings.frame_count,
        )
        height, width = self.settings.resolution
        frames = [
            torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(height, width, 3)
            for payload in payloads
        ]
        return (
            torch.stack(frames)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(dtype=torch.float32)
            .div_(127.5)
            .sub_(1.0)
            .contiguous()
        )

    def _permutation(self, epoch: int) -> tuple[int, ...]:
        found = self._permutations.get(epoch)
        if found is not None:
            return found
        digest = digest_bytes(f"{self.settings.digest}:epoch:{epoch}".encode("ascii"))
        seed = int(digest[7:23], 16) & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        found = tuple(torch.randperm(len(self._items), generator=generator).tolist())
        self._permutations[epoch] = found
        return found

    def _item_index(self, cursor: int) -> int:
        epoch, offset = divmod(cursor, len(self._items))
        return self._permutation(epoch)[offset]

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> WanPreparedBatch:
        del generator
        if cursor < 0:
            raise ValueError("data cursor must be non-negative")
        selected = [
            self._item_index(cursor * self._latent_shape[0] + offset)
            for offset in range(self._latent_shape[0])
        ]
        return WanPreparedBatch(
            latents=torch.stack([self._latents[index] for index in selected]).to(device=device),
            context=torch.stack([self._contexts[index] for index in selected]).to(device=device),
            i2v_conditioning=(
                None
                if self._i2v_conditioning is None
                else torch.stack([self._i2v_conditioning[index] for index in selected]).to(
                    device=device
                )
            ),
        )
