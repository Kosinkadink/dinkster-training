"""Validated configuration for native SD, Flux, Wan, and MiniMax H3 LoRA training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import torch
from dinkster_inference import (
    MINIMAX_H3_CONFIG,
    MINIMAX_MUSIC3_CONFIG,
    SD15_UNET_CONFIG,
    SDXL_UNET_CONFIG,
    ExecutionComposition,
    MiniMaxH3DiTRole,
    UNetConfig,
    compose_execution,
)

from .dataset import (
    DatasetInspection,
    EncoderStateSource,
    Flux2DatasetSettings,
    FluxDatasetSettings,
    Ideogram4DatasetSettings,
    ImageCaptionDatasetSettings,
    MiniMaxH3ComponentStateSource,
    MiniMaxH3DatasetSettings,
    QwenImageDatasetSettings,
    TrainingFamily,
    WanConditioningContract,
    WanDatasetSettings,
    WanVAEContract,
    inspect_flux2_dataset,
    inspect_flux_dataset,
    inspect_ideogram4_dataset,
    inspect_image_caption_dataset,
    inspect_minimax_h3_dataset,
    inspect_qwen_image_dataset,
    inspect_wan_dataset,
    minimax_h3_audio_latent_shape,
    minimax_h3_video_latent_shape,
)
from .distributed import (
    DistributedSettings,
    FileRendezvousSettings,
    TcpRendezvousSettings,
)
from .minimax_music3_training import (
    MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION,
    MINIMAX_MUSIC3_FLOW_OBJECTIVE,
    MiniMaxMusic3ArtifactPin,
    MiniMaxMusic3DatasetSettings,
    inspect_minimax_music3_dataset,
    minimax_music3_encoded_cache_state,
)

OptimizerName = Literal["adamw", "factored-adamw"]
BaseDtypeName = Literal["float32", "bfloat16"]
MiniMaxH3Int8BaseForward = Literal["dequantize", "fused"]
RngPolicyName = Literal["sequential", "counter"]
CheckpointingModeName = Literal["blockReentrant", "wholeModel"]
WanBaseDtypeName = Literal["float16", "bfloat16"]
WanLoraTargetName = Literal["attention.qkvo", "ffn.projections"]
WanVariantName = Literal[
    "wan21-t2v",
    "wan22-ti2v-5b",
    "wan22-t2v-14b",
    "wan22-i2v-14b",
]
WanExpertName = Literal["high-noise", "low-noise"]
FluxBaseDtypeName = Literal["float16", "bfloat16"]
FluxLoraTargetName = Literal["attention.qkv_proj", "mlp.projections"]
FluxVariantName = Literal["flux1-dev", "flux1-schnell"]
Flux2VariantName = Literal["flux2-dev", "flux2-klein-9b", "flux2-klein-4b"]
QwenImageBaseDtypeName = Literal["bfloat16", "float32"]
QwenImageLoraTargetName = Literal["attention.qkvo", "mlp.projections"]
Ideogram4RoleName = Literal["conditional", "unconditional"]
Ideogram4StorageName = Literal["fp8", "int8-convrot"]
Ideogram4LoraTargetName = Literal["attention.qkvo", "mlp.projections"]
MiniMaxMusic3BaseDtypeName = Literal["float16", "bfloat16", "float32"]

_WAN_VARIANTS = (
    "wan21-t2v",
    "wan22-ti2v-5b",
    "wan22-t2v-14b",
    "wan22-i2v-14b",
)
_FLUX_VARIANTS = ("flux1-dev", "flux1-schnell")
_FLUX2_VARIANTS = ("flux2-dev", "flux2-klein-9b", "flux2-klein-4b")
_FLUX2_CONTEXT_WIDTHS = {
    "flux2-dev": 15360,
    "flux2-klein-9b": 12288,
    "flux2-klein-4b": 7680,
}
_QWEN_IMAGE_VARIANTS = ("qwen-image",)
_IDEOGRAM4_DIFFUSION_DIGESTS = {
    (
        "conditional",
        "fp8",
    ): "blake3:dadac522cf4fd911d25c70401df4e1233805887481e21be0e19839458fb69eba",
    (
        "unconditional",
        "fp8",
    ): "blake3:1e26b2ecf7cb7ab57495faff8534875be41a44d6fc2ffe37f934dd65a460958d",
    (
        "conditional",
        "int8-convrot",
    ): "blake3:e3cf071faafcf04192a66fa0324948589e60325b54a7935206f4e30b8ef257ce",
    (
        "unconditional",
        "int8-convrot",
    ): "blake3:eb50781b817ef134e114ffa55def137206c56ce32846f12e43397f0ac2a8e909",
}
_IDEOGRAM4_TEXT_DIGEST = "blake3:b82a81d829c1d8db687c8e9f79fc07e7211f5ac59ed05672423c1cedc8db0924"
_IDEOGRAM4_VAE_DIGEST = "blake3:fcb1d172993424c66d325d139863ccbaadf64a920073b2d005d73a31fa5a851d"
_WAN22_EXPERT_TIMESTEP_RANGES = {
    "wan22-t2v-14b": {
        "high-noise": (1000, 875),
        "low-noise": (875, 0),
    },
    "wan22-i2v-14b": {
        "high-noise": (1000, 900),
        "low-noise": (900, 0),
    },
}

# Runtime-only settings must not change what training computes. They may only
# change how or where equivalent work is accelerated or stored.
_COMMON_RUNTIME_ONLY_FIELDS = (
    "checkpointInterval",
    "loraExportInterval",
    "syncDigestInterval",
)
_DATASET_RUNTIME_ONLY_FIELDS = ("encodedCacheRoot",)
_H3_RUNTIME_ONLY_FIELDS = ("hostLayerPagingFraction",)
_DISTRIBUTED_RUNTIME_ONLY_FIELDS = ("backend", "rendezvous")


class TrainingConfigError(ValueError):
    """The serialized training configuration is invalid."""


@dataclass(frozen=True)
class WanArtifactSource:
    """One immutable training artifact selected from local storage."""

    path: str
    digest: str
    size: int

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "digest": self.digest, "size": self.size}


TrainingArtifactSource = WanArtifactSource


@dataclass(frozen=True)
class NativeTrainingArtifactSource:
    """One immutable artifact and its planned native runtime identity."""

    path: str
    digest: str
    size: int
    identity: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "digest": self.digest,
            "size": self.size,
            "identity": self.identity,
        }


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TrainingConfigError(f"{name} must be an object with string keys")
    return cast("dict[str, object]", value)


def _known(mapping: dict[str, object], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise TrainingConfigError(f"{name} has unknown fields: {', '.join(unknown)}")


def _int(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TrainingConfigError(f"{name} must be an integer >= {minimum}")
    return value


def _distributed_settings(value: object) -> DistributedSettings:
    raw = _object(value, "distributed")
    fields = {"worldSize", "backend", "rendezvous"}
    _known(raw, fields, "distributed")
    missing = sorted(fields - set(raw))
    if missing:
        raise TrainingConfigError(f"distributed is missing fields: {', '.join(missing)}")

    world_size = _int(raw["worldSize"], "distributed.worldSize", minimum=1)
    backend_value = raw["backend"]
    if backend_value not in ("gloo", "nccl"):
        raise TrainingConfigError("distributed.backend must be 'gloo' or 'nccl'")

    rendezvous = _object(raw["rendezvous"], "distributed.rendezvous")
    method = rendezvous.get("method")
    if method == "tcp":
        _known(rendezvous, {"method", "host", "port"}, "distributed.rendezvous")
        missing = sorted({"host", "port"} - set(rendezvous))
        if missing:
            raise TrainingConfigError(
                f"distributed.rendezvous is missing fields: {', '.join(missing)}"
            )
        host = rendezvous["host"]
        if not isinstance(host, str) or not host:
            raise TrainingConfigError("distributed.rendezvous.host must be a non-empty string")
        port = _int(rendezvous["port"], "distributed.rendezvous.port", minimum=1)
        if port > 65535:
            raise TrainingConfigError("distributed.rendezvous.port must be <= 65535")
        parsed_rendezvous = TcpRendezvousSettings(host=host, port=port)
    elif method == "file":
        _known(rendezvous, {"method", "path"}, "distributed.rendezvous")
        if "path" not in rendezvous:
            raise TrainingConfigError("distributed.rendezvous is missing fields: path")
        path = rendezvous["path"]
        if not isinstance(path, str) or not path:
            raise TrainingConfigError("distributed.rendezvous.path must be a non-empty string")
        parsed_rendezvous = FileRendezvousSettings(path=str(Path(path).expanduser().resolve()))
    else:
        raise TrainingConfigError("distributed.rendezvous.method must be 'tcp' or 'file'")

    return DistributedSettings(
        world_size=world_size,
        backend=backend_value,
        rendezvous=parsed_rendezvous,
    )


def _float(
    value: object, name: str, *, minimum: float = 0.0, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (maximum is not None and result > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        raise TrainingConfigError(f"{name} must be finite, >= {minimum}{suffix}")
    return result


def _normalized_float(
    value: object, name: str, *, minimum: float = 0.0, maximum: float | None = None
) -> float:
    if type(value) is not float:
        raise TrainingConfigError(f"{name} must be a normalized float")
    return _float(value, name, minimum=minimum, maximum=maximum)


def _shape(value: object, name: str, rank: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != rank:
        raise TrainingConfigError(f"{name} must be a {rank}-item integer array")
    return tuple(_int(item, f"{name}[{index}]", minimum=1) for index, item in enumerate(value))


def _tuple_int(mapping: dict[str, object], key: str) -> tuple[int, ...]:
    value = mapping[key]
    if not isinstance(value, list) or not value:
        raise TrainingConfigError(f"unet.{key} must be a non-empty integer array")
    return tuple(_int(item, f"unet.{key}[{index}]", minimum=0) for index, item in enumerate(value))


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("blake3:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise TrainingConfigError(f"{name} must be 'blake3:' plus 64 lowercase hex characters")
    return value


def _native_identity(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TrainingConfigError(f"{name} must be a native runtime identity")
    parts = value.split(":")
    if (
        len(parts) != 3
        or parts[0] != "native"
        or parts[1] != MINIMAX_H3_CONFIG.family_id
        or len(parts[2]) != 64
        or any(character not in "0123456789abcdef" for character in parts[2])
    ):
        raise TrainingConfigError(
            f"{name} must be a three-part {MINIMAX_H3_CONFIG.family_id!r} native identity"
        )
    return value


def _family_native_identity(value: object, name: str, family: str) -> str:
    if not isinstance(value, str):
        raise TrainingConfigError(f"{name} must be a native runtime identity")
    parts = value.split(":")
    if (
        len(parts) != 3
        or parts[0] != "native"
        or parts[1] != family
        or len(parts[2]) != 64
        or any(character not in "0123456789abcdef" for character in parts[2])
    ):
        raise TrainingConfigError(f"{name} must be a three-part {family!r} native identity")
    return value


def _native_artifact_source(
    value: object, name: str, *, family: str
) -> NativeTrainingArtifactSource:
    raw = _object(value, name)
    _known(raw, {"path", "digest", "size", "identity"}, name)
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise TrainingConfigError(f"{name}.path must be a non-empty string")
    return NativeTrainingArtifactSource(
        str(Path(path).expanduser().resolve()),
        _digest(raw.get("digest"), f"{name}.digest"),
        _int(raw.get("size"), f"{name}.size", minimum=1),
        _family_native_identity(raw.get("identity"), f"{name}.identity", family),
    )


def _music3_native_identity(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TrainingConfigError(f"{name} must be a native runtime identity")
    parts = value.split(":")
    if (
        len(parts) != 3
        or parts[0] != "native"
        or parts[1] != MINIMAX_MUSIC3_CONFIG.family_id
        or len(parts[2]) != 64
        or any(character not in "0123456789abcdef" for character in parts[2])
    ):
        raise TrainingConfigError(
            f"{name} must be a three-part {MINIMAX_MUSIC3_CONFIG.family_id!r} native identity"
        )
    return value


def _music3_artifact_pin(
    value: object,
    name: str,
    *,
    identity: bool,
) -> MiniMaxMusic3ArtifactPin:
    raw = _object(value, name)
    fields = {"path", "digest", "size", "identity"} if identity else {"path", "digest", "size"}
    _known(raw, fields, name)
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise TrainingConfigError(f"{name}.path must be a non-empty string")
    parsed_identity = (
        _music3_native_identity(raw.get("identity"), f"{name}.identity") if identity else None
    )
    return MiniMaxMusic3ArtifactPin(
        path=str(Path(path).expanduser().resolve()),
        digest=_digest(raw.get("digest"), f"{name}.digest"),
        size=_int(raw.get("size"), f"{name}.size", minimum=1),
        identity=parsed_identity,
    )


def _artifact_source(value: object, name: str) -> TrainingArtifactSource:
    raw = _object(value, name)
    _known(raw, {"path", "digest", "size"}, name)
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise TrainingConfigError(f"{name}.path must be a non-empty string")
    return TrainingArtifactSource(
        path=str(Path(path).expanduser().resolve()),
        digest=_digest(raw.get("digest"), f"{name}.digest"),
        size=_int(raw.get("size"), f"{name}.size"),
    )


def _encoder_source(value: object, name: str) -> EncoderStateSource:
    raw = _object(value, name)
    _known(raw, {"path", "digest", "prefix"}, name)
    path = raw.get("path")
    prefix = raw.get("prefix", "")
    if not isinstance(path, str) or not path:
        raise TrainingConfigError(f"{name}.path must be a non-empty string")
    if not isinstance(prefix, str):
        raise TrainingConfigError(f"{name}.prefix must be a string")
    return EncoderStateSource(
        path=str(Path(path).expanduser().resolve()),
        digest=_digest(raw.get("digest"), f"{name}.digest"),
        prefix=prefix,
    )


def _h3_component_source(value: object, name: str) -> MiniMaxH3ComponentStateSource:
    raw = _object(value, name)
    _known(raw, {"path", "digest", "size", "identity"}, name)
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        raise TrainingConfigError(f"{name}.path must be a non-empty string")
    return MiniMaxH3ComponentStateSource(
        path=str(Path(path).expanduser().resolve()),
        digest=_digest(raw.get("digest"), f"{name}.digest"),
        size=_int(raw.get("size"), f"{name}.size"),
        identity=_native_identity(raw.get("identity"), f"{name}.identity"),
    )


def _dataset_settings(value: object, family: TrainingFamily) -> ImageCaptionDatasetSettings:
    raw = _object(value, "dataset")
    common = {"type", "root", "resolution", "digest", "encodedCacheRoot"}
    family_fields = {"vaeState", "textEncoderState"} if family == "sd15" else {"checkpointState"}
    _known(raw, common | family_fields, "dataset")
    if raw.get("type", "image-caption-folder") != "image-caption-folder":
        raise TrainingConfigError("dataset.type must be 'image-caption-folder'")
    root_value = raw.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise TrainingConfigError("dataset.root must be a non-empty string")
    resolution = cast(
        "tuple[int, int]",
        _shape(raw.get("resolution", [512, 512]), "dataset.resolution", 2),
    )
    if any(side % 8 for side in resolution):
        raise TrainingConfigError("dataset.resolution entries must be divisible by 8")
    root = Path(root_value).expanduser().resolve()
    encoded_cache_value = raw.get("encodedCacheRoot")
    if encoded_cache_value is not None and (
        not isinstance(encoded_cache_value, str) or not encoded_cache_value
    ):
        raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty string")
    encoded_cache_root = (
        None
        if encoded_cache_value is None
        else str(Path(encoded_cache_value).expanduser().resolve())
    )
    if family == "sd15":
        vae_state = _encoder_source(raw.get("vaeState"), "dataset.vaeState")
        text_state = _encoder_source(raw.get("textEncoderState"), "dataset.textEncoderState")
        checkpoint_state = None
    else:
        vae_state = None
        text_state = None
        checkpoint_state = _encoder_source(raw.get("checkpointState"), "dataset.checkpointState")
        if checkpoint_state.prefix:
            raise TrainingConfigError(
                "dataset.checkpointState.prefix must be empty for a standard SDXL checkpoint"
            )
    inspection = inspect_image_caption_dataset(
        root,
        resolution,
        vae_state,
        text_state,
        family=family,
        checkpoint_state=checkpoint_state,
    )
    if (expected := raw.get("digest")) is not None:
        expected_digest = _digest(expected, "dataset.digest")
        if expected_digest != inspection.digest:
            raise TrainingConfigError(
                f"dataset digest changed: expected {expected_digest}, got {inspection.digest}"
            )
    return ImageCaptionDatasetSettings(
        family=family,
        root=str(root),
        resolution=resolution,
        vae_state=vae_state,
        text_encoder_state=text_state,
        checkpoint_state=checkpoint_state,
        inspection=inspection,
        encoded_cache_root=encoded_cache_root,
    )


def _h3_dataset_settings(value: object, *, device: str) -> MiniMaxH3DatasetSettings:
    raw = _object(value, "dataset")
    _known(
        raw,
        {
            "type",
            "root",
            "resolution",
            "frameCount",
            "videoVaeState",
            "audioVaeState",
            "conditionerState",
            "digest",
            "encodedCacheRoot",
        },
        "dataset",
    )
    if raw.get("type") != "h3-video-audio-caption-folder":
        raise TrainingConfigError("dataset.type must be 'h3-video-audio-caption-folder'")
    root_value = raw.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise TrainingConfigError("dataset.root must be a non-empty string")
    resolution = cast(
        "tuple[int, int]",
        _shape(raw.get("resolution"), "dataset.resolution", 2),
    )
    resolution_unit = MINIMAX_H3_CONFIG.video_spatial_downscale * MINIMAX_H3_CONFIG.patch[1]
    if any(side % resolution_unit for side in resolution):
        raise TrainingConfigError(
            f"dataset.resolution entries must be divisible by {resolution_unit}"
        )
    frame_count = _int(raw.get("frameCount"), "dataset.frameCount", minimum=1)
    if frame_count < 5 or (frame_count - 5) % 17:
        raise TrainingConfigError("dataset.frameCount must equal 17k+5 for an integer k >= 0")
    video_vae_state = _h3_component_source(raw.get("videoVaeState"), "dataset.videoVaeState")
    audio_vae_state = _h3_component_source(raw.get("audioVaeState"), "dataset.audioVaeState")
    conditioner_state = _h3_component_source(
        raw.get("conditionerState"), "dataset.conditionerState"
    )
    encoded_cache_value = raw.get("encodedCacheRoot")
    if encoded_cache_value is not None and (
        not isinstance(encoded_cache_value, str) or not encoded_cache_value
    ):
        raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty string")
    root = Path(root_value).expanduser().resolve()
    encoded_cache_root = (
        None
        if encoded_cache_value is None
        else str(Path(encoded_cache_value).expanduser().resolve())
    )
    inspection = inspect_minimax_h3_dataset(
        root,
        resolution,
        frame_count,
        video_vae_state,
        audio_vae_state,
        conditioner_state,
        validate_media=False,
    )
    if (expected := raw.get("digest")) is not None:
        expected_digest = _digest(expected, "dataset.digest")
        if expected_digest != inspection.digest:
            raise TrainingConfigError(
                f"dataset digest changed: expected {expected_digest}, got {inspection.digest}"
            )
    settings = MiniMaxH3DatasetSettings(
        root=str(root),
        resolution=resolution,
        frame_count=frame_count,
        video_vae_state=video_vae_state,
        audio_vae_state=audio_vae_state,
        conditioner_state=conditioner_state,
        inspection=inspection,
        encoded_cache_root=encoded_cache_root,
    )
    if encoded_cache_root is not None:
        from .encoded_cache import minimax_h3_encoded_cache_state

        if minimax_h3_encoded_cache_state(settings, device=torch.device(device)) == "hit":
            return settings
    validated = inspect_minimax_h3_dataset(
        root,
        resolution,
        frame_count,
        video_vae_state,
        audio_vae_state,
        conditioner_state,
    )
    if validated.digest != inspection.digest:
        raise TrainingConfigError("dataset changed during inspection")
    return MiniMaxH3DatasetSettings(
        root=settings.root,
        resolution=settings.resolution,
        frame_count=settings.frame_count,
        video_vae_state=settings.video_vae_state,
        audio_vae_state=settings.audio_vae_state,
        conditioner_state=settings.conditioner_state,
        inspection=validated,
        encoded_cache_root=encoded_cache_root,
    )


def _music3_dataset_settings(
    value: object,
    *,
    device: str,
    text_dtype: BaseDtypeName,
) -> MiniMaxMusic3DatasetSettings:
    raw = _object(value, "dataset")
    _known(
        raw,
        {
            "type",
            "root",
            "audioFrames",
            "davEncoderState",
            "rvqEncoderState",
            "textEncoderState",
            "digest",
            "encodedCacheRoot",
        },
        "dataset",
    )
    if raw.get("type") != "minimax-music3-audio-caption-lyrics-folder":
        raise TrainingConfigError(
            "dataset.type must be 'minimax-music3-audio-caption-lyrics-folder'"
        )
    root_value = raw.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise TrainingConfigError("dataset.root must be a non-empty string")
    root = Path(root_value).expanduser().resolve()
    audio_frames = _int(raw.get("audioFrames"), "dataset.audioFrames", minimum=1)
    if audio_frames > 128:
        raise TrainingConfigError("dataset.audioFrames must be <= 128")
    dav = _music3_artifact_pin(
        raw.get("davEncoderState"), "dataset.davEncoderState", identity=False
    )
    rvq = _music3_artifact_pin(
        raw.get("rvqEncoderState"), "dataset.rvqEncoderState", identity=False
    )
    text = _music3_artifact_pin(
        raw.get("textEncoderState"), "dataset.textEncoderState", identity=True
    )
    cache_value = raw.get("encodedCacheRoot")
    if cache_value is not None and (not isinstance(cache_value, str) or not cache_value):
        raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty string")
    cache_root = None if cache_value is None else str(Path(cache_value).expanduser().resolve())
    inspected = inspect_minimax_music3_dataset(
        root,
        audio_frames,
        dav,
        rvq,
        text,
        text_dtype,
        validate_audio=False,
    )
    if (expected := raw.get("digest")) is not None:
        expected_digest = _digest(expected, "dataset.digest")
        if expected_digest != inspected.digest:
            raise TrainingConfigError(
                f"dataset digest changed: expected {expected_digest}, got {inspected.digest}"
            )
    settings = MiniMaxMusic3DatasetSettings(
        str(root),
        audio_frames,
        dav,
        rvq,
        text,
        text_dtype,
        inspected,
        cache_root,
    )
    if (
        cache_root is not None
        and minimax_music3_encoded_cache_state(settings, device=torch.device(device)) == "hit"
    ):
        return settings
    validated = inspect_minimax_music3_dataset(
        root,
        audio_frames,
        dav,
        rvq,
        text,
        text_dtype,
    )
    if validated.digest != inspected.digest:
        raise TrainingConfigError("dataset changed during inspection")
    if validated.errors:
        raise TrainingConfigError("dataset validation failed: " + "; ".join(validated.errors))
    return replace(settings, inspection=validated)


def _wan_dataset_settings(
    value: object,
    *,
    vae_state: TrainingArtifactSource,
    umt5xxl_state: TrainingArtifactSource,
    latent_shape: tuple[int, int, int, int, int],
    context_shape: tuple[int, int, int],
    device: str,
    vae_contract: WanVAEContract,
    conditioning_contract: WanConditioningContract,
) -> WanDatasetSettings:
    raw = _object(value, "dataset")
    _known(
        raw,
        {"type", "root", "resolution", "frameCount", "digest", "encodedCacheRoot"},
        "dataset",
    )
    dataset_type = (
        "wan22-video-caption-folder" if vae_contract == "wan22" else "wan21-video-caption-folder"
    )
    if raw.get("type") != dataset_type:
        raise TrainingConfigError(f"dataset.type must be {dataset_type!r}")
    root_value = raw.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise TrainingConfigError("dataset.root must be a non-empty string")
    resolution = cast("tuple[int, int]", _shape(raw.get("resolution"), "dataset.resolution", 2))
    spatial_downscale = 16 if vae_contract == "wan22" else 8
    if any(side % spatial_downscale for side in resolution):
        raise TrainingConfigError(
            f"dataset.resolution entries must be divisible by {spatial_downscale}"
        )
    frame_count = _int(raw.get("frameCount"), "dataset.frameCount", minimum=1)
    if (frame_count - 1) % 4:
        raise TrainingConfigError("dataset.frameCount must equal 4k+1 for an integer k >= 0")
    encoded_cache_value = raw.get("encodedCacheRoot")
    if encoded_cache_value is not None and (
        not isinstance(encoded_cache_value, str) or not encoded_cache_value
    ):
        raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty string")
    root = Path(root_value).expanduser().resolve()
    inspection = inspect_wan_dataset(
        root,
        resolution,
        frame_count,
        vae_identity={"digest": vae_state.digest, "size": vae_state.size},
        umt5xxl_identity={"digest": umt5xxl_state.digest, "size": umt5xxl_state.size},
        vae_contract=vae_contract,
        conditioning_contract=conditioning_contract,
        validate_media=False,
    )
    if (expected := raw.get("digest")) is not None:
        expected_digest = _digest(expected, "dataset.digest")
        if expected_digest != inspection.digest:
            raise TrainingConfigError(
                f"dataset digest changed: expected {expected_digest}, got {inspection.digest}"
            )
    settings = WanDatasetSettings(
        root=str(root),
        resolution=resolution,
        frame_count=frame_count,
        inspection=inspection,
        vae_contract=vae_contract,
        conditioning_contract=conditioning_contract,
        encoded_cache_root=(
            None
            if encoded_cache_value is None
            else str(Path(encoded_cache_value).expanduser().resolve())
        ),
    )
    if settings.encoded_cache_root is not None:
        from .encoded_cache import WanEncodedDatasetCache

        cache = WanEncodedDatasetCache(
            settings,
            latent_shape=latent_shape,
            context_shape=context_shape,
            device=torch.device(device),
        )
        if cache.is_valid():
            return settings
    validated = inspect_wan_dataset(
        root,
        resolution,
        frame_count,
        vae_identity={"digest": vae_state.digest, "size": vae_state.size},
        umt5xxl_identity={"digest": umt5xxl_state.digest, "size": umt5xxl_state.size},
        vae_contract=vae_contract,
        conditioning_contract=conditioning_contract,
    )
    if validated.digest != inspection.digest:
        raise TrainingConfigError("dataset changed during inspection")
    if validated.errors:
        raise TrainingConfigError("dataset validation failed: " + "; ".join(validated.errors))
    return replace(settings, inspection=validated)


def _flux_dataset_settings(
    value: object,
    *,
    vae_state: TrainingArtifactSource,
    clip_l_state: TrainingArtifactSource,
    t5xxl_state: TrainingArtifactSource,
    context_tokens: int,
    variant: FluxVariantName,
) -> FluxDatasetSettings:
    raw = _object(value, "dataset")
    _known(
        raw,
        {
            "type",
            "variant",
            "root",
            "resolution",
            "contextTokens",
            "digest",
            "encodedCacheRoot",
        },
        "dataset",
    )
    if raw.get("type") != "flux-image-caption-folder":
        raise TrainingConfigError("dataset.type must be 'flux-image-caption-folder'")
    if raw.get("variant", variant) != variant:
        raise TrainingConfigError("dataset.variant must match the Flux training variant")
    root_value = raw.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise TrainingConfigError("dataset.root must be a non-empty string")
    resolution = cast("tuple[int, int]", _shape(raw.get("resolution"), "dataset.resolution", 2))
    if any(side % 16 for side in resolution):
        raise TrainingConfigError("dataset.resolution entries must be divisible by 16")
    supplied_tokens = _int(
        raw.get("contextTokens", context_tokens), "dataset.contextTokens", minimum=256
    )
    if supplied_tokens != context_tokens:
        raise TrainingConfigError("dataset.contextTokens must match contextShape token count")
    encoded_cache_value = raw.get("encodedCacheRoot")
    if encoded_cache_value is not None and (
        not isinstance(encoded_cache_value, str) or not encoded_cache_value
    ):
        raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty string")
    root = Path(root_value).expanduser().resolve()
    inspection = inspect_flux_dataset(
        root,
        resolution,
        context_tokens,
        variant=variant,
        vae_identity={"digest": vae_state.digest, "size": vae_state.size},
        clip_l_identity={"digest": clip_l_state.digest, "size": clip_l_state.size},
        t5xxl_identity={"digest": t5xxl_state.digest, "size": t5xxl_state.size},
    )
    if (expected := raw.get("digest")) is not None:
        expected_digest = _digest(expected, "dataset.digest")
        if expected_digest != inspection.digest:
            raise TrainingConfigError(
                f"dataset digest changed: expected {expected_digest}, got {inspection.digest}"
            )
    if inspection.errors:
        raise TrainingConfigError("dataset validation failed: " + "; ".join(inspection.errors))
    return FluxDatasetSettings(
        root=str(root),
        variant=variant,
        resolution=resolution,
        context_tokens=context_tokens,
        inspection=inspection,
        encoded_cache_root=(
            None
            if encoded_cache_value is None
            else str(Path(encoded_cache_value).expanduser().resolve())
        ),
    )


def _flux2_dataset_settings(
    value: object,
    *,
    vae_state: TrainingArtifactSource,
    text_encoder_state: TrainingArtifactSource,
    context_tokens: int,
    variant: Flux2VariantName,
) -> Flux2DatasetSettings:
    raw = _object(value, "dataset")
    _known(
        raw,
        {
            "type",
            "variant",
            "root",
            "resolution",
            "contextTokens",
            "digest",
            "encodedCacheRoot",
        },
        "dataset",
    )
    if raw.get("type") != "flux2-image-caption-folder":
        raise TrainingConfigError("dataset.type must be 'flux2-image-caption-folder'")
    variant_value = raw.get("variant", variant)
    if not isinstance(variant_value, str) or variant_value != variant:
        raise TrainingConfigError("dataset.variant must match the Flux2 training variant")
    root_value = raw.get("root")
    if not isinstance(root_value, str) or not root_value:
        raise TrainingConfigError("dataset.root must be a non-empty string")
    resolution = cast("tuple[int, int]", _shape(raw.get("resolution"), "dataset.resolution", 2))
    if any(side % 16 for side in resolution):
        raise TrainingConfigError("dataset.resolution entries must be divisible by 16")
    supplied_tokens = _int(
        raw.get("contextTokens", context_tokens), "dataset.contextTokens", minimum=512
    )
    if supplied_tokens != context_tokens:
        raise TrainingConfigError("dataset.contextTokens must match contextShape token count")
    encoded_cache_value = raw.get("encodedCacheRoot")
    if encoded_cache_value is not None and (
        not isinstance(encoded_cache_value, str) or not encoded_cache_value
    ):
        raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty string")
    root = Path(root_value).expanduser().resolve()
    inspection = inspect_flux2_dataset(
        root,
        resolution,
        context_tokens,
        variant=variant,
        vae_identity={"digest": vae_state.digest, "size": vae_state.size},
        text_encoder_identity={
            "digest": text_encoder_state.digest,
            "size": text_encoder_state.size,
        },
    )
    if (expected := raw.get("digest")) is not None:
        expected_digest = _digest(expected, "dataset.digest")
        if expected_digest != inspection.digest:
            raise TrainingConfigError(
                f"dataset digest changed: expected {expected_digest}, got {inspection.digest}"
            )
    if inspection.errors:
        raise TrainingConfigError("dataset validation failed: " + "; ".join(inspection.errors))
    return Flux2DatasetSettings(
        root=str(root),
        variant=variant,
        resolution=resolution,
        context_tokens=context_tokens,
        inspection=inspection,
        encoded_cache_root=(
            None
            if encoded_cache_value is None
            else str(Path(encoded_cache_value).expanduser().resolve())
        ),
    )


def _qwen_image_dataset_settings(
    value: object,
    *,
    vae_state: TrainingArtifactSource,
    text_encoder_state: TrainingArtifactSource,
    context_tokens: int,
) -> QwenImageDatasetSettings:
    raw = _object(value, "dataset")
    _known(
        raw,
        {
            "type",
            "root",
            "resolution",
            "contextTokens",
            "digest",
            "encodedCacheRoot",
        },
        "dataset",
    )
    dataset_type = raw.get("type")
    if type(dataset_type) is not str or dataset_type != "qwen-image-image-caption-folder":
        raise TrainingConfigError("dataset.type must be 'qwen-image-image-caption-folder'")
    root_value = raw.get("root")
    if type(root_value) is not str or not root_value:
        raise TrainingConfigError("dataset.root must be a non-empty string")
    resolution = cast("tuple[int, int]", _shape(raw.get("resolution"), "dataset.resolution", 2))
    if any(side % 16 for side in resolution):
        raise TrainingConfigError("dataset.resolution entries must be divisible by 16")
    supplied_tokens = _int(
        raw.get("contextTokens", context_tokens), "dataset.contextTokens", minimum=512
    )
    if supplied_tokens != context_tokens:
        raise TrainingConfigError("dataset.contextTokens must match contextShape token count")
    encoded_cache_value = raw.get("encodedCacheRoot")
    if encoded_cache_value is not None and (
        type(encoded_cache_value) is not str or not encoded_cache_value
    ):
        raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty string")
    root = Path(root_value).expanduser().resolve()
    inspection = inspect_qwen_image_dataset(
        root,
        resolution,
        context_tokens,
        vae_identity={"digest": vae_state.digest, "size": vae_state.size},
        text_encoder_identity={
            "digest": text_encoder_state.digest,
            "size": text_encoder_state.size,
        },
    )
    if (expected := raw.get("digest")) is not None:
        expected_digest = _digest(expected, "dataset.digest")
        if expected_digest != inspection.digest:
            raise TrainingConfigError(
                f"dataset digest changed: expected {expected_digest}, got {inspection.digest}"
            )
    if inspection.errors:
        raise TrainingConfigError("dataset validation failed: " + "; ".join(inspection.errors))
    return QwenImageDatasetSettings(
        root=str(root),
        resolution=resolution,
        context_tokens=context_tokens,
        inspection=inspection,
        encoded_cache_root=(
            None
            if encoded_cache_value is None
            else str(Path(encoded_cache_value).expanduser().resolve())
        ),
    )


def _ideogram4_dataset_settings(
    value: object,
    *,
    role: Ideogram4RoleName,
    vae_state: NativeTrainingArtifactSource,
    text_encoder_state: NativeTrainingArtifactSource | None,
    context_tokens: int,
) -> Ideogram4DatasetSettings:
    raw = _object(value, "dataset")
    _known(
        raw,
        {"type", "root", "resolution", "contextTokens", "digest", "encodedCacheRoot"},
        "dataset",
    )
    expected_type = (
        "ideogram4-image-caption-folder" if role == "conditional" else "ideogram4-image-folder"
    )
    if raw.get("type") != expected_type:
        raise TrainingConfigError(f"dataset.type must be {expected_type!r} for role {role!r}")
    root_value = raw.get("root")
    if type(root_value) is not str or not root_value:
        raise TrainingConfigError("dataset.root must be a non-empty string")
    resolution = cast("tuple[int, int]", _shape(raw.get("resolution"), "dataset.resolution", 2))
    if any(side % 16 for side in resolution):
        raise TrainingConfigError("dataset.resolution entries must be divisible by 16")
    if role == "conditional":
        supplied_tokens = _int(
            raw.get("contextTokens", context_tokens), "dataset.contextTokens", minimum=1
        )
        if supplied_tokens != context_tokens:
            raise TrainingConfigError("dataset.contextTokens must match contextShape token count")
    elif "contextTokens" in raw:
        raise TrainingConfigError("unconditional Ideogram 4 datasets do not accept contextTokens")
    encoded_cache_value = raw.get("encodedCacheRoot")
    if encoded_cache_value is not None and (
        type(encoded_cache_value) is not str or not encoded_cache_value
    ):
        raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty string")
    root = Path(root_value).expanduser().resolve()
    inspection = inspect_ideogram4_dataset(
        root,
        resolution,
        context_tokens,
        role=role,
        vae_identity={
            "digest": vae_state.digest,
            "size": vae_state.size,
            "identity": vae_state.identity,
        },
        text_encoder_identity=(
            None
            if text_encoder_state is None
            else {
                "digest": text_encoder_state.digest,
                "size": text_encoder_state.size,
                "identity": text_encoder_state.identity,
            }
        ),
    )
    if (expected := raw.get("digest")) is not None:
        expected_digest = _digest(expected, "dataset.digest")
        if expected_digest != inspection.digest:
            raise TrainingConfigError(
                f"dataset digest changed: expected {expected_digest}, got {inspection.digest}"
            )
    if inspection.errors:
        raise TrainingConfigError("dataset validation failed: " + "; ".join(inspection.errors))
    return Ideogram4DatasetSettings(
        root=str(root),
        role=role,
        resolution=resolution,
        context_tokens=context_tokens,
        inspection=inspection,
        encoded_cache_root=(
            None
            if encoded_cache_value is None
            else str(Path(encoded_cache_value).expanduser().resolve())
        ),
    )


def _unet_from_mapping(value: object | None, family: TrainingFamily) -> UNetConfig:
    if value is None:
        return SD15_UNET_CONFIG if family == "sd15" else SDXL_UNET_CONFIG
    raw = _object(value, "unet")
    fields = {
        "inChannels",
        "outChannels",
        "modelChannels",
        "numResBlocks",
        "channelMult",
        "transformerDepth",
        "transformerDepthOutput",
        "transformerDepthMiddle",
        "contextDim",
        "useLinearInTransformer",
        "admInChannels",
        "numHeads",
        "numHeadChannels",
        "dropout",
    }
    _known(raw, fields, "unet")
    missing = sorted(fields - {"admInChannels", "numHeadChannels", "dropout"} - set(raw))
    if missing:
        raise TrainingConfigError(f"unet is missing fields: {', '.join(missing)}")
    use_linear = raw["useLinearInTransformer"]
    if type(use_linear) is not bool:
        raise TrainingConfigError("unet.useLinearInTransformer must be a boolean")
    try:
        return UNetConfig(
            in_channels=_int(raw["inChannels"], "unet.inChannels", minimum=1),
            out_channels=_int(raw["outChannels"], "unet.outChannels", minimum=1),
            model_channels=_int(raw["modelChannels"], "unet.modelChannels", minimum=1),
            num_res_blocks=_tuple_int(raw, "numResBlocks"),
            channel_mult=_tuple_int(raw, "channelMult"),
            transformer_depth=_tuple_int(raw, "transformerDepth"),
            transformer_depth_output=_tuple_int(raw, "transformerDepthOutput"),
            transformer_depth_middle=_int(
                raw["transformerDepthMiddle"], "unet.transformerDepthMiddle", minimum=1
            ),
            context_dim=_int(raw["contextDim"], "unet.contextDim", minimum=1),
            use_linear_in_transformer=use_linear,
            adm_in_channels=(
                None
                if raw.get("admInChannels") is None
                else _int(raw["admInChannels"], "unet.admInChannels", minimum=1)
            ),
            num_heads=_int(raw["numHeads"], "unet.numHeads", minimum=-1),
            num_head_channels=_int(
                raw.get("numHeadChannels", -1), "unet.numHeadChannels", minimum=-1
            ),
            dropout=_float(raw.get("dropout", 0.0), "unet.dropout", maximum=0.999999),
        )
    except ValueError as exc:
        raise TrainingConfigError(str(exc)) from exc


def _unet_wire(config: UNetConfig) -> dict[str, object]:
    wire: dict[str, object] = {
        "inChannels": config.in_channels,
        "outChannels": config.out_channels,
        "modelChannels": config.model_channels,
        "numResBlocks": list(config.num_res_blocks),
        "channelMult": list(config.channel_mult),
        "transformerDepth": list(config.transformer_depth),
        "transformerDepthOutput": list(config.transformer_depth_output),
        "transformerDepthMiddle": config.transformer_depth_middle,
        "contextDim": config.context_dim,
        "useLinearInTransformer": config.use_linear_in_transformer,
        "numHeads": config.num_heads,
        "numHeadChannels": config.num_head_channels,
        "dropout": config.dropout,
    }
    if config.adm_in_channels is not None:
        wire["admInChannels"] = config.adm_in_channels
    return wire


@dataclass(frozen=True)
class TrainingConfig:
    """Normalized behavior and source configuration for one session."""

    family: TrainingFamily
    unet: UNetConfig
    base_state_path: str | None
    base_state_digest: str | None
    base_state_prefix: str
    prepared_batch_root: str | None
    dataset: ImageCaptionDatasetSettings | None
    distributed: DistributedSettings | None
    device: str
    base_dtype: BaseDtypeName
    rank: int
    alpha: float
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    optimizer: OptimizerName
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    checkpointing_mode: CheckpointingModeName
    checkpoint_interval: int
    lora_export_interval: int
    sync_digest_interval: int
    seed: int
    rng_policy: RngPolicyName
    latent_shape: tuple[int, int, int, int]
    context_shape: tuple[int, int, int]
    pooled_shape: tuple[int, int] | None

    @classmethod
    def parse(
        cls, serialized: str, *, expected_family: TrainingFamily | None = None
    ) -> TrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"config must be valid JSON: {exc.msg}") from exc
        return cls.from_mapping(_object(value, "config"), expected_family=expected_family)

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, object],
        *,
        expected_family: TrainingFamily | None = None,
    ) -> TrainingConfig:
        fields = {
            "schemaVersion",
            "family",
            "unet",
            "baseState",
            "preparedBatchRoot",
            "dataset",
            "distributed",
            "device",
            "baseDtype",
            "rank",
            "alpha",
            "learningRate",
            "weightDecay",
            "betas",
            "epsilon",
            "optimizer",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "checkpointingMode",
            "checkpointInterval",
            "loraExportInterval",
            "syncDigestInterval",
            "seed",
            "rngPolicy",
            "latentShape",
            "contextShape",
            "pooledShape",
            "trainTextEncoder",
        }
        _known(raw, fields, "config")
        if raw.get("schemaVersion", 1) != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        family_value = raw.get("family", "sd15")
        if family_value not in ("sd15", "sdxl"):
            raise TrainingConfigError("family must be 'sd15' or 'sdxl'")
        family = family_value
        if expected_family is not None and family != expected_family:
            raise TrainingConfigError(f"family must be {expected_family!r} for this backend")
        if raw.get("trainTextEncoder", False) is not False:
            raise TrainingConfigError("text-encoder training is not enabled for this backend")

        base_path: str | None = None
        base_digest: str | None = None
        base_prefix = ""
        if (source_value := raw.get("baseState")) is not None:
            source = _object(source_value, "baseState")
            _known(source, {"path", "digest", "prefix"}, "baseState")
            path = source.get("path")
            digest = source.get("digest")
            prefix = source.get("prefix", "")
            if not isinstance(path, str) or not path:
                raise TrainingConfigError("baseState.path must be a non-empty string")
            digest = _digest(digest, "baseState.digest")
            if not isinstance(prefix, str):
                raise TrainingConfigError("baseState.prefix must be a string")
            base_path = str(Path(path).expanduser().resolve()) if family == "sdxl" else path
            base_digest = digest
            base_prefix = prefix
        if family == "sdxl" and base_prefix:
            raise TrainingConfigError(
                "baseState.prefix must be empty for a standard SDXL checkpoint"
            )

        batch_root = raw.get("preparedBatchRoot")
        if batch_root is not None and (not isinstance(batch_root, str) or not batch_root):
            raise TrainingConfigError("preparedBatchRoot must be a non-empty string")
        dataset = None if raw.get("dataset") is None else _dataset_settings(raw["dataset"], family)
        distributed = (
            None if raw.get("distributed") is None else _distributed_settings(raw["distributed"])
        )
        if batch_root is not None and dataset is not None:
            raise TrainingConfigError("preparedBatchRoot and dataset cannot both be set")
        if family == "sdxl" and dataset is not None and base_path is not None:
            assert dataset.checkpoint_state is not None
            if (
                dataset.checkpoint_state.path != str(Path(base_path).expanduser().resolve())
                or dataset.checkpoint_state.digest != base_digest
            ):
                raise TrainingConfigError(
                    "SDXL baseState and dataset.checkpointState must name the same checkpoint"
                )
        device = raw.get("device", "cpu")
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        base_dtype = raw.get("baseDtype", "float32")
        if base_dtype not in ("float32", "bfloat16"):
            raise TrainingConfigError("baseDtype must be 'float32' or 'bfloat16'")
        optimizer = raw.get("optimizer", "adamw")
        if optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        checkpointing = raw.get("gradientCheckpointing", True)
        if type(checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode = raw.get("checkpointingMode", "blockReentrant")
        if checkpointing_mode not in ("blockReentrant", "wholeModel"):
            raise TrainingConfigError("checkpointingMode must be 'blockReentrant' or 'wholeModel'")
        checkpoint_interval = _int(raw.get("checkpointInterval", 0), "checkpointInterval")
        lora_export_interval = _int(raw.get("loraExportInterval", 0), "loraExportInterval")
        sync_digest_interval = _int(raw.get("syncDigestInterval", 0), "syncDigestInterval")
        rng_policy = raw.get("rngPolicy", "sequential")
        if rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        betas_value = raw.get("betas", [0.9, 0.999])
        if not isinstance(betas_value, list) or len(betas_value) != 2:
            raise TrainingConfigError("betas must be a two-item number array")
        beta1 = _float(betas_value[0], "betas[0]", maximum=0.999999999)
        beta2 = _float(betas_value[1], "betas[1]", maximum=0.999999999)
        if beta1 == 0.0 or beta2 == 0.0:
            raise TrainingConfigError("betas entries must be greater than zero")

        latent = cast(
            "tuple[int, int, int, int]",
            _shape(
                raw.get(
                    "latentShape",
                    [
                        1,
                        4,
                        *(
                            [64, 64]
                            if dataset is None
                            else [dataset.resolution[0] // 8, dataset.resolution[1] // 8]
                        ),
                    ],
                ),
                "latentShape",
                4,
            ),
        )
        context = cast(
            "tuple[int, int, int]",
            _shape(
                raw.get("contextShape", [1, 77, 768 if family == "sd15" else 2048]),
                "contextShape",
                3,
            ),
        )
        pooled = (
            None
            if family == "sd15"
            else cast(
                "tuple[int, int]",
                _shape(raw.get("pooledShape", [latent[0], 1280]), "pooledShape", 2),
            )
        )
        if family == "sd15" and raw.get("pooledShape") is not None:
            raise TrainingConfigError("pooledShape is available only for SDXL")
        unet = _unet_from_mapping(raw.get("unet"), family)
        if family == "sd15" and unet.adm_in_channels is not None:
            raise TrainingConfigError("unet.admInChannels is available only for SDXL")
        if family == "sd15" and unet.use_linear_in_transformer:
            raise TrainingConfigError(
                "unet.useLinearInTransformer must be false for the SD1.5 target set"
            )
        if family == "sdxl" and not unet.use_linear_in_transformer:
            raise TrainingConfigError(
                "unet.useLinearInTransformer must be true for the SDXL target set"
            )
        if unet.dropout != 0.0:
            raise TrainingConfigError(
                "unet.dropout must be zero so resume does not depend on process-global RNG state"
            )
        if latent[1] != unet.in_channels:
            raise TrainingConfigError(
                f"latentShape has {latent[1]} channels but the UNet expects {unet.in_channels}"
            )
        if context[2] != unet.context_dim:
            raise TrainingConfigError(
                f"contextShape has width {context[2]} but the UNet expects {unet.context_dim}"
            )
        if latent[0] != context[0]:
            raise TrainingConfigError("latentShape and contextShape batch dimensions must match")
        if pooled is not None:
            if pooled[0] != latent[0]:
                raise TrainingConfigError(
                    "latentShape, contextShape, and pooledShape batch dimensions must match"
                )
            if unet.adm_in_channels != pooled[1] + 6 * 256:
                raise TrainingConfigError(
                    "unet.admInChannels must equal pooledShape width plus six"
                    " 256-wide size embeddings"
                )
        if dataset is not None and latent[2:] != (
            dataset.resolution[0] // 8,
            dataset.resolution[1] // 8,
        ):
            raise TrainingConfigError(
                "latentShape spatial dimensions must equal dataset.resolution divided by 8"
            )

        return cls(
            family=family,
            unet=unet,
            base_state_path=base_path,
            base_state_digest=base_digest,
            base_state_prefix=base_prefix,
            prepared_batch_root=batch_root,
            dataset=dataset,
            distributed=distributed,
            device=device,
            base_dtype=base_dtype,
            rank=_int(raw.get("rank", 4), "rank", minimum=1),
            alpha=_float(raw.get("alpha", 4.0), "alpha", minimum=0.000000001),
            learning_rate=_float(
                raw.get("learningRate", 0.0001), "learningRate", minimum=0.000000000001
            ),
            weight_decay=_float(raw.get("weightDecay", 0.01), "weightDecay"),
            beta1=beta1,
            beta2=beta2,
            epsilon=_float(raw.get("epsilon", 1e-8), "epsilon", minimum=1e-30),
            optimizer=optimizer,
            gradient_accumulation_steps=_int(
                raw.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
                minimum=1,
            ),
            gradient_checkpointing=checkpointing,
            checkpointing_mode=checkpointing_mode,
            checkpoint_interval=checkpoint_interval,
            lora_export_interval=lora_export_interval,
            sync_digest_interval=sync_digest_interval,
            seed=_int(raw.get("seed", 0), "seed"),
            rng_policy=rng_policy,
            latent_shape=latent,
            context_shape=context,
            pooled_shape=pooled,
        )

    def to_mapping(self) -> dict[str, object]:
        base_state: dict[str, object] | None = None
        if self.base_state_path is not None:
            assert self.base_state_digest is not None
            base_state = {
                "path": self.base_state_path,
                "digest": self.base_state_digest,
                "prefix": self.base_state_prefix,
            }
        mapping: dict[str, object] = {
            "schemaVersion": 1,
            "family": self.family,
            "unet": _unet_wire(self.unet),
            "baseState": base_state,
            "preparedBatchRoot": self.prepared_batch_root,
            "dataset": None if self.dataset is None else self.dataset.to_mapping(),
            "device": self.device,
            "baseDtype": self.base_dtype,
            "rank": self.rank,
            "alpha": self.alpha,
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "betas": [self.beta1, self.beta2],
            "epsilon": self.epsilon,
            "optimizer": self.optimizer,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "gradientCheckpointing": self.gradient_checkpointing,
            "checkpointingMode": self.checkpointing_mode,
            "seed": self.seed,
            "latentShape": list(self.latent_shape),
            "contextShape": list(self.context_shape),
            "trainTextEncoder": False,
        }
        if self.pooled_shape is not None:
            mapping["pooledShape"] = list(self.pooled_shape)
        if self.checkpoint_interval:
            mapping["checkpointInterval"] = self.checkpoint_interval
        if self.lora_export_interval:
            mapping["loraExportInterval"] = self.lora_export_interval
        if self.sync_digest_interval:
            mapping["syncDigestInterval"] = self.sync_digest_interval
        if self.distributed is not None:
            mapping["distributed"] = self.distributed.to_mapping()
        if self.rng_policy != "sequential":
            mapping["rngPolicy"] = self.rng_policy
        return mapping

    def identity_mapping(self) -> dict[str, object]:
        mapping = self.to_mapping()
        # Checkpoint cadence changes which safe points become durable
        # checkpoints, never the optimizer, RNG, adapter, or data trajectory.
        for field in _COMMON_RUNTIME_ONLY_FIELDS:
            mapping.pop(field, None)
        dataset = mapping.get("dataset")
        if isinstance(dataset, dict):
            for field in _DATASET_RUNTIME_ONLY_FIELDS:
                dataset.pop(field, None)
        distributed = mapping.get("distributed")
        if isinstance(distributed, dict):
            for field in _DISTRIBUTED_RUNTIME_ONLY_FIELDS:
                distributed.pop(field, None)
        return mapping


@dataclass(frozen=True)
class WanTrainingConfig:
    """Normalized behavior and component identity for Wan LoRA training."""

    family: Literal["wan"]
    variant: WanVariantName
    dit_state: TrainingArtifactSource
    umt5xxl_state: TrainingArtifactSource
    vae_state: TrainingArtifactSource
    dataset_identity: str
    distributed: DistributedSettings | None
    device: str
    base_dtype: WanBaseDtypeName
    rank: int
    alpha: float
    lora_targets: tuple[WanLoraTargetName, WanLoraTargetName]
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    optimizer: OptimizerName
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    checkpointing_mode: Literal["blockNonReentrant", "wholeModel"]
    checkpoint_interval: int
    lora_export_interval: int
    sync_digest_interval: int
    seed: int
    rng_policy: RngPolicyName
    latent_shape: tuple[int, int, int, int, int]
    context_shape: tuple[int, int, int]
    dataset: WanDatasetSettings | None = None
    expert: WanExpertName | None = None
    timestep_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.family != "wan":
            raise TrainingConfigError("family must be 'wan'")
        if self.variant not in _WAN_VARIANTS:
            raise TrainingConfigError("unsupported Wan training variant")
        if self.variant in ("wan22-t2v-14b", "wan22-i2v-14b"):
            ranges = _WAN22_EXPERT_TIMESTEP_RANGES[self.variant]
            expert_range = ranges.get(self.expert) if isinstance(self.expert, str) else None
            if expert_range is None:
                raise TrainingConfigError(
                    "Wan 2.2 14B training requires one high-noise or low-noise expert"
                )
            if (
                type(self.timestep_range) is not tuple
                or any(type(value) is not int for value in self.timestep_range)
                or self.timestep_range != expert_range
            ):
                raise TrainingConfigError(
                    f"timestepRange must be {list(expert_range)} for variant "
                    f"{self.variant!r} expert {self.expert!r}"
                )
        elif self.expert is not None or self.timestep_range is not None:
            raise TrainingConfigError("expert and timestepRange apply only to Wan 2.2 14B training")
        vae_contract = "wan22" if self.variant == "wan22-ti2v-5b" else "wan21"
        if self.dataset is not None and self.dataset.vae_contract != vae_contract:
            raise TrainingConfigError("dataset VAE contract does not match the Wan variant")
        conditioning_contract = "first-frame-i2v" if self.variant == "wan22-i2v-14b" else "none"
        if self.dataset is not None and self.dataset.conditioning_contract != conditioning_contract:
            raise TrainingConfigError(
                "dataset conditioning contract does not match the Wan variant"
            )
        latent_channels = 48 if self.variant == "wan22-ti2v-5b" else 16
        if type(self.latent_shape[1]) is not int or self.latent_shape[1] != latent_channels:
            raise TrainingConfigError(
                f"latentShape must be [batch, {latent_channels}, time, height, width]"
            )
        if self.base_dtype not in ("float16", "bfloat16"):
            raise TrainingConfigError("baseDtype must be 'float16' or 'bfloat16'")
        if self.lora_targets != ("attention.qkvo", "ffn.projections"):
            raise TrainingConfigError("loraTargets must be ['attention.qkvo', 'ffn.projections']")
        if self.checkpointing_mode not in ("blockNonReentrant", "wholeModel"):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel' for Wan training"
            )

    @classmethod
    def parse(cls, serialized: str) -> WanTrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"config must be valid JSON: {exc.msg}") from exc
        return cls.from_mapping(_object(value, "config"))

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> WanTrainingConfig:
        fields = {
            "schemaVersion",
            "family",
            "variant",
            "expert",
            "timestepRange",
            "ditState",
            "umt5xxlState",
            "vaeState",
            "datasetIdentity",
            "dataset",
            "distributed",
            "device",
            "baseDtype",
            "rank",
            "alpha",
            "loraTargets",
            "learningRate",
            "weightDecay",
            "betas",
            "epsilon",
            "optimizer",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "checkpointingMode",
            "checkpointInterval",
            "loraExportInterval",
            "syncDigestInterval",
            "seed",
            "rngPolicy",
            "latentShape",
            "contextShape",
        }
        _known(raw, fields, "config")
        if raw.get("schemaVersion", 1) != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        family = raw.get("family")
        if family != "wan":
            raise TrainingConfigError("family must be 'wan'")
        variant = raw.get("variant")
        if variant not in _WAN_VARIANTS:
            raise TrainingConfigError("unsupported Wan training variant")
        vae_contract: WanVAEContract = "wan22" if variant == "wan22-ti2v-5b" else "wan21"
        conditioning_contract: WanConditioningContract = (
            "first-frame-i2v" if variant == "wan22-i2v-14b" else "none"
        )

        expert_value = raw.get("expert")
        if variant in ("wan22-t2v-14b", "wan22-i2v-14b"):
            ranges = _WAN22_EXPERT_TIMESTEP_RANGES[variant]
            if not isinstance(expert_value, str) or expert_value not in ranges:
                raise TrainingConfigError(
                    "expert must be 'high-noise' or 'low-noise' for Wan 2.2 14B training"
                )
            expert = cast("WanExpertName", expert_value)
            timestep_range = ranges[expert]
            supplied_range = raw.get("timestepRange")
            if supplied_range is not None and (
                not isinstance(supplied_range, list)
                or len(supplied_range) != 2
                or any(type(value) is not int for value in supplied_range)
                or supplied_range != list(timestep_range)
            ):
                raise TrainingConfigError(
                    f"timestepRange must be {list(timestep_range)} for variant "
                    f"{variant!r} expert {expert!r}"
                )
        else:
            if expert_value is not None or raw.get("timestepRange") is not None:
                raise TrainingConfigError(
                    "expert and timestepRange apply only to Wan 2.2 14B training"
                )
            expert = None
            timestep_range = None

        device = raw.get("device", "cpu")
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        base_dtype = raw.get("baseDtype", "bfloat16")
        if base_dtype not in ("float16", "bfloat16"):
            raise TrainingConfigError("baseDtype must be 'float16' or 'bfloat16'")
        target_value = raw.get("loraTargets", ["attention.qkvo", "ffn.projections"])
        if target_value != ["attention.qkvo", "ffn.projections"]:
            raise TrainingConfigError("loraTargets must be ['attention.qkvo', 'ffn.projections']")
        checkpointing = raw.get("gradientCheckpointing", True)
        if type(checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode = raw.get("checkpointingMode", "wholeModel")
        if checkpointing_mode not in ("blockNonReentrant", "wholeModel"):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel' for Wan training"
            )
        optimizer = raw.get("optimizer", "adamw")
        if optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        rng_policy = raw.get("rngPolicy", "sequential")
        if rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        betas_value = raw.get("betas", [0.9, 0.999])
        if not isinstance(betas_value, list) or len(betas_value) != 2:
            raise TrainingConfigError("betas must be a two-item number array")
        beta1 = _float(betas_value[0], "betas[0]", maximum=0.999999999)
        beta2 = _float(betas_value[1], "betas[1]", maximum=0.999999999)
        if beta1 == 0.0 or beta2 == 0.0:
            raise TrainingConfigError("betas entries must be greater than zero")

        latent = cast(
            "tuple[int, int, int, int, int]",
            _shape(raw.get("latentShape", [1, 16, 1, 32, 32]), "latentShape", 5),
        )
        context = cast(
            "tuple[int, int, int]",
            _shape(raw.get("contextShape", [1, 512, 4096]), "contextShape", 3),
        )
        latent_channels = 48 if variant == "wan22-ti2v-5b" else 16
        if latent[1] != latent_channels:
            raise TrainingConfigError(
                f"latentShape must be [batch, {latent_channels}, time, height, width]"
            )
        if context[0] != latent[0] or context[2] != 4096:
            raise TrainingConfigError("contextShape must be [batch, tokens, 4096]")

        dit_state = _artifact_source(raw.get("ditState"), "ditState")
        umt5xxl_state = _artifact_source(raw.get("umt5xxlState"), "umt5xxlState")
        vae_state = _artifact_source(raw.get("vaeState"), "vaeState")
        dataset = (
            None
            if raw.get("dataset") is None
            else _wan_dataset_settings(
                raw["dataset"],
                vae_state=vae_state,
                umt5xxl_state=umt5xxl_state,
                latent_shape=latent,
                context_shape=context,
                device=device,
                vae_contract=vae_contract,
                conditioning_contract=conditioning_contract,
            )
        )
        identity_value = raw.get("datasetIdentity")
        if dataset is None:
            dataset_identity = _digest(identity_value, "datasetIdentity")
        else:
            dataset_identity = dataset.digest
            if (
                identity_value is not None
                and _digest(identity_value, "datasetIdentity") != dataset.digest
            ):
                raise TrainingConfigError("datasetIdentity does not match dataset.digest")
            expected_latent = (
                latent[0],
                48 if vae_contract == "wan22" else 16,
                1 + (dataset.frame_count - 1) // 4,
                dataset.resolution[0] // (16 if vae_contract == "wan22" else 8),
                dataset.resolution[1] // (16 if vae_contract == "wan22" else 8),
            )
            if latent != expected_latent:
                raise TrainingConfigError(
                    f"latentShape must match encoded dataset geometry {list(expected_latent)}"
                )

        return cls(
            family="wan",
            variant=variant,
            dit_state=dit_state,
            umt5xxl_state=umt5xxl_state,
            vae_state=vae_state,
            dataset_identity=dataset_identity,
            dataset=dataset,
            distributed=(
                None
                if raw.get("distributed") is None
                else _distributed_settings(raw["distributed"])
            ),
            device=device,
            base_dtype=base_dtype,
            rank=_int(raw.get("rank", 4), "rank", minimum=1),
            alpha=_float(raw.get("alpha", 4.0), "alpha", minimum=0.000000001),
            lora_targets=("attention.qkvo", "ffn.projections"),
            learning_rate=_float(
                raw.get("learningRate", 0.0001), "learningRate", minimum=0.000000000001
            ),
            weight_decay=_float(raw.get("weightDecay", 0.01), "weightDecay"),
            beta1=beta1,
            beta2=beta2,
            epsilon=_float(raw.get("epsilon", 1e-8), "epsilon", minimum=1e-30),
            optimizer=optimizer,
            gradient_accumulation_steps=_int(
                raw.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
                minimum=1,
            ),
            gradient_checkpointing=checkpointing,
            checkpointing_mode=checkpointing_mode,
            checkpoint_interval=_int(raw.get("checkpointInterval", 0), "checkpointInterval"),
            lora_export_interval=_int(raw.get("loraExportInterval", 0), "loraExportInterval"),
            sync_digest_interval=_int(raw.get("syncDigestInterval", 0), "syncDigestInterval"),
            seed=_int(raw.get("seed", 0), "seed"),
            rng_policy=rng_policy,
            latent_shape=latent,
            context_shape=context,
            expert=expert,
            timestep_range=timestep_range,
        )

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schemaVersion": 1,
            "family": self.family,
            "variant": self.variant,
            "ditState": self.dit_state.to_mapping(),
            "umt5xxlState": self.umt5xxl_state.to_mapping(),
            "vaeState": self.vae_state.to_mapping(),
            "device": self.device,
            "baseDtype": self.base_dtype,
            "rank": self.rank,
            "alpha": self.alpha,
            "loraTargets": list(self.lora_targets),
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "betas": [self.beta1, self.beta2],
            "epsilon": self.epsilon,
            "optimizer": self.optimizer,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "gradientCheckpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "latentShape": list(self.latent_shape),
            "contextShape": list(self.context_shape),
        }
        if self.dataset is None:
            mapping["datasetIdentity"] = self.dataset_identity
        else:
            mapping["dataset"] = self.dataset.to_mapping()
        if self.checkpoint_interval:
            mapping["checkpointInterval"] = self.checkpoint_interval
        if self.lora_export_interval:
            mapping["loraExportInterval"] = self.lora_export_interval
        if self.sync_digest_interval:
            mapping["syncDigestInterval"] = self.sync_digest_interval
        if self.distributed is not None:
            mapping["distributed"] = self.distributed.to_mapping()
        if self.rng_policy != "sequential":
            mapping["rngPolicy"] = self.rng_policy
        if self.checkpointing_mode != "wholeModel":
            mapping["checkpointingMode"] = self.checkpointing_mode
        if self.expert is not None:
            mapping["expert"] = self.expert
            assert self.timestep_range is not None
            mapping["timestepRange"] = list(self.timestep_range)
        return mapping

    def identity_mapping(self) -> dict[str, object]:
        mapping = self.to_mapping()
        for field in _COMMON_RUNTIME_ONLY_FIELDS:
            mapping.pop(field, None)
        for field in ("ditState", "umt5xxlState", "vaeState"):
            artifact = mapping[field]
            assert isinstance(artifact, dict)
            artifact.pop("path")
        dataset = mapping.get("dataset")
        if isinstance(dataset, dict):
            for field in _DATASET_RUNTIME_ONLY_FIELDS:
                dataset.pop(field, None)
        distributed = mapping.get("distributed")
        if isinstance(distributed, dict):
            for field in _DISTRIBUTED_RUNTIME_ONLY_FIELDS:
                distributed.pop(field, None)
        return mapping


@dataclass(frozen=True)
class FluxTrainingConfig:
    """Normalized behavior and component identity for classic Flux LoRA training."""

    family: Literal["flux"]
    variant: FluxVariantName
    dit_state: TrainingArtifactSource
    clip_l_state: TrainingArtifactSource
    t5xxl_state: TrainingArtifactSource
    vae_state: TrainingArtifactSource
    dataset_identity: str
    dataset: FluxDatasetSettings | None
    distributed: DistributedSettings | None
    device: str
    base_dtype: FluxBaseDtypeName
    rank: int
    alpha: float
    lora_targets: tuple[FluxLoraTargetName, FluxLoraTargetName]
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    optimizer: OptimizerName
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    checkpointing_mode: Literal["blockNonReentrant", "wholeModel"]
    checkpoint_interval: int
    lora_export_interval: int
    sync_digest_interval: int
    seed: int
    rng_policy: RngPolicyName
    latent_shape: tuple[int, int, int, int]
    context_shape: tuple[int, int, int]
    pooled_shape: tuple[int, int]
    guidance: float | None

    def __post_init__(self) -> None:
        if cast("object", self.family) != "flux":
            raise TrainingConfigError("family must be 'flux'")
        variant = cast("object", self.variant)
        if not isinstance(variant, str) or variant not in _FLUX_VARIANTS:
            raise TrainingConfigError("unsupported Flux training variant")
        for name, artifact in (
            ("ditState", self.dit_state),
            ("clipLState", self.clip_l_state),
            ("t5xxlState", self.t5xxl_state),
            ("vaeState", self.vae_state),
        ):
            artifact_value = cast("object", artifact)
            if not isinstance(artifact_value, TrainingArtifactSource):
                raise TrainingConfigError(f"{name} must be a normalized training artifact")
            path = cast("object", artifact_value.path)
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise TrainingConfigError(f"{name}.path must be a non-empty absolute path")
            _digest(cast("object", artifact_value.digest), f"{name}.digest")
            _int(cast("object", artifact_value.size), f"{name}.size")
        _digest(cast("object", self.dataset_identity), "datasetIdentity")
        distributed = cast("object", self.distributed)
        if distributed is not None and not isinstance(distributed, DistributedSettings):
            raise TrainingConfigError("distributed must be normalized settings or None")
        device = cast("object", self.device)
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        base_dtype = cast("object", self.base_dtype)
        if not isinstance(base_dtype, str) or base_dtype not in ("float16", "bfloat16"):
            raise TrainingConfigError("baseDtype must be 'float16' or 'bfloat16'")
        _int(cast("object", self.rank), "rank", minimum=1)
        _normalized_float(cast("object", self.alpha), "alpha", minimum=0.000000001)
        if type(self.lora_targets) is not tuple or self.lora_targets != (
            "attention.qkv_proj",
            "mlp.projections",
        ):
            raise TrainingConfigError(
                "loraTargets must be ['attention.qkv_proj', 'mlp.projections']"
            )
        _normalized_float(self.learning_rate, "learningRate", minimum=0.000000000001)
        _normalized_float(self.weight_decay, "weightDecay")
        _normalized_float(self.beta1, "betas[0]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.beta2, "betas[1]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.epsilon, "epsilon", minimum=1e-30)
        optimizer = cast("object", self.optimizer)
        if not isinstance(optimizer, str) or optimizer not in (
            "adamw",
            "factored-adamw",
        ):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        _int(self.gradient_accumulation_steps, "gradientAccumulationSteps", minimum=1)
        if type(self.gradient_checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode = cast("object", self.checkpointing_mode)
        if not isinstance(checkpointing_mode, str) or checkpointing_mode not in (
            "blockNonReentrant",
            "wholeModel",
        ):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel' for Flux training"
            )
        _int(self.checkpoint_interval, "checkpointInterval")
        _int(self.lora_export_interval, "loraExportInterval")
        _int(self.sync_digest_interval, "syncDigestInterval")
        _int(self.seed, "seed")
        rng_policy = cast("object", self.rng_policy)
        if not isinstance(rng_policy, str) or rng_policy not in (
            "sequential",
            "counter",
        ):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        if (
            type(self.latent_shape) is not tuple
            or len(self.latent_shape) != 4
            or any(type(value) is not int or value < 1 for value in self.latent_shape)
            or self.latent_shape[1] != 16
            or self.latent_shape[2] % 2 != 0
            or self.latent_shape[3] % 2 != 0
        ):
            raise TrainingConfigError("latentShape must be [batch, 16, even height, even width]")
        if (
            type(self.context_shape) is not tuple
            or len(self.context_shape) != 3
            or any(type(value) is not int or value < 1 for value in self.context_shape)
            or self.context_shape[0] != self.latent_shape[0]
            or self.context_shape[2] != 4096
        ):
            raise TrainingConfigError("contextShape must be [batch, tokens, 4096]")
        if (
            type(self.pooled_shape) is not tuple
            or len(self.pooled_shape) != 2
            or any(type(value) is not int or value < 1 for value in self.pooled_shape)
            or self.pooled_shape != (self.latent_shape[0], 768)
        ):
            raise TrainingConfigError("pooledShape must be [batch, 768]")
        if self.dataset is not None:
            dataset = cast("object", self.dataset)
            if not isinstance(dataset, FluxDatasetSettings):
                raise TrainingConfigError("dataset must be normalized Flux dataset settings")
            root = cast("object", dataset.root)
            cache_root = cast("object", dataset.encoded_cache_root)
            if (
                not isinstance(root, str)
                or not root
                or not Path(root).is_absolute()
                or type(dataset.resolution) is not tuple
                or len(dataset.resolution) != 2
                or any(type(value) is not int or value < 1 for value in dataset.resolution)
                or any(value % 16 for value in dataset.resolution)
                or type(dataset.context_tokens) is not int
                or dataset.context_tokens < 256
                or dataset.context_tokens != self.context_shape[1]
                or dataset.variant != self.variant
                or not isinstance(cast("object", dataset.inspection), DatasetInspection)
                or (
                    cache_root is not None
                    and (
                        not isinstance(cache_root, str)
                        or not cache_root
                        or not Path(cache_root).is_absolute()
                    )
                )
            ):
                raise TrainingConfigError("dataset must contain normalized Flux dataset settings")
            if dataset.digest != self.dataset_identity:
                raise TrainingConfigError("datasetIdentity does not match dataset.digest")
            expected_latent = (
                self.latent_shape[0],
                16,
                dataset.resolution[0] // 8,
                dataset.resolution[1] // 8,
            )
            if self.latent_shape != expected_latent:
                raise TrainingConfigError(
                    f"latentShape must match encoded dataset geometry {list(expected_latent)}"
                )
        if self.variant == "flux1-dev":
            if self.guidance is None:
                raise TrainingConfigError("flux1-dev requires guidance")
            _normalized_float(self.guidance, "guidance")
        elif self.guidance is not None:
            raise TrainingConfigError("flux1-schnell does not accept guidance")

    @classmethod
    def parse(cls, serialized: str) -> FluxTrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"config must be valid JSON: {exc.msg}") from exc
        return cls.from_mapping(_object(value, "config"))

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> FluxTrainingConfig:
        fields = {
            "schemaVersion",
            "family",
            "variant",
            "ditState",
            "clipLState",
            "t5xxlState",
            "vaeState",
            "datasetIdentity",
            "dataset",
            "distributed",
            "device",
            "baseDtype",
            "rank",
            "alpha",
            "loraTargets",
            "learningRate",
            "weightDecay",
            "betas",
            "epsilon",
            "optimizer",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "checkpointingMode",
            "checkpointInterval",
            "loraExportInterval",
            "syncDigestInterval",
            "seed",
            "rngPolicy",
            "latentShape",
            "contextShape",
            "pooledShape",
            "guidance",
        }
        _known(raw, fields, "config")
        schema_version = _int(raw.get("schemaVersion", 1), "schemaVersion", minimum=1)
        if schema_version != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        if raw.get("family") != "flux":
            raise TrainingConfigError("family must be 'flux'")
        variant_value = raw.get("variant")
        if not isinstance(variant_value, str) or variant_value not in _FLUX_VARIANTS:
            raise TrainingConfigError("unsupported Flux training variant")
        variant = variant_value
        base_dtype_value = raw.get("baseDtype", "bfloat16")
        if not isinstance(base_dtype_value, str) or base_dtype_value not in (
            "float16",
            "bfloat16",
        ):
            raise TrainingConfigError("baseDtype must be 'float16' or 'bfloat16'")
        base_dtype = base_dtype_value
        targets = raw.get("loraTargets", ["attention.qkv_proj", "mlp.projections"])
        if (
            not isinstance(targets, list)
            or len(targets) != 2
            or any(not isinstance(target, str) for target in targets)
            or targets != ["attention.qkv_proj", "mlp.projections"]
        ):
            raise TrainingConfigError(
                "loraTargets must be ['attention.qkv_proj', 'mlp.projections']"
            )
        checkpointing = raw.get("gradientCheckpointing", True)
        if type(checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode_value = raw.get("checkpointingMode", "wholeModel")
        if not isinstance(checkpointing_mode_value, str) or checkpointing_mode_value not in (
            "blockNonReentrant",
            "wholeModel",
        ):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel' for Flux training"
            )
        checkpointing_mode = checkpointing_mode_value
        optimizer_value = raw.get("optimizer", "adamw")
        if not isinstance(optimizer_value, str) or optimizer_value not in (
            "adamw",
            "factored-adamw",
        ):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        optimizer = optimizer_value
        rng_value = raw.get("rngPolicy", "sequential")
        if not isinstance(rng_value, str) or rng_value not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        rng_policy = rng_value
        betas = raw.get("betas", [0.9, 0.999])
        if not isinstance(betas, list) or len(betas) != 2:
            raise TrainingConfigError("betas must be a two-item number array")
        beta1 = _float(betas[0], "betas[0]", minimum=0.000000001, maximum=0.999999999)
        beta2 = _float(betas[1], "betas[1]", minimum=0.000000001, maximum=0.999999999)
        latent = cast(
            "tuple[int, int, int, int]",
            _shape(raw.get("latentShape", [1, 16, 32, 32]), "latentShape", 4),
        )
        context = cast(
            "tuple[int, int, int]",
            _shape(raw.get("contextShape", [1, 512, 4096]), "contextShape", 3),
        )
        pooled = cast(
            "tuple[int, int]",
            _shape(raw.get("pooledShape", [1, 768]), "pooledShape", 2),
        )
        if variant == "flux1-dev":
            guidance = _float(raw["guidance"] if "guidance" in raw else 3.5, "guidance")
        else:
            if "guidance" in raw:
                raise TrainingConfigError("flux1-schnell does not accept guidance")
            guidance = None
        device_value = raw.get("device", "cpu")
        if not isinstance(device_value, str) or not device_value:
            raise TrainingConfigError("device must be a non-empty string")

        dit_state = _artifact_source(raw.get("ditState"), "ditState")
        clip_l_state = _artifact_source(raw.get("clipLState"), "clipLState")
        t5xxl_state = _artifact_source(raw.get("t5xxlState"), "t5xxlState")
        vae_state = _artifact_source(raw.get("vaeState"), "vaeState")
        dataset = (
            None
            if raw.get("dataset") is None
            else _flux_dataset_settings(
                raw["dataset"],
                vae_state=vae_state,
                clip_l_state=clip_l_state,
                t5xxl_state=t5xxl_state,
                context_tokens=context[1],
                variant=variant,
            )
        )
        identity_value = raw.get("datasetIdentity")
        if dataset is None:
            dataset_identity = _digest(identity_value, "datasetIdentity")
        else:
            dataset_identity = dataset.digest
            if (
                identity_value is not None
                and _digest(identity_value, "datasetIdentity") != dataset.digest
            ):
                raise TrainingConfigError("datasetIdentity does not match dataset.digest")

        return cls(
            family="flux",
            variant=variant,
            dit_state=dit_state,
            clip_l_state=clip_l_state,
            t5xxl_state=t5xxl_state,
            vae_state=vae_state,
            dataset_identity=dataset_identity,
            dataset=dataset,
            distributed=(
                None
                if raw.get("distributed") is None
                else _distributed_settings(raw["distributed"])
            ),
            device=device_value,
            base_dtype=base_dtype,
            rank=_int(raw.get("rank", 4), "rank", minimum=1),
            alpha=_float(raw.get("alpha", 4.0), "alpha", minimum=0.000000001),
            lora_targets=("attention.qkv_proj", "mlp.projections"),
            learning_rate=_float(
                raw.get("learningRate", 0.0001), "learningRate", minimum=0.000000000001
            ),
            weight_decay=_float(raw.get("weightDecay", 0.01), "weightDecay"),
            beta1=beta1,
            beta2=beta2,
            epsilon=_float(raw.get("epsilon", 1e-8), "epsilon", minimum=1e-30),
            optimizer=optimizer,
            gradient_accumulation_steps=_int(
                raw.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
                minimum=1,
            ),
            gradient_checkpointing=checkpointing,
            checkpointing_mode=checkpointing_mode,
            checkpoint_interval=_int(raw.get("checkpointInterval", 0), "checkpointInterval"),
            lora_export_interval=_int(raw.get("loraExportInterval", 0), "loraExportInterval"),
            sync_digest_interval=_int(raw.get("syncDigestInterval", 0), "syncDigestInterval"),
            seed=_int(raw.get("seed", 0), "seed"),
            rng_policy=rng_policy,
            latent_shape=latent,
            context_shape=context,
            pooled_shape=pooled,
            guidance=guidance,
        )

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schemaVersion": 1,
            "family": self.family,
            "variant": self.variant,
            "ditState": self.dit_state.to_mapping(),
            "clipLState": self.clip_l_state.to_mapping(),
            "t5xxlState": self.t5xxl_state.to_mapping(),
            "vaeState": self.vae_state.to_mapping(),
            "device": self.device,
            "baseDtype": self.base_dtype,
            "rank": self.rank,
            "alpha": self.alpha,
            "loraTargets": list(self.lora_targets),
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "betas": [self.beta1, self.beta2],
            "epsilon": self.epsilon,
            "optimizer": self.optimizer,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "gradientCheckpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "latentShape": list(self.latent_shape),
            "contextShape": list(self.context_shape),
            "pooledShape": list(self.pooled_shape),
        }
        if self.dataset is None:
            mapping["datasetIdentity"] = self.dataset_identity
        else:
            mapping["dataset"] = self.dataset.to_mapping()
        if self.guidance is not None:
            mapping["guidance"] = self.guidance
        if self.checkpoint_interval:
            mapping["checkpointInterval"] = self.checkpoint_interval
        if self.lora_export_interval:
            mapping["loraExportInterval"] = self.lora_export_interval
        if self.sync_digest_interval:
            mapping["syncDigestInterval"] = self.sync_digest_interval
        if self.distributed is not None:
            mapping["distributed"] = self.distributed.to_mapping()
        if self.rng_policy != "sequential":
            mapping["rngPolicy"] = self.rng_policy
        if self.checkpointing_mode != "wholeModel":
            mapping["checkpointingMode"] = self.checkpointing_mode
        return mapping

    def identity_mapping(self) -> dict[str, object]:
        mapping = self.to_mapping()
        for field in _COMMON_RUNTIME_ONLY_FIELDS:
            mapping.pop(field, None)
        for field in ("ditState", "clipLState", "t5xxlState", "vaeState"):
            artifact = mapping[field]
            assert isinstance(artifact, dict)
            artifact.pop("path")
        distributed = mapping.get("distributed")
        if isinstance(distributed, dict):
            for field in _DISTRIBUTED_RUNTIME_ONLY_FIELDS:
                distributed.pop(field, None)
        dataset = mapping.get("dataset")
        if isinstance(dataset, dict):
            for field in _DATASET_RUNTIME_ONLY_FIELDS:
                dataset.pop(field, None)
        return mapping


@dataclass(frozen=True)
class Flux2TrainingConfig:
    """Normalized behavior and component identity for Flux2 LoRA training."""

    family: Literal["flux2"]
    variant: Flux2VariantName
    dit_state: TrainingArtifactSource
    text_encoder_state: TrainingArtifactSource
    vae_state: TrainingArtifactSource
    dataset_identity: str
    dataset: Flux2DatasetSettings | None
    distributed: DistributedSettings | None
    device: str
    base_dtype: FluxBaseDtypeName
    rank: int
    alpha: float
    lora_targets: tuple[FluxLoraTargetName, FluxLoraTargetName]
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    optimizer: OptimizerName
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    checkpointing_mode: Literal["blockNonReentrant", "wholeModel"]
    checkpoint_interval: int
    lora_export_interval: int
    sync_digest_interval: int
    seed: int
    rng_policy: RngPolicyName
    latent_shape: tuple[int, int, int, int]
    context_shape: tuple[int, int, int]
    guidance: float | None

    def __post_init__(self) -> None:
        if cast("object", self.family) != "flux2":
            raise TrainingConfigError("family must be 'flux2'")
        variant = cast("object", self.variant)
        if not isinstance(variant, str) or variant not in _FLUX2_VARIANTS:
            raise TrainingConfigError("unsupported Flux2 training variant")
        for name, artifact in (
            ("ditState", self.dit_state),
            ("textEncoderState", self.text_encoder_state),
            ("vaeState", self.vae_state),
        ):
            artifact_value = cast("object", artifact)
            if not isinstance(artifact_value, TrainingArtifactSource):
                raise TrainingConfigError(f"{name} must be a normalized training artifact")
            path = cast("object", artifact_value.path)
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise TrainingConfigError(f"{name}.path must be a non-empty absolute path")
            _digest(cast("object", artifact_value.digest), f"{name}.digest")
            _int(cast("object", artifact_value.size), f"{name}.size")
        _digest(cast("object", self.dataset_identity), "datasetIdentity")
        distributed = cast("object", self.distributed)
        if distributed is not None and not isinstance(distributed, DistributedSettings):
            raise TrainingConfigError("distributed must be normalized settings or None")
        device = cast("object", self.device)
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        base_dtype = cast("object", self.base_dtype)
        if not isinstance(base_dtype, str) or base_dtype not in ("float16", "bfloat16"):
            raise TrainingConfigError("baseDtype must be 'float16' or 'bfloat16'")
        _int(cast("object", self.rank), "rank", minimum=1)
        _normalized_float(cast("object", self.alpha), "alpha", minimum=0.000000001)
        if type(self.lora_targets) is not tuple or self.lora_targets != (
            "attention.qkv_proj",
            "mlp.projections",
        ):
            raise TrainingConfigError(
                "loraTargets must be ['attention.qkv_proj', 'mlp.projections']"
            )
        _normalized_float(self.learning_rate, "learningRate", minimum=0.000000000001)
        _normalized_float(self.weight_decay, "weightDecay")
        _normalized_float(self.beta1, "betas[0]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.beta2, "betas[1]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.epsilon, "epsilon", minimum=1e-30)
        optimizer = cast("object", self.optimizer)
        if not isinstance(optimizer, str) or optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        _int(self.gradient_accumulation_steps, "gradientAccumulationSteps", minimum=1)
        if type(self.gradient_checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode = cast("object", self.checkpointing_mode)
        if not isinstance(checkpointing_mode, str) or checkpointing_mode not in (
            "blockNonReentrant",
            "wholeModel",
        ):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel' for Flux2 training"
            )
        _int(self.checkpoint_interval, "checkpointInterval")
        _int(self.lora_export_interval, "loraExportInterval")
        _int(self.sync_digest_interval, "syncDigestInterval")
        _int(self.seed, "seed")
        rng_policy = cast("object", self.rng_policy)
        if not isinstance(rng_policy, str) or rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        if (
            type(self.latent_shape) is not tuple
            or len(self.latent_shape) != 4
            or any(type(value) is not int or value < 1 for value in self.latent_shape)
            or self.latent_shape[1] != 128
        ):
            raise TrainingConfigError("latentShape must be [batch, 128, height, width]")
        expected_context_width = _FLUX2_CONTEXT_WIDTHS[self.variant]
        if (
            type(self.context_shape) is not tuple
            or len(self.context_shape) != 3
            or any(type(value) is not int or value < 1 for value in self.context_shape)
            or self.context_shape[0] != self.latent_shape[0]
            or self.context_shape[1] < 512
            or self.context_shape[2] != expected_context_width
        ):
            raise TrainingConfigError(
                "contextShape must be"
                f" [batch, at least 512 tokens, {expected_context_width}] for {self.variant}"
            )
        if self.dataset is not None:
            dataset = cast("object", self.dataset)
            if not isinstance(dataset, Flux2DatasetSettings):
                raise TrainingConfigError("dataset must be normalized Flux2 dataset settings")
            root = cast("object", dataset.root)
            cache_root = cast("object", dataset.encoded_cache_root)
            if (
                not isinstance(root, str)
                or not root
                or not Path(root).is_absolute()
                or type(dataset.resolution) is not tuple
                or len(dataset.resolution) != 2
                or any(type(value) is not int or value < 1 for value in dataset.resolution)
                or any(value % 16 for value in dataset.resolution)
                or type(dataset.context_tokens) is not int
                or dataset.context_tokens < 512
                or dataset.context_tokens != self.context_shape[1]
                or dataset.variant != self.variant
                or not isinstance(cast("object", dataset.inspection), DatasetInspection)
                or (
                    cache_root is not None
                    and (
                        not isinstance(cache_root, str)
                        or not cache_root
                        or not Path(cache_root).is_absolute()
                    )
                )
            ):
                raise TrainingConfigError("dataset must contain normalized Flux2 dataset settings")
            if dataset.digest != self.dataset_identity:
                raise TrainingConfigError("datasetIdentity does not match dataset.digest")
            expected_latent = (
                self.latent_shape[0],
                128,
                dataset.resolution[0] // 16,
                dataset.resolution[1] // 16,
            )
            if self.latent_shape != expected_latent:
                raise TrainingConfigError(
                    f"latentShape must match encoded dataset geometry {list(expected_latent)}"
                )
        if self.variant == "flux2-dev":
            if self.guidance is None:
                raise TrainingConfigError("flux2-dev requires guidance")
            _normalized_float(self.guidance, "guidance")
        elif self.guidance is not None:
            raise TrainingConfigError(f"{self.variant} does not accept guidance")

    @classmethod
    def parse(cls, serialized: str) -> Flux2TrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"config must be valid JSON: {exc.msg}") from exc
        return cls.from_mapping(_object(value, "config"))

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> Flux2TrainingConfig:
        fields = {
            "schemaVersion",
            "family",
            "variant",
            "ditState",
            "textEncoderState",
            "vaeState",
            "datasetIdentity",
            "dataset",
            "distributed",
            "device",
            "baseDtype",
            "rank",
            "alpha",
            "loraTargets",
            "learningRate",
            "weightDecay",
            "betas",
            "epsilon",
            "optimizer",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "checkpointingMode",
            "checkpointInterval",
            "loraExportInterval",
            "syncDigestInterval",
            "seed",
            "rngPolicy",
            "latentShape",
            "contextShape",
            "guidance",
        }
        _known(raw, fields, "config")
        schema_version = _int(raw.get("schemaVersion", 1), "schemaVersion", minimum=1)
        if schema_version != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        if raw.get("family") != "flux2":
            raise TrainingConfigError("family must be 'flux2'")
        variant_value = raw.get("variant")
        if not isinstance(variant_value, str) or variant_value not in _FLUX2_VARIANTS:
            raise TrainingConfigError("unsupported Flux2 training variant")
        variant = variant_value
        base_dtype_value = raw.get("baseDtype", "bfloat16")
        if not isinstance(base_dtype_value, str) or base_dtype_value not in (
            "float16",
            "bfloat16",
        ):
            raise TrainingConfigError("baseDtype must be 'float16' or 'bfloat16'")
        base_dtype = base_dtype_value
        targets = raw.get("loraTargets", ["attention.qkv_proj", "mlp.projections"])
        if (
            not isinstance(targets, list)
            or len(targets) != 2
            or any(not isinstance(target, str) for target in targets)
            or targets != ["attention.qkv_proj", "mlp.projections"]
        ):
            raise TrainingConfigError(
                "loraTargets must be ['attention.qkv_proj', 'mlp.projections']"
            )
        checkpointing = raw.get("gradientCheckpointing", True)
        if type(checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode_value = raw.get("checkpointingMode", "wholeModel")
        if not isinstance(checkpointing_mode_value, str) or checkpointing_mode_value not in (
            "blockNonReentrant",
            "wholeModel",
        ):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel' for Flux2 training"
            )
        checkpointing_mode = checkpointing_mode_value
        optimizer_value = raw.get("optimizer", "adamw")
        if not isinstance(optimizer_value, str) or optimizer_value not in (
            "adamw",
            "factored-adamw",
        ):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        optimizer = optimizer_value
        rng_value = raw.get("rngPolicy", "sequential")
        if not isinstance(rng_value, str) or rng_value not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        rng_policy = rng_value
        betas = raw.get("betas", [0.9, 0.999])
        if not isinstance(betas, list) or len(betas) != 2:
            raise TrainingConfigError("betas must be a two-item number array")
        beta1 = _float(betas[0], "betas[0]", minimum=0.000000001, maximum=0.999999999)
        beta2 = _float(betas[1], "betas[1]", minimum=0.000000001, maximum=0.999999999)
        latent = cast(
            "tuple[int, int, int, int]",
            _shape(raw.get("latentShape", [1, 128, 32, 32]), "latentShape", 4),
        )
        context = cast(
            "tuple[int, int, int]",
            _shape(
                raw.get("contextShape", [1, 512, _FLUX2_CONTEXT_WIDTHS[variant]]),
                "contextShape",
                3,
            ),
        )
        if variant == "flux2-dev":
            guidance = _float(raw["guidance"] if "guidance" in raw else 3.5, "guidance")
        else:
            if "guidance" in raw:
                raise TrainingConfigError(f"{variant} does not accept guidance")
            guidance = None
        device_value = raw.get("device", "cpu")
        if not isinstance(device_value, str) or not device_value:
            raise TrainingConfigError("device must be a non-empty string")

        dit_state = _artifact_source(raw.get("ditState"), "ditState")
        text_encoder_state = _artifact_source(raw.get("textEncoderState"), "textEncoderState")
        vae_state = _artifact_source(raw.get("vaeState"), "vaeState")
        dataset = (
            None
            if raw.get("dataset") is None
            else _flux2_dataset_settings(
                raw["dataset"],
                vae_state=vae_state,
                text_encoder_state=text_encoder_state,
                context_tokens=context[1],
                variant=variant,
            )
        )
        identity_value = raw.get("datasetIdentity")
        if dataset is None:
            dataset_identity = _digest(identity_value, "datasetIdentity")
        else:
            dataset_identity = dataset.digest
            if (
                identity_value is not None
                and _digest(identity_value, "datasetIdentity") != dataset.digest
            ):
                raise TrainingConfigError("datasetIdentity does not match dataset.digest")

        return cls(
            family="flux2",
            variant=variant,
            dit_state=dit_state,
            text_encoder_state=text_encoder_state,
            vae_state=vae_state,
            dataset_identity=dataset_identity,
            dataset=dataset,
            distributed=(
                None
                if raw.get("distributed") is None
                else _distributed_settings(raw["distributed"])
            ),
            device=device_value,
            base_dtype=base_dtype,
            rank=_int(raw.get("rank", 4), "rank", minimum=1),
            alpha=_float(raw.get("alpha", 4.0), "alpha", minimum=0.000000001),
            lora_targets=("attention.qkv_proj", "mlp.projections"),
            learning_rate=_float(
                raw.get("learningRate", 0.0001), "learningRate", minimum=0.000000000001
            ),
            weight_decay=_float(raw.get("weightDecay", 0.01), "weightDecay"),
            beta1=beta1,
            beta2=beta2,
            epsilon=_float(raw.get("epsilon", 1e-8), "epsilon", minimum=1e-30),
            optimizer=optimizer,
            gradient_accumulation_steps=_int(
                raw.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
                minimum=1,
            ),
            gradient_checkpointing=checkpointing,
            checkpointing_mode=checkpointing_mode,
            checkpoint_interval=_int(raw.get("checkpointInterval", 0), "checkpointInterval"),
            lora_export_interval=_int(raw.get("loraExportInterval", 0), "loraExportInterval"),
            sync_digest_interval=_int(raw.get("syncDigestInterval", 0), "syncDigestInterval"),
            seed=_int(raw.get("seed", 0), "seed"),
            rng_policy=rng_policy,
            latent_shape=latent,
            context_shape=context,
            guidance=guidance,
        )

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schemaVersion": 1,
            "family": self.family,
            "variant": self.variant,
            "ditState": self.dit_state.to_mapping(),
            "textEncoderState": self.text_encoder_state.to_mapping(),
            "vaeState": self.vae_state.to_mapping(),
            "device": self.device,
            "baseDtype": self.base_dtype,
            "rank": self.rank,
            "alpha": self.alpha,
            "loraTargets": list(self.lora_targets),
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "betas": [self.beta1, self.beta2],
            "epsilon": self.epsilon,
            "optimizer": self.optimizer,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "gradientCheckpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "latentShape": list(self.latent_shape),
            "contextShape": list(self.context_shape),
        }
        if self.dataset is None:
            mapping["datasetIdentity"] = self.dataset_identity
        else:
            mapping["dataset"] = self.dataset.to_mapping()
        if self.guidance is not None:
            mapping["guidance"] = self.guidance
        if self.checkpoint_interval:
            mapping["checkpointInterval"] = self.checkpoint_interval
        if self.lora_export_interval:
            mapping["loraExportInterval"] = self.lora_export_interval
        if self.sync_digest_interval:
            mapping["syncDigestInterval"] = self.sync_digest_interval
        if self.distributed is not None:
            mapping["distributed"] = self.distributed.to_mapping()
        if self.rng_policy != "sequential":
            mapping["rngPolicy"] = self.rng_policy
        if self.checkpointing_mode != "wholeModel":
            mapping["checkpointingMode"] = self.checkpointing_mode
        return mapping

    def identity_mapping(self) -> dict[str, object]:
        mapping = self.to_mapping()
        for field in _COMMON_RUNTIME_ONLY_FIELDS:
            mapping.pop(field, None)
        for field in ("ditState", "textEncoderState", "vaeState"):
            artifact = mapping[field]
            assert isinstance(artifact, dict)
            artifact.pop("path")
        distributed = mapping.get("distributed")
        if isinstance(distributed, dict):
            for field in _DISTRIBUTED_RUNTIME_ONLY_FIELDS:
                distributed.pop(field, None)
        dataset = mapping.get("dataset")
        if isinstance(dataset, dict):
            for field in _DATASET_RUNTIME_ONLY_FIELDS:
                dataset.pop(field, None)
        return mapping


@dataclass(frozen=True)
class QwenImageTrainingConfig:
    """Normalized behavior and component identity for Qwen-Image LoRA training."""

    family: Literal["qwen-image"]
    variant: Literal["qwen-image"]
    dit_state: TrainingArtifactSource
    text_encoder_state: TrainingArtifactSource
    vae_state: TrainingArtifactSource
    dataset_identity: str
    dataset: QwenImageDatasetSettings | None
    distributed: DistributedSettings | None
    device: str
    base_dtype: QwenImageBaseDtypeName
    rank: int
    alpha: float
    lora_targets: tuple[QwenImageLoraTargetName, QwenImageLoraTargetName]
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    optimizer: OptimizerName
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    checkpointing_mode: Literal["blockNonReentrant", "wholeModel"]
    checkpoint_interval: int
    lora_export_interval: int
    sync_digest_interval: int
    seed: int
    rng_policy: RngPolicyName
    latent_shape: tuple[int, int, int, int, int]
    context_shape: tuple[int, int, int]
    attention_mask_shape: tuple[int, int]
    guidance: None

    def __post_init__(self) -> None:
        family = cast("object", self.family)
        if not isinstance(family, str) or family != "qwen-image":
            raise TrainingConfigError("family must be 'qwen-image'")
        variant = cast("object", self.variant)
        if not isinstance(variant, str) or variant not in _QWEN_IMAGE_VARIANTS:
            raise TrainingConfigError("unsupported Qwen-Image training variant")
        for name, artifact in (
            ("ditState", self.dit_state),
            ("textEncoderState", self.text_encoder_state),
            ("vaeState", self.vae_state),
        ):
            artifact_value = cast("object", artifact)
            if not isinstance(artifact_value, TrainingArtifactSource):
                raise TrainingConfigError(f"{name} must be a normalized training artifact")
            path = cast("object", artifact_value.path)
            if not isinstance(path, str) or not path or not Path(path).is_absolute():
                raise TrainingConfigError(f"{name}.path must be a non-empty absolute path")
            _digest(cast("object", artifact_value.digest), f"{name}.digest")
            _int(cast("object", artifact_value.size), f"{name}.size")
        dataset = cast("object", self.dataset)
        if dataset is not None and type(dataset) is not QwenImageDatasetSettings:
            raise TrainingConfigError("dataset must be normalized Qwen-Image settings or None")
        dataset_identity = _digest(cast("object", self.dataset_identity), "datasetIdentity")
        if isinstance(dataset, QwenImageDatasetSettings) and dataset.digest != dataset_identity:
            raise TrainingConfigError("datasetIdentity does not match dataset.digest")
        distributed = cast("object", self.distributed)
        if distributed is not None and not isinstance(distributed, DistributedSettings):
            raise TrainingConfigError("distributed must be normalized settings or None")
        device = cast("object", self.device)
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        base_dtype = cast("object", self.base_dtype)
        if not isinstance(base_dtype, str) or base_dtype not in ("bfloat16", "float32"):
            raise TrainingConfigError("baseDtype must be 'bfloat16' or 'float32'")
        _int(cast("object", self.rank), "rank", minimum=1)
        _normalized_float(cast("object", self.alpha), "alpha", minimum=0.000000001)
        if type(self.lora_targets) is not tuple or self.lora_targets != (
            "attention.qkvo",
            "mlp.projections",
        ):
            raise TrainingConfigError("loraTargets must be ['attention.qkvo', 'mlp.projections']")
        _normalized_float(self.learning_rate, "learningRate", minimum=0.000000000001)
        _normalized_float(self.weight_decay, "weightDecay")
        _normalized_float(self.beta1, "betas[0]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.beta2, "betas[1]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.epsilon, "epsilon", minimum=1e-30)
        optimizer = cast("object", self.optimizer)
        if not isinstance(optimizer, str) or optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        _int(self.gradient_accumulation_steps, "gradientAccumulationSteps", minimum=1)
        if type(self.gradient_checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode = cast("object", self.checkpointing_mode)
        if not isinstance(checkpointing_mode, str) or checkpointing_mode not in (
            "blockNonReentrant",
            "wholeModel",
        ):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel'"
                " for Qwen-Image training"
            )
        _int(self.checkpoint_interval, "checkpointInterval")
        _int(self.lora_export_interval, "loraExportInterval")
        _int(self.sync_digest_interval, "syncDigestInterval")
        _int(self.seed, "seed")
        rng_policy = cast("object", self.rng_policy)
        if not isinstance(rng_policy, str) or rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        if (
            type(self.latent_shape) is not tuple
            or len(self.latent_shape) != 5
            or any(type(value) is not int or value < 1 for value in self.latent_shape)
            or self.latent_shape[1] != 16
            or self.latent_shape[2] != 1
            or self.latent_shape[3] % 2
            or self.latent_shape[4] % 2
        ):
            raise TrainingConfigError("latentShape must be [batch, 16, 1, even height, even width]")
        if (
            type(self.context_shape) is not tuple
            or len(self.context_shape) != 3
            or any(type(value) is not int or value < 1 for value in self.context_shape)
            or self.context_shape[0] != self.latent_shape[0]
            or self.context_shape[1] < 512
            or self.context_shape[2] != 3584
        ):
            raise TrainingConfigError("contextShape must be [batch, at least 512 tokens, 3584]")
        if (
            type(self.attention_mask_shape) is not tuple
            or len(self.attention_mask_shape) != 2
            or any(type(value) is not int or value < 1 for value in self.attention_mask_shape)
            or self.attention_mask_shape != self.context_shape[:2]
        ):
            raise TrainingConfigError(
                "attentionMaskShape must match contextShape batch and token dimensions"
            )
        if isinstance(dataset, QwenImageDatasetSettings):
            expected_latent = (
                self.latent_shape[0],
                16,
                1,
                dataset.resolution[0] // 8,
                dataset.resolution[1] // 8,
            )
            if self.latent_shape != expected_latent:
                raise TrainingConfigError(
                    f"latentShape must match encoded dataset geometry {list(expected_latent)}"
                )
            if dataset.context_tokens != self.context_shape[1]:
                raise TrainingConfigError(
                    "dataset.contextTokens must match contextShape token count"
                )
        if self.guidance is not None:
            raise TrainingConfigError("qwen-image does not accept guidance")

    @classmethod
    def parse(cls, serialized: str) -> QwenImageTrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"config must be valid JSON: {exc.msg}") from exc
        return cls.from_mapping(_object(value, "config"))

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> QwenImageTrainingConfig:
        fields = {
            "schemaVersion",
            "family",
            "variant",
            "ditState",
            "textEncoderState",
            "vaeState",
            "datasetIdentity",
            "dataset",
            "distributed",
            "device",
            "baseDtype",
            "rank",
            "alpha",
            "loraTargets",
            "learningRate",
            "weightDecay",
            "betas",
            "epsilon",
            "optimizer",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "checkpointingMode",
            "checkpointInterval",
            "loraExportInterval",
            "syncDigestInterval",
            "seed",
            "rngPolicy",
            "latentShape",
            "contextShape",
            "attentionMaskShape",
            "guidance",
        }
        _known(raw, fields, "config")
        schema_version = _int(raw.get("schemaVersion", 1), "schemaVersion", minimum=1)
        if schema_version != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        family_value = raw.get("family")
        if not isinstance(family_value, str) or family_value != "qwen-image":
            raise TrainingConfigError("family must be 'qwen-image'")
        variant_value = raw.get("variant")
        if not isinstance(variant_value, str) or variant_value not in _QWEN_IMAGE_VARIANTS:
            raise TrainingConfigError("unsupported Qwen-Image training variant")
        base_dtype_value = raw.get("baseDtype", "bfloat16")
        if not isinstance(base_dtype_value, str) or base_dtype_value not in (
            "bfloat16",
            "float32",
        ):
            raise TrainingConfigError("baseDtype must be 'bfloat16' or 'float32'")
        targets = raw.get("loraTargets", ["attention.qkvo", "mlp.projections"])
        if (
            not isinstance(targets, list)
            or len(targets) != 2
            or any(not isinstance(target, str) for target in targets)
            or targets != ["attention.qkvo", "mlp.projections"]
        ):
            raise TrainingConfigError("loraTargets must be ['attention.qkvo', 'mlp.projections']")
        checkpointing = raw.get("gradientCheckpointing", True)
        if type(checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode_value = raw.get("checkpointingMode", "wholeModel")
        if not isinstance(checkpointing_mode_value, str) or checkpointing_mode_value not in (
            "blockNonReentrant",
            "wholeModel",
        ):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel'"
                " for Qwen-Image training"
            )
        optimizer_value = raw.get("optimizer", "adamw")
        if not isinstance(optimizer_value, str) or optimizer_value not in (
            "adamw",
            "factored-adamw",
        ):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        rng_value = raw.get("rngPolicy", "sequential")
        if not isinstance(rng_value, str) or rng_value not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        betas = raw.get("betas", [0.9, 0.999])
        if not isinstance(betas, list) or len(betas) != 2:
            raise TrainingConfigError("betas must be a two-item number array")
        beta1 = _float(betas[0], "betas[0]", minimum=0.000000001, maximum=0.999999999)
        beta2 = _float(betas[1], "betas[1]", minimum=0.000000001, maximum=0.999999999)
        latent = cast(
            "tuple[int, int, int, int, int]",
            _shape(raw.get("latentShape", [1, 16, 1, 64, 64]), "latentShape", 5),
        )
        context = cast(
            "tuple[int, int, int]",
            _shape(raw.get("contextShape", [1, 512, 3584]), "contextShape", 3),
        )
        attention_mask = cast(
            "tuple[int, int]",
            _shape(
                raw.get("attentionMaskShape", [context[0], context[1]]),
                "attentionMaskShape",
                2,
            ),
        )
        if "guidance" in raw:
            raise TrainingConfigError("qwen-image does not accept guidance")
        device_value = raw.get("device", "cpu")
        if not isinstance(device_value, str) or not device_value:
            raise TrainingConfigError("device must be a non-empty string")

        dit_state = _artifact_source(raw.get("ditState"), "ditState")
        text_encoder_state = _artifact_source(raw.get("textEncoderState"), "textEncoderState")
        vae_state = _artifact_source(raw.get("vaeState"), "vaeState")
        dataset = (
            None
            if raw.get("dataset") is None
            else _qwen_image_dataset_settings(
                raw["dataset"],
                vae_state=vae_state,
                text_encoder_state=text_encoder_state,
                context_tokens=context[1],
            )
        )
        identity_value = raw.get("datasetIdentity")
        if dataset is None:
            dataset_identity = _digest(identity_value, "datasetIdentity")
        else:
            dataset_identity = dataset.digest
            if (
                identity_value is not None
                and _digest(identity_value, "datasetIdentity") != dataset.digest
            ):
                raise TrainingConfigError("datasetIdentity does not match dataset.digest")

        return cls(
            family="qwen-image",
            variant="qwen-image",
            dit_state=dit_state,
            text_encoder_state=text_encoder_state,
            vae_state=vae_state,
            dataset_identity=dataset_identity,
            dataset=dataset,
            distributed=(
                None
                if raw.get("distributed") is None
                else _distributed_settings(raw["distributed"])
            ),
            device=device_value,
            base_dtype=base_dtype_value,
            rank=_int(raw.get("rank", 4), "rank", minimum=1),
            alpha=_float(raw.get("alpha", 4.0), "alpha", minimum=0.000000001),
            lora_targets=("attention.qkvo", "mlp.projections"),
            learning_rate=_float(
                raw.get("learningRate", 0.0001), "learningRate", minimum=0.000000000001
            ),
            weight_decay=_float(raw.get("weightDecay", 0.01), "weightDecay"),
            beta1=beta1,
            beta2=beta2,
            epsilon=_float(raw.get("epsilon", 1e-8), "epsilon", minimum=1e-30),
            optimizer=optimizer_value,
            gradient_accumulation_steps=_int(
                raw.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
                minimum=1,
            ),
            gradient_checkpointing=checkpointing,
            checkpointing_mode=checkpointing_mode_value,
            checkpoint_interval=_int(raw.get("checkpointInterval", 0), "checkpointInterval"),
            lora_export_interval=_int(raw.get("loraExportInterval", 0), "loraExportInterval"),
            sync_digest_interval=_int(raw.get("syncDigestInterval", 0), "syncDigestInterval"),
            seed=_int(raw.get("seed", 0), "seed"),
            rng_policy=rng_value,
            latent_shape=latent,
            context_shape=context,
            attention_mask_shape=attention_mask,
            guidance=None,
        )

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schemaVersion": 1,
            "family": self.family,
            "variant": self.variant,
            "ditState": self.dit_state.to_mapping(),
            "textEncoderState": self.text_encoder_state.to_mapping(),
            "vaeState": self.vae_state.to_mapping(),
            "device": self.device,
            "baseDtype": self.base_dtype,
            "rank": self.rank,
            "alpha": self.alpha,
            "loraTargets": list(self.lora_targets),
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "betas": [self.beta1, self.beta2],
            "epsilon": self.epsilon,
            "optimizer": self.optimizer,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "gradientCheckpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "latentShape": list(self.latent_shape),
            "contextShape": list(self.context_shape),
            "attentionMaskShape": list(self.attention_mask_shape),
        }
        if self.dataset is None:
            mapping["datasetIdentity"] = self.dataset_identity
        else:
            mapping["dataset"] = self.dataset.to_mapping()
        if self.checkpoint_interval:
            mapping["checkpointInterval"] = self.checkpoint_interval
        if self.lora_export_interval:
            mapping["loraExportInterval"] = self.lora_export_interval
        if self.sync_digest_interval:
            mapping["syncDigestInterval"] = self.sync_digest_interval
        if self.distributed is not None:
            mapping["distributed"] = self.distributed.to_mapping()
        if self.rng_policy != "sequential":
            mapping["rngPolicy"] = self.rng_policy
        if self.checkpointing_mode != "wholeModel":
            mapping["checkpointingMode"] = self.checkpointing_mode
        return mapping

    def identity_mapping(self) -> dict[str, object]:
        mapping = self.to_mapping()
        for field in _COMMON_RUNTIME_ONLY_FIELDS:
            mapping.pop(field, None)
        for field in ("ditState", "textEncoderState", "vaeState"):
            artifact = mapping[field]
            assert isinstance(artifact, dict)
            artifact.pop("path")
        dataset = mapping.get("dataset")
        if isinstance(dataset, dict):
            for field in _DATASET_RUNTIME_ONLY_FIELDS:
                dataset.pop(field, None)
        distributed = mapping.get("distributed")
        if isinstance(distributed, dict):
            for field in _DISTRIBUTED_RUNTIME_ONLY_FIELDS:
                distributed.pop(field, None)
        return mapping


@dataclass(frozen=True)
class Ideogram4TrainingConfig:
    """Normalized role-bound Ideogram 4 LoRA training configuration."""

    family: Literal["ideogram4"]
    variant: Literal["ideogram4"]
    role: Ideogram4RoleName
    diffusion_state: NativeTrainingArtifactSource
    text_encoder_state: NativeTrainingArtifactSource | None
    vae_state: NativeTrainingArtifactSource
    dataset_identity: str
    dataset: Ideogram4DatasetSettings | None
    distributed: DistributedSettings | None
    device: str
    base_dtype: Literal["bfloat16"]
    base_storage: Ideogram4StorageName
    int8_base_forward: MiniMaxH3Int8BaseForward | None
    rank: int
    alpha: float
    lora_targets: tuple[Ideogram4LoraTargetName, Ideogram4LoraTargetName]
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    optimizer: OptimizerName
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    checkpointing_mode: Literal["blockNonReentrant", "wholeModel"]
    checkpoint_interval: int
    lora_export_interval: int
    sync_digest_interval: int
    seed: int
    rng_policy: RngPolicyName
    latent_shape: tuple[int, int, int, int]
    context_shape: tuple[int, int, int] | None
    attention_mask_shape: tuple[int, int] | None
    flow_objective: Literal["ideogram4-shifted-logit-normal-noise-minus-data-mse-v1"]

    def __post_init__(self) -> None:
        if self.family != "ideogram4" or self.variant != "ideogram4":
            raise TrainingConfigError("family and variant must be 'ideogram4'")
        if self.role not in ("conditional", "unconditional"):
            raise TrainingConfigError("role must be 'conditional' or 'unconditional'")
        artifacts = (
            ("diffusionState", self.diffusion_state, "dinkster.ideogram4"),
            ("vaeState", self.vae_state, "dinkster.flux2"),
        )
        for name, artifact, family in artifacts:
            if type(artifact) is not NativeTrainingArtifactSource:
                raise TrainingConfigError(f"{name} must be a normalized native artifact")
            if not Path(artifact.path).is_absolute():
                raise TrainingConfigError(f"{name}.path must be absolute")
            _digest(artifact.digest, f"{name}.digest")
            _int(artifact.size, f"{name}.size", minimum=1)
            _family_native_identity(artifact.identity, f"{name}.identity", family)
        if self.role == "conditional":
            text = self.text_encoder_state
            if type(text) is not NativeTrainingArtifactSource:
                raise TrainingConfigError("conditional training requires textEncoderState")
            if not Path(text.path).is_absolute():
                raise TrainingConfigError("textEncoderState.path must be absolute")
            _digest(text.digest, "textEncoderState.digest")
            _int(text.size, "textEncoderState.size", minimum=1)
            _family_native_identity(
                text.identity, "textEncoderState.identity", "dinkster.ideogram4"
            )
        elif self.text_encoder_state is not None:
            raise TrainingConfigError("unconditional training does not accept textEncoderState")
        _digest(self.dataset_identity, "datasetIdentity")
        if self.dataset is not None:
            if type(self.dataset) is not Ideogram4DatasetSettings:
                raise TrainingConfigError("dataset must be normalized Ideogram 4 settings")
            if self.dataset.role != self.role or self.dataset.digest != self.dataset_identity:
                raise TrainingConfigError(
                    "dataset role and identity must match the training config"
                )
        if self.distributed is not None and type(self.distributed) is not DistributedSettings:
            raise TrainingConfigError("distributed must be normalized settings or None")
        if type(self.device) is not str or not self.device:
            raise TrainingConfigError("device must be a non-empty string")
        if self.base_dtype != "bfloat16":
            raise TrainingConfigError("baseDtype must be 'bfloat16'")
        if self.base_storage not in ("fp8", "int8-convrot"):
            raise TrainingConfigError("baseStorage must be 'fp8' or 'int8-convrot'")
        if self.base_storage == "fp8" and self.int8_base_forward is not None:
            raise TrainingConfigError("fp8 base storage does not accept int8BaseForward")
        if self.base_storage == "int8-convrot" and self.int8_base_forward not in (
            "dequantize",
            "fused",
        ):
            raise TrainingConfigError(
                "int8-convrot base storage requires int8BaseForward 'dequantize' or 'fused'"
            )
        if self.int8_base_forward == "fused" and not (
            self.device == "cuda" or self.device.startswith("cuda:")
        ):
            raise TrainingConfigError("fused INT8 base forward requires a CUDA device")
        expected_diffusion = _IDEOGRAM4_DIFFUSION_DIGESTS[(self.role, self.base_storage)]
        if self.diffusion_state.digest != expected_diffusion:
            raise TrainingConfigError(
                "diffusionState does not match the selected Ideogram 4 role and base storage"
            )
        if self.vae_state.digest != _IDEOGRAM4_VAE_DIGEST:
            raise TrainingConfigError("vaeState must be the official Flux2 VAE")
        if (
            self.text_encoder_state is not None
            and self.text_encoder_state.digest != _IDEOGRAM4_TEXT_DIGEST
        ):
            raise TrainingConfigError(
                "textEncoderState must be the official Ideogram 4 Qwen3-VL-8B"
            )
        _int(self.rank, "rank", minimum=1)
        _normalized_float(self.alpha, "alpha", minimum=0.000000001)
        if self.lora_targets != ("attention.qkvo", "mlp.projections"):
            raise TrainingConfigError("loraTargets must be ['attention.qkvo', 'mlp.projections']")
        _normalized_float(self.learning_rate, "learningRate", minimum=0.000000000001)
        _normalized_float(self.weight_decay, "weightDecay")
        _normalized_float(self.beta1, "betas[0]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.beta2, "betas[1]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.epsilon, "epsilon", minimum=1e-30)
        if self.optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        _int(self.gradient_accumulation_steps, "gradientAccumulationSteps", minimum=1)
        if type(self.gradient_checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        if self.checkpointing_mode not in ("blockNonReentrant", "wholeModel"):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel'"
            )
        _int(self.checkpoint_interval, "checkpointInterval")
        _int(self.lora_export_interval, "loraExportInterval")
        _int(self.sync_digest_interval, "syncDigestInterval")
        _int(self.seed, "seed")
        if self.rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        if (
            type(self.latent_shape) is not tuple
            or len(self.latent_shape) != 4
            or any(type(value) is not int or value < 1 for value in self.latent_shape)
            or self.latent_shape[1] != 128
        ):
            raise TrainingConfigError("latentShape must be [batch, 128, height, width]")
        if self.role == "conditional":
            context = self.context_shape
            mask = self.attention_mask_shape
            if (
                type(context) is not tuple
                or len(context) != 3
                or any(type(value) is not int or value < 1 for value in context)
                or context[0] != self.latent_shape[0]
                or context[2] != 53248
            ):
                raise TrainingConfigError("contextShape must be [batch, tokens, 53248]")
            if type(mask) is not tuple or mask != context[:2]:
                raise TrainingConfigError("attentionMaskShape must match context batch and tokens")
        elif self.context_shape is not None or self.attention_mask_shape is not None:
            raise TrainingConfigError("unconditional training does not accept context geometry")
        if self.dataset is not None:
            expected_latent = (
                self.latent_shape[0],
                128,
                self.dataset.resolution[0] // 16,
                self.dataset.resolution[1] // 16,
            )
            if self.latent_shape != expected_latent:
                raise TrainingConfigError(
                    f"latentShape must match encoded dataset geometry {list(expected_latent)}"
                )
            expected_tokens = 0 if self.context_shape is None else self.context_shape[1]
            if self.dataset.context_tokens != expected_tokens:
                raise TrainingConfigError("dataset context tokens do not match contextShape")
        if self.flow_objective != "ideogram4-shifted-logit-normal-noise-minus-data-mse-v1":
            raise TrainingConfigError("flowObjective is not supported")

    @classmethod
    def parse(cls, serialized: str) -> Ideogram4TrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"config must be valid JSON: {exc.msg}") from exc
        return cls.from_mapping(_object(value, "config"))

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> Ideogram4TrainingConfig:
        fields = {
            "schemaVersion",
            "family",
            "variant",
            "role",
            "diffusionState",
            "textEncoderState",
            "vaeState",
            "datasetIdentity",
            "dataset",
            "distributed",
            "device",
            "baseDtype",
            "baseStorage",
            "int8BaseForward",
            "rank",
            "alpha",
            "loraTargets",
            "learningRate",
            "weightDecay",
            "betas",
            "epsilon",
            "optimizer",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "checkpointingMode",
            "checkpointInterval",
            "loraExportInterval",
            "syncDigestInterval",
            "seed",
            "rngPolicy",
            "latentShape",
            "contextShape",
            "attentionMaskShape",
            "flowObjective",
        }
        _known(raw, fields, "config")
        if _int(raw.get("schemaVersion", 1), "schemaVersion", minimum=1) != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        if raw.get("family") != "ideogram4" or raw.get("variant", "ideogram4") != "ideogram4":
            raise TrainingConfigError("family and variant must be 'ideogram4'")
        role_value = raw.get("role")
        if role_value not in ("conditional", "unconditional"):
            raise TrainingConfigError("role must be 'conditional' or 'unconditional'")
        role = role_value
        storage_value = raw.get("baseStorage")
        if storage_value not in ("fp8", "int8-convrot"):
            raise TrainingConfigError("baseStorage must be 'fp8' or 'int8-convrot'")
        storage = storage_value
        int8_value = raw.get("int8BaseForward")
        if storage == "int8-convrot":
            if int8_value not in ("dequantize", "fused"):
                raise TrainingConfigError(
                    "int8-convrot base storage requires int8BaseForward 'dequantize' or 'fused'"
                )
            int8_forward = int8_value
        else:
            if "int8BaseForward" in raw:
                raise TrainingConfigError("fp8 base storage does not accept int8BaseForward")
            int8_forward = None
        if raw.get("baseDtype", "bfloat16") != "bfloat16":
            raise TrainingConfigError("baseDtype must be 'bfloat16'")
        if (
            raw.get(
                "flowObjective",
                "ideogram4-shifted-logit-normal-noise-minus-data-mse-v1",
            )
            != "ideogram4-shifted-logit-normal-noise-minus-data-mse-v1"
        ):
            raise TrainingConfigError("flowObjective is not supported")
        targets = raw.get("loraTargets", ["attention.qkvo", "mlp.projections"])
        if targets != ["attention.qkvo", "mlp.projections"]:
            raise TrainingConfigError("loraTargets must be ['attention.qkvo', 'mlp.projections']")
        optimizer = raw.get("optimizer", "adamw")
        if optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        checkpointing = raw.get("gradientCheckpointing", True)
        if type(checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode = raw.get("checkpointingMode", "wholeModel")
        if checkpointing_mode not in ("blockNonReentrant", "wholeModel"):
            raise TrainingConfigError(
                "checkpointingMode must be 'blockNonReentrant' or 'wholeModel'"
            )
        rng_policy = raw.get("rngPolicy", "sequential")
        if rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        betas = raw.get("betas", [0.9, 0.999])
        if not isinstance(betas, list) or len(betas) != 2:
            raise TrainingConfigError("betas must be a two-item number array")
        diffusion = _native_artifact_source(
            raw.get("diffusionState"), "diffusionState", family="dinkster.ideogram4"
        )
        text = (
            _native_artifact_source(
                raw.get("textEncoderState"),
                "textEncoderState",
                family="dinkster.ideogram4",
            )
            if role == "conditional"
            else None
        )
        if role == "unconditional" and "textEncoderState" in raw:
            raise TrainingConfigError("unconditional training does not accept textEncoderState")
        vae = _native_artifact_source(raw.get("vaeState"), "vaeState", family="dinkster.flux2")
        latent = cast(
            "tuple[int, int, int, int]",
            _shape(raw.get("latentShape", [1, 128, 64, 64]), "latentShape", 4),
        )
        if role == "conditional":
            context = cast(
                "tuple[int, int, int]",
                _shape(raw.get("contextShape", [1, 512, 53248]), "contextShape", 3),
            )
            attention_mask = cast(
                "tuple[int, int]",
                _shape(
                    raw.get("attentionMaskShape", [context[0], context[1]]),
                    "attentionMaskShape",
                    2,
                ),
            )
        else:
            if "contextShape" in raw or "attentionMaskShape" in raw:
                raise TrainingConfigError("unconditional training does not accept context geometry")
            context = None
            attention_mask = None
        dataset = (
            None
            if raw.get("dataset") is None
            else _ideogram4_dataset_settings(
                raw["dataset"],
                role=role,
                vae_state=vae,
                text_encoder_state=text,
                context_tokens=0 if context is None else context[1],
            )
        )
        identity_value = raw.get("datasetIdentity")
        if dataset is None:
            dataset_identity = _digest(identity_value, "datasetIdentity")
        else:
            dataset_identity = dataset.digest
            if (
                identity_value is not None
                and _digest(identity_value, "datasetIdentity") != dataset.digest
            ):
                raise TrainingConfigError("datasetIdentity does not match dataset.digest")
        device = raw.get("device", "cpu")
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        return cls(
            family="ideogram4",
            variant="ideogram4",
            role=role,
            diffusion_state=diffusion,
            text_encoder_state=text,
            vae_state=vae,
            dataset_identity=dataset_identity,
            dataset=dataset,
            distributed=(
                None
                if raw.get("distributed") is None
                else _distributed_settings(raw["distributed"])
            ),
            device=device,
            base_dtype="bfloat16",
            base_storage=storage,
            int8_base_forward=int8_forward,
            rank=_int(raw.get("rank", 4), "rank", minimum=1),
            alpha=_float(raw.get("alpha", 4.0), "alpha", minimum=0.000000001),
            lora_targets=("attention.qkvo", "mlp.projections"),
            learning_rate=_float(
                raw.get("learningRate", 0.0001), "learningRate", minimum=0.000000000001
            ),
            weight_decay=_float(raw.get("weightDecay", 0.01), "weightDecay"),
            beta1=_float(betas[0], "betas[0]", minimum=0.000000001, maximum=0.999999999),
            beta2=_float(betas[1], "betas[1]", minimum=0.000000001, maximum=0.999999999),
            epsilon=_float(raw.get("epsilon", 1e-8), "epsilon", minimum=1e-30),
            optimizer=optimizer,
            gradient_accumulation_steps=_int(
                raw.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
                minimum=1,
            ),
            gradient_checkpointing=checkpointing,
            checkpointing_mode=checkpointing_mode,
            checkpoint_interval=_int(raw.get("checkpointInterval", 0), "checkpointInterval"),
            lora_export_interval=_int(raw.get("loraExportInterval", 0), "loraExportInterval"),
            sync_digest_interval=_int(raw.get("syncDigestInterval", 0), "syncDigestInterval"),
            seed=_int(raw.get("seed", 0), "seed"),
            rng_policy=rng_policy,
            latent_shape=latent,
            context_shape=context,
            attention_mask_shape=attention_mask,
            flow_objective="ideogram4-shifted-logit-normal-noise-minus-data-mse-v1",
        )

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schemaVersion": 1,
            "family": self.family,
            "variant": self.variant,
            "role": self.role,
            "diffusionState": self.diffusion_state.to_mapping(),
            "vaeState": self.vae_state.to_mapping(),
            "device": self.device,
            "baseDtype": self.base_dtype,
            "baseStorage": self.base_storage,
            "rank": self.rank,
            "alpha": self.alpha,
            "loraTargets": list(self.lora_targets),
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "betas": [self.beta1, self.beta2],
            "epsilon": self.epsilon,
            "optimizer": self.optimizer,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "gradientCheckpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "latentShape": list(self.latent_shape),
            "flowObjective": self.flow_objective,
        }
        if self.text_encoder_state is not None:
            mapping["textEncoderState"] = self.text_encoder_state.to_mapping()
        if self.int8_base_forward is not None:
            mapping["int8BaseForward"] = self.int8_base_forward
        if self.context_shape is not None:
            assert self.attention_mask_shape is not None
            mapping["contextShape"] = list(self.context_shape)
            mapping["attentionMaskShape"] = list(self.attention_mask_shape)
        if self.dataset is None:
            mapping["datasetIdentity"] = self.dataset_identity
        else:
            mapping["dataset"] = self.dataset.to_mapping()
        if self.checkpoint_interval:
            mapping["checkpointInterval"] = self.checkpoint_interval
        if self.lora_export_interval:
            mapping["loraExportInterval"] = self.lora_export_interval
        if self.sync_digest_interval:
            mapping["syncDigestInterval"] = self.sync_digest_interval
        if self.distributed is not None:
            mapping["distributed"] = self.distributed.to_mapping()
        if self.rng_policy != "sequential":
            mapping["rngPolicy"] = self.rng_policy
        if self.checkpointing_mode != "wholeModel":
            mapping["checkpointingMode"] = self.checkpointing_mode
        return mapping

    def identity_mapping(self) -> dict[str, object]:
        mapping = self.to_mapping()
        for field in _COMMON_RUNTIME_ONLY_FIELDS:
            mapping.pop(field, None)
        for field in ("diffusionState", "textEncoderState", "vaeState"):
            artifact = mapping.get(field)
            if isinstance(artifact, dict):
                artifact.pop("path")
        dataset = mapping.get("dataset")
        if isinstance(dataset, dict):
            for field in _DATASET_RUNTIME_ONLY_FIELDS:
                dataset.pop(field, None)
        distributed = mapping.get("distributed")
        if isinstance(distributed, dict):
            for field in _DISTRIBUTED_RUNTIME_ONLY_FIELDS:
                distributed.pop(field, None)
        return mapping


@dataclass(frozen=True)
class MiniMaxH3TrainingConfig:
    """Normalized behavior and component identity for one H3 session."""

    family: Literal["minimax-h3"]
    dit_role: MiniMaxH3DiTRole
    dit_identity: str
    conditioner_identity: str
    dit_state_path: str | None
    dit_state_digest: str | None
    dit_state_size: int | None
    prepared_batch_root: str | None
    dataset: MiniMaxH3DatasetSettings | None
    distributed: DistributedSettings | None
    device: str
    base_dtype: BaseDtypeName
    quantized_base: bool
    int8_base_forward: MiniMaxH3Int8BaseForward
    host_layer_paging_fraction: float
    rank: int
    alpha: float
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    optimizer: OptimizerName
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    checkpointing_mode: Literal["wholeModel"]
    checkpoint_interval: int
    lora_export_interval: int
    sync_digest_interval: int
    seed: int
    rng_policy: RngPolicyName
    video_latent_shape: tuple[int, int, int, int, int]
    audio_latent_shape: tuple[int, int, int, int]
    conditioner_shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        # Construction-time guard so direct dataclass creation and replace()
        # cannot carry an unsupported mode past the from_mapping validation.
        if self.checkpointing_mode != "wholeModel":
            raise TrainingConfigError(
                "checkpointingMode must be 'wholeModel' for MiniMax H3 training"
            )
        if self.int8_base_forward not in ("dequantize", "fused"):
            raise TrainingConfigError("int8BaseForward must be 'dequantize' or 'fused'")
        if self.int8_base_forward == "fused" and not self.quantized_base:
            raise TrainingConfigError("int8BaseForward 'fused' requires quantizedBase")
        if self.int8_base_forward == "fused" and not (
            self.device == "cuda" or self.device.startswith("cuda:")
        ):
            raise TrainingConfigError("int8BaseForward 'fused' requires a CUDA device")

    @classmethod
    def parse(cls, serialized: str) -> MiniMaxH3TrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"config must be valid JSON: {exc.msg}") from exc
        return cls.from_mapping(_object(value, "config"))

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> MiniMaxH3TrainingConfig:
        fields = {
            "schemaVersion",
            "family",
            "ditRole",
            "ditIdentity",
            "conditionerIdentity",
            "ditState",
            "preparedBatchRoot",
            "dataset",
            "distributed",
            "device",
            "baseDtype",
            "quantizedBase",
            "int8BaseForward",
            "hostLayerPagingFraction",
            "rank",
            "alpha",
            "learningRate",
            "weightDecay",
            "betas",
            "epsilon",
            "optimizer",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "checkpointingMode",
            "checkpointInterval",
            "loraExportInterval",
            "syncDigestInterval",
            "seed",
            "rngPolicy",
            "videoLatentShape",
            "audioLatentShape",
            "conditionerShape",
        }
        _known(raw, fields, "config")
        if raw.get("schemaVersion", 1) != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        if raw.get("family") != "minimax-h3":
            raise TrainingConfigError("family must be 'minimax-h3'")
        role_value = raw.get("ditRole")
        if role_value not in ("fl2va-dit", "ref2va-dit"):
            raise TrainingConfigError("ditRole must be 'fl2va-dit' or 'ref2va-dit'")
        role = role_value
        dit_identity = _native_identity(raw.get("ditIdentity"), "ditIdentity")
        conditioner_identity = _native_identity(
            raw.get("conditionerIdentity"), "conditionerIdentity"
        )
        try:
            compose_execution(
                MINIMAX_H3_CONFIG.family_id,
                {role.replace("-", "_"): dit_identity, "conditioner": conditioner_identity},
            )
        except (TypeError, ValueError) as exc:
            raise TrainingConfigError(f"H3 component identities cannot compose: {exc}") from exc

        state_path: str | None = None
        state_digest: str | None = None
        state_size: int | None = None
        if (state_value := raw.get("ditState")) is not None:
            state = _object(state_value, "ditState")
            _known(state, {"path", "digest", "size"}, "ditState")
            path = state.get("path")
            if not isinstance(path, str) or not path:
                raise TrainingConfigError("ditState.path must be a non-empty string")
            state_path = str(Path(path).expanduser().resolve())
            state_digest = _digest(state.get("digest"), "ditState.digest")
            state_size = _int(state.get("size"), "ditState.size")

        batch_root = raw.get("preparedBatchRoot")
        if batch_root is not None and (not isinstance(batch_root, str) or not batch_root):
            raise TrainingConfigError("preparedBatchRoot must be a non-empty string")
        device = raw.get("device", "cpu")
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        dataset = (
            None
            if raw.get("dataset") is None
            else _h3_dataset_settings(raw["dataset"], device=device)
        )
        distributed = (
            None if raw.get("distributed") is None else _distributed_settings(raw["distributed"])
        )
        if batch_root is not None and dataset is not None:
            raise TrainingConfigError("preparedBatchRoot and dataset cannot both be set")
        if dataset is not None:
            if role != "fl2va-dit":
                raise TrainingConfigError(
                    "H3 video/audio/caption datasets require ditRole 'fl2va-dit'"
                )
            if dataset.conditioner_state.identity != conditioner_identity:
                raise TrainingConfigError(
                    "conditionerIdentity and dataset.conditionerState.identity must match"
                )
        base_dtype = raw.get("baseDtype", "float32")
        if base_dtype not in ("float32", "bfloat16"):
            raise TrainingConfigError("baseDtype must be 'float32' or 'bfloat16'")
        quantized_base = raw.get("quantizedBase", False)
        if type(quantized_base) is not bool:
            raise TrainingConfigError("quantizedBase must be a boolean")
        if quantized_base and base_dtype != "bfloat16":
            raise TrainingConfigError("quantizedBase requires baseDtype 'bfloat16'")
        int8_base_forward = raw.get("int8BaseForward", "dequantize")
        if int8_base_forward not in ("dequantize", "fused"):
            raise TrainingConfigError("int8BaseForward must be 'dequantize' or 'fused'")
        if int8_base_forward == "fused" and not quantized_base:
            raise TrainingConfigError("int8BaseForward 'fused' requires quantizedBase")
        if int8_base_forward == "fused" and not (device == "cuda" or device.startswith("cuda:")):
            raise TrainingConfigError("int8BaseForward 'fused' requires a CUDA device")
        host_layer_paging_fraction = _float(
            raw.get("hostLayerPagingFraction", 0.0),
            "hostLayerPagingFraction",
            maximum=1.0,
        )
        if host_layer_paging_fraction and not (device == "cuda" or device.startswith("cuda:")):
            raise TrainingConfigError("hostLayerPagingFraction requires a CUDA device")
        optimizer = raw.get("optimizer", "adamw")
        if optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        checkpointing = raw.get("gradientCheckpointing", True)
        if type(checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode = raw.get("checkpointingMode", "wholeModel")
        checkpoint_interval = _int(raw.get("checkpointInterval", 0), "checkpointInterval")
        lora_export_interval = _int(raw.get("loraExportInterval", 0), "loraExportInterval")
        sync_digest_interval = _int(raw.get("syncDigestInterval", 0), "syncDigestInterval")
        rng_policy = raw.get("rngPolicy", "sequential")
        if rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        betas_value = raw.get("betas", [0.9, 0.999])
        if not isinstance(betas_value, list) or len(betas_value) != 2:
            raise TrainingConfigError("betas must be a two-item number array")
        beta1 = _float(betas_value[0], "betas[0]", maximum=0.999999999)
        beta2 = _float(betas_value[1], "betas[1]", maximum=0.999999999)
        if beta1 == 0.0 or beta2 == 0.0:
            raise TrainingConfigError("betas entries must be greater than zero")

        video = cast(
            "tuple[int, int, int, int, int]",
            _shape(raw.get("videoLatentShape", [1, 24, 1, 32, 32]), "videoLatentShape", 5),
        )
        audio = cast(
            "tuple[int, int, int, int]",
            _shape(raw.get("audioLatentShape", [1, 32, 2, 1]), "audioLatentShape", 4),
        )
        conditioner = cast(
            "tuple[int, int, int]",
            _shape(
                raw.get("conditionerShape", [1, 1, MINIMAX_H3_CONFIG.text_width]),
                "conditionerShape",
                3,
            ),
        )
        if video[:2] != (MINIMAX_H3_CONFIG.batch_size, MINIMAX_H3_CONFIG.video_latent_channels):
            raise TrainingConfigError("videoLatentShape must be [1, 24, time, height, width]")
        if audio[:3] != (
            MINIMAX_H3_CONFIG.batch_size,
            MINIMAX_H3_CONFIG.audio_latent_channels,
            MINIMAX_H3_CONFIG.audio_content_channels,
        ):
            raise TrainingConfigError("audioLatentShape must be [1, 32, 2, time]")
        if conditioner[0] != MINIMAX_H3_CONFIG.batch_size or conditioner[2] != (
            MINIMAX_H3_CONFIG.text_width
        ):
            raise TrainingConfigError("conditionerShape must be [1, tokens, 5120]")
        if dataset is not None:
            expected_video = minimax_h3_video_latent_shape(dataset.frame_count, dataset.resolution)
            if video != expected_video:
                raise TrainingConfigError(
                    "videoLatentShape does not match dataset frame count and resolution"
                )
            expected_audio = minimax_h3_audio_latent_shape(dataset.frame_count)
            if audio != expected_audio:
                raise TrainingConfigError(
                    "audioLatentShape does not match dataset frame count and sample rate"
                )

        return cls(
            family="minimax-h3",
            dit_role=role,
            dit_identity=dit_identity,
            conditioner_identity=conditioner_identity,
            dit_state_path=state_path,
            dit_state_digest=state_digest,
            dit_state_size=state_size,
            prepared_batch_root=(
                None if batch_root is None else str(Path(batch_root).expanduser().resolve())
            ),
            dataset=dataset,
            distributed=distributed,
            device=device,
            base_dtype=base_dtype,
            quantized_base=quantized_base,
            int8_base_forward=int8_base_forward,
            host_layer_paging_fraction=host_layer_paging_fraction,
            rank=_int(raw.get("rank", 4), "rank", minimum=1),
            alpha=_float(raw.get("alpha", 4.0), "alpha", minimum=0.000000001),
            learning_rate=_float(
                raw.get("learningRate", 0.0001), "learningRate", minimum=0.000000000001
            ),
            weight_decay=_float(raw.get("weightDecay", 0.01), "weightDecay"),
            beta1=beta1,
            beta2=beta2,
            epsilon=_float(raw.get("epsilon", 1e-8), "epsilon", minimum=1e-30),
            optimizer=optimizer,
            gradient_accumulation_steps=_int(
                raw.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
                minimum=1,
            ),
            gradient_checkpointing=checkpointing,
            checkpointing_mode=cast('Literal["wholeModel"]', checkpointing_mode),
            checkpoint_interval=checkpoint_interval,
            lora_export_interval=lora_export_interval,
            sync_digest_interval=sync_digest_interval,
            seed=_int(raw.get("seed", 0), "seed"),
            rng_policy=rng_policy,
            video_latent_shape=video,
            audio_latent_shape=audio,
            conditioner_shape=conditioner,
        )

    @property
    def execution_composition(self) -> ExecutionComposition:
        return compose_execution(
            MINIMAX_H3_CONFIG.family_id,
            {
                self.dit_role.replace("-", "_"): self.dit_identity,
                "conditioner": self.conditioner_identity,
            },
        )

    def to_mapping(self) -> dict[str, object]:
        state: dict[str, object] | None = None
        if self.dit_state_path is not None:
            assert self.dit_state_digest is not None and self.dit_state_size is not None
            state = {
                "path": self.dit_state_path,
                "digest": self.dit_state_digest,
                "size": self.dit_state_size,
            }
        mapping: dict[str, object] = {
            "schemaVersion": 1,
            "family": self.family,
            "ditRole": self.dit_role,
            "ditIdentity": self.dit_identity,
            "conditionerIdentity": self.conditioner_identity,
            "ditState": state,
            "preparedBatchRoot": self.prepared_batch_root,
            "dataset": None if self.dataset is None else self.dataset.to_mapping(),
            "device": self.device,
            "baseDtype": self.base_dtype,
            "rank": self.rank,
            "alpha": self.alpha,
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "betas": [self.beta1, self.beta2],
            "epsilon": self.epsilon,
            "optimizer": self.optimizer,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "gradientCheckpointing": self.gradient_checkpointing,
            "seed": self.seed,
            "videoLatentShape": list(self.video_latent_shape),
            "audioLatentShape": list(self.audio_latent_shape),
            "conditionerShape": list(self.conditioner_shape),
        }
        if self.quantized_base:
            mapping["quantizedBase"] = True
        if self.int8_base_forward != "dequantize":
            mapping["int8BaseForward"] = self.int8_base_forward
        if self.host_layer_paging_fraction:
            mapping["hostLayerPagingFraction"] = self.host_layer_paging_fraction
        if self.checkpoint_interval:
            mapping["checkpointInterval"] = self.checkpoint_interval
        if self.lora_export_interval:
            mapping["loraExportInterval"] = self.lora_export_interval
        if self.sync_digest_interval:
            mapping["syncDigestInterval"] = self.sync_digest_interval
        if self.distributed is not None:
            mapping["distributed"] = self.distributed.to_mapping()
        if self.rng_policy != "sequential":
            mapping["rngPolicy"] = self.rng_policy
        return mapping

    def identity_mapping(self) -> dict[str, object]:
        mapping = self.to_mapping()
        # Runtime-only paging, encoded-cache placement, and checkpoint cadence
        # change where identical bytes live or when they become durable, never
        # the optimizer, RNG, adapter, or data-cursor semantics.
        for field in _COMMON_RUNTIME_ONLY_FIELDS + _H3_RUNTIME_ONLY_FIELDS:
            mapping.pop(field, None)
        dataset = mapping.get("dataset")
        if isinstance(dataset, dict):
            for field in _DATASET_RUNTIME_ONLY_FIELDS:
                dataset.pop(field, None)
        distributed = mapping.get("distributed")
        if isinstance(distributed, dict):
            for field in _DISTRIBUTED_RUNTIME_ONLY_FIELDS:
                distributed.pop(field, None)
        return mapping


@dataclass(frozen=True)
class MiniMaxMusic3TrainingConfig:
    """Strict community-derived MiniMax Music 3 LoRA training identity."""

    family: Literal["minimax-music3"]
    diffusion_state: MiniMaxMusic3ArtifactPin
    dataset: MiniMaxMusic3DatasetSettings
    distributed: DistributedSettings | None
    device: str
    base_dtype: MiniMaxMusic3BaseDtypeName
    text_dtype: BaseDtypeName
    quantized_base: bool
    rank: int
    alpha: float
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    epsilon: float
    optimizer: OptimizerName
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    checkpointing_mode: Literal["wholeModel"]
    checkpoint_interval: int
    lora_export_interval: int
    sync_digest_interval: int
    seed: int
    rng_policy: RngPolicyName

    def __post_init__(self) -> None:
        if self.family != "minimax-music3":
            raise TrainingConfigError("family must be 'minimax-music3'")
        dataset = cast("object", self.dataset)
        if not isinstance(dataset, MiniMaxMusic3DatasetSettings):
            raise TrainingConfigError("dataset must be normalized MiniMax Music 3 settings")
        for name, pin, needs_identity in (
            ("diffusionState", self.diffusion_state, True),
            ("dataset.davEncoderState", self.dataset.dav_encoder_state, False),
            ("dataset.rvqEncoderState", self.dataset.rvq_encoder_state, False),
            ("dataset.textEncoderState", self.dataset.text_encoder_state, True),
        ):
            pin_value = cast("object", pin)
            if not isinstance(pin_value, MiniMaxMusic3ArtifactPin):
                raise TrainingConfigError(f"{name} must be a normalized artifact pin")
            if not Path(pin.path).is_absolute():
                raise TrainingConfigError(f"{name}.path must be absolute")
            _digest(pin.digest, f"{name}.digest")
            _int(pin.size, f"{name}.size", minimum=1)
            if needs_identity:
                _music3_native_identity(pin.identity, f"{name}.identity")
            elif pin.identity is not None:
                raise TrainingConfigError(f"{name}.identity is not accepted")
        if not Path(self.dataset.root).is_absolute():
            raise TrainingConfigError("dataset.root must be absolute")
        _int(self.dataset.audio_frames, "dataset.audioFrames", minimum=1)
        if self.dataset.audio_frames > 128:
            raise TrainingConfigError("dataset.audioFrames must be <= 128")
        if (
            self.dataset.encoded_cache_root is not None
            and not Path(self.dataset.encoded_cache_root).is_absolute()
        ):
            raise TrainingConfigError("dataset.encodedCacheRoot must be absolute")
        distributed = cast("object", self.distributed)
        if not isinstance(distributed, (DistributedSettings, type(None))):
            raise TrainingConfigError("distributed must be normalized settings or None")
        device = cast("object", self.device)
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        if self.base_dtype not in ("float16", "bfloat16", "float32"):
            raise TrainingConfigError("baseDtype must be 'float16', 'bfloat16', or 'float32'")
        if self.text_dtype not in ("bfloat16", "float32"):
            raise TrainingConfigError("textDtype must be 'bfloat16' or 'float32'")
        if self.dataset.text_compute_dtype != self.text_dtype:
            raise TrainingConfigError("dataset text compute dtype must match textDtype")
        if type(self.quantized_base) is not bool:
            raise TrainingConfigError("quantizedBase must be a boolean")
        if self.quantized_base != (self.base_dtype == "bfloat16"):
            raise TrainingConfigError(
                "quantizedBase requires bfloat16 baseDtype and bfloat16 baseDtype requires"
                " quantizedBase"
            )
        _int(self.rank, "rank", minimum=1)
        _normalized_float(self.alpha, "alpha", minimum=0.000000001)
        _normalized_float(self.learning_rate, "learningRate", minimum=0.000000000001)
        _normalized_float(self.weight_decay, "weightDecay")
        _normalized_float(self.beta1, "betas[0]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.beta2, "betas[1]", minimum=0.000000001, maximum=0.999999999)
        _normalized_float(self.epsilon, "epsilon", minimum=1e-30)
        if self.optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        _int(self.gradient_accumulation_steps, "gradientAccumulationSteps", minimum=1)
        if type(self.gradient_checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        if self.checkpointing_mode != "wholeModel":
            raise TrainingConfigError(
                "checkpointingMode must be 'wholeModel' for MiniMax Music 3 training"
            )
        _int(self.checkpoint_interval, "checkpointInterval")
        _int(self.lora_export_interval, "loraExportInterval")
        _int(self.sync_digest_interval, "syncDigestInterval")
        _int(self.seed, "seed")
        if self.rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")

    @classmethod
    def parse(cls, serialized: str) -> MiniMaxMusic3TrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError(f"config must be valid JSON: {exc.msg}") from exc
        return cls.from_mapping(_object(value, "config"))

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> MiniMaxMusic3TrainingConfig:
        fields = {
            "schemaVersion",
            "family",
            "communityRecipeRevision",
            "flowObjective",
            "diffusionState",
            "dataset",
            "distributed",
            "device",
            "baseDtype",
            "textDtype",
            "quantizedBase",
            "rank",
            "alpha",
            "learningRate",
            "weightDecay",
            "betas",
            "epsilon",
            "optimizer",
            "gradientAccumulationSteps",
            "gradientCheckpointing",
            "checkpointingMode",
            "checkpointInterval",
            "loraExportInterval",
            "syncDigestInterval",
            "seed",
            "rngPolicy",
        }
        _known(raw, fields, "config")
        if raw.get("schemaVersion", 1) != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        if raw.get("family") != "minimax-music3":
            raise TrainingConfigError("family must be 'minimax-music3'")
        if (
            raw.get("communityRecipeRevision", MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION)
            != MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION
        ):
            raise TrainingConfigError("communityRecipeRevision is not supported")
        if raw.get("flowObjective", MINIMAX_MUSIC3_FLOW_OBJECTIVE) != MINIMAX_MUSIC3_FLOW_OBJECTIVE:
            raise TrainingConfigError("flowObjective is not supported")
        device = raw.get("device", "cpu")
        if not isinstance(device, str) or not device:
            raise TrainingConfigError("device must be a non-empty string")
        base_dtype = raw.get("baseDtype", "float32")
        if base_dtype not in ("float16", "bfloat16", "float32"):
            raise TrainingConfigError("baseDtype must be 'float16', 'bfloat16', or 'float32'")
        text_dtype = raw.get("textDtype", "bfloat16")
        if text_dtype not in ("bfloat16", "float32"):
            raise TrainingConfigError("textDtype must be 'bfloat16' or 'float32'")
        quantized = raw.get("quantizedBase", False)
        if type(quantized) is not bool:
            raise TrainingConfigError("quantizedBase must be a boolean")
        if quantized != (base_dtype == "bfloat16"):
            raise TrainingConfigError(
                "quantizedBase requires bfloat16 baseDtype and bfloat16 baseDtype requires"
                " quantizedBase"
            )
        optimizer = raw.get("optimizer", "adamw")
        if optimizer not in ("adamw", "factored-adamw"):
            raise TrainingConfigError("optimizer must be 'adamw' or 'factored-adamw'")
        checkpointing = raw.get("gradientCheckpointing", True)
        if type(checkpointing) is not bool:
            raise TrainingConfigError("gradientCheckpointing must be a boolean")
        checkpointing_mode = raw.get("checkpointingMode", "wholeModel")
        if checkpointing_mode != "wholeModel":
            raise TrainingConfigError(
                "checkpointingMode must be 'wholeModel' for MiniMax Music 3 training"
            )
        rng_policy = raw.get("rngPolicy", "sequential")
        if rng_policy not in ("sequential", "counter"):
            raise TrainingConfigError("rngPolicy must be 'sequential' or 'counter'")
        betas = raw.get("betas", [0.9, 0.999])
        if not isinstance(betas, list) or len(betas) != 2:
            raise TrainingConfigError("betas must be a two-item number array")
        beta1 = _float(betas[0], "betas[0]", maximum=0.999999999)
        beta2 = _float(betas[1], "betas[1]", maximum=0.999999999)
        if beta1 == 0.0 or beta2 == 0.0:
            raise TrainingConfigError("betas entries must be greater than zero")
        diffusion = _music3_artifact_pin(raw.get("diffusionState"), "diffusionState", identity=True)
        dataset = _music3_dataset_settings(
            raw.get("dataset"),
            device=device,
            text_dtype=text_dtype,
        )
        distributed = (
            None if raw.get("distributed") is None else _distributed_settings(raw["distributed"])
        )
        return cls(
            family="minimax-music3",
            diffusion_state=diffusion,
            dataset=dataset,
            distributed=distributed,
            device=device,
            base_dtype=base_dtype,
            text_dtype=text_dtype,
            quantized_base=quantized,
            rank=_int(raw.get("rank", 4), "rank", minimum=1),
            alpha=_float(raw.get("alpha", 4.0), "alpha", minimum=0.000000001),
            learning_rate=_float(
                raw.get("learningRate", 0.0001), "learningRate", minimum=0.000000000001
            ),
            weight_decay=_float(raw.get("weightDecay", 0.01), "weightDecay"),
            beta1=beta1,
            beta2=beta2,
            epsilon=_float(raw.get("epsilon", 1e-8), "epsilon", minimum=1e-30),
            optimizer=optimizer,
            gradient_accumulation_steps=_int(
                raw.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
                minimum=1,
            ),
            gradient_checkpointing=checkpointing,
            checkpointing_mode="wholeModel",
            checkpoint_interval=_int(raw.get("checkpointInterval", 0), "checkpointInterval"),
            lora_export_interval=_int(raw.get("loraExportInterval", 0), "loraExportInterval"),
            sync_digest_interval=_int(raw.get("syncDigestInterval", 0), "syncDigestInterval"),
            seed=_int(raw.get("seed", 0), "seed"),
            rng_policy=rng_policy,
        )

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (1, MINIMAX_MUSIC3_CONFIG.latent_channels, self.dataset.latent_frames)

    @property
    def context_shape(self) -> tuple[int, int, int]:
        return (
            1,
            self.dataset.audio_frames,
            MINIMAX_MUSIC3_CONFIG.context_layers * MINIMAX_MUSIC3_CONFIG.context_width,
        )

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schemaVersion": 1,
            "family": self.family,
            "communityRecipeRevision": MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION,
            "flowObjective": MINIMAX_MUSIC3_FLOW_OBJECTIVE,
            "diffusionState": self.diffusion_state.to_mapping(),
            "dataset": self.dataset.to_mapping(),
            "device": self.device,
            "baseDtype": self.base_dtype,
            "textDtype": self.text_dtype,
            "rank": self.rank,
            "alpha": self.alpha,
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "betas": [self.beta1, self.beta2],
            "epsilon": self.epsilon,
            "optimizer": self.optimizer,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "gradientCheckpointing": self.gradient_checkpointing,
            "seed": self.seed,
        }
        if self.quantized_base:
            mapping["quantizedBase"] = True
        if self.checkpoint_interval:
            mapping["checkpointInterval"] = self.checkpoint_interval
        if self.lora_export_interval:
            mapping["loraExportInterval"] = self.lora_export_interval
        if self.sync_digest_interval:
            mapping["syncDigestInterval"] = self.sync_digest_interval
        if self.distributed is not None:
            mapping["distributed"] = self.distributed.to_mapping()
        if self.rng_policy != "sequential":
            mapping["rngPolicy"] = self.rng_policy
        return mapping

    def identity_mapping(self) -> dict[str, object]:
        mapping = self.to_mapping()
        for field in _COMMON_RUNTIME_ONLY_FIELDS:
            mapping.pop(field, None)
        diffusion = cast("dict[str, object]", mapping["diffusionState"])
        diffusion.pop("path")
        dataset = cast("dict[str, object]", mapping["dataset"])
        for field in _DATASET_RUNTIME_ONLY_FIELDS:
            dataset.pop(field, None)
        for field in ("davEncoderState", "rvqEncoderState", "textEncoderState"):
            artifact = cast("dict[str, object]", dataset[field])
            artifact.pop("path")
        distributed = mapping.get("distributed")
        if isinstance(distributed, dict):
            for field in _DISTRIBUTED_RUNTIME_ONLY_FIELDS:
                distributed.pop(field, None)
        return mapping
