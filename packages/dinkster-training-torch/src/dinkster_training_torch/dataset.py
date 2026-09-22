"""Content identity and validation for training dataset folders."""

from __future__ import annotations

import io
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dinkster_api.v1 import digest_bytes
from dinkster_inference import MINIMAX_H3_CONFIG, QWEN_IMAGE_TEXT_CONFIG
from PIL import Image, ImageOps

from .container import (
    MINIMAX_H3_CONTAINER_DECODER_IDENTITY,
    decode_audio_s16_stereo_32k,
    decode_video_rgb24,
)

IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
VIDEO_EXTENSIONS = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"})
SD15_PREPROCESSING_IDENTITY = "sd15-center-crop-bilinear-antialias-rgb-v1"
SDXL_PREPROCESSING_IDENTITY = "sdxl-center-crop-bilinear-antialias-rgb-v1"
MINIMAX_H3_PREPROCESSING_IDENTITY = "minimax-h3-rgb-neg1-pos1-pcm-wave-v1"
WAN21_PREPROCESSING_IDENTITY = "wan21-t2v-rgb-neg1-pos1-causal-fp32-v1"
WAN22_PREPROCESSING_IDENTITY = "wan22-ti2v-rgb-neg1-pos1-causal-16x-fp32-v1"
WAN21_TOKENIZER_IDENTITY = "dinkster-umt5-spiece-plain-truncate-512-fp32-v1"
FLUX_PREPROCESSING_IDENTITY = "flux1-center-crop-bilinear-antialias-rgb-v1"
FLUX_CLIP_TOKENIZER_IDENTITY = "dinkster-clip-bpe-flux-pooled-v1"
FLUX_T5_TOKENIZER_IDENTITY = "dinkster-t5-spiece-flux-chunks-v1"
FLUX2_PREPROCESSING_IDENTITY = "flux2-center-crop-bilinear-antialias-rgb-v1"
FLUX2_DEV_TOKENIZER_IDENTITY = "dinkster-tekken-flux2-dev-instruct-v1"
FLUX2_KLEIN_TOKENIZER_IDENTITY = "dinkster-qwen3-flux2-klein-chat-pad512-right-context-v1"
FLUX2_LATENT_IDENTITY = "dinkster-flux2-kl-mode-pack2x2-batchnorm-v1"
FLUX2_TEXT_ENCODING_DTYPE = "bfloat16"
QWEN_IMAGE_FAMILY_IDENTITY = "qwen-image"
QWEN_IMAGE_PREPROCESSING_IDENTITY = (
    "qwen-image-center-crop-bilinear-antialias-rgb-wan21-neg1-pos1-v1"
)
QWEN_IMAGE_TOKENIZER_IDENTITY = "dinkster-qwen-bpe-qwen-image-text-template-v1"
QWEN_IMAGE_LATENT_IDENTITY = "dinkster-wan21-single-frame-process-in-fp32-v1"
QWEN_IMAGE_CONTEXT_IDENTITY = "dinkster-qwen-image-post-user-boundary-v1"
QWEN_IMAGE_PADDING_IDENTITY = "right-zero-v1"
QWEN_IMAGE_POSITION_IDENTITY = "unmodified-full-template-position-v1"
QWEN_IMAGE_MASK_IDENTITY = "float32-one-real-zero-right-padding-v1"
QWEN_IMAGE_TEXT_ENCODING_DTYPE = "bfloat16"
IDEOGRAM4_PREPROCESSING_IDENTITY = "ideogram4-center-crop-bilinear-antialias-rgb-v1"
IDEOGRAM4_TOKENIZER_IDENTITY = "dinkster-qwen-bpe-ideogram4-chat-template-v1"
IDEOGRAM4_LATENT_IDENTITY = "dinkster-flux2-kl-mode-pack2x2-batchnorm-v1"
IDEOGRAM4_CONTEXT_IDENTITY = "dinkster-qwen3vl-8b-13-tap-concatenation-v1"
IDEOGRAM4_TEXT_ENCODING_DTYPE = "bfloat16"
TOKENIZER_IDENTITY = "dinkster-clip-bpe-plain-truncate-77-v1"
TrainingFamily = Literal["sd15", "sdxl"]
WanVAEContract = Literal["wan21", "wan22"]
WanConditioningContract = Literal["none", "first-frame-i2v"]


def _digest(data: bytes) -> str:
    return digest_bytes(data)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


@dataclass(frozen=True)
class EncoderStateSource:
    """A digest-pinned safetensors component and its key prefix."""

    path: str
    digest: str
    prefix: str

    def to_mapping(self) -> dict[str, object]:
        return {"path": self.path, "digest": self.digest, "prefix": self.prefix}


@dataclass(frozen=True)
class MiniMaxH3ComponentStateSource:
    """One immutable H3 component artifact and its native identity."""

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


@dataclass(frozen=True)
class ImageCaptionItem:
    """One digest-pinned image and its caption sidecar."""

    image_path: Path
    caption_path: Path
    relative_image: str
    relative_caption: str
    image_digest: str
    caption_digest: str | None


@dataclass(frozen=True)
class DatasetInspection:
    """Stable dataset manifest plus user-facing validation facts."""

    digest: str
    items: tuple[ImageCaptionItem, ...]
    caption_files: int
    nonempty_captions: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class MiniMaxH3DatasetItem:
    """One digest-pinned H3 video, audio source, and caption sidecar."""

    video_kind: Literal["frames", "container"]
    audio_kind: Literal["wav", "container"]
    video_path: Path
    audio_path: Path
    caption_path: Path
    relative_item: str
    relative_frames: tuple[str, ...]
    frame_digests: tuple[str, ...]
    relative_container: str | None
    container_digest: str | None
    relative_audio: str
    audio_digest: str | None
    relative_caption: str
    caption_digest: str | None


@dataclass(frozen=True)
class MiniMaxH3DatasetInspection:
    """Stable H3 dataset manifest plus user-facing validation facts."""

    digest: str
    items: tuple[MiniMaxH3DatasetItem, ...]
    caption_files: int
    nonempty_captions: int
    frame_folder_items: int
    container_items: int
    embedded_audio_items: int
    sidecar_audio_items: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class WanDatasetItem:
    """One digest-pinned Wan video and caption sidecar."""

    video_path: Path
    caption_path: Path
    relative_video: str
    relative_caption: str
    video_digest: str
    caption_digest: str | None


@dataclass(frozen=True)
class WanDatasetInspection:
    """Stable Wan video/caption manifest plus validation facts."""

    digest: str
    items: tuple[WanDatasetItem, ...]
    caption_files: int
    nonempty_captions: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class ImageCaptionDatasetSettings:
    """Normalized configuration and current content identity."""

    family: TrainingFamily
    root: str
    resolution: tuple[int, int]
    vae_state: EncoderStateSource | None
    text_encoder_state: EncoderStateSource | None
    checkpoint_state: EncoderStateSource | None
    inspection: DatasetInspection
    encoded_cache_root: str | None = None

    @property
    def digest(self) -> str:
        return self.inspection.digest

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "type": "image-caption-folder",
            "root": self.root,
            "resolution": list(self.resolution),
            "digest": self.digest,
        }
        if self.family == "sd15":
            assert self.vae_state is not None and self.text_encoder_state is not None
            mapping["vaeState"] = self.vae_state.to_mapping()
            mapping["textEncoderState"] = self.text_encoder_state.to_mapping()
        else:
            assert self.checkpoint_state is not None
            mapping["checkpointState"] = self.checkpoint_state.to_mapping()
        if self.encoded_cache_root is not None:
            mapping["encodedCacheRoot"] = self.encoded_cache_root
        return mapping


@dataclass(frozen=True)
class MiniMaxH3DatasetSettings:
    """Normalized H3 video/audio/caption dataset configuration."""

    root: str
    resolution: tuple[int, int]
    frame_count: int
    video_vae_state: MiniMaxH3ComponentStateSource
    audio_vae_state: MiniMaxH3ComponentStateSource
    conditioner_state: MiniMaxH3ComponentStateSource
    inspection: MiniMaxH3DatasetInspection
    encoded_cache_root: str | None = None

    @property
    def digest(self) -> str:
        return self.inspection.digest

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "type": "h3-video-audio-caption-folder",
            "root": self.root,
            "resolution": list(self.resolution),
            "frameCount": self.frame_count,
            "videoVaeState": self.video_vae_state.to_mapping(),
            "audioVaeState": self.audio_vae_state.to_mapping(),
            "conditionerState": self.conditioner_state.to_mapping(),
            "digest": self.digest,
        }
        if self.encoded_cache_root is not None:
            mapping["encodedCacheRoot"] = self.encoded_cache_root
        return mapping


@dataclass(frozen=True)
class WanDatasetSettings:
    """Normalized Wan video/caption dataset configuration."""

    root: str
    resolution: tuple[int, int]
    frame_count: int
    inspection: WanDatasetInspection
    encoded_cache_root: str | None = None
    vae_contract: WanVAEContract = "wan21"
    conditioning_contract: WanConditioningContract = "none"

    def __post_init__(self) -> None:
        if self.vae_contract not in ("wan21", "wan22"):
            raise ValueError("Wan dataset VAE contract must be 'wan21' or 'wan22'")
        if self.conditioning_contract not in ("none", "first-frame-i2v"):
            raise ValueError("unsupported Wan dataset conditioning contract")
        if self.conditioning_contract == "first-frame-i2v" and self.vae_contract != "wan21":
            raise ValueError("Wan first-frame I2V conditioning requires the Wan 2.1 VAE")

    @property
    def digest(self) -> str:
        return self.inspection.digest

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "type": (
                "wan22-video-caption-folder"
                if self.vae_contract == "wan22"
                else "wan21-video-caption-folder"
            ),
            "root": self.root,
            "resolution": list(self.resolution),
            "frameCount": self.frame_count,
            "digest": self.digest,
        }
        if self.encoded_cache_root is not None:
            mapping["encodedCacheRoot"] = self.encoded_cache_root
        return mapping


@dataclass(frozen=True)
class FluxDatasetSettings:
    """Normalized classic Flux image/caption dataset configuration."""

    root: str
    variant: Literal["flux1-dev", "flux1-schnell"]
    resolution: tuple[int, int]
    context_tokens: int
    inspection: DatasetInspection
    encoded_cache_root: str | None = None

    @property
    def digest(self) -> str:
        return self.inspection.digest

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "type": "flux-image-caption-folder",
            "variant": self.variant,
            "root": self.root,
            "resolution": list(self.resolution),
            "contextTokens": self.context_tokens,
            "digest": self.digest,
        }
        if self.encoded_cache_root is not None:
            mapping["encodedCacheRoot"] = self.encoded_cache_root
        return mapping


@dataclass(frozen=True)
class Flux2DatasetSettings:
    """Normalized Flux2 image/caption dataset configuration."""

    root: str
    variant: Literal["flux2-dev", "flux2-klein-9b", "flux2-klein-4b"]
    resolution: tuple[int, int]
    context_tokens: int
    inspection: DatasetInspection
    encoded_cache_root: str | None = None

    @property
    def digest(self) -> str:
        return self.inspection.digest

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "type": "flux2-image-caption-folder",
            "variant": self.variant,
            "root": self.root,
            "resolution": list(self.resolution),
            "contextTokens": self.context_tokens,
            "digest": self.digest,
        }
        if self.encoded_cache_root is not None:
            mapping["encodedCacheRoot"] = self.encoded_cache_root
        return mapping


@dataclass(frozen=True)
class QwenImageDatasetSettings:
    """Normalized fixed-resolution Qwen-Image image/caption dataset."""

    root: str
    resolution: tuple[int, int]
    context_tokens: int
    inspection: DatasetInspection
    encoded_cache_root: str | None = None

    def __post_init__(self) -> None:
        if type(self.root) is not str or not self.root or not Path(self.root).is_absolute():
            raise ValueError("Qwen-Image dataset root must be a non-empty absolute path")
        if (
            type(self.resolution) is not tuple
            or len(self.resolution) != 2
            or any(type(side) is not int or side < 16 or side % 16 for side in self.resolution)
        ):
            raise ValueError("Qwen-Image dataset resolution must be positive multiples of 16")
        if type(self.context_tokens) is not int or self.context_tokens < 512:
            raise ValueError("Qwen-Image dataset context tokens must be at least 512")
        if type(self.inspection) is not DatasetInspection:
            raise TypeError("Qwen-Image dataset inspection must be exact DatasetInspection")
        if self.encoded_cache_root is not None and (
            type(self.encoded_cache_root) is not str
            or not self.encoded_cache_root
            or not Path(self.encoded_cache_root).is_absolute()
        ):
            raise ValueError("Qwen-Image encoded cache root must be an absolute path or None")

    @property
    def digest(self) -> str:
        return self.inspection.digest

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "type": "qwen-image-image-caption-folder",
            "root": self.root,
            "resolution": list(self.resolution),
            "contextTokens": self.context_tokens,
            "digest": self.digest,
        }
        if self.encoded_cache_root is not None:
            mapping["encodedCacheRoot"] = self.encoded_cache_root
        return mapping


@dataclass(frozen=True)
class Ideogram4DatasetSettings:
    """Normalized fixed-resolution Ideogram 4 image dataset."""

    root: str
    role: Literal["conditional", "unconditional"]
    resolution: tuple[int, int]
    context_tokens: int
    inspection: DatasetInspection
    encoded_cache_root: str | None = None

    def __post_init__(self) -> None:
        if type(self.root) is not str or not self.root or not Path(self.root).is_absolute():
            raise ValueError("Ideogram 4 dataset root must be a non-empty absolute path")
        if self.role not in ("conditional", "unconditional"):
            raise ValueError("Ideogram 4 dataset role must be conditional or unconditional")
        if (
            type(self.resolution) is not tuple
            or len(self.resolution) != 2
            or any(type(side) is not int or side < 16 or side % 16 for side in self.resolution)
        ):
            raise ValueError("Ideogram 4 dataset resolution must be positive multiples of 16")
        expected_tokens = (
            self.context_tokens >= 1 if self.role == "conditional" else self.context_tokens == 0
        )
        if type(self.context_tokens) is not int or not expected_tokens:
            raise ValueError("Ideogram 4 dataset context tokens do not match its model role")
        if type(self.inspection) is not DatasetInspection:
            raise TypeError("Ideogram 4 dataset inspection must be exact DatasetInspection")
        if self.encoded_cache_root is not None and (
            type(self.encoded_cache_root) is not str
            or not self.encoded_cache_root
            or not Path(self.encoded_cache_root).is_absolute()
        ):
            raise ValueError("Ideogram 4 encoded cache root must be an absolute path or None")

    @property
    def digest(self) -> str:
        return self.inspection.digest

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "type": (
                "ideogram4-image-caption-folder"
                if self.role == "conditional"
                else "ideogram4-image-folder"
            ),
            "root": self.root,
            "resolution": list(self.resolution),
            "digest": self.digest,
        }
        if self.role == "conditional":
            mapping["contextTokens"] = self.context_tokens
        if self.encoded_cache_root is not None:
            mapping["encodedCacheRoot"] = self.encoded_cache_root
        return mapping


def inspect_image_caption_dataset(
    root: Path,
    resolution: tuple[int, int],
    vae_state: EncoderStateSource | None,
    text_encoder_state: EncoderStateSource | None,
    *,
    family: TrainingFamily = "sd15",
    checkpoint_state: EncoderStateSource | None = None,
) -> DatasetInspection:
    """Enumerate, hash, and validate one folder without encoding tensors."""
    if family == "sd15":
        if vae_state is None or text_encoder_state is None or checkpoint_state is not None:
            raise ValueError("SD1.5 datasets require VAE and CLIP-L state sources")
        encoder_identity: dict[str, object] = {
            "vaeState": vae_state.to_mapping(),
            "textEncoderState": text_encoder_state.to_mapping(),
        }
        latent_scale = 0.18215
        preprocessing = SD15_PREPROCESSING_IDENTITY
        family_identity: dict[str, object] = {}
    else:
        if checkpoint_state is None or vae_state is not None or text_encoder_state is not None:
            raise ValueError("SDXL datasets require one combined checkpoint state source")
        encoder_identity = {"checkpointState": checkpoint_state.to_mapping()}
        latent_scale = 0.13025
        preprocessing = SDXL_PREPROCESSING_IDENTITY
        family_identity = {"family": family}
    resolved = root.expanduser().resolve()
    errors: list[str] = []
    if not resolved.is_dir():
        errors.append(f"dataset root is not a directory: {resolved}")
        paths: list[Path] = []
    else:
        paths = sorted(
            (
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
    if not paths:
        errors.append("dataset contains no supported image files")

    items: list[ImageCaptionItem] = []
    caption_files = 0
    nonempty_captions = 0
    for image_path in paths:
        relative_image = image_path.relative_to(resolved).as_posix()
        caption_path = image_path.with_suffix(".txt")
        relative_caption = caption_path.relative_to(resolved).as_posix()
        image_data = image_path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as exc:
            errors.append(f"{relative_image}: cannot decode image: {exc}")

        caption_digest: str | None = None
        if not caption_path.is_file():
            errors.append(f"{relative_image}: missing caption {relative_caption}")
        else:
            caption_files += 1
            caption_data = caption_path.read_bytes()
            caption_digest = _digest(caption_data)
            try:
                caption = caption_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"{relative_caption}: caption is not UTF-8: {exc}")
            else:
                if caption.strip():
                    nonempty_captions += 1
                else:
                    errors.append(f"{relative_caption}: caption is empty")
        items.append(
            ImageCaptionItem(
                image_path=image_path,
                caption_path=caption_path,
                relative_image=relative_image,
                relative_caption=relative_caption,
                image_digest=_digest(image_data),
                caption_digest=caption_digest,
            )
        )

    identity = [
        "dinkster.image-caption-dataset.v1",
        [
            {
                "image": item.relative_image,
                "imageDigest": item.image_digest,
                "caption": item.relative_caption,
                "captionDigest": item.caption_digest,
            }
            for item in items
        ],
        {
            **family_identity,
            "resolution": list(resolution),
            "preprocessing": preprocessing,
            "tokenizer": TOKENIZER_IDENTITY,
            "latentScale": latent_scale,
            **encoder_identity,
        },
    ]
    return DatasetInspection(
        digest=_digest(_canonical_json(identity)),
        items=tuple(items),
        caption_files=caption_files,
        nonempty_captions=nonempty_captions,
        errors=tuple(errors),
    )


def inspect_flux_dataset(
    root: Path,
    resolution: tuple[int, int],
    context_tokens: int,
    *,
    variant: Literal["flux1-dev", "flux1-schnell"],
    vae_identity: object,
    clip_l_identity: object,
    t5xxl_identity: object,
) -> DatasetInspection:
    """Enumerate, hash, and validate a classic Flux image/caption folder."""
    resolved = root.expanduser().resolve()
    errors: list[str] = []
    paths = (
        sorted(
            (
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
        if resolved.is_dir()
        else []
    )
    if not resolved.is_dir():
        errors.append(f"dataset root is not a directory: {resolved}")
    if not paths:
        errors.append("dataset contains no supported image files")
    items: list[ImageCaptionItem] = []
    caption_files = 0
    nonempty_captions = 0
    for image_path in paths:
        relative_image = image_path.relative_to(resolved).as_posix()
        caption_path = image_path.with_suffix(".txt")
        relative_caption = caption_path.relative_to(resolved).as_posix()
        image_data = image_path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as exc:
            errors.append(f"{relative_image}: cannot decode image: {exc}")
        caption_digest: str | None = None
        if not caption_path.is_file():
            errors.append(f"{relative_image}: missing caption {relative_caption}")
        else:
            caption_files += 1
            caption_data = caption_path.read_bytes()
            caption_digest = _digest(caption_data)
            try:
                caption = caption_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"{relative_caption}: caption is not UTF-8: {exc}")
            else:
                if caption.strip():
                    nonempty_captions += 1
                else:
                    errors.append(f"{relative_caption}: caption is empty")
        items.append(
            ImageCaptionItem(
                image_path=image_path,
                caption_path=caption_path,
                relative_image=relative_image,
                relative_caption=relative_caption,
                image_digest=_digest(image_data),
                caption_digest=caption_digest,
            )
        )
    identity = [
        "dinkster.flux-image-caption-dataset.v1",
        [
            {
                "image": item.relative_image,
                "imageDigest": item.image_digest,
                "caption": item.relative_caption,
                "captionDigest": item.caption_digest,
            }
            for item in items
        ],
        {
            "variant": variant,
            "resolution": list(resolution),
            "contextTokens": context_tokens,
            "preprocessing": FLUX_PREPROCESSING_IDENTITY,
            "clipTokenizer": FLUX_CLIP_TOKENIZER_IDENTITY,
            "t5Tokenizer": FLUX_T5_TOKENIZER_IDENTITY,
            "latentScale": 0.3611,
            "latentShift": 0.1159,
            "vaeState": vae_identity,
            "clipLState": clip_l_identity,
            "t5xxlState": t5xxl_identity,
        },
    ]
    return DatasetInspection(
        digest=_digest(_canonical_json(identity)),
        items=tuple(items),
        caption_files=caption_files,
        nonempty_captions=nonempty_captions,
        errors=tuple(errors),
    )


def inspect_flux2_dataset(
    root: Path,
    resolution: tuple[int, int],
    context_tokens: int,
    *,
    variant: Literal["flux2-dev", "flux2-klein-9b", "flux2-klein-4b"],
    vae_identity: object,
    text_encoder_identity: object,
) -> DatasetInspection:
    """Enumerate, hash, and validate a Flux2 image/caption folder."""
    resolved = root.expanduser().resolve()
    errors: list[str] = []
    paths = (
        sorted(
            (
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
        if resolved.is_dir()
        else []
    )
    if not resolved.is_dir():
        errors.append(f"dataset root is not a directory: {resolved}")
    if not paths:
        errors.append("dataset contains no supported image files")
    items: list[ImageCaptionItem] = []
    caption_files = 0
    nonempty_captions = 0
    for image_path in paths:
        relative_image = image_path.relative_to(resolved).as_posix()
        caption_path = image_path.with_suffix(".txt")
        relative_caption = caption_path.relative_to(resolved).as_posix()
        image_data = image_path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as exc:
            errors.append(f"{relative_image}: cannot decode image: {exc}")
        caption_digest: str | None = None
        if not caption_path.is_file():
            errors.append(f"{relative_image}: missing caption {relative_caption}")
        else:
            caption_files += 1
            caption_data = caption_path.read_bytes()
            caption_digest = _digest(caption_data)
            try:
                caption = caption_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"{relative_caption}: caption is not UTF-8: {exc}")
            else:
                if caption.strip():
                    nonempty_captions += 1
                else:
                    errors.append(f"{relative_caption}: caption is empty")
        items.append(
            ImageCaptionItem(
                image_path=image_path,
                caption_path=caption_path,
                relative_image=relative_image,
                relative_caption=relative_caption,
                image_digest=_digest(image_data),
                caption_digest=caption_digest,
            )
        )
    tokenizer = (
        FLUX2_DEV_TOKENIZER_IDENTITY if variant == "flux2-dev" else FLUX2_KLEIN_TOKENIZER_IDENTITY
    )
    identity = [
        "dinkster.flux2-image-caption-dataset.v1",
        [
            {
                "image": item.relative_image,
                "imageDigest": item.image_digest,
                "caption": item.relative_caption,
                "captionDigest": item.caption_digest,
            }
            for item in items
        ],
        {
            "variant": variant,
            "resolution": list(resolution),
            "contextTokens": context_tokens,
            "preprocessing": FLUX2_PREPROCESSING_IDENTITY,
            "tokenizer": tokenizer,
            "latentContract": FLUX2_LATENT_IDENTITY,
            "textEncodingDtype": FLUX2_TEXT_ENCODING_DTYPE,
            "vaeState": vae_identity,
            "textEncoderState": text_encoder_identity,
        },
    ]
    return DatasetInspection(
        digest=_digest(_canonical_json(identity)),
        items=tuple(items),
        caption_files=caption_files,
        nonempty_captions=nonempty_captions,
        errors=tuple(errors),
    )


def inspect_qwen_image_dataset(
    root: Path,
    resolution: tuple[int, int],
    context_tokens: int,
    *,
    vae_identity: object,
    text_encoder_identity: object,
) -> DatasetInspection:
    """Enumerate, hash, and validate a Qwen-Image image/caption folder."""
    resolved = root.expanduser().resolve()
    errors: list[str] = []
    paths = (
        sorted(
            (
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
        if resolved.is_dir()
        else []
    )
    if not resolved.is_dir():
        errors.append(f"dataset root is not a directory: {resolved}")
    if not paths:
        errors.append("dataset contains no supported image files")
    items: list[ImageCaptionItem] = []
    caption_files = 0
    nonempty_captions = 0
    for image_path in paths:
        relative_image = image_path.relative_to(resolved).as_posix()
        caption_path = image_path.with_suffix(".txt")
        relative_caption = caption_path.relative_to(resolved).as_posix()
        image_data = image_path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as exc:
            errors.append(f"{relative_image}: cannot decode image: {exc}")
        caption_digest: str | None = None
        if not caption_path.is_file():
            errors.append(f"{relative_image}: missing caption {relative_caption}")
        else:
            caption_files += 1
            caption_data = caption_path.read_bytes()
            caption_digest = _digest(caption_data)
            try:
                caption = caption_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"{relative_caption}: caption is not UTF-8: {exc}")
            else:
                if caption.strip():
                    nonempty_captions += 1
                else:
                    errors.append(f"{relative_caption}: caption is empty")
        items.append(
            ImageCaptionItem(
                image_path=image_path,
                caption_path=caption_path,
                relative_image=relative_image,
                relative_caption=relative_caption,
                image_digest=_digest(image_data),
                caption_digest=caption_digest,
            )
        )
    identity = [
        "dinkster.qwen-image-image-caption-dataset.v1",
        [
            {
                "image": item.relative_image,
                "imageDigest": item.image_digest,
                "caption": item.relative_caption,
                "captionDigest": item.caption_digest,
            }
            for item in items
        ],
        {
            "family": QWEN_IMAGE_FAMILY_IDENTITY,
            "resolution": list(resolution),
            "contextTokens": context_tokens,
            "preprocessing": QWEN_IMAGE_PREPROCESSING_IDENTITY,
            "tokenizer": QWEN_IMAGE_TOKENIZER_IDENTITY,
            "promptTemplate": QWEN_IMAGE_TEXT_CONFIG.text_template,
            "latentContract": QWEN_IMAGE_LATENT_IDENTITY,
            "contextContract": QWEN_IMAGE_CONTEXT_IDENTITY,
            "padding": QWEN_IMAGE_PADDING_IDENTITY,
            "positionAlignment": QWEN_IMAGE_POSITION_IDENTITY,
            "attentionMask": QWEN_IMAGE_MASK_IDENTITY,
            "textEncodingDtype": QWEN_IMAGE_TEXT_ENCODING_DTYPE,
            "vaeState": vae_identity,
            "textEncoderState": text_encoder_identity,
        },
    ]
    return DatasetInspection(
        digest=_digest(_canonical_json(identity)),
        items=tuple(items),
        caption_files=caption_files,
        nonempty_captions=nonempty_captions,
        errors=tuple(errors),
    )


def inspect_ideogram4_dataset(
    root: Path,
    resolution: tuple[int, int],
    context_tokens: int,
    *,
    role: Literal["conditional", "unconditional"],
    vae_identity: object,
    text_encoder_identity: object | None,
) -> DatasetInspection:
    """Enumerate, hash, and validate one role-bound Ideogram 4 image dataset."""
    if role == "conditional" and text_encoder_identity is None:
        raise ValueError("conditional Ideogram 4 datasets require a text encoder identity")
    if role == "unconditional" and (context_tokens != 0 or text_encoder_identity is not None):
        raise ValueError("unconditional Ideogram 4 datasets cannot carry text conditioning")
    resolved = root.expanduser().resolve()
    errors: list[str] = []
    paths = (
        sorted(
            (
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
        if resolved.is_dir()
        else []
    )
    if not resolved.is_dir():
        errors.append(f"dataset root is not a directory: {resolved}")
    if not paths:
        errors.append("dataset contains no supported image files")
    items: list[ImageCaptionItem] = []
    caption_files = 0
    nonempty_captions = 0
    for image_path in paths:
        relative_image = image_path.relative_to(resolved).as_posix()
        caption_path = image_path.with_suffix(".txt")
        relative_caption = caption_path.relative_to(resolved).as_posix()
        image_data = image_path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_data)) as image:
                ImageOps.exif_transpose(image).convert("RGB").load()
        except Exception as exc:
            errors.append(f"{relative_image}: cannot decode image: {exc}")
        caption_digest: str | None = None
        if caption_path.is_file():
            caption_files += 1
            caption_data = caption_path.read_bytes()
            caption_digest = _digest(caption_data)
            try:
                caption = caption_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"{relative_caption}: caption is not UTF-8: {exc}")
            else:
                if caption.strip():
                    nonempty_captions += 1
                elif role == "conditional":
                    errors.append(f"{relative_caption}: caption is empty")
        elif role == "conditional":
            errors.append(f"{relative_image}: missing caption {relative_caption}")
        items.append(
            ImageCaptionItem(
                image_path=image_path,
                caption_path=caption_path,
                relative_image=relative_image,
                relative_caption=relative_caption,
                image_digest=_digest(image_data),
                caption_digest=caption_digest,
            )
        )
    identity = [
        "dinkster.ideogram4-image-dataset.v1",
        [
            {
                "image": item.relative_image,
                "imageDigest": item.image_digest,
                **(
                    {
                        "caption": item.relative_caption,
                        "captionDigest": item.caption_digest,
                    }
                    if role == "conditional"
                    else {}
                ),
            }
            for item in items
        ],
        {
            "family": "dinkster.ideogram4",
            "role": role,
            "resolution": list(resolution),
            "contextTokens": context_tokens,
            "preprocessing": IDEOGRAM4_PREPROCESSING_IDENTITY,
            "tokenizer": IDEOGRAM4_TOKENIZER_IDENTITY if role == "conditional" else None,
            "latentContract": IDEOGRAM4_LATENT_IDENTITY,
            "contextContract": IDEOGRAM4_CONTEXT_IDENTITY if role == "conditional" else None,
            "textEncodingDtype": (IDEOGRAM4_TEXT_ENCODING_DTYPE if role == "conditional" else None),
            "vaeState": vae_identity,
            "textEncoderState": text_encoder_identity,
        },
    ]
    return DatasetInspection(
        digest=_digest(_canonical_json(identity)),
        items=tuple(items),
        caption_files=caption_files,
        nonempty_captions=nonempty_captions,
        errors=tuple(errors),
    )


def inspect_wan_dataset(
    root: Path,
    resolution: tuple[int, int],
    frame_count: int,
    *,
    vae_identity: object,
    umt5xxl_identity: object,
    vae_contract: WanVAEContract = "wan21",
    conditioning_contract: WanConditioningContract = "none",
    validate_media: bool = True,
) -> WanDatasetInspection:
    """Enumerate, hash, and validate one Wan video/caption folder."""
    if vae_contract not in ("wan21", "wan22"):
        raise ValueError("Wan dataset VAE contract must be 'wan21' or 'wan22'")
    if conditioning_contract not in ("none", "first-frame-i2v"):
        raise ValueError("unsupported Wan dataset conditioning contract")
    resolved = root.expanduser().resolve()
    errors: list[str] = []
    if not resolved.is_dir():
        errors.append(f"dataset root is not a directory: {resolved}")
        paths: list[Path] = []
    else:
        paths = sorted(
            (
                path
                for path in resolved.rglob("*")
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            ),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
    if not paths:
        errors.append("dataset contains no supported video files")

    grouped: dict[Path, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.with_suffix(""), []).append(path)
    for stem, candidates in grouped.items():
        if len(candidates) > 1:
            relative = stem.relative_to(resolved).as_posix()
            errors.append(f"{relative}: multiple video containers share one caption basename")

    items: list[WanDatasetItem] = []
    caption_files = 0
    nonempty_captions = 0
    for video_path in paths:
        relative_video = video_path.relative_to(resolved).as_posix()
        caption_path = video_path.with_suffix(".txt")
        relative_caption = caption_path.relative_to(resolved).as_posix()
        video_data = video_path.read_bytes()
        if validate_media:
            try:
                decode_video_rgb24(
                    video_data,
                    resolution=resolution,
                    frame_count=frame_count,
                    collect=False,
                )
            except (RuntimeError, ValueError) as exc:
                errors.append(f"{relative_video}: cannot decode required video grid: {exc}")
        caption_digest: str | None = None
        if not caption_path.is_file():
            errors.append(f"{relative_video}: missing caption {relative_caption}")
        else:
            caption_files += 1
            caption_data = caption_path.read_bytes()
            caption_digest = _digest(caption_data)
            try:
                caption = caption_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"{relative_caption}: caption is not UTF-8: {exc}")
            else:
                if caption.strip():
                    nonempty_captions += 1
                else:
                    errors.append(f"{relative_caption}: caption is empty")
        items.append(
            WanDatasetItem(
                video_path=video_path,
                caption_path=caption_path,
                relative_video=relative_video,
                relative_caption=relative_caption,
                video_digest=_digest(video_data),
                caption_digest=caption_digest,
            )
        )

    identity = [
        (
            "dinkster.wan22-video-caption-dataset.v1"
            if vae_contract == "wan22"
            else "dinkster.wan21-video-caption-dataset.v1"
        ),
        [
            {
                "video": item.relative_video,
                "videoDigest": item.video_digest,
                "caption": item.relative_caption,
                "captionDigest": item.caption_digest,
            }
            for item in items
        ],
        {
            "resolution": list(resolution),
            "frameCount": frame_count,
            "preprocessing": (
                WAN22_PREPROCESSING_IDENTITY
                if vae_contract == "wan22"
                else WAN21_PREPROCESSING_IDENTITY
            ),
            "containerDecoder": MINIMAX_H3_CONTAINER_DECODER_IDENTITY,
            "tokenizer": WAN21_TOKENIZER_IDENTITY,
            "vaeState": vae_identity,
            "umt5xxlState": umt5xxl_identity,
            **({"conditioning": conditioning_contract} if conditioning_contract != "none" else {}),
        },
    ]
    return WanDatasetInspection(
        digest=_digest(_canonical_json(identity)),
        items=tuple(items),
        caption_files=caption_files,
        nonempty_captions=nonempty_captions,
        errors=tuple(errors),
    )


def minimax_h3_audio_sample_count(frame_count: int) -> int:
    latent_frames = round(
        frame_count / MINIMAX_H3_CONFIG.video_fps * MINIMAX_H3_CONFIG.audio_latent_rate_hz
    )
    return latent_frames * (
        MINIMAX_H3_CONFIG.audio_sample_rate_hz // MINIMAX_H3_CONFIG.audio_latent_rate_hz
    )


def minimax_h3_video_latent_shape(
    frame_count: int, resolution: tuple[int, int]
) -> tuple[int, int, int, int, int]:
    latent_frames = ((frame_count - 5) // 17) * 5 + 2
    height, width = resolution
    return (
        1,
        MINIMAX_H3_CONFIG.video_latent_channels,
        latent_frames,
        (height + MINIMAX_H3_CONFIG.video_spatial_downscale - 1)
        // MINIMAX_H3_CONFIG.video_spatial_downscale,
        (width + MINIMAX_H3_CONFIG.video_spatial_downscale - 1)
        // MINIMAX_H3_CONFIG.video_spatial_downscale,
    )


def minimax_h3_audio_latent_shape(frame_count: int) -> tuple[int, int, int, int]:
    return (
        1,
        MINIMAX_H3_CONFIG.audio_latent_channels,
        MINIMAX_H3_CONFIG.audio_content_channels,
        round(frame_count / MINIMAX_H3_CONFIG.video_fps * MINIMAX_H3_CONFIG.audio_latent_rate_hz),
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _inspect_h3_audio(
    path: Path,
    relative: str,
    *,
    frame_count: int,
    errors: list[str],
) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            samples = audio.getnframes()
            compression = audio.getcomptype()
            payload = audio.readframes(samples)
    except (EOFError, OSError, wave.Error) as exc:
        errors.append(f"{relative}: cannot decode PCM WAV audio: {exc}")
        return
    if compression != "NONE":
        errors.append(f"{relative}: audio must use uncompressed PCM")
    if channels != MINIMAX_H3_CONFIG.audio_content_channels:
        errors.append(
            f"{relative}: audio must have {MINIMAX_H3_CONFIG.audio_content_channels} channels"
        )
    if sample_width not in (1, 2, 3, 4):
        errors.append(f"{relative}: PCM sample width must be 8, 16, 24, or 32 bits")
    if sample_rate != MINIMAX_H3_CONFIG.audio_sample_rate_hz:
        errors.append(
            f"{relative}: audio sample rate must be {MINIMAX_H3_CONFIG.audio_sample_rate_hz} Hz"
        )
    expected_samples = minimax_h3_audio_sample_count(frame_count)
    if samples != expected_samples:
        errors.append(f"{relative}: audio has {samples} samples; expected {expected_samples}")
    expected_bytes = samples * channels * sample_width
    if len(payload) != expected_bytes:
        errors.append(
            f"{relative}: PCM payload has {len(payload)} bytes; expected {expected_bytes}"
        )


def inspect_minimax_h3_dataset(
    root: Path,
    resolution: tuple[int, int],
    frame_count: int,
    video_vae_state: MiniMaxH3ComponentStateSource,
    audio_vae_state: MiniMaxH3ComponentStateSource,
    conditioner_state: MiniMaxH3ComponentStateSource,
    *,
    validate_media: bool = True,
) -> MiniMaxH3DatasetInspection:
    """Enumerate, hash, and validate one H3 media dataset."""
    resolved = root.expanduser().resolve()
    errors: list[str] = []
    folder_candidates: set[Path] = set()
    container_groups: dict[Path, list[Path]] = {}
    if not resolved.is_dir():
        errors.append(f"dataset root is not a directory: {resolved}")
    else:
        for path in resolved.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                container_groups.setdefault(path.with_suffix(""), []).append(path)
            elif path.is_dir() and path.name == "frames":
                folder_candidates.add(path.parent)
        for path in resolved.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() == ".wav"
                and path.with_suffix("") not in container_groups
            ):
                folder_candidates.add(path.parent)
    item_keys = sorted(
        folder_candidates | set(container_groups), key=lambda path: _relative(path, resolved)
    )
    if not item_keys:
        errors.append("dataset contains no H3 frame-folder or container-video items")

    items: list[MiniMaxH3DatasetItem] = []
    caption_files = 0
    nonempty_captions = 0
    frame_folder_items = 0
    container_items = 0
    embedded_audio_items = 0
    sidecar_audio_items = 0
    target_height, target_width = resolution
    for item_key in item_keys:
        relative_item = _relative(item_key, resolved)
        container_paths = sorted(container_groups.get(item_key, ()), key=lambda path: path.name)
        has_folder = item_key in folder_candidates
        if has_folder and container_paths:
            errors.append(
                f"{relative_item}: ambiguous H3 item has both a frame folder and a container video"
            )
        if len(container_paths) > 1:
            errors.append(
                f"{relative_item}: ambiguous H3 item has multiple container videos: "
                + ", ".join(_relative(path, resolved) for path in container_paths)
            )

        if container_paths and not has_folder:
            container_items += 1
            container_path = container_paths[0]
            relative_container = _relative(container_path, resolved)
            container_data = container_path.read_bytes()
            container_digest = _digest(container_data)
            if validate_media:
                try:
                    decode_video_rgb24(
                        container_data,
                        resolution=resolution,
                        frame_count=frame_count,
                        collect=False,
                    )
                except (RuntimeError, ValueError) as exc:
                    errors.append(f"{relative_container}: {exc}")

            sidecar_audio_paths = sorted(
                (
                    path
                    for path in container_path.parent.iterdir()
                    if path.is_file()
                    and path.stem == container_path.stem
                    and path.suffix.lower() == ".wav"
                ),
                key=lambda path: path.name,
            )
            if len(sidecar_audio_paths) > 1:
                errors.append(
                    f"{relative_item}: multiple same-basename sidecar WAV files: "
                    + ", ".join(_relative(path, resolved) for path in sidecar_audio_paths)
                )
            if sidecar_audio_paths:
                sidecar_audio_items += 1
                audio_kind: Literal["wav", "container"] = "wav"
                audio_path = sidecar_audio_paths[0]
                relative_audio = _relative(audio_path, resolved)
                audio_data = audio_path.read_bytes()
                audio_digest = _digest(audio_data)
                if validate_media:
                    _inspect_h3_audio(
                        audio_path,
                        relative_audio,
                        frame_count=frame_count,
                        errors=errors,
                    )
            else:
                embedded_audio_items += 1
                audio_kind = "container"
                audio_path = container_path
                relative_audio = relative_container
                audio_digest = container_digest
                if validate_media:
                    try:
                        decoded_audio = decode_audio_s16_stereo_32k(
                            container_data,
                            sample_count=minimax_h3_audio_sample_count(frame_count),
                            collect=False,
                        )
                    except (RuntimeError, ValueError) as exc:
                        errors.append(f"{relative_container}: {exc}")
                    else:
                        if decoded_audio is None:
                            errors.append(
                                f"{relative_container}: container has no audio stream and no "
                                "same-basename sidecar WAV"
                            )

            caption_path = container_path.with_suffix(".txt")
            relative_caption = _relative(caption_path, resolved)
            video_kind: Literal["frames", "container"] = "container"
            video_path = container_path
            relative_frames: list[str] = []
            frame_digests: list[str] = []
        else:
            frame_folder_items += 1
            item_path = item_key
            frames_path = item_path / "frames"
            video_kind = "frames"
            video_path = frames_path
            relative_container = None
            container_digest = None
            audio_kind = "wav"
            sidecar_audio_items += 1
            frame_paths: list[Path] = []
            if not frames_path.is_dir():
                errors.append(f"{relative_item}: missing frames directory")
            else:
                frame_files = sorted(
                    (path for path in frames_path.iterdir() if path.is_file()),
                    key=lambda path: path.name,
                )
                unsupported = [
                    _relative(path, resolved)
                    for path in frame_files
                    if path.suffix.lower() not in IMAGE_EXTENSIONS
                ]
                if unsupported:
                    errors.append(
                        f"{relative_item}: frames directory has unsupported files:"
                        f" {', '.join(unsupported)}"
                    )
                frame_paths = [
                    path for path in frame_files if path.suffix.lower() in IMAGE_EXTENSIONS
                ]
                frame_stems = [path.stem for path in frame_paths]
                if frame_stems and (
                    any(not stem.isascii() or not stem.isdecimal() for stem in frame_stems)
                    or len({len(stem) for stem in frame_stems}) != 1
                    or any(
                        int(current) != int(previous) + 1
                        for previous, current in zip(frame_stems, frame_stems[1:], strict=False)
                    )
                ):
                    errors.append(
                        f"{relative_item}: frame image names must have equal-width consecutive"
                        " decimal stems"
                    )
            if len(frame_paths) != frame_count:
                errors.append(
                    f"{relative_item}: frames directory has {len(frame_paths)} images;"
                    f" expected {frame_count}"
                )
            frame_digests = []
            relative_frames = []
            for frame_path in frame_paths:
                relative_frame = _relative(frame_path, resolved)
                data = frame_path.read_bytes()
                relative_frames.append(relative_frame)
                frame_digests.append(_digest(data))
                if not validate_media:
                    continue
                try:
                    with Image.open(io.BytesIO(data)) as opened:
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                        image.load()
                        size = image.size
                except Exception as exc:
                    errors.append(f"{relative_frame}: cannot decode image: {exc}")
                else:
                    if size != (target_width, target_height):
                        errors.append(
                            f"{relative_frame}: frame resolution is {size[1]}x{size[0]};"
                            f" expected {target_height}x{target_width}"
                        )

            audio_paths = sorted(
                (
                    path
                    for path in item_path.iterdir()
                    if path.is_file() and path.suffix.lower() == ".wav"
                ),
                key=lambda path: path.name,
            )
            if len(audio_paths) != 1:
                errors.append(
                    f"{relative_item}: item must contain exactly one WAV audio file;"
                    f" found {len(audio_paths)}"
                )
            audio_path = audio_paths[0] if audio_paths else item_path / f"{item_path.name}.wav"
            relative_audio = _relative(audio_path, resolved)
            audio_digest = None
            if audio_path.is_file():
                audio_data = audio_path.read_bytes()
                audio_digest = _digest(audio_data)
                if validate_media:
                    _inspect_h3_audio(
                        audio_path,
                        relative_audio,
                        frame_count=frame_count,
                        errors=errors,
                    )

            caption_path = audio_path.with_suffix(".txt")
            relative_caption = _relative(caption_path, resolved)

        caption_digest: str | None = None
        if not caption_path.is_file():
            errors.append(f"{relative_audio}: missing caption {relative_caption}")
        else:
            caption_files += 1
            caption_data = caption_path.read_bytes()
            caption_digest = _digest(caption_data)
            try:
                caption = caption_data.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(f"{relative_caption}: caption is not UTF-8: {exc}")
            else:
                if caption.strip():
                    nonempty_captions += 1
                else:
                    errors.append(f"{relative_caption}: caption is empty")
        items.append(
            MiniMaxH3DatasetItem(
                video_kind=video_kind,
                audio_kind=audio_kind,
                video_path=video_path,
                audio_path=audio_path,
                caption_path=caption_path,
                relative_item=relative_item,
                relative_frames=tuple(relative_frames),
                frame_digests=tuple(frame_digests),
                relative_container=relative_container,
                container_digest=container_digest,
                relative_audio=relative_audio,
                audio_digest=audio_digest,
                relative_caption=relative_caption,
                caption_digest=caption_digest,
            )
        )

    identity_facts = {
        "resolution": list(resolution),
        "frameCount": frame_count,
        "videoFps": MINIMAX_H3_CONFIG.video_fps,
        "audioSampleRate": MINIMAX_H3_CONFIG.audio_sample_rate_hz,
        "preprocessing": MINIMAX_H3_PREPROCESSING_IDENTITY,
        "videoVaeState": video_vae_state.to_mapping(),
        "audioVaeState": audio_vae_state.to_mapping(),
        "conditionerState": conditioner_state.to_mapping(),
    }
    if container_items:
        identity = [
            "dinkster.minimax-h3-video-audio-caption-dataset.v2",
            [
                (
                    {
                        "item": item.relative_item,
                        "kind": "container",
                        "container": item.relative_container,
                        "containerDigest": item.container_digest,
                        "audio": item.relative_audio,
                        "audioDigest": item.audio_digest,
                        "audioKind": item.audio_kind,
                        "caption": item.relative_caption,
                        "captionDigest": item.caption_digest,
                    }
                    if item.video_kind == "container"
                    else {
                        "item": item.relative_item,
                        "kind": "frames",
                        "frames": [
                            {"path": path, "digest": digest}
                            for path, digest in zip(
                                item.relative_frames, item.frame_digests, strict=True
                            )
                        ],
                        "audio": item.relative_audio,
                        "audioDigest": item.audio_digest,
                        "caption": item.relative_caption,
                        "captionDigest": item.caption_digest,
                    }
                )
                for item in items
            ],
            {
                **identity_facts,
                "containerDecoder": MINIMAX_H3_CONTAINER_DECODER_IDENTITY,
            },
        ]
    else:
        identity = [
            "dinkster.minimax-h3-video-audio-caption-dataset.v1",
            [
                {
                    "item": item.relative_item,
                    "frames": [
                        {"path": path, "digest": digest}
                        for path, digest in zip(
                            item.relative_frames, item.frame_digests, strict=True
                        )
                    ],
                    "audio": item.relative_audio,
                    "audioDigest": item.audio_digest,
                    "caption": item.relative_caption,
                    "captionDigest": item.caption_digest,
                }
                for item in items
            ],
            identity_facts,
        ]
    return MiniMaxH3DatasetInspection(
        digest=_digest(_canonical_json(identity)),
        items=tuple(items),
        caption_files=caption_files,
        nonempty_captions=nonempty_captions,
        frame_folder_items=frame_folder_items,
        container_items=container_items,
        embedded_audio_items=embedded_audio_items,
        sidecar_audio_items=sidecar_audio_items,
        errors=tuple(errors),
    )
