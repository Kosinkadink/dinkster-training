"""Content-addressed disk cache for encoded image/caption datasets."""

from __future__ import annotations

import json
import platform
import struct
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from dinkster_inference import SafetensorsSource, load_safetensors_header
from dinkster_inference_torch import load_tensors

from .checkpoint import blake3_digest, canonical_json
from .dataset import (
    Flux2DatasetSettings,
    FluxDatasetSettings,
    Ideogram4DatasetSettings,
    ImageCaptionDatasetSettings,
    MiniMaxH3DatasetSettings,
    QwenImageDatasetSettings,
    WanDatasetSettings,
    minimax_h3_audio_latent_shape,
    minimax_h3_video_latent_shape,
)
from .durability import advisory_file_lock, atomic_replace, durable_mkdir

_FORMAT = "dinkster.encoded-dataset.v1"
_H3_FORMAT = "dinkster.encoded-h3-dataset.v1"
_WAN_FORMAT = "dinkster.encoded-wan21-dataset.v1"
_WAN22_FORMAT = "dinkster.encoded-wan22-dataset.v1"
_WAN22_I2V_FORMAT = "dinkster.encoded-wan22-i2v-dataset.v1"
_FLUX_FORMAT = "dinkster.encoded-flux1-dataset.v1"
_FLUX2_FORMAT = "dinkster.encoded-flux2-dataset.v1"
_QWEN_IMAGE_FORMAT = "dinkster.encoded-qwen-image-dataset.v1"
_IDEOGRAM4_FORMAT = "dinkster.encoded-ideogram4-dataset.v1"


def _wan_cache_contract(settings: WanDatasetSettings) -> tuple[str, str, str]:
    if settings.vae_contract == "wan22":
        return _WAN22_FORMAT, "wan22-ti2v", "wan22-v1"
    if settings.conditioning_contract == "first-frame-i2v":
        return _WAN22_I2V_FORMAT, "wan22-i2v", "wan22-i2v-v1"
    return _WAN_FORMAT, "wan21-t2v", "wan21-v1"


@dataclass(frozen=True)
class EncodedDatasetTensors:
    latents: torch.Tensor
    text_embeddings: torch.Tensor
    pooled_embeddings: torch.Tensor | None
    time_ids: torch.Tensor | None


@dataclass(frozen=True)
class EncodedMiniMaxH3DatasetTensors:
    video_latents: torch.Tensor
    audio_latents: torch.Tensor
    conditioner_embeddings: torch.Tensor
    conditioning_text_token_tags: torch.Tensor


@dataclass(frozen=True)
class EncodedWanDatasetTensors:
    latents: torch.Tensor
    context: torch.Tensor
    i2v_conditioning: torch.Tensor | None = None


@dataclass(frozen=True)
class EncodedFluxDatasetTensors:
    latents: torch.Tensor
    context: torch.Tensor
    pooled: torch.Tensor


@dataclass(frozen=True)
class EncodedFlux2DatasetTensors:
    latents: torch.Tensor
    context: torch.Tensor


@dataclass(frozen=True)
class EncodedQwenImageDatasetTensors:
    latents: torch.Tensor
    context: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class EncodedIdeogram4DatasetTensors:
    latents: torch.Tensor
    context: torch.Tensor | None
    attention_mask: torch.Tensor | None


def _execution_fingerprint(*, batch_size: int, device: torch.device) -> dict[str, object]:
    execution: dict[str, object] = {
        "batchSize": batch_size,
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "device": str(device),
        "float32MatmulPrecision": torch.get_float32_matmul_precision(),
        "torchVersion": str(torch.__version__),
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if str(device) == "cuda" else device.index
        properties = torch.cuda.get_device_properties(index)
        execution.update(
            {
                "cudaRuntime": str(torch.version.cuda),
                "cudnnBenchmark": torch.backends.cudnn.benchmark,
                "cudnnDeterministic": torch.backends.cudnn.deterministic,
                "device": "cuda",
                "deviceCapability": [properties.major, properties.minor],
                "deviceName": properties.name,
            }
        )
    else:
        execution["cpuCapability"] = torch.backends.cpu.get_cpu_capability()
        execution["machine"] = platform.machine()
    return execution


def encoded_cache_key(
    settings: ImageCaptionDatasetSettings,
    *,
    batch_size: int,
    device: torch.device,
) -> str:
    execution = _execution_fingerprint(batch_size=batch_size, device=device)
    return blake3_digest(canonical_json([_FORMAT, settings.digest, execution]))


def minimax_h3_encoded_cache_key(
    settings: MiniMaxH3DatasetSettings,
    *,
    device: torch.device,
) -> str:
    execution = _execution_fingerprint(batch_size=1, device=device)
    return blake3_digest(canonical_json([_H3_FORMAT, settings.digest, execution]))


def wan_encoded_cache_key(
    settings: WanDatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int, int],
    context_shape: tuple[int, int, int],
    device: torch.device,
) -> str:
    execution = _execution_fingerprint(batch_size=1, device=device)
    cache_format, _family, _root = _wan_cache_contract(settings)
    return blake3_digest(
        canonical_json(
            [cache_format, settings.digest, list(latent_shape), list(context_shape), execution]
        )
    )


def flux_encoded_cache_key(
    settings: FluxDatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int],
    context_shape: tuple[int, int, int],
    pooled_shape: tuple[int, int],
    device: torch.device,
) -> str:
    execution = _execution_fingerprint(batch_size=latent_shape[0], device=device)
    return blake3_digest(
        canonical_json(
            [
                _FLUX_FORMAT,
                settings.digest,
                list(latent_shape),
                list(context_shape),
                list(pooled_shape),
                execution,
            ]
        )
    )


def flux2_encoded_cache_key(
    settings: Flux2DatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int],
    context_shape: tuple[int, int, int],
    device: torch.device,
) -> str:
    execution = _execution_fingerprint(batch_size=latent_shape[0], device=device)
    return blake3_digest(
        canonical_json(
            [
                _FLUX2_FORMAT,
                settings.variant,
                settings.digest,
                list(latent_shape),
                list(context_shape),
                execution,
            ]
        )
    )


def qwen_image_encoded_cache_key(
    settings: QwenImageDatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int, int],
    context_shape: tuple[int, int, int],
    attention_mask_shape: tuple[int, int],
    device: torch.device,
) -> str:
    execution = _execution_fingerprint(batch_size=latent_shape[0], device=device)
    return blake3_digest(
        canonical_json(
            [
                _QWEN_IMAGE_FORMAT,
                settings.digest,
                list(latent_shape),
                list(context_shape),
                list(attention_mask_shape),
                execution,
            ]
        )
    )


def ideogram4_encoded_cache_key(
    settings: Ideogram4DatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int],
    context_shape: tuple[int, int, int] | None,
    attention_mask_shape: tuple[int, int] | None,
    device: torch.device,
) -> str:
    execution = _execution_fingerprint(batch_size=latent_shape[0], device=device)
    return blake3_digest(
        canonical_json(
            [
                _IDEOGRAM4_FORMAT,
                settings.digest,
                settings.role,
                list(latent_shape),
                None if context_shape is None else list(context_shape),
                None if attention_mask_shape is None else list(attention_mask_shape),
                execution,
            ]
        )
    )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous().resolve_neg()
    return value.numpy().tobytes()


def _safetensors_bytes(tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("encoded dataset caching requires a little-endian host")
    header: dict[str, object] = {"__metadata__": {key: metadata[key] for key in sorted(metadata)}}
    payload = bytearray()
    for key in sorted(tensors):
        tensor = tensors[key]
        dtype = {torch.float32: "F32", torch.int64: "I64"}.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"encoded cache tensor {key!r} must be float32 or int64")
        data = _tensor_bytes(tensor)
        begin = len(payload)
        payload.extend(data)
        header[key] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [begin, len(payload)],
        }
    raw_header = json.dumps(
        header,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    raw_header += b" " * (-len(raw_header) % 8)
    return struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload)


class EncodedDatasetCache:
    """One verified per-dataset safetensors shard behind an atomic manifest."""

    def __init__(
        self,
        settings: ImageCaptionDatasetSettings,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if settings.encoded_cache_root is None:
            raise ValueError("encoded dataset cache root is not configured")
        self._settings = settings
        self._key = encoded_cache_key(settings, batch_size=batch_size, device=device)
        root = Path(settings.encoded_cache_root) / "v1"
        self._entries = root / "entries"
        self._shards = root / "shards"
        self._manifest_path = self._entries / f"{self._key[7:]}.json"
        self._lock_path = root / "locks" / f"{self._key[7:]}.lock"

    def build_lock(self) -> AbstractContextManager[None]:
        return advisory_file_lock(self._lock_path)

    def _read_manifest(self) -> tuple[Path, str]:
        value = json.loads(self._manifest_path.read_bytes().decode("ascii"))
        if not isinstance(value, dict):
            raise ValueError("encoded cache manifest must be an object")
        manifest = cast("dict[str, object]", value)
        if set(manifest) != {
            "schemaVersion",
            "cacheKey",
            "datasetDigest",
            "shardDigest",
        }:
            raise ValueError("encoded cache manifest has an unknown layout")
        if (
            manifest["schemaVersion"] != 1
            or manifest["cacheKey"] != self._key
            or manifest["datasetDigest"] != self._settings.digest
        ):
            raise ValueError("encoded cache manifest identity does not match the dataset")
        shard_digest = manifest["shardDigest"]
        if (
            not isinstance(shard_digest, str)
            or not shard_digest.startswith("blake3:")
            or len(shard_digest) != 71
            or any(character not in "0123456789abcdef" for character in shard_digest[7:])
        ):
            raise ValueError("encoded cache manifest has an invalid shard digest")
        return self._shards / f"{shard_digest[7:]}.safetensors", shard_digest

    def _verified_source(self) -> SafetensorsSource:
        shard_path, expected_digest = self._read_manifest()
        data = shard_path.read_bytes()
        if blake3_digest(data) != expected_digest:
            raise ValueError("encoded cache shard failed digest verification")
        source = load_safetensors_header(shard_path)
        expected_metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": self._settings.family,
            "format": _FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        if dict(source.metadata()) != expected_metadata:
            raise ValueError("encoded cache shard metadata does not match the dataset")
        expected_keys = (
            {"latents", "text_embeddings"}
            if self._settings.family == "sd15"
            else {"latents", "text_embeddings", "pooled_embeddings", "time_ids"}
        )
        if set(source.keys()) != expected_keys:
            raise ValueError("encoded cache shard has an unknown tensor layout")
        item_count = len(self._settings.inspection.items)
        latent = source.entry("latents").geometry
        text = source.entry("text_embeddings").geometry
        expected_latent_shape = (
            item_count,
            4,
            self._settings.resolution[0] // 8,
            self._settings.resolution[1] // 8,
        )
        if latent.dtype.name != "float32" or latent.shape != expected_latent_shape:
            raise ValueError("encoded cache latent tensor has the wrong shape or dtype")
        if (
            text.dtype.name != "float32"
            or len(text.shape) != 3
            or text.shape[0] != item_count
            or text.shape[1] != 77
        ):
            raise ValueError("encoded cache text tensor has the wrong shape or dtype")
        if self._settings.family == "sdxl":
            pooled = source.entry("pooled_embeddings").geometry
            time_ids = source.entry("time_ids").geometry
            if (
                pooled.dtype.name != "float32"
                or len(pooled.shape) != 2
                or pooled.shape[0] != item_count
                or time_ids.dtype.name != "float32"
                or time_ids.shape != (6,)
            ):
                raise ValueError("encoded cache SDXL tensors have the wrong shape or dtype")
        return source

    def is_valid(self) -> bool:
        try:
            self._verified_source()
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def load(self) -> EncodedDatasetTensors | None:
        try:
            source = self._verified_source()
            tensors = load_tensors(source.path)
            latents = tensors["latents"]
            text = tensors["text_embeddings"]
            pooled = tensors.get("pooled_embeddings")
            time_ids = tensors.get("time_ids")
            if time_ids is not None:
                height, width = self._settings.resolution
                expected = torch.tensor([height, width, 0, 0, height, width], dtype=torch.float32)
                if not torch.equal(time_ids, expected):
                    raise ValueError("encoded cache SDXL time ids do not match the dataset")
            return EncodedDatasetTensors(latents, text, pooled, time_ids)
        except (OSError, ValueError, KeyError, RuntimeError):
            return None

    def write(self, tensors: EncodedDatasetTensors) -> None:
        values = {
            "latents": tensors.latents,
            "text_embeddings": tensors.text_embeddings,
        }
        if self._settings.family == "sdxl":
            if tensors.pooled_embeddings is None or tensors.time_ids is None:
                raise ValueError("SDXL encoded cache needs pooled embeddings and time ids")
            values["pooled_embeddings"] = tensors.pooled_embeddings
            values["time_ids"] = tensors.time_ids
        elif tensors.pooled_embeddings is not None or tensors.time_ids is not None:
            raise ValueError("SD1.5 encoded cache cannot contain SDXL tensors")
        metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": self._settings.family,
            "format": _FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        shard = _safetensors_bytes(values, metadata)
        shard_digest = blake3_digest(shard)
        manifest = canonical_json(
            {
                "schemaVersion": 1,
                "cacheKey": self._key,
                "datasetDigest": self._settings.digest,
                "shardDigest": shard_digest,
            }
        )
        durable_mkdir(self._shards)
        durable_mkdir(self._entries)
        atomic_replace(self._shards / f"{shard_digest[7:]}.safetensors", shard)
        atomic_replace(self._manifest_path, manifest)


class FluxEncodedDatasetCache:
    """Verified classic Flux latent, T5 context, and CLIP pooled cache."""

    def __init__(
        self,
        settings: FluxDatasetSettings,
        *,
        latent_shape: tuple[int, int, int, int],
        context_shape: tuple[int, int, int],
        pooled_shape: tuple[int, int],
        device: torch.device,
    ) -> None:
        if settings.encoded_cache_root is None:
            raise ValueError("encoded Flux dataset cache root is not configured")
        self._settings = settings
        self._shapes = (latent_shape, context_shape, pooled_shape)
        self._key = flux_encoded_cache_key(
            settings,
            latent_shape=latent_shape,
            context_shape=context_shape,
            pooled_shape=pooled_shape,
            device=device,
        )
        root = Path(settings.encoded_cache_root) / "flux1-v1"
        self._entries = root / "entries"
        self._shards = root / "shards"
        self._manifest_path = self._entries / f"{self._key[7:]}.json"
        self._lock_path = root / "locks" / f"{self._key[7:]}.lock"

    def build_lock(self) -> AbstractContextManager[None]:
        return advisory_file_lock(self._lock_path)

    def _verified_source(self) -> SafetensorsSource:
        value = json.loads(self._manifest_path.read_bytes().decode("ascii"))
        if not isinstance(value, dict):
            raise ValueError("encoded Flux cache manifest must be an object")
        manifest = cast("dict[str, object]", value)
        if set(manifest) != {"schemaVersion", "cacheKey", "datasetDigest", "shardDigest"}:
            raise ValueError("encoded Flux cache manifest has an unknown layout")
        if (
            manifest["schemaVersion"] != 1
            or manifest["cacheKey"] != self._key
            or manifest["datasetDigest"] != self._settings.digest
        ):
            raise ValueError("encoded Flux cache manifest identity does not match the dataset")
        digest = manifest["shardDigest"]
        if (
            not isinstance(digest, str)
            or not digest.startswith("blake3:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise ValueError("encoded Flux cache manifest has an invalid shard digest")
        path = self._shards / f"{digest[7:]}.safetensors"
        if blake3_digest(path.read_bytes()) != digest:
            raise ValueError("encoded Flux cache shard failed digest verification")
        source = load_safetensors_header(path)
        expected_metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "flux",
            "format": _FLUX_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        if dict(source.metadata()) != expected_metadata or set(source.keys()) != {
            "latents",
            "context",
            "pooled",
        }:
            raise ValueError("encoded Flux cache shard layout does not match the dataset")
        count = len(self._settings.inspection.items)
        for key, shape in zip(("latents", "context", "pooled"), self._shapes, strict=True):
            geometry = source.entry(key).geometry
            if geometry.dtype.name != "float32" or geometry.shape != (count, *shape[1:]):
                raise ValueError(f"encoded Flux {key} tensor has the wrong shape or dtype")
        return source

    def is_valid(self) -> bool:
        try:
            self._verified_source()
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def load(self) -> EncodedFluxDatasetTensors | None:
        try:
            tensors = load_tensors(self._verified_source().path)
            return EncodedFluxDatasetTensors(
                tensors["latents"], tensors["context"], tensors["pooled"]
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            return None

    def write(self, tensors: EncodedFluxDatasetTensors) -> str:
        count = len(self._settings.inspection.items)
        values = {"latents": tensors.latents, "context": tensors.context, "pooled": tensors.pooled}
        for key, shape in zip(("latents", "context", "pooled"), self._shapes, strict=True):
            tensor = values[key]
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != (count, *shape[1:]):
                raise ValueError(f"encoded Flux {key} tensor has the wrong shape or dtype")
        metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "flux",
            "format": _FLUX_FORMAT,
            "itemCount": str(count),
        }
        shard = _safetensors_bytes(values, metadata)
        digest = blake3_digest(shard)
        manifest = canonical_json(
            {
                "schemaVersion": 1,
                "cacheKey": self._key,
                "datasetDigest": self._settings.digest,
                "shardDigest": digest,
            }
        )
        durable_mkdir(self._shards)
        durable_mkdir(self._entries)
        atomic_replace(self._shards / f"{digest[7:]}.safetensors", shard)
        atomic_replace(self._manifest_path, manifest)
        return digest


class Flux2EncodedDatasetCache:
    """Verified Flux2 latent and variant-specific text context cache."""

    def __init__(
        self,
        settings: Flux2DatasetSettings,
        *,
        latent_shape: tuple[int, int, int, int],
        context_shape: tuple[int, int, int],
        device: torch.device,
    ) -> None:
        if settings.encoded_cache_root is None:
            raise ValueError("encoded Flux2 dataset cache root is not configured")
        self._settings = settings
        self._shapes = (latent_shape, context_shape)
        self._key = flux2_encoded_cache_key(
            settings,
            latent_shape=latent_shape,
            context_shape=context_shape,
            device=device,
        )
        root = Path(settings.encoded_cache_root) / "flux2-v1"
        self._entries = root / "entries"
        self._shards = root / "shards"
        self._manifest_path = self._entries / f"{self._key[7:]}.json"
        self._lock_path = root / "locks" / f"{self._key[7:]}.lock"

    def build_lock(self) -> AbstractContextManager[None]:
        return advisory_file_lock(self._lock_path)

    def _verified_source(self) -> SafetensorsSource:
        value = json.loads(self._manifest_path.read_bytes().decode("ascii"))
        if not isinstance(value, dict):
            raise ValueError("encoded Flux2 cache manifest must be an object")
        manifest = cast("dict[str, object]", value)
        if set(manifest) != {"schemaVersion", "cacheKey", "datasetDigest", "shardDigest"}:
            raise ValueError("encoded Flux2 cache manifest has an unknown layout")
        if (
            type(manifest["schemaVersion"]) is not int
            or manifest["schemaVersion"] != 1
            or manifest["cacheKey"] != self._key
            or manifest["datasetDigest"] != self._settings.digest
        ):
            raise ValueError("encoded Flux2 cache manifest identity does not match the dataset")
        digest = manifest["shardDigest"]
        if (
            not isinstance(digest, str)
            or not digest.startswith("blake3:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise ValueError("encoded Flux2 cache manifest has an invalid shard digest")
        path = self._shards / f"{digest[7:]}.safetensors"
        if blake3_digest(path.read_bytes()) != digest:
            raise ValueError("encoded Flux2 cache shard failed digest verification")
        source = load_safetensors_header(path)
        expected_metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "flux2",
            "format": _FLUX2_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
            "variant": self._settings.variant,
        }
        if dict(source.metadata()) != expected_metadata or set(source.keys()) != {
            "latents",
            "context",
        }:
            raise ValueError("encoded Flux2 cache shard layout does not match the dataset")
        count = len(self._settings.inspection.items)
        for key, shape in zip(("latents", "context"), self._shapes, strict=True):
            geometry = source.entry(key).geometry
            if geometry.dtype.name != "float32" or geometry.shape != (count, *shape[1:]):
                raise ValueError(f"encoded Flux2 {key} tensor has the wrong shape or dtype")
        return source

    def is_valid(self) -> bool:
        try:
            self._verified_source()
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def load(self) -> EncodedFlux2DatasetTensors | None:
        try:
            tensors = load_tensors(self._verified_source().path)
            return EncodedFlux2DatasetTensors(tensors["latents"], tensors["context"])
        except (OSError, ValueError, KeyError, RuntimeError):
            return None

    def write(self, tensors: EncodedFlux2DatasetTensors) -> str:
        count = len(self._settings.inspection.items)
        values = {"latents": tensors.latents, "context": tensors.context}
        for key, shape in zip(("latents", "context"), self._shapes, strict=True):
            tensor = values[key]
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != (count, *shape[1:]):
                raise ValueError(f"encoded Flux2 {key} tensor has the wrong shape or dtype")
        metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "flux2",
            "format": _FLUX2_FORMAT,
            "itemCount": str(count),
            "variant": self._settings.variant,
        }
        shard = _safetensors_bytes(values, metadata)
        digest = blake3_digest(shard)
        manifest = canonical_json(
            {
                "schemaVersion": 1,
                "cacheKey": self._key,
                "datasetDigest": self._settings.digest,
                "shardDigest": digest,
            }
        )
        durable_mkdir(self._shards)
        durable_mkdir(self._entries)
        atomic_replace(self._shards / f"{digest[7:]}.safetensors", shard)
        atomic_replace(self._manifest_path, manifest)
        return digest


class QwenImageEncodedDatasetCache:
    """Verified Qwen-Image latent, context, and attention-mask cache."""

    def __init__(
        self,
        settings: QwenImageDatasetSettings,
        *,
        latent_shape: tuple[int, int, int, int, int],
        context_shape: tuple[int, int, int],
        attention_mask_shape: tuple[int, int],
        device: torch.device,
    ) -> None:
        if settings.encoded_cache_root is None:
            raise ValueError("encoded Qwen-Image dataset cache root is not configured")
        self._settings = settings
        self._shapes = (latent_shape, context_shape, attention_mask_shape)
        self._key = qwen_image_encoded_cache_key(
            settings,
            latent_shape=latent_shape,
            context_shape=context_shape,
            attention_mask_shape=attention_mask_shape,
            device=device,
        )
        root = Path(settings.encoded_cache_root) / "qwen-image-v1"
        self._entries = root / "entries"
        self._shards = root / "shards"
        self._manifest_path = self._entries / f"{self._key[7:]}.json"
        self._lock_path = root / "locks" / f"{self._key[7:]}.lock"

    def build_lock(self) -> AbstractContextManager[None]:
        return advisory_file_lock(self._lock_path)

    def _read_manifest(self) -> tuple[Path, str, int]:
        value = json.loads(self._manifest_path.read_bytes().decode("ascii"))
        if not isinstance(value, dict):
            raise ValueError("encoded Qwen-Image cache manifest must be an object")
        manifest = cast("dict[str, object]", value)
        if set(manifest) != {
            "schemaVersion",
            "cacheKey",
            "datasetDigest",
            "shardDigest",
            "shardSize",
        }:
            raise ValueError("encoded Qwen-Image cache manifest has an unknown layout")
        if (
            type(manifest["schemaVersion"]) is not int
            or manifest["schemaVersion"] != 1
            or type(manifest["cacheKey"]) is not str
            or manifest["cacheKey"] != self._key
            or type(manifest["datasetDigest"]) is not str
            or manifest["datasetDigest"] != self._settings.digest
        ):
            raise ValueError("encoded Qwen-Image cache manifest identity does not match")
        digest = manifest["shardDigest"]
        if (
            type(digest) is not str
            or not digest.startswith("blake3:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
        ):
            raise ValueError("encoded Qwen-Image cache manifest has an invalid shard digest")
        size = manifest["shardSize"]
        if type(size) is not int or size < 1:
            raise ValueError("encoded Qwen-Image cache manifest has an invalid shard size")
        return self._shards / f"{digest[7:]}.safetensors", digest, size

    def _verified_shard(
        self, path: Path, expected_digest: str, expected_size: int
    ) -> SafetensorsSource:
        data = path.read_bytes()
        if len(data) != expected_size or path.stat().st_size != expected_size:
            raise ValueError("encoded Qwen-Image cache shard failed size verification")
        if blake3_digest(data) != expected_digest:
            raise ValueError("encoded Qwen-Image cache shard failed digest verification")
        source = load_safetensors_header(path)
        expected_metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "qwen-image",
            "format": _QWEN_IMAGE_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        expected_keys = {"latents", "context", "attention_mask"}
        if dict(source.metadata()) != expected_metadata or set(source.keys()) != expected_keys:
            raise ValueError("encoded Qwen-Image cache shard layout does not match")
        count = len(self._settings.inspection.items)
        for key, shape in zip(("latents", "context", "attention_mask"), self._shapes, strict=True):
            geometry = source.entry(key).geometry
            if geometry.dtype.name != "float32" or geometry.shape != (count, *shape[1:]):
                raise ValueError(f"encoded Qwen-Image {key} tensor has the wrong shape or dtype")
        return source

    def _verified_source(self) -> SafetensorsSource:
        path, digest, size = self._read_manifest()
        return self._verified_shard(path, digest, size)

    def _validate_values(self, tensors: EncodedQwenImageDatasetTensors) -> None:
        count = len(self._settings.inspection.items)
        values = (tensors.latents, tensors.context, tensors.attention_mask)
        for name, tensor, shape in zip(
            ("latents", "context", "attention_mask"), values, self._shapes, strict=True
        ):
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != (count, *shape[1:]):
                raise ValueError(f"encoded Qwen-Image {name} tensor has the wrong shape or dtype")
        if not bool(
            torch.all((tensors.attention_mask == 0.0) | (tensors.attention_mask == 1.0)).item()
        ):
            raise ValueError("encoded Qwen-Image attention mask must be binary")

    def is_valid(self) -> bool:
        try:
            source = self._verified_source()
            mask = load_tensors(source.path, ("attention_mask",))["attention_mask"]
            if not bool(torch.all((mask == 0.0) | (mask == 1.0)).item()):
                raise ValueError("encoded Qwen-Image attention mask must be binary")
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def load(self) -> EncodedQwenImageDatasetTensors | None:
        try:
            tensors = load_tensors(self._verified_source().path)
            result = EncodedQwenImageDatasetTensors(
                tensors["latents"], tensors["context"], tensors["attention_mask"]
            )
            self._validate_values(result)
            return result
        except (OSError, ValueError, KeyError, RuntimeError):
            return None

    def write(self, tensors: EncodedQwenImageDatasetTensors) -> str:
        self._validate_values(tensors)
        values = {
            "latents": tensors.latents,
            "context": tensors.context,
            "attention_mask": tensors.attention_mask,
        }
        metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "qwen-image",
            "format": _QWEN_IMAGE_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        shard = _safetensors_bytes(values, metadata)
        digest = blake3_digest(shard)
        size = len(shard)
        manifest = canonical_json(
            {
                "schemaVersion": 1,
                "cacheKey": self._key,
                "datasetDigest": self._settings.digest,
                "shardDigest": digest,
                "shardSize": size,
            }
        )
        durable_mkdir(self._shards)
        durable_mkdir(self._entries)
        path = self._shards / f"{digest[7:]}.safetensors"
        atomic_replace(path, shard)
        source = self._verified_shard(path, digest, size)
        round_tripped = load_tensors(source.path)
        if any(
            not torch.equal(round_tripped[name], value.detach().cpu())
            for name, value in values.items()
        ):
            raise ValueError("encoded Qwen-Image cache write failed exact readback")
        atomic_replace(self._manifest_path, manifest)
        return digest


class Ideogram4EncodedDatasetCache:
    """Verified role-bound Ideogram 4 latent and conditioning cache."""

    def __init__(
        self,
        settings: Ideogram4DatasetSettings,
        *,
        latent_shape: tuple[int, int, int, int],
        context_shape: tuple[int, int, int] | None,
        attention_mask_shape: tuple[int, int] | None,
        device: torch.device,
    ) -> None:
        if settings.encoded_cache_root is None:
            raise ValueError("encoded Ideogram 4 dataset cache root is not configured")
        if (context_shape is None) != (attention_mask_shape is None):
            raise ValueError("Ideogram 4 cache context and mask geometry must be supplied together")
        if (context_shape is not None) != (settings.role == "conditional"):
            raise ValueError("Ideogram 4 cache geometry does not match the model role")
        self._settings = settings
        self._latent_shape = latent_shape
        self._context_shape = context_shape
        self._attention_mask_shape = attention_mask_shape
        self._key = ideogram4_encoded_cache_key(
            settings,
            latent_shape=latent_shape,
            context_shape=context_shape,
            attention_mask_shape=attention_mask_shape,
            device=device,
        )
        root = Path(settings.encoded_cache_root) / "ideogram4-v1"
        self._entries = root / "entries"
        self._shards = root / "shards"
        self._manifest_path = self._entries / f"{self._key[7:]}.json"
        self._lock_path = root / "locks" / f"{self._key[7:]}.lock"

    def build_lock(self) -> AbstractContextManager[None]:
        return advisory_file_lock(self._lock_path)

    def _read_manifest(self) -> tuple[Path, str, int]:
        value = json.loads(self._manifest_path.read_bytes().decode("ascii"))
        if not isinstance(value, dict):
            raise ValueError("encoded Ideogram 4 cache manifest must be an object")
        manifest = cast("dict[str, object]", value)
        if set(manifest) != {
            "schemaVersion",
            "cacheKey",
            "datasetDigest",
            "role",
            "shardDigest",
            "shardSize",
        }:
            raise ValueError("encoded Ideogram 4 cache manifest has an unknown layout")
        if (
            type(manifest["schemaVersion"]) is not int
            or manifest["schemaVersion"] != 1
            or manifest["cacheKey"] != self._key
            or manifest["datasetDigest"] != self._settings.digest
            or manifest["role"] != self._settings.role
        ):
            raise ValueError("encoded Ideogram 4 cache manifest identity does not match")
        digest = manifest["shardDigest"]
        size = manifest["shardSize"]
        if (
            type(digest) is not str
            or not digest.startswith("blake3:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
            or type(size) is not int
            or size < 1
        ):
            raise ValueError("encoded Ideogram 4 cache manifest has an invalid shard")
        return self._shards / f"{digest[7:]}.safetensors", digest, size

    def _verified_source(self) -> SafetensorsSource:
        path, expected_digest, expected_size = self._read_manifest()
        data = path.read_bytes()
        if len(data) != expected_size or path.stat().st_size != expected_size:
            raise ValueError("encoded Ideogram 4 cache shard failed size verification")
        if blake3_digest(data) != expected_digest:
            raise ValueError("encoded Ideogram 4 cache shard failed digest verification")
        source = load_safetensors_header(path)
        expected_metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "ideogram4",
            "format": _IDEOGRAM4_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
            "role": self._settings.role,
        }
        keys = {"latents"}
        if self._settings.role == "conditional":
            keys.update(("context", "attention_mask"))
        if dict(source.metadata()) != expected_metadata or set(source.keys()) != keys:
            raise ValueError("encoded Ideogram 4 cache shard layout does not match")
        count = len(self._settings.inspection.items)
        geometries: dict[str, tuple[int, ...]] = {"latents": self._latent_shape}
        if self._context_shape is not None and self._attention_mask_shape is not None:
            geometries["context"] = self._context_shape
            geometries["attention_mask"] = self._attention_mask_shape
        for key, shape in geometries.items():
            geometry = source.entry(key).geometry
            if geometry.dtype.name != "float32" or geometry.shape != (count, *shape[1:]):
                raise ValueError(f"encoded Ideogram 4 {key} tensor has the wrong shape or dtype")
        return source

    def _validate_values(self, tensors: EncodedIdeogram4DatasetTensors) -> None:
        count = len(self._settings.inspection.items)
        if tensors.latents.dtype != torch.float32 or tuple(tensors.latents.shape) != (
            count,
            *self._latent_shape[1:],
        ):
            raise ValueError("encoded Ideogram 4 latents have the wrong shape or dtype")
        if self._settings.role == "unconditional":
            if tensors.context is not None or tensors.attention_mask is not None:
                raise ValueError("unconditional Ideogram 4 cache cannot carry text conditioning")
            return
        assert self._context_shape is not None and self._attention_mask_shape is not None
        context = tensors.context
        mask = tensors.attention_mask
        if (
            context is None
            or context.dtype != torch.float32
            or tuple(context.shape) != (count, *self._context_shape[1:])
            or mask is None
            or mask.dtype != torch.float32
            or tuple(mask.shape) != (count, *self._attention_mask_shape[1:])
        ):
            raise ValueError("conditional Ideogram 4 cache has wrong conditioning geometry")
        if not bool(torch.all((mask == 0.0) | (mask == 1.0)).item()):
            raise ValueError("encoded Ideogram 4 attention mask must be binary")

    def is_valid(self) -> bool:
        try:
            source = self._verified_source()
            if self._settings.role == "conditional":
                mask = load_tensors(source.path, ("attention_mask",))["attention_mask"]
                if not bool(torch.all((mask == 0.0) | (mask == 1.0)).item()):
                    raise ValueError("encoded Ideogram 4 attention mask must be binary")
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def load(self) -> EncodedIdeogram4DatasetTensors | None:
        try:
            values = load_tensors(self._verified_source().path)
            result = EncodedIdeogram4DatasetTensors(
                values["latents"], values.get("context"), values.get("attention_mask")
            )
            self._validate_values(result)
            return result
        except (OSError, ValueError, KeyError, RuntimeError):
            return None

    def write(self, tensors: EncodedIdeogram4DatasetTensors) -> str:
        self._validate_values(tensors)
        values = {"latents": tensors.latents}
        if tensors.context is not None and tensors.attention_mask is not None:
            values["context"] = tensors.context
            values["attention_mask"] = tensors.attention_mask
        metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "ideogram4",
            "format": _IDEOGRAM4_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
            "role": self._settings.role,
        }
        shard = _safetensors_bytes(values, metadata)
        digest = blake3_digest(shard)
        size = len(shard)
        manifest = canonical_json(
            {
                "schemaVersion": 1,
                "cacheKey": self._key,
                "datasetDigest": self._settings.digest,
                "role": self._settings.role,
                "shardDigest": digest,
                "shardSize": size,
            }
        )
        durable_mkdir(self._shards)
        durable_mkdir(self._entries)
        path = self._shards / f"{digest[7:]}.safetensors"
        atomic_replace(path, shard)
        round_trip = load_tensors(path)
        if any(
            not torch.equal(round_trip[name], value.detach().cpu())
            for name, value in values.items()
        ):
            raise ValueError("encoded Ideogram 4 cache write failed exact readback")
        atomic_replace(self._manifest_path, manifest)
        self._verified_source()
        return digest


def encoded_cache_state(
    settings: ImageCaptionDatasetSettings,
    *,
    batch_size: int,
    device: torch.device,
) -> str:
    if settings.encoded_cache_root is None:
        return "disabled"
    cache = EncodedDatasetCache(settings, batch_size=batch_size, device=device)
    return "hit" if cache.is_valid() else "miss"


def flux_encoded_cache_state(
    settings: FluxDatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int],
    context_shape: tuple[int, int, int],
    pooled_shape: tuple[int, int],
    device: torch.device,
) -> str:
    if settings.encoded_cache_root is None:
        return "disabled"
    cache = FluxEncodedDatasetCache(
        settings,
        latent_shape=latent_shape,
        context_shape=context_shape,
        pooled_shape=pooled_shape,
        device=device,
    )
    return "hit" if cache.is_valid() else "miss"


def flux2_encoded_cache_state(
    settings: Flux2DatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int],
    context_shape: tuple[int, int, int],
    device: torch.device,
) -> str:
    if settings.encoded_cache_root is None:
        return "disabled"
    cache = Flux2EncodedDatasetCache(
        settings,
        latent_shape=latent_shape,
        context_shape=context_shape,
        device=device,
    )
    return "hit" if cache.is_valid() else "miss"


def qwen_image_encoded_cache_state(
    settings: QwenImageDatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int, int],
    context_shape: tuple[int, int, int],
    attention_mask_shape: tuple[int, int],
    device: torch.device,
) -> str:
    if settings.encoded_cache_root is None:
        return "disabled"
    cache = QwenImageEncodedDatasetCache(
        settings,
        latent_shape=latent_shape,
        context_shape=context_shape,
        attention_mask_shape=attention_mask_shape,
        device=device,
    )
    return "hit" if cache.is_valid() else "miss"


def ideogram4_encoded_cache_state(
    settings: Ideogram4DatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int],
    context_shape: tuple[int, int, int] | None,
    attention_mask_shape: tuple[int, int] | None,
    device: torch.device,
) -> str:
    if settings.encoded_cache_root is None:
        return "disabled"
    cache = Ideogram4EncodedDatasetCache(
        settings,
        latent_shape=latent_shape,
        context_shape=context_shape,
        attention_mask_shape=attention_mask_shape,
        device=device,
    )
    return "hit" if cache.is_valid() else "miss"


class MiniMaxH3EncodedDatasetCache:
    """Verified H3 video, audio, conditioner, and carrier tensor cache."""

    def __init__(self, settings: MiniMaxH3DatasetSettings, *, device: torch.device) -> None:
        if settings.encoded_cache_root is None:
            raise ValueError("encoded H3 dataset cache root is not configured")
        self._settings = settings
        self._key = minimax_h3_encoded_cache_key(settings, device=device)
        root = Path(settings.encoded_cache_root) / "h3-v1"
        self._entries = root / "entries"
        self._shards = root / "shards"
        self._manifest_path = self._entries / f"{self._key[7:]}.json"
        self._lock_path = root / "locks" / f"{self._key[7:]}.lock"

    def build_lock(self) -> AbstractContextManager[None]:
        return advisory_file_lock(self._lock_path)

    def _read_manifest(self) -> tuple[Path, str]:
        value = json.loads(self._manifest_path.read_bytes().decode("ascii"))
        if not isinstance(value, dict):
            raise ValueError("encoded H3 cache manifest must be an object")
        manifest = cast("dict[str, object]", value)
        if set(manifest) != {
            "schemaVersion",
            "cacheKey",
            "datasetDigest",
            "shardDigest",
        }:
            raise ValueError("encoded H3 cache manifest has an unknown layout")
        if (
            manifest["schemaVersion"] != 1
            or manifest["cacheKey"] != self._key
            or manifest["datasetDigest"] != self._settings.digest
        ):
            raise ValueError("encoded H3 cache manifest identity does not match the dataset")
        shard_digest = manifest["shardDigest"]
        if (
            not isinstance(shard_digest, str)
            or not shard_digest.startswith("blake3:")
            or len(shard_digest) != 71
            or any(character not in "0123456789abcdef" for character in shard_digest[7:])
        ):
            raise ValueError("encoded H3 cache manifest has an invalid shard digest")
        return self._shards / f"{shard_digest[7:]}.safetensors", shard_digest

    def _verified_source(self) -> SafetensorsSource:
        shard_path, expected_digest = self._read_manifest()
        data = shard_path.read_bytes()
        if blake3_digest(data) != expected_digest:
            raise ValueError("encoded H3 cache shard failed digest verification")
        source = load_safetensors_header(shard_path)
        expected_metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "minimax-h3",
            "format": _H3_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        if dict(source.metadata()) != expected_metadata:
            raise ValueError("encoded H3 cache shard metadata does not match the dataset")
        expected_keys = {
            "video_latents",
            "audio_latents",
            "conditioner_embeddings",
            "conditioning_text_token_tags",
        }
        if set(source.keys()) != expected_keys:
            raise ValueError("encoded H3 cache shard has an unknown tensor layout")
        item_count = len(self._settings.inspection.items)
        video = source.entry("video_latents").geometry
        audio = source.entry("audio_latents").geometry
        context = source.entry("conditioner_embeddings").geometry
        tags = source.entry("conditioning_text_token_tags").geometry
        if video.dtype.name != "float32" or video.shape != (
            item_count,
            *minimax_h3_video_latent_shape(self._settings.frame_count, self._settings.resolution)[
                1:
            ],
        ):
            raise ValueError("encoded H3 video tensor has the wrong shape or dtype")
        if audio.dtype.name != "float32" or audio.shape != (
            item_count,
            *minimax_h3_audio_latent_shape(self._settings.frame_count)[1:],
        ):
            raise ValueError("encoded H3 audio tensor has the wrong shape or dtype")
        if (
            context.dtype.name != "float32"
            or len(context.shape) != 3
            or context.shape[0] != item_count
            or context.shape[1] < 1
            or context.shape[2] != 5120
        ):
            raise ValueError("encoded H3 conditioner tensor has the wrong shape or dtype")
        if tags.dtype.name != "int64" or tags.shape != context.shape[:2]:
            raise ValueError("encoded H3 conditioning tags have the wrong shape or dtype")
        return source

    def is_valid(self) -> bool:
        try:
            self._verified_source()
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def load(self) -> EncodedMiniMaxH3DatasetTensors | None:
        try:
            source = self._verified_source()
            tensors = load_tensors(source.path)
            return EncodedMiniMaxH3DatasetTensors(
                tensors["video_latents"],
                tensors["audio_latents"],
                tensors["conditioner_embeddings"],
                tensors["conditioning_text_token_tags"],
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            return None

    def write(self, tensors: EncodedMiniMaxH3DatasetTensors) -> None:
        values = {
            "video_latents": tensors.video_latents,
            "audio_latents": tensors.audio_latents,
            "conditioner_embeddings": tensors.conditioner_embeddings,
            "conditioning_text_token_tags": tensors.conditioning_text_token_tags,
        }
        metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "minimax-h3",
            "format": _H3_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        shard = _safetensors_bytes(values, metadata)
        shard_digest = blake3_digest(shard)
        manifest = canonical_json(
            {
                "schemaVersion": 1,
                "cacheKey": self._key,
                "datasetDigest": self._settings.digest,
                "shardDigest": shard_digest,
            }
        )
        durable_mkdir(self._shards)
        durable_mkdir(self._entries)
        atomic_replace(self._shards / f"{shard_digest[7:]}.safetensors", shard)
        atomic_replace(self._manifest_path, manifest)


def minimax_h3_encoded_cache_state(
    settings: MiniMaxH3DatasetSettings,
    *,
    device: torch.device,
) -> str:
    if settings.encoded_cache_root is None:
        return "disabled"
    cache = MiniMaxH3EncodedDatasetCache(settings, device=device)
    return "hit" if cache.is_valid() else "miss"


def wan_encoded_cache_state(
    settings: WanDatasetSettings,
    *,
    latent_shape: tuple[int, int, int, int, int],
    context_shape: tuple[int, int, int],
    device: torch.device,
) -> str:
    if settings.encoded_cache_root is None:
        return "disabled"
    cache = WanEncodedDatasetCache(
        settings,
        latent_shape=latent_shape,
        context_shape=context_shape,
        device=device,
    )
    return "hit" if cache.is_valid() else "miss"


class WanEncodedDatasetCache:
    """Verified Wan video latent and UMT5 context cache."""

    def __init__(
        self,
        settings: WanDatasetSettings,
        *,
        latent_shape: tuple[int, int, int, int, int],
        context_shape: tuple[int, int, int],
        device: torch.device,
    ) -> None:
        if settings.encoded_cache_root is None:
            raise ValueError("encoded Wan dataset cache root is not configured")
        self._settings = settings
        self._latent_shape = latent_shape
        self._context_shape = context_shape
        self._key = wan_encoded_cache_key(
            settings,
            latent_shape=latent_shape,
            context_shape=context_shape,
            device=device,
        )
        _cache_format, _family, cache_family = _wan_cache_contract(settings)
        root = Path(settings.encoded_cache_root) / cache_family
        self._entries = root / "entries"
        self._shards = root / "shards"
        self._manifest_path = self._entries / f"{self._key[7:]}.json"
        self._lock_path = root / "locks" / f"{self._key[7:]}.lock"

    def build_lock(self) -> AbstractContextManager[None]:
        return advisory_file_lock(self._lock_path)

    def _read_manifest(self) -> tuple[Path, str]:
        value = json.loads(self._manifest_path.read_bytes().decode("ascii"))
        if not isinstance(value, dict):
            raise ValueError("encoded Wan cache manifest must be an object")
        manifest = cast("dict[str, object]", value)
        if set(manifest) != {
            "schemaVersion",
            "cacheKey",
            "datasetDigest",
            "shardDigest",
        }:
            raise ValueError("encoded Wan cache manifest has an unknown layout")
        if (
            manifest["schemaVersion"] != 1
            or manifest["cacheKey"] != self._key
            or manifest["datasetDigest"] != self._settings.digest
        ):
            raise ValueError("encoded Wan cache manifest identity does not match the dataset")
        shard_digest = manifest["shardDigest"]
        if (
            not isinstance(shard_digest, str)
            or not shard_digest.startswith("blake3:")
            or len(shard_digest) != 71
            or any(character not in "0123456789abcdef" for character in shard_digest[7:])
        ):
            raise ValueError("encoded Wan cache manifest has an invalid shard digest")
        return self._shards / f"{shard_digest[7:]}.safetensors", shard_digest

    def _verified_source(self) -> SafetensorsSource:
        shard_path, expected_digest = self._read_manifest()
        if blake3_digest(shard_path.read_bytes()) != expected_digest:
            raise ValueError("encoded Wan cache shard failed digest verification")
        source = load_safetensors_header(shard_path)
        cache_format, family, _root = _wan_cache_contract(self._settings)
        expected_metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": family,
            "format": cache_format,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        if dict(source.metadata()) != expected_metadata:
            raise ValueError("encoded Wan cache shard metadata does not match the dataset")
        expected_keys = {"latents", "context"}
        if self._settings.conditioning_contract == "first-frame-i2v":
            expected_keys.add("i2v_conditioning")
        if set(source.keys()) != expected_keys:
            raise ValueError("encoded Wan cache shard has an unknown tensor layout")
        item_count = len(self._settings.inspection.items)
        latent = source.entry("latents").geometry
        context = source.entry("context").geometry
        if latent.dtype.name != "float32" or latent.shape != (
            item_count,
            *self._latent_shape[1:],
        ):
            raise ValueError("encoded Wan latent tensor has the wrong shape or dtype")
        if context.dtype.name != "float32" or context.shape != (
            item_count,
            *self._context_shape[1:],
        ):
            raise ValueError("encoded Wan context tensor has the wrong shape or dtype")
        if "i2v_conditioning" in expected_keys:
            conditioning = source.entry("i2v_conditioning").geometry
            if conditioning.dtype.name != "float32" or conditioning.shape != (
                item_count,
                20,
                *self._latent_shape[2:],
            ):
                raise ValueError("encoded Wan I2V conditioning has the wrong shape or dtype")
        return source

    def is_valid(self) -> bool:
        try:
            self._verified_source()
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def load(self) -> EncodedWanDatasetTensors | None:
        try:
            tensors = load_tensors(self._verified_source().path)
            return EncodedWanDatasetTensors(
                tensors["latents"], tensors["context"], tensors.get("i2v_conditioning")
            )
        except (OSError, ValueError, KeyError, RuntimeError):
            return None

    def write(self, tensors: EncodedWanDatasetTensors) -> str:
        expects_i2v = self._settings.conditioning_contract == "first-frame-i2v"
        if (tensors.i2v_conditioning is not None) != expects_i2v:
            raise ValueError("encoded Wan I2V conditioning does not match the cache contract")
        item_count = len(self._settings.inspection.items)
        if tensors.latents.dtype != torch.float32 or tuple(tensors.latents.shape) != (
            item_count,
            *self._latent_shape[1:],
        ):
            raise ValueError("encoded Wan latent tensor has the wrong shape or dtype")
        if tensors.context.dtype != torch.float32 or tuple(tensors.context.shape) != (
            item_count,
            *self._context_shape[1:],
        ):
            raise ValueError("encoded Wan context tensor has the wrong shape or dtype")
        if tensors.i2v_conditioning is not None and (
            tensors.i2v_conditioning.dtype != torch.float32
            or tuple(tensors.i2v_conditioning.shape) != (item_count, 20, *self._latent_shape[2:])
        ):
            raise ValueError("encoded Wan I2V conditioning has the wrong shape or dtype")
        cache_format, family, _root = _wan_cache_contract(self._settings)
        metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": family,
            "format": cache_format,
            "itemCount": str(item_count),
        }
        values = {"latents": tensors.latents, "context": tensors.context}
        if tensors.i2v_conditioning is not None:
            values["i2v_conditioning"] = tensors.i2v_conditioning
        shard = _safetensors_bytes(values, metadata)
        shard_digest = blake3_digest(shard)
        manifest = canonical_json(
            {
                "schemaVersion": 1,
                "cacheKey": self._key,
                "datasetDigest": self._settings.digest,
                "shardDigest": shard_digest,
            }
        )
        durable_mkdir(self._shards)
        durable_mkdir(self._entries)
        atomic_replace(self._shards / f"{shard_digest[7:]}.safetensors", shard)
        atomic_replace(self._manifest_path, manifest)
        return shard_digest
