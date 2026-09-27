"""Validated SD1.5 training configuration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinkster_api.v1 import digest_bytes


class TrainingConfigError(ValueError):
    """The serialized training configuration is invalid."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise TrainingConfigError(f"{name} must be a positive integer")
    return value


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise TrainingConfigError(f"{name} must be a number")
    result = float(cast("int | float", value))
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise TrainingConfigError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    resolution: tuple[int, int]
    encoded_cache_root: Path
    caption_extension: str = ".txt"

    @classmethod
    def from_mapping(cls, value: object) -> DatasetConfig:
        if not isinstance(value, dict):
            raise TrainingConfigError("dataset must be an object")
        mapping = cast("dict[str, object]", value)
        unknown = sorted(
            set(mapping) - {"root", "resolution", "encodedCacheRoot", "captionExtension"}
        )
        if unknown:
            raise TrainingConfigError(f"dataset has unknown fields: {', '.join(unknown)}")
        root = mapping.get("root")
        cache = mapping.get("encodedCacheRoot")
        resolution = mapping.get("resolution")
        caption_extension = mapping.get("captionExtension", ".txt")
        if not isinstance(root, str) or not root:
            raise TrainingConfigError("dataset.root must be a non-empty path")
        if not isinstance(cache, str) or not cache:
            raise TrainingConfigError("dataset.encodedCacheRoot must be a non-empty path")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(type(item) is not int or item < 8 or item % 8 for item in resolution)
        ):
            raise TrainingConfigError("dataset.resolution must contain two positive multiples of 8")
        if not isinstance(caption_extension, str) or not caption_extension.startswith("."):
            raise TrainingConfigError("dataset.captionExtension must begin with '.'")
        return cls(
            root=Path(root).expanduser().resolve(),
            resolution=(resolution[0], resolution[1]),
            encoded_cache_root=Path(cache).expanduser().resolve(),
            caption_extension=caption_extension,
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "resolution": list(self.resolution),
            "encodedCacheRoot": str(self.encoded_cache_root),
            "captionExtension": self.caption_extension,
        }


@dataclass(frozen=True)
class TrainingConfig:
    checkpoint_path: Path
    checkpoint_digest: str
    dataset: DatasetConfig
    device: str
    rank: int
    alpha: float
    learning_rate: float
    weight_decay: float
    seed: int
    batch_size: int
    gradient_accumulation_steps: int
    checkpoint_interval: int
    target_patterns: tuple[str, ...]

    @classmethod
    def parse(cls, serialized: str) -> TrainingConfig:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise TrainingConfigError("training config must be valid JSON") from exc
        if not isinstance(value, dict):
            raise TrainingConfigError("training config must be an object")
        return cls.from_mapping(cast("dict[str, object]", value))

    @classmethod
    def from_mapping(cls, mapping: dict[str, object]) -> TrainingConfig:
        allowed = {
            "schemaVersion",
            "family",
            "checkpointPath",
            "checkpointDigest",
            "dataset",
            "device",
            "rank",
            "alpha",
            "learningRate",
            "weightDecay",
            "seed",
            "batchSize",
            "gradientAccumulationSteps",
            "checkpointInterval",
            "targetPatterns",
        }
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise TrainingConfigError(f"training config has unknown fields: {', '.join(unknown)}")
        if mapping.get("schemaVersion") != 1:
            raise TrainingConfigError("schemaVersion must be 1")
        if mapping.get("family") != "sd15":
            raise TrainingConfigError("family must be 'sd15'")
        checkpoint_path = mapping.get("checkpointPath")
        checkpoint_digest = mapping.get("checkpointDigest")
        device = mapping.get("device", "cuda")
        seed = mapping.get("seed", 0)
        target_patterns = mapping.get(
            "targetPatterns",
            ["attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"],
        )
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise TrainingConfigError("checkpointPath must be a non-empty path")
        if (
            not isinstance(checkpoint_digest, str)
            or not checkpoint_digest.startswith("sha256:")
            or len(checkpoint_digest) != 71
            or any(character not in "0123456789abcdef" for character in checkpoint_digest[7:])
        ):
            raise TrainingConfigError("checkpointDigest must be a sha256 digest")
        if not isinstance(device, str) or not (device == "cpu" or device.startswith("cuda")):
            raise TrainingConfigError("device must be 'cpu' or a CUDA device")
        if type(seed) is not int or seed < 0:
            raise TrainingConfigError("seed must be a non-negative integer")
        if (
            not isinstance(target_patterns, list)
            or not target_patterns
            or any(not isinstance(pattern, str) or not pattern for pattern in target_patterns)
        ):
            raise TrainingConfigError("targetPatterns must contain non-empty strings")
        checkpoint_interval = mapping.get("checkpointInterval", 1)
        if type(checkpoint_interval) is not int or checkpoint_interval < 1:
            raise TrainingConfigError("checkpointInterval must be a positive integer")
        weight_decay = _finite_float(mapping.get("weightDecay", 0.0), "weightDecay")
        if weight_decay < 0:
            raise TrainingConfigError("weightDecay must be non-negative")
        return cls(
            checkpoint_path=Path(checkpoint_path).expanduser().resolve(),
            checkpoint_digest=checkpoint_digest,
            dataset=DatasetConfig.from_mapping(mapping.get("dataset")),
            device=device,
            rank=_positive_int(mapping.get("rank", 4), "rank"),
            alpha=_finite_float(mapping.get("alpha", 4.0), "alpha", positive=True),
            learning_rate=_finite_float(
                mapping.get("learningRate", 1e-4), "learningRate", positive=True
            ),
            weight_decay=weight_decay,
            seed=seed,
            batch_size=_positive_int(mapping.get("batchSize", 1), "batchSize"),
            gradient_accumulation_steps=_positive_int(
                mapping.get("gradientAccumulationSteps", 1),
                "gradientAccumulationSteps",
            ),
            checkpoint_interval=checkpoint_interval,
            target_patterns=tuple(cast("list[str]", target_patterns)),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "family": "sd15",
            "checkpointPath": str(self.checkpoint_path),
            "checkpointDigest": self.checkpoint_digest,
            "dataset": self.dataset.to_mapping(),
            "device": self.device,
            "rank": self.rank,
            "alpha": self.alpha,
            "learningRate": self.learning_rate,
            "weightDecay": self.weight_decay,
            "seed": self.seed,
            "batchSize": self.batch_size,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "checkpointInterval": self.checkpoint_interval,
            "targetPatterns": list(self.target_patterns),
        }

    @property
    def digest(self) -> str:
        return digest_bytes(canonical_json(self.to_mapping()))
