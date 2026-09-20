"""Content-addressed resumable training checkpoints."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from dinkster_api.v1 import digest_bytes

from .durability import atomic_replace, durable_mkdir


class CheckpointError(ValueError):
    """A checkpoint is missing, corrupt, or inconsistent."""


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def blake3_digest(data: bytes) -> str:
    return digest_bytes(data)


def _cpu_value(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {key: _cpu_value(item) for key, item in cast("dict[object, object]", value).items()}
    if isinstance(value, list):
        return [_cpu_value(item) for item in cast("list[object]", value)]
    if isinstance(value, tuple):
        return tuple(_cpu_value(item) for item in cast("tuple[object, ...]", value))
    return value


@dataclass(frozen=True)
class CheckpointState:
    manifest_digest: str
    session_id: str
    config_digest: str
    extension_snapshot_digest: str
    parent_manifest_digest: str
    step_cursor: int
    config: dict[str, object]
    adapter: dict[str, torch.Tensor]
    optimizer: dict[str, object]
    rng: dict[str, torch.Tensor]
    data_cursor: int
    loss: float | None


class ContentAddressedCheckpointStore:
    """Immutable shards plus a digest-addressed JSON manifest."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._shards = root / "shards"
        self._manifests = root / "manifests"
        durable_mkdir(self._shards)
        durable_mkdir(self._manifests)

    @staticmethod
    def _hex(digest: str) -> str:
        if not digest.startswith("blake3:") or len(digest) != 71:
            raise CheckpointError(f"invalid blake3 digest {digest!r}")
        value = digest[7:]
        if any(ch not in "0123456789abcdef" for ch in value):
            raise CheckpointError(f"invalid blake3 digest {digest!r}")
        return value

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        if path.exists():
            if path.read_bytes() != data:
                raise CheckpointError(f"content-addressed path collision at {path}")
            return
        atomic_replace(path, data)

    def _write_shard(self, data: bytes, suffix: str) -> str:
        digest = blake3_digest(data)
        self._atomic_write(self._shards / f"{self._hex(digest)}.{suffix}", data)
        return digest

    def _write_torch(self, value: object) -> str:
        buffer = io.BytesIO()
        torch.save(_cpu_value(value), buffer)
        return self._write_shard(buffer.getvalue(), "pt")

    def _write_json(self, value: object) -> str:
        return self._write_shard(canonical_json(value), "json")

    def _read_shard(self, digest: str, suffix: str) -> bytes:
        path = self._shards / f"{self._hex(digest)}.{suffix}"
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise CheckpointError(f"checkpoint shard {digest} is missing") from exc
        if blake3_digest(data) != digest:
            raise CheckpointError(f"checkpoint shard {digest} failed digest verification")
        return data

    def _read_torch(self, digest: str) -> object:
        return torch.load(
            io.BytesIO(self._read_shard(digest, "pt")),
            map_location="cpu",
            weights_only=True,
        )

    def _read_json(self, digest: str) -> object:
        try:
            return json.loads(self._read_shard(digest, "json").decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"JSON checkpoint shard {digest} is malformed") from exc

    def write(
        self,
        *,
        session_id: str,
        config_digest: str,
        extension_snapshot_digest: str,
        parent_manifest_digest: str,
        step_cursor: int,
        config: dict[str, object],
        adapter: dict[str, torch.Tensor],
        optimizer: dict[str, object],
        rng: dict[str, torch.Tensor],
        data_cursor: int,
        loss: float | None,
    ) -> str:
        if (step_cursor == 0) != (loss is None):
            raise CheckpointError("checkpoint loss must be absent exactly at optimizer step zero")
        if loss is not None and not math.isfinite(loss):
            raise CheckpointError("checkpoint loss must be finite")
        trainer_state = {"dataCursor": data_cursor, "loss": loss}
        manifest = {
            "schemaVersion": 1,
            "sessionId": session_id,
            "configDigest": config_digest,
            "extensionSnapshotDigest": extension_snapshot_digest,
            "parentManifestDigest": parent_manifest_digest,
            "stepCursor": step_cursor,
            "shards": {
                "config": self._write_json(config),
                "adapter": self._write_torch(adapter),
                "optimizer": self._write_torch(optimizer),
                "rng": self._write_torch(rng),
                "trainerState": self._write_json(trainer_state),
            },
        }
        data = canonical_json(manifest)
        digest = blake3_digest(data)
        self._atomic_write(self._manifests / f"{self._hex(digest)}.json", data)
        return digest

    def load(self, digest: str) -> CheckpointState:
        path = self._manifests / f"{self._hex(digest)}.json"
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise CheckpointError(f"checkpoint manifest {digest} is missing") from exc
        if blake3_digest(data) != digest:
            raise CheckpointError(f"checkpoint manifest {digest} failed digest verification")
        try:
            value = json.loads(data.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"checkpoint manifest {digest} is malformed") from exc
        if not isinstance(value, dict):
            raise CheckpointError(f"checkpoint manifest {digest} must be an object")
        manifest = cast("dict[str, object]", value)
        if manifest.get("schemaVersion") != 1:
            raise CheckpointError(f"checkpoint manifest {digest} has an unknown schema")
        shards_value = manifest.get("shards")
        if not isinstance(shards_value, dict):
            raise CheckpointError(f"checkpoint manifest {digest} has no shard mapping")
        shards = cast("dict[str, object]", shards_value)
        required = {"config", "adapter", "optimizer", "rng", "trainerState"}
        if set(shards) != required or not all(isinstance(shards[name], str) for name in required):
            raise CheckpointError(f"checkpoint manifest {digest} has invalid shard references")

        config_value = self._read_json(cast("str", shards["config"]))
        adapter_value = self._read_torch(cast("str", shards["adapter"]))
        optimizer_value = self._read_torch(cast("str", shards["optimizer"]))
        rng_value = self._read_torch(cast("str", shards["rng"]))
        trainer_value = self._read_json(cast("str", shards["trainerState"]))
        if not isinstance(config_value, dict) or not isinstance(adapter_value, dict):
            raise CheckpointError(f"checkpoint manifest {digest} has invalid config/adapter state")
        if not isinstance(optimizer_value, dict) or not isinstance(rng_value, dict):
            raise CheckpointError(f"checkpoint manifest {digest} has invalid optimizer/RNG state")
        if not isinstance(trainer_value, dict):
            raise CheckpointError(f"checkpoint manifest {digest} has invalid trainer state")
        trainer = cast("dict[str, object]", trainer_value)
        cursor = trainer.get("dataCursor")
        loss_value = trainer.get("loss")
        if type(cursor) is not int or cursor < 0:
            raise CheckpointError(f"checkpoint manifest {digest} has an invalid data cursor")
        if loss_value is not None and type(loss_value) not in (int, float):
            raise CheckpointError(f"checkpoint manifest {digest} has an invalid loss")
        loss = None if loss_value is None else float(cast("int | float", loss_value))
        if loss is not None and not math.isfinite(loss):
            raise CheckpointError(f"checkpoint manifest {digest} has an invalid loss")
        step = manifest.get("stepCursor")
        if type(step) is not int or step < 0:
            raise CheckpointError(f"checkpoint manifest {digest} has an invalid step cursor")
        if (step == 0) != (loss_value is None):
            raise CheckpointError(
                f"checkpoint manifest {digest} has a loss inconsistent with its step cursor"
            )

        adapter_raw = cast("dict[object, object]", adapter_value)
        rng_raw = cast("dict[object, object]", rng_value)
        if not all(
            isinstance(key, str) and isinstance(item, torch.Tensor)
            for key, item in adapter_raw.items()
        ):
            raise CheckpointError(f"checkpoint manifest {digest} has invalid adapter tensors")
        if not all(
            isinstance(key, str) and isinstance(item, torch.Tensor) for key, item in rng_raw.items()
        ):
            raise CheckpointError(f"checkpoint manifest {digest} has invalid RNG tensors")
        string_fields = (
            "sessionId",
            "configDigest",
            "extensionSnapshotDigest",
            "parentManifestDigest",
        )
        if not all(isinstance(manifest.get(name), str) for name in string_fields):
            raise CheckpointError(f"checkpoint manifest {digest} has invalid identity fields")
        return CheckpointState(
            manifest_digest=digest,
            session_id=cast("str", manifest["sessionId"]),
            config_digest=cast("str", manifest["configDigest"]),
            extension_snapshot_digest=cast("str", manifest["extensionSnapshotDigest"]),
            parent_manifest_digest=cast("str", manifest["parentManifestDigest"]),
            step_cursor=step,
            config=cast("dict[str, object]", config_value),
            adapter=cast("dict[str, torch.Tensor]", adapter_value),
            optimizer=cast("dict[str, object]", optimizer_value),
            rng=cast("dict[str, torch.Tensor]", rng_value),
            data_cursor=cursor,
            loss=loss,
        )
