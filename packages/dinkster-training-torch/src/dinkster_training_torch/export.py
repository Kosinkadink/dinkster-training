"""Deterministic LoRA export from fork-mapped checkpoint tensors."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, cast

import torch

from .checkpoint import CheckpointState
from .durability import atomic_replace, durable_mkdir

ExportDtype = Literal["fp16", "bf16", "fp32"]
_DTYPES: dict[ExportDtype, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
_DTYPE_CODES: dict[ExportDtype, str] = {"fp16": "F16", "bf16": "BF16", "fp32": "F32"}


@dataclass(frozen=True)
class LoraExportSettings:
    path: Path
    dtype: ExportDtype

    @classmethod
    def parse(cls, serialized: str, *, export_root: Path) -> LoraExportSettings:
        try:
            value = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise ValueError("export settings must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("export settings must be an object")
        mapping = cast("dict[str, object]", value)
        if set(mapping) - {"path", "dtype"}:
            raise ValueError("export settings contain unknown fields")
        raw_path = mapping.get("path")
        raw_dtype = mapping.get("dtype", "fp16")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError("export settings path must be a non-empty string")
        if PurePosixPath(raw_path).is_absolute() or PureWindowsPath(raw_path).is_absolute():
            raise ValueError("export settings path must be relative")
        root = export_root.resolve()
        path = (root / raw_path).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".safetensors":
            raise ValueError("export path must stay under the export root and end in .safetensors")
        if raw_dtype not in _DTYPES:
            raise ValueError("export dtype must be 'fp16', 'bf16', or 'fp32'")
        return cls(path, raw_dtype)


def export_lora(
    checkpoint: CheckpointState,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    dtype = _DTYPES[settings.dtype]
    tensors = {
        key: value.detach().to(device="cpu", dtype=dtype).contiguous()
        for key, value in sorted(checkpoint.adapter.items())
    }
    if not tensors or not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("LoRA checkpoint contains no finite adapter tensors")
    metadata = {
        "dinkster.config_digest": checkpoint.config_digest,
        "dinkster.runtime": runtime_identity,
        "dinkster.step_cursor": str(checkpoint.step_cursor),
    }
    if sys.byteorder != "little":
        raise RuntimeError("safetensors export requires a little-endian host")
    header: dict[str, object] = {"__metadata__": {key: metadata[key] for key in sorted(metadata)}}
    payload = bytearray()
    for key, tensor in tensors.items():
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        begin = len(payload)
        payload.extend(raw)
        header[key] = {
            "dtype": _DTYPE_CODES[settings.dtype],
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
    data = struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload)
    durable_mkdir(settings.path.parent)
    atomic_replace(settings.path, data)
    return settings.path, f"sha256:{hashlib.sha256(data).hexdigest()}"
