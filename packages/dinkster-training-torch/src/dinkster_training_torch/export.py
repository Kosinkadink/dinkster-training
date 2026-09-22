"""Deterministic LoRA export from committed checkpoints."""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, cast

import torch
from dinkster_api.v1 import digest_bytes
from dinkster_inference import native_unet_key_map, qwen_image_lora_key_map
from dinkster_inference_torch import (
    Flux,
    Ideogram4DiT,
    MiniMaxMusic3DiT,
    QwenImage,
    UNetModel,
    assemble_minimax_h3_dit,
    select_attention,
)

from .attachment import TargetDescriptor, resolve_lora_targets
from .checkpoint import CheckpointError, CheckpointState
from .config import (
    Flux2TrainingConfig,
    FluxTrainingConfig,
    Ideogram4TrainingConfig,
    MiniMaxH3TrainingConfig,
    MiniMaxMusic3TrainingConfig,
    QwenImageTrainingConfig,
    TrainingConfig,
    WanTrainingConfig,
)
from .trainer import minimax_h3_time_embedding_kind

_ExportDtype = Literal["fp16", "bf16", "fp32"]

_DTYPES: dict[_ExportDtype, tuple[torch.dtype, str]] = {
    "fp16": (torch.float16, "F16"),
    "bf16": (torch.bfloat16, "BF16"),
    "fp32": (torch.float32, "F32"),
}


@dataclass(frozen=True)
class LoraExportSource:
    """Adapter state and provenance for one standard LoRA export."""

    checkpoint_manifest_digest: str | None
    session_id: str
    config_digest: str
    step_cursor: int
    adapter: dict[str, torch.Tensor]

    @classmethod
    def from_checkpoint(cls, checkpoint: CheckpointState) -> LoraExportSource:
        return cls(
            checkpoint_manifest_digest=checkpoint.manifest_digest,
            session_id=checkpoint.session_id,
            config_digest=checkpoint.config_digest,
            step_cursor=checkpoint.step_cursor,
            adapter=checkpoint.adapter,
        )


@dataclass(frozen=True)
class LoraExportSettings:
    """Validated destination and storage dtype for one LoRA export."""

    path: Path
    dtype: _ExportDtype

    @classmethod
    def parse(cls, serialized: str, *, export_root: Path) -> LoraExportSettings:
        try:
            value: object = json.loads(serialized)
        except json.JSONDecodeError as exc:
            raise ValueError(f"export settings must be valid JSON: {exc.msg}") from exc
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError("export settings must be an object with string keys")
        raw = cast("dict[str, object]", value)
        unknown = sorted(set(raw) - {"path", "dtype"})
        if unknown:
            raise ValueError(f"export settings have unknown fields: {', '.join(unknown)}")
        path_value = raw.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("export settings path must be a non-empty string")
        relative = Path(path_value)
        if PurePosixPath(path_value).is_absolute() or PureWindowsPath(path_value).is_absolute():
            raise ValueError("export settings path must be relative to the training export root")
        root = export_root.resolve()
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("export settings path must stay within the training export root")
        lexical_first_parts: list[str] = []
        for lexical_path in (PurePosixPath(path_value), PureWindowsPath(path_value)):
            normalized_parts: list[str] = []
            for part in lexical_path.parts:
                if part == "..":
                    if normalized_parts:
                        normalized_parts.pop()
                elif part != ".":
                    normalized_parts.append(part)
            if normalized_parts:
                lexical_first_parts.append(normalized_parts[0])
        cadence_root = (root / "cadence").resolve()
        if any(part.casefold() == "cadence" for part in lexical_first_parts) or path.is_relative_to(
            cadence_root
        ):
            raise ValueError("export settings path uses the reserved cadence directory")
        if path.suffix.lower() != ".safetensors":
            raise ValueError("export settings path must end in .safetensors")
        dtype_value = raw.get("dtype", "fp16")
        if dtype_value not in _DTYPES:
            raise ValueError("export settings dtype must be 'fp16', 'bf16', or 'fp32'")
        return cls(path=path, dtype=dtype_value)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous().clone()
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def _cast_tensor(tensor: torch.Tensor, dtype: torch.dtype, name: str) -> torch.Tensor:
    value = tensor.to(dtype=dtype)
    if not bool(torch.isfinite(value).all().item()):
        raise CheckpointError(f"LoRA tensor {name!r} is not finite in export dtype {dtype}")
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _safetensors_bytes(
    tensors: dict[str, torch.Tensor], metadata: dict[str, str], dtype_code: str
) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("safetensors export requires a little-endian host")
    header: dict[str, object] = {"__metadata__": {key: metadata[key] for key in sorted(metadata)}}
    payload = bytearray()
    for key in sorted(tensors):
        tensor = tensors[key]
        data = _tensor_bytes(tensor)
        begin = len(payload)
        payload.extend(data)
        header[key] = {
            "dtype": dtype_code,
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


def _checkpoint_adapter_tensors(
    source: LoraExportSource,
    targets: tuple[TargetDescriptor, ...],
    rank: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    expected_state_keys = {
        f"{target.target_id}.{role}" for target in targets for role in ("down", "up")
    }
    if set(source.adapter) != expected_state_keys:
        missing = sorted(expected_state_keys - set(source.adapter))
        unknown = sorted(set(source.adapter) - expected_state_keys)
        raise CheckpointError(
            f"LoRA checkpoint keys differ during export: missing={missing}, unknown={unknown}"
        )

    values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for target in targets:
        down = source.adapter[f"{target.target_id}.down"]
        up = source.adapter[f"{target.target_id}.up"]
        down_shape = (rank, math.prod(target.weight_shape[1:]))
        up_shape = (target.weight_shape[0], rank)
        if down.dtype != torch.float32 or tuple(down.shape) != down_shape:
            raise CheckpointError(
                f"LoRA checkpoint tensor {target.target_id!r}.down has shape"
                f" {tuple(down.shape)} and dtype {down.dtype}; expected {down_shape} torch.float32"
            )
        if up.dtype != torch.float32 or tuple(up.shape) != up_shape:
            raise CheckpointError(
                f"LoRA checkpoint tensor {target.target_id!r}.up has shape"
                f" {tuple(up.shape)} and dtype {up.dtype}; expected {up_shape} torch.float32"
            )
        values[target.target_id] = (down, up)
    return values


def _kohya_tensors(
    source: LoraExportSource,
    config: TrainingConfig,
    dtype: torch.dtype,
    targets: tuple[TargetDescriptor, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if targets is None:
        with torch.device("meta"):
            model = UNetModel(config.unet)
        targets = resolve_lora_targets(model, config.rank, family=config.family)
    adapters = _checkpoint_adapter_tensors(source, targets, config.rank)

    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        down, up = adapters[target.target_id]
        if len(target.weight_shape) == 4:
            down = down.reshape(config.rank, *target.weight_shape[1:])
            up = up.reshape(target.weight_shape[0], config.rank, 1, 1)
        stem = "lora_unet_" + target.module_path.replace(".", "_")
        down_name = f"{stem}.lora_down.weight"
        up_name = f"{stem}.lora_up.weight"
        alpha_name = f"{stem}.alpha"
        tensors[down_name] = _cast_tensor(down, dtype, down_name)
        tensors[up_name] = _cast_tensor(up, dtype, up_name)
        tensors[alpha_name] = _cast_tensor(
            torch.tensor(config.alpha, dtype=torch.float64), dtype, alpha_name
        )
    return tensors


def _minimax_h3_tensors(
    source: LoraExportSource,
    config: MiniMaxH3TrainingConfig,
    dtype: torch.dtype,
    targets: tuple[TargetDescriptor, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if targets is None:
        with torch.device("meta"):
            model = assemble_minimax_h3_dit(
                time_embedding_kind=minimax_h3_time_embedding_kind(config),
                attention_selection=select_attention("flux", "sdpa"),
            )
        targets = resolve_lora_targets(model, config.rank, family="minimax-h3")
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    assert model_keys is not None
    key_map = native_unet_key_map(model_keys)
    unmatched = [
        f"diffusion_model.{target.module_path}"
        for target in targets
        if key_map.get(f"diffusion_model.{target.module_path}")
        != f"diffusion_model.{target.module_path}.weight"
    ]
    if unmatched:
        raise CheckpointError(
            "MiniMax H3 LoRA export stems do not match the native DiT state dict: "
            + ", ".join(unmatched)
        )
    adapters = _checkpoint_adapter_tensors(source, targets, config.rank)

    for target in targets:
        expected_target_id = f"minimax-h3/dit/{target.module_path}/weight"
        if target.target_id != expected_target_id:
            raise CheckpointError(
                f"MiniMax H3 target {target.target_id!r} does not encode its module path"
            )

    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        if target.operation != "linear" or len(target.weight_shape) != 2:
            raise CheckpointError(
                f"MiniMax H3 target {target.target_id!r} is not a weight-targeted linear"
            )
        down, up = adapters[target.target_id]
        stem = f"diffusion_model.{target.module_path}"
        down_name = f"{stem}.lora_A.weight"
        up_name = f"{stem}.lora_B.weight"
        alpha_name = f"{stem}.alpha"
        tensors[down_name] = _cast_tensor(down, dtype, down_name)
        tensors[up_name] = _cast_tensor(up, dtype, up_name)
        tensors[alpha_name] = _cast_tensor(
            torch.tensor(config.alpha, dtype=torch.float64), dtype, alpha_name
        )
    return tensors


def _minimax_music3_tensors(
    source: LoraExportSource,
    config: MiniMaxMusic3TrainingConfig,
    dtype: torch.dtype,
    targets: tuple[TargetDescriptor, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if targets is None:
        with torch.device("meta"):
            model = MiniMaxMusic3DiT()
        targets = resolve_lora_targets(model, config.rank, family="minimax-music3")
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    assert model_keys is not None
    key_map = native_unet_key_map(model_keys)
    unmatched = [
        f"diffusion_model.{target.module_path}"
        for target in targets
        if key_map.get(f"diffusion_model.{target.module_path}")
        != f"diffusion_model.{target.module_path}.weight"
    ]
    if unmatched:
        raise CheckpointError(
            "MiniMax Music 3 LoRA export stems do not match the native DiT state dict: "
            + ", ".join(unmatched)
        )
    adapters = _checkpoint_adapter_tensors(source, targets, config.rank)
    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        expected_target_id = f"minimax-music3/dit/{target.module_path}/weight"
        if target.target_id != expected_target_id:
            raise CheckpointError(
                f"MiniMax Music 3 target {target.target_id!r} does not encode its module path"
            )
        if target.operation != "linear" or len(target.weight_shape) != 2:
            raise CheckpointError(
                f"MiniMax Music 3 target {target.target_id!r} is not a linear weight"
            )
        down, up = adapters[target.target_id]
        stem = f"diffusion_model.{target.module_path}"
        down_name = f"{stem}.lora_A.weight"
        up_name = f"{stem}.lora_B.weight"
        alpha_name = f"{stem}.alpha"
        tensors[down_name] = _cast_tensor(down, dtype, down_name)
        tensors[up_name] = _cast_tensor(up, dtype, up_name)
        tensors[alpha_name] = _cast_tensor(
            torch.tensor(config.alpha, dtype=torch.float64), dtype, alpha_name
        )
    return tensors


def _wan_tensors(
    source: LoraExportSource,
    config: WanTrainingConfig,
    dtype: torch.dtype,
    targets: tuple[TargetDescriptor, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if targets is None:
        from dinkster_inference_torch.wan21_model import Wan21Model

        from .trainer import wan_model_assembly_plan

        plan = wan_model_assembly_plan(config)
        with torch.device("meta"):
            model = Wan21Model(plan.diffusion.config)
        targets = resolve_lora_targets(model, config.rank, family="wan")
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    assert model_keys is not None
    key_map = native_unet_key_map(model_keys)
    unmatched = [
        "lora_unet_" + target.module_path.replace(".", "_")
        for target in targets
        if key_map.get("lora_unet_" + target.module_path.replace(".", "_"))
        != f"diffusion_model.{target.module_path}.weight"
    ]
    if unmatched:
        raise CheckpointError(
            "Wan LoRA export stems do not match the native DiT state dict: " + ", ".join(unmatched)
        )
    adapters = _checkpoint_adapter_tensors(source, targets, config.rank)

    for target in targets:
        expected_target_id = f"wan/dit/{target.module_path}/weight"
        if target.target_id != expected_target_id:
            raise CheckpointError(
                f"Wan target {target.target_id!r} does not encode its module path"
            )

    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        if target.operation != "linear" or len(target.weight_shape) != 2:
            raise CheckpointError(f"Wan target {target.target_id!r} is not a linear weight")
        down, up = adapters[target.target_id]
        # The extra underscore is the Wan Fun kohya convention used by downstream loaders.
        stem = "lora_unet__" + target.module_path.replace(".", "_")
        down_name = f"{stem}.lora_down.weight"
        up_name = f"{stem}.lora_up.weight"
        alpha_name = f"{stem}.alpha"
        tensors[down_name] = _cast_tensor(down, dtype, down_name)
        tensors[up_name] = _cast_tensor(up, dtype, up_name)
        tensors[alpha_name] = _cast_tensor(
            torch.tensor(config.alpha, dtype=torch.float64), dtype, alpha_name
        )
    return tensors


def _flux_tensors(
    source: LoraExportSource,
    config: FluxTrainingConfig,
    dtype: torch.dtype,
    targets: tuple[TargetDescriptor, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if targets is None:
        from .trainer import flux_model_assembly_plan

        plan = flux_model_assembly_plan(config)
        with torch.device("meta"):
            model = Flux(plan.diffusion.config)
        targets = resolve_lora_targets(model, config.rank, family="flux")
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    assert model_keys is not None
    key_map = native_unet_key_map(model_keys)
    unmatched = [
        f"diffusion_model.{target.module_path}"
        for target in targets
        if key_map.get(f"diffusion_model.{target.module_path}")
        != f"diffusion_model.{target.module_path}.weight"
    ]
    if unmatched:
        raise CheckpointError(
            "Flux LoRA export stems do not match the native DiT state dict: " + ", ".join(unmatched)
        )
    adapters = _checkpoint_adapter_tensors(source, targets, config.rank)

    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        expected_target_id = f"flux/dit/{target.module_path}/weight"
        if target.target_id != expected_target_id:
            raise CheckpointError(
                f"Flux target {target.target_id!r} does not encode its module path"
            )
        if target.operation != "linear" or len(target.weight_shape) != 2:
            raise CheckpointError(f"Flux target {target.target_id!r} is not a linear weight")
        down, up = adapters[target.target_id]
        stem = f"diffusion_model.{target.module_path}"
        down_name = f"{stem}.lora_A.weight"
        up_name = f"{stem}.lora_B.weight"
        alpha_name = f"{stem}.alpha"
        tensors[down_name] = _cast_tensor(down, dtype, down_name)
        tensors[up_name] = _cast_tensor(up, dtype, up_name)
        tensors[alpha_name] = _cast_tensor(
            torch.tensor(config.alpha, dtype=torch.float64), dtype, alpha_name
        )
    return tensors


def _flux2_tensors(
    source: LoraExportSource,
    config: Flux2TrainingConfig,
    dtype: torch.dtype,
    targets: tuple[TargetDescriptor, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if targets is None:
        from .trainer import flux2_model_assembly_plan

        plan = flux2_model_assembly_plan(config)
        with torch.device("meta"):
            model = Flux(plan.diffusion.config)
        targets = resolve_lora_targets(model, config.rank, family="flux2")
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    assert model_keys is not None
    key_map = native_unet_key_map(model_keys)
    unmatched = [
        f"diffusion_model.{target.module_path}"
        for target in targets
        if key_map.get(f"diffusion_model.{target.module_path}")
        != f"diffusion_model.{target.module_path}.weight"
    ]
    if unmatched:
        raise CheckpointError(
            "Flux2 LoRA export stems do not match the native DiT state dict: "
            + ", ".join(unmatched)
        )
    adapters = _checkpoint_adapter_tensors(source, targets, config.rank)

    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        expected_target_id = f"flux2/dit/{target.module_path}/weight"
        if target.target_id != expected_target_id:
            raise CheckpointError(
                f"Flux2 target {target.target_id!r} does not encode its module path"
            )
        if target.operation != "linear" or len(target.weight_shape) != 2:
            raise CheckpointError(f"Flux2 target {target.target_id!r} is not a linear weight")
        down, up = adapters[target.target_id]
        stem = f"diffusion_model.{target.module_path}"
        down_name = f"{stem}.lora_A.weight"
        up_name = f"{stem}.lora_B.weight"
        alpha_name = f"{stem}.alpha"
        tensors[down_name] = _cast_tensor(down, dtype, down_name)
        tensors[up_name] = _cast_tensor(up, dtype, up_name)
        tensors[alpha_name] = _cast_tensor(
            torch.tensor(config.alpha, dtype=torch.float64), dtype, alpha_name
        )
    return tensors


def _qwen_image_tensors(
    source: LoraExportSource,
    config: QwenImageTrainingConfig,
    dtype: torch.dtype,
    targets: tuple[TargetDescriptor, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if targets is None:
        from .trainer import qwen_image_model_assembly_plan

        plan = qwen_image_model_assembly_plan(config)
        with torch.device("meta"):
            model = QwenImage(plan.diffusion.config)
        targets = resolve_lora_targets(model, config.rank, family="qwen-image")
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    assert model_keys is not None
    key_map = qwen_image_lora_key_map(model_keys)
    mismatched = [
        f"transformer.{target.module_path}"
        for target in targets
        if key_map.get(f"transformer.{target.module_path}")
        != f"diffusion_model.{target.module_path}.weight"
    ]
    if mismatched:
        raise CheckpointError(
            "Qwen-Image LoRA export stems do not match the native DiT state dict: "
            + ", ".join(mismatched)
        )
    adapters = _checkpoint_adapter_tensors(source, targets, config.rank)

    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        expected_target_id = f"qwen-image/dit/{target.module_path}/weight"
        if target.target_id != expected_target_id:
            raise CheckpointError(
                f"Qwen-Image target {target.target_id!r} does not encode its module path"
            )
        if target.operation != "linear" or len(target.weight_shape) != 2:
            raise CheckpointError(
                f"Qwen-Image target {target.target_id!r} is not a weight-targeted linear"
            )
        down, up = adapters[target.target_id]
        stem = f"transformer.{target.module_path}"
        down_name = f"{stem}.lora_A.weight"
        up_name = f"{stem}.lora_B.weight"
        alpha_name = f"{stem}.alpha"
        tensors[down_name] = _cast_tensor(down, dtype, down_name)
        tensors[up_name] = _cast_tensor(up, dtype, up_name)
        tensors[alpha_name] = _cast_tensor(
            torch.tensor(config.alpha, dtype=torch.float64), dtype, alpha_name
        )
    return tensors


def _ideogram4_tensors(
    source: LoraExportSource,
    config: Ideogram4TrainingConfig,
    dtype: torch.dtype,
    targets: tuple[TargetDescriptor, ...] | None = None,
    model_keys: tuple[str, ...] | None = None,
) -> dict[str, torch.Tensor]:
    if targets is None:
        from .trainer import ideogram4_component_plans

        ideogram4_component_plans(config)
        with torch.device("meta"):
            model = Ideogram4DiT()
        targets = resolve_lora_targets(model, config.rank, family="ideogram4", role=config.role)
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    assert model_keys is not None
    key_map = native_unet_key_map(model_keys)
    mismatched = [
        f"diffusion_model.{target.module_path}"
        for target in targets
        if key_map.get(f"diffusion_model.{target.module_path}")
        != f"diffusion_model.{target.module_path}.weight"
    ]
    if mismatched:
        raise CheckpointError(
            "Ideogram 4 LoRA export stems do not match the native DiT state dict: "
            + ", ".join(mismatched)
        )
    adapters = _checkpoint_adapter_tensors(source, targets, config.rank)
    tensors: dict[str, torch.Tensor] = {}
    for target in targets:
        expected_target_id = f"ideogram4/{config.role}/dit/{target.module_path}/weight"
        if target.target_id != expected_target_id:
            raise CheckpointError(
                f"Ideogram 4 target {target.target_id!r} does not encode its role and module path"
            )
        if target.operation != "linear" or len(target.weight_shape) != 2:
            raise CheckpointError(
                f"Ideogram 4 target {target.target_id!r} is not a weight-targeted linear"
            )
        down, up = adapters[target.target_id]
        stem = f"diffusion_model.{target.module_path}"
        down_name = f"{stem}.lora_A.weight"
        up_name = f"{stem}.lora_B.weight"
        alpha_name = f"{stem}.alpha"
        tensors[down_name] = _cast_tensor(down, dtype, down_name)
        tensors[up_name] = _cast_tensor(up, dtype, up_name)
        tensors[alpha_name] = _cast_tensor(
            torch.tensor(config.alpha, dtype=torch.float64), dtype, alpha_name
        )
    return tensors


def _write_export(
    source: LoraExportSource,
    settings: LoraExportSettings,
    tensors: dict[str, torch.Tensor],
    dtype_code: str,
    runtime_identity: str,
    extra_metadata: dict[str, str] | None = None,
) -> tuple[Path, str]:
    metadata = {
        "dinkster_config_digest": source.config_digest,
        "dinkster_runtime_identity": runtime_identity,
        "dinkster_session_id": source.session_id,
        "dinkster_step_cursor": str(source.step_cursor),
    }
    if source.checkpoint_manifest_digest is not None:
        metadata["dinkster_checkpoint_manifest_digest"] = source.checkpoint_manifest_digest
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    data = _safetensors_bytes(tensors, metadata, dtype_code)
    digest = digest_bytes(data)
    settings.path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=settings.path.parent, delete=False) as file:
        temporary = Path(file.name)
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    try:
        if os.name != "nt":
            os.chmod(temporary, 0o644)
        os.replace(temporary, settings.path)
        _fsync_directory(settings.path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return settings.path, digest


def export_kohya_lora(
    checkpoint: CheckpointState,
    config: TrainingConfig,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write one deterministic kohya-layout native SD UNet LoRA file."""
    source = LoraExportSource.from_checkpoint(checkpoint)
    dtype, dtype_code = _DTYPES[settings.dtype]
    tensors = _kohya_tensors(source, config, dtype)
    return _write_export(source, settings, tensors, dtype_code, runtime_identity)


def export_minimax_h3_lora(
    checkpoint: CheckpointState,
    config: MiniMaxH3TrainingConfig,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write one deterministic PEFT-layout native MiniMax H3 DiT LoRA file."""
    source = LoraExportSource.from_checkpoint(checkpoint)
    dtype, dtype_code = _DTYPES[settings.dtype]
    tensors = _minimax_h3_tensors(source, config, dtype)
    return _write_export(source, settings, tensors, dtype_code, runtime_identity)


def export_minimax_music3_lora(
    checkpoint: CheckpointState,
    config: MiniMaxMusic3TrainingConfig,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write one inference-compatible PEFT-layout Music 3 DiT LoRA file."""
    source = LoraExportSource.from_checkpoint(checkpoint)
    dtype, dtype_code = _DTYPES[settings.dtype]
    tensors = _minimax_music3_tensors(source, config, dtype)
    return _write_export(source, settings, tensors, dtype_code, runtime_identity)


def export_wan_lora(
    checkpoint: CheckpointState,
    config: WanTrainingConfig,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write one deterministic Wan Fun kohya-layout Wan DiT LoRA file."""
    source = LoraExportSource.from_checkpoint(checkpoint)
    dtype, dtype_code = _DTYPES[settings.dtype]
    tensors = _wan_tensors(source, config, dtype)
    return _write_export(
        source,
        settings,
        tensors,
        dtype_code,
        runtime_identity,
        _wan_export_metadata(config),
    )


def export_flux_lora(
    checkpoint: CheckpointState,
    config: FluxTrainingConfig,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write one deterministic PEFT-layout classic Flux DiT LoRA file."""
    source = LoraExportSource.from_checkpoint(checkpoint)
    dtype, dtype_code = _DTYPES[settings.dtype]
    tensors = _flux_tensors(source, config, dtype)
    return _write_export(
        source,
        settings,
        tensors,
        dtype_code,
        runtime_identity,
        _flux_export_metadata(config),
    )


def export_flux2_lora(
    checkpoint: CheckpointState,
    config: Flux2TrainingConfig,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write one deterministic PEFT-layout Flux2 DiT LoRA file."""
    source = LoraExportSource.from_checkpoint(checkpoint)
    dtype, dtype_code = _DTYPES[settings.dtype]
    tensors = _flux2_tensors(source, config, dtype)
    return _write_export(
        source,
        settings,
        tensors,
        dtype_code,
        runtime_identity,
        _flux2_export_metadata(config),
    )


def export_qwen_image_lora(
    checkpoint: CheckpointState,
    config: QwenImageTrainingConfig,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write one deterministic PEFT-layout Qwen-Image DiT LoRA file."""
    source = LoraExportSource.from_checkpoint(checkpoint)
    dtype, dtype_code = _DTYPES[settings.dtype]
    tensors = _qwen_image_tensors(source, config, dtype)
    return _write_export(
        source,
        settings,
        tensors,
        dtype_code,
        runtime_identity,
        _qwen_image_export_metadata(config),
    )


def export_ideogram4_lora(
    checkpoint: CheckpointState,
    config: Ideogram4TrainingConfig,
    settings: LoraExportSettings,
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write one deterministic role-bound native Ideogram 4 LoRA file."""
    source = LoraExportSource.from_checkpoint(checkpoint)
    dtype, dtype_code = _DTYPES[settings.dtype]
    tensors = _ideogram4_tensors(source, config, dtype)
    return _write_export(
        source,
        settings,
        tensors,
        dtype_code,
        runtime_identity,
        _ideogram4_export_metadata(config),
    )


def _wan_export_metadata(config: WanTrainingConfig) -> dict[str, str] | None:
    if config.variant == "wan21-t2v":
        return None
    metadata = {"dinkster_wan_variant": config.variant}
    if config.expert is not None:
        metadata["dinkster_wan_expert"] = config.expert
    return metadata


def _flux_export_metadata(config: FluxTrainingConfig) -> dict[str, str]:
    return {"dinkster_flux_variant": config.variant}


def _flux2_export_metadata(config: Flux2TrainingConfig) -> dict[str, str]:
    return {"dinkster_flux2_variant": config.variant}


def _qwen_image_export_metadata(config: QwenImageTrainingConfig) -> dict[str, str]:
    return {"dinkster_qwen_image_variant": config.variant}


def _ideogram4_export_metadata(config: Ideogram4TrainingConfig) -> dict[str, str]:
    return {
        "dinkster_ideogram4_role": config.role,
        "dinkster_ideogram4_base_storage": config.base_storage,
    }


def export_intermediate_lora(
    source: LoraExportSource,
    config: (
        TrainingConfig
        | FluxTrainingConfig
        | Flux2TrainingConfig
        | MiniMaxH3TrainingConfig
        | MiniMaxMusic3TrainingConfig
        | QwenImageTrainingConfig
        | Ideogram4TrainingConfig
        | WanTrainingConfig
    ),
    settings: LoraExportSettings,
    targets: tuple[TargetDescriptor, ...],
    model_state_keys: tuple[str, ...],
    *,
    runtime_identity: str,
) -> tuple[Path, str]:
    """Write adapter masters from an in-progress training runtime."""
    dtype, dtype_code = _DTYPES[settings.dtype]
    model_keys = tuple(f"diffusion_model.{key}" for key in model_state_keys)
    if isinstance(config, MiniMaxH3TrainingConfig):
        tensors = _minimax_h3_tensors(source, config, dtype, targets, model_keys)
    elif isinstance(config, MiniMaxMusic3TrainingConfig):
        tensors = _minimax_music3_tensors(source, config, dtype, targets, model_keys)
    elif isinstance(config, WanTrainingConfig):
        tensors = _wan_tensors(source, config, dtype, targets, model_keys)
    elif isinstance(config, FluxTrainingConfig):
        tensors = _flux_tensors(source, config, dtype, targets, model_keys)
    elif isinstance(config, Flux2TrainingConfig):
        tensors = _flux2_tensors(source, config, dtype, targets, model_keys)
    elif isinstance(config, QwenImageTrainingConfig):
        tensors = _qwen_image_tensors(source, config, dtype, targets, model_keys)
    elif isinstance(config, Ideogram4TrainingConfig):
        tensors = _ideogram4_tensors(source, config, dtype, targets, model_keys)
    else:
        tensors = _kohya_tensors(source, config, dtype, targets)
    return _write_export(
        source,
        settings,
        tensors,
        dtype_code,
        runtime_identity,
        (
            _wan_export_metadata(config)
            if isinstance(config, WanTrainingConfig)
            else _flux_export_metadata(config)
            if isinstance(config, FluxTrainingConfig)
            else _flux2_export_metadata(config)
            if isinstance(config, Flux2TrainingConfig)
            else _qwen_image_export_metadata(config)
            if isinstance(config, QwenImageTrainingConfig)
            else _ideogram4_export_metadata(config)
            if isinstance(config, Ideogram4TrainingConfig)
            else None
        ),
    )


__all__ = [
    "LoraExportSettings",
    "export_flux2_lora",
    "export_flux_lora",
    "export_ideogram4_lora",
    "export_kohya_lora",
    "export_minimax_h3_lora",
    "export_minimax_music3_lora",
    "export_qwen_image_lora",
    "export_wan_lora",
]
