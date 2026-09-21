"""Summarize applied LoRA weight-delta magnitudes without loading a base model."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import torch
from safetensors.torch import load_file


def _family(name: str) -> str:
    lowered = name.lower()
    if "attn1" in lowered:
        return "self_attention"
    if "attn2" in lowered:
        return "cross_attention"
    if "ff" in lowered or "_net_" in lowered:
        return "feed_forward"
    if "proj_in" in lowered or "proj_out" in lowered:
        return "spatial_projection"
    return "other"


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "p25": _quantile(values, 0.25),
        "p75": _quantile(values, 0.75),
        "p95": _quantile(values, 0.95),
        "zero_count": sum(value == 0.0 for value in values),
    }


class _ModuleStatistics(NamedTuple):
    alpha: float
    alpha_tensor_present: float
    applied_delta_rms: float
    down_rms: float
    effective_scale: float
    normalized_delta_rms: float
    rank: float
    up_rms: float


def summarize(path: Path) -> dict[str, object]:
    tensors = load_file(path, device="cpu")
    grouped: dict[str, list[_ModuleStatistics]] = defaultdict(list)
    down_suffixes = (".lora_down.weight", ".lora_A.weight")
    for name, down in tensors.items():
        suffix = next((item for item in down_suffixes if name.endswith(item)), None)
        if suffix is None:
            continue
        stem = name[: -len(suffix)]
        up_names = (stem + ".lora_up.weight", stem + ".lora_B.weight")
        up_name = next((item for item in up_names if item in tensors), None)
        if up_name is None:
            raise ValueError(f"missing LoRA up tensor for {name}")
        up = tensors[up_name].float().reshape(tensors[up_name].shape[0], -1)
        down_matrix = down.float().reshape(down.shape[0], -1)
        alpha_tensor = tensors.get(stem + ".alpha")
        alpha = float(down.shape[0]) if alpha_tensor is None else float(alpha_tensor.item())
        rank = float(down.shape[0])
        effective_scale = alpha / rank
        if effective_scale <= 0.0 or not math.isfinite(effective_scale):
            raise ValueError(f"LoRA application scale for {stem} is not positive and finite")
        normalized_delta = torch.mm(up, down_matrix)
        delta = normalized_delta * effective_scale
        rms = float(delta.square().mean().sqrt().item())
        if not math.isfinite(rms):
            raise ValueError(f"LoRA delta for {stem} is non-finite")
        grouped[_family(stem)].append(
            _ModuleStatistics(
                alpha=alpha,
                alpha_tensor_present=float(alpha_tensor is not None),
                applied_delta_rms=rms,
                down_rms=float(down_matrix.square().mean().sqrt().item()),
                effective_scale=effective_scale,
                normalized_delta_rms=float(normalized_delta.square().mean().sqrt().item()),
                rank=rank,
                up_rms=float(up.square().mean().sqrt().item()),
            )
        )
    if not grouped:
        raise ValueError(f"no LoRA tensor pairs found in {path}")
    families: dict[str, object] = {}
    for family, modules in sorted(grouped.items()):
        families[family] = {
            field: _distribution([getattr(module, field) for module in modules])
            for field in _ModuleStatistics._fields
        }
    return {
        "adapter": str(path.resolve()),
        "application_scale_source": "per-module alpha tensor divided by lora_down rank",
        "families": families,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(summarize(args.adapter), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
