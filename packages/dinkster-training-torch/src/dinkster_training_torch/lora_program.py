"""Trainable LoRA resources bound to the dinkster_inference patch program."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import cast

import torch
from dinkster_inference.lora import model_lora_keys_unet
from dinkster_inference.model_patcher import ModelPatcher
from dinkster_inference.weight_adapter import LoRAAdapter
from dinkster_inference.weight_adapter.lora import LoraDiff


class LoRAProgramError(ValueError):
    """The fork model cannot satisfy the requested LoRA target contract."""


@dataclass(frozen=True)
class Target:
    export_stem: str
    model_key: str


def _target_seed(seed: int, target: str) -> int:
    digest = hashlib.sha256(f"{seed}:adapter:{target}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _module(model: torch.nn.Module, model_key: str) -> torch.nn.Module:
    return model.get_submodule(model_key.removesuffix(".weight"))


def resolve_targets(model: torch.nn.Module, patterns: tuple[str, ...]) -> tuple[Target, ...]:
    key_map = model_lora_keys_unet(model, {})
    by_model_key: dict[str, str] = {}
    for export_stem, model_key in sorted(key_map.items()):
        if not export_stem.startswith("lora_unet_"):
            continue
        if not any(pattern in export_stem or pattern in model_key for pattern in patterns):
            continue
        by_model_key.setdefault(model_key, export_stem)
    targets = tuple(
        Target(export_stem, model_key) for model_key, export_stem in sorted(by_model_key.items())
    )
    if not targets:
        raise LoRAProgramError("the SD1.5 model exposes no requested LoRA targets")
    return targets


class LoRAProgram:
    """Trainable LoRA resources materialized by the model patch program."""

    _PROGRAM_KEY = "dinkster-training"

    def __init__(
        self,
        model_patcher: ModelPatcher,
        *,
        rank: int,
        alpha: float,
        seed: int,
        target_patterns: tuple[str, ...],
        device: torch.device,
    ) -> None:
        self._model_patcher = model_patcher
        model = model_patcher.model
        self.targets = resolve_targets(model, target_patterns)
        self._adapters: dict[str, LoraDiff] = {}
        for target in self.targets:
            module = _module(model, target.model_key)
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.ndim not in (2, 4):
                raise LoRAProgramError(f"LoRA target {target.model_key!r} has no supported weight")
            devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(_target_seed(seed, target.model_key))
                if device.type == "cuda":
                    torch.cuda.manual_seed(_target_seed(seed, target.model_key))
                adapter = LoRAAdapter.create_train(weight, rank=rank, alpha=alpha)
            adapter.to(device=device, dtype=torch.float32)
            self._adapters[target.model_key] = adapter
        model_patcher.set_bypass_adapters(
            self._PROGRAM_KEY,
            {key: (adapter, 1.0) for key, adapter in self._adapters.items()},
        )

    def parameters(self) -> list[torch.nn.Parameter]:
        return [
            cast("torch.nn.Parameter", parameter)
            for adapter in self._adapters.values()
            for parameter in (adapter.lora_down.weight, adapter.lora_up.weight)
        ]

    def state_dict(self) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        for target in self.targets:
            adapter = self._adapters[target.model_key]
            values[f"{target.export_stem}.lora_down.weight"] = (
                adapter.lora_down.weight.detach().float().cpu().contiguous()
            )
            values[f"{target.export_stem}.lora_up.weight"] = (
                adapter.lora_up.weight.detach().float().cpu().contiguous()
            )
            values[f"{target.export_stem}.alpha"] = adapter.alpha.detach().float().cpu()
        return values

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = set(self.state_dict())
        if set(state) != expected:
            raise LoRAProgramError(
                f"LoRA checkpoint keys differ: missing={sorted(expected - set(state))}, "
                f"unknown={sorted(set(state) - expected)}"
            )
        for target in self.targets:
            adapter = self._adapters[target.model_key]
            adapter.lora_down.weight.data.copy_(state[f"{target.export_stem}.lora_down.weight"])
            adapter.lora_up.weight.data.copy_(state[f"{target.export_stem}.lora_up.weight"])

    def inject(self, device: torch.device) -> torch.nn.Module:
        model = self._model_patcher.patch_model(device_to=device)
        for adapter in self._adapters.values():
            adapter.to(device=device, dtype=torch.float32)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        return model

    def close(self) -> None:
        self._model_patcher.unpatch_model(device_to=torch.device("cpu"))
        self._model_patcher.remove_bypass_adapters(self._PROGRAM_KEY)
