"""Live float32 LoRA attachments for native UNet and DiT projections."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn.functional as functional
from dinkster_inference_torch import Fp8Linear, Int8Linear


class AttachmentError(ValueError):
    """The model cannot satisfy the LoRA attachment contract."""


@dataclass(frozen=True)
class TargetDescriptor:
    """One stable trainable target resolved before optimizer creation."""

    target_id: str
    module_path: str
    operation: str
    weight_shape: tuple[int, ...]
    base_parameter_count: int
    adapter_parameter_count: int

    def to_wire(self) -> dict[str, object]:
        return {
            "targetId": self.target_id,
            "modulePath": self.module_path,
            "operation": self.operation,
            "weightShape": list(self.weight_shape),
            "baseParameterCount": self.base_parameter_count,
            "adapterParameterCount": self.adapter_parameter_count,
        }


_MINIMAX_MUSIC3_LORA_TARGET_PATHS = frozenset(
    (
        "diffusion_transformer.transformer.project_in",
        "diffusion_transformer.transformer.project_out",
        *(
            f"diffusion_transformer.transformer.layers.{block}.{suffix}"
            for block in range(36)
            for suffix in (
                "self_attn.to_qkv",
                "self_attn.to_out",
                "ff.ff.0.proj",
                "ff.ff.2",
            )
        ),
    )
)


def _selected(path: str, module: torch.nn.Module, family: str) -> bool:
    if family == "minimax-h3":
        return isinstance(module, (torch.nn.Linear, Int8Linear))
    if family == "minimax-music3":
        return path in _MINIMAX_MUSIC3_LORA_TARGET_PATHS and isinstance(
            module, (torch.nn.Linear, Int8Linear)
        )
    if family == "wan":
        parts = path.split(".")
        return (
            isinstance(module, torch.nn.Linear)
            and len(parts) == 4
            and parts[0] == "blocks"
            and parts[1].isdigit()
            and (
                (parts[2] in ("self_attn", "cross_attn") and parts[3] in ("q", "k", "v", "o"))
                or (parts[2] == "ffn" and parts[3] in ("0", "2"))
            )
        )
    if family in ("flux", "flux2"):
        parts = path.split(".")
        return isinstance(module, torch.nn.Linear) and (
            (
                len(parts) == 4
                and parts[0] == "double_blocks"
                and parts[1].isdigit()
                and (
                    (parts[2] in ("img_attn", "txt_attn") and parts[3] in ("qkv", "proj"))
                    or (parts[2] in ("img_mlp", "txt_mlp") and parts[3] in ("0", "2"))
                )
            )
            or (
                len(parts) == 3
                and parts[0] == "single_blocks"
                and parts[1].isdigit()
                and parts[2] in ("linear1", "linear2")
            )
        )
    if family == "qwen-image":
        parts = path.split(".")
        return isinstance(module, torch.nn.Linear) and (
            (
                len(parts) == 4
                and parts[0] == "transformer_blocks"
                and parts[1].isdigit()
                and parts[2] == "attn"
                and parts[3]
                in (
                    "to_q",
                    "to_k",
                    "to_v",
                    "add_q_proj",
                    "add_k_proj",
                    "add_v_proj",
                    "to_add_out",
                )
            )
            or (
                len(parts) == 5
                and parts[0] == "transformer_blocks"
                and parts[1].isdigit()
                and parts[2] == "attn"
                and parts[3:] == ["to_out", "0"]
            )
            or (
                len(parts) == 5
                and parts[0] == "transformer_blocks"
                and parts[1].isdigit()
                and parts[2] in ("img_mlp", "txt_mlp")
                and parts[3:] == ["net", "2"]
            )
            or (
                len(parts) == 6
                and parts[0] == "transformer_blocks"
                and parts[1].isdigit()
                and parts[2] in ("img_mlp", "txt_mlp")
                and parts[3:] == ["net", "0", "proj"]
            )
        )
    if family == "ideogram4":
        parts = path.split(".")
        return (
            isinstance(module, (torch.nn.Linear, Fp8Linear, Int8Linear))
            and len(parts) == 4
            and parts[0] == "layers"
            and parts[1].isdigit()
            and (
                (parts[2] == "attention" and parts[3] in ("qkv", "o"))
                or (parts[2] == "feed_forward" and parts[3] in ("w1", "w2", "w3"))
            )
        )
    if isinstance(module, torch.nn.Linear):
        return ".transformer_blocks." in f".{path}." or path.endswith((".proj_in", ".proj_out"))
    if isinstance(module, torch.nn.Conv2d):
        return path.endswith((".proj_in", ".proj_out")) or path in ("proj_in", "proj_out")
    return False


def resolve_lora_targets(
    model: torch.nn.Module,
    rank: int,
    *,
    family: Literal[
        "sd15",
        "sdxl",
        "flux",
        "flux2",
        "qwen-image",
        "ideogram4",
        "minimax-h3",
        "minimax-music3",
        "wan",
    ] = "sd15",
    role: Literal["conditional", "unconditional"] | None = None,
) -> tuple[TargetDescriptor, ...]:
    """Resolve the stable projection set for one supported model family."""
    if rank < 1:
        raise AttachmentError("LoRA rank must be >= 1")
    targets: list[TargetDescriptor] = []
    for path, module in model.named_modules():
        if not _selected(path, module, family):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim not in (2, 4):
            raise AttachmentError(f"target {path!r} has no supported rank-2/rank-4 weight")
        if not weight.is_floating_point() and not (
            family in ("minimax-h3", "minimax-music3", "ideogram4")
            and isinstance(module, Int8Linear)
        ):
            raise AttachmentError(f"target {path!r} has non-floating base storage {weight.dtype}")
        out_features = weight.shape[0]
        in_features = math.prod(weight.shape[1:])
        operation = "linear" if weight.ndim == 2 else "conv2d"
        component = (
            "dit"
            if family
            in (
                "flux",
                "flux2",
                "qwen-image",
                "ideogram4",
                "minimax-h3",
                "minimax-music3",
                "wan",
            )
            else "unet"
        )
        target_family = f"{family}/{role}" if family == "ideogram4" else family
        targets.append(
            TargetDescriptor(
                target_id=f"{target_family}/{component}/{path}/weight",
                module_path=path,
                operation=operation,
                weight_shape=tuple(weight.shape),
                base_parameter_count=weight.numel(),
                adapter_parameter_count=rank * (in_features + out_features),
            )
        )
    if family == "minimax-music3":
        actual = frozenset(target.module_path for target in targets)
        if actual != _MINIMAX_MUSIC3_LORA_TARGET_PATHS:
            missing = sorted(_MINIMAX_MUSIC3_LORA_TARGET_PATHS - actual)
            raise AttachmentError(
                "the MiniMax Music 3 model differs from its stable 146-target manifest: missing "
                + ", ".join(missing)
            )
    if family == "ideogram4":
        if role not in ("conditional", "unconditional"):
            raise AttachmentError("Ideogram 4 LoRA targets require a model role")
        if len(targets) != 170:
            raise AttachmentError(
                f"the Ideogram 4 model differs from its stable 170-target manifest: {len(targets)}"
            )
    elif role is not None:
        raise AttachmentError("model role applies only to Ideogram 4 LoRA targets")
    if not targets:
        raise AttachmentError(f"the {family} model exposes no supported LoRA projection targets")
    return tuple(sorted(targets, key=lambda target: target.target_id))


def _target_seed(seed: int, target_id: str, role: str) -> int:
    payload = f"{seed}:adapter:{target_id}:{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


class _LoRABranch(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Linear | torch.nn.Conv2d | Fp8Linear | Int8Linear,
        *,
        rank: int,
        alpha: float,
        target_id: str,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        shape = tuple(module.weight.shape)
        self.operation = "conv2d" if isinstance(module, torch.nn.Conv2d) else "linear"
        self.scale = alpha / rank
        self.rank = rank
        self.weight_shape = shape
        in_features = math.prod(shape[1:])
        down = torch.empty((rank, in_features), dtype=torch.float32, device="cpu")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_target_seed(seed, target_id, "down"))
        gain = math.sqrt(2.0 / (1.0 + math.sqrt(5.0) ** 2))
        bound = math.sqrt(3.0) * gain / math.sqrt(in_features)
        down.uniform_(-bound, bound, generator=generator)
        self.down = torch.nn.Parameter(down.to(device=device))
        self.up = torch.nn.Parameter(
            torch.zeros((shape[0], rank), dtype=torch.float32, device=device)
        )
        if isinstance(module, torch.nn.Conv2d):
            self.stride = module.stride
            self.padding = module.padding
            self.dilation = module.dilation
            self.groups = module.groups
            self.padding_mode = module.padding_mode
            if isinstance(module.padding, str):
                if module.padding == "same":
                    height = module.dilation[0] * (module.kernel_size[0] - 1)
                    width = module.dilation[1] * (module.kernel_size[1] - 1)
                    self.reversed_padding = (
                        width // 2,
                        width - width // 2,
                        height // 2,
                        height - height // 2,
                    )
                else:
                    self.reversed_padding = (0, 0, 0, 0)
            else:
                height, width = module.padding
                self.reversed_padding = (width, width, height, height)

    def _conv2d(self, inputs: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
        if self.padding_mode != "zeros":
            inputs = functional.pad(inputs, self.reversed_padding, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        down = self.down if dtype is None else self.down.to(dtype)
        up = self.up if dtype is None else self.up.to(dtype)
        down = down.reshape(self.rank, *self.weight_shape[1:])
        if self.groups > 1:
            down = down.repeat(self.groups, 1, 1, 1)
        hidden = functional.conv2d(
            inputs,
            down,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return functional.conv2d(
            hidden,
            up.reshape(self.weight_shape[0], self.rank, 1, 1),
            groups=self.groups,
        )

    def forward(self, inputs: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
        if dtype is not None:
            inputs = inputs.to(dtype)
        if self.operation == "linear":
            down = self.down if dtype is None else self.down.to(dtype)
            up = self.up if dtype is None else self.up.to(dtype)
            hidden = functional.linear(inputs, down)
            return functional.linear(hidden, up) * self.scale
        return self._conv2d(inputs, dtype) * self.scale

    def add_to_output(
        self,
        _module: torch.nn.Module,
        args: tuple[object, ...],
        output: object,
    ) -> torch.Tensor:
        inputs = cast("torch.Tensor", args[0])
        base = cast("torch.Tensor", output)
        dtype = None if torch.is_autocast_enabled(base.device.type) else base.dtype
        return base + self(inputs, dtype).to(dtype=base.dtype)


class TrainableAttachment:
    """Session-owned LoRA masters layered over an immutable model base."""

    def __init__(
        self,
        model: torch.nn.Module,
        targets: tuple[TargetDescriptor, ...],
        *,
        rank: int,
        alpha: float,
        seed: int,
        device: torch.device | None = None,
    ) -> None:
        modules = dict(model.named_modules())
        branches: dict[str, _LoRABranch] = {}
        handles: list[torch.utils.hooks.RemovableHandle] = []
        for target in targets:
            module = cast(
                "torch.nn.Linear | torch.nn.Conv2d | Fp8Linear | Int8Linear",
                modules[target.module_path],
            )
            weight = module.weight
            adapter = _LoRABranch(
                module,
                rank=rank,
                alpha=alpha,
                target_id=target.target_id,
                seed=seed,
                device=weight.device if device is None else device,
            )
            handles.append(module.register_forward_hook(adapter.add_to_output))
            branches[target.target_id] = adapter
        self.targets = targets
        self._branches = branches
        self._handles = handles
        self.rank = rank
        self.alpha = alpha
        self._assert_float32()

    @classmethod
    def attach(
        cls,
        model: torch.nn.Module,
        *,
        rank: int,
        alpha: float,
        seed: int,
        family: Literal[
            "sd15",
            "sdxl",
            "flux",
            "flux2",
            "qwen-image",
            "ideogram4",
            "minimax-h3",
            "minimax-music3",
            "wan",
        ] = "sd15",
        device: torch.device | None = None,
        role: Literal["conditional", "unconditional"] | None = None,
    ) -> TrainableAttachment:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        targets = resolve_lora_targets(model, rank, family=family, role=role)
        return cls(model, targets, rank=rank, alpha=alpha, seed=seed, device=device)

    def detach(self) -> None:
        """Remove every adapter hook without changing base parameter storage."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _assert_float32(self) -> None:
        for name, parameter in self.named_parameters():
            if parameter.dtype != torch.float32 or not parameter.requires_grad:
                raise AttachmentError(f"LoRA master {name!r} must be trainable float32")

    def named_parameters(self) -> tuple[tuple[str, torch.nn.Parameter], ...]:
        values: list[tuple[str, torch.nn.Parameter]] = []
        for target in self.targets:
            adapter = self._branches[target.target_id]
            values.append((f"{target.target_id}.down", adapter.down))
            values.append((f"{target.target_id}.up", adapter.up))
        return tuple(values)

    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(parameter for _, parameter in self.named_parameters())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = {name for name, _ in self.named_parameters()}
        if set(state) != expected:
            missing = sorted(expected - set(state))
            unknown = sorted(set(state) - expected)
            raise AttachmentError(
                f"LoRA checkpoint keys differ: missing={missing}, unknown={unknown}"
            )
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                value = state[name]
                if value.shape != parameter.shape or value.dtype != torch.float32:
                    raise AttachmentError(
                        f"LoRA checkpoint tensor {name!r} is {tuple(value.shape)} {value.dtype};"
                        f" expected {tuple(parameter.shape)} torch.float32"
                    )
                parameter.copy_(value.to(device=parameter.device))
        self._assert_float32()
