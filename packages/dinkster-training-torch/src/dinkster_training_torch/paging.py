"""Pinned-host paging for frozen MiniMax H3 transformer layers."""

from __future__ import annotations

import math
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Literal, Self

import torch
from torch.utils.checkpoint import checkpoint


def paged_layer_indices(layer_count: int, fraction: float) -> tuple[int, ...]:
    """Select a deterministic tail fraction of transformer layers."""
    if layer_count < 1:
        raise ValueError("layer_count must be positive")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("paging fraction must be finite and in (0, 1]")
    count = min(layer_count, math.ceil(layer_count * fraction))
    return tuple(range(layer_count - count, layer_count))


@dataclass(frozen=True)
class _TensorBinding:
    owner: torch.nn.Module
    name: str
    parameter: bool
    host: torch.Tensor

    def install(self, value: torch.Tensor) -> None:
        if self.parameter:
            setattr(self.owner, self.name, torch.nn.Parameter(value, requires_grad=False))
        else:
            setattr(self.owner, self.name, value)


@dataclass
class _LayerState:
    index: int
    bindings: tuple[_TensorBinding, ...]
    device_tensors: tuple[torch.Tensor, ...] = ()
    ready: torch.cuda.Event | None = None


class _CheckpointedLayer(torch.nn.Module):
    def __init__(
        self,
        layer: torch.nn.Module,
        *,
        pager: FrozenLayerPager | None,
        index: int,
    ) -> None:
        super().__init__()
        self.layer = layer
        self._pager = pager
        self._index = index

    def _apply(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        recurse: bool = True,
    ) -> Self:
        if self._pager is not None:
            return self
        return super()._apply(fn, recurse=recurse)

    def forward(self, *args: object, **kwargs: object) -> object:
        def evaluate(*values: object) -> object:
            return self.layer(*values, **kwargs)

        pager = self._pager
        if pager is None:
            return checkpoint(
                evaluate,
                *args,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return checkpoint(
            evaluate,
            *args,
            use_reentrant=False,
            preserve_rng_state=False,
            context_fn=lambda: pager.checkpoint_contexts(self._index),
        )


class FrozenLayerPager:
    """Stream frozen layers from immutable pinned CPU storage around each use."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        fraction: float,
        device: torch.device,
        checkpoint_all_layers: bool,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("host layer paging requires a CUDA device")
        blocks = getattr(model, "blocks", None)
        if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
            raise ValueError("MiniMax H3 host layer paging requires a non-empty model.blocks")
        self.device = device
        self.layer_count = len(blocks)
        self.paged_indices = paged_layer_indices(self.layer_count, fraction)
        self._states: dict[int, _LayerState] = {}
        self._module_indices: dict[torch.nn.Module, int] = {}
        self._phase_stack: list[tuple[int, Literal["forward", "recompute"]]] = []
        self._completed_events: list[torch.cuda.Event] = []
        self._transfer_stream = torch.cuda.Stream(device=device)

        paged = frozenset(self.paged_indices)
        original_layers = tuple(blocks)
        for index in self.paged_indices:
            layer = original_layers[index]
            bindings = self._pin_layer(layer)
            self._states[index] = _LayerState(index, bindings)
            self._module_indices[layer] = index
            layer.register_forward_pre_hook(self._before_layer)
            layer.register_forward_hook(self._after_layer, always_call=True)
            layer.register_full_backward_hook(self._after_backward)
        for index, layer in enumerate(original_layers):
            if index in paged or checkpoint_all_layers:
                blocks[index] = _CheckpointedLayer(
                    layer,
                    pager=self if index in paged else None,
                    index=index,
                )

    @staticmethod
    def _pin_layer(layer: torch.nn.Module) -> tuple[_TensorBinding, ...]:
        bindings: list[_TensorBinding] = []
        for owner in layer.modules():
            for name, value in owner.named_parameters(recurse=False):
                if value.requires_grad:
                    raise ValueError("host layer paging accepts only frozen base parameters")
                if value.device.type != "cpu":
                    raise ValueError("paged MiniMax H3 layers must begin on CPU")
                host = value.detach().clone().pin_memory()
                binding = _TensorBinding(owner, name, True, host)
                binding.install(host)
                bindings.append(binding)
            for name, value in owner.named_buffers(recurse=False):
                if value.device.type != "cpu":
                    raise ValueError("paged MiniMax H3 layer buffers must begin on CPU")
                host = value.detach().clone().pin_memory()
                binding = _TensorBinding(owner, name, False, host)
                binding.install(host)
                bindings.append(binding)
        if not bindings:
            raise ValueError("a paged MiniMax H3 layer must own frozen tensors")
        return tuple(bindings)

    @property
    def host_resident_layer_count(self) -> int:
        return sum(not state.device_tensors for state in self._states.values())

    @property
    def device_resident_layer_count(self) -> int:
        return len(self._states) - self.host_resident_layer_count

    @property
    def pinned_host_bytes(self) -> int:
        return sum(
            binding.host.numel() * binding.host.element_size()
            for state in self._states.values()
            for binding in state.bindings
        )

    def checkpoint_contexts(
        self, index: int
    ) -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
        return self._phase(index, "forward"), self._phase(index, "recompute")

    @contextmanager
    def _phase(
        self, index: int, phase: Literal["forward", "recompute"]
    ) -> Generator[None, None, None]:
        self._phase_stack.append((index, phase))
        try:
            yield
        finally:
            actual = self._phase_stack.pop()
            if actual != (index, phase):
                raise RuntimeError("MiniMax H3 paging checkpoint contexts became unbalanced")

    def _active_phase(self, index: int) -> Literal["forward", "recompute"]:
        if not self._phase_stack or self._phase_stack[-1][0] != index:
            raise RuntimeError("paged MiniMax H3 layer executed outside its checkpoint context")
        return self._phase_stack[-1][1]

    def _before_layer(self, module: torch.nn.Module, args: tuple[object, ...]) -> None:
        del args
        index = self._index_for(module)
        phase = self._active_phase(index)
        self._activate(index)
        self._prefetch_neighbor(index, 1 if phase == "forward" else -1)

    def _after_layer(
        self,
        module: torch.nn.Module,
        args: tuple[object, ...],
        output: object,
    ) -> None:
        del args, output
        index = self._index_for(module)
        if self._active_phase(index) == "forward":
            self._retire(index)

    def _after_backward(
        self,
        module: torch.nn.Module,
        grad_input: object,
        grad_output: object,
    ) -> None:
        del grad_input, grad_output
        self._retire(self._index_for(module))

    def _index_for(self, module: torch.nn.Module) -> int:
        try:
            return self._module_indices[module]
        except KeyError as exc:
            raise RuntimeError("unregistered MiniMax H3 layer reached the pager") from exc

    def _activate(self, index: int) -> None:
        self._reap_events()
        state = self._states[index]
        if not state.device_tensors:
            self._stage(state)
        assert state.ready is not None
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(state.ready)
        for tensor in state.device_tensors:
            tensor.record_stream(stream)
        state.ready = None

    def _stage(self, state: _LayerState) -> None:
        device_tensors: list[torch.Tensor] = []
        with torch.cuda.stream(self._transfer_stream):
            for binding in state.bindings:
                value = binding.host.to(device=self.device, non_blocking=True)
                value.record_stream(self._transfer_stream)
                binding.install(value)
                device_tensors.append(value)
            ready = torch.cuda.Event()
            ready.record(self._transfer_stream)
        state.device_tensors = tuple(device_tensors)
        state.ready = ready

    def _prefetch_neighbor(self, index: int, direction: int) -> None:
        position = self.paged_indices.index(index) + direction
        if 0 <= position < len(self.paged_indices):
            state = self._states[self.paged_indices[position]]
            if not state.device_tensors:
                self._stage(state)

    def _retire(self, index: int) -> None:
        state = self._states[index]
        if not state.device_tensors:
            return
        stream = torch.cuda.current_stream(self.device)
        completed = torch.cuda.Event()
        completed.record(stream)
        for tensor in state.device_tensors:
            tensor.record_stream(stream)
        for binding in state.bindings:
            binding.install(binding.host)
        state.device_tensors = ()
        state.ready = None
        self._completed_events.append(completed)

    def _reap_events(self) -> None:
        self._completed_events = [event for event in self._completed_events if not event.query()]

    def assert_idle(self) -> None:
        if self.device_resident_layer_count:
            raise RuntimeError("paged MiniMax H3 layers remained device-resident after backward")
