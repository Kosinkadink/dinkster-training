"""Process-group bootstrap and rank context for distributed training."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    import torch

DistributedBackend = Literal["gloo", "nccl"]
PROCESS_GROUP_TIMEOUT = timedelta(seconds=120)


@dataclass(frozen=True)
class TcpRendezvousSettings:
    """TCP process-group rendezvous settings."""

    host: str
    port: int

    @property
    def method(self) -> Literal["tcp"]:
        return "tcp"

    def to_mapping(self) -> dict[str, object]:
        return {"method": self.method, "host": self.host, "port": self.port}

    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"tcp://{host}:{self.port}"


@dataclass(frozen=True)
class FileRendezvousSettings:
    """Filesystem process-group rendezvous settings."""

    path: str

    @property
    def method(self) -> Literal["file"]:
        return "file"

    def to_mapping(self) -> dict[str, object]:
        return {"method": self.method, "path": self.path}

    def url(self) -> str:
        return Path(self.path).expanduser().resolve().as_uri()


RendezvousSettings = TcpRendezvousSettings | FileRendezvousSettings


@dataclass(frozen=True)
class DistributedSettings:
    """Validated process-group settings from a training config."""

    world_size: int
    backend: DistributedBackend
    rendezvous: RendezvousSettings

    def to_mapping(self) -> dict[str, object]:
        return {
            "worldSize": self.world_size,
            "backend": self.backend,
            "rendezvous": self.rendezvous.to_mapping(),
        }


class _DistributedModule(Protocol):
    def init_process_group(
        self,
        backend: str,
        *,
        init_method: str,
        rank: int,
        world_size: int,
        timeout: timedelta,
    ) -> object: ...

    def is_initialized(self) -> bool: ...

    def get_world_size(self) -> int: ...

    def get_rank(self) -> int: ...

    def all_reduce(self, tensor: torch.Tensor) -> object: ...

    def broadcast(self, tensor: torch.Tensor, src: int) -> object: ...

    def barrier(self) -> object: ...

    def destroy_process_group(self) -> None: ...


class TorchDistributedRankContext(AbstractContextManager["TorchDistributedRankContext"]):
    """Rank collectives backed by one initialized torch process group."""

    def __init__(self, distributed: _DistributedModule) -> None:
        self._distributed = distributed
        self._closed = False

    @property
    def world_size(self) -> int:
        return self._distributed.get_world_size()

    @property
    def rank(self) -> int:
        return self._distributed.get_rank()

    def all_reduce_gradients(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        import torch

        world_size = self.world_size
        gradients = tuple(parameter.grad for parameter in parameters if parameter.grad is not None)
        if not gradients:
            return
        # Flattening into one buffer requires a single dtype and device. Adapter
        # masters are enforced float32 on the one training device; a mixed dtype
        # would make cat promote silently and change the reduction numerics.
        flat = torch.cat(tuple(gradient.reshape(-1) for gradient in gradients))
        self._distributed.all_reduce(flat)
        flat.div_(world_size)
        offset = 0
        for gradient in gradients:
            end = offset + gradient.numel()
            gradient.copy_(flat[offset:end].view_as(gradient))
            offset = end

    def broadcast_parameters(self, parameters: tuple[torch.nn.Parameter, ...]) -> None:
        for parameter in parameters:
            self._distributed.broadcast(parameter.detach(), src=0)

    def barrier(self) -> None:
        self._distributed.barrier()

    def close(self) -> None:
        if self._closed:
            return
        if self._distributed.is_initialized():
            self._distributed.destroy_process_group()
        self._closed = True

    def abort(self) -> None:
        """Abort the default NCCL process group without peer participation."""
        if self._closed:
            return
        if self._distributed.is_initialized():
            from torch.distributed import distributed_c10d

            distributed_c10d._abort_process_group()  # pyright: ignore[reportPrivateUsage]
        self._closed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()


def bootstrap_rank_context(
    settings: DistributedSettings,
    rank: int,
    *,
    timeout: timedelta = PROCESS_GROUP_TIMEOUT,
) -> TorchDistributedRankContext:
    """Initialize a process group and return its rank context."""
    if rank < 0 or rank >= settings.world_size:
        raise ValueError(f"rank must be in [0, {settings.world_size}), got {rank}")

    import torch.distributed as torch_distributed

    distributed = cast("_DistributedModule", torch_distributed)
    if distributed.is_initialized():
        raise RuntimeError("a torch distributed process group is already initialized")
    try:
        distributed.init_process_group(
            settings.backend,
            init_method=settings.rendezvous.url(),
            rank=rank,
            world_size=settings.world_size,
            timeout=timeout,
        )
    except BaseException:
        if distributed.is_initialized():
            distributed.destroy_process_group()
        raise
    return TorchDistributedRankContext(distributed)
