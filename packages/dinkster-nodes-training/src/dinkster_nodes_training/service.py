"""The seam between training graph nodes and the host's training runtime.

Nodes in this pack are thin adapters (training-design.md 3.1): they never
own a microbatch loop, tensor state, or a live autograd graph. Everything
effectful goes through the ``TrainingService`` the host binds - the durable
claim/commit ledger, checkpointing, and safe-point cancellation live behind
it, so a node invocation can crash or be retried at any point without
double-stepping an optimizer trajectory.

Cancellation is deliberately absent from this interface: the host delivers
safe-point cancellation to the trainer it runs (training-design.md 9.3);
the node adapter never polls it.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

from dinkster_api.v1 import TrainingSessionHandle


@dataclass(frozen=True)
class AdvanceOutcome:
    """One advance's result at the graph boundary.

    ``replayed`` is True when the outcome was served from the committed
    operation record without running any step - the retry-safety contract,
    surfaced so callers and tests can distinguish work from replay.
    """

    handle: TrainingSessionHandle
    loss: float
    replayed: bool = False


class TrainingService(Protocol):
    """What the host must bind for training nodes to execute.

    Structural: implementations live host-side and are matched by shape,
    never by importing this pack.
    """

    def dry_run(self, config: str) -> Mapping[str, object]:
        """Validate ``config`` and report capabilities without creating
        anything durable."""
        ...

    def create(
        self, session_key: str, config: str
    ) -> tuple[TrainingSessionHandle, Mapping[str, object]]:
        """Create (or idempotently rejoin) the session for ``session_key``
        and return its initial step-0 handle plus the capability report.
        Always the initial handle, even on rejoin: a re-run graph must feed
        each advance the same input handle as the first run so operation
        identities match and committed outcomes replay. The same key with
        a different config is a loud lineage conflict."""
        ...

    def advance(self, handle: TrainingSessionHandle, steps: int, note: str) -> AdvanceOutcome:
        """Run one retry-safe checkpoint interval of ``steps`` optimizer
        steps from ``handle`` and return the next handle."""
        ...

    def export_lora(self, handle: TrainingSessionHandle, settings: str) -> tuple[str, str]:
        """Export the checkpoint named by ``handle`` using serialized settings."""
        ...

    def complete(self, handle: TrainingSessionHandle) -> TrainingSessionHandle:
        """Mark the session terminal and return its final handle."""
        ...


_training_service: ContextVar[TrainingService | None] = ContextVar(
    "dinkster_training_service", default=None
)


class TrainingServiceUnbound(RuntimeError):
    """A training node executed with no host-bound training service."""


def current_training_service() -> TrainingService:
    service = _training_service.get()
    if service is None:
        raise TrainingServiceUnbound(
            "no training service is bound; the host must wrap execution in"
            " bind_training_service(...)"
        )
    return service


class _TrainingServiceBinding(AbstractContextManager[None]):
    def __init__(self, service: TrainingService) -> None:
        self._service = service
        self._token: Token[TrainingService | None] | None = None

    def __enter__(self) -> None:
        if self._token is not None:
            raise RuntimeError("training service binding is already active")
        self._token = _training_service.set(self._service)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._token is None:
            raise RuntimeError("training service binding is not active")
        _training_service.reset(self._token)
        self._token = None


def bind_training_service(service: TrainingService) -> AbstractContextManager[None]:
    return _TrainingServiceBinding(service)
