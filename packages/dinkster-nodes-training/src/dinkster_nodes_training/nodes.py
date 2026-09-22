"""Training session nodes: the graph-boundary adapters.

The only loop-carried value is the ``training.session_handle`` state port
(training-design.md 3.2): an RPC-clean record naming a committed checkpoint,
never a tensor, module, optimizer, or live object. A fold or while region
feeds the handle through ``training.advance`` and receives the next handle;
one region iteration is one checkpoint interval.

Every effectful node declares ``idempotent=False``: the engine never caches
it, and the durable operation ledger behind the training service - not the
engine cache - decides whether an optimizer step happened. A retried advance
with the same operation identity replays its committed outcome instead of
stepping again.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from dinkster_api.v1 import (
    CORE_FLOAT,
    CORE_INT,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TrainingSessionHandle,
    TypeExpr,
    TypeRegistry,
)

from .service import current_training_service

TRAINING_SESSION_HANDLE = "training.session_handle"
HANDLE = TypeExpr.concrete(TRAINING_SESSION_HANDLE)
INT = TypeExpr.concrete(CORE_INT)
FLOAT = TypeExpr.concrete(CORE_FLOAT)
STRING = TypeExpr.concrete(CORE_STRING)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _encode_handle(value: object) -> bytes:
    assert isinstance(value, TrainingSessionHandle)
    return _canonical_json(value.to_wire()).encode("ascii")


def _decode_handle(data: bytes) -> object:
    return TrainingSessionHandle.from_wire(json.loads(data.decode("ascii")))


def _fingerprint_handle(value: object) -> str:
    assert isinstance(value, TrainingSessionHandle)
    return value.fingerprint()


def _meta_handle(value: object) -> Mapping[str, object]:
    assert isinstance(value, TrainingSessionHandle)
    return {"sessionId": value.session_id, "stepCursor": value.step_cursor}


def _coerce_handle(value: object) -> object:
    """Graph literals arrive as the wire mapping; runtime handles pass
    through. Anything else fails from_wire's fail-closed decode."""
    if isinstance(value, TrainingSessionHandle):
        return value
    return TrainingSessionHandle.from_wire(value)


def register_training_types(registry: TypeRegistry) -> None:
    """The pack's one type entry point: the session handle value."""
    registry.register(
        TRAINING_SESSION_HANDLE,
        encode=_encode_handle,
        decode=_decode_handle,
        fingerprint=_fingerprint_handle,
        meta=_meta_handle,
        coerce=_coerce_handle,
    )


class TrainingDryRun(Node):
    """Validates a config and reports trainer capabilities without creating
    a session. Non-idempotent because the report reflects the host-bound
    service, which is outside cache identity."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="training.dry_run",
            display_name="Training Dry Run",
            category="training",
            inputs=(InputSpec("config", STRING),),
            outputs=(OutputSpec("report", STRING),),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, config: str) -> Mapping[str, object]:
        report = current_training_service().dry_run(config)
        return cls.outputs(report=_canonical_json(report))


class CreateTrainingSession(Node):
    """Creates (or idempotently rejoins) a training session and returns its
    initial step-0 handle. ``session_key`` is the caller's durable
    idempotency key: re-running the same graph rejoins the same session,
    and the same key with a different config fails loudly. The handle is
    the initial one even on rejoin so downstream advances keep their first
    run's operation identities and replay committed outcomes."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="training.create_session",
            display_name="Create Training Session",
            category="training",
            inputs=(InputSpec("session_key", STRING), InputSpec("config", STRING)),
            outputs=(OutputSpec("handle", HANDLE), OutputSpec("report", STRING)),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, session_key: str, config: str) -> Mapping[str, object]:
        handle, report = current_training_service().create(session_key, config)
        return cls.outputs(handle=handle, report=_canonical_json(report))


class AdvanceTraining(Node):
    """One effectful, retry-safe checkpoint interval (training-design.md
    3.2). ``note`` is a per-advance policy input: it enters the operation
    identity, so two advances from the same checkpoint with different notes
    are different operations."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="training.advance",
            display_name="Advance Training",
            category="training",
            inputs=(
                InputSpec("handle", HANDLE),
                InputSpec("steps", INT, default=1),
                InputSpec("note", STRING, default=""),
            ),
            outputs=(
                OutputSpec("handle", HANDLE),
                OutputSpec("step_cursor", INT),
                OutputSpec("loss", FLOAT),
            ),
            idempotent=False,
        )

    @classmethod
    def execute(
        cls, *, handle: TrainingSessionHandle, steps: int, note: str
    ) -> Mapping[str, object]:
        if steps < 1:
            raise ValueError("an advance must request at least one optimizer step")
        outcome = current_training_service().advance(handle, steps, note)
        return cls.outputs(
            handle=outcome.handle,
            step_cursor=outcome.handle.step_cursor,
            loss=outcome.loss,
        )


class CompleteTrainingSession(Node):
    """Marks the session terminal; further advances are refused."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="training.complete_session",
            display_name="Complete Training Session",
            category="training",
            inputs=(InputSpec("handle", HANDLE),),
            outputs=(OutputSpec("session_id", STRING), OutputSpec("step_cursor", INT)),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, handle: TrainingSessionHandle) -> Mapping[str, object]:
        final = current_training_service().complete(handle)
        return cls.outputs(session_id=final.session_id, step_cursor=final.step_cursor)


class ExportTrainingLora(Node):
    """Exports a committed checkpoint as a kohya-layout LoRA file."""

    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="training.export_lora",
            display_name="Export Training LoRA",
            category="training",
            inputs=(InputSpec("handle", HANDLE), InputSpec("settings", STRING)),
            outputs=(OutputSpec("path", STRING), OutputSpec("digest", STRING)),
            idempotent=False,
        )

    @classmethod
    def execute(cls, *, handle: TrainingSessionHandle, settings: str) -> Mapping[str, object]:
        path, digest = current_training_service().export_lora(handle, settings)
        return cls.outputs(path=path, digest=digest)


PACK_NODES: tuple[type[Node], ...] = (
    TrainingDryRun,
    CreateTrainingSession,
    AdvanceTraining,
    CompleteTrainingSession,
    ExportTrainingLora,
)
