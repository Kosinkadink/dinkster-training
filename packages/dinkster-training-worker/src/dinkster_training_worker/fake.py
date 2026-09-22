"""A deterministic torch-free trainer over the durable session protocol.

Training state is a digest chain, so a replacement worker can reconstruct a
recovery checkpoint without model artifacts. Claims, fencing, checkpoints,
commits, cancellation, and replay all use the production session store.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping

from dinkster_api.v1 import TrainingSessionHandle, digest_bytes
from dinkster_nodes_training import AdvanceOutcome
from dinkster_server import (
    TrainingLineageConflict,
    TrainingOperationRecord,
    TrainingSessionStore,
)

FAKE_RUNTIME_IDENTITY = "fake-trainer/1"
FAKE_SNAPSHOT_DIGEST = digest_bytes(b"fake-trainer-extension-snapshot")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class TrainingAdvancePaused(Exception):
    """The advance stopped after publishing a recovery checkpoint."""


class FakeTrainer:
    """A deterministic ``TrainingService`` implementation.

    One instance models one worker process and holds one fence epoch per
    session. A replacement instance acquires a new fence and resumes from the
    latest recovery checkpoint.
    """

    def __init__(
        self,
        store: TrainingSessionStore,
        *,
        scope: str = "local",
        cancelled: Callable[[], bool] | None = None,
        first_step_delay: float = 0.0,
    ) -> None:
        if not math.isfinite(first_step_delay) or first_step_delay < 0:
            raise ValueError("first_step_delay must be finite and non-negative")
        self._store = store
        self._scope = scope
        self._cancelled = cancelled if cancelled is not None else lambda: False
        self._first_step_delay = first_step_delay
        self._fences: dict[str, int] = {}
        self.steps_run = 0
        """Optimizer steps this instance executed rather than replayed."""

    @staticmethod
    def _config_digest(config: str) -> str:
        canonical = _canonical({"config": config, "trainer": FAKE_RUNTIME_IDENTITY})
        return digest_bytes(canonical.encode("ascii"))

    @staticmethod
    def _session_id(session_key: str) -> str:
        return digest_bytes(("session:" + session_key).encode("utf-8")).removeprefix("blake3:")

    @staticmethod
    def _chain(config_digest: str, step: int) -> str:
        digest = hashlib.sha256(config_digest.encode("ascii")).hexdigest()
        for number in range(1, step + 1):
            digest = hashlib.sha256((digest + f":{number}").encode("ascii")).hexdigest()
        return digest

    @classmethod
    def _manifest_digest(cls, session_id: str, config_digest: str, step: int) -> str:
        chain = cls._chain(config_digest, step)
        raw = f"manifest:{session_id}:{step}:{chain}"
        return digest_bytes(raw.encode("ascii"))

    @classmethod
    def _loss(cls, config_digest: str, step: int) -> float:
        return int(cls._chain(config_digest, step)[:8], 16) / 2**32

    def _step_for_manifest(self, session_id: str, config_digest: str, digest: str) -> int:
        for step in range(100_000):
            if self._manifest_digest(session_id, config_digest, step) == digest:
                return step
        raise TrainingLineageConflict(f"manifest {digest!r} is not on this session's fake lineage")

    def _fence(self, session_id: str) -> int:
        epoch = self._fences.get(session_id)
        if epoch is None:
            epoch = self._store.acquire_fence(session_id).fence_epoch
            self._fences[session_id] = epoch
        return epoch

    def _report(self, config: str) -> dict[str, object]:
        config_digest = self._config_digest(config)
        return {
            "trainer": FAKE_RUNTIME_IDENTITY,
            "configDigest": config_digest,
            "sessionExtensionSnapshotDigest": FAKE_SNAPSHOT_DIGEST,
            "capabilities": {
                "autograd": False,
                "families": [],
                "checkpointResume": True,
                "safePointCancellation": True,
            },
        }

    def dry_run(self, config: str) -> Mapping[str, object]:
        return self._report(config)

    def create(
        self, session_key: str, config: str
    ) -> tuple[TrainingSessionHandle, Mapping[str, object]]:
        session_id = self._session_id(session_key)
        config_digest = self._config_digest(config)
        self._store.create_session(
            session_id,
            scope=self._scope,
            config_digest=config_digest,
            extension_snapshot_digest=FAKE_SNAPSHOT_DIGEST,
            initial_manifest_digest=self._manifest_digest(session_id, config_digest, 0),
        )
        initial = self._store.list_checkpoints(session_id)[0]
        handle = TrainingSessionHandle(
            session_id=session_id,
            checkpoint_manifest_digest=initial.manifest_digest,
            step_cursor=0,
            config_digest=config_digest,
            session_extension_snapshot_digest=FAKE_SNAPSHOT_DIGEST,
            journal_seq=initial.journal_seq,
        )
        return handle, self._report(config)

    def advance(self, handle: TrainingSessionHandle, steps: int, note: str) -> AdvanceOutcome:
        if steps < 1:
            raise ValueError("an advance must request at least one optimizer step")
        session_id = handle.session_id
        session = self._store.get_session(session_id)
        if session is None:
            raise TrainingLineageConflict(f"unknown training session {session_id!r}")
        if (
            session.config_digest != handle.config_digest
            or session.extension_snapshot_digest != handle.session_extension_snapshot_digest
        ):
            raise TrainingLineageConflict(
                "handle pins do not match the session's config/extension identity"
            )
        config_digest = handle.config_digest
        input_step = handle.step_cursor
        if self._manifest_digest(session_id, config_digest, input_step) != (
            handle.checkpoint_manifest_digest
        ):
            raise TrainingLineageConflict(
                "handle checkpoint digest does not name the handle's step cursor"
            )
        ledger = {c.manifest_digest: c for c in self._store.list_checkpoints(session_id)}
        checkpoint = ledger.get(handle.checkpoint_manifest_digest)
        if (
            checkpoint is None
            or checkpoint.step_cursor != input_step
            or checkpoint.journal_seq != handle.journal_seq
        ):
            raise TrainingLineageConflict(
                "handle does not match a committed checkpoint of this session"
            )
        operation_id = digest_bytes(
            _canonical(
                [
                    "dinkster.training.advance.v1",
                    session_id,
                    handle.checkpoint_manifest_digest,
                    config_digest,
                    handle.session_extension_snapshot_digest,
                    steps,
                    note,
                    FAKE_RUNTIME_IDENTITY,
                ]
            ).encode("ascii")
        ).removeprefix("blake3:")
        fence = self._fence(session_id)
        record = self._store.claim_advance(
            session_id,
            operation_id,
            input_manifest_digest=handle.checkpoint_manifest_digest,
            fence_epoch=fence,
        )
        if record.status == "committed":
            return self._committed_outcome(handle, record)
        resume_step = input_step
        recovery_published = bool(record.recovery_checkpoint_digest)
        if record.recovery_checkpoint_digest:
            resume_step = self._step_for_manifest(
                session_id, config_digest, record.recovery_checkpoint_digest
            )
        target = input_step + steps
        current = resume_step

        def pause_if_cancelled() -> None:
            if not self._cancelled():
                return
            if not recovery_published:
                self._store.set_recovery_checkpoint(
                    session_id,
                    operation_id,
                    fence_epoch=fence,
                    manifest_digest=self._manifest_digest(session_id, config_digest, current),
                )
            self._store.pause_advance(
                session_id,
                operation_id,
                fence_epoch=fence,
                reason="cancel requested",
            )
            raise TrainingAdvancePaused(
                f"advance paused at safe point (step {current} of {target})"
            )

        while current < target:
            pause_if_cancelled()
            self._chain(config_digest, current + 1)
            self.steps_run += 1
            current += 1
            self._store.set_recovery_checkpoint(
                session_id,
                operation_id,
                fence_epoch=fence,
                manifest_digest=self._manifest_digest(session_id, config_digest, current),
            )
            recovery_published = True
            pause_if_cancelled()
            if current == input_step + 1 and self._first_step_delay:
                time.sleep(self._first_step_delay)
        output_digest = self._manifest_digest(session_id, config_digest, target)
        covered = self._store.read_events(session_id, after=0, limit=1).latest_seq
        committed = self._store.commit_advance(
            session_id,
            operation_id,
            fence_epoch=fence,
            output_manifest_digest=output_digest,
            output_step_cursor=target,
            covered_journal_seq=covered,
        )
        return self._committed_outcome(handle, committed, replayed=False)

    def _committed_outcome(
        self,
        handle: TrainingSessionHandle,
        record: TrainingOperationRecord,
        *,
        replayed: bool = True,
    ) -> AdvanceOutcome:
        output = TrainingSessionHandle(
            session_id=handle.session_id,
            checkpoint_manifest_digest=record.output_manifest_digest,
            step_cursor=record.output_step_cursor,
            config_digest=handle.config_digest,
            session_extension_snapshot_digest=handle.session_extension_snapshot_digest,
            journal_seq=record.output_journal_seq,
        )
        return AdvanceOutcome(
            handle=output,
            loss=self._loss(handle.config_digest, record.output_step_cursor),
            replayed=replayed,
        )

    def export_lora(self, handle: TrainingSessionHandle, settings: str) -> tuple[str, str]:
        del handle, settings
        raise RuntimeError("the fake training backend cannot export a LoRA")

    def complete(self, handle: TrainingSessionHandle) -> TrainingSessionHandle:
        session_id = handle.session_id
        session = self._store.get_session(session_id)
        if session is None:
            raise TrainingLineageConflict(f"unknown training session {session_id!r}")
        if session.handle() != handle:
            raise TrainingLineageConflict(
                "handle does not name the session's committed head; complete from the head"
            )
        record = self._store.complete_session(session_id, fence_epoch=self._fence(session_id))
        return record.handle()
