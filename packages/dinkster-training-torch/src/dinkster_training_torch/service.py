"""Durable isolated SD1.5 LoRA service over the Dinkster training ledger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import torch
from dinkster_api.v1 import TrainingSessionHandle, digest_bytes
from dinkster_nodes_training import AdvanceOutcome
from dinkster_server import (
    TrainingLineageConflict,
    TrainingOperationRecord,
    TrainingSessionStore,
)

from .checkpoint import CheckpointState, ContentAddressedCheckpointStore, blake3_digest
from .config import TrainingConfig
from .export import LoraExportSettings, export_lora
from .trainer import SD15LoRATrainer

TRAINING_RUNTIME_IDENTITY = "dinkster-comfy-sd15-lora/1"
TRAINING_SNAPSHOT_DIGEST = blake3_digest(b"dinkster-comfy-sd15-lora/1")


class TrainingAdvancePaused(Exception):
    """The advance stopped after publishing a complete recovery checkpoint."""


def _session_id(session_key: str) -> str:
    return digest_bytes(("session:" + session_key).encode("utf-8"))[7:]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


class SD15LoRATrainingService:
    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        cancelled: Callable[[], bool] | None = None,
        trainer_factory: Callable[[TrainingConfig], SD15LoRATrainer] = SD15LoRATrainer,
        expected_device: str = "cuda:0",
    ) -> None:
        self._store = store
        self._checkpoints = ContentAddressedCheckpointStore(checkpoint_root)
        self._export_root = checkpoint_root / "exports"
        self._scope = scope
        self._cancelled = cancelled if cancelled is not None else lambda: False
        self._trainer_factory = trainer_factory
        self._expected_device = expected_device
        self._fences: dict[str, int] = {}
        self.steps_run = 0

    @staticmethod
    def session_id(session_key: str) -> str:
        return _session_id(session_key)

    def _fence(self, session_id: str) -> int:
        epoch = self._fences.get(session_id)
        if epoch is None:
            epoch = self._store.acquire_fence(session_id).fence_epoch
            self._fences[session_id] = epoch
        return epoch

    def _parse(self, serialized: str) -> TrainingConfig:
        config = TrainingConfig.parse(serialized)
        if not config.checkpoint_path.is_file():
            raise ValueError(f"checkpoint does not exist: {config.checkpoint_path}")
        if _sha256(config.checkpoint_path) != config.checkpoint_digest:
            raise ValueError("checkpoint bytes do not match checkpointDigest")
        device = torch.device(config.device)
        if device != torch.device(self._expected_device):
            raise ValueError(
                f"config device {config.device!r} does not match the governed worker device "
                f"{self._expected_device!r}"
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA training was requested but CUDA is unavailable")
        return config

    @staticmethod
    def _report(config: TrainingConfig) -> dict[str, object]:
        return {
            "trainer": TRAINING_RUNTIME_IDENTITY,
            "configDigest": config.digest,
            "sessionExtensionSnapshotDigest": TRAINING_SNAPSHOT_DIGEST,
            "capabilities": {
                "autograd": True,
                "families": ["sd15"],
                "checkpointResume": True,
                "safePointCancellation": True,
                "dedicatedProcess": True,
            },
        }

    def dry_run(self, config: str) -> Mapping[str, object]:
        return self._report(self._parse(config))

    def _write_checkpoint(
        self,
        *,
        session_id: str,
        config: TrainingConfig,
        parent: str,
        trainer: SD15LoRATrainer,
    ) -> str:
        return self._checkpoints.write(
            session_id=session_id,
            config_digest=config.digest,
            extension_snapshot_digest=TRAINING_SNAPSHOT_DIGEST,
            parent_manifest_digest=parent,
            step_cursor=trainer.step_cursor,
            config=config.to_mapping(),
            adapter=trainer.attachment.state_dict(),
            optimizer=trainer.optimizer.state_dict(),
            rng=trainer.randomness.state_dict(),
            data_cursor=trainer.data_cursor,
            loss=trainer.last_loss,
        )

    def create(
        self, session_key: str, config: str
    ) -> tuple[TrainingSessionHandle, Mapping[str, object]]:
        normalized = self._parse(config)
        session_id = _session_id(session_key)
        existing = self._store.get_session(session_id)
        if existing is not None:
            if (
                existing.config_digest != normalized.digest
                or existing.extension_snapshot_digest != TRAINING_SNAPSHOT_DIGEST
            ):
                raise TrainingLineageConflict(
                    "session already exists with different identity facts"
                )
            initial = self._store.list_checkpoints(session_id)[0]
        else:
            trainer = self._trainer_factory(normalized)
            try:
                digest = self._write_checkpoint(
                    session_id=session_id,
                    config=normalized,
                    parent="",
                    trainer=trainer,
                )
            finally:
                trainer.close()
            self._store.create_session(
                session_id,
                scope=self._scope,
                config_digest=normalized.digest,
                extension_snapshot_digest=TRAINING_SNAPSHOT_DIGEST,
                initial_manifest_digest=digest,
            )
            initial = self._store.list_checkpoints(session_id)[0]
        return (
            TrainingSessionHandle(
                session_id=session_id,
                checkpoint_manifest_digest=initial.manifest_digest,
                step_cursor=0,
                config_digest=normalized.digest,
                session_extension_snapshot_digest=TRAINING_SNAPSHOT_DIGEST,
                journal_seq=initial.journal_seq,
            ),
            self._report(normalized),
        )

    def _verify_handle(self, handle: TrainingSessionHandle) -> CheckpointState:
        session = self._store.get_session(handle.session_id)
        if session is None:
            raise TrainingLineageConflict(f"unknown training session {handle.session_id!r}")
        if (
            session.config_digest != handle.config_digest
            or session.extension_snapshot_digest != handle.session_extension_snapshot_digest
        ):
            raise TrainingLineageConflict("handle identity does not match the session")
        rows = {row.manifest_digest: row for row in self._store.list_checkpoints(handle.session_id)}
        row = rows.get(handle.checkpoint_manifest_digest)
        if (
            row is None
            or row.step_cursor != handle.step_cursor
            or row.journal_seq != handle.journal_seq
        ):
            raise TrainingLineageConflict("handle does not match a committed checkpoint")
        state = self._checkpoints.load(handle.checkpoint_manifest_digest)
        if (
            state.session_id != handle.session_id
            or state.config_digest != handle.config_digest
            or state.extension_snapshot_digest != handle.session_extension_snapshot_digest
            or state.step_cursor != handle.step_cursor
        ):
            raise TrainingLineageConflict("checkpoint state does not match the handle")
        return state

    def _load_trainer(self, digest: str, handle: TrainingSessionHandle) -> SD15LoRATrainer:
        state = self._checkpoints.load(digest)
        if state.session_id != handle.session_id or state.config_digest != handle.config_digest:
            raise TrainingLineageConflict("recovery checkpoint identity does not match the handle")
        config = self._parse(json.dumps(state.config, separators=(",", ":")))
        if config.digest != handle.config_digest:
            raise TrainingLineageConflict("checkpoint config payload does not match its digest")
        trainer = self._trainer_factory(config)
        trainer.restore(
            adapter=state.adapter,
            optimizer=state.optimizer,
            rng=state.rng,
            step_cursor=state.step_cursor,
            data_cursor=state.data_cursor,
            loss=state.loss,
        )
        return trainer

    def _operation_id(self, handle: TrainingSessionHandle, steps: int, note: str) -> str:
        value = [
            "dinkster.training.advance.v1",
            handle.session_id,
            handle.checkpoint_manifest_digest,
            steps,
            note,
            TRAINING_RUNTIME_IDENTITY,
        ]
        return digest_bytes(json.dumps(value, separators=(",", ":")).encode("ascii"))[7:]

    def advance(self, handle: TrainingSessionHandle, steps: int, note: str) -> AdvanceOutcome:
        if steps < 1:
            raise ValueError("an advance must request at least one optimizer step")
        input_state = self._verify_handle(handle)
        operation_id = self._operation_id(handle, steps, note)
        fence = self._fence(handle.session_id)
        record = self._store.claim_advance(
            handle.session_id,
            operation_id,
            input_manifest_digest=handle.checkpoint_manifest_digest,
            fence_epoch=fence,
        )
        if record.status == "committed":
            return self._committed_outcome(handle, record)
        resume_digest = record.recovery_checkpoint_digest or handle.checkpoint_manifest_digest
        trainer = self._load_trainer(resume_digest, handle)
        config = TrainingConfig.from_mapping(input_state.config)
        target = handle.step_cursor + steps
        recovery_published = bool(record.recovery_checkpoint_digest)
        checkpointed_step = trainer.step_cursor
        try:
            if not handle.step_cursor <= trainer.step_cursor <= target:
                raise TrainingLineageConflict("recovery checkpoint falls outside the advance")

            def publish() -> None:
                nonlocal checkpointed_step, recovery_published, resume_digest
                resume_digest = self._write_checkpoint(
                    session_id=handle.session_id,
                    config=config,
                    parent=handle.checkpoint_manifest_digest,
                    trainer=trainer,
                )
                self._store.set_recovery_checkpoint(
                    handle.session_id,
                    operation_id,
                    fence_epoch=fence,
                    manifest_digest=resume_digest,
                )
                recovery_published = True
                checkpointed_step = trainer.step_cursor

            def pause_if_cancelled() -> None:
                if not self._cancelled():
                    return
                if trainer.step_cursor != checkpointed_step or not recovery_published:
                    publish()
                self._store.pause_advance(
                    handle.session_id,
                    operation_id,
                    fence_epoch=fence,
                    reason="cancel requested",
                )
                raise TrainingAdvancePaused(
                    f"advance paused at safe point (step {trainer.step_cursor} of {target})"
                )

            while trainer.step_cursor < target:
                pause_if_cancelled()
                trainer.train_step()
                self.steps_run += 1
                if (
                    trainer.step_cursor == target
                    or (trainer.step_cursor - handle.step_cursor) % config.checkpoint_interval == 0
                ):
                    publish()
                pause_if_cancelled()
            covered = self._store.read_events(handle.session_id, after=0, limit=1).latest_seq
            committed = self._store.commit_advance(
                handle.session_id,
                operation_id,
                fence_epoch=fence,
                output_manifest_digest=resume_digest,
                output_step_cursor=target,
                covered_journal_seq=covered,
            )
            return self._committed_outcome(handle, committed, replayed=False)
        finally:
            trainer.close()

    def _committed_outcome(
        self,
        handle: TrainingSessionHandle,
        record: TrainingOperationRecord,
        *,
        replayed: bool = True,
    ) -> AdvanceOutcome:
        state = self._checkpoints.load(record.output_manifest_digest)
        if (
            state.session_id != handle.session_id
            or state.config_digest != handle.config_digest
            or state.extension_snapshot_digest != handle.session_extension_snapshot_digest
            or state.parent_manifest_digest != handle.checkpoint_manifest_digest
            or state.step_cursor != record.output_step_cursor
            or state.loss is None
        ):
            raise TrainingLineageConflict("committed checkpoint does not match its ledger row")
        output = TrainingSessionHandle(
            session_id=handle.session_id,
            checkpoint_manifest_digest=record.output_manifest_digest,
            step_cursor=record.output_step_cursor,
            config_digest=handle.config_digest,
            session_extension_snapshot_digest=handle.session_extension_snapshot_digest,
            journal_seq=record.output_journal_seq,
        )
        return AdvanceOutcome(output, state.loss, replayed)

    def export_lora(self, handle: TrainingSessionHandle, settings: str) -> tuple[str, str]:
        state = self._verify_handle(handle)
        export_settings = LoraExportSettings.parse(settings, export_root=self._export_root)
        path, digest = export_lora(
            state,
            export_settings,
            runtime_identity=TRAINING_RUNTIME_IDENTITY,
        )
        return str(path), digest

    def complete(self, handle: TrainingSessionHandle) -> TrainingSessionHandle:
        self._verify_handle(handle)
        session = self._store.get_session(handle.session_id)
        assert session is not None
        if session.handle() != handle:
            raise TrainingLineageConflict("complete requires the session's committed head")
        return self._store.complete_session(
            handle.session_id, fence_epoch=self._fence(handle.session_id)
        ).handle()
