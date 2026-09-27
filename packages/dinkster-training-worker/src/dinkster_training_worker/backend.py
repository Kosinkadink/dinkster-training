"""Trainer backend selection for the isolated training executor."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from dinkster_nodes_training import TrainingService
from dinkster_server import TrainingSessionStore
from dinkster_workers import current_execution_context

from .fake import FakeTrainer


def _invocation_cancelled() -> bool:
    context = current_execution_context()
    return context is not None and context.cancelled()


def create_training_service(
    name: str,
    store: TrainingSessionStore,
    *,
    environment: Mapping[str, str] | None = None,
) -> TrainingService:
    """Create the explicitly selected trainer backend."""
    if name == "fake":
        raw_delay = (environment or {}).get("DINKSTER_TRAINING_FAKE_FIRST_STEP_DELAY", "0")
        try:
            first_step_delay = float(raw_delay)
        except ValueError as exc:
            raise ValueError("DINKSTER_TRAINING_FAKE_FIRST_STEP_DELAY must be a number") from exc
        return FakeTrainer(
            store,
            cancelled=_invocation_cancelled,
            first_step_delay=first_step_delay,
        )
    if name == "sd15-lora":
        checkpoint_root = (environment or {}).get("DINKSTER_TRAINING_CHECKPOINT_ROOT")
        if not checkpoint_root:
            raise ValueError(f"DINKSTER_TRAINING_CHECKPOINT_ROOT must be set for {name}")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from dinkster_training_torch import SD15LoRATrainingService

        return SD15LoRATrainingService(
            store,
            Path(checkpoint_root),
            cancelled=_invocation_cancelled,
            expected_device=(environment or {}).get("DINKSTER_TRAINING_DEVICE", "cuda:0"),
        )
    raise ValueError(f"unknown training backend {name!r}; available backends: fake, sd15-lora")
