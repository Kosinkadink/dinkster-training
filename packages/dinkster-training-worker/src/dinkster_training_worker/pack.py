"""Isolated pack entry that binds the selected training service."""

from __future__ import annotations

import atexit
import os
from pathlib import Path

from dinkster_nodes_training import (
    PACK_NODES,
    AdvanceTraining,
    CompleteTrainingSession,
    CreateTrainingSession,
    ExportTrainingLora,
    TrainingDryRun,
    bind_training_service,
    register_training_types,
)
from dinkster_server import JournalStore, TrainingSessionStore

from .backend import create_training_service


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set for the training worker")
    return value


_store = TrainingSessionStore(
    JournalStore(Path(_required_environment("DINKSTER_TRAINING_JOURNAL")))
)
_service = create_training_service(
    _required_environment("DINKSTER_TRAINING_BACKEND"),
    _store,
    environment=os.environ,
)
_service_binding = bind_training_service(_service)
_service_binding.__enter__()
atexit.register(_store.close)

__all__ = [
    "PACK_NODES",
    "AdvanceTraining",
    "CompleteTrainingSession",
    "CreateTrainingSession",
    "ExportTrainingLora",
    "TrainingDryRun",
    "register_training_types",
]
