"""dinkster-nodes-training: first-party training node pack.

Training sessions are graph-orchestrated and worker-executed
(training-design.md 3.1): these nodes are thin adapters over the host-bound
``TrainingService``, and the durable session supervisor plus a dedicated
training worker process own all trainer state. The pack contributes the
``training.session_handle`` value type - the only loop-carried value a
fold/while region ever sees - and the create/dry-run/advance/complete/export
node family.
"""

from .nodes import (
    HANDLE,
    PACK_NODES,
    TRAINING_SESSION_HANDLE,
    AdvanceTraining,
    CompleteTrainingSession,
    CreateTrainingSession,
    ExportTrainingLora,
    TrainingDryRun,
    register_training_types,
)
from .service import (
    AdvanceOutcome,
    TrainingService,
    TrainingServiceUnbound,
    bind_training_service,
)

__all__ = [
    "HANDLE",
    "PACK_NODES",
    "TRAINING_SESSION_HANDLE",
    "AdvanceOutcome",
    "AdvanceTraining",
    "CompleteTrainingSession",
    "CreateTrainingSession",
    "ExportTrainingLora",
    "TrainingDryRun",
    "TrainingService",
    "TrainingServiceUnbound",
    "bind_training_service",
    "register_training_types",
]
