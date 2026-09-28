"""SD1.5 LoRA training through dinkster_inference."""

import torch as _torch
from dinkster_inference.cli_args import args as _comfy_args

# dinkster_inference selects its device while model_management imports. Training
# owns this process, so select its CPU fallback before importing model modules.
if not _torch.cuda.is_available():
    _comfy_args.cpu = True

from .checkpoint import CheckpointError, CheckpointState, ContentAddressedCheckpointStore
from .config import DatasetConfig, TrainingConfig, TrainingConfigError
from .dataset import EncodedDataset, PreparedBatch
from .lora_program import LoRAProgram, LoRAProgramError, Target, resolve_targets
from .service import (
    TRAINING_RUNTIME_IDENTITY,
    TRAINING_SNAPSHOT_DIGEST,
    SD15LoRATrainingService,
    TrainingAdvancePaused,
)
from .trainer import SD15LoRATrainer

__all__ = [
    "TRAINING_RUNTIME_IDENTITY",
    "TRAINING_SNAPSHOT_DIGEST",
    "CheckpointError",
    "CheckpointState",
    "ContentAddressedCheckpointStore",
    "DatasetConfig",
    "EncodedDataset",
    "LoRAProgram",
    "LoRAProgramError",
    "PreparedBatch",
    "SD15LoRATrainer",
    "SD15LoRATrainingService",
    "Target",
    "TrainingAdvancePaused",
    "TrainingConfig",
    "TrainingConfigError",
    "resolve_targets",
]
