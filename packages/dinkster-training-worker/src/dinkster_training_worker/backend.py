"""Trainer backend selection for the isolated training executor."""

from __future__ import annotations

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
    if name in (
        "sd15-lora",
        "sdxl-lora",
        "flux-lora",
        "flux2-lora",
        "ideogram4-lora",
        "minimax-h3-lora",
        "minimax-music3-lora",
        "qwen-image-lora",
        "wan-lora",
    ):
        checkpoint_root = (environment or {}).get("DINKSTER_TRAINING_CHECKPOINT_ROOT")
        if not checkpoint_root:
            raise ValueError(f"DINKSTER_TRAINING_CHECKPOINT_ROOT must be set for {name}")
        if name == "sd15-lora":
            from dinkster_training_torch import SD15LoRATrainingService

            service_type = SD15LoRATrainingService
        elif name == "sdxl-lora":
            from dinkster_training_torch import SDXLLoRATrainingService

            service_type = SDXLLoRATrainingService
        elif name == "wan-lora":
            from dinkster_training_torch import WanLoRATrainingService

            service_type = WanLoRATrainingService
        elif name == "flux-lora":
            from dinkster_training_torch import FluxLoRATrainingService

            service_type = FluxLoRATrainingService
        elif name == "flux2-lora":
            from dinkster_training_torch import Flux2LoRATrainingService

            service_type = Flux2LoRATrainingService
        elif name == "qwen-image-lora":
            from dinkster_training_torch import QwenImageLoRATrainingService

            service_type = QwenImageLoRATrainingService
        elif name == "ideogram4-lora":
            from dinkster_training_torch import Ideogram4LoRATrainingService

            service_type = Ideogram4LoRATrainingService
        elif name == "minimax-music3-lora":
            from dinkster_training_torch import MiniMaxMusic3LoRATrainingService

            service_type = MiniMaxMusic3LoRATrainingService
        else:
            from dinkster_training_torch import MiniMaxH3LoRATrainingService

            service_type = MiniMaxH3LoRATrainingService
        return service_type(
            store,
            Path(checkpoint_root),
            cancelled=_invocation_cancelled,
        )
    raise ValueError(
        f"unknown training backend {name!r}; available backends: fake, sd15-lora,"
        " sdxl-lora, flux-lora, flux2-lora, minimax-h3-lora, minimax-music3-lora,"
        " ideogram4-lora, qwen-image-lora, wan-lora"
    )
