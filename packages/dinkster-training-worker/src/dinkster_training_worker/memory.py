"""Governor reservation policy for the dedicated training process."""

from __future__ import annotations

import os
from collections.abc import Sequence

from dinkster_memory import InvocationView, ReservationRequest

_MODEL_OPERATIONS = {"training.create_session", "training.advance"}


def _positive_bytes(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer byte count") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def plan_reservations(invocation: InvocationView) -> Sequence[ReservationRequest]:
    if os.environ.get("DINKSTER_TRAINING_BACKEND") != "sd15-lora":
        return ()
    if invocation.node_type not in _MODEL_OPERATIONS:
        return ()
    device = os.environ.get("DINKSTER_TRAINING_DEVICE", "cuda:0")
    ram = _positive_bytes("DINKSTER_TRAINING_RAM_BYTES", 8 * 1024**3)
    if device == "cpu":
        return (ReservationRequest("ram", ram),)
    if not device.startswith("cuda"):
        raise ValueError("DINKSTER_TRAINING_DEVICE must be 'cpu' or a CUDA device")
    index = device.partition(":")[2] or "0"
    vram = _positive_bytes("DINKSTER_TRAINING_VRAM_BYTES", 8 * 1024**3)
    return (
        ReservationRequest("ram", ram),
        ReservationRequest(f"vram:cuda:{index}", vram),
    )
