from __future__ import annotations

from dataclasses import dataclass

import pytest
from dinkster_training_worker.memory import plan_reservations
from dinkster_values import Value


@dataclass
class _Invocation:
    node_type: str
    inputs: dict[str, Value]


def test_fake_backend_reserves_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_TRAINING_BACKEND", "fake")
    assert plan_reservations(_Invocation("training.advance", {})) == ()


def test_model_operations_reserve_governed_process_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DINKSTER_TRAINING_BACKEND", "sd15-lora")
    monkeypatch.setenv("DINKSTER_TRAINING_DEVICE", "cuda:1")
    monkeypatch.setenv("DINKSTER_TRAINING_RAM_BYTES", "100")
    monkeypatch.setenv("DINKSTER_TRAINING_VRAM_BYTES", "200")
    requests = plan_reservations(_Invocation("training.advance", {}))
    assert [(request.residency, request.nbytes) for request in requests] == [
        ("ram", 100),
        ("vram:cuda:1", 200),
    ]
    assert plan_reservations(_Invocation("training.export_lora", {})) == ()
