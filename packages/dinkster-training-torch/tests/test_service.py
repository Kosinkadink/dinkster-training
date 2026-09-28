from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import torch
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import SD15LoRATrainer, TrainingConfig
from dinkster_training_torch.service import SD15LoRATrainingService, TrainingAdvancePaused


class _StateOwner:
    def __init__(self, state: Mapping[str, object]) -> None:
        self.state: dict[str, object] = dict(state)

    def state_dict(self) -> dict[str, object]:
        return self.state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.state = dict(state)


class _Attachment:
    def __init__(self) -> None:
        self.value = torch.tensor([0.0])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"lora_unet_test.lora_up.weight": self.value.clone()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.value = state["lora_unet_test.lora_up.weight"].clone()


class _FakeTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.lora_program = _Attachment()
        self.optimizer = _StateOwner({"step": 0})
        self.randomness = _StateOwner({"data": torch.tensor([1], dtype=torch.uint8)})
        self.step_cursor = 0
        self.data_cursor = 0
        self.last_loss: float | None = None

    def restore(
        self,
        *,
        adapter: dict[str, torch.Tensor],
        optimizer: dict[str, object],
        rng: dict[str, torch.Tensor],
        step_cursor: int,
        data_cursor: int,
        loss: float | None,
    ) -> None:
        self.lora_program.load_state_dict(adapter)
        self.optimizer.load_state_dict(optimizer)
        self.randomness.load_state_dict(rng)
        self.step_cursor = step_cursor
        self.data_cursor = data_cursor
        self.last_loss = loss

    def train_step(self) -> float:
        self.step_cursor += 1
        self.data_cursor += 1
        self.lora_program.value += 1
        self.optimizer.state["step"] = self.step_cursor
        self.last_loss = 1.0 / self.step_cursor
        return self.last_loss

    def close(self) -> None:
        pass


def _config(tmp_path: Path) -> str:
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    return json.dumps(
        {
            "schemaVersion": 1,
            "family": "sd15",
            "checkpointPath": str(checkpoint),
            "checkpointDigest": "sha256:" + hashlib.sha256(b"checkpoint").hexdigest(),
            "device": "cpu",
            "seed": 9,
            "checkpointInterval": 1,
            "dataset": {
                "root": str(dataset),
                "resolution": [8, 8],
                "encodedCacheRoot": str(tmp_path / "cache"),
            },
        }
    )


def _factory(config: TrainingConfig) -> SD15LoRATrainer:
    return cast("SD15LoRATrainer", _FakeTrainer(config))


def test_committed_advance_replays_without_stepping(tmp_path: Path) -> None:
    store = TrainingSessionStore(JournalStore(tmp_path / "journal.sqlite"))
    try:
        service = SD15LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            trainer_factory=_factory,
            expected_device="cpu",
        )
        handle, _ = service.create("session", _config(tmp_path))
        first = service.advance(handle, 2, "test")
        assert first.handle.step_cursor == 2
        assert service.steps_run == 2
        replay = service.advance(handle, 2, "test")
        assert replay.handle == first.handle
        assert replay.replayed
        assert service.steps_run == 2
    finally:
        store.close()


def test_paused_advance_resumes_from_recovery_checkpoint(tmp_path: Path) -> None:
    store = TrainingSessionStore(JournalStore(tmp_path / "journal.sqlite"))
    polls = 0

    def cancel_after_one_step() -> bool:
        nonlocal polls
        polls += 1
        return polls >= 2

    try:
        service = SD15LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            trainer_factory=_factory,
            expected_device="cpu",
            cancelled=cancel_after_one_step,
        )
        handle, _ = service.create("session", _config(tmp_path))
        with pytest.raises(TrainingAdvancePaused):
            service.advance(handle, 3, "resume")
        replacement = SD15LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            trainer_factory=_factory,
            expected_device="cpu",
        )
        outcome = replacement.advance(handle, 3, "resume")
        assert outcome.handle.step_cursor == 3
        assert replacement.steps_run == 2
    finally:
        store.close()
