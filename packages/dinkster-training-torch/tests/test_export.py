from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from dinkster_training_torch.checkpoint import CheckpointState
from dinkster_training_torch.export import LoraExportSettings, export_lora


def _checkpoint(session_id: str) -> CheckpointState:
    return CheckpointState(
        manifest_digest="blake3:" + "a" * 64,
        session_id=session_id,
        config_digest="blake3:" + "b" * 64,
        extension_snapshot_digest="blake3:" + "c" * 64,
        parent_manifest_digest="",
        step_cursor=2,
        config={},
        adapter={
            "lora_unet_z.lora_up.weight": torch.tensor([[3.0], [4.0]]),
            "lora_unet_z.alpha": torch.tensor(1.0),
            "lora_unet_z.lora_down.weight": torch.tensor([[1.0, 2.0]]),
        },
        optimizer={},
        rng={},
        data_cursor=2,
        loss=0.5,
    )


def test_export_is_byte_identical_across_sessions(tmp_path: Path) -> None:
    first_path = tmp_path / "first.safetensors"
    second_path = tmp_path / "second.safetensors"
    first, first_digest = export_lora(
        _checkpoint("first"),
        LoraExportSettings(first_path, "fp32"),
        runtime_identity="test-runtime",
    )
    second, second_digest = export_lora(
        replace(_checkpoint("first"), session_id="second"),
        LoraExportSettings(second_path, "fp32"),
        runtime_identity="test-runtime",
    )
    assert first.read_bytes() == second.read_bytes()
    assert first_digest == second_digest
