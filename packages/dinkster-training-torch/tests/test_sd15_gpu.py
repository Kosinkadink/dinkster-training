from __future__ import annotations

import hashlib
import multiprocessing
import os
from pathlib import Path

import pytest
from PIL import Image


def _train_once(config_mapping: dict[str, object], output: str, session_id: str) -> None:
    from dinkster_training_torch import SD15LoRATrainer, TrainingConfig
    from dinkster_training_torch.checkpoint import CheckpointState
    from dinkster_training_torch.export import LoraExportSettings, export_lora

    config = TrainingConfig.from_mapping(config_mapping)
    trainer = SD15LoRATrainer(config)
    try:
        loss = trainer.train_step()
        state = CheckpointState(
            manifest_digest="blake3:" + "a" * 64,
            session_id=session_id,
            config_digest=config.digest,
            extension_snapshot_digest="blake3:" + "b" * 64,
            parent_manifest_digest="",
            step_cursor=1,
            config=config.to_mapping(),
            adapter=trainer.lora_program.state_dict(),
            optimizer=trainer.optimizer.state_dict(),
            rng=trainer.randomness.state_dict(),
            data_cursor=trainer.data_cursor,
            loss=loss,
        )
        export_lora(
            state,
            LoraExportSettings(Path(output), "fp32"),
            runtime_identity="dinkster-comfy-sd15-lora/1",
        )
    finally:
        trainer.close()


@pytest.mark.skipif(
    "DINKSTER_SD15_CHECKPOINT" not in os.environ,
    reason="set DINKSTER_SD15_CHECKPOINT for the physical-GPU determinism proof",
)
def test_two_cold_processes_export_byte_identical_adapters(tmp_path: Path) -> None:
    checkpoint = Path(os.environ["DINKSTER_SD15_CHECKPOINT"]).resolve()
    checkpoint_digest = "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (64, 64), (40, 80, 120)).save(dataset / "sample.png")
    (dataset / "sample.txt").write_text("a blue square", encoding="utf-8")
    config: dict[str, object] = {
        "schemaVersion": 1,
        "family": "sd15",
        "checkpointPath": str(checkpoint),
        "checkpointDigest": checkpoint_digest,
        "device": "cuda:0",
        "seed": 459,
        "rank": 1,
        "alpha": 1.0,
        "learningRate": 0.0001,
        "batchSize": 1,
        "gradientAccumulationSteps": 1,
        "checkpointInterval": 1,
        "targetPatterns": ["input_blocks.1.1.transformer_blocks.0.attn2.to_q"],
        "dataset": {
            "root": str(dataset),
            "resolution": [64, 64],
            "encodedCacheRoot": str(tmp_path / "encoded"),
        },
    }
    context = multiprocessing.get_context("spawn")
    outputs = [tmp_path / "first.safetensors", tmp_path / "second.safetensors"]
    for index, output in enumerate(outputs):
        process = context.Process(
            target=_train_once,
            args=(config, str(output), f"independent-session-{index}"),
        )
        process.start()
        process.join(timeout=300)
        assert process.exitcode == 0
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
