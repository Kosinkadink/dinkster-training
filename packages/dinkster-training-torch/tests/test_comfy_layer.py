from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch
from dinkster_training_torch.config import DatasetConfig
from dinkster_training_torch.dataset import EncodedDataset, PreparedBatch
from dinkster_training_torch.lora_program import LoRAProgram
from PIL import Image


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(3, 2, bias=False)


class _Diffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn2 = _Attention()


class _TinyForkModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_model = _Diffusion()
        self.model_config = type("Config", (), {"unet_config": {}})()
        self.model_lowvram = False
        self.lowvram_patch_counter = 0
        self.current_weight_patches_uuid = None
        self.model_loaded_weight_memory = 0
        self.model_offload_buffer_memory = 0
        self.device = torch.device("cpu")


def test_package_selects_fork_cpu_before_model_management_import() -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import dinkster_training_torch; "
                "from dinkster_inference import model_management; "
                "assert model_management.get_torch_device().type == 'cpu'"
            ),
        ],
        check=True,
        env=environment,
    )


def test_fork_lora_program_is_trainable_and_ejects() -> None:
    from dinkster_inference.lora import load_lora, model_lora_keys_unet
    from dinkster_inference.model_patcher import ModelPatcher

    model = _TinyForkModel()
    layer = model.diffusion_model.attn2.to_q
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    original_forward = layer.forward
    program = LoRAProgram(
        patcher,
        rank=1,
        alpha=1.0,
        seed=17,
        target_patterns=("attn2.to_q",),
        device=torch.device("cpu"),
    )
    program.inject(torch.device("cpu"))
    inputs = torch.tensor([[1.0, 2.0, 4.0]])
    output = layer(inputs)
    output.sum().backward()
    assert all(parameter.grad is not None for parameter in program.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in program.parameters())
    decoded = load_lora(program.state_dict(), model_lora_keys_unet(model, {}), log_missing=False)
    assert set(decoded) == {"diffusion_model.attn2.to_q.weight"}
    program.close()
    assert layer.forward == original_forward


def test_encoded_cache_hit_does_not_load_fork_encoders(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    Image.new("RGB", (8, 8), (20, 40, 80)).save(dataset_root / "sample.png")
    (dataset_root / "sample.txt").write_text("caption", encoding="utf-8")
    settings = DatasetConfig(dataset_root, (8, 8), tmp_path / "cache")
    loads = 0

    def factory():  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        nonlocal loads
        loads += 1

        def encode(images: torch.Tensor, captions: list[str]) -> PreparedBatch:
            assert images.shape == (1, 8, 8, 3)
            assert captions == ["caption"]
            return PreparedBatch(torch.ones(1, 4, 1, 1), torch.ones(1, 77, 768))

        return encode, lambda: None

    first = EncodedDataset(
        settings, checkpoint_digest="sha256:" + "a" * 64, encoder_factory=factory
    )
    second = EncodedDataset(
        settings, checkpoint_digest="sha256:" + "a" * 64, encoder_factory=factory
    )
    assert first.dataset_digest == second.dataset_digest
    assert loads == 1


def test_dataset_center_crops_before_fork_encoding(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    image = Image.new("RGB", (24, 8), (255, 0, 0))
    image.paste((0, 255, 0), (8, 0, 16, 8))
    image.paste((0, 0, 255), (16, 0, 24, 8))
    image.save(dataset_root / "sample.png")
    (dataset_root / "sample.txt").write_text("caption", encoding="utf-8")

    def factory():  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        def encode(images: torch.Tensor, captions: list[str]) -> PreparedBatch:
            assert images.shape == (1, 8, 8, 3)
            assert torch.equal(images[0], torch.tensor([0.0, 1.0, 0.0]).expand(8, 8, 3))
            return PreparedBatch(torch.ones(1, 4, 1, 1), torch.ones(1, 77, 768))

        return encode, lambda: None

    EncodedDataset(
        DatasetConfig(dataset_root, (8, 8), tmp_path / "cache"),
        checkpoint_digest="sha256:" + "a" * 64,
        encoder_factory=factory,
    )
