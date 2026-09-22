"""Executable proofs for Qwen-Image configuration, assembly, and LoRA attachment."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import dinkster_training_torch.service as training_service
import dinkster_training_torch.trainer as training_trainer
import pytest
import torch
from dinkster_assets import AssetIntegrityError, digest_bytes
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    QWEN_IMAGE,
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_2511_CONFIG,
    QWEN_IMAGE_TEXT_CONFIG,
    WAN21_VAE_CONFIG,
    QwenImageConfig,
    decode_lora,
    load_safetensors_header,
    qwen_image_lora_key_map,
)
from dinkster_inference_torch import QwenImage, load_tensors
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    FLUX2_DEV_TRAINING_RUNTIME_IDENTITY,
    FLUX2_DEV_TRAINING_SNAPSHOT_DIGEST,
    FLUX2_KLEIN_4B_TRAINING_RUNTIME_IDENTITY,
    FLUX2_KLEIN_4B_TRAINING_SNAPSHOT_DIGEST,
    FLUX2_KLEIN_9B_TRAINING_RUNTIME_IDENTITY,
    FLUX2_KLEIN_9B_TRAINING_SNAPSHOT_DIGEST,
    FLUX_DEV_TRAINING_RUNTIME_IDENTITY,
    FLUX_DEV_TRAINING_SNAPSHOT_DIGEST,
    MINIMAX_H3_TRAINING_RUNTIME_IDENTITY,
    MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST,
    QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY,
    QWEN_IMAGE_TRAINING_SNAPSHOT_DIGEST,
    SDXL_TRAINING_RUNTIME_IDENTITY,
    SDXL_TRAINING_SNAPSHOT_DIGEST,
    TRAINING_RUNTIME_IDENTITY,
    TRAINING_SNAPSHOT_DIGEST,
    WAN_TRAINING_RUNTIME_IDENTITY,
    WAN_TRAINING_SNAPSHOT_DIGEST,
    CheckpointError,
    ContentAddressedCheckpointStore,
    QwenImageLoRATrainer,
    QwenImageLoRATrainingService,
    QwenImagePreparedBatch,
    QwenImageTrainingConfig,
    TrainableAttachment,
    TrainingConfigError,
    default_qwen_image_model_factory,
    qwen_image_capability_report,
    qwen_image_training_sigma_table,
)
from dinkster_training_torch.attachment import resolve_lora_targets
from dinkster_training_torch.config import TrainingArtifactSource
from dinkster_training_torch.export import (
    LoraExportSource,
    _qwen_image_tensors,  # pyright: ignore[reportPrivateUsage]
)
from PIL import Image
from test_sd15_lora import assert_tree_equal


@dataclass(frozen=True)
class _ReducedQwenImageConfig:
    transformer_blocks: int = 2
    hidden_width: int = 12
    attention_heads: int = 2
    attention_head_dim: int = 6
    text_width: int = 6
    pooled_width: int = 4
    patchified_input_channels: int = 8
    output_latent_channels: int = 2
    patch: tuple[int, int] = (2, 2)
    rope_axes: tuple[int, int, int] = (2, 2, 2)
    default_ref_method: str = "index"
    use_additional_t_cond: bool = False


@dataclass(frozen=True)
class _ObjectiveQwenImageConfig:
    transformer_blocks: int = 2
    hidden_width: int = 12
    attention_heads: int = 2
    attention_head_dim: int = 6
    text_width: int = 3584
    pooled_width: int = 4
    patchified_input_channels: int = 64
    output_latent_channels: int = 16
    patch: tuple[int, int] = (2, 2)
    rope_axes: tuple[int, int, int] = (2, 2, 2)
    default_ref_method: str = "index"
    use_additional_t_cond: bool = False


def tiny_qwen_image_model() -> QwenImage:
    model = QwenImage(cast("QwenImageConfig", _ReducedQwenImageConfig()))
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) * 0.001)
    return model


def _artifact(path: Path, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "digest": digest_bytes(data), "size": len(data)}


def config_mapping(tmp_path: Path) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "family": "qwen-image",
        "variant": "qwen-image",
        "ditState": _artifact(tmp_path / "dit.safetensors", b"qwen-image-dit"),
        "textEncoderState": _artifact(tmp_path / "text.safetensors", b"qwen-image-text"),
        "vaeState": _artifact(tmp_path / "vae.safetensors", b"wan21-vae"),
        "datasetIdentity": "blake3:" + "7" * 64,
        "device": "cpu",
        "baseDtype": "bfloat16",
        "rank": 2,
        "alpha": 2.0,
        "loraTargets": ["attention.qkvo", "mlp.projections"],
        "learningRate": 0.0001,
        "weightDecay": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "optimizer": "adamw",
        "gradientAccumulationSteps": 1,
        "gradientCheckpointing": True,
        "seed": 1001,
        "latentShape": [1, 16, 1, 64, 64],
        "contextShape": [1, 512, 3584],
        "attentionMaskShape": [1, 512],
    }


def _identity_digest(config: QwenImageTrainingConfig) -> str:
    payload = json.dumps(config.identity_mapping(), sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return digest_bytes(payload)


def test_qwen_image_config_round_trips_exact_contract(tmp_path: Path) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))

    assert QwenImageTrainingConfig.from_mapping(config.to_mapping()) == config
    assert config.latent_shape == (1, 16, 1, 64, 64)
    assert config.context_shape == (1, 512, 3584)
    assert config.attention_mask_shape == (1, 512)
    assert config.guidance is None
    assert "guidance" not in config.to_mapping()


def test_qwen_image_config_has_stable_identity(tmp_path: Path) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))

    assert _identity_digest(config) == (
        "blake3:9e31d8863c685a2d56e90fe5d4708c7238193ed25385de1ccf9832ec0965d87d"
    )
    moved = config.to_mapping()
    for field in ("ditState", "textEncoderState", "vaeState"):
        artifact = cast("dict[str, object]", moved[field])
        artifact["path"] = str(tmp_path / "moved" / field)
    moved["checkpointInterval"] = 10
    moved["loraExportInterval"] = 5
    moved["syncDigestInterval"] = 3
    assert _identity_digest(QwenImageTrainingConfig.from_mapping(moved)) == _identity_digest(config)
    changed = config.to_mapping()
    changed["rank"] = 4
    assert _identity_digest(QwenImageTrainingConfig.from_mapping(changed)) != _identity_digest(
        config
    )


def test_qwen_image_service_identities_are_canonical_and_disjoint(tmp_path: Path) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))
    config_digest = training_service._config_digest(config)  # pyright: ignore[reportPrivateUsage]

    assert config_digest == (
        "blake3:27cc66ff38ccc2d3e2e1b46164f98db02eb037ae7bcc6482d18e173a888b5c0f"
    )
    assert config_digest == training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
        config
    )
    assert QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY == "qwen-image-lora-torch/1"
    assert QWEN_IMAGE_TRAINING_SNAPSHOT_DIGEST == (
        "blake3:fa77484159d59cd7defeacc6f54ea9024b3567ece624d5e245d1334902ed591d"
    )
    assert (
        training_service._runtime_identity(  # pyright: ignore[reportPrivateUsage]
            config
        )
        == QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY
    )
    assert (
        training_service._snapshot_digest(  # pyright: ignore[reportPrivateUsage]
            config
        )
        == QWEN_IMAGE_TRAINING_SNAPSHOT_DIGEST
    )
    existing_runtime_identities = {
        TRAINING_RUNTIME_IDENTITY,
        SDXL_TRAINING_RUNTIME_IDENTITY,
        MINIMAX_H3_TRAINING_RUNTIME_IDENTITY,
        WAN_TRAINING_RUNTIME_IDENTITY,
        FLUX_DEV_TRAINING_RUNTIME_IDENTITY,
        FLUX2_DEV_TRAINING_RUNTIME_IDENTITY,
        FLUX2_KLEIN_9B_TRAINING_RUNTIME_IDENTITY,
        FLUX2_KLEIN_4B_TRAINING_RUNTIME_IDENTITY,
    }
    existing_snapshot_digests = {
        TRAINING_SNAPSHOT_DIGEST,
        SDXL_TRAINING_SNAPSHOT_DIGEST,
        MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST,
        WAN_TRAINING_SNAPSHOT_DIGEST,
        FLUX_DEV_TRAINING_SNAPSHOT_DIGEST,
        FLUX2_DEV_TRAINING_SNAPSHOT_DIGEST,
        FLUX2_KLEIN_9B_TRAINING_SNAPSHOT_DIGEST,
        FLUX2_KLEIN_4B_TRAINING_SNAPSHOT_DIGEST,
    }
    assert QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY not in existing_runtime_identities
    assert QWEN_IMAGE_TRAINING_SNAPSHOT_DIGEST not in existing_snapshot_digests


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schemaVersion", True, "schemaVersion must be an integer"),
        ("family", True, "family must be 'qwen-image'"),
        ("variant", [], "unsupported Qwen-Image training variant"),
        ("baseDtype", "float16", "baseDtype must be 'bfloat16' or 'float32'"),
        ("rank", True, "rank must be an integer"),
        ("latentShape", [1, 16.0, 1, 64, 64], "latentShape.*integer"),
        ("latentShape", [1, 16, False, 64, 64], "latentShape.*integer"),
        ("contextShape", [1, 512.0, 3584], "contextShape.*integer"),
        ("attentionMaskShape", [1, False], "attentionMaskShape.*integer"),
        ("guidance", 3.5, "does not accept guidance"),
        ("loraTargets", ["attention.qkvo"], "loraTargets must be"),
        (
            "textEncoderState",
            {"path": "/tmp/text", "digest": "blake3:" + "0" * 64, "size": True},
            "textEncoderState.size must be an integer",
        ),
    ),
)
def test_qwen_image_config_rejects_malformed_wire_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    mapping = config_mapping(tmp_path)
    mapping[field] = value

    with pytest.raises(TrainingConfigError, match=message):
        QwenImageTrainingConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("variant", "qwen-image-edit", "unsupported Qwen-Image training variant"),
        ("latentShape", [1, 64, 1, 64, 64], "latentShape must be"),
        ("latentShape", [1, 16, 2, 64, 64], "latentShape must be"),
        ("latentShape", [1, 16, 1, 63, 64], "latentShape must be"),
        ("contextShape", [1, 512, 4096], "contextShape must be"),
        ("attentionMaskShape", [1, 511], "attentionMaskShape must match"),
    ),
)
def test_qwen_image_config_rejects_wrong_geometry_and_variants(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    mapping = config_mapping(tmp_path)
    mapping[field] = value

    with pytest.raises(TrainingConfigError, match=message):
        QwenImageTrainingConfig.from_mapping(mapping)


def test_qwen_image_config_direct_construction_requires_normalized_exact_types(
    tmp_path: Path,
) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))

    with pytest.raises(TrainingConfigError, match="family must be 'qwen-image'"):
        replace(config, family=cast("Literal['qwen-image']", True))
    with pytest.raises(TrainingConfigError, match="unsupported Qwen-Image training variant"):
        replace(config, variant=cast('Literal["qwen-image"]', "qwen-image-edit"))
    with pytest.raises(TrainingConfigError, match="baseDtype must be 'bfloat16' or 'float32'"):
        replace(config, base_dtype=cast("Literal['bfloat16', 'float32']", "float16"))
    with pytest.raises(TrainingConfigError, match="rank must be an integer"):
        replace(config, rank=cast("int", True))
    with pytest.raises(TrainingConfigError, match="alpha must be a normalized float"):
        replace(config, alpha=cast("float", 2))
    with pytest.raises(TrainingConfigError, match="latentShape must be"):
        replace(
            config,
            latent_shape=cast("tuple[int, int, int, int, int]", (1, 16.0, 1, 64, 64)),
        )
    with pytest.raises(TrainingConfigError, match="contextShape must be"):
        replace(config, context_shape=cast("tuple[int, int, int]", (1, 512, 3584.0)))
    with pytest.raises(TrainingConfigError, match="attentionMaskShape must match"):
        replace(config, attention_mask_shape=(1, 511))
    with pytest.raises(TrainingConfigError, match="does not accept guidance"):
        replace(config, guidance=cast("None", 3.5))
    with pytest.raises(TrainingConfigError, match="textEncoderState.size must be an integer"):
        replace(
            config,
            text_encoder_state=replace(config.text_encoder_state, size=cast("int", True)),
        )


def test_qwen_image_targets_cover_both_streams_and_only_native_projections() -> None:
    targets = resolve_lora_targets(tiny_qwen_image_model(), 2, family="qwen-image")

    assert len(targets) == 24
    assert all(
        target.target_id == f"qwen-image/dit/{target.module_path}/weight" for target in targets
    )
    expected: set[str] = set()
    for block in range(2):
        root = f"transformer_blocks.{block}"
        expected.update(
            f"{root}.attn.{projection}"
            for projection in (
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
                "to_out.0",
                "to_add_out",
            )
        )
        expected.update(
            (
                f"{root}.img_mlp.net.0.proj",
                f"{root}.img_mlp.net.2",
                f"{root}.txt_mlp.net.0.proj",
                f"{root}.txt_mlp.net.2",
            )
        )
    actual = {target.module_path for target in targets}
    assert actual == expected
    assert not any(
        forbidden in path
        for path in actual
        for forbidden in (
            "img_mod",
            "txt_mod",
            "norm",
            "time_text_embed",
            "img_in",
            "txt_in",
            "proj_out",
        )
    )


def test_qwen_image_attachment_is_deterministic_and_detaches_cleanly() -> None:
    first_model = tiny_qwen_image_model()
    second_model = tiny_qwen_image_model()
    frozen = {name: value.detach().clone() for name, value in first_model.state_dict().items()}
    first = TrainableAttachment.attach(first_model, rank=2, alpha=2.0, seed=91, family="qwen-image")
    second = TrainableAttachment.attach(
        second_model, rank=2, alpha=2.0, seed=91, family="qwen-image"
    )

    assert all(not parameter.requires_grad for parameter in first_model.parameters())
    assert all(
        parameter.dtype == torch.float32 and parameter.requires_grad
        for parameter in first.parameters()
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    first_state = first.state_dict()
    assert not torch.equal(
        first_state["qwen-image/dit/transformer_blocks.0.attn.to_q/weight.down"],
        first_state["qwen-image/dit/transformer_blocks.0.attn.add_q_proj/weight.down"],
    )

    target = first.targets[0]
    module = cast("torch.nn.Linear", first_model.get_submodule(target.module_path))
    inputs = torch.linspace(-1.0, 1.0, module.in_features).reshape(1, -1)
    baseline = module(inputs)
    with torch.no_grad():
        dict(first.named_parameters())[f"{target.target_id}.up"].fill_(0.25)
    assert not torch.equal(module(inputs), baseline)

    first.detach()
    assert torch.equal(module(inputs), baseline)
    assert all(torch.equal(value, frozen[name]) for name, value in first_model.state_dict().items())


class _QwenImageBatches:
    def __init__(self, batch: QwenImagePreparedBatch) -> None:
        self._batch = batch

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> QwenImagePreparedBatch:
        del cursor, generator, device
        return self._batch


def _objective_config(tmp_path: Path, *, device: str = "cpu") -> QwenImageTrainingConfig:
    mapping = config_mapping(tmp_path)
    mapping["device"] = device
    mapping["baseDtype"] = "float32"
    mapping["gradientCheckpointing"] = False
    mapping["latentShape"] = [1, 16, 1, 2, 2]
    mapping["contextShape"] = [1, 512, 3584]
    mapping["attentionMaskShape"] = [1, 512]
    return QwenImageTrainingConfig.from_mapping(mapping)


def _objective_model() -> QwenImage:
    generator = torch.Generator().manual_seed(19)
    model = QwenImage(cast("QwenImageConfig", _ObjectiveQwenImageConfig()))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model


def _objective_batch(
    config: QwenImageTrainingConfig, *, prepared_noise: bool
) -> QwenImagePreparedBatch:
    generator = torch.Generator().manual_seed(23)
    latents = torch.randn(config.latent_shape, generator=generator)
    context = torch.randn(config.context_shape, generator=generator)
    mask = torch.cat((torch.ones((1, 127)), torch.zeros((1, 385))), dim=1)
    if not prepared_noise:
        return QwenImagePreparedBatch(latents, context, mask)
    return QwenImagePreparedBatch(
        latents,
        context,
        mask,
        sigma_indices=torch.tensor([73]),
        noise=torch.randn(config.latent_shape, generator=generator),
    )


def _initialize_lora_up(trainer: QwenImageLoRATrainer) -> None:
    with torch.no_grad():
        for name, parameter in trainer.attachment.named_parameters():
            if name.endswith(".up"):
                parameter.fill_(0.01)


def test_qwen_image_training_schedule_is_bit_exact_inference_reference() -> None:
    from dinkster_inference_torch import schedules

    actual = qwen_image_training_sigma_table("qwen-image")
    expected = schedules._FluxRef(  # pyright: ignore[reportPrivateUsage]
        QWEN_IMAGE_CONFIG.sampling_shift, 10000
    ).sigmas
    wrong_shift = schedules._FluxRef(1.0, 10000).sigmas  # pyright: ignore[reportPrivateUsage]
    wrong_count = schedules._FluxRef(  # pyright: ignore[reportPrivateUsage]
        QWEN_IMAGE_CONFIG.sampling_shift, 9999
    ).sigmas

    assert QWEN_IMAGE_CONFIG.sampling_shift == 1.15
    assert len(actual) == 10000
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, wrong_shift)
    assert not torch.equal(actual[:-1], wrong_count)
    assert actual[0].item() == 0.00031575115281157196
    assert actual[-1].item() == 1.0
    with pytest.raises(ValueError, match="unsupported Qwen-Image training variant"):
        qwen_image_training_sigma_table("qwen-image-edit")


_GPU_TESTS_ENABLED = os.environ.get("DINKSTER_ENABLE_GPU_TESTS") == "1"
_OBJECTIVE_DEVICES = (
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available() or not _GPU_TESTS_ENABLED,
            reason="needs DINKSTER_ENABLE_GPU_TESTS=1 and CUDA",
        ),
    ),
)


@pytest.mark.parametrize("device", _OBJECTIVE_DEVICES)
def test_qwen_image_flow_objective_runs_real_dit_and_trains_image_reachable_targets(
    tmp_path: Path, device: str
) -> None:
    config = _objective_config(tmp_path, device=device)
    batch = _objective_batch(config, prepared_noise=True)
    assert batch.sigma_indices is not None and batch.noise is not None
    model = _objective_model()
    trainer = QwenImageLoRATrainer(config, model, _QwenImageBatches(batch))
    _initialize_lora_up(trainer)
    frozen = {name: value.detach().clone() for name, value in model.state_dict().items()}
    seen_noisy: list[torch.Tensor] = []
    seen_sigmas: list[torch.Tensor] = []
    seen_masks: list[torch.Tensor] = []
    seen_predictions: list[torch.Tensor] = []
    original_forward = model.forward

    def capture_forward(
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        seen_noisy.append(x.detach().clone())
        seen_sigmas.append(timesteps.detach().clone())
        seen_masks.append(attention_mask.detach().clone())
        result = original_forward(x, timesteps, context, attention_mask)
        seen_predictions.append(result.detach().clone())
        return result

    model.forward = capture_forward  # pyright: ignore[reportAttributeAccessIssue]
    loss = trainer.train_step()

    assert math.isfinite(loss)
    assert len(seen_noisy) == len(seen_sigmas) == len(seen_masks) == len(seen_predictions) == 1
    sigma = qwen_image_training_sigma_table("qwen-image")[73].to(device)
    expected_noisy = ((1.0 - sigma) * batch.latents.to(device) + sigma * batch.noise.to(device)).to(
        next(model.parameters()).dtype
    )
    target = batch.noise.to(device) - batch.latents.to(device)
    expected_loss = torch.nn.functional.mse_loss(seen_predictions[0].float(), target)
    assert torch.equal(seen_noisy[0], expected_noisy)
    assert torch.equal(seen_sigmas[0], sigma.reshape(1))
    assert seen_masks[0].dtype == torch.bool
    assert torch.equal(seen_masks[0], batch.attention_mask.to(device, dtype=torch.bool))
    assert loss == expected_loss.item()
    up_gradients = {
        name: parameter.grad
        for name, parameter in trainer.attachment.named_parameters()
        if name.endswith(".up")
    }
    final_text_only = {
        "qwen-image/dit/transformer_blocks.1.attn.add_q_proj/weight.up",
        "qwen-image/dit/transformer_blocks.1.attn.to_add_out/weight.up",
        "qwen-image/dit/transformer_blocks.1.txt_mlp.net.0.proj/weight.up",
        "qwen-image/dit/transformer_blocks.1.txt_mlp.net.2/weight.up",
    }
    image_reachable_gradients = {
        name: gradient for name, gradient in up_gradients.items() if name not in final_text_only
    }
    assert len(up_gradients) == len(trainer.attachment.targets) == 24
    assert set(up_gradients) - set(image_reachable_gradients) == final_text_only
    assert len(image_reachable_gradients) == 20
    assert all(
        gradient is not None
        and bool(torch.isfinite(gradient).all().item())
        and bool(torch.count_nonzero(gradient).item())
        for gradient in image_reachable_gradients.values()
    )
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(value, frozen[name]) for name, value in model.state_dict().items())


def test_qwen_image_same_seed_repeats_objective_and_update(tmp_path: Path) -> None:
    config = _objective_config(tmp_path)
    batch = _objective_batch(config, prepared_noise=False)
    first = QwenImageLoRATrainer(config, _objective_model(), _QwenImageBatches(batch))
    second = QwenImageLoRATrainer(config, _objective_model(), _QwenImageBatches(batch))
    _initialize_lora_up(first)
    _initialize_lora_up(second)

    first_loss = first.train_step()
    second_loss = second.train_step()

    assert first_loss == second_loss
    assert all(
        torch.equal(value, second.attachment.state_dict()[name])
        for name, value in first.attachment.state_dict().items()
    )


@pytest.mark.parametrize("mode", ("blockNonReentrant", "wholeModel"))
def test_qwen_image_checkpointing_modes_execute_real_dit(tmp_path: Path, mode: str) -> None:
    config = replace(
        _objective_config(tmp_path),
        gradient_checkpointing=True,
        checkpointing_mode=cast('Literal["blockNonReentrant", "wholeModel"]', mode),
    )
    model = _objective_model()
    trainer = QwenImageLoRATrainer(
        config, model, _QwenImageBatches(_objective_batch(config, prepared_noise=True))
    )
    assert all(
        ("forward" in block.__dict__) == (mode == "blockNonReentrant")
        for block in model.transformer_blocks
    )

    assert math.isfinite(trainer.train_step())


def _planned_qwen_image(*, dtype: object = BFLOAT16) -> SimpleNamespace:
    diffusion = SimpleNamespace(
        config=QWEN_IMAGE_CONFIG,
        quant={},
        dtypes={"transformer_blocks.0.attn.to_q.weight": dtype},
        component="diffusion",
    )
    text_encoder = SimpleNamespace(
        config=QWEN_IMAGE_TEXT_CONFIG,
        quant={},
        component="qwen2_5_vl_7b",
    )
    vae = SimpleNamespace(config=WAN21_VAE_CONFIG, quant={}, component="vae")
    return SimpleNamespace(
        family=QWEN_IMAGE,
        diffusion=diffusion,
        qwen2_5_vl_7b=text_encoder,
        vae=vae,
    )


def _opaque_artifact_header(_source: TrainingArtifactSource, _family: str) -> object:
    return object()


def _return_plan(plan: SimpleNamespace) -> Callable[..., SimpleNamespace]:
    def provide_plan(**_sources: object) -> SimpleNamespace:
        return plan

    return provide_plan


def _change_family(plan: SimpleNamespace) -> None:
    plan.family = object()


def _change_diffusion_config(plan: SimpleNamespace) -> None:
    plan.diffusion.config = QWEN_IMAGE_EDIT_2511_CONFIG


def _change_text_config(plan: SimpleNamespace) -> None:
    plan.qwen2_5_vl_7b.config = object()


def _change_vae_config(plan: SimpleNamespace) -> None:
    plan.vae.config = object()


def test_qwen_image_model_plan_rejects_wrong_artifact_digest(tmp_path: Path) -> None:
    mapping = config_mapping(tmp_path)
    dit = cast("dict[str, object]", mapping["ditState"])
    dit["digest"] = "blake3:" + "0" * 64
    config = QwenImageTrainingConfig.from_mapping(mapping)

    with pytest.raises(AssetIntegrityError, match="digest_mismatch"):
        training_trainer.qwen_image_model_assembly_plan(config)


def test_qwen_image_model_plan_uses_matching_complete_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_qwen_image()
    calls: list[dict[str, object]] = []

    def fake_artifact_header(
        source: TrainingArtifactSource, family: str
    ) -> tuple[TrainingArtifactSource, str]:
        return source, family

    monkeypatch.setattr(training_trainer, "_artifact_header", fake_artifact_header)

    def fake_plan(**sources: object) -> object:
        calls.append(sources)
        return plan

    monkeypatch.setattr(training_trainer, "plan_qwen_image_assembly", fake_plan)

    assert training_trainer.qwen_image_model_assembly_plan(config) is plan
    assert [set(call) for call in calls] == [{"diffusion", "qwen2_5_vl_7b", "vae"}]
    headers = cast("tuple[tuple[TrainingArtifactSource, str], ...]", tuple(calls[0].values()))
    assert all(value[1] == "Qwen-Image" for value in headers)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (_change_family, "does not match"),
        (_change_diffusion_config, "does not match"),
        (_change_text_config, "Qwen2.5-VL-7B"),
        (_change_vae_config, "Wan 2.1 VAE"),
    ),
)
def test_qwen_image_model_plan_rejects_wrong_family_and_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[SimpleNamespace], None],
    message: str,
) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_qwen_image()
    mutate(plan)

    monkeypatch.setattr(training_trainer, "_artifact_header", _opaque_artifact_header)
    monkeypatch.setattr(training_trainer, "plan_qwen_image_assembly", _return_plan(plan))

    with pytest.raises(ValueError, match=message):
        training_trainer.qwen_image_model_assembly_plan(config)


@pytest.mark.parametrize("role", ("diffusion", "qwen2_5_vl_7b", "vae"))
def test_qwen_image_model_plan_rejects_quantized_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_qwen_image()
    getattr(plan, role).quant = {"layer": object()}

    monkeypatch.setattr(training_trainer, "_artifact_header", _opaque_artifact_header)
    monkeypatch.setattr(training_trainer, "plan_qwen_image_assembly", _return_plan(plan))

    with pytest.raises(ValueError, match="quantized Qwen-Image training components"):
        training_trainer.qwen_image_model_assembly_plan(config)


@pytest.mark.parametrize("dtype", (BFLOAT16, FLOAT32))
def test_qwen_image_model_plan_accepts_supported_dit_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: object
) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_qwen_image(dtype=dtype)
    monkeypatch.setattr(training_trainer, "_artifact_header", _opaque_artifact_header)
    monkeypatch.setattr(training_trainer, "plan_qwen_image_assembly", _return_plan(plan))

    assert training_trainer.qwen_image_model_assembly_plan(config) is plan


def test_qwen_image_model_plan_rejects_unsupported_dit_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = QwenImageTrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_qwen_image(dtype=FLOAT16)
    monkeypatch.setattr(training_trainer, "_artifact_header", _opaque_artifact_header)
    monkeypatch.setattr(training_trainer, "plan_qwen_image_assembly", _return_plan(plan))

    with pytest.raises(ValueError, match="storage must be bfloat16 or float32"):
        training_trainer.qwen_image_model_assembly_plan(config)


def test_qwen_image_factory_strict_loads_only_dit_at_selected_dtype_and_freezes_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = config_mapping(tmp_path)
    mapping["baseDtype"] = "float32"
    config = QwenImageTrainingConfig.from_mapping(mapping)
    plan = _planned_qwen_image(dtype=FLOAT32)
    model = tiny_qwen_image_model()
    calls: list[object] = []

    def provide_plan(_config: QwenImageTrainingConfig) -> SimpleNamespace:
        return plan

    monkeypatch.setattr(training_trainer, "qwen_image_model_assembly_plan", provide_plan)

    def load_component(component: object, factory: object) -> QwenImage:
        calls.extend((component, factory))
        return model

    monkeypatch.setattr(training_trainer, "load_planned_component", load_component)

    loaded = default_qwen_image_model_factory(config)

    assert loaded is model
    assert calls == [plan.diffusion, QwenImage]
    assert all(parameter.dtype == torch.float32 for parameter in loaded.parameters())
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def _service_model_factory(config: QwenImageTrainingConfig) -> QwenImage:
    del config
    return _objective_model()


def _service_data_source_factory(config: QwenImageTrainingConfig) -> _QwenImageBatches:
    return _QwenImageBatches(_objective_batch(config, prepared_noise=True))


def _service_report(config: QwenImageTrainingConfig) -> dict[str, object]:
    return {
        "family": config.family,
        "variant": config.variant,
        "dataset": {"type": "injected", "errors": []},
    }


def _service_config(tmp_path: Path, *, cadence: bool = False) -> QwenImageTrainingConfig:
    mapping = config_mapping(tmp_path)
    mapping["baseDtype"] = "float32"
    mapping["gradientCheckpointing"] = False
    mapping["latentShape"] = [1, 16, 1, 2, 2]
    if cadence:
        mapping["loraExportInterval"] = 1
    return QwenImageTrainingConfig.from_mapping(mapping)


def _objective_assembly_plan(config: QwenImageTrainingConfig) -> SimpleNamespace:
    del config
    return SimpleNamespace(
        diffusion=SimpleNamespace(config=cast("QwenImageConfig", _ObjectiveQwenImageConfig()))
    )


def test_qwen_image_capability_report_is_exact_and_does_not_build_runtime_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _service_config(tmp_path)
    monkeypatch.setattr(
        training_service,
        "qwen_image_model_assembly_plan",
        _objective_assembly_plan,
    )

    report = qwen_image_capability_report(config)

    assert report["trainer"] == QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY
    assert report["variant"] == "qwen-image"
    assert report["configDigest"] == training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
        config
    )
    assert report["sessionExtensionSnapshotDigest"] == QWEN_IMAGE_TRAINING_SNAPSHOT_DIGEST
    targets = cast("list[dict[str, object]]", report["targets"])
    assert len(targets) == 24
    assert all(cast("str", target["targetId"]).startswith("qwen-image/dit/") for target in targets)
    blocks: dict[str, int] = {}
    for target in targets:
        block = cast("str", target["modulePath"]).split(".")[1]
        blocks[block] = blocks.get(block, 0) + 1
    assert blocks == {"0": 12, "1": 12}
    capabilities = cast("dict[str, object]", report["capabilities"])
    assert capabilities["families"] == ["qwen-image"]
    assert capabilities["checkpointResume"] is True
    assert capabilities["textEncoderTraining"] is False
    assert capabilities["guidanceConditioning"] is False
    assert capabilities["datasetBucketing"] is False
    assert capabilities["quantizedBase"] is False
    assert capabilities["loraExport"] is True
    precision = cast("dict[str, object]", report["precisionPlan"])
    assert precision == {
        "baseStorage": "float32",
        "forwardCompute": "float32",
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
        "flowObjective": "float32",
        "preparedBatch": "float32 latents, context, and attention mask",
    }
    memory = cast("dict[str, object]", report["memoryLedger"])
    categories = cast("dict[str, int]", memory["categories"])
    parameter_counts = cast("dict[str, int]", report["parameterCounts"])
    assert categories["frozenBaseParameters"] == parameter_counts["frozenBase"] * 4
    expected_batch_bytes = (
        4 * math.prod(config.latent_shape)
        + math.prod(config.context_shape)
        + math.prod(config.attention_mask_shape)
    ) * 4 + math.prod(config.attention_mask_shape)
    assert categories["preparedBatchAndNoisedStreamsLowerBound"] == expected_batch_bytes
    bfloat16_report = qwen_image_capability_report(replace(config, base_dtype="bfloat16"))
    bfloat16_categories = cast(
        "dict[str, int]",
        cast("dict[str, object]", bfloat16_report["memoryLedger"])["categories"],
    )
    assert bfloat16_categories["frozenBaseParameters"] == parameter_counts["frozenBase"] * 2


@pytest.mark.parametrize("device", ("mps", "not-a-device"))
def test_qwen_image_capability_report_rejects_invalid_devices(tmp_path: Path, device: str) -> None:
    config = replace(_service_config(tmp_path), device=device)
    with pytest.raises(ValueError, match="invalid training device|supports cpu and cuda"):
        qwen_image_capability_report(config)


def test_qwen_image_capability_report_rejects_nonexistent_cuda_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config = replace(_service_config(tmp_path), device="cuda:2")

    with pytest.raises(ValueError, match="CUDA device cuda:2 does not exist"):
        qwen_image_capability_report(config)


@pytest.mark.parametrize("cache_state", ("hit", "miss"))
def test_qwen_image_service_reports_cache_without_constructing_runtime_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_state: str
) -> None:
    mapping = config_mapping(tmp_path / "artifacts")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (16, 16)).save(dataset / "item.png")
    (dataset / "item.txt").write_text("caption", encoding="utf-8")
    mapping.pop("datasetIdentity")
    mapping["baseDtype"] = "float32"
    mapping["gradientCheckpointing"] = False
    mapping["latentShape"] = [1, 16, 1, 2, 2]
    mapping["dataset"] = {
        "type": "qwen-image-image-caption-folder",
        "root": str(dataset.resolve()),
        "resolution": [16, 16],
        "contextTokens": 512,
        "encodedCacheRoot": str((tmp_path / "cache").resolve()),
    }
    config = QwenImageTrainingConfig.from_mapping(mapping)
    monkeypatch.setattr(
        training_service,
        "qwen_image_model_assembly_plan",
        _objective_assembly_plan,
    )

    def cache_probe(*_args: object, **_kwargs: object) -> str:
        return cache_state

    monkeypatch.setattr(
        training_service,
        "qwen_image_encoded_cache_state",
        cache_probe,
    )

    def component_bytes(*_args: object, **_kwargs: object) -> dict[str, int]:
        return {
            "vaeEncoderParametersTransientLowerBound": 11,
            "qwen25VlTextEncoderParametersTransientLowerBound": 13,
        }

    monkeypatch.setattr(
        training_service,
        "_qwen_image_component_bytes",
        component_bytes,
    )
    constructions: list[str] = []

    def model_factory(config: QwenImageTrainingConfig) -> QwenImage:
        del config
        constructions.append("dit")
        raise AssertionError("dry-run must not construct the DiT")

    def data_factory(config: QwenImageTrainingConfig) -> _QwenImageBatches:
        del config
        constructions.append("encoders")
        raise AssertionError("dry-run must not construct dataset encoders")

    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = QwenImageLoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_factory,
            data_source_factory=data_factory,
        )
        report = service.dry_run(json.dumps(config.to_mapping()))
        dataset_report = cast("dict[str, object]", report["dataset"])
        assert dataset_report["type"] == "qwen-image-image-caption-folder"
        assert dataset_report["variant"] == "qwen-image"
        assert dataset_report["fixedResolution"] is True
        assert dataset_report["resolution"] == {"height": 16, "width": 16}
        assert dataset_report["contextTokens"] == 512
        assert dataset_report["encodedCache"] == {"state": cache_state}
        precision = cast("dict[str, object]", report["precisionPlan"])
        assert precision["datasetPrecomputeComponents"] == (
            "not loaded on encoded cache hit"
            if cache_state == "hit"
            else "Wan 2.1 VAE float32, then Qwen2.5-VL-7B bfloat16"
        )
        categories = cast(
            "dict[str, int]", cast("dict[str, object]", report["memoryLedger"])["categories"]
        )
        expected_transients = (0, 0) if cache_state == "hit" else (11, 13)
        assert (
            categories["vaeEncoderParametersTransientLowerBound"],
            categories["qwen25VlTextEncoderParametersTransientLowerBound"],
        ) == expected_transients
        assert constructions == []
    finally:
        store.close()


def test_qwen_image_service_builds_dataset_before_loading_dit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = config_mapping(tmp_path / "artifacts")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (16, 16)).save(dataset / "item.png")
    (dataset / "item.txt").write_text("caption", encoding="utf-8")
    mapping.pop("datasetIdentity")
    mapping["baseDtype"] = "float32"
    mapping["gradientCheckpointing"] = False
    mapping["latentShape"] = [1, 16, 1, 2, 2]
    mapping["dataset"] = {
        "type": "qwen-image-image-caption-folder",
        "root": str(dataset.resolve()),
        "resolution": [16, 16],
        "contextTokens": 512,
    }
    events: list[str] = []

    def report_config(config: QwenImageTrainingConfig) -> dict[str, object]:
        assert config.dataset is not None
        return {"dataset": {"errors": list(config.dataset.inspection.errors)}}

    monkeypatch.setattr(
        training_service,
        "qwen_image_capability_report",
        report_config,
    )

    def data_factory(config: QwenImageTrainingConfig) -> _QwenImageBatches:
        assert config.dataset is not None
        events.append("dataset")
        return _service_data_source_factory(config)

    def model_factory(config: QwenImageTrainingConfig) -> QwenImage:
        assert config.dataset is not None
        assert events == ["dataset"]
        events.append("dit")
        return _service_model_factory(config)

    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = QwenImageLoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_factory,
            data_source_factory=data_factory,
        )
        service.create("qwen-image-precompute-order", json.dumps(mapping))
        assert events == ["dataset", "dit"]
    finally:
        store.close()


def test_qwen_image_service_lifecycle_resume_and_exports_decode_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_service, "qwen_image_capability_report", _service_report)
    config = _service_config(tmp_path / "artifacts", cadence=True)
    monkeypatch.setattr(
        training_trainer,
        "qwen_image_model_assembly_plan",
        _objective_assembly_plan,
    )
    root = tmp_path / "checkpoints"
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    baseline_store = TrainingSessionStore(JournalStore(tmp_path / "baseline.sqlite"))
    try:
        service = QwenImageLoRATrainingService(
            store,
            root,
            model_factory=_service_model_factory,
            data_source_factory=_service_data_source_factory,
        )
        initial, report = service.create("qwen-image-session", json.dumps(config.to_mapping()))
        assert report == _service_report(config)
        assert initial.config_digest == training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
            config
        )
        first = service.advance(initial, 1, "train Qwen-Image adapter")
        assert first.handle.step_cursor == 1
        assert first.loss is not None and math.isfinite(first.loss)
        service._evict_runtime(initial.session_id)  # pyright: ignore[reportPrivateUsage]
        resumed = service.advance(first.handle, 1, "resume Qwen-Image adapter")
        assert resumed.handle.step_cursor == 2
        assert resumed.loss is not None and math.isfinite(resumed.loss)

        baseline = QwenImageLoRATrainingService(
            baseline_store,
            tmp_path / "baseline-checkpoints",
            model_factory=_service_model_factory,
            data_source_factory=_service_data_source_factory,
        )
        baseline_initial, _ = baseline.create(
            "qwen-image-baseline", json.dumps(config.to_mapping())
        )
        uninterrupted = baseline.advance(baseline_initial, 2, "train without eviction")
        resumed_state = ContentAddressedCheckpointStore(root).load(
            resumed.handle.checkpoint_manifest_digest
        )
        uninterrupted_state = ContentAddressedCheckpointStore(
            tmp_path / "baseline-checkpoints"
        ).load(uninterrupted.handle.checkpoint_manifest_digest)
        assert resumed.loss == uninterrupted.loss
        assert resumed_state.step_cursor == uninterrupted_state.step_cursor == 2
        assert resumed_state.data_cursor == uninterrupted_state.data_cursor == 2
        assert_tree_equal(resumed_state.adapter, uninterrupted_state.adapter)
        assert_tree_equal(resumed_state.optimizer, uninterrupted_state.optimizer)
        assert_tree_equal(resumed_state.rng, uninterrupted_state.rng)

        model = _service_model_factory(config)
        targets = resolve_lora_targets(model, config.rank, family="qwen-image")
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
        key_map = qwen_image_lora_key_map(model_keys)
        expected_keys = {
            f"transformer.{target.module_path}.{suffix}"
            for target in targets
            for suffix in ("lora_A.weight", "lora_B.weight", "alpha")
        }
        cadence = (
            root / "exports" / "cadence" / initial.session_id / "step-000000000001.safetensors"
        )
        cadence_source = load_safetensors_header(cadence)
        assert cadence_source.metadata() == {
            "dinkster_config_digest": first.handle.config_digest,
            "dinkster_qwen_image_variant": "qwen-image",
            "dinkster_runtime_identity": QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY,
            "dinkster_session_id": first.handle.session_id,
            "dinkster_step_cursor": "1",
        }
        first_state = ContentAddressedCheckpointStore(root).load(
            first.handle.checkpoint_manifest_digest
        )
        sources = [(cadence_source, first_state, "fp32")]
        for dtype in ("fp16", "bf16", "fp32"):
            path_value, export_digest = service.export_lora(
                resumed.handle,
                json.dumps({"path": f"qwen-image-{dtype}.safetensors", "dtype": dtype}),
            )
            path = Path(path_value)
            assert export_digest == digest_bytes(path.read_bytes())
            source = load_safetensors_header(path)
            assert source.metadata() == {
                "dinkster_checkpoint_manifest_digest": resumed.handle.checkpoint_manifest_digest,
                "dinkster_config_digest": resumed.handle.config_digest,
                "dinkster_qwen_image_variant": "qwen-image",
                "dinkster_runtime_identity": QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY,
                "dinkster_session_id": resumed.handle.session_id,
                "dinkster_step_cursor": "2",
            }
            sources.append((source, resumed_state, dtype))

        for source, checkpoint, dtype in sources:
            assert set(source.keys()) == expected_keys
            geometries = {key: source.entry(key).geometry for key in source.keys()}
            decoded = decode_lora(geometries, key_map)
            assert decoded.unmatched == ()
            assert decoded.diagnostics == ()
            assert {patch.key for patch in decoded.patches} == {
                f"diffusion_model.{target.module_path}.weight" for target in targets
            }
            tensors = load_tensors(source.path)
            expected_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
                dtype
            ]
            assert all(tensor.dtype == expected_dtype for tensor in tensors.values())
            if dtype == "fp32":
                for target in targets:
                    stem = f"transformer.{target.module_path}"
                    down = tensors[f"{stem}.lora_A.weight"]
                    up = tensors[f"{stem}.lora_B.weight"]
                    alpha = tensors[f"{stem}.alpha"]
                    expected = (
                        checkpoint.adapter[f"{target.target_id}.up"]
                        @ checkpoint.adapter[f"{target.target_id}.down"]
                    ) * (config.alpha / config.rank)
                    actual = (up @ down) * (float(alpha.item()) / config.rank)
                    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        malformed = {key: cadence_source.entry(key).geometry for key in cadence_source.keys()}
        key = next(key for key in malformed if key.endswith(".lora_A.weight"))
        malformed[key + ".renamed"] = malformed.pop(key)
        rejected = decode_lora(malformed, key_map)
        assert rejected.unmatched or rejected.diagnostics
        assert {patch.key for patch in rejected.patches} != {
            f"diffusion_model.{target.module_path}.weight" for target in targets
        }

        assert service.complete(resumed.handle) == resumed.handle
        assert baseline.complete(uninterrupted.handle) == uninterrupted.handle
    finally:
        store.close()
        baseline_store.close()


def test_qwen_image_export_rejects_wrong_mapping_and_adapter_coverage(tmp_path: Path) -> None:
    config = _service_config(tmp_path)
    model = _service_model_factory(config)
    targets = resolve_lora_targets(model, config.rank, family="qwen-image")
    attachment = TrainableAttachment.attach(
        model, rank=config.rank, alpha=config.alpha, seed=config.seed, family="qwen-image"
    )
    source = LoraExportSource(
        checkpoint_manifest_digest=None,
        session_id="strict-export",
        config_digest="blake3:" + "1" * 64,
        step_cursor=0,
        adapter=attachment.state_dict(),
    )
    model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    missing_native = tuple(
        key for key in model_keys if key != f"diffusion_model.{targets[0].module_path}.weight"
    )
    with pytest.raises(CheckpointError, match="do not match the native DiT state dict"):
        _qwen_image_tensors(source, config, torch.float32, targets, missing_native)

    missing_adapter = dict(source.adapter)
    missing_adapter.pop(next(iter(missing_adapter)))
    with pytest.raises(CheckpointError, match="keys differ during export"):
        _qwen_image_tensors(
            replace(source, adapter=missing_adapter), config, torch.float32, targets, model_keys
        )

    unknown_adapter = dict(source.adapter)
    unknown_adapter["qwen-image/dit/unknown/weight.down"] = torch.zeros(2, 2)
    with pytest.raises(CheckpointError, match="keys differ during export"):
        _qwen_image_tensors(
            replace(source, adapter=unknown_adapter), config, torch.float32, targets, model_keys
        )
