"""Executable proofs for Flux2 configuration, assembly, and LoRA attachment."""

from __future__ import annotations

import json
import math
from dataclasses import replace
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
    FLOAT32,
    FLUX2_DEV,
    FLUX2_DEV_CONFIG,
    FLUX2_KLEIN_4B,
    FLUX2_KLEIN_4B_CONFIG,
    FLUX2_KLEIN_9B,
    FLUX2_KLEIN_9B_CONFIG,
    KLEIN_QWEN3_4B_CONFIG,
    KLEIN_QWEN3_8B_CONFIG,
    MISTRAL3_24B_PRUNED_CONFIG,
    FluxConfig,
    decode_lora,
    load_safetensors_header,
    native_unet_key_map,
)
from dinkster_inference_torch import Flux, load_tensors
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    FLUX2_DEV_TRAINING_RUNTIME_IDENTITY,
    FLUX2_DEV_TRAINING_SNAPSHOT_DIGEST,
    FLUX2_KLEIN_4B_TRAINING_RUNTIME_IDENTITY,
    FLUX2_KLEIN_4B_TRAINING_SNAPSHOT_DIGEST,
    FLUX2_KLEIN_9B_TRAINING_RUNTIME_IDENTITY,
    FLUX2_KLEIN_9B_TRAINING_SNAPSHOT_DIGEST,
    ContentAddressedCheckpointStore,
    Flux2LoRATrainer,
    Flux2LoRATrainingService,
    Flux2PreparedBatch,
    Flux2TrainingConfig,
    TrainableAttachment,
    TrainingConfigError,
    default_flux2_model_factory,
    flux2_capability_report,
)
from dinkster_training_torch.attachment import resolve_lora_targets
from dinkster_training_torch.config import TrainingArtifactSource
from dinkster_training_torch.trainer import flux2_training_sigma_table
from PIL import Image

_VARIANTS = (
    ("flux2-dev", 15360, True),
    ("flux2-klein-9b", 12288, False),
    ("flux2-klein-4b", 7680, False),
)


def _artifact(path: Path, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "digest": digest_bytes(data), "size": len(data)}


def config_mapping(tmp_path: Path, variant: str = "flux2-dev") -> dict[str, object]:
    width = dict((name, context) for name, context, _ in _VARIANTS)[variant]
    mapping: dict[str, object] = {
        "schemaVersion": 1,
        "family": "flux2",
        "variant": variant,
        "ditState": _artifact(tmp_path / "dit.safetensors", b"flux2-dit"),
        "textEncoderState": _artifact(tmp_path / "text.safetensors", b"flux2-text"),
        "vaeState": _artifact(tmp_path / "vae.safetensors", b"flux2-vae"),
        "datasetIdentity": "blake3:" + "6" * 64,
        "device": "cpu",
        "baseDtype": "bfloat16",
        "rank": 2,
        "alpha": 2.0,
        "loraTargets": ["attention.qkv_proj", "mlp.projections"],
        "learningRate": 0.0001,
        "weightDecay": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "optimizer": "adamw",
        "gradientAccumulationSteps": 1,
        "gradientCheckpointing": True,
        "seed": 1234,
        "latentShape": [1, 128, 8, 8],
        "contextShape": [1, 512, width],
    }
    if variant == "flux2-dev":
        mapping["guidance"] = 3.5
    return mapping


def tiny_flux2_model(*, guidance_embed: bool) -> Flux:
    model = Flux(
        FluxConfig(
            in_channels=128,
            out_channels=128,
            vec_in_dim=None,
            context_in_dim=24,
            hidden_size=16,
            depth=2,
            depth_single_blocks=2,
            num_heads=2,
            axes_dim=(2, 2, 2, 2),
            theta=2000,
            patch_size=1,
            mlp_ratio=3.0,
            qkv_bias=False,
            guidance_embed=guidance_embed,
            txt_ids_dims=(3,),
            global_modulation=True,
            mlp_silu_act=True,
            ops_bias=False,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) * 0.001)
    return model


def _identity_digest(config: Flux2TrainingConfig) -> str:
    payload = json.dumps(config.identity_mapping(), sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return digest_bytes(payload)


def _opaque_artifact_header(_source: TrainingArtifactSource, _family: str) -> object:
    return object()


@pytest.mark.parametrize(("variant", "width", "guidance_embed"), _VARIANTS)
def test_flux2_config_round_trips_variant_contract(
    tmp_path: Path, variant: str, width: int, guidance_embed: bool
) -> None:
    config = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path, variant))

    assert Flux2TrainingConfig.from_mapping(config.to_mapping()) == config
    assert config.context_shape == (1, 512, width)
    assert config.latent_shape == (1, 128, 8, 8)
    assert (config.guidance is not None) == guidance_embed
    assert ("guidance" in config.to_mapping()) == guidance_embed


def test_flux2_config_has_stable_identity(tmp_path: Path) -> None:
    config = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path))

    assert _identity_digest(config) == (
        "blake3:55bca1cc9581973190f80537d5444f449d6bbbb95f6b8df11915ed734dfcd49d"
    )
    moved = config.to_mapping()
    for field in ("ditState", "textEncoderState", "vaeState"):
        artifact = cast("dict[str, object]", moved[field])
        artifact["path"] = str(tmp_path / "moved" / field)
    moved["checkpointInterval"] = 10
    moved["loraExportInterval"] = 5
    moved["syncDigestInterval"] = 3
    assert _identity_digest(Flux2TrainingConfig.from_mapping(moved)) == _identity_digest(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schemaVersion", True, "schemaVersion must be an integer"),
        ("variant", [], "unsupported Flux2 training variant"),
        ("rank", True, "rank must be an integer"),
        ("latentShape", [1, 128.0, 8, 8], "latentShape.*integer"),
        ("latentShape", [1, 128, False, 8], "latentShape.*integer"),
        ("contextShape", [1, 512.0, 15360], "contextShape.*integer"),
        ("guidance", True, "guidance must be a finite number"),
        ("loraTargets", ["attention.qkv_proj"], "loraTargets must be"),
        (
            "textEncoderState",
            {"path": "/tmp/text", "digest": "blake3:" + "0" * 64, "size": True},
            "textEncoderState.size must be an integer",
        ),
    ),
)
def test_flux2_config_rejects_malformed_wire_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    mapping = config_mapping(tmp_path)
    mapping[field] = value

    with pytest.raises(TrainingConfigError, match=message):
        Flux2TrainingConfig.from_mapping(mapping)


def test_flux2_config_rejects_unsupported_variants_and_klein_guidance(tmp_path: Path) -> None:
    unsupported = config_mapping(tmp_path)
    unsupported["variant"] = "flux2-fill"
    with pytest.raises(TrainingConfigError, match="unsupported Flux2 training variant"):
        Flux2TrainingConfig.from_mapping(unsupported)

    klein = config_mapping(tmp_path, "flux2-klein-4b")
    klein["guidance"] = 3.5
    with pytest.raises(TrainingConfigError, match="does not accept guidance"):
        Flux2TrainingConfig.from_mapping(klein)


def test_flux2_config_direct_construction_requires_normalized_exact_types(tmp_path: Path) -> None:
    dev = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path))
    klein = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path, "flux2-klein-4b"))

    with pytest.raises(TrainingConfigError, match="unsupported Flux2 training variant"):
        replace(dev, variant=cast('Literal["flux2-dev"]', "flux2-fill"))
    with pytest.raises(TrainingConfigError, match="rank must be an integer"):
        replace(dev, rank=cast("int", True))
    with pytest.raises(TrainingConfigError, match="alpha must be a normalized float"):
        replace(dev, alpha=cast("float", 2))
    with pytest.raises(TrainingConfigError, match="latentShape must be"):
        replace(dev, latent_shape=cast("tuple[int, int, int, int]", (1, 128.0, 8, 8)))
    with pytest.raises(TrainingConfigError, match="contextShape must be"):
        replace(dev, context_shape=cast("tuple[int, int, int]", (1, 512, 15360.0)))
    with pytest.raises(TrainingConfigError, match="guidance must be a normalized float"):
        replace(dev, guidance=cast("float", 3))
    with pytest.raises(TrainingConfigError, match="requires guidance"):
        replace(dev, guidance=None)
    with pytest.raises(TrainingConfigError, match="does not accept guidance"):
        replace(klein, guidance=3.5)
    with pytest.raises(TrainingConfigError, match="textEncoderState.size must be an integer"):
        replace(
            dev,
            text_encoder_state=replace(dev.text_encoder_state, size=cast("int", True)),
        )


@pytest.mark.parametrize("guidance_embed", (True, False))
def test_flux2_targets_cover_real_double_and_single_stream_projections(
    guidance_embed: bool,
) -> None:
    targets = resolve_lora_targets(
        tiny_flux2_model(guidance_embed=guidance_embed), 2, family="flux2"
    )

    assert len(targets) == 20
    assert all(target.target_id == f"flux2/dit/{target.module_path}/weight" for target in targets)
    expected = {
        f"double_blocks.{block}.{stream}_{group}.{projection}"
        for block in range(2)
        for stream in ("img", "txt")
        for group, projections in (("attn", ("qkv", "proj")), ("mlp", ("0", "2")))
        for projection in projections
    }
    expected.update(
        f"single_blocks.{block}.{projection}"
        for block in range(2)
        for projection in ("linear1", "linear2")
    )
    assert {target.module_path for target in targets} == expected


@pytest.mark.parametrize("guidance_embed", (True, False))
def test_flux2_attachment_is_deterministic_and_detaches_cleanly(guidance_embed: bool) -> None:
    first_model = tiny_flux2_model(guidance_embed=guidance_embed)
    second_model = tiny_flux2_model(guidance_embed=guidance_embed)
    frozen = {name: value.detach().clone() for name, value in first_model.state_dict().items()}
    first = TrainableAttachment.attach(first_model, rank=2, alpha=2.0, seed=91, family="flux2")
    second = TrainableAttachment.attach(second_model, rank=2, alpha=2.0, seed=91, family="flux2")

    assert all(not parameter.requires_grad for parameter in first_model.parameters())
    assert all(
        parameter.dtype == torch.float32 and parameter.requires_grad
        for parameter in first.parameters()
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    first_state = first.state_dict()
    assert not torch.equal(
        first_state["flux2/dit/double_blocks.0.img_attn.qkv/weight.down"],
        first_state["flux2/dit/double_blocks.0.txt_attn.qkv/weight.down"],
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


def _planned_flux2(
    config: FluxConfig,
    family: object,
    text_config: object,
    *,
    dtype: object = BFLOAT16,
) -> SimpleNamespace:
    diffusion = SimpleNamespace(
        config=config,
        quant={},
        dtypes={"double_blocks.0.img_attn.qkv.weight": dtype},
        component="diffusion",
    )
    text_encoder = SimpleNamespace(config=text_config, quant={}, component="text_encoder")
    vae = SimpleNamespace(
        config=SimpleNamespace(batch_norm_latent=True, latent_channels=128),
        quant={},
        component="vae",
    )
    return SimpleNamespace(
        family=family,
        diffusion=diffusion,
        text_encoder=text_encoder,
        vae=vae,
    )


_PLAN_VARIANTS = (
    ("flux2-dev", FLUX2_DEV_CONFIG, FLUX2_DEV, MISTRAL3_24B_PRUNED_CONFIG),
    ("flux2-klein-9b", FLUX2_KLEIN_9B_CONFIG, FLUX2_KLEIN_9B, KLEIN_QWEN3_8B_CONFIG),
    ("flux2-klein-4b", FLUX2_KLEIN_4B_CONFIG, FLUX2_KLEIN_4B, KLEIN_QWEN3_4B_CONFIG),
)


def test_flux2_model_plan_rejects_wrong_artifact_digest(tmp_path: Path) -> None:
    mapping = config_mapping(tmp_path)
    dit = cast("dict[str, object]", mapping["ditState"])
    dit["digest"] = "blake3:" + "0" * 64
    config = Flux2TrainingConfig.from_mapping(mapping)

    with pytest.raises(AssetIntegrityError, match="digest_mismatch"):
        training_trainer.flux2_model_assembly_plan(config)


@pytest.mark.parametrize(
    ("variant", "expected_config", "expected_family", "text_config"), _PLAN_VARIANTS
)
def test_flux2_model_plan_uses_matching_complete_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expected_config: FluxConfig,
    expected_family: object,
    text_config: object,
) -> None:
    config = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path, variant))
    plan = _planned_flux2(expected_config, expected_family, text_config)
    calls: list[dict[str, object]] = []

    def fake_artifact_header(
        source: TrainingArtifactSource, family: str
    ) -> tuple[TrainingArtifactSource, str]:
        return source, family

    monkeypatch.setattr(training_trainer, "_artifact_header", fake_artifact_header)

    def fake_plan(**sources: object) -> object:
        calls.append(sources)
        return plan

    monkeypatch.setattr(training_trainer, "plan_flux2_assembly", fake_plan)

    assert training_trainer.flux2_model_assembly_plan(config) is plan
    assert [set(call) for call in calls] == [{"diffusion", "text_encoder", "vae"}]
    headers = cast("tuple[tuple[TrainingArtifactSource, str], ...]", tuple(calls[0].values()))
    assert all(value[1] == "Flux2" for value in headers)


@pytest.mark.parametrize(
    ("plan", "message"),
    (
        (
            _planned_flux2(FLUX2_KLEIN_4B_CONFIG, FLUX2_KLEIN_4B, KLEIN_QWEN3_4B_CONFIG),
            "does not match Flux2",
        ),
        (
            _planned_flux2(FLUX2_DEV_CONFIG, FLUX2_DEV, MISTRAL3_24B_PRUNED_CONFIG, dtype=FLOAT32),
            "storage must be",
        ),
    ),
)
def test_flux2_model_plan_rejects_wrong_variant_and_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: object,
    message: str,
) -> None:
    config = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path))

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_artifact_header", _opaque_artifact_header)
    monkeypatch.setattr(training_trainer, "plan_flux2_assembly", fake_plan)

    with pytest.raises(ValueError, match=message):
        training_trainer.flux2_model_assembly_plan(config)


def test_flux2_model_plan_rejects_quantized_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_flux2(FLUX2_DEV_CONFIG, FLUX2_DEV, MISTRAL3_24B_PRUNED_CONFIG)
    plan.text_encoder.quant = {"layers.0.mlp": object()}

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_artifact_header", _opaque_artifact_header)
    monkeypatch.setattr(training_trainer, "plan_flux2_assembly", fake_plan)

    with pytest.raises(ValueError, match="quantized Flux2 training components"):
        training_trainer.flux2_model_assembly_plan(config)


def test_flux2_model_plan_rejects_wrong_vae_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_flux2(FLUX2_DEV_CONFIG, FLUX2_DEV, MISTRAL3_24B_PRUNED_CONFIG)
    plan.vae.config.latent_channels = 16

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_artifact_header", _opaque_artifact_header)
    monkeypatch.setattr(training_trainer, "plan_flux2_assembly", fake_plan)

    with pytest.raises(ValueError, match="128-channel packed batch-norm KL VAE"):
        training_trainer.flux2_model_assembly_plan(config)


def test_flux2_factory_strict_loads_only_dit_at_selected_dtype_and_freezes_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = config_mapping(tmp_path)
    mapping["baseDtype"] = "float16"
    config = Flux2TrainingConfig.from_mapping(mapping)
    plan = _planned_flux2(FLUX2_DEV_CONFIG, FLUX2_DEV, MISTRAL3_24B_PRUNED_CONFIG)
    model = tiny_flux2_model(guidance_embed=True)
    calls: list[object] = []

    def fake_model_plan(_config: Flux2TrainingConfig) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "flux2_model_assembly_plan", fake_model_plan)

    def load_component(component: object, factory: object) -> Flux:
        calls.extend((component, factory))
        return model

    monkeypatch.setattr(training_trainer, "load_planned_component", load_component)

    loaded = default_flux2_model_factory(config)

    assert loaded is model
    assert calls == [plan.diffusion, Flux]
    assert all(parameter.dtype == torch.float16 for parameter in loaded.parameters())
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


class _Flux2BatchSource:
    def __init__(self, batch: Flux2PreparedBatch) -> None:
        self.batch_value = batch

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> Flux2PreparedBatch:
        del cursor, generator, device
        return self.batch_value


def _objective_model(context_width: int, *, guidance_embed: bool) -> Flux:
    generator = torch.Generator().manual_seed(7)
    model = Flux(
        FluxConfig(
            in_channels=128,
            out_channels=128,
            vec_in_dim=None,
            context_in_dim=context_width,
            hidden_size=16,
            depth=1,
            depth_single_blocks=1,
            num_heads=2,
            axes_dim=(2, 2, 2, 2),
            theta=2000,
            patch_size=1,
            mlp_ratio=2.0,
            qkv_bias=False,
            guidance_embed=guidance_embed,
            txt_ids_dims=(3,),
            global_modulation=True,
            mlp_silu_act=True,
            ops_bias=False,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model


@pytest.mark.parametrize(("variant", "_width", "_guidance_embed"), _VARIANTS)
def test_flux2_training_schedule_is_bit_exact_inference_reference(
    variant: str, _width: int, _guidance_embed: bool
) -> None:
    from dinkster_inference_torch import schedules

    expected = schedules._FluxRef(2.02, 10000).sigmas  # pyright: ignore[reportPrivateUsage]
    actual = flux2_training_sigma_table(cast("object", variant))  # pyright: ignore[reportArgumentType]

    assert len(actual) == 10000
    assert torch.equal(actual, expected)
    assert actual[0].item() == 0.0007533399038948119
    assert actual[-1].item() == 1.0


@pytest.mark.parametrize(("variant", "width", "guidance_embed"), _VARIANTS)
def test_flux2_flow_objective_has_finite_loss_and_lora_gradients(
    tmp_path: Path, variant: str, width: int, guidance_embed: bool
) -> None:
    config = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path, variant))
    generator = torch.Generator().manual_seed(19)
    batch = Flux2PreparedBatch(
        latents=torch.randn(config.latent_shape, generator=generator),
        context=torch.randn(config.context_shape, generator=generator),
        sigma_indices=torch.tensor([31]),
        noise=torch.randn(config.latent_shape, generator=generator),
    )
    model = _objective_model(width, guidance_embed=guidance_embed)
    seen_guidance: list[torch.Tensor | None] = []
    seen_noisy: list[torch.Tensor] = []
    original_forward = model.forward

    def capture_forward(
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seen_guidance.append(guidance)
        seen_noisy.append(x.detach().clone())
        return original_forward(x, timesteps, context, y, guidance, image_position_ids)

    model.forward = capture_forward  # pyright: ignore[reportAttributeAccessIssue]
    trainer = Flux2LoRATrainer(config, model, _Flux2BatchSource(batch))
    with torch.no_grad():
        for name, parameter in trainer.attachment.named_parameters():
            if name.endswith(".up"):
                parameter.fill_(0.01)

    loss = trainer.train_step()

    assert torch.isfinite(torch.tensor(loss))
    gradients = [parameter.grad for parameter in trainer.attachment.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    sigma = flux2_training_sigma_table(cast("object", variant))[31]  # pyright: ignore[reportArgumentType]
    expected_noisy = ((1.0 - sigma) * batch.latents + sigma * cast("torch.Tensor", batch.noise)).to(
        next(model.parameters()).dtype
    )
    assert seen_noisy and all(torch.equal(value, expected_noisy) for value in seen_noisy)
    if guidance_embed:
        assert seen_guidance
        assert all(
            value is not None and torch.equal(value, torch.tensor([3.5])) for value in seen_guidance
        )
    else:
        assert seen_guidance and all(value is None for value in seen_guidance)


@pytest.mark.parametrize(("variant", "width", "guidance_embed"), _VARIANTS)
def test_flux2_same_seed_repeats_noise_timestep_and_update(
    tmp_path: Path, variant: str, width: int, guidance_embed: bool
) -> None:
    config = replace(
        Flux2TrainingConfig.from_mapping(config_mapping(tmp_path, variant)),
        gradient_checkpointing=False,
    )
    generator = torch.Generator().manual_seed(23)
    batch = Flux2PreparedBatch(
        torch.randn(config.latent_shape, generator=generator),
        torch.randn(config.context_shape, generator=generator),
    )
    first = Flux2LoRATrainer(
        config, _objective_model(width, guidance_embed=guidance_embed), _Flux2BatchSource(batch)
    )
    second = Flux2LoRATrainer(
        config, _objective_model(width, guidance_embed=guidance_embed), _Flux2BatchSource(batch)
    )
    with torch.no_grad():
        for trainer in (first, second):
            for name, parameter in trainer.attachment.named_parameters():
                if name.endswith(".up"):
                    parameter.fill_(0.01)

    first_loss = first.train_step()
    second_loss = second.train_step()

    assert first_loss == second_loss
    assert all(
        torch.equal(value, second.attachment.state_dict()[name])
        for name, value in first.attachment.state_dict().items()
    )


def _service_model_factory(config: Flux2TrainingConfig) -> Flux:
    generator = torch.Generator().manual_seed(31)
    model = Flux(
        FluxConfig(
            in_channels=128,
            out_channels=128,
            vec_in_dim=None,
            context_in_dim=config.context_shape[2],
            hidden_size=16,
            depth=1,
            depth_single_blocks=1,
            num_heads=2,
            axes_dim=(2, 2, 2, 2),
            theta=2000,
            patch_size=1,
            mlp_ratio=2.0,
            qkv_bias=False,
            guidance_embed=config.variant == "flux2-dev",
            txt_ids_dims=(3,),
            global_modulation=True,
            mlp_silu_act=True,
            ops_bias=False,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model


def _service_data_source_factory(config: Flux2TrainingConfig) -> _Flux2BatchSource:
    generator = torch.Generator().manual_seed(37)
    return _Flux2BatchSource(
        Flux2PreparedBatch(
            latents=torch.randn(config.latent_shape, generator=generator),
            context=torch.randn(config.context_shape, generator=generator),
            sigma_indices=torch.tensor([17]),
            noise=torch.randn(config.latent_shape, generator=generator),
        )
    )


def _service_report(config: Flux2TrainingConfig) -> dict[str, object]:
    return {
        "family": config.family,
        "variant": config.variant,
        "dataset": {"type": "injected", "errors": []},
    }


_FLUX2_SERVICE_CASES = (
    (
        "flux2-dev",
        FLUX2_DEV_TRAINING_RUNTIME_IDENTITY,
        FLUX2_DEV_TRAINING_SNAPSHOT_DIGEST,
    ),
    (
        "flux2-klein-9b",
        FLUX2_KLEIN_9B_TRAINING_RUNTIME_IDENTITY,
        FLUX2_KLEIN_9B_TRAINING_SNAPSHOT_DIGEST,
    ),
    (
        "flux2-klein-4b",
        FLUX2_KLEIN_4B_TRAINING_RUNTIME_IDENTITY,
        FLUX2_KLEIN_4B_TRAINING_SNAPSHOT_DIGEST,
    ),
)


def test_flux2_service_identities_are_variant_specific() -> None:
    assert len({case[1] for case in _FLUX2_SERVICE_CASES}) == len(_FLUX2_SERVICE_CASES)
    assert len({case[2] for case in _FLUX2_SERVICE_CASES}) == len(_FLUX2_SERVICE_CASES)


@pytest.mark.parametrize(
    ("variant", "runtime_identity", "snapshot_digest"),
    _FLUX2_SERVICE_CASES,
)
def test_flux2_capability_report_resolves_variant_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    runtime_identity: str,
    snapshot_digest: str,
) -> None:
    config = Flux2TrainingConfig.from_mapping(config_mapping(tmp_path, variant))
    model_config = _service_model_factory(config).config

    def assembly_plan(_config: Flux2TrainingConfig) -> object:
        return SimpleNamespace(diffusion=SimpleNamespace(config=model_config))

    monkeypatch.setattr(training_service, "flux2_model_assembly_plan", assembly_plan)

    report = flux2_capability_report(config)

    assert report["trainer"] == runtime_identity
    assert report["variant"] == variant
    assert report["sessionExtensionSnapshotDigest"] == snapshot_digest
    assert report["dataset"] == {
        "type": "injected",
        "statisticsAvailable": False,
        "errors": [],
    }
    targets = cast("list[dict[str, object]]", report["targets"])
    assert len(targets) == 10
    assert all(cast("str", target["targetId"]).startswith("flux2/dit/") for target in targets)
    capabilities = cast("dict[str, object]", report["capabilities"])
    assert capabilities["families"] == ["flux2"]
    assert capabilities["checkpointResume"] is True
    assert capabilities["loraExport"] is True
    memory = cast("dict[str, object]", report["memoryLedger"])
    assert cast("int", memory["deviceMemoryLowerBoundBytes"]) > 0


@pytest.mark.parametrize(
    ("variant", "runtime_identity", "snapshot_digest"),
    _FLUX2_SERVICE_CASES,
)
def test_flux2_service_lifecycle_and_export_decode_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    runtime_identity: str,
    snapshot_digest: str,
) -> None:
    monkeypatch.setattr(training_service, "flux2_capability_report", _service_report)
    config = replace(
        Flux2TrainingConfig.from_mapping(config_mapping(tmp_path / "artifacts", variant)),
        lora_export_interval=1,
    )
    model_config = _service_model_factory(config).config

    def assembly_plan(_config: Flux2TrainingConfig) -> object:
        return SimpleNamespace(diffusion=SimpleNamespace(config=model_config))

    monkeypatch.setattr(training_trainer, "flux2_model_assembly_plan", assembly_plan)
    root = tmp_path / "checkpoints"
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = Flux2LoRATrainingService(
            store,
            root,
            model_factory=_service_model_factory,
            data_source_factory=_service_data_source_factory,
        )
        initial, report = service.create("flux2-session", json.dumps(config.to_mapping()))
        assert initial.session_extension_snapshot_digest == snapshot_digest
        assert report == _service_report(config)

        advanced = service.advance(initial, 1, "train Flux2 adapter")
        assert advanced.handle.step_cursor == 1
        assert advanced.loss is not None and math.isfinite(advanced.loss)
        cadence = (
            root / "exports" / "cadence" / initial.session_id / "step-000000000001.safetensors"
        )
        cadence_source = load_safetensors_header(cadence)
        assert cadence_source.metadata()["dinkster_runtime_identity"] == runtime_identity
        assert cadence_source.metadata()["dinkster_flux2_variant"] == variant

        service._evict_runtime(initial.session_id)  # pyright: ignore[reportPrivateUsage]
        recovered = service.advance(advanced.handle, 1, "resume Flux2 adapter")
        assert recovered.handle.step_cursor == 2
        assert recovered.loss is not None and math.isfinite(recovered.loss)

        path_value, export_digest = service.export_lora(
            recovered.handle,
            json.dumps({"path": f"{variant}.safetensors", "dtype": "fp32"}),
        )
        path = Path(path_value)
        assert export_digest == digest_bytes(path.read_bytes())
        manual_source = load_safetensors_header(path)
        assert manual_source.metadata() == {
            "dinkster_checkpoint_manifest_digest": recovered.handle.checkpoint_manifest_digest,
            "dinkster_config_digest": recovered.handle.config_digest,
            "dinkster_flux2_variant": variant,
            "dinkster_runtime_identity": runtime_identity,
            "dinkster_session_id": recovered.handle.session_id,
            "dinkster_step_cursor": "2",
        }

        model = _service_model_factory(config)
        targets = resolve_lora_targets(model, config.rank, family="flux2")
        expected_keys = {
            f"diffusion_model.{target.module_path}.{suffix}"
            for target in targets
            for suffix in ("lora_A.weight", "lora_B.weight", "alpha")
        }
        key_map = native_unet_key_map(f"diffusion_model.{key}" for key in model.state_dict())
        for source in (cadence_source, manual_source):
            assert set(source.keys()) == expected_keys
            geometries = {key: source.entry(key).geometry for key in source.keys()}
            decoded = decode_lora(geometries, key_map)
            assert decoded.unmatched == ()
            assert decoded.diagnostics == ()
            assert {patch.key for patch in decoded.patches} == {
                f"diffusion_model.{target.module_path}.weight" for target in targets
            }

            renamed = dict(geometries)
            key = next(iter(renamed))
            renamed[key + ".renamed"] = renamed.pop(key)
            malformed = decode_lora(renamed, key_map)
            assert malformed.unmatched or malformed.diagnostics

        checkpoint = ContentAddressedCheckpointStore(root).load(
            recovered.handle.checkpoint_manifest_digest
        )
        tensors = load_tensors(path)
        for target in targets:
            stem = f"diffusion_model.{target.module_path}"
            down = tensors[f"{stem}.lora_A.weight"]
            up = tensors[f"{stem}.lora_B.weight"]
            alpha = tensors[f"{stem}.alpha"]
            expected = (
                checkpoint.adapter[f"{target.target_id}.up"]
                @ checkpoint.adapter[f"{target.target_id}.down"]
            ) * (float(alpha.item()) / config.rank)
            actual = (up @ down) * (float(alpha.item()) / config.rank)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        assert service.complete(recovered.handle) == recovered.handle
    finally:
        store.close()


@pytest.mark.parametrize("cache_state", ("hit", "miss"))
def test_flux2_service_reports_cache_state_without_constructing_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache_state: str
) -> None:
    mapping = config_mapping(tmp_path / "artifacts")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (64, 64)).save(dataset / "item.png")
    (dataset / "item.txt").write_text("caption", encoding="utf-8")
    mapping.pop("datasetIdentity")
    mapping["latentShape"] = [1, 128, 4, 4]
    mapping["dataset"] = {
        "type": "flux2-image-caption-folder",
        "variant": "flux2-dev",
        "root": str(dataset),
        "resolution": [64, 64],
        "contextTokens": 512,
    }
    config = Flux2TrainingConfig.from_mapping(mapping)
    model_config = _service_model_factory(config).config

    def assembly_plan(_config: Flux2TrainingConfig) -> object:
        return SimpleNamespace(diffusion=SimpleNamespace(config=model_config))

    def encoded_cache_state(*_args: object, **_kwargs: object) -> str:
        return cache_state

    def component_bytes(*_args: object, **_kwargs: object) -> dict[str, int]:
        return {
            "vaeEncoderParametersTransientLowerBound": 11,
            "textEncoderParametersTransientLowerBound": 13,
        }

    monkeypatch.setattr(
        training_service,
        "flux2_model_assembly_plan",
        assembly_plan,
    )
    monkeypatch.setattr(training_service, "flux2_encoded_cache_state", encoded_cache_state)
    monkeypatch.setattr(
        training_service,
        "_flux2_component_bytes",
        component_bytes,
    )
    constructions: list[str] = []

    def model_factory(config: Flux2TrainingConfig) -> Flux:
        del config
        constructions.append("dit")
        raise AssertionError("dry-run must not construct the DiT")

    def data_factory(config: Flux2TrainingConfig) -> _Flux2BatchSource:
        del config
        constructions.append("encoders")
        raise AssertionError("dry-run must not construct dataset encoders")

    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = Flux2LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_factory,
            data_source_factory=data_factory,
        )
        report = service.dry_run(json.dumps(config.to_mapping()))
        assert cast("dict[str, object]", report["dataset"])["encodedCache"] == {
            "state": cache_state
        }
        assert constructions == []
    finally:
        store.close()


def test_flux2_service_builds_dataset_before_loading_dit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = config_mapping(tmp_path / "artifacts")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (64, 64)).save(dataset / "item.png")
    (dataset / "item.txt").write_text("caption", encoding="utf-8")
    mapping.pop("datasetIdentity")
    mapping["latentShape"] = [1, 128, 4, 4]
    mapping["dataset"] = {
        "type": "flux2-image-caption-folder",
        "variant": "flux2-dev",
        "root": str(dataset),
        "resolution": [64, 64],
        "contextTokens": 512,
    }
    events: list[str] = []

    def report(config: Flux2TrainingConfig) -> dict[str, object]:
        assert config.dataset is not None
        return {"dataset": {"errors": list(config.dataset.inspection.errors)}}

    def data_factory(config: Flux2TrainingConfig) -> _Flux2BatchSource:
        assert config.dataset is not None
        events.append("dataset")
        return _service_data_source_factory(config)

    def model_factory(config: Flux2TrainingConfig) -> Flux:
        assert events == ["dataset"]
        events.append("dit")
        return _service_model_factory(config)

    monkeypatch.setattr(training_service, "flux2_capability_report", report)
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = Flux2LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_factory,
            data_source_factory=data_factory,
        )
        service.create("flux2-precompute-order", json.dumps(mapping))
        assert events == ["dataset", "dit"]
    finally:
        store.close()
