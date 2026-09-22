"""Executable proofs for classic Flux configuration and LoRA attachment."""

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
    CLIP_L_TEXT_CONFIG,
    FLOAT32,
    FLUX_DEV,
    FLUX_DEV_CONFIG,
    FLUX_SCHNELL,
    FLUX_SCHNELL_CONFIG,
    T5_XXL_CONFIG,
    FluxConfig,
    decode_lora,
    load_safetensors_header,
    native_unet_key_map,
)
from dinkster_inference_torch import Flux, load_tensors
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    FLUX_DEV_TRAINING_RUNTIME_IDENTITY,
    FLUX_DEV_TRAINING_SNAPSHOT_DIGEST,
    FLUX_SCHNELL_TRAINING_RUNTIME_IDENTITY,
    FLUX_SCHNELL_TRAINING_SNAPSHOT_DIGEST,
    ContentAddressedCheckpointStore,
    FluxLoRATrainer,
    FluxLoRATrainingService,
    FluxPreparedBatch,
    FluxTrainingConfig,
    TrainableAttachment,
    TrainingConfigError,
    default_flux_model_factory,
    flux_capability_report,
)
from dinkster_training_torch.attachment import resolve_lora_targets
from PIL import Image


def _artifact(path: Path, data: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "digest": digest_bytes(data), "size": len(data)}


def config_mapping(tmp_path: Path, variant: str = "flux1-dev") -> dict[str, object]:
    mapping: dict[str, object] = {
        "schemaVersion": 1,
        "family": "flux",
        "variant": variant,
        "ditState": _artifact(tmp_path / "dit.safetensors", b"flux-dit"),
        "clipLState": _artifact(tmp_path / "clip-l.safetensors", b"clip-l"),
        "t5xxlState": _artifact(tmp_path / "t5xxl.safetensors", b"t5xxl"),
        "vaeState": _artifact(tmp_path / "vae.safetensors", b"vae"),
        "datasetIdentity": "blake3:" + "4" * 64,
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
        "latentShape": [1, 16, 8, 8],
        "contextShape": [1, 8, 4096],
        "pooledShape": [1, 768],
    }
    if variant == "flux1-dev":
        mapping["guidance"] = 3.5
    return mapping


def tiny_flux_model(*, guidance_embed: bool) -> Flux:
    model = Flux(
        FluxConfig(
            in_channels=16,
            out_channels=16,
            vec_in_dim=4,
            context_in_dim=8,
            hidden_size=12,
            depth=2,
            depth_single_blocks=2,
            num_heads=2,
            axes_dim=(2, 2, 2),
            mlp_ratio=2.0,
            guidance_embed=guidance_embed,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) * 0.001)
    return model


class _FluxBatchSource:
    def __init__(self, batch: FluxPreparedBatch) -> None:
        self.batch_value = batch

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> FluxPreparedBatch:
        del cursor, generator, device
        return self.batch_value


@pytest.mark.parametrize("variant", ["flux1-dev", "flux1-schnell"])
def test_flux_flow_objective_has_finite_loss_and_lora_gradients(
    tmp_path: Path, variant: str
) -> None:
    config = replace(
        FluxTrainingConfig.from_mapping(config_mapping(tmp_path, variant)),
        checkpointing_mode="blockNonReentrant",
    )
    model = Flux(
        FluxConfig(
            in_channels=16,
            out_channels=16,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=12,
            depth=1,
            depth_single_blocks=1,
            num_heads=2,
            axes_dim=(2, 2, 2),
            mlp_ratio=2.0,
            guidance_embed=variant == "flux1-dev",
        )
    )
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    batch = FluxPreparedBatch(
        latents=torch.randn(config.latent_shape, generator=generator),
        context=torch.randn(config.context_shape, generator=generator),
        pooled=torch.randn(config.pooled_shape, generator=generator),
        sigma_indices=torch.tensor([17]),
        noise=torch.randn(config.latent_shape, generator=generator),
    )
    trainer = FluxLoRATrainer(config, model, _FluxBatchSource(batch))
    with torch.no_grad():
        for name, parameter in trainer.attachment.named_parameters():
            if name.endswith(".up"):
                parameter.fill_(0.01)

    loss = trainer.train_step()

    assert torch.isfinite(torch.tensor(loss))
    gradients = [parameter.grad for parameter in trainer.attachment.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_flux_training_sigma_tables_match_variant_sampling_contracts(tmp_path: Path) -> None:
    batch = FluxPreparedBatch(
        torch.zeros(1, 16, 8, 8),
        torch.zeros(1, 8, 4096),
        torch.zeros(1, 768),
    )
    for variant, expected_count, expected_first in (
        ("flux1-dev", 10000, 0.00031575115281157196),
        ("flux1-schnell", 1000, 0.0010000000474974513),
    ):
        variant_path = tmp_path / variant
        variant_path.mkdir()
        config = replace(
            FluxTrainingConfig.from_mapping(config_mapping(variant_path, variant)),
            gradient_checkpointing=False,
        )
        model = Flux(
            FluxConfig(
                in_channels=16,
                out_channels=16,
                vec_in_dim=768,
                context_in_dim=4096,
                hidden_size=12,
                depth=1,
                depth_single_blocks=1,
                num_heads=2,
                axes_dim=(2, 2, 2),
                mlp_ratio=2.0,
                guidance_embed=variant == "flux1-dev",
            )
        )
        trainer = FluxLoRATrainer(config, model, _FluxBatchSource(batch))
        table = cast("torch.Tensor", trainer.__dict__["_sigma_table"])
        assert len(table) == expected_count
        assert table[0].item() == expected_first
        assert table[-1].item() == 1.0


def _identity_digest(config: FluxTrainingConfig) -> str:
    payload = json.dumps(config.identity_mapping(), sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return digest_bytes(payload)


@pytest.mark.parametrize("variant", ["flux1-dev", "flux1-schnell"])
def test_flux_config_round_trips_with_variant_guidance_contract(
    tmp_path: Path, variant: str
) -> None:
    config = FluxTrainingConfig.from_mapping(config_mapping(tmp_path, variant))

    assert FluxTrainingConfig.from_mapping(config.to_mapping()) == config
    assert (config.guidance is not None) == (variant == "flux1-dev")
    assert ("guidance" in config.to_mapping()) == (variant == "flux1-dev")


def test_flux_config_round_trips_dataset_identity(tmp_path: Path) -> None:
    mapping = config_mapping(tmp_path)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (64, 64)).save(dataset / "item.png")
    (dataset / "item.txt").write_text("caption", encoding="utf-8")
    mapping.pop("datasetIdentity")
    mapping["contextShape"] = [1, 256, 4096]
    mapping["dataset"] = {
        "type": "flux-image-caption-folder",
        "variant": "flux1-dev",
        "root": str(dataset),
        "resolution": [64, 64],
        "contextTokens": 256,
        "encodedCacheRoot": str(tmp_path / "cache"),
    }

    config = FluxTrainingConfig.from_mapping(mapping)

    assert config.dataset is not None
    assert config.dataset_identity == config.dataset.digest
    assert FluxTrainingConfig.from_mapping(config.to_mapping()) == config
    identity_dataset = cast("dict[str, object]", config.identity_mapping()["dataset"])
    assert "encodedCacheRoot" not in identity_dataset
    with pytest.raises(TrainingConfigError, match="normalized Flux dataset settings"):
        replace(
            config,
            dataset=replace(config.dataset, context_tokens=cast("int", True)),
        )


def test_flux_config_has_stable_identity(tmp_path: Path) -> None:
    config = FluxTrainingConfig.from_mapping(config_mapping(tmp_path))

    assert _identity_digest(config) == (
        "blake3:fef8a543c1d584a511766e7a86edefc3806060281d65cddacf8be4ca41a0918d"
    )
    moved = config.to_mapping()
    for field in ("ditState", "clipLState", "t5xxlState", "vaeState"):
        artifact = cast("dict[str, object]", moved[field])
        artifact["path"] = str(tmp_path / "moved" / field)
    moved["checkpointInterval"] = 10
    moved["loraExportInterval"] = 5
    moved["syncDigestInterval"] = 3
    assert _identity_digest(FluxTrainingConfig.from_mapping(moved)) == _identity_digest(config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schemaVersion", True, "schemaVersion must be an integer"),
        ("variant", [], "unsupported Flux training variant"),
        ("rank", True, "rank must be an integer"),
        ("latentShape", [1, 16.0, 8, 8], "latentShape.*integer"),
        ("latentShape", [1, 16, False, 8], "latentShape.*integer"),
        ("latentShape", [1, 16, 7, 8], "latentShape must be.*even height"),
        ("guidance", True, "guidance must be a finite number"),
        ("loraTargets", ["attention.qkv_proj"], "loraTargets must be"),
        (
            "ditState",
            {"path": "/tmp/dit", "digest": "blake3:" + "0" * 64, "size": True},
            "ditState.size must be an integer",
        ),
    ],
)
def test_flux_config_rejects_malformed_wire_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    mapping = config_mapping(tmp_path)
    mapping[field] = value

    with pytest.raises(TrainingConfigError, match=message):
        FluxTrainingConfig.from_mapping(mapping)


def test_flux_config_rejects_unsupported_variants_and_schnell_guidance(tmp_path: Path) -> None:
    unsupported = config_mapping(tmp_path)
    unsupported["variant"] = "flux-kontext"
    with pytest.raises(TrainingConfigError, match="unsupported Flux training variant"):
        FluxTrainingConfig.from_mapping(unsupported)

    schnell = config_mapping(tmp_path, "flux1-schnell")
    schnell["guidance"] = None
    with pytest.raises(TrainingConfigError, match="does not accept guidance"):
        FluxTrainingConfig.from_mapping(schnell)


def test_flux_config_direct_construction_requires_normalized_exact_types(tmp_path: Path) -> None:
    dev = FluxTrainingConfig.from_mapping(config_mapping(tmp_path))
    schnell = FluxTrainingConfig.from_mapping(config_mapping(tmp_path, "flux1-schnell"))

    with pytest.raises(TrainingConfigError, match="unsupported Flux training variant"):
        replace(dev, variant=cast('Literal["flux1-dev"]', "flux2-dev"))
    with pytest.raises(TrainingConfigError, match="rank must be an integer"):
        replace(dev, rank=cast("int", True))
    with pytest.raises(TrainingConfigError, match="alpha must be a normalized float"):
        replace(dev, alpha=cast("float", 2))
    with pytest.raises(TrainingConfigError, match="latentShape must be"):
        replace(dev, latent_shape=cast("tuple[int, int, int, int]", (1, 16.0, 8, 8)))
    with pytest.raises(TrainingConfigError, match="guidance must be a normalized float"):
        replace(dev, guidance=cast("float", 3))
    with pytest.raises(TrainingConfigError, match="requires guidance"):
        replace(dev, guidance=None)
    with pytest.raises(TrainingConfigError, match="does not accept guidance"):
        replace(schnell, guidance=3.5)
    with pytest.raises(TrainingConfigError, match="ditState.path must be"):
        replace(dev, dit_state=replace(dev.dit_state, path="relative.safetensors"))
    with pytest.raises(TrainingConfigError, match="ditState.size must be an integer"):
        replace(dev, dit_state=replace(dev.dit_state, size=cast("int", True)))


@pytest.mark.parametrize("guidance_embed", [False, True])
def test_flux_targets_cover_double_and_single_stream_projections(guidance_embed: bool) -> None:
    targets = resolve_lora_targets(tiny_flux_model(guidance_embed=guidance_embed), 2, family="flux")

    assert len(targets) == 20
    assert all(target.target_id == f"flux/dit/{target.module_path}/weight" for target in targets)
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


def test_flux_attachment_is_deterministic_float32_and_detaches_cleanly() -> None:
    first_model = tiny_flux_model(guidance_embed=True)
    second_model = tiny_flux_model(guidance_embed=True)
    frozen = {name: value.detach().clone() for name, value in first_model.state_dict().items()}
    first = TrainableAttachment.attach(first_model, rank=2, alpha=2.0, seed=91, family="flux")
    second = TrainableAttachment.attach(second_model, rank=2, alpha=2.0, seed=91, family="flux")

    assert all(not parameter.requires_grad for parameter in first_model.parameters())
    assert all(
        parameter.dtype == torch.float32 and parameter.requires_grad
        for parameter in first.parameters()
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    first_state = first.state_dict()
    assert not torch.equal(
        first_state["flux/dit/double_blocks.0.img_attn.qkv/weight.down"],
        first_state["flux/dit/double_blocks.0.txt_attn.qkv/weight.down"],
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


def _planned_flux(
    config: FluxConfig, family: object, *, dtype: object = BFLOAT16
) -> SimpleNamespace:
    diffusion = SimpleNamespace(
        config=config,
        quant={},
        dtypes={"double_blocks.0.img_attn.qkv.weight": dtype},
        component="diffusion",
    )
    clip_l = SimpleNamespace(config=CLIP_L_TEXT_CONFIG, quant={}, component="clip_l")
    t5xxl = SimpleNamespace(config=T5_XXL_CONFIG, quant={}, component="t5xxl")
    vae = SimpleNamespace(config=object(), quant={}, component="vae")
    return SimpleNamespace(
        family=family,
        diffusion=diffusion,
        clip_l=clip_l,
        t5xxl=t5xxl,
        vae=vae,
        qwen3_2b=None,
    )


def test_flux_model_plan_rejects_wrong_artifact_digest(tmp_path: Path) -> None:
    mapping = config_mapping(tmp_path)
    dit = cast("dict[str, object]", mapping["ditState"])
    dit["digest"] = "blake3:" + "0" * 64
    config = FluxTrainingConfig.from_mapping(mapping)

    with pytest.raises(AssetIntegrityError, match="digest_mismatch"):
        training_trainer.flux_model_assembly_plan(config)


@pytest.mark.parametrize(
    ("variant", "expected_config", "expected_family"),
    [
        ("flux1-dev", FLUX_DEV_CONFIG, FLUX_DEV),
        ("flux1-schnell", FLUX_SCHNELL_CONFIG, FLUX_SCHNELL),
    ],
)
def test_flux_model_plan_uses_matching_complete_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expected_config: FluxConfig,
    expected_family: object,
) -> None:
    config = FluxTrainingConfig.from_mapping(config_mapping(tmp_path, variant))
    plan = _planned_flux(expected_config, expected_family)
    calls: list[dict[str, object]] = []

    def fake_header(_source: object) -> object:
        return object()

    def fake_plan(**sources: object) -> object:
        calls.append(sources)
        return plan

    monkeypatch.setattr(training_trainer, "_flux_artifact_header", fake_header)
    monkeypatch.setattr(training_trainer, "plan_flux_assembly", fake_plan)

    assert training_trainer.flux_model_assembly_plan(config) is plan
    assert [set(call) for call in calls] == [{"diffusion", "clip_l", "t5xxl", "vae"}]


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (_planned_flux(FLUX_SCHNELL_CONFIG, FLUX_SCHNELL), "does not match Flux"),
        (_planned_flux(FLUX_DEV_CONFIG, FLUX_DEV, dtype=FLOAT32), "storage must be"),
    ],
)
def test_flux_model_plan_rejects_wrong_variant_and_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: object,
    message: str,
) -> None:
    config = FluxTrainingConfig.from_mapping(config_mapping(tmp_path))

    def fake_header(_source: object) -> object:
        return object()

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_flux_artifact_header", fake_header)
    monkeypatch.setattr(training_trainer, "plan_flux_assembly", fake_plan)

    with pytest.raises(ValueError, match=message):
        training_trainer.flux_model_assembly_plan(config)


def test_flux_model_plan_rejects_quantized_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FluxTrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_flux(FLUX_DEV_CONFIG, FLUX_DEV)
    plan.clip_l.quant = {"text_model.encoder.layers.0.mlp.fc1": object()}

    def fake_header(_source: object) -> object:
        return object()

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_flux_artifact_header", fake_header)
    monkeypatch.setattr(training_trainer, "plan_flux_assembly", fake_plan)

    with pytest.raises(ValueError, match="quantized Flux training components"):
        training_trainer.flux_model_assembly_plan(config)


def test_flux_model_plan_rejects_alternative_text_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = FluxTrainingConfig.from_mapping(config_mapping(tmp_path))
    plan = _planned_flux(FLUX_DEV_CONFIG, FLUX_DEV)
    plan.qwen3_2b = SimpleNamespace(config=object(), quant={}, component="qwen3_2b")

    def fake_header(_source: object) -> object:
        return object()

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_flux_artifact_header", fake_header)
    monkeypatch.setattr(training_trainer, "plan_flux_assembly", fake_plan)

    with pytest.raises(ValueError, match="requires CLIP-L and T5-XXL"):
        training_trainer.flux_model_assembly_plan(config)


def test_flux_factory_strict_loads_only_dit_at_selected_dtype_and_freezes_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = config_mapping(tmp_path)
    mapping["baseDtype"] = "float16"
    config = FluxTrainingConfig.from_mapping(mapping)
    plan = _planned_flux(FLUX_DEV_CONFIG, FLUX_DEV)
    model = tiny_flux_model(guidance_embed=True)
    calls: list[object] = []

    def fake_assembly_plan(_config: FluxTrainingConfig) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "flux_model_assembly_plan", fake_assembly_plan)

    def load_component(component: object, factory: object) -> Flux:
        calls.extend((component, factory))
        return model

    monkeypatch.setattr(training_trainer, "load_planned_component", load_component)

    loaded = default_flux_model_factory(config)

    assert loaded is model
    assert calls == [plan.diffusion, Flux]
    assert all(parameter.dtype == torch.float16 for parameter in loaded.parameters())
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def _service_model_factory(config: FluxTrainingConfig) -> Flux:
    generator = torch.Generator().manual_seed(19)
    model = Flux(
        FluxConfig(
            in_channels=16,
            out_channels=16,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=12,
            depth=1,
            depth_single_blocks=1,
            num_heads=2,
            axes_dim=(2, 2, 2),
            mlp_ratio=2.0,
            guidance_embed=config.variant == "flux1-dev",
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model


def _service_data_source_factory(config: FluxTrainingConfig) -> _FluxBatchSource:
    generator = torch.Generator().manual_seed(23)
    return _FluxBatchSource(
        FluxPreparedBatch(
            latents=torch.randn(config.latent_shape, generator=generator),
            context=torch.randn(config.context_shape, generator=generator),
            pooled=torch.randn(config.pooled_shape, generator=generator),
            sigma_indices=torch.tensor([17]),
            noise=torch.randn(config.latent_shape, generator=generator),
        )
    )


def _service_report(config: FluxTrainingConfig) -> dict[str, object]:
    return {
        "family": config.family,
        "variant": config.variant,
        "dataset": {"type": "injected", "errors": []},
    }


_FLUX_SERVICE_CASES = (
    (
        "flux1-dev",
        FLUX_DEV_TRAINING_RUNTIME_IDENTITY,
        FLUX_DEV_TRAINING_SNAPSHOT_DIGEST,
    ),
    (
        "flux1-schnell",
        FLUX_SCHNELL_TRAINING_RUNTIME_IDENTITY,
        FLUX_SCHNELL_TRAINING_SNAPSHOT_DIGEST,
    ),
)


@pytest.mark.parametrize(
    ("variant", "runtime_identity", "snapshot_digest"),
    _FLUX_SERVICE_CASES,
)
def test_flux_capability_report_resolves_variant_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    runtime_identity: str,
    snapshot_digest: str,
) -> None:
    config = FluxTrainingConfig.from_mapping(config_mapping(tmp_path, variant))
    model_config = _service_model_factory(config).config

    def assembly_plan(_config: FluxTrainingConfig) -> object:
        return SimpleNamespace(diffusion=SimpleNamespace(config=model_config))

    monkeypatch.setattr(
        training_service,
        "flux_model_assembly_plan",
        assembly_plan,
    )

    report = flux_capability_report(config)

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
    assert all(cast("str", target["targetId"]).startswith("flux/dit/") for target in targets)
    capabilities = cast("dict[str, object]", report["capabilities"])
    assert capabilities["families"] == ["flux"]
    assert capabilities["checkpointResume"] is True
    assert capabilities["loraExport"] is True
    memory = cast("dict[str, object]", report["memoryLedger"])
    assert cast("int", memory["deviceMemoryLowerBoundBytes"]) > 0


@pytest.mark.parametrize(
    ("variant", "runtime_identity", "snapshot_digest"),
    _FLUX_SERVICE_CASES,
)
def test_flux_service_lifecycle_and_export_decode_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    runtime_identity: str,
    snapshot_digest: str,
) -> None:
    monkeypatch.setattr(training_service, "flux_capability_report", _service_report)
    config = replace(
        FluxTrainingConfig.from_mapping(config_mapping(tmp_path / "artifacts", variant)),
        lora_export_interval=1,
    )
    model_config = _service_model_factory(config).config

    def assembly_plan(_config: FluxTrainingConfig) -> object:
        return SimpleNamespace(diffusion=SimpleNamespace(config=model_config))

    monkeypatch.setattr(
        training_trainer,
        "flux_model_assembly_plan",
        assembly_plan,
    )
    root = tmp_path / "checkpoints"
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = FluxLoRATrainingService(
            store,
            root,
            model_factory=_service_model_factory,
            data_source_factory=_service_data_source_factory,
        )
        initial, report = service.create("flux-session", json.dumps(config.to_mapping()))
        assert initial.session_extension_snapshot_digest == snapshot_digest
        assert report == _service_report(config)

        advanced = service.advance(initial, 1, "train Flux adapter")
        assert advanced.handle.step_cursor == 1
        assert advanced.loss is not None and math.isfinite(advanced.loss)
        cadence = (
            root / "exports" / "cadence" / initial.session_id / "step-000000000001.safetensors"
        )
        cadence_metadata = load_safetensors_header(cadence).metadata()
        assert cadence_metadata["dinkster_runtime_identity"] == runtime_identity
        assert cadence_metadata["dinkster_flux_variant"] == variant

        service._evict_runtime(initial.session_id)  # pyright: ignore[reportPrivateUsage]
        recovered = service.advance(advanced.handle, 1, "resume Flux adapter")
        assert recovered.handle.step_cursor == 2
        assert recovered.loss is not None and math.isfinite(recovered.loss)

        path_value, export_digest = service.export_lora(
            recovered.handle,
            json.dumps({"path": f"{variant}.safetensors", "dtype": "fp32"}),
        )
        path = Path(path_value)
        assert export_digest == digest_bytes(path.read_bytes())
        source = load_safetensors_header(path)
        assert source.metadata() == {
            "dinkster_checkpoint_manifest_digest": recovered.handle.checkpoint_manifest_digest,
            "dinkster_config_digest": recovered.handle.config_digest,
            "dinkster_flux_variant": variant,
            "dinkster_runtime_identity": runtime_identity,
            "dinkster_session_id": recovered.handle.session_id,
            "dinkster_step_cursor": "2",
        }

        model = _service_model_factory(config)
        targets = resolve_lora_targets(model, config.rank, family="flux")
        expected_keys = {
            f"diffusion_model.{target.module_path}.{suffix}"
            for target in targets
            for suffix in ("lora_A.weight", "lora_B.weight", "alpha")
        }
        assert set(source.keys()) == expected_keys
        decoded = decode_lora(
            {key: source.entry(key).geometry for key in source.keys()},
            native_unet_key_map(f"diffusion_model.{key}" for key in model.state_dict()),
        )
        assert decoded.unmatched == ()
        assert decoded.diagnostics == ()
        assert {patch.key for patch in decoded.patches} == {
            f"diffusion_model.{target.module_path}.weight" for target in targets
        }

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


def test_flux_service_builds_dataset_before_loading_dit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = config_mapping(tmp_path / "artifacts")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    Image.new("RGB", (64, 64)).save(dataset / "item.png")
    (dataset / "item.txt").write_text("caption", encoding="utf-8")
    mapping.pop("datasetIdentity")
    mapping["contextShape"] = [1, 256, 4096]
    mapping["dataset"] = {
        "type": "flux-image-caption-folder",
        "variant": "flux1-dev",
        "root": str(dataset),
        "resolution": [64, 64],
        "contextTokens": 256,
    }
    events: list[str] = []

    def report(config: FluxTrainingConfig) -> dict[str, object]:
        assert config.dataset is not None
        return {"dataset": {"errors": list(config.dataset.inspection.errors)}}

    def data_factory(config: FluxTrainingConfig) -> _FluxBatchSource:
        assert config.dataset is not None
        events.append("dataset")
        return _service_data_source_factory(config)

    def model_factory(config: FluxTrainingConfig) -> Flux:
        assert events == ["dataset"]
        events.append("dit")
        return _service_model_factory(config)

    monkeypatch.setattr(training_service, "flux_capability_report", report)
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = FluxLoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_factory,
            data_source_factory=data_factory,
        )
        service.create("flux-precompute-order", json.dumps(mapping))
        assert events == ["dataset", "dit"]
    finally:
        store.close()
