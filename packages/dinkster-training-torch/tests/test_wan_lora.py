"""Executable proofs for Wan configuration and LoRA attachment."""

from __future__ import annotations

import gc
import json
import math
import os
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import patch

import dinkster_training_torch.service as training_service
import dinkster_training_torch.trainer as training_trainer
import pytest
import torch
from dinkster_assets import AssetIntegrityError, digest_bytes
from dinkster_inference import (
    FLOAT8_E4M3,
    FLOAT16,
    FLOAT32,
    WAN21_I2V_14B,
    WAN21_SIGMAS,
    WAN21_T2V_1_3B,
    WAN21_T2V_14B,
    WAN22_I2V_14B,
    WAN22_TI2V_5B,
    Wan21Config,
    decode_lora,
    load_safetensors_header,
    native_unet_key_map,
)
from dinkster_inference_torch import load_tensors
from dinkster_inference_torch.wan21_model import Wan21Model
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    WAN22_I2V_TRAINING_RUNTIME_IDENTITY,
    WAN22_I2V_TRAINING_SNAPSHOT_DIGEST,
    WAN22_T2V_TRAINING_RUNTIME_IDENTITY,
    WAN22_T2V_TRAINING_SNAPSHOT_DIGEST,
    WAN22_TI2V_TRAINING_RUNTIME_IDENTITY,
    WAN22_TI2V_TRAINING_SNAPSHOT_DIGEST,
    WAN_TRAINING_RUNTIME_IDENTITY,
    WAN_TRAINING_SNAPSHOT_DIGEST,
    ContentAddressedCheckpointStore,
    TrainableAttachment,
    TrainingConfigError,
    WanLoRATrainer,
    WanLoRATrainingService,
    WanPreparedBatch,
    WanTrainingConfig,
    default_wan_model_factory,
    wan_capability_report,
)
from dinkster_training_torch.attachment import resolve_lora_targets
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def _artifact(path: Path, data: bytes) -> dict[str, object]:
    path.write_bytes(data)
    return {"path": str(path), "digest": digest_bytes(data), "size": len(data)}


def config_mapping(tmp_path: Path) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "family": "wan",
        "variant": "wan21-t2v",
        "ditState": _artifact(tmp_path / "dit.safetensors", b"dit"),
        "umt5xxlState": _artifact(tmp_path / "umt5xxl.safetensors", b"umt5"),
        "vaeState": _artifact(tmp_path / "vae.safetensors", b"vae"),
        "datasetIdentity": "blake3:" + "4" * 64,
        "device": "cpu",
        "baseDtype": "bfloat16",
        "rank": 2,
        "alpha": 2.0,
        "loraTargets": ["attention.qkvo", "ffn.projections"],
        "learningRate": 0.0001,
        "weightDecay": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "optimizer": "adamw",
        "gradientAccumulationSteps": 1,
        "gradientCheckpointing": True,
        "seed": 1234,
        "latentShape": [1, 16, 3, 8, 8],
        "contextShape": [1, 8, 4096],
    }


def wan22_config_mapping(
    tmp_path: Path,
    variant: str,
    *,
    expert: str | None = None,
) -> dict[str, object]:
    mapping = config_mapping(tmp_path)
    mapping["variant"] = variant
    if variant == "wan22-ti2v-5b":
        mapping["latentShape"] = [1, 48, 3, 8, 8]
    if expert is not None:
        mapping["expert"] = expert
    return mapping


def tiny_wan_model(*, ti2v: bool = False) -> Wan21Model:
    model = Wan21Model(
        Wan21Config(
            model_type="ti2v" if ti2v else "t2v",
            in_channels=48 if ti2v else 16,
            hidden_size=12,
            ffn_hidden_size=16,
            num_heads=2,
            num_layers=2,
            text_dim=8,
            time_freq_dim=4,
            out_channels=48 if ti2v else 16,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.fill_((index + 1) * 0.001)
    return model


def _identity_digest(config: WanTrainingConfig) -> str:
    payload = json.dumps(config.identity_mapping(), sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    return digest_bytes(payload)


def test_wan_config_round_trips_and_has_stable_identity(tmp_path: Path) -> None:
    config = WanTrainingConfig.from_mapping(config_mapping(tmp_path))
    round_tripped = WanTrainingConfig.from_mapping(config.to_mapping())

    assert round_tripped == config
    assert _identity_digest(round_tripped) == _identity_digest(config)
    assert _identity_digest(config) == (
        "blake3:11150b89c55386288875e7465b7fb89cbeada2ecf5998906754ba2e019922741"
    )
    runtime_mapping = config.to_mapping()
    runtime_mapping["checkpointInterval"] = 7
    runtime_mapping["loraExportInterval"] = 5
    runtime_mapping["syncDigestInterval"] = 3
    cast("dict[str, object]", runtime_mapping["ditState"])["path"] = str(
        tmp_path / "relocated.safetensors"
    )
    assert _identity_digest(WanTrainingConfig.from_mapping(runtime_mapping)) == _identity_digest(
        config
    )
    changed = config.to_mapping()
    changed["datasetIdentity"] = "blake3:" + "5" * 64
    assert _identity_digest(WanTrainingConfig.from_mapping(changed)) != _identity_digest(config)


@pytest.mark.parametrize(
    ("variant", "expert", "timestep_range"),
    [
        ("wan22-ti2v-5b", None, None),
        ("wan22-t2v-14b", "high-noise", [1000, 875]),
        ("wan22-t2v-14b", "low-noise", [875, 0]),
        ("wan22-i2v-14b", "high-noise", [1000, 900]),
        ("wan22-i2v-14b", "low-noise", [900, 0]),
    ],
)
def test_wan22_config_round_trips_one_expert_and_timestep_range(
    tmp_path: Path,
    variant: str,
    expert: str | None,
    timestep_range: list[int] | None,
) -> None:
    config = WanTrainingConfig.from_mapping(wan22_config_mapping(tmp_path, variant, expert=expert))
    mapping = config.to_mapping()

    assert mapping["variant"] == variant
    assert mapping.get("expert") == expert
    assert mapping.get("timestepRange") == timestep_range
    assert config.identity_mapping().get("timestepRange") == timestep_range
    assert WanTrainingConfig.from_mapping(mapping) == config


def test_wan22_expert_and_boundary_change_identity(tmp_path: Path) -> None:
    high = WanTrainingConfig.from_mapping(
        wan22_config_mapping(tmp_path, "wan22-t2v-14b", expert="high-noise")
    )
    low = WanTrainingConfig.from_mapping(
        wan22_config_mapping(tmp_path, "wan22-t2v-14b", expert="low-noise")
    )

    assert _identity_digest(high) != _identity_digest(low)


@pytest.mark.parametrize("variant", ["wan22-t2v-14b", "wan22-i2v-14b"])
def test_wan22_14b_config_requires_exactly_one_expert(tmp_path: Path, variant: str) -> None:
    mapping = wan22_config_mapping(tmp_path, variant)
    with pytest.raises(TrainingConfigError, match="expert must be"):
        WanTrainingConfig.from_mapping(mapping)

    mapping["expert"] = "both"
    with pytest.raises(TrainingConfigError, match="expert must be"):
        WanTrainingConfig.from_mapping(mapping)


@pytest.mark.parametrize("expert", [[], 1])
def test_wan22_config_rejects_non_string_expert(tmp_path: Path, expert: object) -> None:
    mapping = wan22_config_mapping(tmp_path, "wan22-t2v-14b")
    mapping["expert"] = expert

    with pytest.raises(TrainingConfigError, match="expert must be"):
        WanTrainingConfig.from_mapping(mapping)


def test_wan22_config_rejects_mismatched_timestep_range(tmp_path: Path) -> None:
    mapping = wan22_config_mapping(tmp_path, "wan22-t2v-14b", expert="high-noise")
    mapping["timestepRange"] = [875, 0]

    with pytest.raises(TrainingConfigError, match="timestepRange must be"):
        WanTrainingConfig.from_mapping(mapping)

    i2v = wan22_config_mapping(tmp_path, "wan22-i2v-14b", expert="high-noise")
    i2v["timestepRange"] = [1000, 875]
    with pytest.raises(
        TrainingConfigError,
        match=r"timestepRange must be \[1000, 900\].*wan22-i2v-14b.*high-noise",
    ):
        WanTrainingConfig.from_mapping(i2v)


@pytest.mark.parametrize(
    ("expert", "timestep_range"),
    [
        ("high-noise", [1000.0, 875.0]),
        ("low-noise", [875, False]),
    ],
)
def test_wan22_config_rejects_non_integer_timestep_range(
    tmp_path: Path, expert: str, timestep_range: list[object]
) -> None:
    mapping = wan22_config_mapping(tmp_path, "wan22-t2v-14b", expert=expert)
    mapping["timestepRange"] = timestep_range

    with pytest.raises(TrainingConfigError, match="timestepRange must be"):
        WanTrainingConfig.from_mapping(mapping)


def test_wan_checkpointing_modes_round_trip_and_change_identity(tmp_path: Path) -> None:
    whole = WanTrainingConfig.from_mapping(config_mapping(tmp_path))
    mapping = config_mapping(tmp_path)
    mapping["checkpointingMode"] = "blockNonReentrant"
    block = WanTrainingConfig.from_mapping(mapping)

    assert whole.checkpointing_mode == "wholeModel"
    assert "checkpointingMode" not in whole.to_mapping()
    assert "checkpointingMode" not in whole.identity_mapping()
    assert block.checkpointing_mode == "blockNonReentrant"
    assert block.to_mapping()["checkpointingMode"] == "blockNonReentrant"
    assert block.identity_mapping()["checkpointingMode"] == "blockNonReentrant"
    assert _identity_digest(block) != _identity_digest(whole)

    mapping["checkpointingMode"] = "blockReentrant"
    with pytest.raises(TrainingConfigError, match="checkpointingMode"):
        WanTrainingConfig.from_mapping(mapping)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("family", "wan21", "family must be 'wan'"),
        ("variant", "wan21-i2v", "unsupported Wan training variant"),
        ("variant", "wan22-t2v", "unsupported Wan training variant"),
        ("baseDtype", "float32", "baseDtype must be 'float16' or 'bfloat16'"),
        ("loraExportInterval", -1, "loraExportInterval must be an integer >= 0"),
        (
            "loraTargets",
            ["attention.qkvo"],
            "loraTargets must be \\['attention.qkvo', 'ffn.projections'\\]",
        ),
        (
            "latentShape",
            [1, 32, 3, 8, 8],
            "latentShape must be \\[batch, 16, time, height, width\\]",
        ),
        (
            "contextShape",
            [2, 8, 4096],
            "contextShape must be \\[batch, tokens, 4096\\]",
        ),
    ],
)
def test_wan_config_rejects_unsupported_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    mapping = config_mapping(tmp_path)
    mapping[field] = value
    with pytest.raises(TrainingConfigError, match=message):
        WanTrainingConfig.from_mapping(mapping)


def test_wan_config_direct_construction_guards_single_value_invariants(tmp_path: Path) -> None:
    config = WanTrainingConfig.from_mapping(config_mapping(tmp_path))
    ti2v = WanTrainingConfig.from_mapping(wan22_config_mapping(tmp_path, "wan22-ti2v-5b"))
    expert = WanTrainingConfig.from_mapping(
        wan22_config_mapping(tmp_path, "wan22-t2v-14b", expert="high-noise")
    )

    with pytest.raises(TrainingConfigError, match="unsupported Wan training variant"):
        replace(config, variant=cast('Literal["wan21-t2v"]', "wan21-i2v"))
    with pytest.raises(TrainingConfigError, match=r"latentShape must be \[batch, 16"):
        replace(config, latent_shape=(1, 48, 3, 8, 8))
    with pytest.raises(TrainingConfigError, match=r"latentShape must be \[batch, 48"):
        replace(ti2v, latent_shape=(1, 16, 3, 8, 8))
    with pytest.raises(TrainingConfigError, match=r"latentShape must be \[batch, 16"):
        replace(config, latent_shape=cast("tuple[int, int, int, int, int]", (1, 16.0, 3, 8, 8)))
    with pytest.raises(TrainingConfigError, match="timestepRange must be"):
        replace(expert, timestep_range=cast("tuple[int, int]", (1000.0, 875.0)))
    with pytest.raises(TrainingConfigError, match="baseDtype must be"):
        replace(config, base_dtype=cast('Literal["float16", "bfloat16"]', "float32"))
    with pytest.raises(TrainingConfigError, match="loraTargets must be"):
        replace(
            config,
            lora_targets=cast(
                'tuple[Literal["attention.qkvo"], Literal["ffn.projections"]]',
                ("attention.qkvo", "attention.qkvo"),
            ),
        )
    with pytest.raises(TrainingConfigError, match="checkpointingMode"):
        replace(
            config,
            checkpointing_mode=cast('Literal["blockNonReentrant", "wholeModel"]', "blocks"),
        )


@pytest.mark.parametrize("ti2v", [False, True])
def test_wan_targets_cover_attention_and_ffn_projections(ti2v: bool) -> None:
    targets = resolve_lora_targets(tiny_wan_model(ti2v=ti2v), 2, family="wan")

    assert len(targets) == 20
    assert all(target.target_id == f"wan/dit/{target.module_path}/weight" for target in targets)
    expected = {
        f"blocks.{block}.{group}.{projection}"
        for block in range(2)
        for group, projections in (
            ("self_attn", ("q", "k", "v", "o")),
            ("cross_attn", ("q", "k", "v", "o")),
            ("ffn", ("0", "2")),
        )
        for projection in projections
    }
    assert {target.module_path for target in targets} == expected


def test_wan_attachment_is_deterministic_float32_and_detaches_cleanly() -> None:
    first_model = tiny_wan_model()
    second_model = tiny_wan_model()
    frozen = {name: value.detach().clone() for name, value in first_model.state_dict().items()}
    first = TrainableAttachment.attach(first_model, rank=2, alpha=2.0, seed=91, family="wan")
    second = TrainableAttachment.attach(second_model, rank=2, alpha=2.0, seed=91, family="wan")

    assert all(not parameter.requires_grad for parameter in first_model.parameters())
    assert all(
        parameter.dtype == torch.float32 and parameter.requires_grad
        for parameter in first.parameters()
    )
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    first_state = first.state_dict()
    assert not torch.equal(
        first_state["wan/dit/blocks.0.self_attn.q/weight.down"],
        first_state["wan/dit/blocks.0.self_attn.k/weight.down"],
    )

    target = first.targets[0]
    module = cast("torch.nn.Linear", first_model.get_submodule(target.module_path))
    inputs = torch.linspace(-1.0, 1.0, module.in_features).reshape(1, -1)
    baseline = module(inputs)
    with torch.no_grad():
        dict(first.named_parameters())[f"{target.target_id}.up"].fill_(0.25)
    adapted = module(inputs)
    assert not torch.equal(adapted, baseline)

    first.detach()
    assert torch.equal(module(inputs), baseline)
    assert all(torch.equal(value, frozen[name]) for name, value in first_model.state_dict().items())


def test_wan_model_plan_rejects_wrong_artifact_digest(tmp_path: Path) -> None:
    mapping = config_mapping(tmp_path)
    dit = cast("dict[str, object]", mapping["ditState"])
    dit["digest"] = "blake3:" + "0" * 64
    config = WanTrainingConfig.from_mapping(mapping)

    with pytest.raises(AssetIntegrityError, match="digest_mismatch"):
        training_trainer.wan_model_assembly_plan(config)


def test_wan_model_plan_rejects_non_t2v_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = WanTrainingConfig.from_mapping(config_mapping(tmp_path))
    component = SimpleNamespace(config=WAN21_I2V_14B, quant={})
    plan = SimpleNamespace(
        diffusion=component, umt5xxl=component, vae=component, clip_vision=object()
    )

    def fake_header(_source: object) -> object:
        return object()

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_wan_artifact_header", fake_header)
    monkeypatch.setattr(training_trainer, "plan_wan21_assembly", fake_plan)

    with pytest.raises(ValueError, match="DiT does not match Wan training variant"):
        training_trainer.wan_model_assembly_plan(config)


@pytest.mark.parametrize(
    ("variant", "expert", "expected_config", "planner_name"),
    [
        ("wan22-ti2v-5b", None, WAN22_TI2V_5B, "plan_wan22_assembly"),
        ("wan22-t2v-14b", "high-noise", WAN21_T2V_14B, "plan_wan21_assembly"),
        ("wan22-t2v-14b", "low-noise", WAN21_T2V_14B, "plan_wan21_assembly"),
        ("wan22-i2v-14b", "high-noise", WAN22_I2V_14B, "plan_wan21_assembly"),
        ("wan22-i2v-14b", "low-noise", WAN22_I2V_14B, "plan_wan21_assembly"),
    ],
)
def test_wan22_model_plan_uses_matching_inference_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expert: str | None,
    expected_config: Wan21Config,
    planner_name: str,
) -> None:
    config = WanTrainingConfig.from_mapping(wan22_config_mapping(tmp_path, variant, expert=expert))
    diffusion = SimpleNamespace(
        config=expected_config,
        quant={},
        dtypes={"blocks.0.self_attn.q.weight": FLOAT16},
        component="diffusion",
    )
    sibling = SimpleNamespace(quant={}, component="component")
    plan = SimpleNamespace(
        diffusion=diffusion,
        umt5xxl=sibling,
        vae=sibling,
        clip_vision=None,
    )
    calls: list[str] = []

    def fake_header(_source: object) -> object:
        return object()

    def fake_plan(**_sources: object) -> object:
        calls.append(planner_name)
        return plan

    monkeypatch.setattr(training_trainer, "_wan_artifact_header", fake_header)
    monkeypatch.setattr(training_trainer, planner_name, fake_plan)

    assert training_trainer.wan_model_assembly_plan(config) is plan
    assert calls == [planner_name]


def test_wan_model_plan_accepts_official_mixed_float_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = WanTrainingConfig.from_mapping(config_mapping(tmp_path))
    diffusion = SimpleNamespace(
        config=WAN21_T2V_1_3B,
        quant={},
        dtypes={
            "blocks.0.self_attn.q.weight": FLOAT16,
            "blocks.0.self_attn.k.weight": FLOAT16,
            "blocks.0.self_attn.v.weight": FLOAT16,
            "blocks.0.self_attn.o.weight": FLOAT16,
            "patch_embedding.weight": FLOAT32,
            "patch_embedding.bias": FLOAT32,
        },
        component="diffusion",
    )
    sibling = SimpleNamespace(quant={}, component="component")
    plan = SimpleNamespace(
        diffusion=diffusion,
        umt5xxl=sibling,
        vae=sibling,
        clip_vision=None,
    )

    def fake_header(_source: object) -> object:
        return object()

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_wan_artifact_header", fake_header)
    monkeypatch.setattr(training_trainer, "plan_wan21_assembly", fake_plan)

    assert training_trainer.wan_model_assembly_plan(config) is plan


@pytest.mark.parametrize(
    ("quant", "dtypes", "message"),
    [
        ({"blocks.0.self_attn.q": object()}, {}, "quantized Wan training components"),
        (
            {},
            {"blocks.0.self_attn.q.weight": FLOAT8_E4M3},
            "storage must be float16, bfloat16, or float32, got float8_e4m3fn",
        ),
    ],
)
def test_wan_model_plan_rejects_quantized_and_unsupported_bases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quant: dict[str, object],
    dtypes: dict[str, object],
    message: str,
) -> None:
    config = WanTrainingConfig.from_mapping(config_mapping(tmp_path))
    diffusion = SimpleNamespace(
        config=WAN21_T2V_1_3B,
        quant=quant,
        dtypes=dtypes,
        component="diffusion",
    )
    sibling = SimpleNamespace(quant={}, component="component")
    plan = SimpleNamespace(
        diffusion=diffusion,
        umt5xxl=sibling,
        vae=sibling,
        clip_vision=None,
    )

    def fake_header(_source: object) -> object:
        return object()

    def fake_plan(**_sources: object) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "_wan_artifact_header", fake_header)
    monkeypatch.setattr(training_trainer, "plan_wan21_assembly", fake_plan)

    with pytest.raises(ValueError, match=message):
        training_trainer.wan_model_assembly_plan(config)


def test_wan_factory_loads_selected_dtype_and_freezes_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = config_mapping(tmp_path)
    mapping["baseDtype"] = "float16"
    config = WanTrainingConfig.from_mapping(mapping)
    plan = SimpleNamespace(diffusion=SimpleNamespace(config=WAN21_T2V_1_3B))
    model = tiny_wan_model()

    def selected_plan(_config: WanTrainingConfig) -> object:
        return plan

    def load_component(_component: object, _factory: object) -> Wan21Model:
        return model

    monkeypatch.setattr(training_trainer, "wan_model_assembly_plan", selected_plan)
    monkeypatch.setattr(training_trainer, "load_planned_component", load_component)

    loaded = default_wan_model_factory(config)

    assert loaded is model
    assert all(parameter.dtype == torch.float16 for parameter in loaded.parameters())
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


class _WanBatches:
    def __init__(self, batch: WanPreparedBatch) -> None:
        self.batch_value = batch

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> WanPreparedBatch:
        del cursor, generator
        return WanPreparedBatch(
            latents=self.batch_value.latents.to(device),
            context=self.batch_value.context.to(device),
            sigma_indices=(
                None
                if self.batch_value.sigma_indices is None
                else self.batch_value.sigma_indices.to(device)
            ),
            noise=None if self.batch_value.noise is None else self.batch_value.noise.to(device),
            i2v_conditioning=(
                None
                if self.batch_value.i2v_conditioning is None
                else self.batch_value.i2v_conditioning.to(device)
            ),
        )


def _training_config(
    tmp_path: Path,
    *,
    seed: int = 1234,
    variant: str = "wan21-t2v",
    expert: str | None = None,
    checkpointing_mode: Literal["blockNonReentrant", "wholeModel"] = "wholeModel",
    checkpointing: bool = True,
    device: str = "cpu",
) -> WanTrainingConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    mapping = wan22_config_mapping(tmp_path, variant, expert=expert)
    mapping["seed"] = seed
    mapping["device"] = device
    mapping["gradientCheckpointing"] = checkpointing
    mapping["checkpointingMode"] = checkpointing_mode
    mapping["latentShape"] = [1, 48, 1, 4, 4] if variant == "wan22-ti2v-5b" else [1, 16, 1, 4, 4]
    mapping["contextShape"] = [1, 2, 4096]
    return WanTrainingConfig.from_mapping(mapping)


def _training_model(*, ti2v: bool = False, i2v: bool = False) -> Wan21Model:
    model = Wan21Model(
        Wan21Config(
            model_type="ti2v" if ti2v else ("i2v" if i2v else "t2v"),
            in_channels=48 if ti2v else (36 if i2v else 16),
            hidden_size=12,
            ffn_hidden_size=16,
            num_heads=2,
            num_layers=1,
            text_dim=4096,
            time_freq_dim=4,
            out_channels=48 if ti2v else 16,
        )
    )
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.copy_(
                torch.linspace(-0.02, 0.02, parameter.numel()).reshape(parameter.shape)
                + index * 0.0001
            )
    return model


def test_wan_flow_objective_produces_finite_lora_gradients_and_preserves_base(
    tmp_path: Path,
) -> None:
    config = _training_config(tmp_path)
    model = _training_model()
    batch = WanPreparedBatch(
        latents=torch.linspace(-0.5, 0.5, 256).reshape(config.latent_shape),
        context=torch.linspace(-0.25, 0.25, 8192).reshape(config.context_shape),
        sigma_indices=torch.tensor([499]),
        noise=torch.linspace(0.5, -0.5, 256).reshape(config.latent_shape),
    )
    trainer = WanLoRATrainer(config, model, _WanBatches(batch))
    frozen = {name: value.detach().clone() for name, value in model.state_dict().items()}

    loss = trainer.train_step()

    gradients = [parameter.grad for parameter in trainer.attachment.parameters()]
    assert math.isfinite(loss)
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
    assert all(parameter.dtype == torch.float32 for parameter in trainer.attachment.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())
    assert all(torch.equal(value, frozen[name]) for name, value in model.state_dict().items())


def test_wan22_ti2v_flow_objective_trains_48_channel_latents(tmp_path: Path) -> None:
    config = _training_config(tmp_path, variant="wan22-ti2v-5b")
    model = _training_model(ti2v=True)
    batch = WanPreparedBatch(
        latents=torch.linspace(-0.5, 0.5, math.prod(config.latent_shape)).reshape(
            config.latent_shape
        ),
        context=torch.linspace(-0.25, 0.25, math.prod(config.context_shape)).reshape(
            config.context_shape
        ),
        sigma_indices=torch.tensor([499]),
        noise=torch.linspace(0.5, -0.5, math.prod(config.latent_shape)).reshape(
            config.latent_shape
        ),
    )
    trainer = WanLoRATrainer(config, model, _WanBatches(batch))

    loss = trainer.train_step()

    assert math.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in trainer.attachment.parameters()
    )


def test_wan22_i2v_flow_objective_trains_with_first_frame_conditioning(
    tmp_path: Path,
) -> None:
    config = _training_config(
        tmp_path,
        variant="wan22-i2v-14b",
        expert="high-noise",
    )
    model = _training_model(i2v=True)
    batch = WanPreparedBatch(
        latents=torch.linspace(-0.5, 0.5, math.prod(config.latent_shape)).reshape(
            config.latent_shape
        ),
        context=torch.linspace(-0.25, 0.25, math.prod(config.context_shape)).reshape(
            config.context_shape
        ),
        sigma_indices=torch.tensor([700]),
        noise=torch.linspace(0.5, -0.5, math.prod(config.latent_shape)).reshape(
            config.latent_shape
        ),
        i2v_conditioning=torch.linspace(-0.2, 0.2, 320).reshape(1, 20, 1, 4, 4),
    )
    trainer = WanLoRATrainer(config, model, _WanBatches(batch))

    loss = trainer.train_step()

    assert math.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in trainer.attachment.parameters()
    )


def test_wan_flow_training_is_deterministic_for_identically_seeded_trainers(
    tmp_path: Path,
) -> None:
    first_config = _training_config(tmp_path / "first", seed=77)
    second_config = _training_config(tmp_path / "second", seed=77)
    batch = WanPreparedBatch(
        latents=torch.linspace(-1.0, 1.0, 256).reshape(first_config.latent_shape),
        context=torch.linspace(-0.1, 0.1, 8192).reshape(first_config.context_shape),
    )
    first = WanLoRATrainer(first_config, _training_model(), _WanBatches(batch))
    second = WanLoRATrainer(second_config, _training_model(), _WanBatches(batch))

    first_losses = [first.train_step(), first.train_step()]
    second_losses = [second.train_step(), second.train_step()]

    assert first_losses == second_losses
    assert first.data_cursor == second.data_cursor == 2
    for name, value in first.attachment.state_dict().items():
        assert torch.equal(value, second.attachment.state_dict()[name])


@pytest.mark.parametrize(
    ("variant", "boundary", "split_index"),
    [
        ("wan22-t2v-14b", 875, 466),
        ("wan22-i2v-14b", 900, 529),
    ],
)
def test_wan22_expert_objectives_sample_only_their_timestep_windows(
    tmp_path: Path, variant: str, boundary: int, split_index: int
) -> None:
    high_config = _training_config(
        tmp_path / "high",
        seed=77,
        variant=variant,
        expert="high-noise",
    )
    low_config = _training_config(
        tmp_path / "low",
        seed=77,
        variant=variant,
        expert="low-noise",
    )
    batch = WanPreparedBatch(
        latents=torch.zeros(high_config.latent_shape),
        context=torch.zeros(high_config.context_shape),
    )
    high = WanLoRATrainer(high_config, _training_model(), _WanBatches(batch))
    low = WanLoRATrainer(low_config, _training_model(), _WanBatches(batch))
    high_indices = set(
        range(high._sigma_index_start, high._sigma_index_stop)  # pyright: ignore[reportPrivateUsage]
    )
    low_indices = set(
        range(low._sigma_index_start, low._sigma_index_stop)  # pyright: ignore[reportPrivateUsage]
    )
    assert high._sigma_index_start == low._sigma_index_stop == split_index  # pyright: ignore[reportPrivateUsage]
    assert low._sigma_index_start == 0  # pyright: ignore[reportPrivateUsage]
    assert high._sigma_index_stop == len(WAN21_SIGMAS.table or ())  # pyright: ignore[reportPrivateUsage]
    assert high_indices.isdisjoint(low_indices)
    assert high_indices | low_indices == set(range(len(WAN21_SIGMAS.table or ())))
    high_draws = [high._draw_noise(batch, cursor) for cursor in range(32)]  # pyright: ignore[reportPrivateUsage]
    low_draws = [low._draw_noise(batch, cursor) for cursor in range(32)]  # pyright: ignore[reportPrivateUsage]

    high_timesteps = [
        high._sigma_table[int(indices.item())] * WAN21_SIGMAS.multiplier  # pyright: ignore[reportPrivateUsage]
        for indices, _noise in high_draws
    ]
    low_timesteps = [
        low._sigma_table[int(indices.item())] * WAN21_SIGMAS.multiplier  # pyright: ignore[reportPrivateUsage]
        for indices, _noise in low_draws
    ]
    assert all(boundary <= timestep <= 1000 for timestep in high_timesteps)
    assert all(0 <= timestep < boundary for timestep in low_timesteps)
    assert high_timesteps != low_timesteps
    assert all(
        torch.equal(high_noise, low_noise)
        for (_high_indices, high_noise), (_low_indices, low_noise) in zip(
            high_draws, low_draws, strict=True
        )
    )


def test_wan22_exact_boundary_timestep_belongs_only_to_high_expert() -> None:
    table = (0.8, 0.875, 0.9)

    high = training_trainer._wan_expert_sigma_indices(  # pyright: ignore[reportPrivateUsage]
        table, 1000.0, "high-noise", (1000, 875)
    )
    low = training_trainer._wan_expert_sigma_indices(  # pyright: ignore[reportPrivateUsage]
        table, 1000.0, "low-noise", (875, 0)
    )

    assert high == (1, 2)
    assert low == (0,)


@pytest.mark.parametrize(
    ("expert", "sigma_index"),
    [
        ("high-noise", 465),
        ("low-noise", 466),
    ],
)
def test_wan22_expert_objective_rejects_prepared_timestep_outside_window(
    tmp_path: Path, expert: str, sigma_index: int
) -> None:
    config = _training_config(
        tmp_path,
        variant="wan22-t2v-14b",
        expert=expert,
    )
    batch = WanPreparedBatch(
        latents=torch.zeros(config.latent_shape),
        context=torch.zeros(config.context_shape),
        sigma_indices=torch.tensor([sigma_index]),
        noise=torch.ones(config.latent_shape),
    )
    trainer = WanLoRATrainer(config, _training_model(), _WanBatches(batch))

    with pytest.raises(ValueError, match="configured timestep range"):
        trainer.train_step()


def test_wan_block_checkpointing_preserves_modules_and_state_keys(tmp_path: Path) -> None:
    config = _training_config(tmp_path, checkpointing_mode="blockNonReentrant")
    model = _training_model()
    state_keys = tuple(model.state_dict())

    trainer = WanLoRATrainer(
        config,
        model,
        _WanBatches(
            WanPreparedBatch(
                latents=torch.zeros(config.latent_shape),
                context=torch.zeros(config.context_shape),
                sigma_indices=torch.tensor([499]),
                noise=torch.ones(config.latent_shape),
            )
        ),
    )

    assert trainer.model is model
    assert tuple(model.state_dict()) == state_keys
    assert all("forward" in block.__dict__ for block in model.blocks)
    assert "forward" not in model.patch_embedding.__dict__
    assert "forward" not in model.head.__dict__
    with patch("dinkster_training_torch.trainer.checkpoint", wraps=torch_checkpoint) as wrapped:
        trainer.train_step()
    assert wrapped.call_count == len(model.blocks)
    for call in wrapped.call_args_list:
        assert call.kwargs["use_reentrant"] is False
        assert call.kwargs["preserve_rng_state"] is True
    block_ref = weakref.ref(model.blocks[0])
    trainer.attachment.detach()
    del trainer, model
    gc.collect()
    assert block_ref() is None


_GPU_TESTS_ENABLED = os.environ.get("DINKSTER_ENABLE_GPU_TESTS") == "1"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _GPU_TESTS_ENABLED,
    reason="needs DINKSTER_ENABLE_GPU_TESTS=1 and CUDA",
)
def test_wan_checkpointing_modes_have_bit_exact_cuda_loss_and_gradients(
    tmp_path: Path,
) -> None:
    batch = WanPreparedBatch(
        latents=torch.linspace(-0.5, 0.5, 256).reshape(1, 16, 1, 4, 4),
        context=torch.linspace(-0.25, 0.25, 8192).reshape(1, 2, 4096),
        sigma_indices=torch.tensor([499]),
        noise=torch.linspace(0.5, -0.5, 256).reshape(1, 16, 1, 4, 4),
    )

    evidence: list[tuple[float, dict[str, torch.Tensor]]] = []
    for name, mode, enabled in (
        ("block", "blockNonReentrant", True),
        ("whole", "wholeModel", True),
        ("disabled", "wholeModel", False),
    ):
        config = _training_config(
            tmp_path / name,
            checkpointing_mode=cast('Literal["blockNonReentrant", "wholeModel"]', mode),
            checkpointing=enabled,
            device="cuda",
        )
        trainer = WanLoRATrainer(config, _training_model(), _WanBatches(batch))
        loss = trainer.train_step()
        gradients = {
            key: parameter.grad.detach().cpu().clone()
            for key, parameter in trainer.attachment.named_parameters()
            if parameter.grad is not None
        }
        evidence.append((loss, gradients))

    reference_loss, reference_gradients = evidence[0]
    for loss, gradients in evidence[1:]:
        assert loss == reference_loss
        assert gradients.keys() == reference_gradients.keys()
        for key, gradient in gradients.items():
            assert torch.equal(gradient, reference_gradients[key]), key


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _GPU_TESTS_ENABLED,
    reason="needs DINKSTER_ENABLE_GPU_TESTS=1 and CUDA",
)
def test_wan_block_checkpointing_reduces_cuda_step_peak(tmp_path: Path) -> None:
    latent_shape = (1, 16, 1, 32, 32)
    context_shape = (1, 32, 4096)
    batch = WanPreparedBatch(
        latents=torch.linspace(-0.5, 0.5, math.prod(latent_shape)).reshape(latent_shape),
        context=torch.linspace(-0.25, 0.25, math.prod(context_shape)).reshape(context_shape),
        sigma_indices=torch.tensor([499]),
        noise=torch.linspace(0.5, -0.5, math.prod(latent_shape)).reshape(latent_shape),
    )

    def peak(mode: Literal["blockNonReentrant", "wholeModel"], name: str) -> int:
        artifact_root = tmp_path / name
        artifact_root.mkdir()
        mapping = config_mapping(artifact_root)
        mapping["device"] = "cuda"
        mapping["checkpointingMode"] = mode
        mapping["latentShape"] = list(latent_shape)
        mapping["contextShape"] = list(context_shape)
        config = WanTrainingConfig.from_mapping(mapping)
        model = Wan21Model(
            Wan21Config(
                hidden_size=256,
                ffn_hidden_size=1024,
                num_heads=8,
                num_layers=8,
                text_dim=4096,
                time_freq_dim=32,
            )
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(
                    torch.linspace(-0.002, 0.002, parameter.numel()).reshape(parameter.shape)
                )
        trainer = WanLoRATrainer(config, model, _WanBatches(batch))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        trainer.train_step()
        torch.cuda.synchronize()
        result = torch.cuda.max_memory_allocated()
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    block_peak = peak("blockNonReentrant", "block")
    whole_peak = peak("wholeModel", "whole")

    assert block_peak < whole_peak


def _service_report(config: WanTrainingConfig) -> dict[str, object]:
    return {
        "family": config.family,
        "configDigest": "test",
        "dataset": {"type": "injected", "errors": []},
    }


def _service_model_factory(config: WanTrainingConfig) -> Wan21Model:
    return _training_model(
        ti2v=config.variant == "wan22-ti2v-5b",
        i2v=config.variant == "wan22-i2v-14b",
    )


def _service_data_source_factory(config: WanTrainingConfig) -> _WanBatches:
    conditioning = (
        torch.linspace(-0.2, 0.2, 320).reshape(1, 20, 1, 4, 4)
        if config.variant == "wan22-i2v-14b"
        else None
    )
    return _WanBatches(
        WanPreparedBatch(
            latents=torch.linspace(-0.5, 0.5, math.prod(config.latent_shape)).reshape(
                config.latent_shape
            ),
            context=torch.linspace(-0.25, 0.25, math.prod(config.context_shape)).reshape(
                config.context_shape
            ),
            i2v_conditioning=conditioning,
        )
    )


_WAN_SERVICE_CASES = (
    ("wan21-t2v", None, WAN_TRAINING_RUNTIME_IDENTITY, WAN_TRAINING_SNAPSHOT_DIGEST),
    (
        "wan22-ti2v-5b",
        None,
        WAN22_TI2V_TRAINING_RUNTIME_IDENTITY,
        WAN22_TI2V_TRAINING_SNAPSHOT_DIGEST,
    ),
    (
        "wan22-t2v-14b",
        "high-noise",
        WAN22_T2V_TRAINING_RUNTIME_IDENTITY,
        WAN22_T2V_TRAINING_SNAPSHOT_DIGEST,
    ),
    (
        "wan22-i2v-14b",
        "high-noise",
        WAN22_I2V_TRAINING_RUNTIME_IDENTITY,
        WAN22_I2V_TRAINING_SNAPSHOT_DIGEST,
    ),
)


@pytest.mark.parametrize(
    ("variant", "expert", "runtime_identity", "snapshot_digest"),
    _WAN_SERVICE_CASES,
)
def test_wan_capability_report_resolves_selected_dit_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expert: str | None,
    runtime_identity: str,
    snapshot_digest: str,
) -> None:
    config = _training_config(tmp_path / "artifacts", variant=variant, expert=expert)

    def assembly_plan(config: WanTrainingConfig) -> object:
        return SimpleNamespace(
            diffusion=SimpleNamespace(config=_service_model_factory(config).config)
        )

    monkeypatch.setattr(
        training_service,
        "wan_model_assembly_plan",
        assembly_plan,
    )

    report = wan_capability_report(config)

    assert report["trainer"] == runtime_identity
    assert report["device"] == "cpu"
    assert report["variant"] == variant
    assert report["sessionExtensionSnapshotDigest"] == snapshot_digest
    assert report["dataset"] == {
        "type": "injected",
        "statisticsAvailable": False,
        "errors": [],
    }
    assert len(cast("list[object]", report["targets"])) == 10
    capabilities = cast("dict[str, object]", report["capabilities"])
    assert capabilities["families"] == ["wan"]
    assert capabilities["checkpointResume"] is True
    assert capabilities["safePointCancellation"] is True
    assert capabilities["gradientAccumulation"] is True
    assert capabilities["loraExport"] is True
    memory = cast("dict[str, object]", report["memoryLedger"])
    assert memory["estimated"] is True
    assert cast("int", memory["deviceMemoryLowerBoundBytes"]) > 0


def test_training_identity_helpers_reject_unknown_families() -> None:
    malformed = cast("WanTrainingConfig", SimpleNamespace(family="unknown"))

    with pytest.raises(ValueError, match="unsupported training family 'unknown'"):
        training_service._runtime_identity(malformed)  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match="unsupported training family 'unknown'"):
        training_service._snapshot_digest(malformed)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("variant", "expert", "runtime_identity", "snapshot_digest"),
    _WAN_SERVICE_CASES,
)
def test_wan_training_identity_selects_variant_runtime(
    tmp_path: Path,
    variant: str,
    expert: str | None,
    runtime_identity: str,
    snapshot_digest: str,
) -> None:
    config = _training_config(tmp_path, variant=variant, expert=expert)

    assert (
        training_service._runtime_identity(config)  # pyright: ignore[reportPrivateUsage]
        == runtime_identity
    )
    assert (
        training_service._snapshot_digest(config)  # pyright: ignore[reportPrivateUsage]
        == snapshot_digest
    )


@pytest.mark.parametrize(
    ("variant", "expert", "runtime_identity", "snapshot_digest"),
    _WAN_SERVICE_CASES,
)
def test_wan_service_parses_config_and_runs_session_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expert: str | None,
    runtime_identity: str,
    snapshot_digest: str,
) -> None:
    monkeypatch.setattr(training_service, "wan_capability_report", _service_report)
    config = replace(
        _training_config(tmp_path / "artifacts", variant=variant, expert=expert),
        lora_export_interval=1,
    )
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = WanLoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=_service_model_factory,
            data_source_factory=_service_data_source_factory,
        )
        initial, report = service.create("wan-session", json.dumps(config.to_mapping()))
        assert initial.session_extension_snapshot_digest == snapshot_digest
        assert report == _service_report(config)

        advanced = service.advance(initial, 1, "train Wan adapter")
        assert advanced.handle.step_cursor == 1
        assert advanced.loss is not None and math.isfinite(advanced.loss)
        cadence = (
            tmp_path
            / "checkpoints"
            / "exports"
            / "cadence"
            / initial.session_id
            / "step-000000000001.safetensors"
        )
        assert cadence.is_file()
        cadence_metadata = load_safetensors_header(cadence).metadata()
        assert cadence_metadata["dinkster_step_cursor"] == "1"
        assert cadence_metadata["dinkster_runtime_identity"] == runtime_identity
        if variant != "wan21-t2v":
            assert cadence_metadata["dinkster_wan_variant"] == variant
        if expert is not None:
            assert cadence_metadata["dinkster_wan_expert"] == expert
        service._evict_runtime(initial.session_id)  # pyright: ignore[reportPrivateUsage]
        recovered = service.advance(advanced.handle, 1, "resume Wan adapter")
        assert recovered.handle.step_cursor == 2
        assert recovered.loss is not None and math.isfinite(recovered.loss)
        completed = service.complete(recovered.handle)
        assert completed == recovered.handle

        wrong_family = config.to_mapping()
        wrong_family["family"] = "sdxl"
        with pytest.raises(TrainingConfigError, match="family must be 'wan'"):
            service.dry_run(json.dumps(wrong_family))
    finally:
        store.close()


@pytest.mark.parametrize(
    ("dtype", "torch_dtype", "geometry_dtype"),
    [
        ("fp16", torch.float16, "float16"),
        ("bf16", torch.bfloat16, "bfloat16"),
        ("fp32", torch.float32, "float32"),
    ],
)
def test_wan_export_uses_wan_fun_keys_and_round_trips_adapter_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
    torch_dtype: torch.dtype,
    geometry_dtype: str,
) -> None:
    monkeypatch.setattr(training_service, "wan_capability_report", _service_report)
    config = _training_config(tmp_path / "artifacts")
    model_config = _training_model().config

    def assembly_plan(_config: WanTrainingConfig) -> object:
        del _config
        return SimpleNamespace(diffusion=SimpleNamespace(config=model_config))

    monkeypatch.setattr(
        training_trainer,
        "wan_model_assembly_plan",
        assembly_plan,
    )
    root = tmp_path / "checkpoints"
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = WanLoRATrainingService(
            store,
            root,
            model_factory=_service_model_factory,
            data_source_factory=_service_data_source_factory,
        )
        initial, _ = service.create("wan-export", json.dumps(config.to_mapping()))
        final = service.advance(initial, 1, "train Wan adapter").handle
        path_value, export_digest = service.export_lora(
            final, json.dumps({"path": f"wan-{dtype}.safetensors", "dtype": dtype})
        )
        path = Path(path_value)
        assert export_digest == digest_bytes(path.read_bytes())

        source = load_safetensors_header(path)
        assert source.metadata() == {
            "dinkster_checkpoint_manifest_digest": final.checkpoint_manifest_digest,
            "dinkster_config_digest": final.config_digest,
            "dinkster_runtime_identity": WAN_TRAINING_RUNTIME_IDENTITY,
            "dinkster_session_id": final.session_id,
            "dinkster_step_cursor": "1",
        }
        assert {source.entry(key).geometry.dtype.name for key in source.keys()} == {geometry_dtype}

        model = _training_model()
        targets = resolve_lora_targets(model, config.rank, family="wan")
        expected_keys = {
            f"lora_unet__{target.module_path.replace('.', '_')}.{suffix}"
            for target in targets
            for suffix in ("lora_down.weight", "lora_up.weight", "alpha")
        }
        assert set(source.keys()) == expected_keys
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
        decoded = decode_lora(
            {key: source.entry(key).geometry for key in source.keys()},
            native_unet_key_map(model_keys),
        )
        assert decoded.dialect == "wan_fun"
        assert decoded.unmatched == ()
        assert decoded.diagnostics == ()
        assert len(decoded.patches) == len(targets)

        state = ContentAddressedCheckpointStore(root).load(final.checkpoint_manifest_digest)
        tensors = load_tensors(path)
        for target in targets:
            stem = f"lora_unet__{target.module_path.replace('.', '_')}"
            down = tensors[f"{stem}.lora_down.weight"]
            up = tensors[f"{stem}.lora_up.weight"]
            alpha = tensors[f"{stem}.alpha"]
            assert down.shape == (config.rank, target.weight_shape[1])
            assert up.shape == (target.weight_shape[0], config.rank)
            assert down.dtype == up.dtype == alpha.dtype == torch_dtype
            expected = (
                state.adapter[f"{target.target_id}.up"].to(torch_dtype)
                @ state.adapter[f"{target.target_id}.down"].to(torch_dtype)
            ).float() * (float(alpha.float().item()) / config.rank)
            actual = (up @ down).float() * (float(alpha.float().item()) / config.rank)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    finally:
        store.close()


@pytest.mark.parametrize(
    ("variant", "expert", "runtime_identity"),
    [
        ("wan22-ti2v-5b", None, WAN22_TI2V_TRAINING_RUNTIME_IDENTITY),
        ("wan22-t2v-14b", "high-noise", WAN22_T2V_TRAINING_RUNTIME_IDENTITY),
        ("wan22-t2v-14b", "low-noise", WAN22_T2V_TRAINING_RUNTIME_IDENTITY),
        ("wan22-i2v-14b", "high-noise", WAN22_I2V_TRAINING_RUNTIME_IDENTITY),
        ("wan22-i2v-14b", "low-noise", WAN22_I2V_TRAINING_RUNTIME_IDENTITY),
    ],
)
def test_wan22_export_decodes_against_variant_dit_and_records_expert(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expert: str | None,
    runtime_identity: str,
) -> None:
    monkeypatch.setattr(training_service, "wan_capability_report", _service_report)
    config = _training_config(tmp_path / "artifacts", variant=variant, expert=expert)

    def assembly_plan(config: WanTrainingConfig) -> object:
        return SimpleNamespace(
            diffusion=SimpleNamespace(config=_service_model_factory(config).config)
        )

    monkeypatch.setattr(training_trainer, "wan_model_assembly_plan", assembly_plan)
    root = tmp_path / "checkpoints"
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = WanLoRATrainingService(
            store,
            root,
            model_factory=_service_model_factory,
            data_source_factory=_service_data_source_factory,
        )
        initial, _ = service.create("wan22-export", json.dumps(config.to_mapping()))
        final = service.advance(initial, 1, "train Wan 2.2 adapter").handle
        path_value, export_digest = service.export_lora(
            final,
            json.dumps({"path": "wan22.safetensors", "dtype": "fp32"}),
        )
        path = Path(path_value)
        assert export_digest == digest_bytes(path.read_bytes())

        source = load_safetensors_header(path)
        metadata = dict(source.metadata())
        assert metadata["dinkster_runtime_identity"] == runtime_identity
        assert metadata["dinkster_wan_variant"] == variant
        if expert is None:
            assert "dinkster_wan_expert" not in metadata
        else:
            assert metadata["dinkster_wan_expert"] == expert

        model = _service_model_factory(config)
        targets = resolve_lora_targets(model, config.rank, family="wan")
        expected_keys = {
            f"lora_unet__{target.module_path.replace('.', '_')}.{suffix}"
            for target in targets
            for suffix in ("lora_down.weight", "lora_up.weight", "alpha")
        }
        assert set(source.keys()) == expected_keys
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
        decoded = decode_lora(
            {key: source.entry(key).geometry for key in source.keys()},
            native_unet_key_map(model_keys),
        )
        assert decoded.dialect == "wan_fun"
        assert decoded.unmatched == ()
        assert decoded.diagnostics == ()
        assert {patch.key for patch in decoded.patches} == {
            f"diffusion_model.{target.module_path}.weight" for target in targets
        }
    finally:
        store.close()
