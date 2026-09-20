"""Executable proofs for Ideogram 4 configuration, attachment, training, and export."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import dinkster_training_torch.export as training_export
import dinkster_training_torch.service as training_service
import dinkster_training_torch.trainer as training_trainer
import pytest
import torch
from dinkster_assets import digest_bytes
from dinkster_inference import (
    FLOAT8_E4M3,
    decode_lora,
    load_safetensors_header,
    native_unet_key_map,
)
from dinkster_inference_torch import Fp8Linear, Int8Linear, load_tensors
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    IDEOGRAM4_TRAINING_RUNTIME_IDENTITY,
    CheckpointError,
    ContentAddressedCheckpointStore,
    Ideogram4LoRATrainer,
    Ideogram4LoRATrainingService,
    Ideogram4PreparedBatch,
    Ideogram4TrainingConfig,
    SequentialRandomnessPolicy,
    TrainingConfigError,
    ideogram4_capability_report,
    ideogram4_training_sigmas,
)
from dinkster_training_torch.attachment import resolve_lora_targets
from dinkster_training_torch.export import (
    LoraExportSource,
    _ideogram4_tensors,  # pyright: ignore[reportPrivateUsage]
)

_DIFFUSION_DIGESTS = {
    ("conditional", "fp8"): (
        "blake3:dadac522cf4fd911d25c70401df4e1233805887481e21be0e19839458fb69eba"
    ),
    ("unconditional", "fp8"): (
        "blake3:1e26b2ecf7cb7ab57495faff8534875be41a44d6fc2ffe37f934dd65a460958d"
    ),
    ("conditional", "int8-convrot"): (
        "blake3:e3cf071faafcf04192a66fa0324948589e60325b54a7935206f4e30b8ef257ce"
    ),
    ("unconditional", "int8-convrot"): (
        "blake3:eb50781b817ef134e114ffa55def137206c56ce32846f12e43397f0ac2a8e909"
    ),
}
_TEXT_DIGEST = "blake3:b82a81d829c1d8db687c8e9f79fc07e7211f5ac59ed05672423c1cedc8db0924"
_VAE_DIGEST = "blake3:fcb1d172993424c66d325d139863ccbaadf64a920073b2d005d73a31fa5a851d"


def _native_artifact(path: Path, digest: str, family: str, marker: str) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "digest": digest,
        "size": 1,
        "identity": f"native:{family}:{marker * 64}",
    }


def config_mapping(
    tmp_path: Path,
    *,
    role: Literal["conditional", "unconditional"] = "conditional",
    storage: Literal["fp8", "int8-convrot"] = "fp8",
) -> dict[str, object]:
    mapping: dict[str, object] = {
        "schemaVersion": 1,
        "family": "ideogram4",
        "variant": "ideogram4",
        "role": role,
        "diffusionState": _native_artifact(
            tmp_path / "diffusion.safetensors",
            _DIFFUSION_DIGESTS[(role, storage)],
            "dinkster.ideogram4",
            "1",
        ),
        "vaeState": _native_artifact(
            tmp_path / "vae.safetensors", _VAE_DIGEST, "dinkster.flux2", "3"
        ),
        "datasetIdentity": "blake3:" + "4" * 64,
        "device": "cpu",
        "baseDtype": "bfloat16",
        "baseStorage": storage,
        "rank": 1,
        "alpha": 1.0,
        "loraTargets": ["attention.qkvo", "mlp.projections"],
        "learningRate": 0.0001,
        "weightDecay": 0.01,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "optimizer": "adamw",
        "gradientAccumulationSteps": 1,
        "gradientCheckpointing": False,
        "seed": 1199,
        "latentShape": [1, 128, 1, 1],
    }
    if role == "conditional":
        mapping["textEncoderState"] = _native_artifact(
            tmp_path / "text.safetensors", _TEXT_DIGEST, "dinkster.ideogram4", "2"
        )
        mapping["contextShape"] = [1, 2, 53248]
        mapping["attentionMaskShape"] = [1, 2]
    if storage == "int8-convrot":
        mapping["int8BaseForward"] = "dequantize"
    return mapping


@pytest.mark.parametrize("role", ("conditional", "unconditional"))
def test_ideogram4_config_round_trips_role_bound_contract(tmp_path: Path, role: str) -> None:
    config = Ideogram4TrainingConfig.from_mapping(
        config_mapping(tmp_path, role=cast('Literal["conditional", "unconditional"]', role))
    )

    assert Ideogram4TrainingConfig.from_mapping(config.to_mapping()) == config
    assert config.role == role
    assert (config.text_encoder_state is not None) == (role == "conditional")
    assert (config.context_shape is not None) == (role == "conditional")
    assert config.flow_objective == "ideogram4-shifted-logit-normal-noise-minus-data-mse-v1"
    assert config.identity_mapping()["role"] == role


def test_ideogram4_config_binds_official_artifact_role_and_storage(tmp_path: Path) -> None:
    mapping = config_mapping(tmp_path)
    diffusion = cast("dict[str, object]", mapping["diffusionState"])
    diffusion["digest"] = _DIFFUSION_DIGESTS[("unconditional", "fp8")]
    with pytest.raises(TrainingConfigError, match="selected Ideogram 4 role and base storage"):
        Ideogram4TrainingConfig.from_mapping(mapping)

    unconditional = config_mapping(tmp_path, role="unconditional")
    unconditional["textEncoderState"] = _native_artifact(
        tmp_path / "text.safetensors", _TEXT_DIGEST, "dinkster.ideogram4", "2"
    )
    with pytest.raises(TrainingConfigError, match="does not accept textEncoderState"):
        Ideogram4TrainingConfig.from_mapping(unconditional)

    int8 = config_mapping(tmp_path, storage="int8-convrot")
    int8.pop("int8BaseForward")
    with pytest.raises(TrainingConfigError, match="requires int8BaseForward"):
        Ideogram4TrainingConfig.from_mapping(int8)

    objective = config_mapping(tmp_path)
    objective["flowObjective"] = "uniform-flow"
    with pytest.raises(TrainingConfigError, match="flowObjective is not supported"):
        Ideogram4TrainingConfig.from_mapping(objective)


def test_ideogram4_targets_are_exact_and_role_disjoint() -> None:
    from dinkster_inference_torch import Ideogram4DiT

    with torch.device("meta"):
        model = Ideogram4DiT()
    conditional = resolve_lora_targets(model, 2, family="ideogram4", role="conditional")
    unconditional = resolve_lora_targets(model, 2, family="ideogram4", role="unconditional")

    assert len(conditional) == len(unconditional) == 170
    expected_paths = {
        f"layers.{layer}.{suffix}"
        for layer in range(34)
        for suffix in (
            "attention.qkv",
            "attention.o",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
        )
    }
    assert {target.module_path for target in conditional} == expected_paths
    assert all(
        target.target_id == f"ideogram4/conditional/dit/{target.module_path}/weight"
        for target in conditional
    )
    assert {target.target_id for target in conditional}.isdisjoint(
        target.target_id for target in unconditional
    )


class _TinyAttention(torch.nn.Module):
    def __init__(self, projection: Callable[[], torch.nn.Module]) -> None:
        super().__init__()
        self.qkv = projection()
        self.o = projection()


class _TinyFeedForward(torch.nn.Module):
    def __init__(self, projection: Callable[[], torch.nn.Module]) -> None:
        super().__init__()
        self.w1 = projection()
        self.w2 = projection()
        self.w3 = projection()


class _TinyBlock(torch.nn.Module):
    def __init__(self, projection: Callable[[], torch.nn.Module]) -> None:
        super().__init__()
        self.attention = _TinyAttention(projection)
        self.feed_forward = _TinyFeedForward(projection)


def _fp8_projection() -> torch.nn.Module:
    layer = Fp8Linear(1, 1, bias=False, compute_dtype=torch.bfloat16)
    layer.load_state_dict(
        {
            "weight": torch.ones(1, 1).to(torch.float8_e4m3fn),
            "weight_scale": torch.ones(()),
            "input_scale": torch.ones(()),
        },
        assign=True,
    )
    return layer


def _linear_projection() -> torch.nn.Module:
    return torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)


class _TinyIdeogram4(torch.nn.Module):
    def __init__(self, projection: Callable[[], torch.nn.Module] | None = None) -> None:
        super().__init__()
        factory = _linear_projection if projection is None else projection
        self.layers = torch.nn.ModuleList(_TinyBlock(factory) for _ in range(34))
        self.seen: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
        ] = []
        if projection is None:
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.fill_(1.0)

    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.seen.append(
            (
                latent.detach().clone(),
                timesteps.detach().clone(),
                None if context is None else context.detach().clone(),
                None if attention_mask is None else attention_mask.detach().clone(),
            )
        )
        value = latent.mean(dim=1, keepdim=True).movedim(1, -1)
        for layer in self.layers:
            block = cast("_TinyBlock", layer)
            value = block.attention.qkv(value)
            value = block.attention.o(value)
            value = block.feed_forward.w1(value)
            value = block.feed_forward.w2(value)
            value = block.feed_forward.w3(value)
        return value.movedim(-1, 1).expand_as(latent)


class _Batches:
    def __init__(self, batch: Ideogram4PreparedBatch) -> None:
        self._batch = batch

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> Ideogram4PreparedBatch:
        del cursor, generator, device
        return self._batch


def _batch(config: Ideogram4TrainingConfig) -> Ideogram4PreparedBatch:
    conditional = config.role == "conditional"
    context = None
    if config.context_shape is not None:
        context = torch.ones(config.context_shape)
    return Ideogram4PreparedBatch(
        torch.zeros(config.latent_shape),
        context,
        torch.tensor([[1.0, 0.0]]) if conditional else None,
        sigmas=torch.tensor([0.5]),
        noise=torch.full(config.latent_shape, 0.75),
    )


def _model() -> _TinyIdeogram4:
    return _TinyIdeogram4(_fp8_projection)


@pytest.mark.parametrize("role", ("conditional", "unconditional"))
def test_ideogram4_objective_trains_fp8_base_without_enabling_fp8_matmul(
    tmp_path: Path, role: str
) -> None:
    config = Ideogram4TrainingConfig.from_mapping(
        config_mapping(tmp_path, role=cast('Literal["conditional", "unconditional"]', role))
    )
    model = _model()
    trainer = Ideogram4LoRATrainer(
        config,
        cast("object", model),  # pyright: ignore[reportArgumentType]
        _Batches(_batch(config)),
    )
    frozen = {name: value.detach().clone() for name, value in model.state_dict().items()}

    loss = trainer.train_step()

    assert math.isfinite(loss)
    assert trainer.step_cursor == trainer.data_cursor == 1
    assert len(trainer.attachment.targets) == 170
    assert all(not module.fp8_matmul for module in model.modules() if isinstance(module, Fp8Linear))
    assert len(model.seen) == 1
    noisy, sigma, context, mask = model.seen[0]
    assert torch.equal(noisy, torch.full_like(noisy, 0.375))
    assert torch.equal(sigma, torch.tensor([0.5]))
    assert (context is not None) == (role == "conditional")
    assert (mask is not None and mask.dtype == torch.bool) == (role == "conditional")
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(value, frozen[name]) for name, value in model.state_dict().items())
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
        for name, parameter in trainer.attachment.named_parameters()
        if name.endswith(".up")
    )


def test_ideogram4_objective_uses_native_resolution_aware_timestep_sampling(
    tmp_path: Path,
) -> None:
    config = Ideogram4TrainingConfig.from_mapping(config_mapping(tmp_path))
    batch = replace(_batch(config), sigmas=None, noise=None)

    class FixedRandomness(SequentialRandomnessPolicy):
        def draw_ideogram4_timestep_noise(
            self,
            data_cursor: int,
            *,
            latent_shape: tuple[int, ...],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del data_cursor
            return torch.tensor([0.5]), torch.full(latent_shape, 0.75)

    model = _model()
    trainer = Ideogram4LoRATrainer(
        config,
        cast("object", model),  # pyright: ignore[reportArgumentType]
        _Batches(batch),
        randomness_policy=FixedRandomness(config.seed, torch.device("cpu")),
    )

    trainer.train_step()

    expected_sigma = ideogram4_training_sigmas(torch.tensor([0.5]), config.latent_shape)
    noisy, sigma, _, _ = model.seen[0]
    torch.testing.assert_close(sigma, expected_sigma, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        noisy,
        (expected_sigma.reshape(1, 1, 1, 1) * 0.75).expand_as(noisy).to(noisy.dtype),
        rtol=0.0,
        atol=0.0,
    )


def test_ideogram4_int8_forward_policy_is_explicit(tmp_path: Path) -> None:
    class Quantized(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = Int8Linear(
                16,
                16,
                bias=False,
                compute_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=4,
            )

    config = Ideogram4TrainingConfig.from_mapping(config_mapping(tmp_path, storage="int8-convrot"))
    dequantized = Quantized()
    training_trainer._configure_ideogram4_base(  # pyright: ignore[reportPrivateUsage]
        cast("object", dequantized),  # pyright: ignore[reportArgumentType]
        config,
    )
    assert dequantized.projection.full_precision_matmul is True
    assert dequantized.projection.fused_training is False

    fused = Quantized()
    training_trainer._configure_ideogram4_base(  # pyright: ignore[reportPrivateUsage]
        cast("object", fused),  # pyright: ignore[reportArgumentType]
        replace(config, device="cuda", int8_base_forward="fused"),
    )
    assert fused.projection.fused_training is True


def test_ideogram4_component_plan_accepts_quantized_text_and_selected_fp8_diffusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Ideogram4TrainingConfig.from_mapping(config_mapping(tmp_path))
    fp8_quant = SimpleNamespace(format=None, parameters={})
    diffusion = SimpleNamespace(
        quant={"layers.0.attention.qkv.weight": fp8_quant},
        dtypes={"layers.0.attention.qkv.weight": FLOAT8_E4M3},
    )
    text = SimpleNamespace(quant={"model.layers.0.mlp.down_proj.weight": fp8_quant})
    vae = SimpleNamespace(family_id="dinkster.flux2", plan=SimpleNamespace(quant={}))

    def artifact_header(_source: object, _family: str) -> object:
        return object()

    def ideogram_plan(_source: object, *, role: str, path: Path) -> SimpleNamespace:
        del path
        return diffusion if role == "diffusion" else text

    def flux2_plan(_source: object, *, role: str, path: Path) -> SimpleNamespace:
        del role, path
        return vae

    text_state = config.text_encoder_state
    assert text_state is not None

    def ideogram_identity(_plan: object, role: str, _dtype: object) -> str:
        return config.diffusion_state.identity if role == "diffusion" else text_state.identity

    def flux2_identity(_plan: object, _dtype: object) -> str:
        return config.vae_state.identity

    monkeypatch.setattr(training_trainer, "_native_artifact_header", artifact_header)
    monkeypatch.setattr(
        training_trainer,
        "plan_ideogram4_split_component",
        ideogram_plan,
    )
    monkeypatch.setattr(training_trainer, "plan_flux2_split_component", flux2_plan)
    monkeypatch.setattr(
        training_trainer,
        "ideogram4_component_runtime_identity",
        ideogram_identity,
    )
    monkeypatch.setattr(
        training_trainer,
        "flux2_component_runtime_identity",
        flux2_identity,
    )

    plans = training_trainer.ideogram4_component_plans(config)

    assert plans.diffusion is diffusion
    assert plans.text_encoder is text
    assert plans.vae is vae


def test_ideogram4_capability_report_binds_role_targets_and_quantized_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Ideogram4TrainingConfig.from_mapping(config_mapping(tmp_path))

    def component_plans(_config: Ideogram4TrainingConfig) -> object:
        return object()

    monkeypatch.setattr(training_service, "ideogram4_component_plans", component_plans)

    report = ideogram4_capability_report(config)

    capabilities = cast("dict[str, object]", report["capabilities"])
    assert report["role"] == "conditional"
    assert capabilities["roles"] == ["conditional"]
    assert capabilities["quantizedBase"] is True
    assert capabilities["datasetBucketing"] is False
    assert len(cast("list[object]", report["targets"])) == 170


def test_ideogram4_capability_refuses_invalid_cuda_ordinal_before_component_planning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        Ideogram4TrainingConfig.from_mapping(config_mapping(tmp_path)),
        device="cuda:2",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    def fail(_config: Ideogram4TrainingConfig) -> object:
        raise AssertionError("component planning must follow CUDA admission")

    monkeypatch.setattr(training_service, "ideogram4_component_plans", fail)

    with pytest.raises(ValueError, match="CUDA device cuda:2 does not exist"):
        ideogram4_capability_report(config)


def _service_report(config: Ideogram4TrainingConfig) -> dict[str, object]:
    return {
        "family": config.family,
        "role": config.role,
        "dataset": {"type": "injected", "errors": []},
    }


def test_ideogram4_service_resumes_and_exports_role_bound_native_lora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = config_mapping(tmp_path / "artifacts")
    mapping["loraExportInterval"] = 1
    config = Ideogram4TrainingConfig.from_mapping(mapping)
    monkeypatch.setattr(training_service, "ideogram4_capability_report", _service_report)

    def component_plans(_config: Ideogram4TrainingConfig) -> object:
        return object()

    monkeypatch.setattr(training_trainer, "ideogram4_component_plans", component_plans)
    monkeypatch.setattr(training_export, "Ideogram4DiT", _TinyIdeogram4)

    def model_factory(_config: Ideogram4TrainingConfig) -> object:
        return _model()

    def data_factory(config: Ideogram4TrainingConfig) -> _Batches:
        return _Batches(_batch(config))

    checkpoint_root = tmp_path / "checkpoints"
    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = Ideogram4LoRATrainingService(
            store,
            checkpoint_root,
            model_factory=cast("object", model_factory),  # pyright: ignore[reportArgumentType]
            data_source_factory=data_factory,
        )
        initial, report = service.create("ideogram4-session", json.dumps(config.to_mapping()))
        assert report == _service_report(config)
        first = service.advance(initial, 1, "train Ideogram 4 adapter")
        assert first.handle.step_cursor == 1
        service._evict_runtime(initial.session_id)  # pyright: ignore[reportPrivateUsage]
        resumed = service.advance(first.handle, 1, "resume Ideogram 4 adapter")
        assert resumed.handle.step_cursor == 2
        assert resumed.loss is not None and math.isfinite(resumed.loss)
        state = ContentAddressedCheckpointStore(checkpoint_root).load(
            resumed.handle.checkpoint_manifest_digest
        )
        assert state.step_cursor == state.data_cursor == 2

        path_value, export_digest = service.export_lora(
            resumed.handle,
            json.dumps({"path": "ideogram4.safetensors", "dtype": "fp32"}),
        )
        path = Path(path_value)
        assert export_digest == digest_bytes(path.read_bytes())
        source = load_safetensors_header(path)
        assert source.metadata() == {
            "dinkster_checkpoint_manifest_digest": resumed.handle.checkpoint_manifest_digest,
            "dinkster_config_digest": resumed.handle.config_digest,
            "dinkster_ideogram4_base_storage": "fp8",
            "dinkster_ideogram4_role": "conditional",
            "dinkster_runtime_identity": IDEOGRAM4_TRAINING_RUNTIME_IDENTITY,
            "dinkster_session_id": resumed.handle.session_id,
            "dinkster_step_cursor": "2",
        }
        assert len(source.keys()) == 510
        model = _model()
        targets = resolve_lora_targets(model, config.rank, family="ideogram4", role=config.role)
        model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
        decoded = decode_lora(
            {key: source.entry(key).geometry for key in source.keys()},
            native_unet_key_map(model_keys),
        )
        assert decoded.unmatched == ()
        assert decoded.diagnostics == ()
        assert len(decoded.patches) == 170
        tensors = load_tensors(path)
        assert all(tensor.dtype == torch.float32 for tensor in tensors.values())
        for target in targets:
            stem = f"diffusion_model.{target.module_path}"
            expected = (
                state.adapter[f"{target.target_id}.up"] @ state.adapter[f"{target.target_id}.down"]
            ) * (config.alpha / config.rank)
            actual = (tensors[f"{stem}.lora_B.weight"] @ tensors[f"{stem}.lora_A.weight"]) * (
                float(tensors[f"{stem}.alpha"].item()) / config.rank
            )
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

        cadence = (
            checkpoint_root
            / "exports"
            / "cadence"
            / initial.session_id
            / "step-000000000001.safetensors"
        )
        assert cadence.is_file()
        assert service.complete(resumed.handle) == resumed.handle
    finally:
        store.close()


def test_ideogram4_export_rejects_adapter_role_mismatch(tmp_path: Path) -> None:
    config = Ideogram4TrainingConfig.from_mapping(config_mapping(tmp_path))
    model = _model()
    conditional = resolve_lora_targets(model, 1, family="ideogram4", role="conditional")
    unconditional = resolve_lora_targets(model, 1, family="ideogram4", role="unconditional")
    adapter: dict[str, torch.Tensor] = {}
    for target in unconditional:
        adapter[f"{target.target_id}.down"] = torch.ones(1, 1)
        adapter[f"{target.target_id}.up"] = torch.zeros(1, 1)
    source = LoraExportSource(None, "session", "blake3:" + "5" * 64, 0, adapter)

    with pytest.raises(CheckpointError, match="keys differ during export"):
        _ideogram4_tensors(
            source,
            config,
            torch.float32,
            conditional,
            tuple(f"diffusion_model.{key}" for key in model.state_dict()),
        )


def test_ideogram4_training_schedule_is_resolution_aware_logit_normal() -> None:
    samples = torch.tensor([0.0, 0.5, 1.0])

    at_512 = ideogram4_training_sigmas(samples, (3, 128, 32, 32))
    at_1024 = ideogram4_training_sigmas(samples, (3, 128, 64, 64))

    assert at_512.dtype == torch.float32
    assert at_512[0].item() == pytest.approx(0.0005527786)
    assert at_512[1].item() == 0.5
    assert at_512[2].item() == pytest.approx(0.9995900393)
    assert at_1024[1].item() == pytest.approx(2.0 / 3.0)
