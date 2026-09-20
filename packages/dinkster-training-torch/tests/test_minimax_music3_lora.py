"""Executable proofs for the native MiniMax Music 3 LoRA trainer."""

from __future__ import annotations

import math
import wave
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import dinkster_training_torch.export as training_export
import dinkster_training_torch.trainer as training_trainer
import pytest
import torch
from dinkster_assets import digest_bytes
from dinkster_inference import FLOAT32, decode_lora, native_unet_key_map
from dinkster_inference.weights import TensorGeometry
from dinkster_inference_torch import Int8Linear, MiniMaxMusic3DiT, build_patch_set
from dinkster_inference_torch.apply import apply_patches
from dinkster_training_torch import (
    MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION,
    MINIMAX_MUSIC3_FLOW_OBJECTIVE,
    AttachmentError,
    MiniMaxMusic3LoRATrainer,
    MiniMaxMusic3PreparedBatch,
    MiniMaxMusic3TrainingConfig,
    TrainingConfigError,
)
from dinkster_training_torch.attachment import resolve_lora_targets
from dinkster_training_torch.export import LoraExportSource
from dinkster_training_torch.trainer import (
    MiniMaxMusic3RandomnessPolicy,
    default_minimax_music3_data_source_factory,
    default_minimax_music3_model_factory,
)

_NATIVE_IDENTITY = "native:dinkster.minimax_music3:" + "1" * 64


def _artifact(path: Path, payload: bytes, *, identity: bool = False) -> dict[str, object]:
    path.write_bytes(payload)
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "digest": digest_bytes(payload),
        "size": len(payload),
    }
    if identity:
        value["identity"] = _NATIVE_IDENTITY
    return value


def _write_wav(path: Path) -> None:
    samples = 44_100 // 25
    payload = torch.arange(samples * 2, dtype=torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44_100)
        audio.writeframes(payload)


def _config_mapping(
    tmp_path: Path,
    *,
    device: str = "cpu",
    base_dtype: str = "float32",
    quantized: bool = False,
    rng_policy: str = "sequential",
) -> dict[str, object]:
    if quantized and base_dtype == "float32":
        base_dtype = "bfloat16"
    dataset = tmp_path / "dataset"
    dataset.mkdir(exist_ok=True)
    _write_wav(dataset / "song.wav")
    (dataset / "song.caption.txt").write_text("bright pop", encoding="utf-8")
    (dataset / "song.lyrics.txt").write_text("hello chorus", encoding="utf-8")
    return {
        "schemaVersion": 1,
        "family": "minimax-music3",
        "communityRecipeRevision": MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION,
        "flowObjective": MINIMAX_MUSIC3_FLOW_OBJECTIVE,
        "diffusionState": _artifact(tmp_path / "diffusion.safetensors", b"dit", identity=True),
        "dataset": {
            "type": "minimax-music3-audio-caption-lyrics-folder",
            "root": str(dataset.resolve()),
            "audioFrames": 1,
            "davEncoderState": _artifact(tmp_path / "dav.pth", b"dav"),
            "rvqEncoderState": _artifact(tmp_path / "rvq.safetensors", b"rvq"),
            "textEncoderState": _artifact(tmp_path / "text.safetensors", b"text", identity=True),
            "encodedCacheRoot": str((tmp_path / "cache").resolve()),
        },
        "device": device,
        "baseDtype": base_dtype,
        "textDtype": "float32",
        "quantizedBase": quantized,
        "rank": 2,
        "alpha": 2.0,
        "learningRate": 0.001,
        "weightDecay": 0.0,
        "gradientCheckpointing": False,
        "seed": 1234,
        "rngPolicy": rng_policy,
    }


class _Projection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _FeedForward(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ff = torch.nn.Sequential(_Projection(), torch.nn.SiLU(), torch.nn.Linear(4, 4))


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_qkv = torch.nn.Linear(4, 4)
        self.to_out = torch.nn.Linear(4, 4)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.ff = _FeedForward()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.self_attn.to_out(torch.tanh(self.self_attn.to_qkv(hidden)))
        first = cast("_Projection", self.ff.ff[0])
        last = cast("torch.nn.Linear", self.ff.ff[2])
        return hidden + last(torch.nn.functional.silu(first.proj(hidden)))


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.project_in = torch.nn.Linear(4, 4)
        self.project_out = torch.nn.Linear(4, 4)
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(36))


class _DiffusionTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Transformer()


class TinyMiniMaxMusic3DiT(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(7)
            self.diffusion_transformer = _DiffusionTransformer()
        self.last_latent: torch.Tensor | None = None
        self.last_timestep: torch.Tensor | None = None
        self.last_context: torch.Tensor | None = None
        self.last_scale: torch.Tensor | None = None
        self.last_output: torch.Tensor | None = None

    def prepare_condition(
        self,
        context: torch.Tensor,
        conditioning_scale: torch.Tensor,
    ) -> torch.Tensor:
        self.last_context = context.detach().float().cpu().clone()
        self.last_scale = conditioning_scale.detach().float().cpu().clone()
        return context[..., :4].transpose(1, 2) * conditioning_scale[:, :1, :1]

    def prepare_rotary(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.empty((0,), device=latent.device, dtype=latent.dtype)

    def forward_prepared(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        rotary: torch.Tensor,
    ) -> torch.Tensor:
        del rotary
        transformer = self.diffusion_transformer.transformer
        hidden = transformer.project_in(latent[:, :4].transpose(1, 2) + condition.transpose(1, 2))
        for layer in self.diffusion_transformer.transformer.layers:
            hidden = layer(hidden)
        hidden = transformer.project_out(hidden)
        update = hidden.mean(dim=-1).unsqueeze(1).expand_as(latent)
        output = latent + update + timestep[:, None, None] * 0.01
        self.last_latent = latent.detach().float().cpu().clone()
        self.last_timestep = timestep.detach().float().cpu().clone()
        self.last_output = output.detach().float().cpu().clone()
        return output

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        conditioning_scale: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.prepare_condition(context, conditioning_scale)
        condition = torch.nn.functional.interpolate(
            condition, size=latent.shape[-1], mode="nearest"
        )
        return self.forward_prepared(latent, timestep, condition, self.prepare_rotary(latent))


class _FixedBatches:
    def __init__(self, config: MiniMaxMusic3TrainingConfig) -> None:
        self.batch_value = MiniMaxMusic3PreparedBatch(
            torch.linspace(-0.5, 0.5, math.prod(config.latent_shape)).reshape(config.latent_shape),
            torch.linspace(-0.2, 0.2, math.prod(config.context_shape)).reshape(
                config.context_shape
            ),
        )

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxMusic3PreparedBatch:
        del cursor, generator, device
        return self.batch_value


def _model_factory(config: MiniMaxMusic3TrainingConfig) -> TinyMiniMaxMusic3DiT:
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[config.base_dtype]
    return TinyMiniMaxMusic3DiT().to(dtype=dtype)


def _quantize_linear(linear: torch.nn.Linear) -> Int8Linear:
    weight = linear.weight.detach().float()
    scale = weight.abs().max().div(127).clamp_min(torch.finfo(torch.float32).tiny)
    replacement = Int8Linear(
        linear.in_features,
        linear.out_features,
        bias=True,
        compute_dtype=torch.bfloat16,
        convrot=False,
        convrot_groupsize=256,
        full_precision_matmul=True,
    )
    state = {
        "weight": weight.div(scale).round().clamp(-128, 127).to(torch.int8),
        "weight_scale": scale,
    }
    state["bias"] = linear.bias.detach().float()
    replacement.load_state_dict(state, strict=True, assign=True)
    return replacement


def _quantized_model() -> TinyMiniMaxMusic3DiT:
    model = TinyMiniMaxMusic3DiT()
    for path, module in tuple(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        parent_path, _, name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, name, _quantize_linear(module))
    return model


def _trainer(
    config: MiniMaxMusic3TrainingConfig, *, quantized: bool = False
) -> MiniMaxMusic3LoRATrainer:
    model = _quantized_model() if quantized else _model_factory(config)
    return MiniMaxMusic3LoRATrainer(
        config,
        cast("MiniMaxMusic3DiT", model),
        _FixedBatches(config),
    )


def _tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _tree_equal(left_item, right_item)
    else:
        assert left == right


def test_config_round_trip_and_identity_exclude_runtime_locations(tmp_path: Path) -> None:
    config = MiniMaxMusic3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    assert MiniMaxMusic3TrainingConfig.from_mapping(config.to_mapping()) == config
    assert config.latent_shape == (1, 128, 4)
    assert config.context_shape == (1, 1, 32768)

    identity = config.identity_mapping()
    assert "path" not in cast("dict[str, object]", identity["diffusionState"])
    dataset = cast("dict[str, object]", identity["dataset"])
    assert "encodedCacheRoot" not in dataset
    for name in ("davEncoderState", "rvqEncoderState", "textEncoderState"):
        assert "path" not in cast("dict[str, object]", dataset[name])

    with pytest.raises(TrainingConfigError, match="wholeModel"):
        replace(config, checkpointing_mode=cast("object", "blockNonReentrant"))  # pyright: ignore[reportArgumentType]
    with pytest.raises(TrainingConfigError, match="learningRate"):
        replace(config, learning_rate=float("nan"))
    with pytest.raises(TrainingConfigError, match="text compute dtype"):
        replace(
            config,
            dataset=replace(config.dataset, text_compute_dtype="bfloat16"),
        )


def test_production_factories_forward_registered_attention_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MiniMaxMusic3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    model = _model_factory(config)
    calls: list[tuple[str, str | None]] = []

    def load_component(
        path: Path,
        *,
        asset: object,
        expected_role: str,
        expected_identity: str,
        compute_dtype: torch.dtype,
        attention_backend: str | None = None,
    ) -> object:
        del path, asset, expected_identity, compute_dtype
        calls.append((expected_role, attention_backend))
        return SimpleNamespace(module=model, tokenizer=object())

    marker = object()

    def dataset_source(
        settings: object,
        dav_factory: object,
        rvq_factory: object,
        text_factory: Callable[[], object],
        *,
        device: torch.device,
    ) -> object:
        del settings, dav_factory, rvq_factory, device
        text_factory()
        return marker

    monkeypatch.setattr(training_trainer, "load_minimax_music3_component", load_component)
    monkeypatch.setattr(training_trainer, "MiniMaxMusic3DatasetSource", dataset_source)

    assert default_minimax_music3_model_factory(config) is model
    assert default_minimax_music3_data_source_factory(config) is marker
    assert calls == [("diffusion", "flux"), ("text", "qwen")]


def test_config_rejects_recipe_artifact_and_dataset_drift(tmp_path: Path) -> None:
    mapping = _config_mapping(tmp_path)
    mapping["flowObjective"] = "official-unknown"
    with pytest.raises(TrainingConfigError, match="flowObjective"):
        MiniMaxMusic3TrainingConfig.from_mapping(mapping)

    mapping = _config_mapping(tmp_path)
    diffusion = cast("dict[str, object]", mapping["diffusionState"])
    diffusion["identity"] = "native:dinkster.minimax_music3:" + "a" * 63
    with pytest.raises(TrainingConfigError, match="native identity"):
        MiniMaxMusic3TrainingConfig.from_mapping(mapping)

    mapping = _config_mapping(tmp_path)
    dataset = cast("dict[str, object]", mapping["dataset"])
    dataset["digest"] = "blake3:" + "f" * 64
    with pytest.raises(TrainingConfigError, match="dataset digest changed"):
        MiniMaxMusic3TrainingConfig.from_mapping(mapping)


def test_production_model_exposes_exact_stable_target_set() -> None:
    with torch.device("meta"):
        model = MiniMaxMusic3DiT()
    targets = resolve_lora_targets(model, 2, family="minimax-music3")

    assert len(targets) == 36 * 4 + 2
    assert {target.module_path for target in targets if ".layers." not in target.module_path} == {
        "diffusion_transformer.transformer.project_in",
        "diffusion_transformer.transformer.project_out",
    }
    for block in range(36):
        prefix = f"diffusion_transformer.transformer.layers.{block}."
        assert {
            target.module_path.removeprefix(prefix)
            for target in targets
            if target.module_path.startswith(prefix)
        } == {
            "self_attn.to_qkv",
            "self_attn.to_out",
            "ff.ff.0.proj",
            "ff.ff.2",
        }

    del model.diffusion_transformer.transformer.layers[-1]
    with pytest.raises(AttachmentError, match="stable 146-target manifest"):
        resolve_lora_targets(model, 2, family="minimax-music3")


def test_flow_objective_uses_data_time_logistic_normal_and_exact_velocity(tmp_path: Path) -> None:
    config = MiniMaxMusic3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    trainer = _trainer(config)
    prepared = cast("_FixedBatches", trainer.data_source).batch_value
    timestep = torch.tensor([0.75])
    noise = torch.full(config.latent_shape, 0.25)
    cast("_FixedBatches", trainer.data_source).batch_value = replace(
        prepared, timesteps=timestep, noise=noise
    )

    loss = trainer.train_step()
    model = cast("TinyMiniMaxMusic3DiT", trainer.model)
    data = prepared.latents.float()
    expected_noisy = timestep[:, None, None] * data + (1.0 - timestep[:, None, None]) * noise
    assert model.last_latent is not None
    assert model.last_timestep is not None
    assert model.last_scale is not None
    assert torch.equal(model.last_latent, expected_noisy)
    assert torch.equal(model.last_timestep, timestep)
    assert torch.equal(model.last_scale, torch.ones((1, 1, 1)))
    assert model.last_output is not None
    expected_loss = torch.nn.functional.mse_loss(model.last_output, data - noise).item()
    assert loss == pytest.approx(expected_loss)

    policy = cast("MiniMaxMusic3RandomnessPolicy", trainer.randomness_policy)
    actual_time, actual_noise = policy.draw_minimax_music3_timestep_noise(
        0, latent_shape=config.latent_shape
    )
    assert 0.0 < actual_time.item() < 1.0
    assert actual_noise.shape == config.latent_shape


@pytest.mark.parametrize("base_dtype", ("float32", "float16"))
def test_float_bases_train_only_lora_masters(tmp_path: Path, base_dtype: str) -> None:
    config = MiniMaxMusic3TrainingConfig.from_mapping(
        _config_mapping(tmp_path, base_dtype=base_dtype)
    )
    trainer = _trainer(config)
    before = {
        name: value.detach().cpu().clone() for name, value in trainer.model.state_dict().items()
    }

    first_loss = trainer.train_step()
    second_loss = trainer.train_step()

    assert math.isfinite(first_loss) and math.isfinite(second_loss)
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in trainer.model.parameters()
    )
    assert all(
        torch.equal(before[name], value.detach().cpu())
        for name, value in trainer.model.state_dict().items()
    )
    attachment_parameters = trainer.attachment.parameters()
    assert {
        id(parameter) for group in trainer.optimizer.param_groups for parameter in group["params"]
    } == {id(parameter) for parameter in attachment_parameters}
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in attachment_parameters
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in attachment_parameters
    )


def test_int8_base_trains_without_base_gradients_or_optimizer_state(tmp_path: Path) -> None:
    config = MiniMaxMusic3TrainingConfig.from_mapping(_config_mapping(tmp_path, quantized=True))
    trainer = _trainer(config, quantized=True)
    before = {
        name: value.detach().cpu().clone() for name, value in trainer.model.state_dict().items()
    }

    assert len(resolve_lora_targets(trainer.model, 2, family="minimax-music3")) == 146
    assert all(
        isinstance(module, Int8Linear)
        for module in trainer.model.modules()
        if hasattr(module, "weight")
        and isinstance(module.weight, torch.Tensor)
        and module.weight.ndim == 2
    )
    assert math.isfinite(trainer.train_step())
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    assert all(
        torch.equal(before[name], value.detach().cpu())
        for name, value in trainer.model.state_dict().items()
    )
    assert len(trainer.optimizer.state) == len(trainer.attachment.parameters())


@pytest.mark.parametrize("rng_policy", ("sequential", "counter"))
def test_checkpoint_restore_is_bit_exact(tmp_path: Path, rng_policy: str) -> None:
    config = MiniMaxMusic3TrainingConfig.from_mapping(
        _config_mapping(tmp_path, rng_policy=rng_policy)
    )
    uninterrupted = _trainer(config)
    uninterrupted.train_step()
    adapter = uninterrupted.attachment.state_dict()
    optimizer = deepcopy(uninterrupted.optimizer_state_dict())
    rng = uninterrupted.rng_state_dict()
    step = uninterrupted.step_cursor
    cursor = uninterrupted.data_cursor
    loss = uninterrupted.last_loss

    expected_loss = uninterrupted.train_step()
    expected_adapter = uninterrupted.attachment.state_dict()
    expected_optimizer = uninterrupted.optimizer_state_dict()

    resumed = _trainer(config)
    resumed.restore(
        adapter=adapter,
        optimizer=optimizer,
        rng=rng,
        step_cursor=step,
        data_cursor=cursor,
        loss=loss,
    )
    assert resumed.train_step() == expected_loss
    _tree_equal(resumed.attachment.state_dict(), expected_adapter)
    _tree_equal(resumed.optimizer_state_dict(), expected_optimizer)


def test_export_mapping_decodes_and_applies_through_native_inference(tmp_path: Path) -> None:
    config = MiniMaxMusic3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    trainer = _trainer(config)
    trainer.train_step()
    trainer.train_step()
    target = trainer.attachment.targets[0]
    adapter = trainer.attachment.state_dict()
    selected_adapter = {
        f"{target.target_id}.down": adapter[f"{target.target_id}.down"],
        f"{target.target_id}.up": adapter[f"{target.target_id}.up"],
    }
    source = LoraExportSource(None, "session", "config", 2, selected_adapter)
    model_keys = tuple(f"diffusion_model.{key}" for key in trainer.model.state_dict())
    tensors = training_export._minimax_music3_tensors(  # pyright: ignore[reportPrivateUsage]
        source,
        config,
        torch.float32,
        targets=(target,),
        model_keys=model_keys,
    )
    geometries = {
        name: TensorGeometry(tuple(tensor.shape), FLOAT32) for name, tensor in tensors.items()
    }
    decoded = decode_lora(geometries, native_unet_key_map(model_keys))
    patches = build_patch_set(decoded.patches, tensors)
    target_key = f"diffusion_model.{target.module_path}.weight"
    base = trainer.model.state_dict()[f"{target.module_path}.weight"].detach().float().clone()
    patched = apply_patches(base.clone(), patches.entries(target_key), key=target_key)

    assert not decoded.unmatched and not decoded.diagnostics
    assert len(decoded.patches) == 1
    decoded_target, decoded_patch = next(iter(decoded.patches.items()))
    assert decoded_target.key == target_key
    assert getattr(decoded_patch, "variant", None) == "peft"
    assert not torch.equal(patched, base)
