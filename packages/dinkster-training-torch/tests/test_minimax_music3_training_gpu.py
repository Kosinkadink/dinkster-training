"""Physical-GPU proofs for community-derived MiniMax Music 3 training."""

from __future__ import annotations

import json
import math
import os
import time
import wave
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
import torch
from dinkster_assets import digest_file
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    DType,
    MiniMaxMusic3ComponentRole,
    decode_lora,
    load_safetensors_header,
    minimax_music3_component_runtime_identity,
    native_unet_key_map,
    plan_minimax_music3_split_component,
)
from dinkster_inference_torch import (
    Int8Linear,
    MiniMaxMusic3Dav,
    MiniMaxMusic3DiT,
    MiniMaxMusic3TextModel,
    build_patch_set,
)
from dinkster_inference_torch.apply import apply_patches
from dinkster_inference_torch.sources import load_tensors
from dinkster_training_torch import (
    MiniMaxMusic3ArtifactPin,
    MiniMaxMusic3DatasetSource,
    MiniMaxMusic3LoRATrainer,
    MiniMaxMusic3PreparedBatchSource,
    MiniMaxMusic3TrainingConfig,
)
from dinkster_training_torch.export import (
    LoraExportSettings,
    LoraExportSource,
    export_intermediate_lora,
)
from dinkster_training_torch.minimax_music3_training import (
    MiniMaxMusic3DavEncoder,
    MiniMaxMusic3RvqEncoder,
    load_minimax_music3_dav_encoder,
)
from dinkster_training_torch.service import MINIMAX_MUSIC3_TRAINING_RUNTIME_IDENTITY
from dinkster_training_torch.trainer import (
    default_minimax_music3_data_source_factory,
    default_minimax_music3_model_factory,
)
from tokenizers import Tokenizer

_PREFLIGHT = os.environ.get("DINKSTER_MINIMAX_MUSIC3_TRAINING_PREFLIGHT") == "1"
pytestmark = pytest.mark.skipif(
    not _PREFLIGHT or not torch.cuda.is_available(),
    reason="explicit MiniMax Music 3 CUDA preflight required (see README)",
)

_MODELS_ENV = os.environ.get("DINKSTER_MINIMAX_MUSIC3_MODELS")
_COMMUNITY_ENV = os.environ.get("DINKSTER_MINIMAX_MUSIC3_TRAINING_ARTIFACTS")
if _PREFLIGHT and (not _MODELS_ENV or not _COMMUNITY_ENV):
    raise RuntimeError(
        "DINKSTER_MINIMAX_MUSIC3_MODELS and "
        "DINKSTER_MINIMAX_MUSIC3_TRAINING_ARTIFACTS must be set for the preflight"
    )
_MODELS = Path(_MODELS_ENV or ".")
_COMMUNITY = Path(_COMMUNITY_ENV or ".")
_DAV = _COMMUNITY / "minimax_music3_dav_full_fce0d00b.safetensors"
_RVQ = (
    _COMMUNITY / "minimax_music3_rvq_encoder_v4_169m_autoregressive_depth_recommended.safetensors"
)
_TEXT = _MODELS / "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors"
_DIFFUSION = (
    (
        _MODELS / "diffusion_models/minimax_music3_dit_fp16.safetensors",
        "float16",
        False,
        FLOAT16,
    ),
    (
        _MODELS / "diffusion_models/minimax_music3_dit_fp32.safetensors",
        "float32",
        False,
        FLOAT32,
    ),
    (
        _MODELS / "diffusion_models/minimax_music3_dit_int8_convrot.safetensors",
        "bfloat16",
        True,
        BFLOAT16,
    ),
)


def _rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise AssertionError("process RSS is unavailable")


def _artifact(
    path: Path, *, role: MiniMaxMusic3ComponentRole | None = None, dtype: DType | None = None
) -> dict[str, object]:
    assert path.is_file(), f"required MiniMax Music 3 artifact is absent: {path}"
    digest = digest_file(path)
    result: dict[str, object] = {
        "path": str(path),
        "digest": digest,
        "size": path.stat().st_size,
    }
    if role is not None:
        assert dtype is not None
        source = load_safetensors_header(
            path,
            asset_digest=digest,
            asset_size=path.stat().st_size,
        )
        plan = plan_minimax_music3_split_component(source, role=role, path=path)
        result["identity"] = minimax_music3_component_runtime_identity(plan, role, dtype)
    return result


def _write_dataset(root: Path) -> None:
    root.mkdir()
    samples = 44_100 // 25
    position = torch.arange(samples * 2, dtype=torch.float32)
    payload = torch.sin(position * 0.01).mul_(12_000).to(torch.int16).numpy().tobytes()
    with wave.open(str(root / "song.wav"), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(44_100)
        audio.writeframes(payload)
    (root / "song.caption.txt").write_text(
        "A clear electronic pop song with bright synthesizers.", encoding="utf-8"
    )
    (root / "song.lyrics.txt").write_text("[Verse]\nMusic in the morning light", encoding="utf-8")


def _config(
    tmp_path: Path,
    diffusion: Path,
    base_dtype: str,
    quantized: bool,
    identity_dtype: DType,
) -> MiniMaxMusic3TrainingConfig:
    dataset = tmp_path / "dataset"
    if not dataset.exists():
        _write_dataset(dataset)
    return MiniMaxMusic3TrainingConfig.from_mapping(
        {
            "schemaVersion": 1,
            "family": "minimax-music3",
            "communityRecipeRevision": ("SimpleTuner/def7bbc065e5f15d9e551827e247ba046fe36eb6"),
            "flowObjective": "community-minimax-music3-logistic-normal-flow-v1",
            "diffusionState": _artifact(
                diffusion,
                role="diffusion",
                dtype=identity_dtype,
            ),
            "dataset": {
                "type": "minimax-music3-audio-caption-lyrics-folder",
                "root": str(dataset),
                "audioFrames": 1,
                "davEncoderState": _artifact(_DAV),
                "rvqEncoderState": _artifact(_RVQ),
                "textEncoderState": _artifact(_TEXT, role="text", dtype=BFLOAT16),
                "encodedCacheRoot": str(tmp_path / "cache"),
            },
            "device": "cuda:0",
            "baseDtype": base_dtype,
            "textDtype": "bfloat16",
            "quantizedBase": quantized,
            "rank": 1,
            "alpha": 1.0,
            "learningRate": 0.0001,
            "weightDecay": 0.0,
            "gradientCheckpointing": True,
            "seed": 1176,
        }
    )


def _tree_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(
                _tree_equal(left_item, right_item)
                for left_item, right_item in zip(left, right, strict=True)
            )
        )
    return left == right


def test_community_dav_encoder_and_decoder_share_the_native_latent_contract() -> None:
    assert digest_file(_DAV) == (
        "blake3:ab71ad1a4706c5532ad34e2ea7af1027357e54ef1c8e2b1dfdf8d7d08fbd9df7"
    )
    pin = MiniMaxMusic3ArtifactPin(str(_DAV), digest_file(_DAV), _DAV.stat().st_size)
    encoder = load_minimax_music3_dav_encoder(pin).to("cuda")
    source = load_safetensors_header(_DAV)
    with torch.device("meta"):
        decoder = MiniMaxMusic3Dav()
    expected = set(decoder.state_dict())
    selected = {
        key for key in source.keys() if key.startswith("decoder.") or key.startswith("dec_in_proj.")
    }
    assert selected == expected
    decoder.load_state_dict(load_tensors(_DAV, tuple(sorted(selected))), strict=True, assign=True)
    decoder = decoder.requires_grad_(False).eval().to("cuda")

    samples = 44_100 // 25
    position = torch.arange(samples, device="cuda", dtype=torch.float32) / 44_100
    waveform = torch.stack(
        (
            0.1 * torch.sin(2 * math.pi * 440 * position),
            0.1 * torch.sin(2 * math.pi * 660 * position),
        )
    ).unsqueeze(0)
    with torch.no_grad():
        latents = encoder.encode(waveform)
        decoded = decoder.decode(latents)

    assert tuple(latents.shape) == (1, 128, 4)
    assert tuple(decoded.shape) == (1, 2, 2048)
    assert torch.isfinite(latents).all() and torch.isfinite(decoded).all()
    assert torch.count_nonzero(latents) and torch.count_nonzero(decoded)


def test_real_audio_conditioning_cache_and_all_native_diffusion_artifacts(
    tmp_path: Path,
) -> None:
    fp16, fp16_dtype, fp16_quantized, fp16_identity_dtype = _DIFFUSION[0]
    config = _config(tmp_path, fp16, fp16_dtype, fp16_quantized, fp16_identity_dtype)
    rss_before = _rss_bytes()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    source = default_minimax_music3_data_source_factory(config)
    torch.cuda.synchronize()
    cold_cache_seconds = time.perf_counter() - start
    batch = source.batch(
        0,
        generator=torch.Generator(device="cuda"),
        device=torch.device("cuda"),
    )
    assert tuple(batch.latents.shape) == config.latent_shape
    assert tuple(batch.context.shape) == config.context_shape
    assert torch.isfinite(batch.latents).all() and torch.isfinite(batch.context).all()
    assert torch.count_nonzero(batch.latents) and torch.count_nonzero(batch.context)
    del source, batch
    torch.cuda.empty_cache()

    def fail_dav() -> MiniMaxMusic3DavEncoder:
        raise AssertionError("a verified cache hit must not load a model")

    def fail_rvq() -> MiniMaxMusic3RvqEncoder:
        raise AssertionError("a verified cache hit must not load a model")

    def fail_text() -> tuple[MiniMaxMusic3TextModel, Tokenizer, torch.dtype]:
        raise AssertionError("a verified cache hit must not load a model")

    start = time.perf_counter()
    cached = MiniMaxMusic3DatasetSource(
        config.dataset,
        fail_dav,
        fail_rvq,
        fail_text,
        device=torch.device("cuda"),
    )
    warm_cache_seconds = time.perf_counter() - start
    del cached

    print(
        json.dumps(
            {
                "conditioningColdSeconds": cold_cache_seconds,
                "conditioningWarmSeconds": warm_cache_seconds,
                "conditioningPeakGpuBytes": torch.cuda.max_memory_allocated(),
                "conditioningHostRssDeltaBytes": _rss_bytes() - rss_before,
            },
            sort_keys=True,
        )
    )

    for diffusion, base_dtype, quantized, identity_dtype in _DIFFUSION:
        config = _config(tmp_path, diffusion, base_dtype, quantized, identity_dtype)
        torch.cuda.empty_cache()
        baseline_allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        rss_before = _rss_bytes()
        start = time.perf_counter()
        source = default_minimax_music3_data_source_factory(config)
        model = default_minimax_music3_model_factory(config)
        trainer = MiniMaxMusic3LoRATrainer(config, model, source)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start

        int8_layers = tuple(
            module for module in trainer.model.modules() if isinstance(module, Int8Linear)
        )
        assert bool(int8_layers) == quantized
        assert len(trainer.attachment.targets) == 146
        optimizer_ids = {
            id(parameter)
            for group in trainer.optimizer.param_groups
            for parameter in group["params"]
        }
        adapter_ids = {id(parameter) for parameter in trainer.attachment.parameters()}
        base_ids = {id(parameter) for parameter in trainer.model.parameters()}
        assert optimizer_ids == adapter_ids
        assert optimizer_ids.isdisjoint(base_ids)
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in trainer.model.parameters()
        )

        start = time.perf_counter()
        loss = trainer.train_step()
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - start
        adapter_parameters = trainer.attachment.parameters()
        assert math.isfinite(loss)
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in adapter_parameters
        )
        assert all(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in adapter_parameters[1::2]
        )
        assert len(trainer.optimizer.state) == len(adapter_parameters)
        assert all(
            not parameter.requires_grad and parameter.grad is None
            for parameter in trainer.model.parameters()
        )

        if base_dtype == "float16":
            checkpoint_adapter = trainer.attachment.state_dict()
            checkpoint_optimizer = deepcopy(trainer.optimizer_state_dict())
            checkpoint_rng = trainer.rng_state_dict()
            checkpoint_step = trainer.step_cursor
            checkpoint_cursor = trainer.data_cursor
            checkpoint_loss = trainer.last_loss
            expected_loss = trainer.train_step()
            expected_adapter = trainer.attachment.state_dict()
            expected_optimizer = deepcopy(trainer.optimizer_state_dict())
            trainer.restore(
                adapter=checkpoint_adapter,
                optimizer=checkpoint_optimizer,
                rng=checkpoint_rng,
                step_cursor=checkpoint_step,
                data_cursor=checkpoint_cursor,
                loss=checkpoint_loss,
            )
            assert trainer.train_step() == expected_loss
            assert _tree_equal(trainer.attachment.state_dict(), expected_adapter)
            assert _tree_equal(trainer.optimizer_state_dict(), expected_optimizer)
            _prove_inference_export_round_trip(trainer, source, tmp_path)
            del (
                checkpoint_adapter,
                checkpoint_optimizer,
                checkpoint_rng,
                expected_adapter,
                expected_optimizer,
            )

        print(
            json.dumps(
                {
                    "artifact": diffusion.name,
                    "loadSeconds": load_seconds,
                    "stepSeconds": step_seconds,
                    "loss": loss,
                    "peakGpuBytes": torch.cuda.max_memory_allocated(),
                    "hostRssDeltaBytes": _rss_bytes() - rss_before,
                },
                sort_keys=True,
            )
        )
        trainer.attachment.detach()
        del trainer, model, source, int8_layers, adapter_parameters
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() <= baseline_allocated + 64 * 1024 * 1024


def _prove_inference_export_round_trip(
    trainer: MiniMaxMusic3LoRATrainer,
    source: MiniMaxMusic3PreparedBatchSource,
    tmp_path: Path,
) -> None:
    export_source = LoraExportSource(
        None,
        "gpu-proof",
        "config-proof",
        trainer.step_cursor,
        trainer.attachment.state_dict(),
    )
    path, _digest = export_intermediate_lora(
        export_source,
        trainer.config,
        LoraExportSettings(tmp_path / "music3-lora.safetensors", "fp16"),
        trainer.attachment.targets,
        trainer.lora_export_model_state_keys,
        runtime_identity=MINIMAX_MUSIC3_TRAINING_RUNTIME_IDENTITY,
    )
    header = load_safetensors_header(path)
    tensors = load_tensors(path)
    model_keys = tuple(f"diffusion_model.{key}" for key in trainer.lora_export_model_state_keys)
    decoded = decode_lora(
        {key: header.entry(key).geometry for key in header.keys()},
        native_unet_key_map(model_keys),
    )
    assert not decoded.unmatched and not decoded.diagnostics
    assert len(decoded.patches) == 146
    patches = build_patch_set(decoded.patches, tensors)
    batch = source.batch(
        0,
        generator=torch.Generator(device="cuda"),
        device=torch.device("cuda"),
    )
    latent = batch.latents.to(torch.float16)
    context = batch.context.to(torch.float16)
    timestep = torch.tensor([0.5], device="cuda")
    scale = torch.ones((1, 1, 1), device="cuda", dtype=torch.float16)
    model = cast("MiniMaxMusic3DiT", trainer.model)
    with torch.no_grad():
        attached = model(latent, timestep, context, scale)
    trainer.attachment.detach()
    with torch.no_grad():
        for target in trainer.attachment.targets:
            module = cast("torch.nn.Linear", trainer.model.get_submodule(target.module_path))
            key = f"diffusion_model.{target.module_path}.weight"
            patched = apply_patches(module.weight.detach(), patches.entries(key), key=key)
            module.weight.copy_(patched)
        inference = model(latent, timestep, context, scale)
    assert torch.isfinite(attached).all() and torch.isfinite(inference).all()
    assert torch.allclose(attached, inference, rtol=0.002, atol=0.05)
