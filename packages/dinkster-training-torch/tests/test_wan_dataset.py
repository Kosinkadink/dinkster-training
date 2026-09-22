"""Executable proofs for the Wan 2.1 encoded dataset cache."""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import dinkster_training_torch.data as training_data
import dinkster_training_torch.service as training_service
import dinkster_training_torch.trainer as training_trainer
import pytest
import torch
from dinkster_assets import digest_bytes
from dinkster_inference import FLOAT32, ComponentPlan, Conditioning, Wan21Config
from dinkster_inference_torch import CastOperations
from dinkster_inference_torch.wan21_component import Wan21TextRuntime
from dinkster_inference_torch.wan21_model import Wan21Model
from dinkster_inference_torch.wan21_vae import LATENTS_MEAN, LATENTS_STD, WanVAE, WanVAEConfig
from dinkster_inference_torch.wan22_vae import Wan22VAE, Wan22VAEConfig
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    TrainingConfigError,
    WanDatasetSource,
    WanLoRATrainingService,
    WanPreparedBatchSource,
    WanTrainingConfig,
    wan_capability_report,
)
from dinkster_training_torch.data import load_planned_component
from dinkster_training_torch.dataset import (
    WanDatasetInspection,
    WanDatasetItem,
    WanDatasetSettings,
    inspect_wan_dataset,
)
from dinkster_training_torch.encoded_cache import (
    EncodedWanDatasetTensors,
    WanEncodedDatasetCache,
    wan_encoded_cache_key,
)


def _settings(
    tmp_path: Path,
    *,
    cache_name: str = "cache",
    vae_contract: Literal["wan21", "wan22"] = "wan21",
    conditioning_contract: Literal["none", "first-frame-i2v"] = "none",
) -> WanDatasetSettings:
    root = tmp_path / "dataset"
    root.mkdir(parents=True, exist_ok=True)
    video = root / "clip.mp4"
    caption = root / "clip.txt"
    video.write_bytes(b"video")
    caption.write_text("a moving subject", encoding="utf-8")
    item = WanDatasetItem(
        video_path=video,
        caption_path=caption,
        relative_video="clip.mp4",
        relative_caption="clip.txt",
        video_digest=digest_bytes(video.read_bytes()),
        caption_digest=digest_bytes(caption.read_bytes()),
    )
    inspection = WanDatasetInspection(
        digest=digest_bytes(b"wan-dataset"),
        items=(item,),
        caption_files=1,
        nonempty_captions=1,
        errors=(),
    )
    return WanDatasetSettings(
        root=str(root),
        resolution=(32, 32),
        frame_count=1,
        inspection=inspection,
        vae_contract=vae_contract,
        conditioning_contract=conditioning_contract,
        encoded_cache_root=str(tmp_path / cache_name),
    )


def _tensors() -> EncodedWanDatasetTensors:
    return EncodedWanDatasetTensors(
        latents=torch.linspace(-1.0, 1.0, 256).reshape(1, 16, 1, 4, 4),
        context=torch.linspace(-0.5, 0.5, 8192).reshape(1, 2, 4096),
    )


def _cache(settings: WanDatasetSettings) -> WanEncodedDatasetCache:
    channels, spatial = (48, 2) if settings.vae_contract == "wan22" else (16, 4)
    return WanEncodedDatasetCache(
        settings,
        latent_shape=(1, channels, 1, spatial, spatial),
        context_shape=(1, 2, 4096),
        device=torch.device("cpu"),
    )


def _write_video(path: Path, *, frame_count: int = 1) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w", format="mp4") as container:
        stream = cast(Any, container.add_stream("libx264", rate=16))
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        pixels = torch.arange(32 * 32 * 3, dtype=torch.uint8).reshape(32, 32, 3)
        for index in range(frame_count):
            frame = av.VideoFrame.from_ndarray(pixels.numpy(), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 16)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _artifact(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {"path": str(path), "digest": digest_bytes(payload), "size": len(payload)}


def _config_mapping(
    tmp_path: Path,
    *,
    variant: str = "wan21-t2v",
    expert: str | None = None,
    frame_count: int = 1,
) -> dict[str, object]:
    dataset = tmp_path / "videos"
    _write_video(dataset / "clip.mp4", frame_count=frame_count)
    (dataset / "clip.txt").write_text("a subject moving", encoding="utf-8")
    mapping: dict[str, object] = {
        "family": "wan",
        "variant": variant,
        "ditState": _artifact(tmp_path / "dit.safetensors", b"dit"),
        "umt5xxlState": _artifact(tmp_path / "umt5.safetensors", b"umt5"),
        "vaeState": _artifact(tmp_path / "vae.safetensors", b"vae"),
        "device": "cpu",
        "baseDtype": "bfloat16",
        "rank": 2,
        "alpha": 2.0,
        "latentShape": (
            [1, 48, 1 + (frame_count - 1) // 4, 2, 2]
            if variant == "wan22-ti2v-5b"
            else [1, 16, 1 + (frame_count - 1) // 4, 4, 4]
        ),
        "contextShape": [1, 2, 4096],
        "dataset": {
            "type": (
                "wan22-video-caption-folder"
                if variant == "wan22-ti2v-5b"
                else "wan21-video-caption-folder"
            ),
            "root": str(dataset),
            "resolution": [32, 32],
            "frameCount": frame_count,
            "encodedCacheRoot": str(tmp_path / "encoded-cache"),
        },
    }
    if expert is not None:
        mapping["expert"] = expert
    return mapping


def test_wan_encoded_cache_is_deterministic_and_digest_verified(tmp_path: Path) -> None:
    first = _cache(_settings(tmp_path, cache_name="first"))
    second = _cache(_settings(tmp_path, cache_name="second"))

    first_digest = first.write(_tensors())
    second_digest = second.write(_tensors())

    assert first_digest == second_digest
    first_loaded = first.load()
    second_loaded = second.load()
    assert first_loaded is not None and second_loaded is not None
    assert torch.equal(first_loaded.latents, second_loaded.latents)
    assert torch.equal(first_loaded.context, second_loaded.context)
    manifest = next((tmp_path / "first/wan21-v1/entries").glob("*.json"))
    assert json.loads(manifest.read_text(encoding="ascii"))["shardDigest"] == first_digest
    shard = next((tmp_path / "first/wan21-v1/shards").glob("*.safetensors"))
    corrupted = bytearray(shard.read_bytes())
    corrupted[-1] ^= 1
    shard.write_bytes(corrupted)
    assert first.load() is None


def test_wan22_encoded_cache_uses_separate_verified_contract(tmp_path: Path) -> None:
    cache = _cache(_settings(tmp_path, vae_contract="wan22"))
    tensors = EncodedWanDatasetTensors(
        latents=torch.linspace(-1.0, 1.0, 192).reshape(1, 48, 1, 2, 2),
        context=torch.linspace(-0.5, 0.5, 8192).reshape(1, 2, 4096),
    )

    digest = cache.write(tensors)
    loaded = cache.load()

    assert loaded is not None
    assert torch.equal(loaded.latents, tensors.latents)
    assert torch.equal(loaded.context, tensors.context)
    manifest = next((tmp_path / "cache/wan22-v1/entries").glob("*.json"))
    assert json.loads(manifest.read_text(encoding="ascii"))["shardDigest"] == digest


def test_wan22_i2v_encoded_cache_carries_first_frame_conditioning(tmp_path: Path) -> None:
    settings = _settings(tmp_path, conditioning_contract="first-frame-i2v")
    cache = _cache(settings)
    tensors = EncodedWanDatasetTensors(
        latents=_tensors().latents,
        context=_tensors().context,
        i2v_conditioning=torch.linspace(-1.0, 1.0, 320).reshape(1, 20, 1, 4, 4),
    )

    cache.write(tensors)
    loaded = cache.load()

    assert loaded is not None and loaded.i2v_conditioning is not None
    expected_conditioning = tensors.i2v_conditioning
    assert expected_conditioning is not None
    assert torch.equal(loaded.i2v_conditioning, expected_conditioning)
    assert next((tmp_path / "cache/wan22-i2v-v1/entries").glob("*.json")).is_file()
    with pytest.raises(ValueError, match="I2V conditioning"):
        cache.write(_tensors())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("latents", torch.zeros((1, 15, 1, 4, 4)), "latent tensor"),
        ("context", torch.zeros((1, 1, 4096)), "context tensor"),
        ("i2v_conditioning", torch.zeros((1, 19, 1, 4, 4)), "I2V conditioning"),
        (
            "i2v_conditioning",
            torch.zeros((1, 20, 1, 4, 4), dtype=torch.float64),
            "I2V conditioning",
        ),
    ],
)
def test_wan_encoded_cache_rejects_malformed_tensors_before_publishing(
    tmp_path: Path,
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    cache = _cache(_settings(tmp_path, conditioning_contract="first-frame-i2v"))
    tensors = EncodedWanDatasetTensors(
        latents=_tensors().latents,
        context=_tensors().context,
        i2v_conditioning=torch.zeros((1, 20, 1, 4, 4)),
    )

    with pytest.raises(ValueError, match=message):
        cache.write(replace(tensors, **{field: value}))

    assert not (tmp_path / "cache").exists()


def test_wan_encoded_cache_identity_rotates_with_dataset_and_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    base = wan_encoded_cache_key(
        settings,
        latent_shape=(1, 16, 1, 4, 4),
        context_shape=(1, 2, 4096),
        device=torch.device("cpu"),
    )
    changed_dataset = replace(
        settings.inspection,
        digest=digest_bytes(b"changed-wan-dataset"),
    )
    dataset_key = wan_encoded_cache_key(
        replace(settings, inspection=changed_dataset),
        latent_shape=(1, 16, 1, 4, 4),
        context_shape=(1, 2, 4096),
        device=torch.device("cpu"),
    )
    context_key = wan_encoded_cache_key(
        settings,
        latent_shape=(1, 16, 1, 4, 4),
        context_shape=(1, 3, 4096),
        device=torch.device("cpu"),
    )
    wan22_key = wan_encoded_cache_key(
        replace(settings, vae_contract="wan22"),
        latent_shape=(1, 48, 1, 2, 2),
        context_shape=(1, 2, 4096),
        device=torch.device("cpu"),
    )
    i2v_key = wan_encoded_cache_key(
        replace(settings, conditioning_contract="first-frame-i2v"),
        latent_shape=(1, 16, 1, 4, 4),
        context_shape=(1, 2, 4096),
        device=torch.device("cpu"),
    )

    assert base != dataset_key
    assert base != context_key
    assert base != wan22_key
    assert base != i2v_key


def test_wan_dataset_config_uses_video_caption_and_encoder_content_identity(tmp_path: Path) -> None:
    mapping = _config_mapping(tmp_path)
    first = WanTrainingConfig.from_mapping(mapping)
    round_tripped = WanTrainingConfig.from_mapping(first.to_mapping())

    assert first.dataset is not None
    assert round_tripped.dataset_identity == first.dataset_identity
    caption = Path(first.dataset.root) / "clip.txt"
    caption.write_text("a different motion", encoding="utf-8")
    changed = WanTrainingConfig.from_mapping(mapping)
    assert changed.dataset_identity != first.dataset_identity


def test_wan21_dataset_identity_digest_stays_stable(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "clip.mp4").write_bytes(b"video")
    (root / "clip.txt").write_text("caption", encoding="utf-8")

    inspection = inspect_wan_dataset(
        root,
        (32, 32),
        1,
        vae_identity={"digest": "blake3:" + "1" * 64, "size": 1},
        umt5xxl_identity={"digest": "blake3:" + "2" * 64, "size": 2},
        validate_media=False,
    )

    assert inspection.digest == (
        "blake3:e66725d254a77ef358fc6fd877e99e91084d7b1b235583f3479a53f2a5d00cab"
    )


@pytest.mark.parametrize(
    ("variant", "expert", "vae_contract", "conditioning_contract", "latent_shape"),
    [
        ("wan22-ti2v-5b", None, "wan22", "none", (1, 48, 1, 2, 2)),
        ("wan22-t2v-14b", "high-noise", "wan21", "none", (1, 16, 1, 4, 4)),
        (
            "wan22-i2v-14b",
            "low-noise",
            "wan21",
            "first-frame-i2v",
            (1, 16, 1, 4, 4),
        ),
    ],
)
def test_wan22_dataset_config_uses_matching_vae_contract(
    tmp_path: Path,
    variant: str,
    expert: str | None,
    vae_contract: str,
    conditioning_contract: str,
    latent_shape: tuple[int, int, int, int, int],
) -> None:
    config = WanTrainingConfig.from_mapping(
        _config_mapping(tmp_path, variant=variant, expert=expert)
    )

    assert config.dataset is not None
    assert config.dataset.vae_contract == vae_contract
    assert config.dataset.conditioning_contract == conditioning_contract
    assert config.latent_shape == latent_shape
    assert WanTrainingConfig.from_mapping(config.to_mapping()) == config


def test_wan22_ti2v_dataset_requires_sixteen_pixel_resolution_units(tmp_path: Path) -> None:
    mapping = _config_mapping(tmp_path, variant="wan22-ti2v-5b")
    dataset = cast("dict[str, object]", mapping["dataset"])
    dataset["resolution"] = [24, 24]
    mapping["latentShape"] = [1, 48, 1, 1, 1]

    with pytest.raises(TrainingConfigError, match="divisible by 16"):
        WanTrainingConfig.from_mapping(mapping)


def test_wan22_dataset_identity_differs_from_wan21_for_same_encoder_pins(tmp_path: Path) -> None:
    wan21 = WanTrainingConfig.from_mapping(_config_mapping(tmp_path))
    wan22 = WanTrainingConfig.from_mapping(_config_mapping(tmp_path, variant="wan22-ti2v-5b"))

    assert wan21.dataset_identity != wan22.dataset_identity


def test_wan22_direct_config_rejects_mismatched_dataset_vae_contract(tmp_path: Path) -> None:
    config = WanTrainingConfig.from_mapping(_config_mapping(tmp_path, variant="wan22-ti2v-5b"))
    assert config.dataset is not None

    with pytest.raises(TrainingConfigError, match="dataset VAE contract"):
        replace(config, dataset=replace(config.dataset, vae_contract="wan21"))


def test_wan22_direct_config_rejects_mismatched_dataset_conditioning_contract(
    tmp_path: Path,
) -> None:
    config = WanTrainingConfig.from_mapping(
        _config_mapping(tmp_path, variant="wan22-i2v-14b", expert="high-noise")
    )
    assert config.dataset is not None

    with pytest.raises(TrainingConfigError, match="dataset conditioning contract"):
        replace(config, dataset=replace(config.dataset, conditioning_contract="none"))


@pytest.mark.parametrize(
    ("variant", "expert"),
    [
        ("wan21-t2v", None),
        ("wan22-ti2v-5b", None),
        ("wan22-t2v-14b", "high-noise"),
        ("wan22-i2v-14b", "low-noise"),
    ],
)
def test_service_builds_wan_dataset_before_loading_dit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expert: str | None,
) -> None:
    mapping = _config_mapping(tmp_path, variant=variant, expert=expert)
    events: list[str] = []

    def report(config: WanTrainingConfig) -> dict[str, object]:
        assert config.dataset is not None
        return {"dataset": {"errors": list(config.dataset.inspection.errors)}}

    monkeypatch.setattr(training_service, "wan_capability_report", report)

    def data_factory(config: WanTrainingConfig) -> WanPreparedBatchSource:
        assert config.dataset is not None
        events.append("dataset")
        return cast(WanPreparedBatchSource, object())

    def model_factory(config: WanTrainingConfig) -> Wan21Model:
        del config
        assert events == ["dataset"]
        events.append("dit")
        return Wan21Model(
            Wan21Config(
                hidden_size=12,
                ffn_hidden_size=16,
                num_heads=2,
                num_layers=1,
                text_dim=4096,
                time_freq_dim=4,
            )
        )

    store = TrainingSessionStore(JournalStore(tmp_path / "training.sqlite"))
    try:
        service = WanLoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_factory,
            data_source_factory=data_factory,
        )
        service.create("precompute-order", json.dumps(mapping))
        assert events == ["dataset", "dit"]
    finally:
        store.close()


def test_wan_dataset_report_includes_precompute_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = WanTrainingConfig.from_mapping(_config_mapping(tmp_path))
    model_config = Wan21Config(
        hidden_size=12,
        ffn_hidden_size=16,
        num_heads=2,
        num_layers=1,
        text_dim=4096,
        time_freq_dim=4,
    )

    def assembly_plan(config: WanTrainingConfig) -> object:
        del config
        return SimpleNamespace(diffusion=SimpleNamespace(config=model_config))

    def component_bytes(plan: object, config: WanTrainingConfig) -> dict[str, int]:
        del plan, config
        return {
            "vaeEncoderParametersTransientLowerBound": 120,
            "umt5xxlTextEncoderParametersTransientLowerBound": 240,
        }

    monkeypatch.setattr(training_service, "wan_model_assembly_plan", assembly_plan)
    monkeypatch.setattr(
        training_service,
        "_wan_component_float32_bytes",
        component_bytes,
    )

    report = wan_capability_report(config)

    memory = cast("dict[str, object]", report["memoryLedger"])
    categories = cast("dict[str, int]", memory["categories"])
    assert categories["vaeEncoderParametersTransientLowerBound"] == 120
    assert categories["umt5xxlTextEncoderParametersTransientLowerBound"] == 240
    assert categories["encoderInputVideoTransientLowerBound"] == 3 * 32 * 32 * 4
    assert categories["encodedDatasetStoreResidentCpu"] > 0
    assert "released before the DiT loads" in cast("str", memory["datasetEncoderEstimateBasis"])


class _TinyTextRuntime:
    def encode_text(self, text: str) -> Conditioning[torch.Tensor]:
        value = sum(text.encode("utf-8")) / 10_000.0
        return Conditioning(torch.full((1, 2, 4096), value, dtype=torch.float32))


def _tiny_low_precision_vae() -> WanVAE:
    model = WanVAE(
        WanVAEConfig(
            dim=32,
            dim_mult=(1, 1, 1, 1),
            num_res_blocks=1,
            temporal_downsample=(False, True, True),
        ),
        operations=CastOperations(torch.float32),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.001)
    return model.to(dtype=torch.bfloat16)


def _tiny_wan22_vae() -> Wan22VAE:
    model = Wan22VAE(
        Wan22VAEConfig(
            dim=2,
            decoder_dim=2,
            z_dim=48,
            num_res_blocks=1,
        ),
        operations=CastOperations(torch.float32),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.001)
    return model


def _write_float32_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for key, tensor in tensors.items():
        value = tensor.detach().contiguous()
        data = bytes(value.untyped_storage())[: value.numel() * value.element_size()]
        header[key] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("ascii")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))


def _load_planned_tiny_vae(tmp_path: Path) -> WanVAE:
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = WanVAEConfig(
        dim=32,
        dim_mult=(1, 1, 1, 1),
        num_res_blocks=1,
        temporal_downsample=(False, True, True),
    )
    source = WanVAE(config, operations=CastOperations(torch.float32))
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.001)
    state = dict(source.state_dict())
    path = tmp_path / "tiny-vae.safetensors"
    _write_float32_safetensors(path, state)
    plan = ComponentPlan(
        component="vae",
        path=path,
        config=config,
        keys={key: key for key in state},
        dtypes={key: FLOAT32 for key in state},
        quant={},
        absent=(),
        transforms={},
    )
    return load_planned_component(
        plan,
        lambda value: WanVAE(value, operations=CastOperations(torch.float32)),
    )


def test_planned_wan_vae_materializes_nonpersistent_buffers(tmp_path: Path) -> None:
    loaded = _load_planned_tiny_vae(tmp_path)

    assert not any(parameter.is_meta for parameter in loaded.parameters())
    assert not any(buffer.is_meta for buffer in loaded.buffers())
    assert torch.equal(loaded.latents_mean, torch.tensor(LATENTS_MEAN, dtype=torch.float32))
    assert torch.equal(loaded.latents_std, torch.tensor(LATENTS_STD, dtype=torch.float32))


def test_wan_dataset_source_moves_planned_vae_to_device(tmp_path: Path) -> None:
    config = WanTrainingConfig.from_mapping(_config_mapping(tmp_path))
    assert config.dataset is not None
    loaded = _load_planned_tiny_vae(tmp_path / "planned")
    source = WanDatasetSource(
        config.dataset,
        lambda: loaded,
        lambda: cast(Wan21TextRuntime, _TinyTextRuntime()),
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        vae_identity={"digest": config.vae_state.digest, "size": config.vae_state.size},
        umt5xxl_identity={
            "digest": config.umt5xxl_state.digest,
            "size": config.umt5xxl_state.size,
        },
        device=torch.device("cpu"),
    )

    batch = source.batch(
        0,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )
    assert batch.latents.shape == config.latent_shape
    assert batch.context.shape == config.context_shape


def test_wan22_dataset_source_encodes_48_channel_latents(tmp_path: Path) -> None:
    config = WanTrainingConfig.from_mapping(_config_mapping(tmp_path, variant="wan22-ti2v-5b"))
    assert config.dataset is not None
    source = WanDatasetSource(
        config.dataset,
        _tiny_wan22_vae,
        lambda: cast(Wan21TextRuntime, _TinyTextRuntime()),
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        vae_identity={"digest": config.vae_state.digest, "size": config.vae_state.size},
        umt5xxl_identity={
            "digest": config.umt5xxl_state.digest,
            "size": config.umt5xxl_state.size,
        },
        device=torch.device("cpu"),
    )

    batch = source.batch(
        0,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )

    assert batch.latents.shape == (1, 48, 1, 2, 2)
    assert batch.context.shape == config.context_shape


def test_wan22_i2v_dataset_source_encodes_first_frame_conditioning(tmp_path: Path) -> None:
    config = WanTrainingConfig.from_mapping(
        _config_mapping(
            tmp_path,
            variant="wan22-i2v-14b",
            expert="high-noise",
            frame_count=5,
        )
    )
    assert config.dataset is not None
    source = WanDatasetSource(
        config.dataset,
        _tiny_low_precision_vae,
        lambda: cast(Wan21TextRuntime, _TinyTextRuntime()),
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        vae_identity={"digest": config.vae_state.digest, "size": config.vae_state.size},
        umt5xxl_identity={
            "digest": config.umt5xxl_state.digest,
            "size": config.umt5xxl_state.size,
        },
        device=torch.device("cpu"),
    )

    batch = source.batch(
        0,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )

    assert batch.i2v_conditioning is not None
    assert batch.i2v_conditioning.shape == (1, 20, 2, 4, 4)
    expected_mask = torch.zeros((1, 4, 2, 4, 4))
    expected_mask[:, :, :1] = 1.0
    assert torch.equal(batch.i2v_conditioning[:, :4], expected_mask)

    def forbidden() -> WanVAE:
        raise AssertionError("cache hit loaded the VAE")

    cached = WanDatasetSource(
        config.dataset,
        forbidden,
        lambda: cast(Wan21TextRuntime, forbidden()),
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        vae_identity={"digest": config.vae_state.digest, "size": config.vae_state.size},
        umt5xxl_identity={
            "digest": config.umt5xxl_state.digest,
            "size": config.umt5xxl_state.size,
        },
        device=torch.device("cpu"),
    ).batch(
        0,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    assert cached.i2v_conditioning is not None
    assert torch.equal(cached.i2v_conditioning, batch.i2v_conditioning)


def test_wan22_default_dataset_factory_loads_planned_wan22_vae(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = WanTrainingConfig.from_mapping(_config_mapping(tmp_path, variant="wan22-ti2v-5b"))
    vae_config = Wan22VAEConfig(dim=2, decoder_dim=2, z_dim=48, num_res_blocks=1)
    vae_plan = SimpleNamespace(config=vae_config)
    plan = SimpleNamespace(vae=vae_plan)
    loaded: list[type[torch.nn.Module]] = []

    def assembly_plan(_config: WanTrainingConfig) -> object:
        return plan

    monkeypatch.setattr(training_trainer, "wan_model_assembly_plan", assembly_plan)

    def load_component(component: object, factory: Any) -> torch.nn.Module:
        assert component is vae_plan
        return factory(vae_config)

    def dataset_source(
        _settings: object,
        vae_factory: Any,
        _text_factory: object,
        **_kwargs: object,
    ) -> WanPreparedBatchSource:
        vae = vae_factory()
        loaded.append(type(vae))
        return cast(WanPreparedBatchSource, object())

    monkeypatch.setattr(training_trainer, "load_planned_component", load_component)
    monkeypatch.setattr(training_trainer, "WanDatasetSource", dataset_source)

    assert training_trainer.default_wan_data_source_factory(config) is not None
    assert loaded == [Wan22VAE]


def test_wan_dataset_source_encodes_once_then_serves_verified_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WanTrainingConfig.from_mapping(_config_mapping(tmp_path))
    assert config.dataset is not None

    def text_factory() -> Wan21TextRuntime:
        return cast(Wan21TextRuntime, _TinyTextRuntime())

    vae_storage_dtypes: list[torch.dtype] = []

    def vae_factory() -> WanVAE:
        vae = _tiny_low_precision_vae()
        vae_storage_dtypes.append(next(vae.parameters()).dtype)
        return vae

    source = WanDatasetSource(
        config.dataset,
        vae_factory,
        text_factory,
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        vae_identity={"digest": config.vae_state.digest, "size": config.vae_state.size},
        umt5xxl_identity={
            "digest": config.umt5xxl_state.digest,
            "size": config.umt5xxl_state.size,
        },
        device=torch.device("cpu"),
    )
    expected = source.batch(
        0,
        generator=torch.Generator().manual_seed(0),
        device=torch.device("cpu"),
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cache hit loaded an encoder or decoder")

    def forbidden_vae() -> WanVAE:
        raise AssertionError("cache hit loaded the VAE")

    def forbidden_text() -> Wan21TextRuntime:
        raise AssertionError("cache hit loaded the text encoder")

    monkeypatch.setattr(training_data, "decode_video_rgb24", forbidden)
    cached = WanDatasetSource(
        config.dataset,
        forbidden_vae,
        forbidden_text,
        latent_shape=config.latent_shape,
        context_shape=config.context_shape,
        vae_identity={"digest": config.vae_state.digest, "size": config.vae_state.size},
        umt5xxl_identity={
            "digest": config.umt5xxl_state.digest,
            "size": config.umt5xxl_state.size,
        },
        device=torch.device("cpu"),
    ).batch(
        0,
        generator=torch.Generator().manual_seed(99),
        device=torch.device("cpu"),
    )

    assert expected.latents.shape == config.latent_shape
    assert expected.context.shape == config.context_shape
    assert expected.latents.dtype == expected.context.dtype == torch.float32
    assert vae_storage_dtypes == [torch.bfloat16]
    assert torch.equal(cached.latents, expected.latents)
    assert torch.equal(cached.context, expected.context)
