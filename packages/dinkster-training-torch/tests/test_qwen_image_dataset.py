"""Qwen-Image dataset encoding, identity, and cache proofs."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import dinkster_training_torch.data as training_data
import dinkster_training_torch.dataset as training_dataset
import dinkster_training_torch.encoded_cache as encoded_cache
import dinkster_training_torch.trainer as training_trainer
import pytest
import torch
import torch.nn.functional as functional
from dinkster_assets import digest_bytes
from dinkster_inference import QWEN_IMAGE_TEXT_CONFIG, WAN21_VAE_CONFIG, load_qwen_bpe
from dinkster_inference.qwen_image_text import (
    format_qwen_image_prompt,
    select_qwen_image_output,
)
from dinkster_inference_torch.qwen_image_text import QwenImageTextModel
from dinkster_inference_torch.sources import load_tensors
from dinkster_inference_torch.wan21_vae import WanVAE
from dinkster_training_torch import (
    QwenImageDatasetSource,
    QwenImageTrainingConfig,
    default_qwen_image_data_source_factory,
)
from dinkster_training_torch.config import TrainingConfigError
from dinkster_training_torch.dataset import (
    DatasetInspection,
    QwenImageDatasetSettings,
    inspect_qwen_image_dataset,
)
from dinkster_training_torch.encoded_cache import (
    EncodedQwenImageDatasetTensors,
    QwenImageEncodedDatasetCache,
    qwen_image_encoded_cache_key,
)
from PIL import Image

_VAE_IDENTITY = {"digest": "blake3:" + "1" * 64, "size": 101}
_TEXT_IDENTITY = {"digest": "blake3:" + "2" * 64, "size": 202}
_LATENT_SHAPE = (1, 16, 1, 2, 2)
_CONTEXT_SHAPE = (1, 512, 3584)
_MASK_SHAPE = (1, 512)


def _write_dataset(root: Path, caption: str = "a small red cube") -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (23, 19)).save(root / "item.png")
    (root / "item.txt").write_text(caption, encoding="utf-8")


def _inspection(
    root: Path,
    *,
    resolution: tuple[int, int] = (16, 16),
    context_tokens: int = 512,
    vae_identity: object = _VAE_IDENTITY,
    text_identity: object = _TEXT_IDENTITY,
) -> DatasetInspection:
    result = inspect_qwen_image_dataset(
        root,
        resolution,
        context_tokens,
        vae_identity=vae_identity,
        text_encoder_identity=text_identity,
    )
    assert not result.errors
    return result


def _settings(
    tmp_path: Path,
    *,
    context_tokens: int = 512,
    caption: str = "a small red cube",
    cache: bool = False,
) -> QwenImageDatasetSettings:
    root = tmp_path / "dataset"
    _write_dataset(root, caption)
    return QwenImageDatasetSettings(
        root=str(root.resolve()),
        resolution=(16, 16),
        context_tokens=context_tokens,
        inspection=_inspection(root, context_tokens=context_tokens),
        encoded_cache_root=str((tmp_path / "cache").resolve()) if cache else None,
    )


def _artifact(path: Path, payload: bytes) -> dict[str, object]:
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "digest": digest_bytes(payload), "size": len(payload)}


def _config_mapping(tmp_path: Path, *, context_tokens: int = 640) -> dict[str, object]:
    root = tmp_path / "config-dataset"
    _write_dataset(root)
    return {
        "schemaVersion": 1,
        "family": "qwen-image",
        "variant": "qwen-image",
        "ditState": _artifact(tmp_path / "dit.safetensors", b"dit"),
        "textEncoderState": _artifact(tmp_path / "text.safetensors", b"text"),
        "vaeState": _artifact(tmp_path / "vae.safetensors", b"vae"),
        "dataset": {
            "type": "qwen-image-image-caption-folder",
            "root": str(root.resolve()),
            "resolution": [16, 16],
            "contextTokens": context_tokens,
            "encodedCacheRoot": str((tmp_path / "encoded").resolve()),
        },
        "device": "cpu",
        "baseDtype": "float32",
        "rank": 2,
        "alpha": 2.0,
        "loraTargets": ["attention.qkvo", "mlp.projections"],
        "gradientCheckpointing": False,
        "latentShape": [1, 16, 1, 2, 2],
        "contextShape": [1, context_tokens, 3584],
        "attentionMaskShape": [1, context_tokens],
    }


class _FakeWanVAE(torch.nn.Module):
    def __init__(self, seen: list[torch.Tensor]) -> None:
        super().__init__()
        self._seen = seen

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        self._seen.append(content.detach().cpu())
        spatial = functional.interpolate(content[:, :, 0], size=(2, 2), mode="bilinear")
        return spatial.mean(dim=1, keepdim=True).unsqueeze(2).repeat(1, 16, 1, 1, 1)

    @staticmethod
    def process_in(latent: torch.Tensor) -> torch.Tensor:
        return latent + 0.25


class _FakeQwenImageText(torch.nn.Module):
    def __init__(self, calls: list[tuple[torch.Tensor, torch.Tensor | None]]) -> None:
        super().__init__()
        self._calls = calls

    def forward(
        self, ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._calls.append((ids.detach().cpu(), attention_mask))
        row = cast("list[list[int]]", ids.detach().cpu().tolist())
        selection = select_qwen_image_output(row, None)
        positions = torch.arange(ids.shape[1], device=ids.device, dtype=torch.bfloat16)
        hidden = positions.reshape(1, -1, 1).expand(1, -1, 3584)
        return hidden[:, selection.slice_start :], None


def _factories(
    events: list[str],
    seen_pixels: list[torch.Tensor],
    text_calls: list[tuple[torch.Tensor, torch.Tensor | None]],
) -> tuple[Callable[[], WanVAE], Callable[[], QwenImageTextModel]]:
    def vae_factory() -> WanVAE:
        events.append("vae")
        return cast("WanVAE", _FakeWanVAE(seen_pixels))

    def text_factory() -> QwenImageTextModel:
        events.append("text")
        return cast("QwenImageTextModel", _FakeQwenImageText(text_calls))

    return vae_factory, text_factory


def _source(
    settings: QwenImageDatasetSettings,
    vae_factory: Callable[[], WanVAE],
    text_factory: Callable[[], QwenImageTextModel],
) -> QwenImageDatasetSource:
    tokens = settings.context_tokens
    return QwenImageDatasetSource(
        settings,
        vae_factory,
        text_factory,
        latent_shape=_LATENT_SHAPE,
        context_shape=(1, tokens, 3584),
        attention_mask_shape=(1, tokens),
        vae_identity=_VAE_IDENTITY,
        text_encoder_identity=_TEXT_IDENTITY,
        device=torch.device("cpu"),
    )


def _cache(
    settings: QwenImageDatasetSettings,
) -> QwenImageEncodedDatasetCache:
    return QwenImageEncodedDatasetCache(
        settings,
        latent_shape=_LATENT_SHAPE,
        context_shape=_CONTEXT_SHAPE,
        attention_mask_shape=_MASK_SHAPE,
        device=torch.device("cpu"),
    )


def _cache_values() -> EncodedQwenImageDatasetTensors:
    return EncodedQwenImageDatasetTensors(
        torch.linspace(-1.0, 1.0, 64).reshape(1, 16, 1, 2, 2),
        torch.arange(512, dtype=torch.float32).reshape(1, 512, 1).expand(-1, -1, 3584),
        functional.pad(torch.ones((1, 37)), (0, 475)),
    )


def _wrong_latent_dtype(
    value: EncodedQwenImageDatasetTensors,
) -> EncodedQwenImageDatasetTensors:
    return EncodedQwenImageDatasetTensors(
        value.latents.to(torch.bfloat16), value.context, value.attention_mask
    )


def _wrong_context_geometry(
    value: EncodedQwenImageDatasetTensors,
) -> EncodedQwenImageDatasetTensors:
    return EncodedQwenImageDatasetTensors(
        value.latents, value.context[:, :-1], value.attention_mask
    )


def _nonbinary_mask(
    value: EncodedQwenImageDatasetTensors,
) -> EncodedQwenImageDatasetTensors:
    return EncodedQwenImageDatasetTensors(
        value.latents, value.context, value.attention_mask.fill_(0.5)
    )


def test_qwen_image_dataset_config_round_trips_fixed_geometry_above_minimum(
    tmp_path: Path,
) -> None:
    config = QwenImageTrainingConfig.from_mapping(_config_mapping(tmp_path))

    assert config.dataset is not None
    assert config.dataset.context_tokens == 640
    assert config.dataset.resolution == (16, 16)
    assert config.dataset_identity == config.dataset.digest
    assert QwenImageTrainingConfig.from_mapping(config.to_mapping()) == config
    identity_dataset = cast("dict[str, object]", config.identity_mapping()["dataset"])
    assert "encodedCacheRoot" not in identity_dataset


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("type", True, "dataset.type"),
        ("resolution", [16.0, 16], "dataset.resolution.*integer"),
        ("resolution", [16, 24], "divisible by 16"),
        ("contextTokens", True, "dataset.contextTokens must be an integer"),
        ("contextTokens", 511, ">= 512"),
    ),
)
def test_qwen_image_dataset_config_rejects_wrong_exact_types(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    mapping = _config_mapping(tmp_path)
    dataset = cast("dict[str, object]", mapping["dataset"])
    dataset[field] = value

    with pytest.raises(TrainingConfigError, match=message):
        QwenImageTrainingConfig.from_mapping(mapping)


def test_qwen_image_dataset_config_rejects_latent_geometry_mismatch(tmp_path: Path) -> None:
    mapping = _config_mapping(tmp_path)
    mapping["latentShape"] = [1, 16, 1, 4, 2]

    with pytest.raises(TrainingConfigError, match="encoded dataset geometry"):
        QwenImageTrainingConfig.from_mapping(mapping)


def test_default_qwen_image_data_factory_pins_encoder_roles_and_compute_dtypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = QwenImageTrainingConfig.from_mapping(_config_mapping(tmp_path))
    plan = SimpleNamespace(
        vae=SimpleNamespace(component="vae", config=WAN21_VAE_CONFIG),
        qwen2_5_vl_7b=SimpleNamespace(component="qwen2_5_vl_7b", config=QWEN_IMAGE_TEXT_CONFIG),
    )
    events: list[tuple[str, object]] = []

    def fake_load(component: object, factory: Callable[[object], object]) -> object:
        typed = cast("SimpleNamespace", component)
        events.append(("load", typed.component))
        return factory(typed.config)

    def fake_vae(_config: object, *, operations: object) -> WanVAE:
        events.append(("vae-dtype", cast("SimpleNamespace", operations).dtype))
        return cast("WanVAE", torch.nn.Identity())

    def fake_text(*, operations: object) -> QwenImageTextModel:
        events.append(("text-dtype", cast("SimpleNamespace", operations).dtype))
        return cast("QwenImageTextModel", torch.nn.Identity())

    def fake_source(
        settings: QwenImageDatasetSettings,
        vae_factory: Callable[[], WanVAE],
        text_encoder_factory: Callable[[], QwenImageTextModel],
        *,
        latent_shape: tuple[int, int, int, int, int],
        context_shape: tuple[int, int, int],
        attention_mask_shape: tuple[int, int],
        vae_identity: object,
        text_encoder_identity: object,
        device: torch.device,
    ) -> QwenImageDatasetSource:
        assert settings is config.dataset
        assert latent_shape == config.latent_shape
        assert context_shape == config.context_shape
        assert attention_mask_shape == config.attention_mask_shape
        assert vae_identity == {"digest": config.vae_state.digest, "size": config.vae_state.size}
        assert text_encoder_identity == {
            "digest": config.text_encoder_state.digest,
            "size": config.text_encoder_state.size,
        }
        assert device == torch.device("cpu")
        vae_factory()
        text_encoder_factory()
        return cast("QwenImageDatasetSource", object())

    def provide_plan(_config: QwenImageTrainingConfig) -> SimpleNamespace:
        return plan

    monkeypatch.setattr(training_trainer, "qwen_image_model_assembly_plan", provide_plan)
    monkeypatch.setattr(training_trainer, "load_planned_component", fake_load)
    monkeypatch.setattr(training_trainer, "WanVAE", fake_vae)
    monkeypatch.setattr(training_trainer, "QwenImageTextModel", fake_text)
    monkeypatch.setattr(training_trainer, "QwenImageDatasetSource", fake_source)

    default_qwen_image_data_source_factory(config)

    assert events == [
        ("load", "vae"),
        ("vae-dtype", torch.float32),
        ("load", "qwen2_5_vl_7b"),
        ("text-dtype", torch.bfloat16),
    ]


def test_qwen_image_cache_key_covers_every_dataset_identity_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "identity"
    _write_dataset(root)
    baseline = _inspection(root)
    inspections = [baseline]
    constants = (
        "QWEN_IMAGE_FAMILY_IDENTITY",
        "QWEN_IMAGE_PREPROCESSING_IDENTITY",
        "QWEN_IMAGE_TOKENIZER_IDENTITY",
        "QWEN_IMAGE_LATENT_IDENTITY",
        "QWEN_IMAGE_CONTEXT_IDENTITY",
        "QWEN_IMAGE_PADDING_IDENTITY",
        "QWEN_IMAGE_POSITION_IDENTITY",
        "QWEN_IMAGE_MASK_IDENTITY",
        "QWEN_IMAGE_TEXT_ENCODING_DTYPE",
    )
    for name in constants:
        with monkeypatch.context() as scoped:
            scoped.setattr(
                training_dataset, name, cast("str", getattr(training_dataset, name)) + "x"
            )
            inspections.append(_inspection(root))
    with monkeypatch.context() as scoped:
        template = training_dataset.QWEN_IMAGE_TEXT_CONFIG.text_template  # pyright: ignore[reportPrivateImportUsage]
        scoped.setattr(
            training_dataset,
            "QWEN_IMAGE_TEXT_CONFIG",
            SimpleNamespace(text_template=template + "x"),
        )
        inspections.append(_inspection(root))
    inspections.extend(
        (
            _inspection(root, resolution=(16, 32)),
            _inspection(root, context_tokens=513),
            _inspection(root, vae_identity={**_VAE_IDENTITY, "digest": "blake3:" + "3" * 64}),
            _inspection(root, vae_identity={**_VAE_IDENTITY, "size": 102}),
            _inspection(root, text_identity={**_TEXT_IDENTITY, "digest": "blake3:" + "4" * 64}),
            _inspection(root, text_identity={**_TEXT_IDENTITY, "size": 203}),
        )
    )
    (root / "item.txt").write_text("different caption", encoding="utf-8")
    inspections.append(_inspection(root))
    Image.new("RGB", (24, 19)).save(root / "item.png")
    inspections.append(_inspection(root))

    assert len({value.digest for value in inspections}) == len(inspections)
    keys = {
        qwen_image_encoded_cache_key(
            QwenImageDatasetSettings(
                str(root.resolve()),
                (16, 16),
                512,
                inspection,
                str((tmp_path / "cache").resolve()),
            ),
            latent_shape=_LATENT_SHAPE,
            context_shape=_CONTEXT_SHAPE,
            attention_mask_shape=_MASK_SHAPE,
            device=torch.device("cpu"),
        )
        for inspection in inspections
    }
    assert len(keys) == len(inspections)
    settings = QwenImageDatasetSettings(
        str(root.resolve()),
        (16, 16),
        512,
        baseline,
        str((tmp_path / "cache").resolve()),
    )
    geometry_keys = {
        qwen_image_encoded_cache_key(
            settings,
            latent_shape=latent_shape,
            context_shape=context_shape,
            attention_mask_shape=mask_shape,
            device=torch.device("cpu"),
        )
        for latent_shape, context_shape, mask_shape in (
            (_LATENT_SHAPE, _CONTEXT_SHAPE, _MASK_SHAPE),
            ((1, 16, 1, 2, 4), _CONTEXT_SHAPE, _MASK_SHAPE),
            (_LATENT_SHAPE, (1, 513, 3584), _MASK_SHAPE),
            (_LATENT_SHAPE, _CONTEXT_SHAPE, (1, 513)),
        )
    }
    assert len(geometry_keys) == 4


def test_qwen_image_encoding_uses_inference_prompt_positions_and_right_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, context_tokens=640)
    events: list[str] = []
    seen_pixels: list[torch.Tensor] = []
    text_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    factories = _factories(events, seen_pixels, text_calls)

    def record_release(_device: torch.device) -> None:
        events.append("release")

    monkeypatch.setattr(
        training_data.QwenImageDatasetSource,
        "_release",
        staticmethod(record_release),
    )

    source = _source(settings, *factories)
    batch = source.batch(0, generator=torch.Generator().manual_seed(1), device=torch.device("cpu"))

    assert events == ["vae", "release", "text", "release"]
    assert len(seen_pixels) == 1
    assert seen_pixels[0].shape == (1, 3, 1, 16, 16)
    assert seen_pixels[0].dtype == torch.float32
    assert float(seen_pixels[0].min()) >= -1.0
    assert float(seen_pixels[0].max()) <= 1.0
    assert len(text_calls) == 1 and text_calls[0][1] is None
    ids = text_calls[0][0]
    caption = (Path(settings.root) / "item.txt").read_text(encoding="utf-8").strip()
    expected_row = load_qwen_bpe().encode(format_qwen_image_prompt(caption).text)
    assert ids.tolist() == [expected_row]
    selection = select_qwen_image_output((expected_row,), None)
    selected_tokens = len(selection.token_rows[0])
    expected_positions = torch.arange(len(expected_row), dtype=torch.bfloat16).float()
    expected_positions = expected_positions[selection.slice_start :]
    assert torch.equal(batch.context[0, :selected_tokens, 0], expected_positions)
    assert torch.equal(
        batch.context[0, selected_tokens:], torch.zeros_like(batch.context[0, selected_tokens:])
    )
    assert torch.equal(batch.attention_mask[0, :selected_tokens], torch.ones(selected_tokens))
    assert torch.equal(
        batch.attention_mask[0, selected_tokens:], torch.zeros(640 - selected_tokens)
    )
    assert batch.latents.dtype == batch.context.dtype == batch.attention_mask.dtype == torch.float32


def test_qwen_image_overlength_rejection_constructs_zero_encoders(tmp_path: Path) -> None:
    settings = _settings(tmp_path, caption="token " * 3000)
    calls: list[str] = []

    def vae_factory() -> WanVAE:
        calls.append("vae")
        raise AssertionError("VAE factory must not run")

    def text_factory() -> QwenImageTextModel:
        calls.append("text")
        raise AssertionError("text factory must not run")

    with pytest.raises(ValueError, match="exceeds configured 512-token context"):
        _source(settings, vae_factory, text_factory)
    assert calls == []


def test_qwen_image_position_limit_rejection_constructs_zero_encoders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    config = QWEN_IMAGE_TEXT_CONFIG
    row = (
        config.im_start_token_id,
        *(0 for _ in range(config.max_position_embeddings)),
        config.im_start_token_id,
        config.user_token_id,
        config.newline_token_id,
        42,
    )
    selected = select_qwen_image_output((row,), None)
    assert len(row) > config.max_position_embeddings
    assert len(selected.token_rows[0]) <= settings.context_tokens

    def encode(_text: str) -> tuple[int, ...]:
        return row

    monkeypatch.setattr(training_data, "load_qwen_bpe", lambda: SimpleNamespace(encode=encode))
    calls: list[str] = []

    def vae_factory() -> WanVAE:
        calls.append("vae")
        raise AssertionError("VAE factory must not run")

    def text_factory() -> QwenImageTextModel:
        calls.append("text")
        raise AssertionError("text factory must not run")

    with pytest.raises(ValueError, match="exceeds 128000-token position limit"):
        _source(settings, vae_factory, text_factory)
    assert calls == []


def test_qwen_image_cache_hit_constructs_zero_encoders_and_round_trips_exactly(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, cache=True)
    events: list[str] = []
    seen_pixels: list[torch.Tensor] = []
    text_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    first = _source(settings, *_factories(events, seen_pixels, text_calls))
    expected = first.batch(
        0, generator=torch.Generator().manual_seed(3), device=torch.device("cpu")
    )

    def unexpected_vae() -> WanVAE:
        raise AssertionError("cache hit constructed the VAE")

    def unexpected_text() -> QwenImageTextModel:
        raise AssertionError("cache hit constructed the text encoder")

    second = _source(settings, unexpected_vae, unexpected_text)
    actual = second.batch(0, generator=torch.Generator().manual_seed(9), device=torch.device("cpu"))
    assert torch.equal(actual.latents, expected.latents)
    assert torch.equal(actual.context, expected.context)
    assert torch.equal(actual.attention_mask, expected.attention_mask)


def test_qwen_image_cache_write_validates_exact_readback_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, cache=True)
    cache = _cache(settings)
    original = encoded_cache.load_tensors  # pyright: ignore[reportPrivateImportUsage]

    def corrupt_readback(path: Path, keys: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        values = original(path, keys)
        if keys is None and "context" in values:
            values["context"] = values["context"].clone()
            values["context"][0, 0, 0] += 1.0
        return values

    monkeypatch.setattr(encoded_cache, "load_tensors", corrupt_readback)
    with pytest.raises(ValueError, match="exact readback"):
        cache.write(_cache_values())
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_qwen_image_cache_publishes_shard_before_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, cache=True)
    cache = _cache(settings)
    original = encoded_cache.atomic_replace  # pyright: ignore[reportPrivateImportUsage]
    publications: list[str] = []

    def recording_replace(path: Path, data: bytes) -> None:
        publications.append(path.suffix)
        original(path, data)

    monkeypatch.setattr(encoded_cache, "atomic_replace", recording_replace)
    cache.write(_cache_values())
    assert publications == [".safetensors", ".json"]


def _replace_cache_shard(
    cache: QwenImageEncodedDatasetCache,
    manifest_path: Path,
    values: dict[str, torch.Tensor],
    *,
    metadata_change: tuple[str, str] | None = None,
) -> None:
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="ascii")))
    cache_key = cache._key  # pyright: ignore[reportPrivateUsage]
    settings = cache._settings  # pyright: ignore[reportPrivateUsage]
    metadata = {
        "cacheKey": cache_key,
        "datasetDigest": settings.digest,
        "family": "qwen-image",
        "format": "dinkster.encoded-qwen-image-dataset.v1",
        "itemCount": "1",
    }
    if metadata_change is not None:
        metadata[metadata_change[0]] = metadata_change[1]
    raw = encoded_cache._safetensors_bytes(values, metadata)  # pyright: ignore[reportPrivateUsage]
    digest = encoded_cache.blake3_digest(raw)  # pyright: ignore[reportPrivateImportUsage]
    path = manifest_path.parent.parent / "shards" / f"{digest[7:]}.safetensors"
    path.write_bytes(raw)
    manifest["shardDigest"] = digest
    manifest["shardSize"] = len(raw)
    manifest_path.write_bytes(
        encoded_cache.canonical_json(manifest)  # pyright: ignore[reportPrivateImportUsage]
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "malformed-manifest",
        "manifest-layout",
        "schema-bool",
        "manifest-key",
        "manifest-dataset",
        "size",
        "digest",
        "shard-bytes",
        "metadata",
        "dtype",
        "geometry",
        "layout",
        "mask-values",
    ),
)
def test_qwen_image_cache_rejects_malformed_and_corrupt_matrix(
    tmp_path: Path, corruption: str
) -> None:
    settings = _settings(tmp_path, cache=True)
    cache = _cache(settings)
    values = _cache_values()
    cache.write(values)
    manifest_path = next((tmp_path / "cache").rglob("*.json"))
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="ascii")))
    shard_path = (
        manifest_path.parent.parent
        / "shards"
        / f"{cast('str', manifest['shardDigest'])[7:]}.safetensors"
    )
    if corruption == "malformed-manifest":
        manifest_path.write_bytes(b"{")
    elif corruption == "manifest-layout":
        manifest["extra"] = True
        manifest_path.write_bytes(
            encoded_cache.canonical_json(manifest)  # pyright: ignore[reportPrivateImportUsage]
        )
    elif corruption == "schema-bool":
        manifest["schemaVersion"] = True
        manifest_path.write_bytes(
            encoded_cache.canonical_json(manifest)  # pyright: ignore[reportPrivateImportUsage]
        )
    elif corruption == "manifest-key":
        manifest["cacheKey"] = "blake3:" + "0" * 64
        manifest_path.write_bytes(
            encoded_cache.canonical_json(manifest)  # pyright: ignore[reportPrivateImportUsage]
        )
    elif corruption == "manifest-dataset":
        manifest["datasetDigest"] = "blake3:" + "0" * 64
        manifest_path.write_bytes(
            encoded_cache.canonical_json(manifest)  # pyright: ignore[reportPrivateImportUsage]
        )
    elif corruption == "size":
        manifest["shardSize"] = cast("int", manifest["shardSize"]) + 1
        manifest_path.write_bytes(
            encoded_cache.canonical_json(manifest)  # pyright: ignore[reportPrivateImportUsage]
        )
    elif corruption == "digest":
        manifest["shardDigest"] = "blake3:" + "0" * 64
        manifest_path.write_bytes(
            encoded_cache.canonical_json(manifest)  # pyright: ignore[reportPrivateImportUsage]
        )
    elif corruption == "shard-bytes":
        shard_path.write_bytes(shard_path.read_bytes() + b"x")
    else:
        tensors = load_tensors(shard_path)
        if corruption == "metadata":
            _replace_cache_shard(cache, manifest_path, tensors, metadata_change=("family", "flux2"))
        elif corruption == "dtype":
            tensors["context"] = tensors["context"].to(torch.int64)
            _replace_cache_shard(cache, manifest_path, tensors)
        elif corruption == "geometry":
            tensors["context"] = tensors["context"][:, :-1]
            _replace_cache_shard(cache, manifest_path, tensors)
        elif corruption == "layout":
            del tensors["context"]
            _replace_cache_shard(cache, manifest_path, tensors)
        elif corruption == "mask-values":
            tensors["attention_mask"].fill_(0.5)
            _replace_cache_shard(cache, manifest_path, tensors)
        else:
            raise AssertionError(corruption)

    assert not cache.is_valid()
    assert cache.load() is None


@pytest.mark.parametrize(
    "mutate",
    (
        _wrong_latent_dtype,
        _wrong_context_geometry,
        _nonbinary_mask,
    ),
)
def test_qwen_image_cache_rejects_malformed_writes(
    tmp_path: Path,
    mutate: Callable[[EncodedQwenImageDatasetTensors], EncodedQwenImageDatasetTensors],
) -> None:
    cache = _cache(_settings(tmp_path, cache=True))
    with pytest.raises(ValueError, match="wrong shape or dtype|must be binary"):
        cache.write(mutate(_cache_values()))
    assert not list((tmp_path / "cache").rglob("*.json"))
