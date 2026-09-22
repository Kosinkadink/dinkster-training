"""Executable proofs for Flux2 dataset identity and encoded caching."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from dinkster_inference import (
    TEKKEN_BOS,
    Flux2DevPromptTokens,
    Flux2KleinPromptTokens,
    tokenize_flux2_dev_prompt,
    tokenize_flux2_klein_prompt,
)
from dinkster_inference_torch import AutoencoderKL, QwenTextModel
from dinkster_training_torch import Flux2TrainingConfig, TrainingConfigError
from dinkster_training_torch.data import Flux2DatasetSource
from dinkster_training_torch.dataset import Flux2DatasetSettings, inspect_flux2_dataset
from dinkster_training_torch.encoded_cache import (
    EncodedFlux2DatasetTensors,
    Flux2EncodedDatasetCache,
)
from PIL import Image
from test_flux2_lora import config_mapping

_VARIANTS = ("flux2-dev", "flux2-klein-9b", "flux2-klein-4b")


def _settings(
    tmp_path: Path,
    *,
    variant: str = "flux2-dev",
    caption: str = "caption",
    context_tokens: int = 512,
) -> Flux2DatasetSettings:
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    Image.new("RGB", (16, 16), (20, 40, 60)).save(root / "item.png")
    (root / "item.txt").write_text(caption, encoding="utf-8")
    identities = ({"digest": "vae", "size": 3}, {"digest": "text", "size": 4})
    inspection = inspect_flux2_dataset(
        root,
        (16, 16),
        context_tokens,
        variant=cast("object", variant),  # pyright: ignore[reportArgumentType]
        vae_identity=identities[0],
        text_encoder_identity=identities[1],
    )
    assert not inspection.errors
    return Flux2DatasetSettings(
        str(root),
        cast("object", variant),  # pyright: ignore[reportArgumentType]
        (16, 16),
        context_tokens,
        inspection,
        str(tmp_path / "cache"),
    )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_flux2_encoded_cache_round_trips_bitwise(tmp_path: Path, variant: str) -> None:
    settings = _settings(tmp_path, variant=variant)
    cache = Flux2EncodedDatasetCache(
        settings,
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, 512, 4),
        device=torch.device("cpu"),
    )
    tensors = EncodedFlux2DatasetTensors(
        torch.randn(1, 128, 1, 1),
        torch.randn(1, 512, 4),
    )

    cache.write(tensors)
    loaded = cache.load()

    assert loaded is not None
    assert torch.equal(loaded.latents, tensors.latents)
    assert torch.equal(loaded.context, tensors.context)


@pytest.mark.parametrize(
    "tensors",
    (
        EncodedFlux2DatasetTensors(torch.zeros(1, 127, 1, 1), torch.zeros(1, 512, 4)),
        EncodedFlux2DatasetTensors(torch.zeros(1, 128, 1, 1), torch.zeros(1, 511, 4)),
        EncodedFlux2DatasetTensors(
            torch.zeros(1, 128, 1, 1, dtype=torch.float16), torch.zeros(1, 512, 4)
        ),
    ),
)
def test_flux2_encoded_cache_rejects_malformed_writes(
    tmp_path: Path, tensors: EncodedFlux2DatasetTensors
) -> None:
    settings = _settings(tmp_path)
    cache = Flux2EncodedDatasetCache(
        settings,
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, 512, 4),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="wrong shape or dtype"):
        cache.write(tensors)
    assert not list((Path(cast("str", settings.encoded_cache_root)) / "flux2-v1").rglob("*"))


def test_flux2_encoded_cache_rejects_corrupt_manifest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cache = Flux2EncodedDatasetCache(
        settings,
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, 512, 4),
        device=torch.device("cpu"),
    )
    cache.write(EncodedFlux2DatasetTensors(torch.zeros(1, 128, 1, 1), torch.zeros(1, 512, 4)))
    manifest = next((Path(cast("str", settings.encoded_cache_root)) / "flux2-v1").rglob("*.json"))
    manifest.write_text("{}", encoding="ascii")

    assert cache.load() is None


def test_flux2_encoded_cache_rejects_boolean_manifest_version(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cache = Flux2EncodedDatasetCache(
        settings,
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, 512, 4),
        device=torch.device("cpu"),
    )
    cache.write(EncodedFlux2DatasetTensors(torch.zeros(1, 128, 1, 1), torch.zeros(1, 512, 4)))
    manifest = next((Path(cast("str", settings.encoded_cache_root)) / "flux2-v1").rglob("*.json"))
    value = json.loads(manifest.read_text(encoding="ascii"))
    value["schemaVersion"] = True
    manifest.write_text(json.dumps(value), encoding="ascii")

    assert cache.load() is None


def test_flux2_native_token_layouts() -> None:
    class FakeBpe:
        def encode(self, _value: str) -> list[int]:
            return [41, 42]

    dev = tokenize_flux2_dev_prompt(
        "caption",
        tokenizer=cast("object", FakeBpe()),  # pyright: ignore[reportArgumentType]
    )
    assert dev.ids[0] == TEKKEN_BOS
    assert dev.ids[1:] == (41, 42)
    assert dev.attention_mask == (1, 1, 1)

    klein = tokenize_flux2_klein_prompt(
        "caption",
        tokenizer=cast("object", FakeBpe()),  # pyright: ignore[reportArgumentType]
    )
    first_pad = klein.ids.index(151643)
    assert klein.ids[:first_pad] == (41, 42)
    assert klein.ids[first_pad:] == (151643,) * (512 - first_pad)
    assert klein.attention_mask == (1,) * first_pad + (0,) * (512 - first_pad)


@pytest.mark.parametrize(
    ("variant", "context_tokens"),
    (("flux2-dev", 512), ("flux2-klein-9b", 513), ("flux2-klein-4b", 513)),
)
def test_flux2_dataset_releases_vae_then_text_and_hits_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    context_tokens: int,
) -> None:
    settings = _settings(tmp_path, variant=variant, context_tokens=context_tokens)
    events: list[str] = []
    encoded_rows: list[tuple[int, ...]] = []

    class FakeVAE(torch.nn.Module):
        def encode(self, value: torch.Tensor) -> torch.Tensor:
            del value
            return torch.arange(128, dtype=torch.float32).reshape(1, 128, 1, 1)

    class FakeEncoder:
        def __init__(self, model: torch.nn.Module, *_args: object) -> None:
            del model

        def encode(self, _caption: str) -> SimpleNamespace:
            tokens = (
                Flux2DevPromptTokens((TEKKEN_BOS, 41, 42), (1, 1, 1))
                if variant == "flux2-dev"
                else Flux2KleinPromptTokens(
                    (41, 42) + (151643,) * 510,
                    (1, 1) + (0,) * 510,
                )
            )
            encoded_rows.append(tokens.ids)
            return SimpleNamespace(embeddings=torch.ones(1, len(tokens.ids), 4))

    def dev_tokens(_caption: str, **_kwargs: object) -> Flux2DevPromptTokens:
        return Flux2DevPromptTokens((TEKKEN_BOS, 41, 42), (1, 1, 1))

    def klein_tokens(_caption: str) -> Flux2KleinPromptTokens:
        return Flux2KleinPromptTokens(
            (41, 42) + (151643,) * 510,
            (1, 1) + (0,) * 510,
        )

    monkeypatch.setattr("dinkster_training_torch.data.tokenize_flux2_dev_prompt", dev_tokens)
    monkeypatch.setattr("dinkster_training_torch.data.tokenize_flux2_klein_prompt", klein_tokens)
    monkeypatch.setattr("dinkster_training_torch.data.Flux2DevTextEncoder", FakeEncoder)
    monkeypatch.setattr("dinkster_training_torch.data.Flux2KleinTextEncoder", FakeEncoder)
    monkeypatch.setattr("dinkster_training_torch.data.load_flux2_tekken_bpe", object)

    def release(_device: torch.device) -> None:
        events.append("release")

    monkeypatch.setattr(Flux2DatasetSource, "_release", staticmethod(release))

    def factory(name: str, value: torch.nn.Module) -> Callable[[], torch.nn.Module]:
        def build() -> torch.nn.Module:
            events.append(name)
            return value

        return build

    vae_factory = cast("Callable[[], AutoencoderKL]", factory("vae", FakeVAE()))
    text_factory = cast("Callable[[], QwenTextModel]", factory("text", torch.nn.Linear(1, 1)))
    identities = ({"digest": "vae", "size": 3}, {"digest": "text", "size": 4})
    first = Flux2DatasetSource(
        settings,
        vae_factory,
        text_factory,
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, context_tokens, 4),
        artifact_identities=identities,
        device=torch.device("cpu"),
    )
    assert events == ["vae", "release", "text", "release"]
    assert len(encoded_rows) == 1
    first_batch = first.batch(0, generator=torch.Generator(), device=torch.device("cpu"))
    if variant == "flux2-dev":
        assert encoded_rows == [(TEKKEN_BOS, 41, 42)]
        assert torch.count_nonzero(first_batch.context[:, :-3]) == 0
        assert torch.equal(first_batch.context[:, -3:], torch.ones(1, 3, 4))
    else:
        assert encoded_rows == [(41, 42) + (151643,) * 510]
        assert torch.equal(first_batch.context[:, :512], torch.ones(1, 512, 4))
        assert torch.count_nonzero(first_batch.context[:, 512:]) == 0
    events.clear()
    second = Flux2DatasetSource(
        settings,
        vae_factory,
        text_factory,
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, context_tokens, 4),
        artifact_identities=identities,
        device=torch.device("cpu"),
    )
    second_batch = second.batch(0, generator=torch.Generator(), device=torch.device("cpu"))
    assert events == []
    assert torch.equal(first_batch.latents, second_batch.latents)
    assert torch.equal(first_batch.context, second_batch.context)


def test_flux2_overlong_caption_stops_before_text_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, caption="overlong")
    events: list[str] = []

    class FakeVAE(torch.nn.Module):
        def encode(self, _value: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 128, 1, 1)

    monkeypatch.setattr("dinkster_training_torch.data.load_flux2_tekken_bpe", object)

    def overlong_tokens(_caption: str, **_kwargs: object) -> Flux2DevPromptTokens:
        return Flux2DevPromptTokens((TEKKEN_BOS,) + tuple(range(512)), (1,) * 513)

    monkeypatch.setattr("dinkster_training_torch.data.tokenize_flux2_dev_prompt", overlong_tokens)

    def text_factory() -> QwenTextModel:
        events.append("text")
        raise AssertionError("over-length caption reached the text encoder")

    with pytest.raises(ValueError, match="exceeds configured 512-token context"):
        Flux2DatasetSource(
            settings,
            cast("Callable[[], AutoencoderKL]", lambda: FakeVAE()),
            text_factory,
            latent_shape=(1, 128, 1, 1),
            context_shape=(1, 512, 4),
            artifact_identities=(
                {"digest": "vae", "size": 3},
                {"digest": "text", "size": 4},
            ),
            device=torch.device("cpu"),
        )
    assert events == []


@pytest.mark.parametrize("variant", _VARIANTS)
def test_flux2_config_dataset_round_trip_and_identity(tmp_path: Path, variant: str) -> None:
    mapping = config_mapping(tmp_path / "artifacts", variant)
    settings = _settings(tmp_path / "input", variant=variant)
    mapping.pop("datasetIdentity")
    mapping["latentShape"] = [1, 128, 1, 1]
    mapping["dataset"] = {
        "type": "flux2-image-caption-folder",
        "variant": variant,
        "root": settings.root,
        "resolution": [16, 16],
        "contextTokens": 512,
        "encodedCacheRoot": settings.encoded_cache_root,
    }
    config = Flux2TrainingConfig.from_mapping(mapping)

    assert Flux2TrainingConfig.from_mapping(config.to_mapping()) == config
    assert config.dataset is not None
    assert config.dataset_identity == config.dataset.digest
    assert config.identity_mapping()["dataset"] == {
        key: value
        for key, value in config.dataset.to_mapping().items()
        if key != "encodedCacheRoot"
    }

    changed = config.to_mapping()
    dataset = cast("dict[str, object]", changed["dataset"])
    dataset["contextTokens"] = 513
    changed["contextShape"] = [1, 513, config.context_shape[2]]
    with pytest.raises(TrainingConfigError, match="dataset digest changed"):
        Flux2TrainingConfig.from_mapping(changed)

    malformed_wire = config.to_mapping()
    wire_dataset = cast("dict[str, object]", malformed_wire["dataset"])
    wire_dataset["resolution"] = [16.0, 16]
    with pytest.raises(TrainingConfigError, match="dataset.resolution.*integer"):
        Flux2TrainingConfig.from_mapping(malformed_wire)

    with pytest.raises(TrainingConfigError, match="normalized Flux2 dataset settings"):
        replace(config, dataset=replace(config.dataset, context_tokens=cast("int", True)))


def test_flux2_dataset_identity_covers_variant_geometry_and_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (16, 16)).save(root / "item.png")
    (root / "item.txt").write_text("caption", encoding="utf-8")

    def identity(
        variant: str = "flux2-dev",
        resolution: tuple[int, int] = (16, 16),
        tokens: int = 512,
        vae: object = "vae-a",
        text: object = "text-a",
    ) -> str:
        return inspect_flux2_dataset(
            root,
            resolution,
            tokens,
            variant=cast("object", variant),  # pyright: ignore[reportArgumentType]
            vae_identity=vae,
            text_encoder_identity=text,
        ).digest

    baseline = identity()
    assert (
        len(
            {
                baseline,
                identity(variant="flux2-klein-9b"),
                identity(resolution=(32, 16)),
                identity(tokens=513),
                identity(vae="vae-b"),
                identity(text="text-b"),
            }
        )
        == 6
    )
    assert baseline.startswith("blake3:")
