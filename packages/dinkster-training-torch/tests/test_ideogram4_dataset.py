"""Executable proofs for role-bound Ideogram 4 dataset caching."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from dinkster_training_torch.data import Ideogram4DatasetSource
from dinkster_training_torch.dataset import Ideogram4DatasetSettings, inspect_ideogram4_dataset
from dinkster_training_torch.encoded_cache import (
    EncodedIdeogram4DatasetTensors,
    Ideogram4EncodedDatasetCache,
)
from PIL import Image


def _settings(tmp_path: Path, *, role: str, caption: bool = True) -> Ideogram4DatasetSettings:
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    Image.new("RGB", (16, 16), (20, 40, 60)).save(root / "item.png")
    if caption:
        (root / "item.txt").write_text("caption", encoding="utf-8")
    conditional = role == "conditional"
    inspection = inspect_ideogram4_dataset(
        root,
        (16, 16),
        4 if conditional else 0,
        role=cast("object", role),  # pyright: ignore[reportArgumentType]
        vae_identity={"digest": "vae", "size": 1, "identity": "vae-native"},
        text_encoder_identity=(
            {"digest": "text", "size": 1, "identity": "text-native"} if conditional else None
        ),
    )
    assert inspection.errors == ()
    return Ideogram4DatasetSettings(
        str(root.resolve()),
        cast("object", role),  # pyright: ignore[reportArgumentType]
        (16, 16),
        4 if conditional else 0,
        inspection,
        str((tmp_path / "cache").resolve()),
    )


@pytest.mark.parametrize(
    ("role", "caption", "expected_type"),
    (
        ("conditional", True, "ideogram4-image-caption-folder"),
        ("unconditional", False, "ideogram4-image-folder"),
    ),
)
def test_ideogram4_dataset_identity_is_role_bound(
    tmp_path: Path, role: str, caption: bool, expected_type: str
) -> None:
    settings = _settings(tmp_path, role=role, caption=caption)

    assert settings.to_mapping()["type"] == expected_type
    assert settings.to_mapping().get("contextTokens") == (4 if role == "conditional" else None)
    opposite = inspect_ideogram4_dataset(
        Path(settings.root),
        settings.resolution,
        0 if role == "conditional" else 4,
        role="unconditional" if role == "conditional" else "conditional",
        vae_identity={"digest": "vae", "size": 1, "identity": "vae-native"},
        text_encoder_identity=(
            None
            if role == "conditional"
            else {"digest": "text", "size": 1, "identity": "text-native"}
        ),
    )
    assert opposite.digest != settings.digest


def test_conditional_dataset_builds_vae_then_text_and_hits_verified_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, role="conditional")
    events: list[str] = []

    class VAE(torch.nn.Module):
        def encode(self, _value: torch.Tensor) -> torch.Tensor:
            events.append("vae-encode")
            return torch.arange(128, dtype=torch.float32).reshape(1, 128, 1, 1)

    class TextEncoder:
        def encode(self, _caption: str) -> SimpleNamespace:
            events.append("text-encode")
            return SimpleNamespace(
                embeddings=torch.ones(1, 2, 53248),
                attention_mask=torch.tensor([[1, 0]]),
            )

    def prompt_tokens(_caption: str) -> SimpleNamespace:
        return SimpleNamespace(ids=(1, 2), attention_mask=(1, 1))

    monkeypatch.setattr("dinkster_training_torch.data.tokenize_ideogram4_prompt", prompt_tokens)

    def vae_factory() -> VAE:
        events.append("vae-construct")
        return VAE()

    def text_factory() -> object:
        events.append("text-construct")
        return TextEncoder()

    source = Ideogram4DatasetSource(
        settings,
        cast("object", vae_factory),  # pyright: ignore[reportArgumentType]
        cast("object", text_factory),  # pyright: ignore[reportArgumentType]
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, 4, 53248),
        attention_mask_shape=(1, 4),
        vae_identity={"digest": "vae", "size": 1, "identity": "vae-native"},
        text_encoder_identity={"digest": "text", "size": 1, "identity": "text-native"},
        device=torch.device("cpu"),
    )
    batch = source.batch(0, generator=torch.Generator(), device=torch.device("cpu"))

    assert events == ["vae-construct", "vae-encode", "text-construct", "text-encode"]
    assert batch.latents.shape == (1, 128, 1, 1)
    assert batch.context is not None and batch.context.shape == (1, 4, 53248)
    assert batch.attention_mask is not None
    assert torch.equal(batch.attention_mask, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    def fail() -> object:
        raise AssertionError("cache hit must construct neither encoder")

    cached = Ideogram4DatasetSource(
        settings,
        cast("object", fail),  # pyright: ignore[reportArgumentType]
        cast("object", fail),  # pyright: ignore[reportArgumentType]
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, 4, 53248),
        attention_mask_shape=(1, 4),
        vae_identity={"digest": "vae", "size": 1, "identity": "vae-native"},
        text_encoder_identity={"digest": "text", "size": 1, "identity": "text-native"},
        device=torch.device("cpu"),
    )
    cached_batch = cached.batch(0, generator=torch.Generator(), device=torch.device("cpu"))
    assert torch.equal(cached_batch.latents, batch.latents)
    assert torch.equal(cast("torch.Tensor", cached_batch.context), batch.context)
    assert torch.equal(cast("torch.Tensor", cached_batch.attention_mask), batch.attention_mask)


def test_unconditional_dataset_never_constructs_text_and_cache_contains_only_latents(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path, role="unconditional", caption=False)
    calls: list[str] = []

    class VAE(torch.nn.Module):
        def encode(self, _value: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, 128, 1, 1)

    def vae_factory() -> VAE:
        calls.append("vae")
        return VAE()

    source = Ideogram4DatasetSource(
        settings,
        cast("object", vae_factory),  # pyright: ignore[reportArgumentType]
        None,
        latent_shape=(1, 128, 1, 1),
        context_shape=None,
        attention_mask_shape=None,
        vae_identity={"digest": "vae", "size": 1, "identity": "vae-native"},
        text_encoder_identity=None,
        device=torch.device("cpu"),
    )
    batch = source.batch(0, generator=torch.Generator(), device=torch.device("cpu"))

    assert calls == ["vae"]
    assert batch.context is batch.attention_mask is None
    cache = Ideogram4EncodedDatasetCache(
        settings,
        latent_shape=(1, 128, 1, 1),
        context_shape=None,
        attention_mask_shape=None,
        device=torch.device("cpu"),
    )
    loaded = cache.load()
    assert loaded is not None
    assert loaded.context is loaded.attention_mask is None


def test_ideogram4_cache_rejects_corrupt_manifest_and_nonbinary_mask(tmp_path: Path) -> None:
    settings = _settings(tmp_path, role="conditional")
    cache = Ideogram4EncodedDatasetCache(
        settings,
        latent_shape=(1, 128, 1, 1),
        context_shape=(1, 4, 53248),
        attention_mask_shape=(1, 4),
        device=torch.device("cpu"),
    )
    tensors = EncodedIdeogram4DatasetTensors(
        torch.zeros(1, 128, 1, 1),
        torch.zeros(1, 4, 53248),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
    )
    cache.write(tensors)
    assert cache.load() is not None

    invalid = EncodedIdeogram4DatasetTensors(
        tensors.latents,
        tensors.context,
        torch.tensor([[1.0, 0.5, 0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="mask must be binary"):
        cache.write(invalid)

    manifest = next(
        (Path(cast("str", settings.encoded_cache_root)) / "ideogram4-v1").rglob("*.json")
    )
    manifest.write_text(json.dumps({"schemaVersion": 1}), encoding="ascii")
    assert cache.load() is None
