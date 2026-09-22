"""Executable proofs for Flux dataset identity and encoded caching."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from dinkster_inference import PackedToken, WeightedSpan
from dinkster_inference_torch import AutoencoderKL, ClipTextModel, T5TextModel
from dinkster_training_torch.data import FluxDatasetSource
from dinkster_training_torch.dataset import FluxDatasetSettings, inspect_flux_dataset
from dinkster_training_torch.encoded_cache import (
    EncodedFluxDatasetTensors,
    FluxEncodedDatasetCache,
)
from PIL import Image


def _settings(
    tmp_path: Path, *, caption: str = "caption", context_tokens: int = 8
) -> FluxDatasetSettings:
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    Image.new("RGB", (16, 16), (20, 40, 60)).save(root / "item.png")
    (root / "item.txt").write_text(caption, encoding="utf-8")
    identities = ({"digest": "v"}, {"digest": "c"}, {"digest": "t"})
    inspection = inspect_flux_dataset(
        root,
        (16, 16),
        context_tokens,
        variant="flux1-dev",
        vae_identity=identities[0],
        clip_l_identity=identities[1],
        t5xxl_identity=identities[2],
    )
    assert not inspection.errors
    return FluxDatasetSettings(
        str(root),
        "flux1-dev",
        (16, 16),
        context_tokens,
        inspection,
        str(tmp_path / "cache"),
    )


def test_flux_encoded_cache_round_trips_bitwise(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cache = FluxEncodedDatasetCache(
        settings,
        latent_shape=(1, 16, 2, 2),
        context_shape=(1, 8, 4096),
        pooled_shape=(1, 768),
        device=torch.device("cpu"),
    )
    tensors = EncodedFluxDatasetTensors(
        torch.randn(1, 16, 2, 2),
        torch.randn(1, 8, 4096),
        torch.randn(1, 768),
    )

    cache.write(tensors)
    loaded = cache.load()

    assert loaded is not None
    assert torch.equal(loaded.latents, tensors.latents)
    assert torch.equal(loaded.context, tensors.context)
    assert torch.equal(loaded.pooled, tensors.pooled)


@pytest.mark.parametrize(
    "tensors",
    [
        EncodedFluxDatasetTensors(
            torch.zeros(1, 15, 2, 2), torch.zeros(1, 8, 4096), torch.zeros(1, 768)
        ),
        EncodedFluxDatasetTensors(
            torch.zeros(1, 16, 2, 2), torch.zeros(1, 7, 4096), torch.zeros(1, 768)
        ),
        EncodedFluxDatasetTensors(
            torch.zeros(1, 16, 2, 2), torch.zeros(1, 8, 4096), torch.zeros(1, 767)
        ),
    ],
)
def test_flux_encoded_cache_rejects_malformed_writes(
    tmp_path: Path, tensors: EncodedFluxDatasetTensors
) -> None:
    cache = FluxEncodedDatasetCache(
        _settings(tmp_path),
        latent_shape=(1, 16, 2, 2),
        context_shape=(1, 8, 4096),
        pooled_shape=(1, 768),
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="wrong shape or dtype"):
        cache.write(tensors)


def test_flux_dataset_builds_encoder_first_then_hits_cache_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, context_tokens=256)
    events: list[str] = []
    encoded_t5_chunks: list[tuple[tuple[int, ...], ...]] = []

    class FakeVAE(torch.nn.Module):
        def encode(self, value: torch.Tensor) -> torch.Tensor:
            return value.mean(dim=1, keepdim=True).expand(-1, 16, -1, -1)

    class FakeTokenizer:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def tokenize(self, caption: str) -> tuple[WeightedSpan, ...]:
            token_count = 256 if caption == "overlong" else 2
            return (WeightedSpan(tuple(range(10, 10 + token_count)), 1.0),)

    class FakeClipEncoder:
        def __init__(self, model: torch.nn.Module) -> None:
            del model

        def encode(self, spans: object) -> SimpleNamespace:
            del spans
            return SimpleNamespace(pooled=torch.arange(768, dtype=torch.float32).unsqueeze(0))

    class FakeT5Encoder:
        def __init__(self, model: torch.nn.Module) -> None:
            del model

        def encode_chunks(self, chunks: object) -> SimpleNamespace:
            packed = cast("tuple[tuple[PackedToken, ...], ...]", chunks)
            units = tuple(tuple(cast("int", token.unit) for token in chunk) for chunk in packed)
            encoded_t5_chunks.append(units)
            return SimpleNamespace(
                embeddings=torch.ones(1, sum(len(chunk) for chunk in packed), 4096)
            )

    monkeypatch.setattr("dinkster_training_torch.data.PromptTokenizer", FakeTokenizer)
    monkeypatch.setattr("dinkster_training_torch.data.ClipTextEncoder", FakeClipEncoder)
    monkeypatch.setattr("dinkster_training_torch.data.T5TextEncoder", FakeT5Encoder)
    monkeypatch.setattr(
        "dinkster_training_torch.data.load_clip_bpe", lambda: SimpleNamespace(encode=None)
    )
    monkeypatch.setattr(
        "dinkster_training_torch.data.load_t5_spm", lambda: SimpleNamespace(encode=None)
    )

    def factory(name: str, value: torch.nn.Module):
        def build() -> torch.nn.Module:
            events.append(name)
            return value

        return build

    identities = ({"digest": "v"}, {"digest": "c"}, {"digest": "t"})
    vae_factory = cast("Callable[[], AutoencoderKL]", factory("vae", FakeVAE()))
    clip_factory = cast("Callable[[], ClipTextModel]", factory("clip", torch.nn.Linear(1, 1)))
    t5_factory = cast("Callable[[], T5TextModel]", factory("t5", torch.nn.Linear(1, 1)))
    first = FluxDatasetSource(
        settings,
        vae_factory,
        clip_factory,
        t5_factory,
        latent_shape=(1, 16, 16, 16),
        context_shape=(1, 256, 4096),
        pooled_shape=(1, 768),
        artifact_identities=identities,
        device=torch.device("cpu"),
    )
    assert events == ["vae", "clip", "t5"]
    assert len(encoded_t5_chunks) == 1
    assert encoded_t5_chunks[0][0][:3] == (10, 11, 1)
    assert encoded_t5_chunks[0][0][3:] == (0,) * 253
    first_batch = first.batch(0, generator=torch.Generator(), device=torch.device("cpu"))
    events.clear()
    second = FluxDatasetSource(
        settings,
        vae_factory,
        clip_factory,
        t5_factory,
        latent_shape=(1, 16, 16, 16),
        context_shape=(1, 256, 4096),
        pooled_shape=(1, 768),
        artifact_identities=identities,
        device=torch.device("cpu"),
    )
    second_batch = second.batch(0, generator=torch.Generator(), device=torch.device("cpu"))
    assert events == []
    assert torch.equal(first_batch.latents, second_batch.latents)
    assert torch.equal(first_batch.context, second_batch.context)
    assert torch.equal(first_batch.pooled, second_batch.pooled)

    overlong = _settings(tmp_path / "overlong", caption="overlong", context_tokens=256)
    with pytest.raises(ValueError, match="exceeds configured 256-token context"):
        FluxDatasetSource(
            overlong,
            vae_factory,
            clip_factory,
            t5_factory,
            latent_shape=(1, 16, 16, 16),
            context_shape=(1, 256, 4096),
            pooled_shape=(1, 768),
            artifact_identities=identities,
            device=torch.device("cpu"),
        )
    assert len(encoded_t5_chunks) == 1
