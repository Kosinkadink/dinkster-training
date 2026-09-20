"""MiniMax Music 3 community codec, dataset identity, and cache proofs."""

from __future__ import annotations

import math
import wave
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from dinkster_assets import digest_bytes
from dinkster_training_torch import (
    MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION,
    MINIMAX_MUSIC3_DAV_ENCODER_CONTRACT,
    MINIMAX_MUSIC3_DAV_SOURCE_REVISION,
    MINIMAX_MUSIC3_RVQ_ENCODER_CONTRACT,
    MINIMAX_MUSIC3_RVQ_SOURCE_REVISION,
    MINIMAX_MUSIC3_TEACHER_FORCING_CONTRACT,
    MiniMaxMusic3ArtifactPin,
    MiniMaxMusic3DatasetSettings,
    inspect_minimax_music3_dataset,
)
from dinkster_training_torch.minimax_music3_training import (
    EncodedMiniMaxMusic3DatasetTensors,
    MiniMaxMusic3EncodedDatasetCache,
    MiniMaxMusic3RvqEncoder,
    minimax_music3_encoded_cache_key,
    minimax_music3_rvq_pool,
    teacher_forced_minimax_music3_context,
)
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace


def _pin(path: Path, payload: bytes, *, identity: str | None = None) -> MiniMaxMusic3ArtifactPin:
    path.write_bytes(payload)
    return MiniMaxMusic3ArtifactPin(
        str(path.resolve()), digest_bytes(payload), len(payload), identity
    )


def _write_wav(path: Path, samples: int, *, channels: int = 2, rate: int = 44_100) -> None:
    values = torch.arange(samples * channels, dtype=torch.int32)
    payload = ((values % 1024) - 512).to(torch.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(payload)


def _settings(tmp_path: Path, *, cache: bool = True) -> MiniMaxMusic3DatasetSettings:
    root = tmp_path / "dataset"
    root.mkdir()
    _write_wav(root / "song.wav", 44_100 // 25)
    (root / "song.caption.txt").write_text("bright synth pop", encoding="utf-8")
    (root / "song.lyrics.txt").write_text("hello from the chorus", encoding="utf-8")
    dav = _pin(tmp_path / "dav.pth", b"dav")
    rvq = _pin(tmp_path / "rvq.safetensors", b"rvq")
    text = _pin(
        tmp_path / "text.safetensors",
        b"text",
        identity="native:dinkster.minimax_music3:" + "1" * 64,
    )
    inspection = inspect_minimax_music3_dataset(root, 1, dav, rvq, text, "float32")
    assert not inspection.errors
    return MiniMaxMusic3DatasetSettings(
        str(root.resolve()),
        1,
        dav,
        rvq,
        text,
        "float32",
        inspection,
        str((tmp_path / "cache").resolve()) if cache else None,
    )


def test_community_training_contracts_are_revision_pinned() -> None:
    assert MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION.endswith(
        "def7bbc065e5f15d9e551827e247ba046fe36eb6"
    )
    assert MINIMAX_MUSIC3_RVQ_SOURCE_REVISION.endswith("326964c2f4edcc642c1ea116274dd2dd94081713")
    assert MINIMAX_MUSIC3_DAV_ENCODER_CONTRACT == (
        "community-minimax-music3-dav-mean-stereo-44100-v1"
    )
    assert MINIMAX_MUSIC3_DAV_SOURCE_REVISION.endswith("fce0d00b1ae42ee47874babb8c06fb859eb01443")
    assert MINIMAX_MUSIC3_RVQ_ENCODER_CONTRACT == ("community-minimax-music3-rvq-v4-169m-argmax-v1")
    assert MINIMAX_MUSIC3_TEACHER_FORCING_CONTRACT == (
        "community-minimax-music3-official-ar-teacher-forcing-v1"
    )


def test_dataset_requires_exact_audio_caption_and_lyrics_contract(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    inspection = settings.inspection

    assert len(inspection.items) == 1
    assert inspection.caption_files == inspection.nonempty_captions == 1
    assert inspection.lyrics_files == inspection.nonempty_lyrics == 1

    item = inspection.items[0]
    item.lyrics_path.write_text("", encoding="utf-8")
    changed = inspect_minimax_music3_dataset(
        Path(settings.root),
        settings.audio_frames,
        settings.dav_encoder_state,
        settings.rvq_encoder_state,
        settings.text_encoder_state,
        settings.text_compute_dtype,
    )
    assert changed.digest != inspection.digest
    assert changed.nonempty_lyrics == 0
    assert any("lyrics is empty" in error for error in changed.errors)


@pytest.mark.parametrize(
    ("channels", "rate", "message"),
    ((3, 44_100, "one or two channels"), (2, 48_000, "sample rate must be 44100 Hz")),
)
def test_dataset_rejects_wrong_pcm_geometry(
    tmp_path: Path, channels: int, rate: int, message: str
) -> None:
    settings = _settings(tmp_path)
    _write_wav(Path(settings.root) / "song.wav", 44_100 // 25, channels=channels, rate=rate)

    inspection = inspect_minimax_music3_dataset(
        Path(settings.root),
        1,
        settings.dav_encoder_state,
        settings.rvq_encoder_state,
        settings.text_encoder_state,
        settings.text_compute_dtype,
    )

    assert any(message in error for error in inspection.errors)


def test_dataset_digest_covers_artifact_and_source_bytes(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    original = settings.digest
    caption = Path(settings.root) / "song.caption.txt"
    caption.write_text("dark synth pop", encoding="utf-8")
    changed_caption = inspect_minimax_music3_dataset(
        Path(settings.root),
        1,
        settings.dav_encoder_state,
        settings.rvq_encoder_state,
        settings.text_encoder_state,
        settings.text_compute_dtype,
    )
    assert changed_caption.digest != original

    alternate_rvq = MiniMaxMusic3ArtifactPin(
        settings.rvq_encoder_state.path,
        "blake3:" + "f" * 64,
        settings.rvq_encoder_state.size,
    )
    changed_artifact = inspect_minimax_music3_dataset(
        Path(settings.root),
        1,
        settings.dav_encoder_state,
        alternate_rvq,
        settings.text_encoder_state,
        settings.text_compute_dtype,
    )
    assert changed_artifact.digest != changed_caption.digest


@pytest.mark.parametrize("audio_frames", (1, 25, 128))
def test_rvq_pool_uses_exact_real_audio_boundaries(audio_frames: int) -> None:
    dav_frames = math.ceil(audio_frames * (44_100 // 25) / 512)
    pool = minimax_music3_rvq_pool(audio_frames, dav_frames).squeeze(0)

    assert tuple(pool.shape) == (audio_frames, dav_frames)
    assert torch.equal(pool.sum(dim=1), torch.ones(audio_frames))
    for frame, row in enumerate(pool):
        start = frame * 441 // 128
        stop = (frame + 1) * 441 // 128
        expected = torch.zeros(dav_frames)
        expected[start:stop] = 1.0 / (stop - start)
        assert torch.equal(row, expected)


def test_pinned_rvq_architecture_has_exact_stable_state_layout() -> None:
    with torch.device("meta"):
        model = MiniMaxMusic3RvqEncoder()
    keys = tuple(model.state_dict())

    assert len(keys) == 210
    assert keys[0] == "position"
    assert "transformer.7.linear2.bias" in keys
    assert "depth_decoder.heads.6.bias" in keys


def test_encoded_cache_is_content_addressed_and_rejects_corruption(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    cache = MiniMaxMusic3EncodedDatasetCache(settings, device=torch.device("cpu"))
    values = EncodedMiniMaxMusic3DatasetTensors(
        torch.arange(128 * settings.latent_frames, dtype=torch.float32).reshape(
            1, 128, settings.latent_frames
        ),
        torch.arange(settings.audio_frames * 8 * 4096, dtype=torch.float32).reshape(
            1, settings.audio_frames, 8 * 4096
        ),
    )

    digest = cache.write(values)
    loaded = cache.load()
    assert digest.startswith("blake3:")
    assert loaded is not None
    assert torch.equal(loaded.latents, values.latents)
    assert torch.equal(loaded.context, values.context)
    assert cache.is_valid()

    invalid = replace(values, context=values.context.clone().fill_(float("nan")))
    with pytest.raises(ValueError, match="invalid shape, dtype, or values"):
        cache.write(invalid)

    manifest = next(
        (
            Path(settings.encoded_cache_root or "") / "minimax-music3-community-v1" / "entries"
        ).iterdir()
    )
    manifest.write_text("{}", encoding="ascii")
    assert cache.load() is None
    assert not cache.is_valid()


def test_encoded_cache_key_includes_execution_and_dataset_identity(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = minimax_music3_encoded_cache_key(settings, device=torch.device("cpu"))
    alternate = replace(
        settings,
        inspection=replace(settings.inspection, digest="blake3:" + "a" * 64),
    )

    assert first.startswith("blake3:")
    assert minimax_music3_encoded_cache_key(alternate, device=torch.device("cpu")) != first
    changed_dtype = replace(settings, text_compute_dtype="bfloat16")
    assert minimax_music3_encoded_cache_key(changed_dtype, device=torch.device("cpu")) != first


class _GeneratedEmbedding(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return ids.to(torch.float32).unsqueeze(-1).expand(*ids.shape, 4096) + self.anchor


class _FakeDepthDecoder(torch.nn.Module):
    projection = torch.nn.Identity()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _FakeTextCore(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens_prefill = _GeneratedEmbedding()
        self.embed_tokens_audio = _GeneratedEmbedding()
        self.audio_extra_embedding = _GeneratedEmbedding()
        self.audio_decoder = _FakeDepthDecoder()

    def forward_causal(
        self,
        cache_position: object,
        cache: object = (),
        *,
        embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        assert cache_position is None
        assert cache == ()
        return embeds, None


class _FakeTextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            pruned=True,
            audio_vocab_size=1024,
            audio_num_codebooks=8,
        )
        self.model = _FakeTextCore()


def test_teacher_forcing_keeps_c0_and_seven_depth_positions() -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    codes = torch.tensor(
        [[3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18]],
        dtype=torch.int64,
    )

    context = teacher_forced_minimax_music3_context(
        cast("object", _FakeTextModel()),  # pyright: ignore[reportArgumentType]
        tokenizer,
        "bright pop",
        "hello chorus",
        codes,
        compute_dtype=torch.float32,
    )

    assert context.dtype == torch.float32
    assert tuple(context.shape) == (1, 2, 8 * 4096)
    assert torch.isfinite(context).all()
