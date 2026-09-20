"""Community-derived MiniMax Music 3 training dataset and codec seams.

The waveform and RVQ encoders follow the Apache-2.0 engineering adapters at
SimpleTuner commit def7bbc065e5f15d9e551827e247ba046fe36eb6 and RVQ artifact
revision 326964c2f4edcc642c1ea116274dd2dd94081713. They are not an official
MiniMax training recipe or source-parity authority.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import struct
import sys
import wave
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import torch
import torch.nn.functional as functional
from dinkster_api.v1 import AssetRef, digest_bytes
from dinkster_inference import (
    AUDIO_CODE_OFFSET,
    C0_VOCAB_SIZE,
    MAX_PROMPT_TOKENS,
    SafetensorsSource,
    load_safetensors_header,
)
from dinkster_inference_torch import (
    MiniMaxMusic3TextModel,
    load_tensors,
    tokenize_music_prompt,
)
from tokenizers import Tokenizer

from .checkpoint import blake3_digest, canonical_json
from .durability import advisory_file_lock, atomic_replace, durable_mkdir

MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION = "SimpleTuner/def7bbc065e5f15d9e551827e247ba046fe36eb6"
MINIMAX_MUSIC3_RVQ_SOURCE_REVISION = (
    "SimpleTuner/open-rvq-encoder-minimax-music3@326964c2f4edcc642c1ea116274dd2dd94081713"
)
MINIMAX_MUSIC3_DAV_SOURCE_REVISION = (
    "SimpleTuner/MiniMax-Music-3-Encoder@fce0d00b1ae42ee47874babb8c06fb859eb01443"
)
MINIMAX_MUSIC3_DAV_ENCODER_CONTRACT = "community-minimax-music3-dav-mean-stereo-44100-v1"
MINIMAX_MUSIC3_RVQ_ENCODER_CONTRACT = "community-minimax-music3-rvq-v4-169m-argmax-v1"
MINIMAX_MUSIC3_TEACHER_FORCING_CONTRACT = "community-minimax-music3-official-ar-teacher-forcing-v1"
MINIMAX_MUSIC3_FLOW_OBJECTIVE = "community-minimax-music3-logistic-normal-flow-v1"
MINIMAX_MUSIC3_SAMPLE_RATE = 44_100
MINIMAX_MUSIC3_AUDIO_FRAMES_PER_SECOND = 25
MINIMAX_MUSIC3_DAV_HOP = 512
MINIMAX_MUSIC3_RVQ_CODEBOOKS = 8
MINIMAX_MUSIC3_MAX_RVQ_FRAMES = 128
_CACHE_FORMAT = "dinkster.encoded-minimax-music3-community-dataset.v1"


@dataclass(frozen=True)
class MiniMaxMusic3ArtifactPin:
    """One immutable local artifact and any native runtime identity it carries."""

    path: str
    digest: str
    size: int
    identity: str | None = None

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path": self.path,
            "digest": self.digest,
            "size": self.size,
        }
        if self.identity is not None:
            result["identity"] = self.identity
        return result

    def identity_mapping(self) -> dict[str, object]:
        result = self.to_mapping()
        result.pop("path")
        return result


@dataclass(frozen=True)
class MiniMaxMusic3DatasetItem:
    audio_path: Path
    caption_path: Path
    lyrics_path: Path
    relative_audio: str
    relative_caption: str
    relative_lyrics: str
    audio_digest: str
    caption_digest: str | None
    lyrics_digest: str | None


@dataclass(frozen=True)
class MiniMaxMusic3DatasetInspection:
    digest: str
    items: tuple[MiniMaxMusic3DatasetItem, ...]
    caption_files: int
    nonempty_captions: int
    lyrics_files: int
    nonempty_lyrics: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class MiniMaxMusic3DatasetSettings:
    root: str
    audio_frames: int
    dav_encoder_state: MiniMaxMusic3ArtifactPin
    rvq_encoder_state: MiniMaxMusic3ArtifactPin
    text_encoder_state: MiniMaxMusic3ArtifactPin
    text_compute_dtype: Literal["bfloat16", "float32"]
    inspection: MiniMaxMusic3DatasetInspection
    encoded_cache_root: str | None = None

    @property
    def digest(self) -> str:
        return self.inspection.digest

    @property
    def samples_per_item(self) -> int:
        return self.audio_frames * (
            MINIMAX_MUSIC3_SAMPLE_RATE // MINIMAX_MUSIC3_AUDIO_FRAMES_PER_SECOND
        )

    @property
    def latent_frames(self) -> int:
        return math.ceil(self.samples_per_item / MINIMAX_MUSIC3_DAV_HOP)

    def to_mapping(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": "minimax-music3-audio-caption-lyrics-folder",
            "root": self.root,
            "audioFrames": self.audio_frames,
            "davEncoderState": self.dav_encoder_state.to_mapping(),
            "rvqEncoderState": self.rvq_encoder_state.to_mapping(),
            "textEncoderState": self.text_encoder_state.to_mapping(),
            "digest": self.digest,
        }
        if self.encoded_cache_root is not None:
            result["encodedCacheRoot"] = self.encoded_cache_root
        return result


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _inspect_utf8_sidecar(
    path: Path,
    relative: str,
    kind: str,
    errors: list[str],
) -> tuple[str | None, bool]:
    if not path.is_file():
        errors.append(f"missing {kind} {relative}")
        return None, False
    data = path.read_bytes()
    digest = digest_bytes(data)
    try:
        value = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        errors.append(f"{relative}: {kind} is not UTF-8: {exc}")
        return digest, False
    if not value.strip():
        errors.append(f"{relative}: {kind} is empty")
        return digest, False
    return digest, True


def _validate_wav(path: Path, relative: str, samples: int, errors: list[str]) -> None:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frame_count = audio.getnframes()
            compression = audio.getcomptype()
            payload = audio.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        errors.append(f"{relative}: cannot decode PCM WAV audio: {exc}")
        return
    if compression != "NONE":
        errors.append(f"{relative}: audio must use uncompressed PCM")
    if channels not in (1, 2):
        errors.append(f"{relative}: audio must have one or two channels")
    if sample_width != 2:
        errors.append(f"{relative}: audio must use signed 16-bit PCM")
    if sample_rate != MINIMAX_MUSIC3_SAMPLE_RATE:
        errors.append(f"{relative}: audio sample rate must be 44100 Hz")
    if frame_count != samples:
        errors.append(f"{relative}: audio has {frame_count} samples; expected {samples}")
    if len(payload) != frame_count * channels * sample_width:
        errors.append(f"{relative}: PCM payload byte count is inconsistent")


def inspect_minimax_music3_dataset(
    root: Path,
    audio_frames: int,
    dav_encoder_state: MiniMaxMusic3ArtifactPin,
    rvq_encoder_state: MiniMaxMusic3ArtifactPin,
    text_encoder_state: MiniMaxMusic3ArtifactPin,
    text_compute_dtype: Literal["bfloat16", "float32"],
    *,
    validate_audio: bool = True,
) -> MiniMaxMusic3DatasetInspection:
    """Hash and validate fixed-duration Music 3 audio/caption/lyrics items."""
    resolved = root.expanduser().resolve()
    errors: list[str] = []
    paths = (
        sorted(
            (path for path in resolved.rglob("*.wav") if path.is_file()),
            key=lambda path: path.relative_to(resolved).as_posix(),
        )
        if resolved.is_dir()
        else []
    )
    if not resolved.is_dir():
        errors.append(f"dataset root is not a directory: {resolved}")
    if not paths:
        errors.append("dataset contains no .wav audio files")
    samples = audio_frames * (MINIMAX_MUSIC3_SAMPLE_RATE // MINIMAX_MUSIC3_AUDIO_FRAMES_PER_SECOND)
    items: list[MiniMaxMusic3DatasetItem] = []
    caption_files = nonempty_captions = lyrics_files = nonempty_lyrics = 0
    for audio_path in paths:
        relative_audio = audio_path.relative_to(resolved).as_posix()
        caption_path = audio_path.with_suffix(".caption.txt")
        lyrics_path = audio_path.with_suffix(".lyrics.txt")
        relative_caption = caption_path.relative_to(resolved).as_posix()
        relative_lyrics = lyrics_path.relative_to(resolved).as_posix()
        audio_data = audio_path.read_bytes()
        if validate_audio:
            _validate_wav(audio_path, relative_audio, samples, errors)
        caption_digest, caption_nonempty = _inspect_utf8_sidecar(
            caption_path, relative_caption, "caption", errors
        )
        lyrics_digest, lyrics_nonempty = _inspect_utf8_sidecar(
            lyrics_path, relative_lyrics, "lyrics", errors
        )
        caption_files += caption_digest is not None
        nonempty_captions += caption_nonempty
        lyrics_files += lyrics_digest is not None
        nonempty_lyrics += lyrics_nonempty
        items.append(
            MiniMaxMusic3DatasetItem(
                audio_path,
                caption_path,
                lyrics_path,
                relative_audio,
                relative_caption,
                relative_lyrics,
                digest_bytes(audio_data),
                caption_digest,
                lyrics_digest,
            )
        )
    identity = [
        _CACHE_FORMAT,
        [
            {
                "audio": item.relative_audio,
                "audioDigest": item.audio_digest,
                "caption": item.relative_caption,
                "captionDigest": item.caption_digest,
                "lyrics": item.relative_lyrics,
                "lyricsDigest": item.lyrics_digest,
            }
            for item in items
        ],
        {
            "audioFrames": audio_frames,
            "sampleRateHz": MINIMAX_MUSIC3_SAMPLE_RATE,
            "samplesPerItem": samples,
            "davContract": MINIMAX_MUSIC3_DAV_ENCODER_CONTRACT,
            "davSource": MINIMAX_MUSIC3_DAV_SOURCE_REVISION,
            "davState": dav_encoder_state.identity_mapping(),
            "rvqContract": MINIMAX_MUSIC3_RVQ_ENCODER_CONTRACT,
            "rvqSource": MINIMAX_MUSIC3_RVQ_SOURCE_REVISION,
            "rvqState": rvq_encoder_state.identity_mapping(),
            "teacherForcingContract": MINIMAX_MUSIC3_TEACHER_FORCING_CONTRACT,
            "textState": text_encoder_state.identity_mapping(),
            "textComputeDtype": text_compute_dtype,
        },
    ]
    return MiniMaxMusic3DatasetInspection(
        digest_bytes(_canonical_json(identity)),
        tuple(items),
        caption_files,
        nonempty_captions,
        lyrics_files,
        nonempty_lyrics,
        tuple(errors),
    )


class _WeightNormalizedConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.weight_g = torch.nn.Parameter(torch.empty(out_channels, 1, 1))
        self.weight_v = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        self.stride = stride
        self.dilation = dilation
        self.padding = padding

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        weight = torch._weight_norm(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
            self.weight_v, self.weight_g, 0
        )
        return functional.conv1d(
            hidden,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class _Snake(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.empty(1, channels, 1))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + torch.sin(self.alpha * hidden).square() / (self.alpha + 1e-9)


class _DavResidual(torch.nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            _Snake(channels),
            _WeightNormalizedConv1d(channels, channels, 7, dilation=dilation, padding=3 * dilation),
            _Snake(channels),
            _WeightNormalizedConv1d(channels, channels, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden)
        if residual.shape[-1] != hidden.shape[-1]:
            offset = (hidden.shape[-1] - residual.shape[-1]) // 2
            hidden = hidden[..., offset : offset + residual.shape[-1]]
        return hidden + residual


class _DavEncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            _DavResidual(in_channels, 1),
            _DavResidual(in_channels, 3),
            _DavResidual(in_channels, 9),
            _Snake(in_channels),
            _WeightNormalizedConv1d(
                in_channels,
                out_channels,
                2 * stride,
                stride=stride,
                padding=stride // 2,
            ),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.block(hidden)


class MiniMaxMusic3DavEncoder(torch.nn.Module):
    """DAV posterior-mean encoder exposed by the pinned community conversion."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.block = torch.nn.Sequential(  # pyright: ignore[reportAttributeAccessIssue]
            _WeightNormalizedConv1d(1, 64, 7, padding=3),
            _DavEncoderBlock(64, 128, 2),
            _DavEncoderBlock(128, 256, 4),
            _DavEncoderBlock(256, 512, 8),
            _DavEncoderBlock(512, 1024, 8),
            _Snake(1024),
            _WeightNormalizedConv1d(1024, 1024, 3, padding=1),
        )
        self.mean_proj = torch.nn.Conv1d(1024, 64, 1)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] not in (1, 2):
            raise ValueError(
                "MiniMax Music 3 waveform must be [batch, one-or-two channels, samples]"
            )
        if waveform.shape[1] == 1:
            waveform = waveform.expand(-1, 2, -1)
        padding = -waveform.shape[-1] % MINIMAX_MUSIC3_DAV_HOP
        if padding:
            waveform = functional.pad(waveform, (0, padding))
        folded = waveform.reshape(waveform.shape[0] * 2, 1, waveform.shape[-1])
        block = cast(torch.nn.Sequential, self.encoder.block)  # pyright: ignore[reportAttributeAccessIssue]
        encoded = self.mean_proj(block(folded))
        return encoded.reshape(waveform.shape[0], 128, encoded.shape[-1])

    forward = encode


class _RvqResidual(torch.nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.norm = torch.nn.GroupNorm(1, channels)
        self.conv1 = torch.nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = torch.nn.Conv1d(channels, channels, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = self.conv1(functional.gelu(self.norm(hidden)))
        return hidden + self.conv2(functional.gelu(value))


class _RvqTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        width: int,
        heads: int,
        ff_width: int,
        dropout: float,
        *,
        attention_multiplier: float,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.attention_multiplier = attention_multiplier
        self.norm1 = torch.nn.LayerNorm(width)
        self.norm2 = torch.nn.LayerNorm(width)
        self.q_proj = torch.nn.Linear(width, width)
        self.k_proj = torch.nn.Linear(width, width)
        self.v_proj = torch.nn.Linear(width, width)
        self.out_proj = torch.nn.Linear(width, width)
        self.linear1 = torch.nn.Linear(width, ff_width)
        self.linear2 = torch.nn.Linear(ff_width, width)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
        normalized = self.norm1(hidden)
        shape = (*normalized.shape[:2], self.heads, self.head_dim)
        query = self.q_proj(normalized).reshape(shape).transpose(1, 2)
        key = self.k_proj(normalized).reshape(shape).transpose(1, 2)
        value = self.v_proj(normalized).reshape(shape).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores * (self.attention_multiplier / self.head_dim)
        if causal:
            mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).triu_(1)
            scores = scores.masked_fill(mask, -torch.inf)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        attended = torch.matmul(self.dropout(weights), value)
        attended = attended.transpose(1, 2).reshape_as(normalized)
        hidden = hidden + self.dropout(self.out_proj(attended))
        value = self.linear2(self.dropout(functional.gelu(self.linear1(self.norm2(hidden)))))
        return hidden + self.dropout(value)


class _RvqDepthDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_projection = torch.nn.Linear(1088, 512, bias=False)
        self.position = torch.nn.Parameter(torch.empty(1, 8, 512))
        self.prior_embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(C0_VOCAB_SIZE, 512)]
            + [torch.nn.Embedding(1024, 512) for _ in range(6)]
        )
        self.layers = torch.nn.ModuleList(
            _RvqTransformerLayer(
                512,
                8,
                2048,
                0.1,
                attention_multiplier=math.sqrt(64),
            )
            for _ in range(2)
        )
        self.norm = torch.nn.LayerNorm(512)
        self.heads = torch.nn.ModuleList(torch.nn.Linear(512, 1024) for _ in range(7))

    def codes(self, context: torch.Tensor, semantic: torch.Tensor) -> tuple[torch.Tensor, ...]:
        sequence = [self.context_projection(context).unsqueeze(1)]
        sequence.append(self.prior_embeddings[0](semantic).unsqueeze(1))
        result: list[torch.Tensor] = []
        for index, head in enumerate(self.heads):
            hidden = torch.cat(sequence, dim=1)
            hidden = hidden + self.position[:, : hidden.shape[1]]
            for layer in self.layers:
                hidden = layer(hidden, causal=True)
            code = head(self.norm(hidden[:, -1])).argmax(dim=-1)
            result.append(code)
            if index < len(self.prior_embeddings) - 1:
                sequence.append(self.prior_embeddings[index + 1](code).unsqueeze(1))
        return tuple(result)


class MiniMaxMusic3RvqEncoder(torch.nn.Module):
    """Pinned community 169M DAV-to-RVQ approximation used for conditioning."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_in = torch.nn.Conv1d(128, 1088, 7, padding=3)
        self.blocks = torch.nn.ModuleList(_RvqResidual(1088, dilation) for dilation in (1, 3, 9))
        self.position = torch.nn.Parameter(torch.empty(1, 128, 1088))
        self.transformer = torch.nn.ModuleList(
            _RvqTransformerLayer(1088, 17, 4352, 0.1, attention_multiplier=8.0) for _ in range(8)
        )
        self.norm_out = torch.nn.LayerNorm(1088)
        self.heads = torch.nn.ModuleList((torch.nn.Linear(1088, C0_VOCAB_SIZE),))
        self.depth_decoder = _RvqDepthDecoder()

    def encode_codes(self, latents: torch.Tensor, pool: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[-1] != 128:
            raise ValueError("RVQ encoder latents must be [batch, DAV frames, 128]")
        if pool.ndim != 3 or pool.shape[0] != latents.shape[0] or pool.shape[2] != latents.shape[1]:
            raise ValueError("RVQ encoder pool must be [batch, audio frames, DAV frames]")
        if pool.shape[1] > MINIMAX_MUSIC3_MAX_RVQ_FRAMES:
            raise ValueError("RVQ encoder accepts at most 128 audio frames")
        hidden = self.conv_in(latents.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        hidden = torch.matmul(pool, hidden.transpose(1, 2))
        hidden = hidden + self.position[:, : hidden.shape[1]]
        for layer in self.transformer:
            hidden = layer(hidden)
        hidden = self.norm_out(hidden)
        semantic = self.heads[0](hidden).argmax(dim=-1)
        flattened = hidden.reshape(-1, hidden.shape[-1])
        semantic_flat = semantic.reshape(-1)
        acoustic = self.depth_decoder.codes(flattened, semantic_flat)
        return torch.stack((semantic_flat, *acoustic), dim=-1).reshape(
            latents.shape[0], pool.shape[1], MINIMAX_MUSIC3_RVQ_CODEBOOKS
        )

    forward = encode_codes


def minimax_music3_rvq_pool(
    audio_frames: int,
    dav_frames: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Map contiguous DAV positions to ordinary real-audio 25 Hz frames."""
    if not 1 <= audio_frames <= MINIMAX_MUSIC3_MAX_RVQ_FRAMES:
        raise ValueError("audio frame count must be in [1, 128]")
    expected = math.ceil(audio_frames * (MINIMAX_MUSIC3_SAMPLE_RATE // 25) / 512)
    if dav_frames != expected:
        raise ValueError(f"DAV frame count must be {expected} for {audio_frames} audio frames")
    pool = torch.zeros((audio_frames, dav_frames), dtype=torch.float32, device=device)
    for frame in range(audio_frames):
        start = frame * 441 // 128
        stop = (frame + 1) * 441 // 128
        pool[frame, start:stop] = 1.0 / (stop - start)
    return pool.unsqueeze(0)


@dataclass(frozen=True)
class MiniMaxMusic3PreparedBatch:
    latents: torch.Tensor
    context: torch.Tensor
    timesteps: torch.Tensor | None = None
    noise: torch.Tensor | None = None


class MiniMaxMusic3PreparedBatchSource(Protocol):
    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxMusic3PreparedBatch: ...


def teacher_forced_minimax_music3_context(
    model: MiniMaxMusic3TextModel,
    tokenizer: Tokenizer,
    caption: str,
    lyrics: str,
    codes: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Build source-aligned official AR/depth hiddens from community RVQ codes."""
    if codes.ndim != 2 or codes.shape[1] != 8 or codes.dtype != torch.int64:
        raise ValueError("Music 3 teacher-forcing codes must be int64 [frames, 8]")
    if bool(torch.any((codes[:, 0] < 0) | (codes[:, 0] >= C0_VOCAB_SIZE)).item()):
        raise ValueError("Music 3 C0 codes fall outside the 16384-entry vocabulary")
    if bool(torch.any((codes[:, 1:] < 0) | (codes[:, 1:] >= 1024)).item()):
        raise ValueError("Music 3 depth codes fall outside the 1024-entry vocabularies")
    ids = tuple(tokenize_music_prompt(tokenizer, caption, lyrics))
    if not ids or len(ids) > MAX_PROMPT_TOKENS:
        raise ValueError("Music 3 prompt token count must be in [1, 5000]")
    embedding = (
        model.model.embed_tokens_prefill if model.config.pruned else model.model.embed_tokens
    )
    device = next(embedding.parameters()).device
    prompt = embedding(torch.tensor([ids], dtype=torch.int64, device=device))
    codes = codes.to(device)
    c0 = (
        model.model.embed_tokens_audio(codes[:, 0])
        if model.config.pruned
        else model.model.embed_tokens(codes[:, 0] + AUDIO_CODE_OFFSET)
    )
    offsets = torch.arange(7, device=device) * model.config.audio_vocab_size
    extra = model.model.audio_extra_embedding(codes[:, 1:] + offsets).sum(dim=1)
    frames = (c0 + extra) * (model.config.audio_num_codebooks**-0.5)
    sequence = torch.cat((prompt.squeeze(0), frames), dim=0).unsqueeze(0).to(compute_dtype)
    hidden, _cache = model.model.forward_causal(None, embeds=sequence)
    predictor = hidden[0, len(ids) - 1 : len(ids) - 1 + codes.shape[0]]
    decoder = model.model.audio_decoder
    depth_inputs = [decoder.projection(predictor).unsqueeze(1)]
    depth_inputs.append(decoder.projection(c0).unsqueeze(1))
    for index in range(6):
        embedded = model.model.audio_extra_embedding(
            codes[:, index + 1] + index * model.config.audio_vocab_size
        )
        depth_inputs.append(decoder.projection(embedded).unsqueeze(1))
    depth_hidden = decoder(torch.cat(depth_inputs, dim=1))[:, 1:]
    context = torch.cat((predictor, depth_hidden.flatten(1)), dim=-1)
    return context.unsqueeze(0).float().cpu()


@dataclass(frozen=True)
class _FixedResolver:
    path: Path

    def resolve(self, digest: str) -> Path:
        del digest
        return self.path


def _verified_path(pin: MiniMaxMusic3ArtifactPin) -> Path:
    path = Path(pin.path)
    if path.stat().st_size != pin.size:
        raise ValueError(f"artifact {path} byte size differs from its training config")
    return AssetRef(pin.digest, path.name, pin.size, resolver=_FixedResolver(path)).local_path()


def load_minimax_music3_dav_encoder(
    pin: MiniMaxMusic3ArtifactPin,
) -> MiniMaxMusic3DavEncoder:
    """Strict-load the encoder from the pinned community safetensors conversion."""
    path = _verified_path(pin)
    source = load_safetensors_header(path, asset_digest=pin.digest, asset_size=pin.size)
    with torch.device("meta"):
        model = MiniMaxMusic3DavEncoder()
    expected = set(model.state_dict())
    selected = {
        key for key in source.keys() if key.startswith("encoder.") or key.startswith("mean_proj.")
    }
    if selected != expected:
        raise ValueError(
            "community Music 3 DAV encoder state layout differs:"
            f" missing={sorted(expected - selected)},"
            f" unknown={sorted(selected - expected)}"
        )
    model.load_state_dict(load_tensors(path, tuple(sorted(selected))), strict=True, assign=True)
    return model.requires_grad_(False).eval()


def load_minimax_music3_rvq_encoder(
    pin: MiniMaxMusic3ArtifactPin,
) -> MiniMaxMusic3RvqEncoder:
    """Digest-verify and strict-load the pinned community RVQ v4 encoder."""
    path = _verified_path(pin)
    source = load_safetensors_header(path, asset_digest=pin.digest, asset_size=pin.size)
    with torch.device("meta"):
        model = MiniMaxMusic3RvqEncoder()
    expected = set(model.state_dict())
    if set(source.keys()) != expected:
        raise ValueError(
            "community Music 3 RVQ state layout differs:"
            f" missing={sorted(expected - set(source.keys()))},"
            f" unknown={sorted(set(source.keys()) - expected)}"
        )
    model.load_state_dict(load_tensors(path), strict=True, assign=True)
    return model.requires_grad_(False).eval()


@dataclass(frozen=True)
class EncodedMiniMaxMusic3DatasetTensors:
    latents: torch.Tensor
    context: torch.Tensor


def _execution_fingerprint(device: torch.device) -> dict[str, object]:
    value: dict[str, object] = {
        "device": device.type,
        "torchVersion": str(torch.__version__),
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "float32MatmulPrecision": torch.get_float32_matmul_precision(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        value.update(
            {
                "cudaRuntime": str(torch.version.cuda),
                "cudnnBenchmark": torch.backends.cudnn.benchmark,
                "cudnnDeterministic": torch.backends.cudnn.deterministic,
                "deviceCapability": [properties.major, properties.minor],
                "deviceName": properties.name,
            }
        )
    else:
        value["cpuCapability"] = torch.backends.cpu.get_cpu_capability()
        value["machine"] = platform.machine()
    return value


def minimax_music3_encoded_cache_key(
    settings: MiniMaxMusic3DatasetSettings,
    *,
    device: torch.device,
) -> str:
    return blake3_digest(
        canonical_json(
            [
                _CACHE_FORMAT,
                settings.digest,
                settings.text_compute_dtype,
                _execution_fingerprint(device),
            ]
        )
    )


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def _safetensors_bytes(tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> bytes:
    if sys.byteorder != "little":
        raise RuntimeError("encoded dataset caching requires a little-endian host")
    header: dict[str, object] = {"__metadata__": metadata}
    payload = bytearray()
    for name in sorted(tensors):
        tensor = tensors[name]
        if tensor.dtype != torch.float32:
            raise ValueError(f"encoded Music 3 cache tensor {name!r} must be float32")
        data = _tensor_bytes(tensor)
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    raw = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + bytes(payload)


class MiniMaxMusic3EncodedDatasetCache:
    """Verified community-derived DAV latent and native AR context cache."""

    def __init__(self, settings: MiniMaxMusic3DatasetSettings, *, device: torch.device) -> None:
        if settings.encoded_cache_root is None:
            raise ValueError("encoded Music 3 dataset cache root is not configured")
        self._settings = settings
        self._key = minimax_music3_encoded_cache_key(settings, device=device)
        root = Path(settings.encoded_cache_root) / "minimax-music3-community-v1"
        self._entries = root / "entries"
        self._shards = root / "shards"
        self._manifest = self._entries / f"{self._key[7:]}.json"
        self._lock = root / "locks" / f"{self._key[7:]}.lock"

    def build_lock(self) -> AbstractContextManager[None]:
        return advisory_file_lock(self._lock)

    def _verified_source(self) -> SafetensorsSource:
        value = json.loads(self._manifest.read_text(encoding="ascii"))
        if not isinstance(value, dict) or set(value) != {
            "schemaVersion",
            "cacheKey",
            "datasetDigest",
            "shardDigest",
        }:
            raise ValueError("encoded Music 3 cache manifest has an unknown layout")
        if (
            value["schemaVersion"] != 1
            or value["cacheKey"] != self._key
            or value["datasetDigest"] != self._settings.digest
        ):
            raise ValueError("encoded Music 3 cache manifest identity does not match")
        digest = value["shardDigest"]
        if not isinstance(digest, str) or not digest.startswith("blake3:") or len(digest) != 71:
            raise ValueError("encoded Music 3 cache manifest has an invalid shard digest")
        path = self._shards / f"{digest[7:]}.safetensors"
        if blake3_digest(path.read_bytes()) != digest:
            raise ValueError("encoded Music 3 cache shard failed digest verification")
        source = load_safetensors_header(path)
        expected_metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "minimax-music3",
            "format": _CACHE_FORMAT,
            "itemCount": str(len(self._settings.inspection.items)),
        }
        if dict(source.metadata()) != expected_metadata or set(source.keys()) != {
            "latents",
            "context",
        }:
            raise ValueError("encoded Music 3 cache shard layout does not match")
        count = len(self._settings.inspection.items)
        shapes = {
            "latents": (count, 128, self._settings.latent_frames),
            "context": (count, self._settings.audio_frames, 8 * 4096),
        }
        for name, shape in shapes.items():
            geometry = source.entry(name).geometry
            if geometry.dtype.name != "float32" or geometry.shape != shape:
                raise ValueError(f"encoded Music 3 {name} has the wrong shape or dtype")
        return source

    def load(self) -> EncodedMiniMaxMusic3DatasetTensors | None:
        try:
            values = load_tensors(self._verified_source().path)
            return EncodedMiniMaxMusic3DatasetTensors(values["latents"], values["context"])
        except (OSError, ValueError, KeyError, RuntimeError):
            return None

    def is_valid(self) -> bool:
        try:
            self._verified_source()
        except (OSError, ValueError, KeyError, RuntimeError):
            return False
        return True

    def write(self, tensors: EncodedMiniMaxMusic3DatasetTensors) -> str:
        count = len(self._settings.inspection.items)
        expected = {
            "latents": (count, 128, self._settings.latent_frames),
            "context": (count, self._settings.audio_frames, 8 * 4096),
        }
        values = {"latents": tensors.latents, "context": tensors.context}
        for name, shape in expected.items():
            tensor = values[name]
            if (
                tensor.dtype != torch.float32
                or tuple(tensor.shape) != shape
                or not bool(torch.isfinite(tensor).all())
            ):
                raise ValueError(f"encoded Music 3 {name} has invalid shape, dtype, or values")
        metadata = {
            "cacheKey": self._key,
            "datasetDigest": self._settings.digest,
            "family": "minimax-music3",
            "format": _CACHE_FORMAT,
            "itemCount": str(count),
        }
        shard = _safetensors_bytes(values, metadata)
        digest = blake3_digest(shard)
        durable_mkdir(self._shards)
        durable_mkdir(self._entries)
        atomic_replace(self._shards / f"{digest[7:]}.safetensors", shard)
        atomic_replace(
            self._manifest,
            canonical_json(
                {
                    "schemaVersion": 1,
                    "cacheKey": self._key,
                    "datasetDigest": self._settings.digest,
                    "shardDigest": digest,
                }
            ),
        )
        return digest


def minimax_music3_encoded_cache_state(
    settings: MiniMaxMusic3DatasetSettings,
    *,
    device: torch.device,
) -> str:
    if settings.encoded_cache_root is None:
        return "disabled"
    cache = MiniMaxMusic3EncodedDatasetCache(settings, device=device)
    return "hit" if cache.is_valid() else "miss"


MiniMaxMusic3TextFactory = Callable[[], tuple[MiniMaxMusic3TextModel, Tokenizer, torch.dtype]]


class MiniMaxMusic3DatasetSource:
    """Precompute and serve fixed-duration community-derived Music 3 batches."""

    def __init__(
        self,
        settings: MiniMaxMusic3DatasetSettings,
        dav_factory: Callable[[], MiniMaxMusic3DavEncoder],
        rvq_factory: Callable[[], MiniMaxMusic3RvqEncoder],
        text_factory: MiniMaxMusic3TextFactory,
        *,
        device: torch.device,
    ) -> None:
        current = inspect_minimax_music3_dataset(
            Path(settings.root),
            settings.audio_frames,
            settings.dav_encoder_state,
            settings.rvq_encoder_state,
            settings.text_encoder_state,
            settings.text_compute_dtype,
        )
        if current.digest != settings.digest:
            raise ValueError(
                f"dataset digest changed: expected {settings.digest}, got {current.digest}"
            )
        if current.errors:
            raise ValueError("dataset validation failed: " + "; ".join(current.errors))
        self.settings = settings
        self._items = current.items
        self._permutations: dict[int, tuple[int, ...]] = {}
        cache = (
            None
            if settings.encoded_cache_root is None
            else MiniMaxMusic3EncodedDatasetCache(settings, device=device)
        )
        cached = None if cache is None else cache.load()
        if cached is not None:
            self._use_cached(cached)
            return
        if cache is None:
            self._build(dav_factory, rvq_factory, text_factory, device)
            return
        with cache.build_lock():
            cached = cache.load()
            if cached is not None:
                self._use_cached(cached)
                return
            self._build(dav_factory, rvq_factory, text_factory, device)
            cache.write(self._cache_tensors())

    @staticmethod
    def _release(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    @staticmethod
    def _text(item: MiniMaxMusic3DatasetItem, path: Path, digest: str | None) -> str:
        data = path.read_bytes()
        if digest_bytes(data) != digest:
            raise ValueError(f"dataset sidecar changed after validation: {path.name}")
        return data.decode("utf-8").strip()

    @staticmethod
    def _waveform(item: MiniMaxMusic3DatasetItem) -> torch.Tensor:
        if sys.byteorder != "little":
            raise RuntimeError("MiniMax Music 3 PCM decoding requires a little-endian host")
        data = item.audio_path.read_bytes()
        if digest_bytes(data) != item.audio_digest:
            raise ValueError(f"dataset audio changed after validation: {item.relative_audio}")
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels = audio.getnchannels()
            samples = audio.getnframes()
            payload = audio.readframes(samples)
        values = torch.frombuffer(bytearray(payload), dtype=torch.int16).reshape(samples, channels)
        return values.transpose(0, 1).float().div_(32768.0).unsqueeze(0)

    def _build(
        self,
        dav_factory: Callable[[], MiniMaxMusic3DavEncoder],
        rvq_factory: Callable[[], MiniMaxMusic3RvqEncoder],
        text_factory: MiniMaxMusic3TextFactory,
        device: torch.device,
    ) -> None:
        try:
            dav = dav_factory().requires_grad_(False).eval().to(device=device, dtype=torch.float32)
            with torch.no_grad():
                self._latents = tuple(
                    dav.encode(self._waveform(item).to(device)).float().squeeze(0).cpu()
                    for item in self._items
                )
            del dav
        finally:
            self._release(device)
        try:
            rvq = rvq_factory().requires_grad_(False).eval().to(device=device, dtype=torch.float32)
            pool = minimax_music3_rvq_pool(
                self.settings.audio_frames,
                self.settings.latent_frames,
                device=device,
            )
            with torch.no_grad():
                codes = tuple(
                    rvq.encode_codes(latent.transpose(0, 1).unsqueeze(0).to(device), pool)
                    .squeeze(0)
                    .to(dtype=torch.int64, device="cpu")
                    for latent in self._latents
                )
            del rvq
        finally:
            self._release(device)
        try:
            text, tokenizer, compute_dtype = text_factory()
            expected_text_dtype = {
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.settings.text_compute_dtype]
            if compute_dtype != expected_text_dtype:
                raise ValueError(
                    f"text factory selected {compute_dtype}; expected {expected_text_dtype}"
                )
            text = text.requires_grad_(False).eval().to(device=device)
            with torch.no_grad():
                self._contexts = tuple(
                    teacher_forced_minimax_music3_context(
                        text,
                        tokenizer,
                        self._text(item, item.caption_path, item.caption_digest),
                        self._text(item, item.lyrics_path, item.lyrics_digest),
                        code,
                        compute_dtype=compute_dtype,
                    ).squeeze(0)
                    for item, code in zip(self._items, codes, strict=True)
                )
            del text
        finally:
            self._release(device)
        self._validate()

    def _use_cached(self, tensors: EncodedMiniMaxMusic3DatasetTensors) -> None:
        self._latents = tuple(tensors.latents.unbind())
        self._contexts = tuple(tensors.context.unbind())
        self._validate()

    def _validate(self) -> None:
        if len(self._latents) != len(self._items) or len(self._contexts) != len(self._items):
            raise ValueError("encoded Music 3 dataset item counts do not match")
        if any(
            value.dtype != torch.float32
            or tuple(value.shape) != (128, self.settings.latent_frames)
            or not bool(torch.isfinite(value).all())
            for value in self._latents
        ):
            raise ValueError("encoded Music 3 DAV latents have invalid shape, dtype, or values")
        if any(
            value.dtype != torch.float32
            or tuple(value.shape) != (self.settings.audio_frames, 8 * 4096)
            or not bool(torch.isfinite(value).all())
            for value in self._contexts
        ):
            raise ValueError("encoded Music 3 AR context has invalid shape, dtype, or values")

    def _cache_tensors(self) -> EncodedMiniMaxMusic3DatasetTensors:
        return EncodedMiniMaxMusic3DatasetTensors(
            torch.stack(self._latents), torch.stack(self._contexts)
        )

    def _item_index(self, cursor: int) -> int:
        epoch, offset = divmod(cursor, len(self._items))
        found = self._permutations.get(epoch)
        if found is None:
            digest = hashlib.sha256(
                f"{self.settings.digest}:epoch:{epoch}".encode("ascii")
            ).digest()
            generator = torch.Generator(device="cpu").manual_seed(
                int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
            )
            found = tuple(torch.randperm(len(self._items), generator=generator).tolist())
            self._permutations[epoch] = found
        return found[offset]

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxMusic3PreparedBatch:
        del generator
        if cursor < 0:
            raise ValueError("data cursor must be non-negative")
        index = self._item_index(cursor)
        return MiniMaxMusic3PreparedBatch(
            self._latents[index].unsqueeze(0).to(device),
            self._contexts[index].unsqueeze(0).to(device),
        )


__all__ = [
    "MINIMAX_MUSIC3_AUDIO_FRAMES_PER_SECOND",
    "MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION",
    "MINIMAX_MUSIC3_DAV_ENCODER_CONTRACT",
    "MINIMAX_MUSIC3_DAV_SOURCE_REVISION",
    "MINIMAX_MUSIC3_FLOW_OBJECTIVE",
    "MINIMAX_MUSIC3_RVQ_ENCODER_CONTRACT",
    "MINIMAX_MUSIC3_RVQ_SOURCE_REVISION",
    "MINIMAX_MUSIC3_SAMPLE_RATE",
    "MINIMAX_MUSIC3_TEACHER_FORCING_CONTRACT",
    "EncodedMiniMaxMusic3DatasetTensors",
    "MiniMaxMusic3ArtifactPin",
    "MiniMaxMusic3DatasetInspection",
    "MiniMaxMusic3DatasetItem",
    "MiniMaxMusic3DatasetSettings",
    "MiniMaxMusic3DatasetSource",
    "MiniMaxMusic3DavEncoder",
    "MiniMaxMusic3EncodedDatasetCache",
    "MiniMaxMusic3PreparedBatch",
    "MiniMaxMusic3PreparedBatchSource",
    "MiniMaxMusic3RvqEncoder",
    "inspect_minimax_music3_dataset",
    "load_minimax_music3_dav_encoder",
    "load_minimax_music3_rvq_encoder",
    "minimax_music3_encoded_cache_key",
    "minimax_music3_encoded_cache_state",
    "minimax_music3_rvq_pool",
    "teacher_forced_minimax_music3_context",
]
