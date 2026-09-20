"""Executable proofs for MiniMax H3 video/audio/caption datasets."""

from __future__ import annotations

import contextlib
import gc
import json
import shutil
import subprocess
import sys
import threading
import wave
import weakref
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Never, cast

import dinkster_training_torch.config as training_config
import dinkster_training_torch.data as training_data
import dinkster_training_torch.dataset as training_dataset
import dinkster_training_torch.encoded_cache as training_encoded_cache
import dinkster_training_torch.service as training_service
import pytest
import torch
import torch.nn.functional as functional
from dinkster_inference import MINIMAX_H3_CONFIG, LatentStream, MultiStreamLatent
from dinkster_inference_torch import (
    MiniMaxH3AudioVaeRuntime,
    MiniMaxH3ConditionerRuntime,
    MiniMaxH3VideoVaeRuntime,
)
from dinkster_inference_torch.minimax_h3_conditioning import MiniMaxH3ConditionerInputs
from dinkster_inference_torch.minimax_h3_dit import MiniMaxH3DiTConditioning
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    MINIMAX_H3_TRAINING_RUNTIME_IDENTITY,
    MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST,
    ContentAddressedCheckpointStore,
    MiniMaxH3DatasetSource,
    MiniMaxH3LoRATrainingService,
    MiniMaxH3PreparedBatch,
    MiniMaxH3TrainingConfig,
    TrainingAdvancePaused,
    TrainingConfigError,
    minimax_h3_capability_report,
)
from dinkster_training_torch.checkpoint import blake3_digest, canonical_json
from dinkster_training_torch.container import (
    MINIMAX_H3_CONTAINER_DECODER_IDENTITY,
    decode_audio_s16_stereo_32k,
    decode_video_rgb24,
)
from dinkster_training_torch.dataset import (
    MINIMAX_H3_PREPROCESSING_IDENTITY,
    MiniMaxH3DatasetInspection,
    MiniMaxH3DatasetSettings,
    minimax_h3_audio_sample_count,
)

_DIT_IDENTITY = "native:dinkster.minimax_h3:" + "1" * 64
_CONDITIONER_IDENTITY = "native:dinkster.minimax_h3:" + "2" * 64
_VIDEO_IDENTITY = "native:dinkster.minimax_h3:" + "3" * 64
_AUDIO_IDENTITY = "native:dinkster.minimax_h3:" + "4" * 64
_ComponentFactories = tuple[
    Callable[[], MiniMaxH3VideoVaeRuntime],
    Callable[[], MiniMaxH3AudioVaeRuntime],
    Callable[[], MiniMaxH3ConditionerRuntime],
]


def _component(path: Path, identity: str, digit: str) -> dict[str, object]:
    return {
        "path": str(path),
        "digest": "blake3:" + digit * 64,
        "size": 1,
        "identity": identity,
    }


def _write_item(
    root: Path,
    name: str,
    *,
    caption: str,
    frame_value: int,
    resolution: tuple[int, int] = (32, 32),
) -> None:
    from PIL import Image

    item = root / "nested" / name
    frames = item / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    width, height = resolution[1], resolution[0]
    for index in range(5):
        value = frame_value + index
        Image.new("RGB", (width, height), (value, value // 2, 255 - value)).save(
            frames / f"{index:03d}.png"
        )
    samples = minimax_h3_audio_sample_count(5)
    with wave.open(str(item / f"{name}.wav"), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(32_000)
        audio.writeframes((frame_value.to_bytes(2, "little", signed=True) * 2) * samples)
    (item / f"{name}.txt").write_text(caption, encoding="utf-8")


def _write_pcm_wav(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(32_000)
        audio.writeframes(payload)


def _write_container(
    path: Path,
    *,
    resolution: tuple[int, int] = (32, 32),
    frame_count: int = 5,
    include_audio: bool = True,
    audio_sample_rate: int = 32_000,
    audio_channels: int = 2,
    audio_sample_count_32k: int | None = None,
    audio_step_at_32k: int | None = None,
) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = resolution
    rate = Fraction(MINIMAX_H3_CONFIG.video_fps)
    target_audio_samples = (
        minimax_h3_audio_sample_count(5)
        if audio_sample_count_32k is None
        else audio_sample_count_32k
    )
    sample_count = target_audio_samples * audio_sample_rate // 32_000
    audio_layout = "mono" if audio_channels == 1 else "stereo"
    with av.open(str(path), mode="w", format="mp4") as container:
        video = cast(Any, container.add_stream("libx264", rate=rate))
        video.width = width
        video.height = height
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "0"}
        audio = (
            cast(Any, container.add_stream("aac", rate=audio_sample_rate))
            if include_audio
            else None
        )
        if audio is not None:
            audio.layout = audio_layout
        for index in range(frame_count):
            pixels = torch.empty((height, width, 3), dtype=torch.uint8)
            pixels[..., 0].fill_(32 + index * 7)
            pixels[..., 1].copy_(
                torch.arange(width, dtype=torch.uint8).unsqueeze(0).expand(height, -1)
            )
            pixels[..., 2].copy_(
                torch.arange(height, dtype=torch.uint8).unsqueeze(1).expand(-1, width)
            )
            frame = av.VideoFrame.from_ndarray(pixels.numpy(), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(rate.denominator, rate.numerator)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio is not None:
            waveform = torch.empty((audio_channels, sample_count), dtype=torch.float32)
            if audio_step_at_32k is None:
                values = torch.arange(sample_count, dtype=torch.float32).remainder_(97).div_(4096.0)
            else:
                step = audio_step_at_32k * audio_sample_rate // 32_000
                values = torch.zeros(sample_count, dtype=torch.float32)
                values[step:].fill_(0.5)
            waveform[0].copy_(values)
            if audio_channels == 2:
                waveform[1].copy_(-values)
            resampler = av.AudioResampler(
                format=audio.codec_context.format.name,
                layout=audio_layout,
                rate=audio_sample_rate,
            )
            for offset in range(0, sample_count, 1024):
                chunk = waveform[:, offset : offset + 1024].contiguous().numpy()
                frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=audio_layout)
                frame.sample_rate = audio_sample_rate
                frame.pts = offset
                frame.time_base = Fraction(1, audio_sample_rate)
                for converted in resampler.resample(frame):
                    for packet in audio.encode(converted):
                        container.mux(packet)
            for converted in resampler.resample(None):
                for packet in audio.encode(converted):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)


def _write_container_item(
    root: Path,
    name: str = "clip",
    *,
    include_audio: bool = True,
    sidecar_audio: bytes | None = None,
    resolution: tuple[int, int] = (32, 32),
    frame_count: int = 5,
    audio_sample_rate: int = 32_000,
    audio_channels: int = 2,
    audio_sample_count_32k: int | None = None,
) -> Path:
    path = root / "nested" / f"{name}.mp4"
    _write_container(
        path,
        resolution=resolution,
        frame_count=frame_count,
        include_audio=include_audio,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        audio_sample_count_32k=audio_sample_count_32k,
    )
    path.with_suffix(".txt").write_text("container clip", encoding="utf-8")
    if sidecar_audio is not None:
        _write_pcm_wav(path.with_suffix(".wav"), sidecar_audio)
    return path


def _write_folder_from_container(root: Path, container: Path, name: str = "clip") -> None:
    from PIL import Image

    data = container.read_bytes()
    frames = decode_video_rgb24(data, resolution=(32, 32), frame_count=5)
    audio = decode_audio_s16_stereo_32k(
        data,
        sample_count=minimax_h3_audio_sample_count(5),
    )
    assert audio is not None
    item = root / "nested" / name
    frame_root = item / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    for index, payload in enumerate(frames):
        Image.frombytes("RGB", (32, 32), payload).save(frame_root / f"{index:03d}.png")
    _write_pcm_wav(item / f"{name}.wav", audio)
    (item / f"{name}.txt").write_text("container clip", encoding="utf-8")


def _dataset_mapping(
    tmp_path: Path,
    *,
    cache_root: Path | None = None,
    create: bool = True,
) -> dict[str, object]:
    root = tmp_path / "dataset"
    if create:
        _write_item(root, "a", caption="clip zero", frame_value=32)
        _write_item(root, "b", caption="clip one", frame_value=96)
        _write_item(root, "c", caption="clip two", frame_value=160)
    mapping: dict[str, object] = {
        "type": "h3-video-audio-caption-folder",
        "root": str(root),
        "resolution": [32, 32],
        "frameCount": 5,
        "videoVaeState": _component(tmp_path / "video.safetensors", _VIDEO_IDENTITY, "3"),
        "audioVaeState": _component(tmp_path / "audio.safetensors", _AUDIO_IDENTITY, "4"),
        "conditionerState": _component(
            tmp_path / "conditioner.safetensors", _CONDITIONER_IDENTITY, "2"
        ),
    }
    if cache_root is not None:
        mapping["encodedCacheRoot"] = str(cache_root)
    return mapping


def _config_mapping(
    tmp_path: Path,
    *,
    cache_root: Path | None = None,
    create: bool = True,
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "family": "minimax-h3",
        "ditRole": "fl2va-dit",
        "ditIdentity": _DIT_IDENTITY,
        "conditionerIdentity": _CONDITIONER_IDENTITY,
        "dataset": _dataset_mapping(tmp_path, cache_root=cache_root, create=create),
        "device": "cpu",
        "baseDtype": "float32",
        "rank": 2,
        "alpha": 2.0,
        "learningRate": 0.0005,
        "gradientCheckpointing": False,
        "seed": 1234,
        "videoLatentShape": [1, 24, 2, 2, 2],
        "audioLatentShape": [1, 32, 2, 8],
        "conditionerShape": [1, 2, 5120],
    }


def _container_config_mapping(
    tmp_path: Path,
    *,
    cache_root: Path | None = None,
    include_audio: bool = True,
    sidecar_audio: bytes | None = None,
    resolution: tuple[int, int] = (32, 32),
    frame_count: int = 5,
    audio_sample_rate: int = 32_000,
    audio_channels: int = 2,
    audio_sample_count_32k: int | None = None,
) -> tuple[dict[str, object], Path]:
    root = tmp_path / "dataset"
    container = _write_container_item(
        root,
        include_audio=include_audio,
        sidecar_audio=sidecar_audio,
        resolution=resolution,
        frame_count=frame_count,
        audio_sample_rate=audio_sample_rate,
        audio_channels=audio_channels,
        audio_sample_count_32k=audio_sample_count_32k,
    )
    return _config_mapping(tmp_path, cache_root=cache_root, create=False), container


class _TinyVideoVae:
    @staticmethod
    def encode_output_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        return shape[0], 24, 2, shape[3] // 16, shape[4] // 16

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        pooled = functional.avg_pool3d(content.mean(dim=1, keepdim=True), (1, 16, 16))
        pooled = torch.stack((pooled[:, :, 0], pooled[:, :, -1]), dim=2)
        return pooled.expand(-1, 24, -1, -1, -1).contiguous()


class _RecordingVideoVae(_TinyVideoVae):
    def __init__(self, inputs: list[torch.Tensor]) -> None:
        self.inputs = inputs

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        self.inputs.append(content.detach().cpu().clone())
        return super().encode(content)


class _TinyAudioVae:
    @staticmethod
    def encode_output_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
        return shape[0], 32, shape[1], (shape[2] + 799) // 800

    def encode(self, waveform: torch.Tensor, *, sample_rate: int) -> torch.Tensor:
        assert sample_rate == 32_000
        frames = (waveform.shape[-1] + 799) // 800
        padded = functional.pad(waveform, (0, frames * 800 - waveform.shape[-1]))
        pooled = padded.reshape(*waveform.shape[:2], frames, 800).mean(dim=-1)
        return pooled.unsqueeze(1).expand(-1, 32, -1, -1).contiguous()


class _TinyConditioner:
    def encode(self, inputs: MiniMaxH3ConditionerInputs) -> torch.Tensor:
        values = inputs.ids.to(dtype=torch.float32).unsqueeze(-1)
        offsets = torch.arange(5120, device=values.device, dtype=torch.float32) / 8192.0
        return values / 128.0 + offsets


def _factories(
    loads: list[str] | None = None,
    *,
    assert_release: bool = False,
    live_components: dict[str, weakref.ReferenceType[object]] | None = None,
) -> _ComponentFactories:
    events = [] if loads is None else loads
    live = {} if live_components is None else live_components

    def video() -> MiniMaxH3VideoVaeRuntime:
        events.append("video")
        module = _TinyVideoVae()
        live["video"] = weakref.ref(module)
        return MiniMaxH3VideoVaeRuntime(
            module,  # type: ignore[arg-type]
            runtime_identity=_VIDEO_IDENTITY,
            compute_dtype=torch.float32,
        )

    def audio() -> MiniMaxH3AudioVaeRuntime:
        if assert_release:
            assert live["video"]() is None
        events.append("audio")
        module = _TinyAudioVae()
        live["audio"] = weakref.ref(module)
        return MiniMaxH3AudioVaeRuntime(
            module,  # type: ignore[arg-type]
            runtime_identity=_AUDIO_IDENTITY,
            compute_dtype=torch.float32,
        )

    def conditioner() -> MiniMaxH3ConditionerRuntime:
        if assert_release:
            assert live["audio"]() is None
        events.append("conditioner")
        module = _TinyConditioner()
        live["conditioner"] = weakref.ref(module)
        return MiniMaxH3ConditionerRuntime(
            module,  # type: ignore[arg-type]
            runtime_identity=_CONDITIONER_IDENTITY,
        )

    return video, audio, conditioner


def _source(
    config: MiniMaxH3TrainingConfig,
    factories: _ComponentFactories,
) -> MiniMaxH3DatasetSource:
    assert config.dataset is not None
    return MiniMaxH3DatasetSource(
        config.dataset,
        *factories,
        video_latent_shape=config.video_latent_shape,
        audio_latent_shape=config.audio_latent_shape,
        conditioner_shape=config.conditioner_shape,
        device=torch.device(config.device),
    )


def _batch(source: MiniMaxH3DatasetSource, cursor: int) -> MiniMaxH3PreparedBatch:
    return source.batch(
        cursor,
        generator=torch.Generator().manual_seed(99),
        device=torch.device("cpu"),
    )


def _assert_batch_equal(left: MiniMaxH3PreparedBatch, right: MiniMaxH3PreparedBatch) -> None:
    assert torch.equal(left.video_latents, right.video_latents)
    assert torch.equal(left.audio_latents, right.audio_latents)
    assert torch.equal(left.conditioner_embeddings, right.conditioner_embeddings)
    assert left.conditioning.text_token_tags is not None
    assert right.conditioning.text_token_tags is not None
    assert torch.equal(left.conditioning.text_token_tags, right.conditioning.text_token_tags)


def test_importing_container_decoder_does_not_import_pyav() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; assert 'av' not in sys.modules; "
            "import dinkster_training_torch.container; assert 'av' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fresh_sources_are_bit_exact_and_release_components_sequentially(tmp_path: Path) -> None:
    config = MiniMaxH3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    loads: list[str] = []
    first = _source(config, _factories(loads, assert_release=True))
    second = _source(config, _factories(loads, assert_release=True))

    assert loads == ["video", "audio", "conditioner"] * 2
    assert config.dataset is not None
    assert [item.relative_item for item in config.dataset.inspection.items] == [
        "nested/a",
        "nested/b",
        "nested/c",
    ]
    for cursor in (0, 1, 2, 3, 7):
        _assert_batch_equal(_batch(first, cursor), _batch(second, cursor))


def test_video_vae_receives_signed_rgb_pixels(tmp_path: Path) -> None:
    config = MiniMaxH3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    inputs: list[torch.Tensor] = []

    def video() -> MiniMaxH3VideoVaeRuntime:
        return MiniMaxH3VideoVaeRuntime(
            _RecordingVideoVae(inputs),  # type: ignore[arg-type]
            runtime_identity=_VIDEO_IDENTITY,
            compute_dtype=torch.float32,
        )

    _, audio, conditioner = _factories()
    _source(config, (video, audio, conditioner))

    assert len(inputs) == 3
    assert inputs[0].shape == (1, 3, 5, 32, 32)
    for frame_index in range(5):
        value = 32 + frame_index
        expected = (
            torch.tensor([value, value // 2, 255 - value], dtype=torch.float32)
            .div_(255.0)
            .mul_(2.0)
            .sub_(1.0)
        )
        assert torch.equal(inputs[0][0, :, frame_index, 0, 0], expected)
    assert inputs[0].amin().item() >= -1.0
    assert inputs[0].amax().item() <= 1.0


def test_mp4_decodes_to_the_same_tensors_as_its_materialized_frames_and_pcm(
    tmp_path: Path,
) -> None:
    container_mapping, container_path = _container_config_mapping(tmp_path / "container")
    folder_root = tmp_path / "folder" / "dataset"
    _write_folder_from_container(folder_root, container_path)
    folder_mapping = _config_mapping(tmp_path / "folder", create=False)

    container_config = MiniMaxH3TrainingConfig.from_mapping(container_mapping)
    folder_config = MiniMaxH3TrainingConfig.from_mapping(folder_mapping)
    container_source = _source(container_config, _factories())
    folder_source = _source(folder_config, _factories())
    assert container_config.dataset is not None
    assert folder_config.dataset is not None
    container_item = container_config.dataset.inspection.items[0]
    folder_item = folder_config.dataset.inspection.items[0]
    read_container_video = cast(Any, container_source)._read_video
    read_folder_video = cast(Any, folder_source)._read_video
    read_container_audio = cast(Any, container_source)._read_audio
    read_folder_audio = cast(Any, folder_source)._read_audio

    assert torch.equal(
        read_container_video(container_item),
        read_folder_video(folder_item),
    )
    assert torch.equal(
        read_container_audio(container_item),
        read_folder_audio(folder_item),
    )
    _assert_batch_equal(_batch(container_source, 0), _batch(folder_source, 0))
    assert torch.equal(
        read_container_video(container_item),
        read_container_video(container_item),
    )
    assert torch.equal(
        read_container_audio(container_item),
        read_container_audio(container_item),
    )


def test_container_sidecar_wav_takes_precedence_over_embedded_audio(tmp_path: Path) -> None:
    sample_count = minimax_h3_audio_sample_count(5)
    sidecar = ((123).to_bytes(2, "little", signed=True) * 2) * sample_count
    mapping, _container = _container_config_mapping(
        tmp_path,
        sidecar_audio=sidecar,
    )
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert config.dataset is not None
    inspection = config.dataset.inspection
    assert inspection.embedded_audio_items == 0
    assert inspection.sidecar_audio_items == 1
    item = inspection.items[0]
    assert item.audio_kind == "wav"

    source = _source(config, _factories())
    expected = torch.full((1, 2, sample_count), 123 / 32768, dtype=torch.float32)
    assert torch.equal(cast(Any, source)._read_audio(item), expected)


def test_container_audio_resamples_mono_48k_to_stereo_32k(tmp_path: Path) -> None:
    mapping, _container = _container_config_mapping(
        tmp_path,
        audio_sample_rate=48_000,
        audio_channels=1,
    )
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert config.dataset is not None
    assert config.dataset.inspection.errors == ()
    source = _source(config, _factories())
    waveform = cast(Any, source)._read_audio(config.dataset.inspection.items[0])
    assert waveform.shape == (1, 2, minimax_h3_audio_sample_count(5))
    assert torch.equal(waveform[:, 0], waveform[:, 1])


def test_container_audio_starts_at_the_declared_timeline_origin(tmp_path: Path) -> None:
    container = tmp_path / "step.mp4"
    _write_container(container, audio_channels=1, audio_step_at_32k=1_024)
    payload = decode_audio_s16_stereo_32k(
        container.read_bytes(),
        sample_count=minimax_h3_audio_sample_count(5),
    )
    assert payload is not None
    waveform = (
        torch.frombuffer(bytearray(payload), dtype=torch.int16)
        .reshape(-1, 2)
        .transpose(0, 1)
        .to(dtype=torch.float32)
        .div_(32_768)
    )
    assert waveform[0, :512].abs().mean().item() < 0.01
    assert waveform[0, 2_048:3_072].mean().item() > 0.3


def test_service_releases_dataset_components_before_loading_dit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _config_mapping(tmp_path)
    events: list[str] = []
    live: dict[str, weakref.ReferenceType[object]] = {}

    def report(config: MiniMaxH3TrainingConfig) -> dict[str, object]:
        assert config.dataset is not None
        return {"dataset": {"errors": list(config.dataset.inspection.errors)}}

    monkeypatch.setattr(training_service, "minimax_h3_capability_report", report)

    def data_factory(config: MiniMaxH3TrainingConfig) -> MiniMaxH3DatasetSource:
        return _source(
            config,
            _factories(
                events,
                assert_release=True,
                live_components=live,
            ),
        )

    def model_factory(config: MiniMaxH3TrainingConfig) -> torch.nn.Module:
        del config
        assert live["conditioner"]() is None
        events.append("dit")
        return _TinyDiT()

    store = _store(tmp_path / "training.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_factory,
            data_source_factory=data_factory,
        )
        service.create("precompute-order", json.dumps(mapping))
        assert events == ["video", "audio", "conditioner", "dit"]
    finally:
        store.close()


def test_dataset_digest_rotates_for_media_caption_and_component_pins(tmp_path: Path) -> None:
    mapping = _config_mapping(tmp_path)
    first = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert first.dataset is not None
    baseline = first.dataset.digest
    root = Path(first.dataset.root)

    frame = root / "nested/a/frames/000.png"
    original_frame = frame.read_bytes()
    frame.write_bytes(original_frame + b"\x00")
    changed_config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert changed_config.dataset is not None and changed_config.dataset.digest != baseline
    frame.write_bytes(original_frame)

    audio = root / "nested/a/a.wav"
    original_audio = audio.read_bytes()
    audio.write_bytes(original_audio[:-1] + bytes([original_audio[-1] ^ 1]))
    changed_config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert changed_config.dataset is not None and changed_config.dataset.digest != baseline
    audio.write_bytes(original_audio)

    caption = root / "nested/a/a.txt"
    caption.write_text("clip nine", encoding="utf-8")
    changed_config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert changed_config.dataset is not None and changed_config.dataset.digest != baseline
    caption.write_text("clip zero", encoding="utf-8")

    changed = cast(
        "dict[str, object]", cast("dict[str, object]", mapping["dataset"])["videoVaeState"]
    )
    changed["identity"] = "native:dinkster.minimax_h3:" + "9" * 64
    changed_config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert changed_config.dataset is not None and changed_config.dataset.digest != baseline
    cast("dict[str, object]", mapping["dataset"])["digest"] = baseline
    with pytest.raises(TrainingConfigError, match="dataset digest changed"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)


def test_frame_folder_digest_preserves_the_v1_identity(tmp_path: Path) -> None:
    mapping = _config_mapping(tmp_path)
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert config.dataset is not None
    settings = config.dataset
    expected_identity = [
        "dinkster.minimax-h3-video-audio-caption-dataset.v1",
        [
            {
                "item": item.relative_item,
                "frames": [
                    {"path": path, "digest": digest}
                    for path, digest in zip(
                        item.relative_frames,
                        item.frame_digests,
                        strict=True,
                    )
                ],
                "audio": item.relative_audio,
                "audioDigest": item.audio_digest,
                "caption": item.relative_caption,
                "captionDigest": item.caption_digest,
            }
            for item in settings.inspection.items
        ],
        {
            "resolution": list(settings.resolution),
            "frameCount": settings.frame_count,
            "videoFps": MINIMAX_H3_CONFIG.video_fps,
            "audioSampleRate": MINIMAX_H3_CONFIG.audio_sample_rate_hz,
            "preprocessing": MINIMAX_H3_PREPROCESSING_IDENTITY,
            "videoVaeState": settings.video_vae_state.to_mapping(),
            "audioVaeState": settings.audio_vae_state.to_mapping(),
            "conditionerState": settings.conditioner_state.to_mapping(),
        },
    ]
    assert settings.digest == blake3_digest(canonical_json(expected_identity))


def test_container_digest_tracks_bytes_and_decoder_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping, container = _container_config_mapping(tmp_path)
    initial = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert initial.dataset is not None
    baseline = initial.dataset.digest

    original = container.read_bytes()
    container.write_bytes(original + b"\x00")
    changed_bytes = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert changed_bytes.dataset is not None
    assert changed_bytes.dataset.digest != baseline
    container.write_bytes(original)

    monkeypatch.setattr(
        training_dataset,
        "MINIMAX_H3_CONTAINER_DECODER_IDENTITY",
        MINIMAX_H3_CONTAINER_DECODER_IDENTITY + "-changed",
    )
    changed_decoder = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert changed_decoder.dataset is not None
    assert changed_decoder.dataset.digest != baseline

    folder_mapping = _config_mapping(tmp_path / "folder")
    folder_before = MiniMaxH3TrainingConfig.from_mapping(folder_mapping)
    monkeypatch.setattr(
        training_dataset,
        "MINIMAX_H3_CONTAINER_DECODER_IDENTITY",
        MINIMAX_H3_CONTAINER_DECODER_IDENTITY + "-changed-again",
    )
    folder_after = MiniMaxH3TrainingConfig.from_mapping(folder_mapping)
    assert folder_before.dataset is not None and folder_after.dataset is not None
    assert folder_before.dataset.digest == folder_after.dataset.digest


def test_config_refuses_media_changes_between_fast_and_full_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = _config_mapping(tmp_path)
    inspect = training_dataset.inspect_minimax_h3_dataset
    calls = 0

    def changing_inspection(*args: Any, **kwargs: Any) -> MiniMaxH3DatasetInspection:
        nonlocal calls
        calls += 1
        found = inspect(*args, **kwargs)
        return replace(found, digest=found.digest + "changed") if calls == 2 else found

    monkeypatch.setattr(training_config, "inspect_minimax_h3_dataset", changing_inspection)
    with pytest.raises(TrainingConfigError, match="dataset changed during inspection"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)


def test_dataset_source_refuses_a_stale_component_runtime_identity(tmp_path: Path) -> None:
    mapping = _config_mapping(tmp_path)
    identity = "native:dinkster.minimax_h3:" + "9" * 64
    mapping["conditionerIdentity"] = identity
    dataset = cast("dict[str, object]", mapping["dataset"])
    conditioner = cast("dict[str, object]", dataset["conditionerState"])
    conditioner["identity"] = identity
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)

    with pytest.raises(ValueError, match="conditioner runtime identity"):
        _source(config, _factories())


def test_encoded_cache_hit_repair_and_digest_isolation(tmp_path: Path) -> None:
    memory_config = MiniMaxH3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    memory_source = _source(memory_config, _factories())
    expected = [_batch(memory_source, cursor) for cursor in range(5)]

    cached_mapping = _config_mapping(tmp_path, cache_root=tmp_path / "encoded-cache")
    cached_config = MiniMaxH3TrainingConfig.from_mapping(cached_mapping)
    miss_loads: list[str] = []
    miss = _source(cached_config, _factories(miss_loads))
    assert miss_loads == ["video", "audio", "conditioner"]
    for cursor, batch in enumerate(expected):
        _assert_batch_equal(batch, _batch(miss, cursor))

    def forbidden() -> Never:
        raise AssertionError("component loaded on encoded cache hit")

    hit = _source(cached_config, (forbidden, forbidden, forbidden))
    for cursor, batch in enumerate(expected):
        _assert_batch_equal(batch, _batch(hit, cursor))

    # Cache-hit tensors are memory-mapped from the shard; Windows refuses
    # to rewrite a mapped file, so release the source before corrupting it.
    del hit
    gc.collect()
    cache_root = tmp_path / "encoded-cache"
    shard = next((cache_root / "h3-v1/shards").glob("*.safetensors"))
    shard.write_bytes(shard.read_bytes()[:-17])
    repair_loads: list[str] = []
    repaired = _source(cached_config, _factories(repair_loads))
    assert repair_loads == ["video", "audio", "conditioner"]
    for cursor, batch in enumerate(expected):
        _assert_batch_equal(batch, _batch(repaired, cursor))

    root = Path(cast("str", cast("dict[str, object]", cached_mapping["dataset"])["root"]))
    (root / "nested/a/a.txt").write_text("clip nine", encoding="utf-8")
    changed_config = MiniMaxH3TrainingConfig.from_mapping(cached_mapping)
    assert changed_config.dataset is not None and cached_config.dataset is not None
    assert changed_config.dataset.digest != cached_config.dataset.digest
    changed_loads: list[str] = []
    _source(changed_config, _factories(changed_loads))
    assert changed_loads == ["video", "audio", "conditioner"]
    assert len(tuple((cache_root / "h3-v1/entries").glob("*.json"))) == 2


def test_h3_encoded_cache_cuda_keys_ignore_index_but_keep_hardware_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MiniMaxH3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    assert config.dataset is not None
    properties = {
        0: SimpleNamespace(name="same-gpu", major=9, minor=0),
        1: SimpleNamespace(name="same-gpu", major=9, minor=0),
    }
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties.__getitem__)

    first = training_encoded_cache.minimax_h3_encoded_cache_key(
        config.dataset,
        device=torch.device("cuda:0"),
    )
    second = training_encoded_cache.minimax_h3_encoded_cache_key(
        config.dataset,
        device=torch.device("cuda:1"),
    )
    assert first == second

    properties[1] = SimpleNamespace(name="same-gpu", major=8, minor=9)
    different_hardware = training_encoded_cache.minimax_h3_encoded_cache_key(
        config.dataset,
        device=torch.device("cuda:1"),
    )
    assert different_hardware != first


def test_h3_encoded_cache_concurrent_callers_build_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MiniMaxH3TrainingConfig.from_mapping(
        _config_mapping(tmp_path, cache_root=tmp_path / "encoded-cache")
    )
    original_load = training_encoded_cache.MiniMaxH3EncodedDatasetCache.load
    miss_threads: set[int] = set()
    miss_lock = threading.Lock()
    both_callers_missed = threading.Event()

    def observed_load(
        cache: training_encoded_cache.MiniMaxH3EncodedDatasetCache,
    ) -> training_encoded_cache.EncodedMiniMaxH3DatasetTensors | None:
        found = original_load(cache)
        if found is None:
            with miss_lock:
                miss_threads.add(threading.get_ident())
                if len(miss_threads) == 2:
                    both_callers_missed.set()
        return found

    monkeypatch.setattr(
        training_encoded_cache.MiniMaxH3EncodedDatasetCache,
        "load",
        observed_load,
    )
    loaded: list[str] = []
    video, audio, conditioner = _factories(loaded)

    def delayed_video() -> MiniMaxH3VideoVaeRuntime:
        assert both_callers_missed.wait(timeout=10)
        return video()

    def build_source() -> MiniMaxH3DatasetSource:
        return _source(config, (delayed_video, audio, conditioner))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(build_source)
        second_future = executor.submit(build_source)
        first = first_future.result(timeout=30)
        second = second_future.result(timeout=30)

    assert loaded == ["video", "audio", "conditioner"]
    _assert_batch_equal(_batch(first, 0), _batch(second, 0))


def test_container_encoded_cache_hit_skips_components_and_decoders_and_repairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "encoded-cache"
    mapping, container = _container_config_mapping(tmp_path, cache_root=cache_root)
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    miss_loads: list[str] = []
    miss = _source(config, _factories(miss_loads))
    expected = _batch(miss, 0)
    assert miss_loads == ["video", "audio", "conditioner"]

    def forbidden_component() -> Never:
        raise AssertionError("component loaded on encoded container cache hit")

    def forbidden_decoder(*_args: object, **_kwargs: object) -> Never:
        raise AssertionError("decoder loaded on encoded container cache hit")

    with monkeypatch.context() as context:
        context.setattr(training_dataset, "decode_video_rgb24", forbidden_decoder)
        context.setattr(training_dataset, "decode_audio_s16_stereo_32k", forbidden_decoder)
        context.setattr(training_data, "decode_video_rgb24", forbidden_decoder)
        context.setattr(training_data, "decode_audio_s16_stereo_32k", forbidden_decoder)
        hit_config = MiniMaxH3TrainingConfig.from_mapping(mapping)
        hit = _source(
            hit_config,
            (forbidden_component, forbidden_component, forbidden_component),
        )
        _assert_batch_equal(expected, _batch(hit, 0))

    # Cache-hit tensors are memory-mapped from the shard; Windows refuses
    # to rewrite a mapped file, so release the source before corrupting it.
    del hit
    gc.collect()
    shard = next((cache_root / "h3-v1/shards").glob("*.safetensors"))
    shard.write_bytes(shard.read_bytes()[:-17])
    repair_loads: list[str] = []
    repaired = _source(config, _factories(repair_loads))
    assert repair_loads == ["video", "audio", "conditioner"]
    _assert_batch_equal(expected, _batch(repaired, 0))

    container.write_bytes(container.read_bytes() + b"\x00")
    changed = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert changed.dataset is not None and config.dataset is not None
    assert changed.dataset.digest != config.dataset.digest
    changed_loads: list[str] = []
    _source(changed, _factories(changed_loads))
    assert changed_loads == ["video", "audio", "conditioner"]
    assert len(tuple((cache_root / "h3-v1/entries").glob("*.json"))) == 2


class _TinyDiT(torch.nn.Module):
    def __init__(
        self, *, time_embedding_kind: object = "curve", attention_selection: object = None
    ) -> None:
        super().__init__()
        del time_embedding_kind, attention_selection
        self.video = torch.nn.Linear(2, 2)
        self.audio = torch.nn.Linear(8, 8)
        self.context = torch.nn.Linear(5120, 1)
        with torch.no_grad():
            for index, parameter in enumerate(self.parameters(), start=1):
                parameter.fill_(index * 0.001)

    def forward(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        sigma: float,
        context: torch.Tensor,
        *,
        conditioning: MiniMaxH3DiTConditioning,
        sigmas: object,
    ) -> MultiStreamLatent[torch.Tensor]:
        del conditioning, sigmas
        offset = self.context(context).mean() * 0.01 + sigma * 0.001
        return MultiStreamLatent(
            (
                LatentStream("video", self.video(latent.by_role("video")) + offset),
                LatentStream("audio", self.audio(latent.by_role("audio")) + offset),
            )
        )


def _model_factory(config: MiniMaxH3TrainingConfig) -> torch.nn.Module:
    return _TinyDiT().to(dtype=torch.float32 if config.base_dtype == "float32" else torch.bfloat16)


def _store(path: Path) -> TrainingSessionStore:
    return TrainingSessionStore(JournalStore(path))


def _tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key, value in cast("dict[object, object]", left).items():
            _tree_equal(value, cast("dict[object, object]", right)[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for left_item, right_item in zip(
            left, cast("list[object] | tuple[object, ...]", right), strict=True
        ):
            _tree_equal(left_item, right_item)
    else:
        assert left == right


@contextlib.contextmanager
def _deterministic_algorithms() -> Generator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def test_fresh_service_resume_through_dataset_is_bit_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _config_mapping(tmp_path)
    serialized = json.dumps(mapping, sort_keys=True)

    def report(config: MiniMaxH3TrainingConfig) -> dict[str, object]:
        assert config.dataset is not None
        return {"dataset": {"errors": list(config.dataset.inspection.errors)}}

    monkeypatch.setattr(training_service, "minimax_h3_capability_report", report)

    def data_factory(config: MiniMaxH3TrainingConfig) -> MiniMaxH3DatasetSource:
        return _source(config, _factories())

    uninterrupted_store = _store(tmp_path / "uninterrupted.sqlite")
    resumed_store = _store(tmp_path / "resumed.sqlite")
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    try:
        with _deterministic_algorithms():
            uninterrupted = MiniMaxH3LoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=_model_factory,
                data_source_factory=data_factory,
            )
            initial_a, _ = uninterrupted.create("dataset-resume", serialized)
            final_a = uninterrupted.advance(initial_a, 2, "same operation").handle

            armed = False
            paused: MiniMaxH3LoRATrainingService

            def cancelled() -> bool:
                return armed and paused.steps_run >= 1

            paused = MiniMaxH3LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=_model_factory,
                data_source_factory=data_factory,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("dataset-resume", serialized)
            armed = True
            with pytest.raises(TrainingAdvancePaused):
                paused.advance(initial_b, 2, "same operation")
            successor = MiniMaxH3LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=_model_factory,
                data_source_factory=data_factory,
            )
            final_b = successor.advance(initial_b, 2, "same operation").handle

        left = ContentAddressedCheckpointStore(uninterrupted_root).load(
            final_a.checkpoint_manifest_digest
        )
        right = ContentAddressedCheckpointStore(resumed_root).load(
            final_b.checkpoint_manifest_digest
        )
        assert left.loss == right.loss and left.data_cursor == right.data_cursor == 2
        _tree_equal(left.adapter, right.adapter)
        _tree_equal(left.optimizer, right.optimizer)
        _tree_equal(left.rng, right.rng)
    finally:
        uninterrupted_store.close()
        resumed_store.close()


@pytest.mark.parametrize(
    "damage, error",
    [
        ("missing-frames", "missing frames directory"),
        ("frame-count", "frames directory has 6 images; expected 5"),
        ("ambiguous-frame-order", "equal-width consecutive decimal stems"),
        ("bad-audio", "cannot decode PCM WAV audio"),
        ("missing-caption", "missing caption"),
        ("resolution", "frame resolution is 16x16; expected 32x32"),
        ("configured-frame-count", "frames directory has 5 images; expected 22"),
    ],
)
def test_service_create_refuses_invalid_h3_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    error: str,
) -> None:
    from PIL import Image

    mapping = _config_mapping(tmp_path)
    dataset = cast("dict[str, object]", mapping["dataset"])
    root = Path(cast("str", dataset["root"]))
    item = root / "nested/a"
    if damage == "missing-frames":
        shutil.rmtree(item / "frames")
    elif damage == "frame-count":
        shutil.copyfile(item / "frames/000.png", item / "frames/005.png")
    elif damage == "ambiguous-frame-order":
        (item / "frames/000.png").rename(item / "frames/0.png")
    elif damage == "bad-audio":
        (item / "a.wav").write_bytes(b"not a wave file")
    elif damage == "missing-caption":
        (item / "a.txt").unlink()
    elif damage == "resolution":
        Image.new("RGB", (16, 16)).save(item / "frames/000.png")
    else:
        dataset["frameCount"] = 22
        mapping["videoLatentShape"] = [1, 24, 7, 2, 2]
        mapping["audioLatentShape"] = [1, 32, 2, 37]

    def report(config: MiniMaxH3TrainingConfig) -> dict[str, object]:
        assert config.dataset is not None
        return {"dataset": {"errors": list(config.dataset.inspection.errors)}}

    monkeypatch.setattr(training_service, "minimax_h3_capability_report", report)

    def model_not_loaded(config: MiniMaxH3TrainingConfig) -> torch.nn.Module:
        del config
        raise AssertionError("model loaded for invalid dataset")

    def data_not_loaded(config: MiniMaxH3TrainingConfig) -> MiniMaxH3DatasetSource:
        del config
        raise AssertionError("data source loaded for invalid dataset")

    store = _store(tmp_path / "training.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_not_loaded,
            data_source_factory=data_not_loaded,
        )
        with pytest.raises(ValueError, match=error):
            service.create("invalid-dataset", json.dumps(mapping))
    finally:
        store.close()


@pytest.mark.parametrize(
    "damage, error",
    [
        ("corrupt", "cannot decode container video"),
        ("missing-audio", "no audio stream and no same-basename sidecar WAV"),
        ("audio-length", "container audio duration is 3200 samples at 32000 Hz; expected 6400"),
        ("resolution", "expected uint8/\\(32, 32, 3\\)"),
        ("frame-count", "container video has 4 frames; expected 5"),
        ("ambiguous", "both a frame folder and a container video"),
    ],
)
def test_service_create_refuses_invalid_container_dataset_before_loading_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
    error: str,
) -> None:
    if damage == "missing-audio":
        mapping, container = _container_config_mapping(tmp_path, include_audio=False)
    elif damage == "audio-length":
        mapping, container = _container_config_mapping(tmp_path, audio_sample_count_32k=3_200)
    elif damage == "resolution":
        mapping, container = _container_config_mapping(tmp_path, resolution=(16, 16))
    elif damage == "frame-count":
        mapping, container = _container_config_mapping(tmp_path, frame_count=4)
    else:
        mapping, container = _container_config_mapping(tmp_path)
    if damage == "corrupt":
        container.write_bytes(b"not a media container")
    elif damage == "ambiguous":
        _write_item(
            tmp_path / "dataset",
            "clip",
            caption="ambiguous clip",
            frame_value=32,
        )

    def report(config: MiniMaxH3TrainingConfig) -> dict[str, object]:
        assert config.dataset is not None
        return {"dataset": {"errors": list(config.dataset.inspection.errors)}}

    monkeypatch.setattr(training_service, "minimax_h3_capability_report", report)

    def model_not_loaded(config: MiniMaxH3TrainingConfig) -> torch.nn.Module:
        del config
        raise AssertionError("model loaded for invalid container dataset")

    def data_not_loaded(config: MiniMaxH3TrainingConfig) -> MiniMaxH3DatasetSource:
        del config
        raise AssertionError("data source loaded for invalid container dataset")

    store = _store(tmp_path / "training.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_not_loaded,
            data_source_factory=data_not_loaded,
        )
        with pytest.raises(ValueError, match=error):
            service.create("invalid-container-dataset", json.dumps(mapping))
    finally:
        store.close()


@pytest.mark.parametrize(
    "field, value, error",
    [
        ("frameCount", 6, "frameCount must equal 17k\\+5"),
        ("resolution", [16, 16], "resolution entries must be divisible by 32"),
    ],
)
def test_config_refuses_non_native_h3_dataset_geometry(
    tmp_path: Path,
    field: str,
    value: object,
    error: str,
) -> None:
    mapping = _config_mapping(tmp_path)
    dataset = cast("dict[str, object]", mapping["dataset"])
    dataset[field] = value
    with pytest.raises(TrainingConfigError, match=error):
        MiniMaxH3TrainingConfig.from_mapping(mapping)


def test_capability_report_dry_run_and_runtime_v2_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapping = _config_mapping(tmp_path, cache_root=tmp_path / "cache")
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert config.dataset is not None
    without_cache = MiniMaxH3TrainingConfig.from_mapping(_config_mapping(tmp_path))
    assert config.identity_mapping() == without_cache.identity_mapping()

    monkeypatch.setattr(training_service, "assemble_minimax_h3_dit", _TinyDiT)

    def encoder_memory(
        settings: MiniMaxH3DatasetSettings,
    ) -> tuple[dict[str, int], tuple[str, ...]]:
        del settings
        return (
            {
                "videoVaeParametersTransientLowerBound": 11,
                "audioVaeParametersTransientLowerBound": 12,
                "conditionerParametersTransientLowerBound": 13,
            },
            (),
        )

    monkeypatch.setattr(
        training_service,
        "inspect_minimax_h3_encoder_memory",
        encoder_memory,
    )
    report = minimax_h3_capability_report(config)
    assert report["trainer"] == MINIMAX_H3_TRAINING_RUNTIME_IDENTITY
    assert MINIMAX_H3_TRAINING_RUNTIME_IDENTITY == "minimax-h3-lora-torch/2"
    assert report["sessionExtensionSnapshotDigest"] == MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST
    dataset = cast("dict[str, object]", report["dataset"])
    assert dataset["itemCount"] == 3
    assert dataset["frameCount"] == 5
    assert dataset["encodedCache"] == {"state": "miss"}
    assert dataset["errors"] == []
    capabilities = cast("dict[str, object]", report["capabilities"])
    assert capabilities["videoAudioCaptionDataset"] is True
    assert capabilities["containerVideoDataset"] is True
    assert capabilities["encodedDatasetDiskCache"] is True
    categories = cast(
        "dict[str, int]", cast("dict[str, object]", report["memoryLedger"])["categories"]
    )
    assert categories["encodedDatasetStoreResidentCpu"] > 0
    precision = cast("dict[str, str]", report["precisionPlan"])
    assert precision["datasetPrecomputeComponents"].startswith("video VAE")

    container_mapping, _container = _container_config_mapping(tmp_path / "container")
    container_config = MiniMaxH3TrainingConfig.from_mapping(container_mapping)
    container_report = minimax_h3_capability_report(container_config)
    container_dataset = cast("dict[str, object]", container_report["dataset"])
    assert container_dataset["itemSources"] == {"frameFolders": 0, "containers": 1}
    assert container_dataset["containerDecoder"] == MINIMAX_H3_CONTAINER_DECODER_IDENTITY
    assert container_dataset["errors"] == []
    container_audio = cast("dict[str, int]", container_dataset["audio"])
    assert container_audio["embeddedContainerItems"] == 1
    assert container_audio["sidecarWavItems"] == 0

    store = _store(tmp_path / "training.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(store, tmp_path / "checkpoints")
        assert service.dry_run(json.dumps(mapping)) == report
    finally:
        store.close()

    current_mapping = config.to_mapping()
    legacy_digest = blake3_digest(
        canonical_json(["dinkster.training.config.v1", "minimax-h3-lora-torch/1", current_mapping])
    )
    current_digest = cast("str", report["configDigest"])
    assert legacy_digest != current_digest
    assert MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST != blake3_digest(
        b"dinkster.minimax-h3-lora-torch.extension-snapshot.v1"
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
    ],
)
def test_dataset_precompute_and_batches_run_on_configured_device(
    tmp_path: Path, device: str
) -> None:
    mapping = _config_mapping(tmp_path)
    mapping["device"] = device
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    source = _source(config, _factories())
    batch = source.batch(
        0,
        generator=torch.Generator(device=device).manual_seed(1),
        device=torch.device(device),
    )
    assert batch.video_latents.device.type == device
    assert batch.audio_latents.device.type == device
    assert batch.conditioner_embeddings.device.type == device
    assert batch.conditioning.text_token_tags is not None
    assert batch.conditioning.text_token_tags.device.type == device


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
    ],
)
def test_container_precompute_and_batches_run_on_configured_device(
    tmp_path: Path,
    device: str,
) -> None:
    mapping, _container = _container_config_mapping(tmp_path)
    mapping["device"] = device
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    source = _source(config, _factories())
    batch = source.batch(
        0,
        generator=torch.Generator(device=device).manual_seed(1),
        device=torch.device(device),
    )
    assert batch.video_latents.device.type == device
    assert batch.audio_latents.device.type == device
    assert batch.conditioner_embeddings.device.type == device
    assert batch.conditioning.text_token_tags is not None
    assert batch.conditioning.text_token_tags.device.type == device
