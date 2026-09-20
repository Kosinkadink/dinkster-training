"""Deterministic PyAV decoding for H3 training containers."""
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import io
from fractions import Fraction
from importlib import import_module
from typing import Any

PYAV_VERSION = "16.0.1"
MINIMAX_H3_CONTAINER_DECODER_IDENTITY = "minimax-h3-pyav-16.0.1-rgb24-s16-stereo-32000-v1"


class ContainerDecodeError(ValueError):
    """A container violates the deterministic H3 decode profile."""


def _av() -> Any:
    try:
        module = import_module("av")
    except ImportError as exc:
        raise RuntimeError(
            "container video datasets require av==16.0.1 in the training worker environment"
        ) from exc
    if getattr(module, "__version__", None) != PYAV_VERSION:
        raise RuntimeError(
            f"container video datasets require av=={PYAV_VERSION}, got "
            f"{getattr(module, '__version__', 'unknown')}"
        )
    return module


def decode_video_rgb24(
    data: bytes,
    *,
    resolution: tuple[int, int],
    frame_count: int,
    collect: bool = True,
) -> tuple[bytes, ...]:
    """Decode one strictly timed video stream to ordered RGB24 frames."""
    av = _av()
    height, width = resolution
    frames: list[bytes] = []
    decoded_frames = 0
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1:
                raise ContainerDecodeError(
                    f"container must have exactly one video stream; found {len(streams)}"
                )
            stream = streams[0]
            last_timestamp: Fraction | None = None
            for decoded in container.decode(stream):
                decoded_frames += 1
                frame = decoded
                time_base = frame.time_base or stream.time_base
                if frame.pts is None or time_base is None:
                    raise ContainerDecodeError("container video frame has no usable timestamp")
                timestamp = Fraction(frame.pts) * Fraction(time_base)
                if last_timestamp is not None and timestamp <= last_timestamp:
                    raise ContainerDecodeError(
                        "container video frame timestamps must be strictly increasing"
                    )
                last_timestamp = timestamp
                converted = frame.reformat(format="rgb24")
                array = converted.to_ndarray()
                expected_shape = (height, width, 3)
                if array.dtype.name != "uint8" or tuple(array.shape) != expected_shape:
                    raise ContainerDecodeError(
                        f"decoded video frame has dtype/shape {array.dtype}/{array.shape}; "
                        f"expected uint8/{expected_shape}"
                    )
                if collect:
                    frames.append(array.tobytes(order="C"))
    except ContainerDecodeError:
        raise
    except Exception as exc:
        raise ContainerDecodeError(f"cannot decode container video: {exc}") from exc
    if decoded_frames != frame_count:
        raise ContainerDecodeError(
            f"container video has {decoded_frames} frames; expected {frame_count}"
        )
    return tuple(frames)


def decode_audio_s16_stereo_32k(
    data: bytes, *, sample_count: int, collect: bool = True
) -> bytes | None:
    """Decode one audio stream to interleaved stereo 32 kHz signed PCM."""
    av = _av()
    pieces: list[bytes] = []
    decoded_samples = 0
    declared_samples: int | None = None
    source_end: Fraction | None = None
    try:
        with av.open(io.BytesIO(data), mode="r") as container:
            streams = tuple(container.streams.audio)
            if not streams:
                return None
            if len(streams) != 1:
                raise ContainerDecodeError(
                    f"container must have at most one audio stream; found {len(streams)}"
                )
            stream = streams[0]
            if stream.start_time not in (None, 0):
                raise ContainerDecodeError("container audio stream must start at timestamp zero")
            if stream.duration is not None and stream.time_base is not None:
                declared_samples = round(
                    Fraction(stream.duration) * Fraction(stream.time_base) * 32_000
                )
                if declared_samples != sample_count:
                    raise ContainerDecodeError(
                        f"container audio duration is {declared_samples} samples at 32000 Hz; "
                        f"expected {sample_count}"
                    )
            resampler = av.AudioResampler(format="s16", layout="stereo", rate=32_000)

            def append(frame: Any) -> None:
                nonlocal decoded_samples
                if (
                    frame.format.name != "s16"
                    or frame.layout.name != "stereo"
                    or frame.sample_rate != 32_000
                ):
                    raise ContainerDecodeError(
                        "PyAV audio resampler returned an unexpected PCM layout"
                    )
                time_base = frame.time_base
                if frame.pts is None or time_base is None:
                    raise ContainerDecodeError("resampled container audio has no usable timestamp")
                timestamp = Fraction(frame.pts) * Fraction(time_base)
                expected_timestamp = Fraction(decoded_samples, 32_000)
                if timestamp != expected_timestamp:
                    raise ContainerDecodeError(
                        "resampled container audio timestamps must be contiguous from zero"
                    )
                array = frame.to_ndarray()
                expected_values = int(frame.samples) * 2
                if array.dtype.name != "int16" or int(array.size) != expected_values:
                    raise ContainerDecodeError(
                        f"decoded audio frame has dtype/value count {array.dtype}/{array.size}; "
                        f"expected int16/{expected_values}"
                    )
                if collect:
                    pieces.append(array.astype("<i2", copy=False).tobytes(order="C"))
                decoded_samples += int(frame.samples)

            for decoded in container.decode(stream):
                if decoded.sample_rate is None or decoded.sample_rate <= 0:
                    raise ContainerDecodeError("container audio frame has no usable sample rate")
                time_base = decoded.time_base or stream.time_base
                if decoded.pts is None or time_base is None:
                    raise ContainerDecodeError("container audio frame has no usable timestamp")
                timestamp = Fraction(decoded.pts) * Fraction(time_base)
                if source_end is None:
                    if timestamp != 0:
                        raise ContainerDecodeError(
                            "container audio frame timestamps must start at zero"
                        )
                elif timestamp != source_end:
                    raise ContainerDecodeError(
                        "container audio frame timestamps must be contiguous"
                    )
                source_end = timestamp + Fraction(int(decoded.samples), int(decoded.sample_rate))
                for converted in resampler.resample(decoded):
                    append(converted)
            for converted in resampler.resample(None):
                append(converted)
    except ContainerDecodeError:
        raise
    except Exception as exc:
        raise ContainerDecodeError(f"cannot decode container audio: {exc}") from exc

    if declared_samples is None and decoded_samples != sample_count:
        raise ContainerDecodeError(
            f"container audio decodes to {decoded_samples} samples at 32000 Hz; "
            f"expected {sample_count}"
        )
    if decoded_samples < sample_count:
        raise ContainerDecodeError(
            f"container audio decodes to {decoded_samples} samples at 32000 Hz; "
            f"expected at least {sample_count}"
        )
    return b"".join(pieces)[: sample_count * 2 * 2] if collect else b""
