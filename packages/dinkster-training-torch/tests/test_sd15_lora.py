"""Executable proofs for the native SD1.5 LoRA trainer."""

from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import json
import math
import struct
import threading
import weakref
import zlib
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import cast

import dinkster_assets.integrity as asset_integrity
import dinkster_training_torch.data as training_data
import dinkster_training_torch.encoded_cache as training_encoded_cache
import pytest
import torch
from dinkster_assets import AssetIntegrityError, clear_verified_cache, digest_bytes
from dinkster_inference import (
    CLIP_L_PROFILE,
    SD15,
    ClipTextConfig,
    ClipTextDetectError,
    KLConfig,
    PromptTokenizer,
    TensorGeometry,
    decode_lora,
    load_clip_bpe,
    load_safetensors_header,
    native_unet_key_map,
    pack_spans,
)
from dinkster_inference_torch import (
    AutoencoderKL,
    ClipTextEncoder,
    ClipTextModel,
    UNetModel,
    build_patch_set,
    load_tensors,
    patch_weights,
    process_input,
)
from dinkster_protocol import TrainingEventName, TrainingJournalEvent
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    TRAINING_RUNTIME_IDENTITY,
    TRAINING_SNAPSHOT_DIGEST,
    CheckpointError,
    CheckpointState,
    ContentAddressedCheckpointStore,
    ImageCaptionDatasetSource,
    LoraExportSettings,
    PreparedBatch,
    SD15LoRATrainer,
    SD15LoRATrainingService,
    TrainableAttachment,
    TrainingAdvancePaused,
    TrainingConfig,
    TrainingConfigError,
    capability_report,
    default_model_factory,
)
from dinkster_training_torch.attachment import resolve_lora_targets
from PIL import Image

_MASK32 = 0xFFFFFFFF
_UNET: dict[str, object] = {
    "inChannels": 4,
    "outChannels": 4,
    "modelChannels": 32,
    "numResBlocks": [1],
    "channelMult": [1],
    "transformerDepth": [1],
    "transformerDepthOutput": [1, 1],
    "transformerDepthMiddle": 1,
    "contextDim": 16,
    "useLinearInTransformer": False,
    "numHeads": 8,
}
_TINY_KL = KLConfig(
    in_channels=3,
    out_channels=3,
    ch=32,
    decoder_ch=32,
    ch_mult=(1, 2, 4, 4),
    num_res_blocks=1,
    z_channels=4,
    embed_dim=4,
)
_TINY_CLIP = ClipTextConfig(
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    intermediate_size=32,
    hidden_act="quick_gelu",
)


def config_mapping(
    *,
    device: str = "cpu",
    base_dtype: str = "float32",
    optimizer: str = "factored-adamw",
    accumulation: int = 1,
    checkpointing: bool = True,
    checkpointing_mode: str | None = None,
) -> dict[str, object]:
    mapping: dict[str, object] = {
        "schemaVersion": 1,
        "family": "sd15",
        "unet": _UNET,
        "device": device,
        "baseDtype": base_dtype,
        "rank": 2,
        "alpha": 2.0,
        "learningRate": 0.0005,
        "weightDecay": 0.01,
        "optimizer": optimizer,
        "gradientAccumulationSteps": accumulation,
        "gradientCheckpointing": checkpointing,
        "seed": 1234,
        "latentShape": [1, 4, 4, 4],
        "contextShape": [1, 2, 16],
    }
    if checkpointing_mode is not None:
        mapping["checkpointingMode"] = checkpointing_mode
    return mapping


def serialized_config(**overrides: object) -> str:
    values = config_mapping()
    values.update(overrides)
    return json.dumps(values, sort_keys=True)


def _hash_uniform(seed: int, count: int) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.int64) + (seed & _MASK32)
    values = (values * 1664525 + 1013904223) & _MASK32
    values = values ^ (values >> 13)
    values = (values * 214013 + 2531011) & _MASK32
    values = values ^ (values >> 17)
    values = (values * 69069 + 1) & _MASK32
    values = values ^ (values >> 5)
    return (values.to(torch.float64) / float(_MASK32 + 1)).to(torch.float32)


def _fill_value(key: str, shape: tuple[int, ...]) -> torch.Tensor:
    count = math.prod(shape) if shape else 1
    uniform = _hash_uniform(zlib.crc32(key.encode("ascii")), count)
    if key.endswith(".weight") and len(shape) == 1:
        values = 1.0 + (uniform - 0.5) * 0.1
    else:
        values = (uniform - 0.5) * 0.04
    return values.reshape(shape)


def model_factory(config: TrainingConfig) -> torch.nn.Module:
    model = UNetModel(config.unet)
    state = {
        key: _fill_value(key, tuple(value.shape)).to(dtype=value.dtype)
        for key, value in model.state_dict().items()
    }
    model.load_state_dict(state, strict=True)
    return model


def _filled(module: torch.nn.Module) -> torch.nn.Module:
    state = {
        key: _fill_value(key, tuple(value.shape)).to(dtype=value.dtype)
        for key, value in module.state_dict().items()
    }
    module.load_state_dict(state, strict=True)
    return module


def tiny_vae() -> AutoencoderKL:
    return cast("AutoencoderKL", _filled(AutoencoderKL(_TINY_KL)))


def tiny_clip() -> ClipTextModel:
    return cast("ClipTextModel", _filled(ClipTextModel(_TINY_CLIP)))


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().contiguous()
    return bytes(contiguous.untyped_storage())[: contiguous.numel() * contiguous.element_size()]


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    header: dict[str, object] = {}
    payload = bytearray()
    for key, tensor in tensors.items():
        data = _tensor_bytes(tensor)
        header[key] = {
            "dtype": {torch.float32: "F32", torch.int64: "I64"}[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    raw = json.dumps(header, separators=(",", ":")).encode("ascii")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(payload))
    return path


def file_digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def image_dataset_mapping(
    tmp_path: Path,
    *,
    invalid_captions: bool = False,
    clip_extras: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    root = tmp_path / "dataset"
    root.mkdir()
    entries = (
        ("zebra.png", (12, 8), (240, 20, 30), "striped animal"),
        ("nested/ant.png", (8, 12), (10, 180, 60), "small ant"),
        ("middle.png", (10, 10), (40, 50, 220), "blue square"),
    )
    for index, (relative, size, color, caption) in enumerate(entries):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path)
        if invalid_captions and index == 1:
            continue
        text = "" if invalid_captions and index == 2 else caption
        path.with_suffix(".txt").write_text(text, encoding="utf-8")

    vae_path = write_safetensors(
        tmp_path / "vae.safetensors",
        {key: value for key, value in tiny_vae().state_dict().items()},
    )
    clip_tensors = {
        key: value
        for key, value in tiny_clip().state_dict().items()
        if key != "text_projection.weight"
    }
    clip_tensors.update(clip_extras or {})
    clip_path = write_safetensors(tmp_path / "clip.safetensors", clip_tensors)
    return {
        "type": "image-caption-folder",
        "root": str(root),
        "resolution": [16, 16],
        "vaeState": {"path": str(vae_path), "digest": file_digest(vae_path), "prefix": ""},
        "textEncoderState": {
            "path": str(clip_path),
            "digest": file_digest(clip_path),
            "prefix": "",
        },
    }


def image_dataset_config(
    tmp_path: Path,
    *,
    device: str = "cpu",
    batch_size: int = 1,
    invalid_captions: bool = False,
    clip_extras: Mapping[str, torch.Tensor] | None = None,
) -> TrainingConfig:
    mapping = config_mapping(device=device)
    mapping["latentShape"] = [batch_size, 4, 2, 2]
    mapping["contextShape"] = [batch_size, 77, 16]
    mapping["dataset"] = image_dataset_mapping(
        tmp_path,
        invalid_captions=invalid_captions,
        clip_extras=clip_extras,
    )
    return TrainingConfig.from_mapping(mapping)


def image_data_source_factory(config: TrainingConfig) -> ImageCaptionDatasetSource:
    assert config.dataset is not None
    return ImageCaptionDatasetSource(
        config.dataset,
        tiny_vae,
        tiny_clip,
        batch_size=config.latent_shape[0],
        device=torch.device(config.device),
    )


def tracked_image_data_source_factory(
    loaded: list[str],
) -> Callable[[TrainingConfig], ImageCaptionDatasetSource]:
    def factory(config: TrainingConfig) -> ImageCaptionDatasetSource:
        assert config.dataset is not None

        def vae_factory() -> AutoencoderKL:
            loaded.append("vae")
            return tiny_vae()

        def clip_factory() -> ClipTextModel:
            loaded.append("clip")
            return tiny_clip()

        return ImageCaptionDatasetSource(
            config.dataset,
            vae_factory,
            clip_factory,
            batch_size=config.latent_shape[0],
            device=torch.device(config.device),
        )

    return factory


def encoded_cache_shard(cache_root: Path) -> Path:
    manifest_path = next((cache_root / "v1" / "entries").glob("*.json"))
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="ascii")))
    digest = cast("str", manifest["shardDigest"])
    return cache_root / "v1" / "shards" / f"{digest[7:]}.safetensors"


def assert_prepared_batch_equal(left: PreparedBatch, right: PreparedBatch) -> None:
    for name in (
        "latents",
        "text_embeddings",
        "pooled_embeddings",
        "time_ids",
        "timesteps",
        "noise",
    ):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if left_value is None:
            assert right_value is None
        else:
            assert isinstance(right_value, torch.Tensor)
            assert torch.equal(left_value, right_value)


class SyntheticBatches:
    """Prepared batches keyed only by the absolute cursor."""

    def __init__(self, config: TrainingConfig) -> None:
        self._latent_shape = config.latent_shape
        self._context_shape = config.context_shape

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        del generator, device
        return PreparedBatch(
            latents=_fill_value(f"latent:{cursor}", self._latent_shape),
            text_embeddings=_fill_value(f"text:{cursor}", self._context_shape),
        )


class FixedBatches:
    def __init__(self, batch: PreparedBatch) -> None:
        self._batch = batch

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        del cursor, generator, device
        return self._batch


def data_source_factory(config: TrainingConfig) -> SyntheticBatches:
    return SyntheticBatches(config)


def make_store(path: Path) -> TrainingSessionStore:
    return TrainingSessionStore(JournalStore(path))


def checkpoint_state(root: Path, digest: str) -> CheckpointState:
    return ContentAddressedCheckpointStore(root).load(digest)


def recovery_losses(
    root: Path,
    store: TrainingSessionStore,
    session_id: str,
) -> tuple[float, ...]:
    checkpoints = ContentAddressedCheckpointStore(root)
    losses: list[float] = []
    for record in store.read_events(session_id, after=0, limit=100).records:
        if record.name != TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED:
            continue
        event = TrainingJournalEvent.from_wire(record.payload)
        digest = event.data["manifestDigest"]
        assert isinstance(digest, str)
        loss = checkpoints.load(digest).loss
        assert loss is not None
        losses.append(loss)
    return tuple(losses)


def recovery_manifest_digests(
    store: TrainingSessionStore,
    session_id: str,
) -> tuple[str, ...]:
    digests: list[str] = []
    for record in store.read_events(session_id, after=0, limit=100).records:
        if record.name != TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED:
            continue
        event = TrainingJournalEvent.from_wire(record.payload)
        digest = event.data["manifestDigest"]
        assert isinstance(digest, str)
        digests.append(digest)
    return tuple(digests)


def manifest_digests(root: Path) -> set[str]:
    return {f"blake3:{path.stem}" for path in (root / "manifests").glob("*.json")}


def manifest_step_cursors(root: Path) -> list[int]:
    cursors: list[int] = []
    for path in (root / "manifests").glob("*.json"):
        manifest = cast("dict[str, object]", json.loads(path.read_text("ascii")))
        cursor = manifest["stepCursor"]
        assert isinstance(cursor, int)
        cursors.append(cursor)
    return sorted(cursors)


def manifest_shards(root: Path, digest: str) -> dict[str, str]:
    path = root / "manifests" / f"{digest[7:]}.json"
    manifest = cast("dict[str, object]", json.loads(path.read_text("ascii")))
    return cast("dict[str, str]", manifest["shards"])


def assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        left_dict = cast("dict[object, object]", left)
        right_dict = cast("dict[object, object]", right)
        assert left_dict.keys() == right_dict.keys()
        for key in left_dict:
            assert_tree_equal(left_dict[key], right_dict[key])
        return
    if isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        right_items = cast("list[object] | tuple[object, ...]", right)
        assert len(left) == len(right_items)
        for left_item, right_item in zip(left, right_items, strict=True):
            assert_tree_equal(left_item, right_item)
        return
    assert left == right


def tree_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for item in cast("dict[object, object]", value).values()
            for tensor in tree_tensors(item)
        ]
    if isinstance(value, (list, tuple)):
        return [
            tensor
            for item in cast("list[object] | tuple[object, ...]", value)
            for tensor in tree_tensors(item)
        ]
    return []


@contextlib.contextmanager
def deterministic_algorithms() -> Generator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def test_capability_report_resolves_targets_precision_and_memory() -> None:
    factored = capability_report(TrainingConfig.from_mapping(config_mapping()))
    adamw = capability_report(TrainingConfig.from_mapping(config_mapping(optimizer="adamw")))

    targets = cast("list[dict[str, object]]", factored["targets"])
    assert {target["operation"] for target in targets} == {"conv2d", "linear"}
    assert any("attn" in str(target["modulePath"]) for target in targets)
    assert any("ff.net" in str(target["modulePath"]) for target in targets)
    counts = cast("dict[str, int]", factored["parameterCounts"])
    assert counts["frozenBase"] > counts["targetedBase"] > counts["trainableLora"] > 0
    assert factored["precisionPlan"] == {
        "baseStorage": "float32",
        "forwardAutocast": "bfloat16",
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
    }
    factored_memory = cast("dict[str, object]", factored["memoryLedger"])
    adamw_memory = cast("dict[str, object]", adamw["memoryLedger"])
    categories = cast("dict[str, int]", factored_memory["categories"])
    assert set(categories) == {
        "frozenBaseParameters",
        "loraMasterParameters",
        "loraGradients",
        "optimizerState",
        "activationsLowerBound",
    }
    assert factored_memory["estimated"] is True
    assert cast("int", factored_memory["totalLowerBoundBytes"]) < cast(
        "int", adamw_memory["totalLowerBoundBytes"]
    )
    assert "excludes intermediate activations" in cast(
        "str", factored_memory["activationEstimateBasis"]
    )


@pytest.mark.parametrize("shape", [(2048, 1024), (320, 320, 3, 3)])
def test_lora_down_initialization_matches_reference_kaiming_uniform(
    shape: tuple[int, ...],
) -> None:
    rank = 64

    def initialized() -> tuple[torch.Tensor, torch.Tensor]:
        model = torch.nn.Module()
        if len(shape) == 2:
            model.add_module(
                "transformer_blocks",
                torch.nn.ModuleList([torch.nn.Linear(shape[1], shape[0], bias=False)]),
            )
        else:
            model.add_module(
                "proj_in",
                torch.nn.Conv2d(shape[1], shape[0], kernel_size=(shape[2], shape[3]), bias=False),
            )
        attached = TrainableAttachment.attach(model, rank=rank, alpha=rank, seed=1234)
        state = attached.state_dict()
        down = next(value for key, value in state.items() if key.endswith(".down"))
        up = next(value for key, value in state.items() if key.endswith(".up"))
        return down, up

    down, up = initialized()
    repeated, _ = initialized()
    fan_in = math.prod(shape[1:])
    bound = 1.0 / math.sqrt(fan_in)
    expected_rms = bound / math.sqrt(3.0)

    assert torch.equal(down, repeated)
    assert torch.count_nonzero(up) == 0
    assert down.amin().item() >= -bound
    assert down.amax().item() <= bound
    assert math.isclose(down.square().mean().sqrt().item(), expected_rms, rel_tol=0.01)


@pytest.mark.parametrize("operation", ["linear", "conv2d", "conv2d_same_reflect"])
def test_lora_low_rank_forward_matches_full_effective_weight(operation: str) -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260822)
    model = torch.nn.Module()
    if operation == "linear":
        target: torch.nn.Linear | torch.nn.Conv2d = torch.nn.Linear(24, 32, bias=True)
        model.add_module("transformer_blocks", torch.nn.ModuleList([target]))
        inputs = torch.randn((2, 5, 24), generator=generator)
    else:
        same_reflect = operation == "conv2d_same_reflect"
        target = torch.nn.Conv2d(
            6,
            8,
            kernel_size=3,
            stride=1 if same_reflect else 2,
            padding="same" if same_reflect else 2,
            dilation=2,
            groups=2,
            bias=True,
            padding_mode="reflect" if same_reflect else "zeros",
        )
        model.add_module("proj_in", target)
        inputs = torch.randn((2, 6, 13, 15), generator=generator)
    base_weight = target.weight.detach().clone()
    base_bias = target.bias.detach().clone() if target.bias is not None else None
    attachment = TrainableAttachment.attach(model, rank=4, alpha=3.0, seed=1234)
    state = attachment.state_dict()
    down_name = next(name for name in state if name.endswith(".down"))
    up_name = next(name for name in state if name.endswith(".up"))
    state[down_name].copy_(torch.randn(state[down_name].shape, generator=generator) * 0.2)
    state[up_name].copy_(torch.randn(state[up_name].shape, generator=generator) * 0.2)
    attachment.load_state_dict(state)

    delta = (state[up_name] @ state[down_name]).reshape(base_weight.shape) * (3.0 / 4.0)
    effective_weight = base_weight + delta
    if operation == "linear":
        expected = torch.nn.functional.linear(inputs, effective_weight, base_bias)
    else:
        assert isinstance(target, torch.nn.Conv2d)
        conv_inputs = inputs
        padding = target.padding
        if target.padding_mode != "zeros":
            height = target.dilation[0] * (target.kernel_size[0] - 1)
            width = target.dilation[1] * (target.kernel_size[1] - 1)
            conv_inputs = torch.nn.functional.pad(
                conv_inputs,
                (width // 2, width - width // 2, height // 2, height - height // 2),
                mode=target.padding_mode,
            )
            padding = 0
        expected = torch.nn.functional.conv2d(
            conv_inputs,
            effective_weight,
            base_bias,
            stride=target.stride,
            padding=padding,
            dilation=target.dilation,
            groups=target.groups,
        )
    actual = target(inputs)

    # Float32 reassociation moves the tested outputs by at most 5.96e-7.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)


def _direct_image_caption_batch(
    config: TrainingConfig,
    source: ImageCaptionDatasetSource,
    cursor: int,
    device: torch.device,
    vae: AutoencoderKL,
    text_encoder: ClipTextEncoder,
    tokenizer: PromptTokenizer,
) -> PreparedBatch:
    assert config.dataset is not None
    indices = []
    for offset in range(config.latent_shape[0]):
        absolute_sample = cursor * config.latent_shape[0] + offset
        epoch, epoch_offset = divmod(absolute_sample, len(config.dataset.inspection.items))
        preimage = f"{config.dataset.digest}:epoch:{epoch}".encode("ascii")
        seed = int.from_bytes(hashlib.sha256(preimage).digest()[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(
            len(config.dataset.inspection.items), generator=generator
        ).tolist()
        indices.append(permutation[epoch_offset])
    items = [config.dataset.inspection.items[index] for index in indices]
    pixels_from = cast(
        "Callable[[bytes], torch.Tensor]", object.__getattribute__(source, "_pixels")
    )
    pixels = torch.stack([pixels_from(item.image_path.read_bytes()) for item in items])
    with torch.no_grad():
        latents = vae.encode(process_input(pixels.to(device=device)))
        latents = latents.float() * SD15.single_stream_latent().scale_factor
        embeddings = []
        for item in items:
            caption = item.caption_path.read_bytes().decode("utf-8").strip()
            chunks = pack_spans(tokenizer.tokenize(caption), CLIP_L_PROFILE)
            embeddings.append(text_encoder.encode_chunks(chunks[:1]).embeddings)
    return PreparedBatch(
        latents=latents.detach(),
        text_embeddings=torch.cat(embeddings, dim=0).detach(),
    )


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
    ],
)
def test_precomputed_image_caption_batches_match_direct_encoders_across_epochs(
    tmp_path: Path,
    device: torch.device,
) -> None:
    config = image_dataset_config(tmp_path, device=str(device), batch_size=2)
    assert config.dataset is not None
    mapping = config.to_mapping()
    dataset_digest = config.dataset.digest
    config_digest = capability_report(config)["configDigest"]

    with deterministic_algorithms():
        source = image_data_source_factory(config)
        vae = tiny_vae().requires_grad_(False).eval().to(device=device, dtype=torch.float32)
        text_model = tiny_clip().requires_grad_(False).eval().to(device=device, dtype=torch.float32)
        text_encoder = ClipTextEncoder(text_model)
        tokenizer = PromptTokenizer(encode_word=load_clip_bpe().encode, disable_weights=True)
        for cursor in (0, 1, 2, 3, 4, 7):
            expected = _direct_image_caption_batch(
                config,
                source,
                cursor,
                device,
                vae,
                text_encoder,
                tokenizer,
            )
            actual = source.batch(
                cursor,
                generator=torch.Generator(device=device).manual_seed(cursor + 10),
                device=device,
            )
            assert torch.equal(actual.latents, expected.latents)
            assert torch.equal(actual.text_embeddings, expected.text_embeddings)

    assert config.to_mapping() == mapping
    assert config.dataset.digest == dataset_digest
    assert capability_report(config)["configDigest"] == config_digest
    assert TRAINING_RUNTIME_IDENTITY == "sd15-lora-torch/4"
    assert (
        TRAINING_SNAPSHOT_DIGEST
        == "blake3:8272c8b00a7aed9e4e412ee8b3207486cede4c494dfb08b1f0c85cee9c9b62d2"
    )


def test_image_caption_source_releases_each_encoder_before_loading_the_next(
    tmp_path: Path,
) -> None:
    config = image_dataset_config(tmp_path)
    assert config.dataset is not None
    references: list[weakref.ReferenceType[torch.nn.Module]] = []
    loaded: list[str] = []

    def vae_factory() -> AutoencoderKL:
        vae = tiny_vae()
        loaded.append("vae")
        references.append(weakref.ref(vae))
        return vae

    def text_model_factory() -> ClipTextModel:
        gc.collect()
        assert references[0]() is None
        text_model = tiny_clip()
        loaded.append("clip")
        references.append(weakref.ref(text_model))
        return text_model

    source = ImageCaptionDatasetSource(
        config.dataset,
        vae_factory,
        text_model_factory,
        batch_size=1,
        device=torch.device("cpu"),
    )
    gc.collect()

    assert loaded == ["vae", "clip"]
    assert all(reference() is None for reference in references)
    assert not {"_vae", "_text_model", "_text_encoder"} & set(vars(source))
    latents = cast("tuple[torch.Tensor, ...]", object.__getattribute__(source, "_latents"))
    text_embeddings = cast(
        "tuple[torch.Tensor, ...]", object.__getattribute__(source, "_text_embeddings")
    )
    assert all(tensor.device.type == "cpu" for tensor in (*latents, *text_embeddings))


def test_image_caption_batch_is_bit_exact_across_fresh_sources(tmp_path: Path) -> None:
    config = image_dataset_config(tmp_path)
    assert config.dataset is not None
    assert tuple(item.relative_image for item in config.dataset.inspection.items) == (
        "middle.png",
        "nested/ant.png",
        "zebra.png",
    )
    first = image_data_source_factory(config).batch(
        4,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    second = image_data_source_factory(config).batch(
        4,
        generator=torch.Generator().manual_seed(999),
        device=torch.device("cpu"),
    )

    assert torch.equal(first.latents, second.latents)
    assert torch.equal(first.text_embeddings, second.text_embeddings)
    assert first.latents.shape == config.latent_shape
    assert first.text_embeddings.shape == config.context_shape
    assert not first.latents.requires_grad
    assert not first.text_embeddings.requires_grad


def test_encoded_cache_miss_and_hit_match_memory_without_loading_encoders(
    tmp_path: Path,
) -> None:
    mapping = config_mapping()
    mapping["latentShape"] = [2, 4, 2, 2]
    mapping["contextShape"] = [2, 77, 16]
    mapping["dataset"] = image_dataset_mapping(tmp_path)
    memory_config = TrainingConfig.from_mapping(mapping)
    cached_mapping = copy.deepcopy(mapping)
    cached_dataset = cast("dict[str, object]", cached_mapping["dataset"])
    cache_root = tmp_path / "encoded-cache"
    cached_dataset["encodedCacheRoot"] = str(cache_root)
    cached_config = TrainingConfig.from_mapping(cached_mapping)

    memory_loaded: list[str] = []
    miss_loaded: list[str] = []
    hit_loaded: list[str] = []
    memory = tracked_image_data_source_factory(memory_loaded)(memory_config)
    miss = tracked_image_data_source_factory(miss_loaded)(cached_config)
    hit = tracked_image_data_source_factory(hit_loaded)(cached_config)

    assert memory_loaded == miss_loaded == ["vae", "clip"]
    assert hit_loaded == []
    for cursor in (0, 1, 2, 3, 4, 7):
        generator = torch.Generator().manual_seed(cursor)
        expected = memory.batch(cursor, generator=generator, device=torch.device("cpu"))
        actual_miss = miss.batch(cursor, generator=generator, device=torch.device("cpu"))
        actual_hit = hit.batch(cursor, generator=generator, device=torch.device("cpu"))
        assert_prepared_batch_equal(expected, actual_miss)
        assert_prepared_batch_equal(expected, actual_hit)

    report = capability_report(cached_config)
    dataset = cast("dict[str, object]", report["dataset"])
    assert dataset["encodedCache"] == {"state": "hit"}
    memory_report = cast("dict[str, object]", report["memoryLedger"])
    categories = cast("dict[str, int]", memory_report["categories"])
    assert categories["vaeEncoderParametersTransientLowerBound"] == 0
    assert categories["clipTextEncoderParametersTransientLowerBound"] == 0
    assert categories["encoderInputBatchTransientLowerBound"] == 0
    precision = cast("dict[str, str]", report["precisionPlan"])
    assert precision["datasetPrecomputeEncoders"] == "not loaded on encoded cache hit"


def test_encoded_cache_cuda_keys_ignore_index_but_keep_hardware_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = image_dataset_config(tmp_path)
    assert config.dataset is not None
    properties = {
        0: SimpleNamespace(name="same-gpu", major=9, minor=0),
        1: SimpleNamespace(name="same-gpu", major=9, minor=0),
    }
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties.__getitem__)

    first = training_encoded_cache.encoded_cache_key(
        config.dataset,
        batch_size=1,
        device=torch.device("cuda:0"),
    )
    second = training_encoded_cache.encoded_cache_key(
        config.dataset,
        batch_size=1,
        device=torch.device("cuda:1"),
    )
    assert first == second

    properties[1] = SimpleNamespace(name="other-gpu", major=9, minor=0)
    different_name = training_encoded_cache.encoded_cache_key(
        config.dataset,
        batch_size=1,
        device=torch.device("cuda:1"),
    )
    properties[1] = SimpleNamespace(name="same-gpu", major=8, minor=9)
    different_capability = training_encoded_cache.encoded_cache_key(
        config.dataset,
        batch_size=1,
        device=torch.device("cuda:1"),
    )
    assert different_name != first
    assert different_capability != first


def test_encoded_cache_concurrent_callers_build_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = config_mapping()
    mapping["latentShape"] = [1, 4, 2, 2]
    mapping["contextShape"] = [1, 77, 16]
    dataset_mapping = image_dataset_mapping(tmp_path)
    dataset_mapping["encodedCacheRoot"] = str(tmp_path / "encoded-cache")
    mapping["dataset"] = dataset_mapping
    config = TrainingConfig.from_mapping(mapping)
    assert config.dataset is not None
    settings = config.dataset

    original_load = training_encoded_cache.EncodedDatasetCache.load
    miss_threads: set[int] = set()
    miss_lock = threading.Lock()
    both_callers_missed = threading.Event()

    def observed_load(
        cache: training_encoded_cache.EncodedDatasetCache,
    ) -> training_encoded_cache.EncodedDatasetTensors | None:
        found = original_load(cache)
        if found is None:
            with miss_lock:
                miss_threads.add(threading.get_ident())
                if len(miss_threads) == 2:
                    both_callers_missed.set()
        return found

    monkeypatch.setattr(training_encoded_cache.EncodedDatasetCache, "load", observed_load)
    loaded: list[str] = []

    def vae_factory() -> AutoencoderKL:
        assert both_callers_missed.wait(timeout=10)
        loaded.append("vae")
        return tiny_vae()

    def clip_factory() -> ClipTextModel:
        loaded.append("clip")
        return tiny_clip()

    def build_source() -> ImageCaptionDatasetSource:
        return ImageCaptionDatasetSource(
            settings,
            vae_factory,
            clip_factory,
            batch_size=1,
            device=torch.device("cpu"),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(build_source)
        second_future = executor.submit(build_source)
        first = first_future.result(timeout=30)
        second = second_future.result(timeout=30)

    assert loaded == ["vae", "clip"]
    assert_prepared_batch_equal(
        first.batch(0, generator=torch.Generator().manual_seed(1), device=torch.device("cpu")),
        second.batch(0, generator=torch.Generator().manual_seed(2), device=torch.device("cpu")),
    )


def test_encoded_cache_builder_failure_releases_lock(tmp_path: Path) -> None:
    mapping = config_mapping()
    mapping["latentShape"] = [1, 4, 2, 2]
    mapping["contextShape"] = [1, 77, 16]
    dataset_mapping = image_dataset_mapping(tmp_path)
    dataset_mapping["encodedCacheRoot"] = str(tmp_path / "encoded-cache")
    mapping["dataset"] = dataset_mapping
    config = TrainingConfig.from_mapping(mapping)
    assert config.dataset is not None

    def failed_vae() -> AutoencoderKL:
        raise RuntimeError("injected builder failure")

    with pytest.raises(RuntimeError, match="injected builder failure"):
        ImageCaptionDatasetSource(
            config.dataset,
            failed_vae,
            tiny_clip,
            batch_size=1,
            device=torch.device("cpu"),
        )

    loaded: list[str] = []
    recovered = tracked_image_data_source_factory(loaded)(config)
    assert loaded == ["vae", "clip"]
    assert (
        recovered.batch(
            0,
            generator=torch.Generator().manual_seed(1),
            device=torch.device("cpu"),
        ).latents.shape
        == config.latent_shape
    )


def test_encoded_cache_repeated_builds_are_bit_identical(tmp_path: Path) -> None:
    base_mapping = config_mapping()
    base_mapping["latentShape"] = [1, 4, 2, 2]
    base_mapping["contextShape"] = [1, 77, 16]
    base_mapping["dataset"] = image_dataset_mapping(tmp_path)
    cache_roots = (tmp_path / "cache-a", tmp_path / "cache-b")

    for cache_root in cache_roots:
        mapping = copy.deepcopy(base_mapping)
        dataset_mapping = cast("dict[str, object]", mapping["dataset"])
        dataset_mapping["encodedCacheRoot"] = str(cache_root)
        tracked_image_data_source_factory([])(TrainingConfig.from_mapping(mapping))

    manifests = [next((root / "v1/entries").glob("*.json")).read_bytes() for root in cache_roots]
    shards = [next((root / "v1/shards").glob("*.safetensors")).read_bytes() for root in cache_roots]
    assert manifests[0] == manifests[1]
    assert shards[0] == shards[1]


def test_encoded_cache_tensor_bytes_match_storage_bytes_in_logical_order() -> None:
    transposed = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()
    assert not transposed.is_contiguous()
    indices = torch.arange(-6, 6, dtype=torch.int64).reshape(3, 4)
    sliced = indices[:, 1:3]
    assert not sliced.is_contiguous()
    negated = torch._neg_view(  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
        torch.tensor([1.5, -2.5], dtype=torch.float32)
    )
    assert negated.is_neg()
    for tensor in (transposed, indices, sliced, negated):
        production_bytes = training_encoded_cache._tensor_bytes(  # pyright: ignore[reportPrivateUsage]
            tensor
        )
        assert production_bytes == _tensor_bytes(tensor.clone())


def test_encoded_cache_repairs_truncation_and_does_not_cross_dataset_digests(
    tmp_path: Path,
) -> None:
    mapping = config_mapping()
    mapping["latentShape"] = [1, 4, 2, 2]
    mapping["contextShape"] = [1, 77, 16]
    dataset_mapping = image_dataset_mapping(tmp_path)
    cache_root = tmp_path / "encoded-cache"
    dataset_mapping["encodedCacheRoot"] = str(cache_root)
    mapping["dataset"] = dataset_mapping
    original = TrainingConfig.from_mapping(mapping)

    expected = tracked_image_data_source_factory([])(original).batch(
        3,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    shard = encoded_cache_shard(cache_root)
    full_size = shard.stat().st_size
    shard.write_bytes(shard.read_bytes()[:32])
    recomputed: list[str] = []
    repaired = tracked_image_data_source_factory(recomputed)(original).batch(
        3,
        generator=torch.Generator().manual_seed(2),
        device=torch.device("cpu"),
    )
    assert recomputed == ["vae", "clip"]
    assert shard.stat().st_size == full_size
    assert_prepared_batch_equal(expected, repaired)

    caption = Path(cast("str", dataset_mapping["root"])) / "middle.txt"
    caption.write_text("changed caption", encoding="utf-8")
    changed_mapping = copy.deepcopy(mapping)
    changed_dataset = cast("dict[str, object]", changed_mapping["dataset"])
    changed_dataset.pop("digest", None)
    changed = TrainingConfig.from_mapping(changed_mapping)
    assert changed.dataset is not None and original.dataset is not None
    assert changed.dataset.digest != original.dataset.digest
    changed_loaded: list[str] = []
    tracked_image_data_source_factory(changed_loaded)(changed)
    assert changed_loaded == ["vae", "clip"]
    assert len(tuple((cache_root / "v1" / "entries").glob("*.json"))) == 2


def test_encoded_cache_root_is_runtime_only_but_persisted_for_cold_resume(
    tmp_path: Path,
) -> None:
    base_mapping = config_mapping()
    base_mapping["latentShape"] = [1, 4, 2, 2]
    base_mapping["contextShape"] = [1, 77, 16]
    base_mapping["dataset"] = image_dataset_mapping(tmp_path)

    mappings = [copy.deepcopy(base_mapping) for _ in range(3)]
    cache_roots = (None, tmp_path / "cache-a", tmp_path / "cache-b")
    for mapping, cache_root in zip(mappings, cache_roots, strict=True):
        if cache_root is not None:
            dataset = cast("dict[str, object]", mapping["dataset"])
            dataset["encodedCacheRoot"] = str(cache_root)
    configs = [TrainingConfig.from_mapping(mapping) for mapping in mappings]
    digests = [capability_report(config)["configDigest"] for config in configs]
    assert digests[0] == digests[1] == digests[2]
    assert configs[1].dataset is not None
    configured_cache_root = cache_roots[1]
    assert configs[1].dataset.encoded_cache_root == str(configured_cache_root.resolve())

    changed_mapping = copy.deepcopy(base_mapping)
    changed_dataset = cast("dict[str, object]", changed_mapping["dataset"])
    text_state = cast("dict[str, object]", changed_dataset["textEncoderState"])
    text_state["prefix"] = "different."
    changed = TrainingConfig.from_mapping(changed_mapping)
    assert capability_report(changed)["configDigest"] != digests[0]

    lineage_store = make_store(tmp_path / "lineage.sqlite")
    lineage_root = tmp_path / "lineage-checkpoints"
    try:
        lineage = SD15LoRATrainingService(
            lineage_store,
            lineage_root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        handles = [
            lineage.create("same-lineage", json.dumps(config.to_mapping()))[0] for config in configs
        ]
        assert handles[0] == handles[1] == handles[2]
    finally:
        lineage_store.close()

    cache_root = configured_cache_root
    cached_config = configs[1]
    with deterministic_algorithms():
        tracked_image_data_source_factory([])(cached_config)
    serialized = json.dumps(cached_config.to_mapping(), sort_keys=True)
    uninterrupted_root = tmp_path / "cache-uninterrupted-checkpoints"
    resumed_root = tmp_path / "cache-resumed-checkpoints"
    uninterrupted_store = make_store(tmp_path / "cache-uninterrupted.sqlite")
    resumed_store = make_store(tmp_path / "cache-resumed.sqlite")
    resumed_configs: list[str | None] = []
    resumed_encoder_loads: list[str] = []

    def cached_source(config: TrainingConfig) -> ImageCaptionDatasetSource:
        assert config.dataset is not None
        resumed_configs.append(config.dataset.encoded_cache_root)
        return tracked_image_data_source_factory(resumed_encoder_loads)(config)

    try:
        with deterministic_algorithms():
            uninterrupted = SD15LoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=model_factory,
                data_source_factory=cached_source,
            )
            initial_a, _ = uninterrupted.create("cache-resume", serialized)
            final_a = uninterrupted.advance(initial_a, 2, "same operation").handle

            armed = False
            paused: SD15LoRATrainingService

            def cancelled() -> bool:
                return armed and paused.steps_run >= 1

            paused = SD15LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=cached_source,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("cache-resume", serialized)
            initial_state = checkpoint_state(resumed_root, initial_b.checkpoint_manifest_digest)
            persisted_dataset = cast("dict[str, object]", initial_state.config["dataset"])
            assert persisted_dataset["encodedCacheRoot"] == str(cache_root.resolve())
            armed = True
            with pytest.raises(TrainingAdvancePaused, match="step 1 of 2"):
                paused.advance(initial_b, 2, "same operation")
            successor = SD15LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=cached_source,
            )
            final_b = successor.advance(initial_b, 2, "same operation").handle

        state_a = checkpoint_state(uninterrupted_root, final_a.checkpoint_manifest_digest)
        state_b = checkpoint_state(resumed_root, final_b.checkpoint_manifest_digest)
        assert resumed_encoder_loads == []
        assert resumed_configs and set(resumed_configs) == {str(cache_root.resolve())}
        assert state_a.loss == state_b.loss
        assert_tree_equal(state_a.adapter, state_b.adapter)
        assert_tree_equal(state_a.optimizer, state_b.optimizer)
        assert_tree_equal(state_a.rng, state_b.rng)
    finally:
        uninterrupted_store.close()
        resumed_store.close()


def test_default_encoder_loader_matches_injected_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = image_dataset_config(tmp_path)
    assert config.dataset is not None

    def detect_tiny_clip(_geometries: Mapping[str, TensorGeometry]) -> ClipTextConfig:
        return _TINY_CLIP

    monkeypatch.setattr(training_data, "detect_clip_text_config", detect_tiny_clip)
    loaded = training_data.default_image_caption_source(
        config.dataset, batch_size=1, device=torch.device("cpu")
    ).batch(
        0,
        generator=torch.Generator().manual_seed(4),
        device=torch.device("cpu"),
    )
    injected = image_data_source_factory(config).batch(
        0,
        generator=torch.Generator().manual_seed(4),
        device=torch.device("cpu"),
    )
    assert torch.equal(loaded.latents, injected.latents)
    assert torch.equal(loaded.text_embeddings, injected.text_embeddings)


def test_text_encoder_loader_discards_inert_position_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = image_dataset_config(
        tmp_path,
        clip_extras={"text_model.embeddings.position_ids": torch.arange(77).unsqueeze(0)},
    )
    assert config.dataset is not None
    detected: dict[str, TensorGeometry] = {}

    def detect_tiny_clip(geometries: Mapping[str, TensorGeometry]) -> ClipTextConfig:
        detected.update(geometries)
        return _TINY_CLIP

    monkeypatch.setattr(training_data, "detect_clip_text_config", detect_tiny_clip)
    training_data.default_image_caption_source(
        config.dataset, batch_size=1, device=torch.device("cpu")
    )

    assert "text_model.embeddings.position_ids" not in detected
    memory = cast("dict[str, object]", capability_report(config)["memoryLedger"])
    categories = cast("dict[str, int]", memory["categories"])
    expected_elements = sum(
        value.numel()
        for key, value in tiny_clip().state_dict().items()
        if key != "text_projection.weight"
    )
    assert categories["clipTextEncoderParametersTransientLowerBound"] == expected_elements * 4


def test_text_encoder_loader_rejects_keys_outside_inference_inert_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unexpected = "text_model.embeddings.unexpected"
    config = image_dataset_config(
        tmp_path,
        clip_extras={
            "text_model.embeddings.position_ids": torch.arange(77).unsqueeze(0),
            "embeddings.position_ids": torch.arange(77).unsqueeze(0),
            unexpected: torch.zeros(1),
        },
    )
    assert config.dataset is not None
    expected = set(tiny_clip().state_dict()) - {"text_projection.weight"}

    def detect_tiny_clip(geometries: Mapping[str, TensorGeometry]) -> ClipTextConfig:
        extras = set(geometries) - expected
        if extras:
            raise ClipTextDetectError(f"unexpected key {sorted(extras)[0]}")
        return _TINY_CLIP

    monkeypatch.setattr(training_data, "detect_clip_text_config", detect_tiny_clip)
    with pytest.raises(ClipTextDetectError, match="unexpected key"):
        training_data.default_image_caption_source(
            config.dataset, batch_size=1, device=torch.device("cpu")
        )


def test_dataset_digest_changes_with_content_and_refuses_stale_identity(tmp_path: Path) -> None:
    mapping = config_mapping()
    mapping["latentShape"] = [1, 4, 2, 2]
    mapping["contextShape"] = [1, 77, 16]
    dataset = image_dataset_mapping(tmp_path)
    mapping["dataset"] = dataset
    first = TrainingConfig.from_mapping(mapping)
    assert first.dataset is not None

    dataset_root = cast("str", dataset["root"])
    image = Path(dataset_root) / "middle.png"
    Image.new("RGB", (10, 10), (1, 2, 3)).save(image)
    changed = TrainingConfig.from_mapping(mapping)
    assert changed.dataset is not None
    assert changed.dataset.digest != first.dataset.digest
    assert capability_report(changed)["configDigest"] != capability_report(first)["configDigest"]
    with pytest.raises(TrainingConfigError, match="dataset digest changed"):
        TrainingConfig.from_mapping(first.to_mapping())


def test_dataset_resolution_must_match_latent_shape(tmp_path: Path) -> None:
    mapping = config_mapping()
    mapping["contextShape"] = [1, 77, 16]
    mapping["dataset"] = image_dataset_mapping(tmp_path)
    with pytest.raises(TrainingConfigError, match="dataset.resolution divided by 8"):
        TrainingConfig.from_mapping(mapping)


def test_dry_run_lists_caption_errors_and_encoder_memory(tmp_path: Path) -> None:
    config = image_dataset_config(tmp_path, invalid_captions=True)
    serialized = json.dumps(config.to_mapping(), sort_keys=True)
    store = make_store(tmp_path / "invalid-captions.sqlite")
    try:
        service = SD15LoRATrainingService(
            store,
            tmp_path / "invalid-caption-checkpoints",
            model_factory=model_factory,
            data_source_factory=image_data_source_factory,
        )
        report = service.dry_run(serialized)
        dataset = cast("dict[str, object]", report["dataset"])
        coverage = cast("dict[str, int]", dataset["captionCoverage"])
        errors = cast("list[str]", dataset["errors"])
        assert dataset["itemCount"] == 3
        assert dataset["resolution"] == {"height": 16, "width": 16}
        assert coverage == {"total": 3, "present": 2, "nonEmpty": 1}
        assert any("missing caption" in error for error in errors)
        assert any("caption is empty" in error for error in errors)
        memory = cast("dict[str, object]", report["memoryLedger"])
        categories = cast("dict[str, int]", memory["categories"])
        assert categories["vaeEncoderParametersTransientLowerBound"] > 0
        assert categories["clipTextEncoderParametersTransientLowerBound"] > 0
        assert categories["encoderInputBatchTransientLowerBound"] == 3 * 16 * 16 * 4
        assert categories["encodedDatasetStoreResidentCpu"] == 3 * (4 * 2 * 2 + 77 * 16) * 4
        estimate_basis = cast("str", memory["datasetEncoderEstimateBasis"])
        assert "one encoder loaded at a time" in estimate_basis
        assert "excludes encoder activations" in estimate_basis
        precision = cast("dict[str, str]", report["precisionPlan"])
        assert precision["datasetPrecomputeEncoders"] == "float32, transient, one at a time"
        assert precision["encodedDatasetStore"] == "float32, CPU resident"
        with pytest.raises(ValueError, match="dataset validation failed"):
            service.create("invalid-captions", serialized)
    finally:
        store.close()


def test_image_caption_training_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    config = image_dataset_config(tmp_path)
    serialized = json.dumps(config.to_mapping(), sort_keys=True)
    uninterrupted_root = tmp_path / "folder-uninterrupted-checkpoints"
    resumed_root = tmp_path / "folder-resumed-checkpoints"
    uninterrupted_store = make_store(tmp_path / "folder-uninterrupted.sqlite")
    resumed_store = make_store(tmp_path / "folder-resumed.sqlite")
    try:
        with deterministic_algorithms():
            uninterrupted = SD15LoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=model_factory,
                data_source_factory=image_data_source_factory,
            )
            initial_a, _ = uninterrupted.create("folder-resume-proof", serialized)
            final_a = uninterrupted.advance(initial_a, 2, "same operation").handle

            cancellation_armed = False
            paused: SD15LoRATrainingService

            def cancelled() -> bool:
                return cancellation_armed and paused.steps_run >= 1

            paused = SD15LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=image_data_source_factory,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("folder-resume-proof", serialized)
            cancellation_armed = True
            with pytest.raises(TrainingAdvancePaused, match="step 1 of 2"):
                paused.advance(initial_b, 2, "same operation")

            successor = SD15LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=image_data_source_factory,
            )
            final_b = successor.advance(initial_b, 2, "same operation").handle

            state_a = checkpoint_state(uninterrupted_root, final_a.checkpoint_manifest_digest)
            state_b = checkpoint_state(resumed_root, final_b.checkpoint_manifest_digest)
            assert state_a.step_cursor == state_b.step_cursor == 2
            assert state_a.data_cursor == state_b.data_cursor == 2
            assert state_a.loss == state_b.loss
            assert_tree_equal(state_a.adapter, state_b.adapter)
            assert_tree_equal(state_a.optimizer, state_b.optimizer)
            assert_tree_equal(state_a.rng, state_b.rng)
    finally:
        uninterrupted_store.close()
        resumed_store.close()


def test_nonzero_unet_dropout_is_refused_for_exact_resume() -> None:
    unet = dict(_UNET)
    unet["dropout"] = 0.1
    mapping = config_mapping()
    mapping["unet"] = unet
    with pytest.raises(TrainingConfigError, match="process-global RNG"):
        TrainingConfig.from_mapping(mapping)


def test_linear_transformer_projection_mode_is_refused() -> None:
    unet = dict(_UNET)
    unet["useLinearInTransformer"] = True
    mapping = config_mapping()
    mapping["unet"] = unet
    with pytest.raises(TrainingConfigError, match="must be false"):
        TrainingConfig.from_mapping(mapping)


def test_adm_conditioning_is_refused_for_sd15() -> None:
    unet = dict(_UNET)
    unet["admInChannels"] = 2816
    mapping = config_mapping()
    mapping["unet"] = unet
    with pytest.raises(TrainingConfigError, match="available only for SDXL"):
        TrainingConfig.from_mapping(mapping)


def test_dry_run_refuses_unsupported_device() -> None:
    config = TrainingConfig.from_mapping(config_mapping(device="meta"))
    with pytest.raises(ValueError, match="supports cpu and cuda"):
        capability_report(config)


def test_base_state_requires_content_identity() -> None:
    mapping = config_mapping()
    mapping["baseState"] = {"path": "/models/sd15.safetensors"}
    with pytest.raises(TrainingConfigError, match="baseState.digest"):
        TrainingConfig.from_mapping(mapping)


def test_base_state_rejects_sha256_identity() -> None:
    mapping = config_mapping()
    mapping["baseState"] = {
        "path": "/models/sd15.safetensors",
        "digest": "sha256:" + "0" * 64,
    }
    with pytest.raises(TrainingConfigError, match="'blake3:' plus 64 lowercase hex"):
        TrainingConfig.from_mapping(mapping)


def test_base_state_content_is_verified_before_loading(tmp_path: Path) -> None:
    base = tmp_path / "base.safetensors"
    base.write_bytes(b"not the pinned base")
    mapping = config_mapping()
    mapping["baseState"] = {
        "path": str(base),
        "digest": "blake3:" + "0" * 64,
    }
    config = TrainingConfig.from_mapping(mapping)
    with pytest.raises(AssetIntegrityError, match="digest_mismatch"):
        default_model_factory(config)


def test_base_state_verification_is_reused_for_unchanged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base.safetensors"
    model = model_factory(TrainingConfig.from_mapping(config_mapping()))
    write_safetensors(base, dict(model.state_dict()))
    mapping = config_mapping()
    mapping["baseState"] = {"path": str(base), "digest": file_digest(base)}
    config = TrainingConfig.from_mapping(mapping)
    hash_count = 0
    original_hash_handle = asset_integrity._hash_handle  # pyright: ignore[reportPrivateUsage]

    def count_hashes(handle: object) -> str:
        nonlocal hash_count
        hash_count += 1
        return original_hash_handle(handle)  # type: ignore[arg-type]

    clear_verified_cache()
    monkeypatch.setattr(asset_integrity, "_hash_handle", count_hashes)
    default_model_factory(config)
    default_model_factory(config)

    # Fingerprint reuse requires POSIX change stamps; Windows rehashes each
    # resolution by design, so expect one hash per factory call there.
    trusted = asset_integrity._TRUST_FINGERPRINTS  # pyright: ignore[reportPrivateUsage]
    assert hash_count == (1 if trusted else 2)


def test_trainer_freezes_base_and_accumulates_with_float32_masters() -> None:
    config = TrainingConfig.from_mapping(config_mapping(accumulation=2))
    trainer = SD15LoRATrainer(config, model_factory(config), SyntheticBatches(config))

    adapter_names = {name for name, _ in trainer.attachment.named_parameters()}
    assert adapter_names
    assert all(
        parameter.dtype == torch.float32 and parameter.requires_grad
        for parameter in trainer.attachment.parameters()
    )
    for name, parameter in trainer.model.named_parameters():
        assert not parameter.requires_grad, name

    loss = trainer.train_step()
    assert math.isfinite(loss)
    assert trainer.step_cursor == 1
    assert trainer.data_cursor == 2
    assert all(parameter.dtype == torch.float32 for parameter in trainer.attachment.parameters())


def _sd_blocks(model: torch.nn.Module) -> tuple[tuple[str, torch.nn.Module], ...]:
    unet = cast("UNetModel", model)
    return (
        *((f"input_blocks.{index}", block) for index, block in enumerate(unet.input_blocks)),
        ("middle_block", unet.middle_block),
        *((f"output_blocks.{index}", block) for index, block in enumerate(unet.output_blocks)),
    )


def _clone_sd_block_inputs(model: torch.nn.Module) -> None:
    def cloned(
        original_forward: Callable[..., torch.Tensor],
    ) -> Callable[..., torch.Tensor]:
        def forward(
            _block: torch.nn.Module,
            x: torch.Tensor,
            emb: torch.Tensor,
            context: torch.Tensor,
            output_shape: tuple[int, ...] | None = None,
            attention_guidance: object | None = None,
            ipadapter: object | None = None,
        ) -> torch.Tensor:
            del _block
            return original_forward(
                x.clone(), emb, context, output_shape, attention_guidance, ipadapter
            )

        return forward

    for _, block in _sd_blocks(model):
        block.forward = MethodType(cloned(block.forward), block)  # type: ignore[method-assign]


def _first_step_evidence(
    trainer: SD15LoRATrainer,
) -> tuple[float, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    gradients: dict[str, torch.Tensor] = {}
    activations: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def capture_gradients(
        _optimizer: torch.optim.Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        for name, parameter in trainer.attachment.named_parameters():
            assert parameter.grad is not None, name
            gradients[name] = parameter.grad.detach().cpu().clone()

    def capture_activation(
        name: str,
    ) -> Callable[[torch.nn.Module, tuple[object, ...], object], None]:
        def hook(
            _module: torch.nn.Module,
            _args: tuple[object, ...],
            output: object,
        ) -> None:
            if name not in activations:
                activations[name] = cast("torch.Tensor", output).detach().cpu().clone()

        return hook

    handles.append(trainer.optimizer.register_step_pre_hook(capture_gradients))
    for name, block in _sd_blocks(trainer.model):
        handles.append(block.register_forward_hook(capture_activation(name)))
    try:
        loss = trainer.train_step()
    finally:
        for handle in handles:
            handle.remove()
    return loss, gradients, activations


def test_checkpointing_mode_defaults_round_trips_and_changes_identity() -> None:
    default = TrainingConfig.from_mapping(config_mapping())
    whole = TrainingConfig.from_mapping(config_mapping(checkpointing_mode="wholeModel"))

    assert default.checkpointing_mode == "blockReentrant"
    assert default.to_mapping()["checkpointingMode"] == "blockReentrant"
    assert default.identity_mapping()["checkpointingMode"] == "blockReentrant"
    assert whole.checkpointing_mode == "wholeModel"
    assert whole.to_mapping()["checkpointingMode"] == "wholeModel"
    assert default.identity_mapping() != whole.identity_mapping()

    mapping = config_mapping()
    mapping["checkpointingMode"] = "blocks"
    with pytest.raises(TrainingConfigError, match="checkpointingMode"):
        TrainingConfig.from_mapping(mapping)


def test_checkpointing_modes_pin_forward_and_gradient_contracts() -> None:
    block_config = TrainingConfig.from_mapping(config_mapping())
    whole_config = TrainingConfig.from_mapping(config_mapping(checkpointing_mode="wholeModel"))
    disabled_config = TrainingConfig.from_mapping(config_mapping(checkpointing=False))

    block = SD15LoRATrainer(
        block_config, model_factory(block_config), SyntheticBatches(block_config)
    )
    whole = SD15LoRATrainer(
        whole_config, model_factory(whole_config), SyntheticBatches(whole_config)
    )
    disabled = SD15LoRATrainer(
        disabled_config, model_factory(disabled_config), SyntheticBatches(disabled_config)
    )
    clone_reference = SD15LoRATrainer(
        disabled_config, model_factory(disabled_config), SyntheticBatches(disabled_config)
    )
    _clone_sd_block_inputs(clone_reference.model)

    assert all("forward" in module.__dict__ for _, module in _sd_blocks(block.model))
    assert all("forward" not in module.__dict__ for _, module in _sd_blocks(whole.model))

    with deterministic_algorithms():
        block_loss, block_gradients, block_activations = _first_step_evidence(block)
        whole_loss, whole_gradients, whole_activations = _first_step_evidence(whole)
        disabled_loss, disabled_gradients, _ = _first_step_evidence(disabled)
        clone_loss, clone_gradients, clone_activations = _first_step_evidence(clone_reference)

    assert block_loss == whole_loss == disabled_loss == clone_loss
    assert block_activations.keys() == whole_activations.keys() == clone_activations.keys()
    for name, activation in block_activations.items():
        assert torch.equal(activation, whole_activations[name]), name
        assert torch.equal(activation, clone_activations[name]), name
    for name, gradient in block_gradients.items():
        assert torch.equal(gradient, clone_gradients[name]), name
    for name, gradient in whole_gradients.items():
        assert torch.equal(gradient, disabled_gradients[name]), name


def test_block_reentrant_checkpointing_is_bit_exact_across_runs() -> None:
    config = TrainingConfig.from_mapping(config_mapping())
    trainers = tuple(
        SD15LoRATrainer(config, model_factory(config), SyntheticBatches(config)) for _ in range(2)
    )

    with deterministic_algorithms():
        losses = tuple(tuple(trainer.train_step() for _ in range(3)) for trainer in trainers)

    assert losses[0] == losses[1]
    states = tuple(trainer.attachment.state_dict() for trainer in trainers)
    assert states[0].keys() == states[1].keys()
    for name, value in states[0].items():
        assert torch.equal(value, states[1][name]), name


def test_prepared_timesteps_require_integer_dtype() -> None:
    config = TrainingConfig.from_mapping(config_mapping())
    batch = PreparedBatch(
        latents=torch.zeros(config.latent_shape),
        text_embeddings=torch.zeros(config.context_shape),
        timesteps=torch.tensor([1.0]),
        noise=torch.zeros(config.latent_shape),
    )
    trainer = SD15LoRATrainer(config, model_factory(config), FixedBatches(batch))
    with pytest.raises(ValueError, match="integer dtype"):
        trainer.train_step()


@pytest.mark.parametrize("base_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("optimizer", ["adamw", "factored-adamw"])
def test_resume_replay_and_safe_point_cancellation_are_bit_exact(
    tmp_path: Path, optimizer: str, base_dtype: str
) -> None:
    # checkpointInterval=1 keeps the per-step recovery stream this test pins.
    config = serialized_config(optimizer=optimizer, baseDtype=base_dtype, checkpointInterval=1)
    uninterrupted_root = tmp_path / "uninterrupted-checkpoints"
    resumed_root = tmp_path / "resumed-checkpoints"
    uninterrupted_store = make_store(tmp_path / "uninterrupted.sqlite")
    resumed_store = make_store(tmp_path / "resumed.sqlite")
    try:
        with deterministic_algorithms():
            uninterrupted = SD15LoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
            )
            initial_a, _ = uninterrupted.create("resume-proof", config)
            final_a = uninterrupted.advance(initial_a, 3, "same operation").handle

            cancellation_armed = False
            paused: SD15LoRATrainingService

            def cancelled() -> bool:
                return cancellation_armed and paused.steps_run >= 1

            paused = SD15LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("resume-proof", config)
            cancellation_armed = True
            with pytest.raises(TrainingAdvancePaused, match="step 1 of 3"):
                paused.advance(initial_b, 3, "same operation")
            assert paused.steps_run == 1

            names_after_pause = [
                record.name
                for record in resumed_store.read_events(
                    initial_b.session_id, after=0, limit=100
                ).records
            ]
            assert names_after_pause.count(TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED) == 1
            assert names_after_pause.count(TrainingEventName.ADVANCE_PAUSED) == 1

            successor = SD15LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
            )
            final_b = successor.advance(initial_b, 3, "same operation").handle
            assert successor.steps_run == 2
            session = resumed_store.get_session(initial_b.session_id)
            assert session is not None and session.fence_epoch == 3

            state_a = checkpoint_state(uninterrupted_root, final_a.checkpoint_manifest_digest)
            state_b = checkpoint_state(resumed_root, final_b.checkpoint_manifest_digest)
            assert state_a.step_cursor == state_b.step_cursor == 3
            assert state_a.data_cursor == state_b.data_cursor == 3
            losses_a = recovery_losses(
                uninterrupted_root, uninterrupted_store, initial_a.session_id
            )
            losses_b = recovery_losses(resumed_root, resumed_store, initial_b.session_id)
            assert len(losses_a) == 3
            assert losses_a == losses_b
            assert state_a.loss == state_b.loss == losses_a[-1]
            assert_tree_equal(state_a.adapter, state_b.adapter)
            assert_tree_equal(state_a.optimizer, state_b.optimizer)
            assert_tree_equal(state_a.rng, state_b.rng)
            assert all(tensor.dtype == torch.float32 for tensor in state_b.adapter.values())
            assert tree_tensors(state_b.optimizer)
            assert all(tensor.dtype == torch.float32 for tensor in tree_tensors(state_b.optimizer))

            before_replay = successor.steps_run
            replay = successor.advance(initial_b, 3, "same operation")
            assert replay.replayed
            assert replay.handle == final_b
            assert successor.steps_run == before_replay

            all_names = [
                record.name
                for record in resumed_store.read_events(
                    initial_b.session_id, after=0, limit=100
                ).records
            ]
            assert all_names.count(TrainingEventName.ADVANCE_RESUMED) == 1
            assert all_names.count(TrainingEventName.RECOVERY_CHECKPOINT_PUBLISHED) == 3
            assert all_names.count(TrainingEventName.ADVANCE_COMMITTED) == 1
    finally:
        uninterrupted_store.close()
        resumed_store.close()


def test_cancellation_during_final_step_pauses_before_commit(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        cancelling: SD15LoRATrainingService

        def cancelled() -> bool:
            return cancelling.steps_run == 1

        cancelling = SD15LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )
        initial, _ = cancelling.create("final-safe-point", serialized_config())
        with pytest.raises(TrainingAdvancePaused, match="step 1 of 1"):
            cancelling.advance(initial, 1, "one step")
        assert cancelling.steps_run == 1
        names = [
            record.name
            for record in store.read_events(initial.session_id, after=0, limit=100).records
        ]
        assert TrainingEventName.ADVANCE_PAUSED in names
        assert TrainingEventName.ADVANCE_COMMITTED not in names

        successor = SD15LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        outcome = successor.advance(initial, 1, "one step")
        assert successor.steps_run == 0
        assert outcome.handle.step_cursor == 1
        assert not outcome.replayed
    finally:
        store.close()


def test_default_cadence_checkpoints_only_at_advance_boundaries(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = SD15LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create("boundary-cadence", serialized_config())
        final = service.advance(initial, 3, "three steps").handle

        assert manifest_step_cursors(root) == [0, 3]
        assert not (root / "exports").exists()
        published = recovery_manifest_digests(store, initial.session_id)
        assert published == (final.checkpoint_manifest_digest,)
        assert manifest_digests(root) == {initial.checkpoint_manifest_digest, *published}
    finally:
        store.close()


@pytest.mark.parametrize("checkpoint_interval", [0, 2])
def test_lora_export_interval_writes_only_adapter_snapshots_at_its_cadence(
    tmp_path: Path,
    checkpoint_interval: int,
) -> None:
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = SD15LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create(
            "lora-export-cadence",
            serialized_config(
                checkpointInterval=checkpoint_interval,
                loraExportInterval=2,
            ),
        )
        service.advance(initial, 5, "five steps")

        export_dir = root / "exports" / "cadence" / initial.session_id
        assert [path.name for path in sorted(export_dir.iterdir())] == [
            "step-000000000002.safetensors",
            "step-000000000004.safetensors",
        ]
        assert manifest_step_cursors(root) == ([0, 5] if checkpoint_interval == 0 else [0, 2, 4, 5])
        for step in (2, 4):
            source = load_safetensors_header(export_dir / f"step-{step:012d}.safetensors")
            assert source.metadata() == {
                "dinkster_config_digest": initial.config_digest,
                "dinkster_runtime_identity": "sd15-lora-torch/4",
                "dinkster_session_id": initial.session_id,
                "dinkster_step_cursor": str(step),
            }
            assert {source.entry(key).geometry.dtype.name for key in source.keys()} == {"float32"}
    finally:
        store.close()


def test_cadence_export_preserves_full_checkpoint_adapter_masters_byte_exactly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = SD15LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create(
            "lora-export-byte-proof",
            serialized_config(checkpointInterval=2, loraExportInterval=2),
        )
        service.advance(initial, 3, "three steps")

        step_two_digest = recovery_manifest_digests(store, initial.session_id)[0]
        state = checkpoint_state(root, step_two_digest)
        config = TrainingConfig.from_mapping(state.config)
        targets = resolve_lora_targets(model_factory(config), config.rank)
        exported = load_tensors(
            root / "exports" / "cadence" / initial.session_id / "step-000000000002.safetensors"
        )

        for target in targets:
            stem = "lora_unet_" + target.module_path.replace(".", "_")
            down = exported[f"{stem}.lora_down.weight"].reshape(
                state.adapter[f"{target.target_id}.down"].shape
            )
            up = exported[f"{stem}.lora_up.weight"].reshape(
                state.adapter[f"{target.target_id}.up"].shape
            )
            assert torch.equal(down, state.adapter[f"{target.target_id}.down"])
            assert torch.equal(up, state.adapter[f"{target.target_id}.up"])
    finally:
        store.close()


def test_failed_cadence_export_is_retried_before_checkpoint_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = SD15LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create(
            "lora-export-retry",
            serialized_config(loraExportInterval=1),
        )
        attempts = 0

        def fail_export(
            self: SD15LoRATrainingService,
            *,
            session_id: str,
            config_digest: str,
            runtime: object,
        ) -> tuple[Path, str]:
            nonlocal attempts
            del self, session_id, config_digest, runtime
            attempts += 1
            raise OSError("injected cadence export failure")

        service._write_intermediate_lora = MethodType(  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
            fail_export,
            service,
        )
        with pytest.raises(OSError, match="injected cadence export failure"):
            service.advance(initial, 1, "one step")
        assert manifest_step_cursors(root) == [0]

        successor = SD15LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        final = successor.advance(initial, 1, "one step").handle
        assert attempts == 1
        assert successor.steps_run == 1
        assert final.step_cursor == 1
        assert manifest_step_cursors(root) == [0, 1]
        assert (
            root / "exports" / "cadence" / initial.session_id / "step-000000000001.safetensors"
        ).is_file()
    finally:
        store.close()


def test_checkpoint_interval_one_reproduces_per_step_checkpoints(tmp_path: Path) -> None:
    boundary_root = tmp_path / "boundary-checkpoints"
    per_step_root = tmp_path / "per-step-checkpoints"
    boundary_store = make_store(tmp_path / "boundary.sqlite")
    per_step_store = make_store(tmp_path / "per-step.sqlite")
    try:
        with deterministic_algorithms():
            boundary = SD15LoRATrainingService(
                boundary_store,
                boundary_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
            )
            initial_boundary, _ = boundary.create("cadence-proof", serialized_config())
            final_boundary = boundary.advance(initial_boundary, 3, "three steps").handle

            per_step = SD15LoRATrainingService(
                per_step_store,
                per_step_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
            )
            initial_per_step, _ = per_step.create(
                "cadence-proof", serialized_config(checkpointInterval=1)
            )
            final_per_step = per_step.advance(initial_per_step, 3, "three steps").handle

        assert manifest_step_cursors(per_step_root) == [0, 1, 2, 3]
        published = recovery_manifest_digests(per_step_store, initial_per_step.session_id)
        assert len(published) == 3
        assert [checkpoint_state(per_step_root, digest).step_cursor for digest in published] == [
            1,
            2,
            3,
        ]

        # The training state shards are byte-identical to the boundary-only
        # run; only the config shard differs because it carries the interval.
        boundary_shards = manifest_shards(boundary_root, final_boundary.checkpoint_manifest_digest)
        per_step_shards = manifest_shards(per_step_root, final_per_step.checkpoint_manifest_digest)
        for name in ("adapter", "optimizer", "rng", "trainerState"):
            assert boundary_shards[name] == per_step_shards[name]
        assert boundary_shards["config"] != per_step_shards["config"]
    finally:
        boundary_store.close()
        per_step_store.close()


def test_checkpoint_interval_writes_mid_advance_checkpoints(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = SD15LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create("interval-two", serialized_config(checkpointInterval=2))
        final = service.advance(initial, 5, "five steps").handle

        assert manifest_step_cursors(root) == [0, 2, 4, 5]
        published = recovery_manifest_digests(store, initial.session_id)
        assert [checkpoint_state(root, digest).step_cursor for digest in published] == [2, 4, 5]
        assert published[-1] == final.checkpoint_manifest_digest
        # Every recovery ledger row points at a written manifest and every
        # manifest beyond the initial one has exactly one ledger row.
        assert manifest_digests(root) == {initial.checkpoint_manifest_digest, *published}
    finally:
        store.close()


def test_resume_from_mid_advance_interval_checkpoint_is_bit_exact(tmp_path: Path) -> None:
    config = serialized_config(checkpointInterval=2)
    uninterrupted_root = tmp_path / "uninterrupted-checkpoints"
    resumed_root = tmp_path / "resumed-checkpoints"
    uninterrupted_store = make_store(tmp_path / "uninterrupted.sqlite")
    resumed_store = make_store(tmp_path / "resumed.sqlite")
    try:
        with deterministic_algorithms():
            uninterrupted = SD15LoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
            )
            initial_a, _ = uninterrupted.create("interval-resume", config)
            final_a = uninterrupted.advance(initial_a, 4, "same operation").handle
            assert manifest_step_cursors(uninterrupted_root) == [0, 2, 4]

            cancellation_armed = False
            paused: SD15LoRATrainingService

            def cancelled() -> bool:
                return cancellation_armed and paused.steps_run >= 3

            paused = SD15LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("interval-resume", config)
            cancellation_armed = True
            with pytest.raises(TrainingAdvancePaused, match="step 3 of 4"):
                paused.advance(initial_b, 4, "same operation")
            assert paused.steps_run == 3
            # Step 3 is off-interval, so the pause itself published a durable
            # checkpoint covering the uncheckpointed step.
            published_after_pause = recovery_manifest_digests(resumed_store, initial_b.session_id)
            assert [
                checkpoint_state(resumed_root, digest).step_cursor
                for digest in published_after_pause
            ] == [2, 3]

            successor = SD15LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
            )
            final_b = successor.advance(initial_b, 4, "same operation").handle
            assert successor.steps_run == 1
            assert manifest_step_cursors(resumed_root) == [0, 2, 3, 4]

        shards_a = manifest_shards(uninterrupted_root, final_a.checkpoint_manifest_digest)
        shards_b = manifest_shards(resumed_root, final_b.checkpoint_manifest_digest)
        assert shards_a == shards_b
        losses_a = recovery_losses(uninterrupted_root, uninterrupted_store, initial_a.session_id)
        losses_b = recovery_losses(resumed_root, resumed_store, initial_b.session_id)
        assert losses_a == (losses_b[0], losses_b[2])
    finally:
        uninterrupted_store.close()
        resumed_store.close()


def test_checkpoint_shards_are_digest_verified(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    store = ContentAddressedCheckpointStore(root)
    digest = store.write(
        session_id="a" * 64,
        config_digest="blake3:" + "b" * 64,
        extension_snapshot_digest="blake3:" + "c" * 64,
        parent_manifest_digest="",
        step_cursor=0,
        config={"family": "sd15"},
        adapter={"weight": torch.zeros(2, 2)},
        optimizer={},
        rng={"data": torch.zeros(1, dtype=torch.uint8)},
        data_cursor=0,
        loss=None,
    )
    assert digest.startswith("blake3:")
    manifest_path = root / "manifests" / f"{digest[7:]}.json"
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text("ascii")))
    shards = cast("dict[str, str]", manifest["shards"])
    adapter_path = root / "shards" / f"{shards['adapter'][7:]}.pt"
    adapter_path.write_bytes(adapter_path.read_bytes() + b"corrupt")
    with pytest.raises(CheckpointError, match="digest verification"):
        store.load(digest)


def test_committed_checkpoint_exports_deterministic_kohya_lora_and_round_trips(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = SD15LoRATrainingService(
            store,
            checkpoint_root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create("export-proof", serialized_config())
        final = service.advance(initial, 2, "train before export").handle
        first = checkpoint_root / "exports" / "first.safetensors"
        second = checkpoint_root / "exports" / "second.safetensors"
        first_path, first_digest = service.export_lora(
            final, json.dumps({"path": first.name}, sort_keys=True)
        )
        second_path, second_digest = service.export_lora(
            final, json.dumps({"path": second.name}, sort_keys=True)
        )

        assert Path(first_path).read_bytes() == Path(second_path).read_bytes()
        assert first_digest == second_digest == file_digest(first)
        source = load_safetensors_header(first)
        assert source.metadata() == {
            "dinkster_checkpoint_manifest_digest": final.checkpoint_manifest_digest,
            "dinkster_config_digest": final.config_digest,
            "dinkster_runtime_identity": "sd15-lora-torch/4",
            "dinkster_session_id": final.session_id,
            "dinkster_step_cursor": "2",
        }
        assert all(str(tmp_path) not in value for value in source.metadata().values())
        assert {source.entry(key).geometry.dtype.name for key in source.keys()} == {"float16"}

        state = checkpoint_state(checkpoint_root, final.checkpoint_manifest_digest)
        config = TrainingConfig.from_mapping(state.config)
        targets = resolve_lora_targets(model_factory(config), config.rank)
        model_keys = [f"diffusion_model.{key}" for key in model_factory(config).state_dict()]
        geometries = {key: source.entry(key).geometry for key in source.keys()}
        decoded = decode_lora(geometries, native_unet_key_map(model_keys))
        assert decoded.unmatched == ()
        assert decoded.diagnostics == ()
        assert len(decoded.patches) == len(targets)
        tensors = load_tensors(first)
        patch_set = build_patch_set(decoded.patches, tensors)
        weights = {
            f"diffusion_model.{target.module_path}.weight": torch.zeros(
                target.weight_shape, dtype=torch.float32
            )
            for target in targets
        }
        patch_weights(weights, patch_set)

        for target in targets:
            down = state.adapter[f"{target.target_id}.down"]
            up = state.adapter[f"{target.target_id}.up"]
            expected = (up @ down).reshape(target.weight_shape) * (config.alpha / config.rank)
            cast_expected = (up.half().float() @ down.half().float()).reshape(
                target.weight_shape
            ) * (torch.tensor(config.alpha).half().float() / config.rank)
            actual = weights[f"diffusion_model.{target.module_path}.weight"]
            torch.testing.assert_close(actual, cast_expected, rtol=0.0, atol=0.0)
            # Both matrix factors are stored in fp16. Under the pinned torch 2.13
            # validation runtime, the maximum master-to-export drift is 6.17e-5.
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.0,
                atol=7e-5,
            )

        for dtype, expected_dtype in (("bf16", "bfloat16"), ("fp32", "float32")):
            path = checkpoint_root / "exports" / f"export-{dtype}.safetensors"
            service.export_lora(
                final,
                json.dumps({"path": path.name, "dtype": dtype}, sort_keys=True),
            )
            typed_source = load_safetensors_header(path)
            assert {typed_source.entry(key).geometry.dtype.name for key in typed_source.keys()} == {
                expected_dtype
            }
    finally:
        store.close()


@pytest.mark.parametrize(
    "settings, message",
    [
        ("[]", "must be an object"),
        ('{"path":"lora.pt"}', "must end in .safetensors"),
        ('{"path":"lora.safetensors","dtype":"float16"}', "dtype must be"),
        ('{"path":"lora.safetensors","extra":true}', "unknown fields"),
        ('{"path":"../lora.safetensors"}', "must stay within"),
        ('{"path":"/tmp/lora.safetensors"}', "must be relative"),
        ('{"path":"C:\\\\tmp\\\\lora.safetensors"}', "must be relative"),
        ('{"path":"cadence/manual.safetensors"}', "reserved cadence directory"),
        ('{"path":"CADENCE/manual.safetensors"}', "reserved cadence directory"),
        ('{"path":"other/../cadence/manual.safetensors"}', "reserved cadence directory"),
        ('{"path":"other/../CADENCE/manual.safetensors"}', "reserved cadence directory"),
    ],
)
def test_export_settings_fail_closed(tmp_path: Path, settings: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LoraExportSettings.parse(settings, export_root=tmp_path)


def test_export_settings_reserve_cadence_symlink_target_but_allow_similar_filename(
    tmp_path: Path,
) -> None:
    export_root = tmp_path / "exports"
    automatic = export_root / "automatic"
    automatic.mkdir(parents=True)
    try:
        (export_root / "cadence").symlink_to(automatic, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="reserved cadence directory"):
        LoraExportSettings.parse(
            '{"path":"automatic/manual.safetensors"}',
            export_root=export_root,
        )
    allowed = LoraExportSettings.parse(
        '{"path":"cadence.safetensors"}',
        export_root=export_root,
    )
    assert allowed.path == export_root / "cadence.safetensors"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("optimizer", ["adamw", "factored-adamw"])
def test_cuda_bfloat16_step_keeps_float32_lora_masters(optimizer: str) -> None:
    config = TrainingConfig.from_mapping(
        config_mapping(device="cuda", base_dtype="bfloat16", optimizer=optimizer)
    )
    trainer = SD15LoRATrainer(config, model_factory(config), SyntheticBatches(config))
    loss = trainer.train_step()
    torch.cuda.synchronize()
    assert math.isfinite(loss)
    assert all(
        parameter.device.type == "cuda"
        and parameter.dtype == torch.float32
        and parameter.requires_grad
        for parameter in trainer.attachment.parameters()
    )
    for name, parameter in trainer.model.named_parameters():
        assert not parameter.requires_grad, name
        assert parameter.dtype == torch.bfloat16, name

    replacement = SD15LoRATrainer(config, model_factory(config), SyntheticBatches(config))
    replacement.restore(
        adapter=trainer.attachment.state_dict(),
        optimizer=trainer.optimizer_state_dict(),
        rng=trainer.rng_state_dict(),
        step_cursor=trainer.step_cursor,
        data_cursor=trainer.data_cursor,
        loss=trainer.last_loss,
    )
    assert math.isfinite(replacement.train_step())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_image_caption_dataset_precomputes_on_cuda_and_serves_cuda_batches(
    tmp_path: Path,
) -> None:
    config = image_dataset_config(tmp_path, device="cuda")
    source = image_data_source_factory(config)
    batch = source.batch(
        0,
        generator=torch.Generator(device="cuda").manual_seed(1),
        device=torch.device("cuda"),
    )
    torch.cuda.synchronize()
    assert batch.latents.device.type == "cuda"
    assert batch.text_embeddings.device.type == "cuda"
    assert batch.latents.shape == config.latent_shape
    assert batch.text_embeddings.shape == config.context_shape
    assert not batch.latents.requires_grad
    assert not batch.text_embeddings.requires_grad
    assert not {"_vae", "_text_model", "_text_encoder"} & set(vars(source))
    latents = cast("tuple[torch.Tensor, ...]", object.__getattribute__(source, "_latents"))
    text_embeddings = cast(
        "tuple[torch.Tensor, ...]", object.__getattribute__(source, "_text_embeddings")
    )
    assert all(tensor.device.type == "cpu" for tensor in (*latents, *text_embeddings))
