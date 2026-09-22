"""Executable proofs for the native MiniMax H3 LoRA trainer."""

from __future__ import annotations

import contextlib
import json
import math
import zlib
from collections.abc import Generator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import dinkster_training_torch.export as training_export
import dinkster_training_torch.service as training_service
import dinkster_training_torch.trainer as training_trainer
import pytest
import torch
from dinkster_assets import digest_bytes
from dinkster_inference import (
    BFLOAT16,
    FLOAT32,
    MINIMAX_H3_SIGMAS,
    LatentStream,
    MultiStreamLatent,
    PatchTarget,
    decode_lora,
    load_safetensors_header,
    native_unet_key_map,
)
from dinkster_inference.minimax_h3_assembly import MiniMaxH3ModelAssemblyPlan
from dinkster_inference.minimax_h3_dit import (
    MiniMaxH3TimeEmbeddingKind,
    minimax_h3_dit_layout,
)
from dinkster_inference.weights import TensorGeometry, WeightEntry
from dinkster_inference_torch import (
    INITLESS,
    AssembledMiniMaxH3Model,
    Int8Linear,
    build_patch_set,
    enroll_assembled,
    load_tensors,
)
from dinkster_inference_torch import quant_linear as quant_linear_mod
from dinkster_inference_torch.attention import (
    BUILTIN_SDPA_PROVIDER,
    AttentionSelection,
    builtin_sdpa_kernel,
    select_attention,
)
from dinkster_inference_torch.minimax_h3_dit import (
    MiniMaxH3Attention,
    MiniMaxH3AttentionGeometry,
    MiniMaxH3AttentionProviderEvidence,
    MiniMaxH3DiTConditioning,
    MiniMaxH3ReferenceKind,
    MiniMaxH3ReferenceLatents,
    assemble_minimax_h3_dit,
)
from dinkster_protocol import TrainingEventName, TrainingJournalEvent
from dinkster_server import JournalStore, TrainingLineageConflict, TrainingSessionStore
from dinkster_training_torch import (
    MINIMAX_H3_TRAINING_RUNTIME_IDENTITY,
    MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST,
    CheckpointError,
    CheckpointState,
    ContentAddressedCheckpointStore,
    FilesystemMiniMaxH3PreparedBatchSource,
    MiniMaxH3LoRATrainer,
    MiniMaxH3LoRATrainingService,
    MiniMaxH3PreparedBatch,
    MiniMaxH3TrainingConfig,
    TrainingAdvancePaused,
    TrainingConfigError,
    default_minimax_h3_model_factory,
    minimax_h3_capability_report,
)
from dinkster_training_torch.attachment import resolve_lora_targets

_MASK32 = 0xFFFFFFFF
_DIT_IDENTITY = "native:dinkster.minimax_h3:" + "1" * 64
_CONDITIONER_IDENTITY = "native:dinkster.minimax_h3:" + "2" * 64


def config_mapping(
    *,
    device: str = "cpu",
    base_dtype: str = "float32",
    optimizer: str = "adamw",
    accumulation: int = 1,
    checkpointing: bool = True,
    role: str = "fl2va-dit",
    paging_fraction: float = 0.0,
) -> dict[str, object]:
    mapping: dict[str, object] = {
        "schemaVersion": 1,
        "family": "minimax-h3",
        "ditRole": role,
        "ditIdentity": _DIT_IDENTITY,
        "conditionerIdentity": _CONDITIONER_IDENTITY,
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
        "videoLatentShape": [1, 24, 1, 2, 2],
        "audioLatentShape": [1, 32, 2, 1],
        "conditionerShape": [1, 1, 5120],
    }
    if paging_fraction:
        mapping["hostLayerPagingFraction"] = paging_fraction
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
    return ((uniform - 0.5) * 0.08).reshape(shape)


class TinyMiniMaxH3Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_projection = INITLESS.linear(2, 2)
        self.access_count = 0

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.access_count += 1
        return self.video_projection(video), audio, context


class TinyMiniMaxH3TimeEmbedder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = INITLESS.linear(1, 2)
        self.proj_out = INITLESS.linear(2, 1)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.proj_out(torch.nn.functional.silu(self.proj_in(sigma)))


class TinyMiniMaxH3DiT(torch.nn.Module):
    """A shape-faithful multistream seam with small linear projections."""

    def __init__(self, time_embedding_kind: MiniMaxH3TimeEmbeddingKind = "curve") -> None:
        super().__init__()
        self.time_embedding_kind = time_embedding_kind
        self.video_projection = INITLESS.linear(2, 2)
        self.audio_projection = INITLESS.linear(1, 1)
        self.context_projection = INITLESS.linear(5120, 1)
        self.time_embedder = TinyMiniMaxH3TimeEmbedder() if time_embedding_kind == "mlp" else None
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(_fill_value(name, tuple(parameter.shape)))
        self.last_video: torch.Tensor | None = None
        self.last_audio: torch.Tensor | None = None
        self.last_context: torch.Tensor | None = None
        self.last_output_video: torch.Tensor | None = None
        self.last_output_audio: torch.Tensor | None = None
        self.last_sigma: float | None = None
        self.last_sigmas: object | None = None
        self.last_conditioning: MiniMaxH3DiTConditioning | None = None

    def forward(
        self,
        latent: MultiStreamLatent[torch.Tensor],
        sigma: float,
        context: torch.Tensor,
        *,
        conditioning: MiniMaxH3DiTConditioning,
        sigmas: object,
    ) -> MultiStreamLatent[torch.Tensor]:
        assert latent.roles == ("video", "audio")
        video = latent.by_role("video")
        audio = latent.by_role("audio")
        context_value = self._project_context(context)
        output_video = self.video_projection(video)
        output_audio = self.audio_projection(audio)
        if self.time_embedder is not None:
            sigma_value = torch.tensor([[sigma]], device=context.device, dtype=context.dtype)
            context_value = context_value + self.time_embedder(sigma_value).mean()
        output_video, output_audio, context_value = self._transform(
            output_video,
            output_audio,
            context_value,
        )
        output_video = output_video + context_value * 0.01 + sigma * 0.001
        output_audio = output_audio + context_value * 0.01 + sigma * 0.001
        self.last_video = video.detach().cpu().clone()
        self.last_audio = audio.detach().cpu().clone()
        self.last_context = context.detach().cpu().clone()
        self.last_output_video = output_video.detach().float().cpu().clone()
        self.last_output_audio = output_audio.detach().float().cpu().clone()
        self.last_sigma = sigma
        self.last_sigmas = sigmas
        self.last_conditioning = conditioning
        return MultiStreamLatent(
            (
                LatentStream("video", output_video),
                LatentStream("audio", output_audio),
            )
        )

    def _project_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_projection(context).mean().reshape(1)

    def _transform(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return video, audio, context


class PaddedTinyProjection(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aligned_in_features = ((in_features + 15) // 16) * 16
        aligned_out_features = ((out_features + 3) // 4) * 4
        self.linear = INITLESS.linear(self.aligned_in_features, aligned_out_features)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(input, (0, self.aligned_in_features - self.in_features))
        return self.linear(padded)[..., : self.out_features]


class FusedTinyMiniMaxH3DiT(TinyMiniMaxH3DiT):
    def __init__(self) -> None:
        super().__init__()
        self.video_projection = PaddedTinyProjection(2, 2)
        self.audio_projection = PaddedTinyProjection(1, 1)
        self.context_projection = PaddedTinyProjection(5120, 1)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(_fill_value(name, tuple(parameter.shape)))


class PagingTinyMiniMaxH3DiT(TinyMiniMaxH3DiT):
    def __init__(self) -> None:
        super().__init__()
        blocks = tuple(TinyMiniMaxH3Block() for _ in range(4))
        self.blocks = torch.nn.ModuleList(blocks)
        object.__setattr__(self, "_block_refs", blocks)
        with torch.no_grad():
            for name, parameter in self.blocks.named_parameters():
                parameter.copy_(_fill_value(f"blocks.{name}", tuple(parameter.shape)))

    def _transform(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            video, audio, context = block(video, audio, context)
        return video, audio, context

    @property
    def block_access_counts(self) -> tuple[int, ...]:
        blocks = cast("tuple[TinyMiniMaxH3Block, ...]", self._block_refs)
        return tuple(block.access_count for block in blocks)


class AttentionTinyMiniMaxH3DiT(TinyMiniMaxH3DiT):
    """A tiny DiT seam that runs the production H3 attention primitive."""

    def __init__(self) -> None:
        super().__init__()
        width = 256
        self.context_projection = INITLESS.linear(5120, width)
        self.token_refiner_attention = MiniMaxH3Attention(
            MiniMaxH3AttentionGeometry(width, 4, 64, 48),
            builtin_sdpa_kernel(),
            MiniMaxH3AttentionProviderEvidence(BUILTIN_SDPA_PROVIDER, str(torch.__version__)),
            operations=INITLESS,
        )
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                parameter.copy_(_fill_value(name, tuple(parameter.shape)))
        self.last_model_context_dtype: torch.dtype | None = None
        self.last_attention_hidden_dtype: torch.dtype | None = None
        self.last_autocast_enabled: bool | None = None

    def _project_context(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.context_projection(context)
        self.last_model_context_dtype = context.dtype
        self.last_attention_hidden_dtype = hidden.dtype
        self.last_autocast_enabled = torch.is_autocast_enabled(context.device.type)
        return self.token_refiner_attention(hidden).mean().reshape(1)


def model_factory(config: MiniMaxH3TrainingConfig) -> TinyMiniMaxH3DiT:
    dtype = torch.float32 if config.base_dtype == "float32" else torch.bfloat16
    return TinyMiniMaxH3DiT().to(dtype=dtype)


def fused_model_factory(config: MiniMaxH3TrainingConfig) -> FusedTinyMiniMaxH3DiT:
    dtype = torch.float32 if config.base_dtype == "float32" else torch.bfloat16
    return FusedTinyMiniMaxH3DiT().to(dtype=dtype)


class MiniMaxH3LayoutHeaderSource:
    def __init__(self, path: Path, time_embedding_kind: MiniMaxH3TimeEmbeddingKind) -> None:
        self.path = path
        layout = minimax_h3_dit_layout(time_embedding_kind=time_embedding_kind)
        self.geometries = {
            key: TensorGeometry(
                shape,
                FLOAT32 if key in layout.fp32_storage_keys else BFLOAT16,
            )
            for key, shape in layout.keys.items()
        }

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}


def _artifact_config_mapping(
    tmp_path: Path, time_embedding_kind: MiniMaxH3TimeEmbeddingKind
) -> dict[str, object]:
    artifact = tmp_path / f"{time_embedding_kind}.safetensors"
    artifact.write_text(time_embedding_kind, encoding="ascii")
    mapping = config_mapping()
    mapping["ditState"] = {
        "path": str(artifact),
        "digest": "blake3:" + ("3" if time_embedding_kind == "curve" else "4") * 64,
        "size": artifact.stat().st_size,
    }
    return mapping


def _patch_artifact_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    def load_header(
        path: Path, *, asset_digest: str, asset_size: int
    ) -> MiniMaxH3LayoutHeaderSource:
        assert asset_digest.startswith("blake3:")
        assert asset_size == path.stat().st_size
        kind = cast("MiniMaxH3TimeEmbeddingKind", path.read_text(encoding="ascii"))
        return MiniMaxH3LayoutHeaderSource(path, kind)

    monkeypatch.setattr(training_trainer, "load_safetensors_header", load_header)


def artifact_model_factory(config: MiniMaxH3TrainingConfig) -> TinyMiniMaxH3DiT:
    assert config.dit_state_path is not None
    kind = cast(
        "MiniMaxH3TimeEmbeddingKind",
        Path(config.dit_state_path).read_text(encoding="ascii"),
    )
    dtype = torch.float32 if config.base_dtype == "float32" else torch.bfloat16
    return TinyMiniMaxH3DiT(kind).to(dtype=dtype)


def assemble_tiny_minimax_h3_dit(
    *,
    time_embedding_kind: MiniMaxH3TimeEmbeddingKind = "curve",
    attention_selection: AttentionSelection,
) -> TinyMiniMaxH3DiT:
    assert type(attention_selection) is AttentionSelection
    return TinyMiniMaxH3DiT(time_embedding_kind)


def paging_model_factory(config: MiniMaxH3TrainingConfig) -> PagingTinyMiniMaxH3DiT:
    dtype = torch.float32 if config.base_dtype == "float32" else torch.bfloat16
    return PagingTinyMiniMaxH3DiT().to(dtype=dtype)


def _quantize_linear(linear: torch.nn.Linear) -> Int8Linear:
    weight = linear.weight.detach().to(torch.float32)
    convrot = linear.in_features % 256 == 0
    scale = (
        (weight.abs().amax(dim=1, keepdim=True) if convrot else weight.abs().max())
        .div(127)
        .clamp_min(torch.finfo(torch.float32).tiny)
    )
    quantized = weight.div(scale).round().clamp(-128, 127).to(torch.int8)
    bias = cast("torch.nn.Parameter | None", linear.bias)
    replacement = Int8Linear(
        linear.in_features,
        linear.out_features,
        bias=bias is not None,
        compute_dtype=torch.bfloat16,
        convrot=convrot,
        convrot_groupsize=256,
    )
    state = {"weight": quantized, "weight_scale": scale}
    if bias is not None:
        state["bias"] = bias.detach().to(torch.bfloat16)
    replacement.load_state_dict(state, strict=True, assign=True)
    return replacement


def _quantize_model(model: torch.nn.Module) -> None:
    for path, module in tuple(model.named_modules()):
        if not isinstance(module, torch.nn.Linear):
            continue
        parent_path, _, name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, name, _quantize_linear(module))


def quantized_model_factory(config: MiniMaxH3TrainingConfig) -> TinyMiniMaxH3DiT:
    if not config.quantized_base:
        raise ValueError("quantized model factory requires quantizedBase")
    model = TinyMiniMaxH3DiT()
    _quantize_model(model)
    return model


def quantized_fused_model_factory(config: MiniMaxH3TrainingConfig) -> FusedTinyMiniMaxH3DiT:
    if not config.quantized_base:
        raise ValueError("quantized model factory requires quantizedBase")
    model = FusedTinyMiniMaxH3DiT()
    _quantize_model(model)
    return model


def quantized_attention_model_factory(
    config: MiniMaxH3TrainingConfig,
) -> AttentionTinyMiniMaxH3DiT:
    if not config.quantized_base:
        raise ValueError("quantized attention model factory requires quantizedBase")
    model = AttentionTinyMiniMaxH3DiT().to(dtype=torch.bfloat16)
    _quantize_model(model)
    return model


def quantized_paging_model_factory(
    config: MiniMaxH3TrainingConfig,
) -> PagingTinyMiniMaxH3DiT:
    if not config.quantized_base:
        raise ValueError("quantized paging model factory requires quantizedBase")
    model = PagingTinyMiniMaxH3DiT()
    _quantize_model(model)
    return model


class SyntheticH3Batches:
    def __init__(self, config: MiniMaxH3TrainingConfig) -> None:
        self._config = config

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxH3PreparedBatch:
        del generator, device
        references = (
            (
                MiniMaxH3ReferenceLatents(
                    MiniMaxH3ReferenceKind.IMAGE,
                    video=_fill_value(f"reference:{cursor}", self._config.video_latent_shape),
                ),
            )
            if self._config.dit_role == "ref2va-dit"
            else ()
        )
        return MiniMaxH3PreparedBatch(
            video_latents=_fill_value(f"video:{cursor}", self._config.video_latent_shape),
            audio_latents=_fill_value(f"audio:{cursor}", self._config.audio_latent_shape),
            conditioner_embeddings=_fill_value(
                f"conditioner:{cursor}", self._config.conditioner_shape
            ),
            conditioning=MiniMaxH3DiTConditioning(references=references, seed=cursor),
        )


class FixedH3Batches:
    def __init__(self, batch: MiniMaxH3PreparedBatch) -> None:
        self._batch = batch

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> MiniMaxH3PreparedBatch:
        del cursor, generator, device
        return self._batch


def data_source_factory(config: MiniMaxH3TrainingConfig) -> SyntheticH3Batches:
    return SyntheticH3Batches(config)


def make_store(path: Path) -> TrainingSessionStore:
    return TrainingSessionStore(JournalStore(path))


def checkpoint_state(root: Path, digest: str) -> CheckpointState:
    return ContentAddressedCheckpointStore(root).load(digest)


def assert_tree_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        left_mapping = cast("dict[object, object]", left)
        right_mapping = cast("dict[object, object]", right)
        for key in left_mapping:
            assert_tree_equal(left_mapping[key], right_mapping[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        right_values = cast("list[object] | tuple[object, ...]", right)
        for left_item, right_item in zip(left, right_values, strict=True):
            assert_tree_equal(left_item, right_item)
    else:
        assert left == right


def tree_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for child in cast("dict[object, object]", value).values()
            for tensor in tree_tensors(child)
        ]
    if isinstance(value, (list, tuple)):
        return [tensor for child in value for tensor in tree_tensors(child)]
    return []


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


@contextlib.contextmanager
def deterministic_algorithms() -> Generator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _fast_report(config: MiniMaxH3TrainingConfig) -> dict[str, object]:
    del config
    return {"dataset": {"errors": []}}


def test_config_composes_role_specific_execution_identity() -> None:
    config = MiniMaxH3TrainingConfig.from_mapping(config_mapping())
    composition = config.execution_composition

    assert composition.family_id == "dinkster.minimax_h3"
    assert composition.execution_identity.startswith("native:dinkster.minimax_h3:")
    assert [(item.role, item.identity) for item in composition.components] == [
        ("conditioner", _CONDITIONER_IDENTITY),
        ("fl2va_dit", _DIT_IDENTITY),
    ]
    assert config.identity_mapping() == config.to_mapping()
    assert config.checkpointing_mode == "wholeModel"
    assert "checkpointingMode" not in config.to_mapping()

    mapping = config_mapping()
    mapping["checkpointingMode"] = "blockReentrant"
    with pytest.raises(TrainingConfigError, match="must be 'wholeModel' for MiniMax H3"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)

    with pytest.raises(TrainingConfigError, match="must be 'wholeModel' for MiniMax H3"):
        replace(config, checkpointing_mode=cast('Literal["wholeModel"]', "blockReentrant"))

    mapping = config_mapping()
    mapping["conditionerIdentity"] = "native:dinkster.flux:" + "2" * 64
    with pytest.raises(TrainingConfigError, match="must be a three-part"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), True])
def test_host_layer_paging_config_refuses_invalid_fractions(value: object) -> None:
    mapping = config_mapping(device="cuda")
    mapping["hostLayerPagingFraction"] = value
    with pytest.raises(TrainingConfigError, match="hostLayerPagingFraction"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)


def test_host_layer_paging_config_refuses_cpu_devices() -> None:
    mapping = config_mapping()
    mapping["hostLayerPagingFraction"] = 0.5
    with pytest.raises(TrainingConfigError, match="requires a CUDA device"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_host_layer_paging_is_runtime_only_and_persisted(tmp_path: Path) -> None:
    mappings = [config_mapping(device="cuda", paging_fraction=value) for value in (0.0, 0.25, 1.0)]
    configs = [MiniMaxH3TrainingConfig.from_mapping(mapping) for mapping in mappings]
    digests = [
        training_service._config_digest(config)  # pyright: ignore[reportPrivateUsage]
        for config in configs
    ]
    assert digests[0] == digests[1] == digests[2]
    assert configs[0].identity_mapping() == configs[1].identity_mapping()
    assert configs[1].to_mapping()["hostLayerPagingFraction"] == 0.25

    lineage_store = make_store(tmp_path / "lineage.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(
            lineage_store,
            tmp_path / "lineage-checkpoints",
            model_factory=paging_model_factory,
            data_source_factory=data_source_factory,
        )
        handles = [
            service.create("paging-lineage", json.dumps(config.to_mapping()))[0]
            for config in configs
        ]
        assert handles[0] == handles[1] == handles[2]
    finally:
        lineage_store.close()

    persisted_store = make_store(tmp_path / "persisted.sqlite")
    persisted_root = tmp_path / "persisted-checkpoints"
    try:
        service = MiniMaxH3LoRATrainingService(
            persisted_store,
            persisted_root,
            model_factory=paging_model_factory,
            data_source_factory=data_source_factory,
        )
        handle, _ = service.create("paging-persisted", json.dumps(configs[1].to_mapping()))
        state = checkpoint_state(persisted_root, handle.checkpoint_manifest_digest)
        assert state.config["hostLayerPagingFraction"] == 0.25
    finally:
        persisted_store.close()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_quantized_base_is_explicit_validated_and_identity_bearing(tmp_path: Path) -> None:
    base = MiniMaxH3TrainingConfig.from_mapping(config_mapping(base_dtype="bfloat16"))
    mapping = config_mapping(base_dtype="bfloat16")
    mapping["quantizedBase"] = True
    quantized = MiniMaxH3TrainingConfig.from_mapping(mapping)
    fused_mapping = config_mapping(device="cuda", base_dtype="bfloat16")
    fused_mapping.update({"quantizedBase": True, "int8BaseForward": "fused"})
    fused = MiniMaxH3TrainingConfig.from_mapping(fused_mapping)

    assert not base.quantized_base
    assert "quantizedBase" not in base.to_mapping()
    assert base.int8_base_forward == "dequantize"
    assert quantized.quantized_base
    assert quantized.to_mapping()["quantizedBase"] is True
    assert fused.int8_base_forward == "fused"
    assert fused.to_mapping()["int8BaseForward"] == "fused"
    assert training_service._config_digest(base) != training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
        quantized
    )
    assert training_service._config_digest(quantized) != training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
        fused
    )

    mapping["baseDtype"] = "float32"
    with pytest.raises(TrainingConfigError, match="requires baseDtype 'bfloat16'"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)
    mapping["baseDtype"] = "bfloat16"
    mapping["quantizedBase"] = "true"
    with pytest.raises(TrainingConfigError, match="must be a boolean"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)
    mapping = config_mapping(device="cuda", base_dtype="bfloat16")
    mapping["int8BaseForward"] = "fused"
    with pytest.raises(TrainingConfigError, match="requires quantizedBase"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)
    mapping["quantizedBase"] = True
    mapping["int8BaseForward"] = "unknown"
    with pytest.raises(TrainingConfigError, match="must be 'dequantize' or 'fused'"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)
    mapping["int8BaseForward"] = "fused"
    mapping["device"] = "cpu"
    with pytest.raises(TrainingConfigError, match="requires a CUDA device"):
        MiniMaxH3TrainingConfig.from_mapping(mapping)

    source = tmp_path / "dit.safetensors"
    mapping = config_mapping(base_dtype="bfloat16")
    mapping["quantizedBase"] = True
    mapping["ditState"] = {
        "path": str(source),
        "digest": "blake3:" + "3" * 64,
        "size": 123,
    }
    first = MiniMaxH3TrainingConfig.from_mapping(mapping)
    cast("dict[str, object]", mapping["ditState"])["digest"] = "blake3:" + "4" * 64
    second = MiniMaxH3TrainingConfig.from_mapping(mapping)
    assert training_service._config_digest(first) != training_service._config_digest(  # pyright: ignore[reportPrivateUsage]
        second
    )


def test_attachment_freezes_h3_base_and_uses_float32_kaiming_masters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapped: list[torch.nn.Module] = []
    monkeypatch.setattr(training_trainer, "_checkpoint_sd_unet_blocks", wrapped.append)
    config = MiniMaxH3TrainingConfig.from_mapping(config_mapping())
    trainer = MiniMaxH3LoRATrainer(config, model_factory(config), SyntheticH3Batches(config))

    assert wrapped == []
    assert len(trainer.attachment.targets) == 3
    assert all(
        target.target_id.startswith("minimax-h3/dit/") for target in trainer.attachment.targets
    )
    assert all(not parameter.requires_grad for parameter in trainer.model.parameters())
    named = trainer.attachment.named_parameters()
    assert named
    for name, parameter in named:
        assert parameter.dtype == torch.float32 and parameter.requires_grad
        if name.endswith(".up"):
            assert torch.count_nonzero(parameter) == 0
        else:
            target_id = name.removesuffix(".down")
            target = next(
                item for item in trainer.attachment.targets if item.target_id == target_id
            )
            fan_in = math.prod(target.weight_shape[1:])
            assert parameter.abs().max().item() <= 1.0 / math.sqrt(fan_in)


@pytest.mark.parametrize(
    ("device", "int8_base_forward"),
    [
        ("cpu", "dequantize"),
        pytest.param(
            "cuda",
            "dequantize",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
        pytest.param(
            "cuda",
            "fused",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
    ],
)
def test_int8_base_trains_lora_without_mutating_storage(
    device: str,
    int8_base_forward: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = config_mapping(device=device, base_dtype="bfloat16", checkpointing=False)
    mapping["quantizedBase"] = True
    if int8_base_forward == "fused":
        mapping["int8BaseForward"] = "fused"
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    actual_routes: dict[int, set[str]] = {}
    if int8_base_forward == "fused":
        model = quantized_fused_model_factory(config)
        float_model = fused_model_factory(config)
        int8_linear = quant_linear_mod._int8_linear  # pyright: ignore[reportPrivateUsage]
        dequantize_int8 = quant_linear_mod._dequantize_int8  # pyright: ignore[reportPrivateUsage]

        def record_fused_forward(
            input: torch.Tensor,
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.Tensor | None,
            *,
            out_dtype: torch.dtype,
            convrot: bool,
            convrot_groupsize: int,
            input_act: Literal["gelu_tanh", "swiglu"] | None = None,
        ) -> torch.Tensor:
            actual_routes.setdefault(id(weight), set()).add("fused")
            return int8_linear(
                input,
                weight,
                weight_scale,
                bias,
                out_dtype=out_dtype,
                convrot=convrot,
                convrot_groupsize=convrot_groupsize,
                input_act=input_act,
            )

        def record_dequantized_forward(
            weight: torch.Tensor,
            weight_scale: torch.Tensor,
            *,
            dtype: torch.dtype,
            convrot: bool,
            convrot_groupsize: int,
        ) -> torch.Tensor:
            actual_routes.setdefault(id(weight), set()).add("dequantize")
            return dequantize_int8(
                weight,
                weight_scale,
                dtype=dtype,
                convrot=convrot,
                convrot_groupsize=convrot_groupsize,
            )

        monkeypatch.setattr(quant_linear_mod, "_int8_linear", record_fused_forward)
        monkeypatch.setattr(
            quant_linear_mod,
            "_dequantize_int8",
            record_dequantized_forward,
        )
    else:
        model = quantized_model_factory(config)
        float_model = model_factory(config)
    float_targets = resolve_lora_targets(float_model, config.rank, family="minimax-h3")
    int8_targets = resolve_lora_targets(model, config.rank, family="minimax-h3")
    assert int8_targets == float_targets
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    try:
        trainer = MiniMaxH3LoRATrainer(config, model, SyntheticH3Batches(config))
        int8_layers = tuple(
            module for module in trainer.model.modules() if isinstance(module, Int8Linear)
        )
        assert len(int8_layers) == len(int8_targets)
        if int8_base_forward == "fused":
            assert all(
                layer.fused_training
                and not layer.full_precision_matmul
                and layer.in_features % 16 == 0
                and (not layer.convrot or layer.in_features % layer.convrot_groupsize == 0)
                for layer in int8_layers
            )
        else:
            assert all(
                not layer.fused_training and layer.full_precision_matmul for layer in int8_layers
            )
        trainer.train_step()
        trainer.train_step()
        if int8_base_forward == "fused":
            assert actual_routes == {id(layer.weight): {"fused"} for layer in int8_layers}

        gradients = dict(trainer.attachment.named_parameters())
        assert all(parameter.grad is not None for parameter in gradients.values())
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for name, parameter in gradients.items()
            if name.endswith(".down")
        )
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for name, parameter in gradients.items()
            if name.endswith(".up")
        )
        assert all(not parameter.requires_grad for parameter in trainer.model.parameters())
        after = trainer.model.state_dict()
        assert after.keys() == frozen.keys()
        assert all(torch.equal(after[name].detach().cpu(), value) for name, value in frozen.items())
    finally:
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def test_fused_int8_base_refuses_ineligible_projection() -> None:
    mapping = config_mapping(device="cuda", base_dtype="bfloat16", checkpointing=False)
    mapping.update({"quantizedBase": True, "int8BaseForward": "fused"})
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)

    with pytest.raises(ValueError, match="input features divisible by 16, got 2"):
        training_trainer._configure_minimax_h3_base(  # pyright: ignore[reportPrivateUsage]
            quantized_model_factory(config),
            quantized=True,
            int8_base_forward="fused",
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
def test_bfloat16_h3_train_step_matches_inference_dtype_policy(device: str) -> None:
    mapping = config_mapping(device=device, base_dtype="bfloat16", checkpointing=False)
    mapping["quantizedBase"] = True
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    model = quantized_attention_model_factory(config)
    qkv = model.token_refiner_attention.qkv_proj
    assert isinstance(qkv, Int8Linear) and qkv.convrot

    try:
        trainer = MiniMaxH3LoRATrainer(config, model, SyntheticH3Batches(config))
        assert math.isfinite(trainer.train_step())
        assert model.last_model_context_dtype is torch.bfloat16
        assert model.last_attention_hidden_dtype is torch.bfloat16
        assert model.last_autocast_enabled is False
        assert model.last_video is not None and model.last_video.dtype is torch.bfloat16
        assert model.last_audio is not None and model.last_audio.dtype is torch.bfloat16
    finally:
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_cuda_host_layer_paging_is_bit_identical_to_resident_training(fraction: float) -> None:
    resident_config = MiniMaxH3TrainingConfig.from_mapping(
        config_mapping(device="cuda", base_dtype="bfloat16")
    )
    paged_config = MiniMaxH3TrainingConfig.from_mapping(
        config_mapping(device="cuda", base_dtype="bfloat16", paging_fraction=fraction)
    )
    resident_model = paging_model_factory(resident_config)
    paged_model = paging_model_factory(paged_config)
    resident = MiniMaxH3LoRATrainer(
        resident_config,
        resident_model,
        SyntheticH3Batches(resident_config),
    )
    paged = MiniMaxH3LoRATrainer(
        paged_config,
        paged_model,
        SyntheticH3Batches(paged_config),
    )
    assert paged.layer_pager is not None
    frozen = {
        name: value.detach().cpu().clone() for name, value in paged.model.state_dict().items()
    }

    try:
        with deterministic_algorithms():
            resident_losses = tuple(resident.train_step() for _ in range(3))
            paged_losses = tuple(paged.train_step() for _ in range(3))

        assert resident_losses == paged_losses
        assert_tree_equal(resident.attachment.state_dict(), paged.attachment.state_dict())
        assert_tree_equal(resident.optimizer_state_dict(), paged.optimizer_state_dict())
        assert_tree_equal(resident.rng_state_dict(), paged.rng_state_dict())
        assert paged.layer_pager.host_resident_layer_count == math.ceil(4 * fraction)
        assert paged.layer_pager.device_resident_layer_count == 0
        assert paged.layer_pager.pinned_host_bytes > 0
        assert all(parameter.device.type == "cuda" for parameter in paged.attachment.parameters())
        assert paged_model.block_access_counts == (6, 6, 6, 6)
        after = paged.model.state_dict()
        assert after.keys() == frozen.keys()
        assert all(torch.equal(after[name].detach().cpu(), value) for name, value in frozen.items())
    finally:
        del resident, paged, resident_model, paged_model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("checkpointing", [False, True], ids=["paged-only", "all-wrapped"])
def test_cuda_h3_cadence_export_uses_keys_captured_before_paging_wrappers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checkpointing: bool,
) -> None:
    monkeypatch.setattr(training_service, "minimax_h3_capability_report", _fast_report)
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(
            store,
            root,
            model_factory=paging_model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create(
            f"paged-cadence-{checkpointing}",
            serialized_config(
                device="cuda",
                baseDtype="bfloat16",
                hostLayerPagingFraction=0.5,
                gradientCheckpointing=checkpointing,
                loraExportInterval=1,
            ),
        )
        final = service.advance(initial, 1, "export paged adapter").handle
        cadence = (
            root / "exports" / "cadence" / initial.session_id / "step-000000000001.safetensors"
        )
        assert cadence.is_file()
        source = load_safetensors_header(cadence)
        assert source.metadata()["dinkster_step_cursor"] == "1"
        assert source.keys()
        service.complete(final)
    finally:
        store.close()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "int8"])
def test_cuda_host_layer_paging_trains_both_h3_base_forms_without_mutation(
    quantized: bool,
) -> None:
    mapping = config_mapping(device="cuda", base_dtype="bfloat16", paging_fraction=0.5)
    mapping["quantizedBase"] = quantized
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)
    factory = quantized_paging_model_factory if quantized else paging_model_factory
    model = factory(config)
    trainer = MiniMaxH3LoRATrainer(config, model, SyntheticH3Batches(config))
    assert trainer.layer_pager is not None
    frozen = {
        name: value.detach().cpu().clone() for name, value in trainer.model.state_dict().items()
    }

    try:
        trainer.train_step()
        trainer.train_step()
        assert trainer.layer_pager.host_resident_layer_count == 2
        assert trainer.layer_pager.device_resident_layer_count == 0
        assert model.block_access_counts == (4, 4, 4, 4)
        after = trainer.model.state_dict()
        assert after.keys() == frozen.keys()
        assert all(torch.equal(after[name].detach().cpu(), value) for name, value in frozen.items())
        if quantized:
            layers = tuple(
                module for module in trainer.model.modules() if isinstance(module, Int8Linear)
            )
            assert layers
            assert all(layer.weight.dtype == torch.int8 for layer in layers)
            assert all(layer.weight_scale.dtype == torch.float32 for layer in layers)
    finally:
        del trainer, model
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_flow_objective_matches_h3_multistream_sampling_coordinates() -> None:
    config = MiniMaxH3TrainingConfig.from_mapping(config_mapping(checkpointing=False))
    video = torch.full(config.video_latent_shape, 0.25)
    audio = torch.full(config.audio_latent_shape, -0.5)
    video_noise = torch.full(config.video_latent_shape, 0.75)
    audio_noise = torch.full(config.audio_latent_shape, 0.125)
    context = torch.full(config.conditioner_shape, 0.05)
    conditioning = MiniMaxH3DiTConditioning(frame_count=1, seed=99)
    batch = MiniMaxH3PreparedBatch(
        video,
        audio,
        context,
        conditioning,
        sigma_indices=torch.tensor([3]),
        video_noise=video_noise,
        audio_noise=audio_noise,
    )
    model = model_factory(config)
    trainer = MiniMaxH3LoRATrainer(config, model, FixedH3Batches(batch))

    loss = trainer.train_step()
    table = MINIMAX_H3_SIGMAS.table
    assert table is not None
    sigma = table[3]
    scaled_audio = audio * MINIMAX_H3_SIGMAS.audio_scale
    expected_video = (1.0 - sigma) * video + sigma * video_noise
    expected_audio = (1.0 - sigma) * scaled_audio + sigma * audio_noise
    assert model.last_sigma == sigma
    assert model.last_sigmas is MINIMAX_H3_SIGMAS
    assert model.last_conditioning == conditioning
    assert model.last_video is not None and torch.equal(model.last_video, expected_video)
    assert model.last_audio is not None and torch.equal(model.last_audio, expected_audio)
    assert model.last_context is not None and torch.equal(model.last_context, context)
    assert model.last_output_video is not None and model.last_output_audio is not None
    target_video = video_noise - video
    target_audio = audio_noise - scaled_audio
    expected_loss = (model.last_output_video - target_video).square().sum()
    expected_loss += (model.last_output_audio - target_audio).square().sum()
    expected_loss /= target_video.numel() + target_audio.numel()
    assert loss == expected_loss.item()


def test_filesystem_batches_restore_the_complete_conditioning_carrier(tmp_path: Path) -> None:
    config = MiniMaxH3TrainingConfig.from_mapping(config_mapping(role="ref2va-dit"))
    root = tmp_path / "prepared"
    root.mkdir()
    keyframe = torch.full(config.video_latent_shape, 0.5)
    reference_audio = torch.full(config.audio_latent_shape, -0.25)
    torch.save(
        {
            "video_latents": torch.zeros(config.video_latent_shape),
            "audio_latents": torch.ones(config.audio_latent_shape),
            "conditioner_embeddings": torch.zeros(config.conditioner_shape),
            "conditioning": {
                "text_token_tags": torch.tensor([[1]], dtype=torch.int64),
                "references": [{"kind": "audio", "video": None, "audio": reference_audio}],
                "frame_count": 1,
                "visual_noise_timestep": 0.9,
                "audio_noise_timestep": 0.8,
                "seed": 7,
            },
            "sigma_indices": torch.tensor([2]),
            "video_noise": torch.ones(config.video_latent_shape),
            "audio_noise": torch.zeros(config.audio_latent_shape),
        },
        root / "000000000000.pt",
    )
    torch.save(
        {
            "video_latents": torch.zeros(config.video_latent_shape),
            "audio_latents": torch.ones(config.audio_latent_shape),
            "conditioner_embeddings": torch.zeros(config.conditioner_shape),
            "conditioning": {
                "keyframes": [{"resolved_frame_index": 0, "video": keyframe}],
                "frame_count": 1,
            },
        },
        root / "000000000001.pt",
    )

    source = FilesystemMiniMaxH3PreparedBatchSource(root)
    batch = source.batch(
        0,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    assert batch.conditioning.frame_count == 1
    assert batch.conditioning.seed == 7
    assert batch.conditioning.text_token_tags is not None
    assert not batch.conditioning.keyframes
    assert batch.conditioning.references[0].kind is MiniMaxH3ReferenceKind.AUDIO
    assert batch.conditioning.references[0].audio is not None
    assert torch.equal(batch.conditioning.references[0].audio, reference_audio)
    keyframe_batch = source.batch(
        1,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    assert not keyframe_batch.conditioning.references
    assert torch.equal(keyframe_batch.conditioning.keyframes[0].video, keyframe)

    trainer = MiniMaxH3LoRATrainer(config, model_factory(config), FixedH3Batches(batch))
    assert math.isfinite(trainer.train_step())


@pytest.mark.parametrize(
    ("base_dtype", "quantized"),
    [("float32", False), ("bfloat16", False), ("bfloat16", True)],
    ids=["fp32", "bf16", "int8"],
)
@pytest.mark.parametrize("optimizer", ["adamw", "factored-adamw"])
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
def test_resume_and_safe_point_cancellation_are_bit_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer: str,
    base_dtype: str,
    quantized: bool,
    device: str,
) -> None:
    monkeypatch.setattr(training_service, "minimax_h3_capability_report", _fast_report)
    # checkpointInterval=1 keeps the per-step recovery stream this test compares.
    config = serialized_config(
        optimizer=optimizer,
        baseDtype=base_dtype,
        quantizedBase=quantized,
        device=device,
        gradientAccumulationSteps=2,
        checkpointInterval=1,
    )
    selected_model_factory = quantized_model_factory if quantized else model_factory
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted_store = make_store(tmp_path / "uninterrupted.sqlite")
    resumed_store = make_store(tmp_path / "resumed.sqlite")
    try:
        with deterministic_algorithms():
            uninterrupted = MiniMaxH3LoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=selected_model_factory,
                data_source_factory=data_source_factory,
            )
            initial_a, _ = uninterrupted.create("h3-resume", config)
            final_a = uninterrupted.advance(initial_a, 2, "same operation").handle

            armed = False
            paused: MiniMaxH3LoRATrainingService

            def cancelled() -> bool:
                return armed and paused.steps_run >= 1

            paused = MiniMaxH3LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=selected_model_factory,
                data_source_factory=data_source_factory,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("h3-resume", config)
            armed = True
            with pytest.raises(TrainingAdvancePaused, match="step 1 of 2"):
                paused.advance(initial_b, 2, "same operation")
            successor = MiniMaxH3LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=selected_model_factory,
                data_source_factory=data_source_factory,
            )
            final_b = successor.advance(initial_b, 2, "same operation").handle

        state_a = checkpoint_state(uninterrupted_root, final_a.checkpoint_manifest_digest)
        state_b = checkpoint_state(resumed_root, final_b.checkpoint_manifest_digest)
        assert state_a.step_cursor == state_b.step_cursor == 2
        assert state_a.data_cursor == state_b.data_cursor == 4
        assert state_a.loss == state_b.loss
        assert recovery_losses(
            uninterrupted_root, uninterrupted_store, initial_a.session_id
        ) == recovery_losses(resumed_root, resumed_store, initial_b.session_id)
        assert_tree_equal(state_a.adapter, state_b.adapter)
        assert_tree_equal(state_a.optimizer, state_b.optimizer)
        assert_tree_equal(state_a.rng, state_b.rng)
        assert all(value.dtype == torch.float32 for value in state_b.adapter.values())
        assert tree_tensors(state_b.optimizer)
        assert all(value.dtype == torch.float32 for value in tree_tensors(state_b.optimizer))

        names = [
            record.name
            for record in resumed_store.read_events(
                initial_b.session_id, after=0, limit=100
            ).records
        ]
        assert names.count(TrainingEventName.ADVANCE_PAUSED) == 1
        assert names.count(TrainingEventName.ADVANCE_RESUMED) == 1
        assert names.count(TrainingEventName.ADVANCE_COMMITTED) == 1
        before_replay = successor.steps_run
        replay = successor.advance(initial_b, 2, "same operation")
        assert replay.replayed and replay.handle == final_b
        assert successor.steps_run == before_replay
    finally:
        uninterrupted_store.close()
        resumed_store.close()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_host_layer_paging_resume_is_bit_exact_across_fresh_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(training_service, "minimax_h3_capability_report", _fast_report)
    # checkpointInterval=1 keeps the per-step recovery stream this test compares.
    config = serialized_config(
        device="cuda",
        baseDtype="bfloat16",
        hostLayerPagingFraction=0.75,
        gradientAccumulationSteps=2,
        checkpointInterval=1,
    )
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted_store = make_store(tmp_path / "uninterrupted.sqlite")
    resumed_store = make_store(tmp_path / "resumed.sqlite")
    try:
        with deterministic_algorithms():
            uninterrupted = MiniMaxH3LoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=paging_model_factory,
                data_source_factory=data_source_factory,
            )
            initial_a, _ = uninterrupted.create("paged-resume", config)
            final_a = uninterrupted.advance(initial_a, 2, "same operation").handle

            armed = False
            paused: MiniMaxH3LoRATrainingService

            def cancelled() -> bool:
                return armed and paused.steps_run >= 1

            paused = MiniMaxH3LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=paging_model_factory,
                data_source_factory=data_source_factory,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("paged-resume", config)
            persisted = checkpoint_state(resumed_root, initial_b.checkpoint_manifest_digest)
            assert persisted.config["hostLayerPagingFraction"] == 0.75
            armed = True
            with pytest.raises(TrainingAdvancePaused, match="step 1 of 2"):
                paused.advance(initial_b, 2, "same operation")
            successor = MiniMaxH3LoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=paging_model_factory,
                data_source_factory=data_source_factory,
            )
            final_b = successor.advance(initial_b, 2, "same operation").handle

        state_a = checkpoint_state(uninterrupted_root, final_a.checkpoint_manifest_digest)
        state_b = checkpoint_state(resumed_root, final_b.checkpoint_manifest_digest)
        assert state_a.loss == state_b.loss
        assert recovery_losses(
            uninterrupted_root, uninterrupted_store, initial_a.session_id
        ) == recovery_losses(resumed_root, resumed_store, initial_b.session_id)
        assert_tree_equal(state_a.adapter, state_b.adapter)
        assert_tree_equal(state_a.optimizer, state_b.optimizer)
        assert_tree_equal(state_a.rng, state_b.rng)
    finally:
        uninterrupted_store.close()
        resumed_store.close()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def test_loading_refuses_wrong_structure_and_unselected_quantized_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dit.safetensors"
    source.write_bytes(b"")
    mapping = config_mapping()
    mapping["ditState"] = {
        "path": str(source),
        "digest": "blake3:" + "3" * 64,
        "size": 0,
    }
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)

    fake_plan = cast(
        "MiniMaxH3ModelAssemblyPlan",
        SimpleNamespace(diffusion=SimpleNamespace(quant={"layer": object()})),
    )

    def load_header(path: Path, *, asset_digest: str, asset_size: int) -> object:
        del path, asset_digest, asset_size
        return object()

    def reject_structure(
        source_value: object, *, role: str, path: Path
    ) -> MiniMaxH3ModelAssemblyPlan:
        del source_value, role, path
        raise ValueError("geometry mismatch")

    monkeypatch.setattr(training_trainer, "load_safetensors_header", load_header)
    monkeypatch.setattr(training_trainer, "plan_minimax_h3_model_assembly", reject_structure)
    with pytest.raises(ValueError, match="geometry mismatch"):
        default_minimax_h3_model_factory(config)

    def plan_h3(source_value: object, *, role: str, path: Path) -> MiniMaxH3ModelAssemblyPlan:
        del source_value, role, path
        return fake_plan

    monkeypatch.setattr(training_trainer, "plan_minimax_h3_model_assembly", plan_h3)
    with pytest.raises(ValueError, match="quantized"):
        default_minimax_h3_model_factory(config)


def test_loading_accepts_only_selected_int8_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dit.safetensors"
    source.write_bytes(b"")
    mapping = config_mapping(base_dtype="bfloat16")
    mapping["quantizedBase"] = True
    mapping["ditState"] = {
        "path": str(source),
        "digest": "blake3:" + "3" * 64,
        "size": 0,
    }
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)

    def load_header(path: Path, *, asset_digest: str, asset_size: int) -> object:
        del path, asset_digest, asset_size
        return object()

    monkeypatch.setattr(training_trainer, "load_safetensors_header", load_header)

    def plan_with(format_name: str | None) -> MiniMaxH3ModelAssemblyPlan:
        quant = {} if format_name is None else {"projection": SimpleNamespace(format=format_name)}
        return cast(
            "MiniMaxH3ModelAssemblyPlan",
            SimpleNamespace(diffusion=SimpleNamespace(quant=quant)),
        )

    planned_format: str | None = "nvfp4"

    def plan_h3(source_value: object, *, role: str, path: Path) -> MiniMaxH3ModelAssemblyPlan:
        del source_value, role, path
        return plan_with(planned_format)

    monkeypatch.setattr(training_trainer, "plan_minimax_h3_model_assembly", plan_h3)
    with pytest.raises(ValueError, match="only INT8 tensorwise"):
        default_minimax_h3_model_factory(config)

    planned_format = None
    with pytest.raises(ValueError, match="ditState is unquantized"):
        default_minimax_h3_model_factory(config)

    model = quantized_model_factory(config)
    loaded = SimpleNamespace(
        model_role="fl2va-dit",
        assembled=SimpleNamespace(diffusion=model),
    )
    planned_format = "int8_tensorwise"
    loads: list[tuple[Path, str, torch.dtype, str]] = []

    def load_model(
        path: Path,
        *,
        asset: object,
        role: str,
        expected_identity: str,
        diffusion_dtype: torch.dtype,
        attention_backend: str,
    ) -> object:
        assert role == "fl2va-dit"
        del asset
        loads.append((path, expected_identity, diffusion_dtype, attention_backend))
        return loaded

    monkeypatch.setattr(training_trainer, "load_minimax_h3_model", load_model)
    result = default_minimax_h3_model_factory(config)
    assert result is model
    assert loads == [(source.resolve(), _DIT_IDENTITY, torch.bfloat16, "flux")]
    assert all(
        not module.fused_training and module.full_precision_matmul
        for module in result.modules()
        if isinstance(module, Int8Linear)
    )


def test_capability_report_and_dry_run_cover_h3_precision_and_memory(tmp_path: Path) -> None:
    config = MiniMaxH3TrainingConfig.from_mapping(config_mapping())
    report = minimax_h3_capability_report(config)

    assert report["trainer"] == MINIMAX_H3_TRAINING_RUNTIME_IDENTITY
    assert report["sessionExtensionSnapshotDigest"] == MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST
    targets = cast("list[dict[str, object]]", report["targets"])
    assert len(targets) == 264
    assert all(str(target["targetId"]).startswith("minimax-h3/dit/") for target in targets)
    counts = cast("dict[str, int]", report["parameterCounts"])
    assert counts == {
        "frozenBase": 33_107_157_888,
        "trainableLora": 19_728_128,
        "targetedBase": 33_101_709_312,
    }
    capabilities = cast("dict[str, object]", report["capabilities"])
    assert capabilities["preparedMultistreamBatches"] is True
    assert capabilities["loraExport"] is True
    assert capabilities["quantizedBase"] is True
    assert capabilities["hostRamLayerPaging"] is True
    assert report["layerPaging"] == {
        "enabled": False,
        "configuredHostResidentFraction": 0.0,
        "hostResidentFraction": 0.0,
        "hostResidentLayerCount": 0,
        "transformerLayerCount": 50,
        "forwardPrefetchDistance": 0,
        "backwardPrefetchDistance": 0,
    }
    precision = cast("dict[str, str]", report["precisionPlan"])
    assert precision == {
        "baseStorage": "float32",
        "forwardAutocast": "bfloat16",
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
        "flowObjective": "float32",
    }
    memory = cast("dict[str, object]", report["memoryLedger"])
    categories = cast("dict[str, int]", memory["categories"])
    assert memory["totalLowerBoundBytes"] == sum(categories.values())
    composition = cast("dict[str, object]", report["executionComposition"])
    assert composition["executionIdentity"] == config.execution_composition.execution_identity

    quantized_mapping = config_mapping(base_dtype="bfloat16")
    quantized_mapping["quantizedBase"] = True
    quantized = MiniMaxH3TrainingConfig.from_mapping(quantized_mapping)
    quantized_report = minimax_h3_capability_report(quantized)
    quantized_precision = cast("dict[str, str]", quantized_report["precisionPlan"])
    assert quantized_precision["baseStorage"] == "int8"
    assert quantized_precision["baseScales"] == "float32"
    assert quantized_precision["baseDequantization"] == "bfloat16 per projection"
    quantized_memory = cast("dict[str, object]", quantized_report["memoryLedger"])
    quantized_categories = cast("dict[str, int]", quantized_memory["categories"])
    assert quantized_categories["frozenBaseParameters"] == (
        counts["targetedBase"] + (counts["frozenBase"] - counts["targetedBase"]) * 2
    )
    assert quantized_categories["quantizationScalesLowerBound"] == len(targets) * 4
    assert quantized_memory["totalLowerBoundBytes"] == sum(quantized_categories.values())
    assert quantized_report["configDigest"] != report["configDigest"]

    fused_mapping = config_mapping(device="cuda", base_dtype="bfloat16")
    fused_mapping.update({"quantizedBase": True, "int8BaseForward": "fused"})
    fused = MiniMaxH3TrainingConfig.from_mapping(fused_mapping)
    if torch.cuda.is_available():
        fused_report = minimax_h3_capability_report(fused)
        fused_precision = cast("dict[str, str]", fused_report["precisionPlan"])
        assert fused_precision["baseForward"] == "fused W8A8 per projection"
        assert (
            fused_precision["baseBackwardDequantization"]
            == "bounded bfloat16 input-gradient chunks"
        )
        assert "baseDequantization" not in fused_precision
        assert fused_report["configDigest"] != quantized_report["configDigest"]

    if torch.cuda.is_available():
        paged_mapping = config_mapping(device="cuda", paging_fraction=0.5)
        paged = MiniMaxH3TrainingConfig.from_mapping(paged_mapping)
        paged_report = minimax_h3_capability_report(paged)
        paging = cast("dict[str, object]", paged_report["layerPaging"])
        assert paging == {
            "enabled": True,
            "configuredHostResidentFraction": 0.5,
            "hostResidentFraction": 0.5,
            "hostResidentLayerCount": 25,
            "transformerLayerCount": 50,
            "forwardPrefetchDistance": 1,
            "backwardPrefetchDistance": 1,
        }
        paged_memory = cast("dict[str, object]", paged_report["memoryLedger"])
        paged_categories = cast("dict[str, int]", paged_memory["categories"])
        assert paged_categories["frozenBaseParametersPinnedCpu"] > 0
        assert paged_categories["frozenBaseParametersDevice"] > 0
        assert (
            paged_memory["hostRamLowerBoundBytes"]
            == paged_categories["frozenBaseParametersPinnedCpu"]
        )
        assert cast("int", paged_memory["deviceMemoryLowerBoundBytes"]) < cast(
            "int", memory["deviceMemoryLowerBoundBytes"]
        )
        resident_cuda = MiniMaxH3TrainingConfig.from_mapping(config_mapping(device="cuda"))
        assert (
            paged_report["configDigest"]
            == minimax_h3_capability_report(resident_cuda)["configDigest"]
        )

    store = make_store(tmp_path / "training.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(store, tmp_path / "checkpoints")
        assert service.dry_run(serialized_config()) == report
    finally:
        store.close()


@pytest.mark.parametrize(
    ("time_embedding_kind", "expected_targets", "expected_counts"),
    [
        (
            "curve",
            264,
            {
                "frozenBase": 33_107_157_888,
                "trainableLora": 19_728_128,
                "targetedBase": 33_101_709_312,
            },
        ),
        (
            "mlp",
            266,
            {
                "frozenBase": 33_122_992_896,
                "trainableLora": 19_755_520,
                "targetedBase": 33_117_536_256,
            },
        ),
    ],
)
def test_h3_dry_run_uses_pinned_artifact_time_embedding_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    time_embedding_kind: MiniMaxH3TimeEmbeddingKind,
    expected_targets: int,
    expected_counts: dict[str, int],
) -> None:
    _patch_artifact_planning(monkeypatch)
    mapping = _artifact_config_mapping(tmp_path, time_embedding_kind)
    serialized = json.dumps(mapping, sort_keys=True)
    config = MiniMaxH3TrainingConfig.from_mapping(mapping)

    report = minimax_h3_capability_report(config)
    assert len(cast("list[object]", report["targets"])) == expected_targets
    assert report["parameterCounts"] == expected_counts
    memory = cast("dict[str, object]", report["memoryLedger"])
    categories = cast("dict[str, int]", memory["categories"])
    assert memory["totalLowerBoundBytes"] == sum(categories.values())

    store = make_store(tmp_path / "training.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(store, tmp_path / "checkpoints")
        assert service.dry_run(serialized) == report
    finally:
        store.close()


@pytest.mark.parametrize(
    ("time_embedding_kind", "expected_targets"), [("curve", 264), ("mlp", 266)]
)
def test_real_h3_dit_targets_match_native_lora_key_map(
    time_embedding_kind: MiniMaxH3TimeEmbeddingKind,
    expected_targets: int,
) -> None:
    with torch.device("meta"):
        model = assemble_minimax_h3_dit(
            time_embedding_kind=time_embedding_kind,
            attention_selection=select_attention("flux", "sdpa"),
        )
    targets = resolve_lora_targets(model, 2, family="minimax-h3")
    model_keys = tuple(f"diffusion_model.{key}" for key in model.state_dict())
    key_map = native_unet_key_map(model_keys)

    assert len(targets) == expected_targets
    assert all(
        key_map.get(f"diffusion_model.{target.module_path}")
        == f"diffusion_model.{target.module_path}.weight"
        for target in targets
    )


@pytest.mark.parametrize("time_embedding_kind", ["curve", "mlp"])
def test_h3_export_uses_pinned_artifact_layout_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    time_embedding_kind: MiniMaxH3TimeEmbeddingKind,
) -> None:
    _patch_artifact_planning(monkeypatch)
    monkeypatch.setattr(training_service, "assemble_minimax_h3_dit", assemble_tiny_minimax_h3_dit)
    monkeypatch.setattr(training_export, "assemble_minimax_h3_dit", assemble_tiny_minimax_h3_dit)
    mapping = _artifact_config_mapping(tmp_path, time_embedding_kind)
    mapping["loraExportInterval"] = 1
    serialized = json.dumps(mapping, sort_keys=True)
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(
            store,
            root,
            model_factory=artifact_model_factory,
            data_source_factory=data_source_factory,
        )
        expected_target_count = 3 if time_embedding_kind == "curve" else 5
        initial, _ = service.create(f"h3-{time_embedding_kind}-export", serialized)
        final = service.advance(initial, 2, "train adapter before export").handle
        cadence = (
            root / "exports" / "cadence" / initial.session_id / "step-000000000002.safetensors"
        )
        assert cadence.is_file()
        assert load_safetensors_header(cadence).metadata()["dinkster_step_cursor"] == "2"
        state = checkpoint_state(root, final.checkpoint_manifest_digest)
        first_path, first_digest = service.export_lora(final, '{"path":"first.safetensors"}')
        second_path, second_digest = service.export_lora(final, '{"path":"second.safetensors"}')
        first = Path(first_path)
        second = Path(second_path)
        assert first.read_bytes() == second.read_bytes()
        assert first_digest == second_digest

        config = MiniMaxH3TrainingConfig.from_mapping(state.config)
        native_model = artifact_model_factory(config)
        targets = resolve_lora_targets(native_model, config.rank, family="minimax-h3")
        assert len(targets) == expected_target_count
        source = load_safetensors_header(first)
        model_keys = tuple(f"diffusion_model.{key}" for key in native_model.state_dict())
        decoded = decode_lora(
            {key: source.entry(key).geometry for key in source.keys()},
            native_unet_key_map(model_keys),
        )
        assert decoded.unmatched == ()
        assert decoded.diagnostics == ()
        assert len(decoded.patches) == len(targets)

        tensors = load_tensors(first)
        routed = {
            PatchTarget(target.key.removeprefix("diffusion_model."), target.offset): spec
            for target, spec in decoded.patches.items()
        }
        strength = 0.25
        patch_set = build_patch_set(routed, tensors, strength=strength)
        applied_model = artifact_model_factory(config)
        applied_modules = dict(applied_model.named_modules())
        with torch.no_grad():
            for target in targets:
                cast("torch.nn.Linear", applied_modules[target.module_path]).weight.zero_()
        enrolled = enroll_assembled(
            AssembledMiniMaxH3Model(
                cast("object", applied_model),  # pyright: ignore[reportArgumentType]
                _component_compute_dtypes={"diffusion": torch.float32},
            ),
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            patch_sets={"diffusion": patch_set},
        )
        enrolled["diffusion"].partially_load(None)

        for target in targets:
            down = state.adapter[f"{target.target_id}.down"]
            up = state.adapter[f"{target.target_id}.up"]
            stem = f"diffusion_model.{target.module_path}"
            exported_down = tensors[f"{stem}.lora_A.weight"].to(torch.float32)
            exported_up = tensors[f"{stem}.lora_B.weight"].to(torch.float32)
            exported_alpha = float(tensors[f"{stem}.alpha"].item())
            expected_from_export = (
                strength * (exported_alpha / config.rank) * (exported_up @ exported_down)
            )
            actual = cast("torch.nn.Linear", applied_modules[target.module_path]).weight.detach()
            torch.testing.assert_close(actual, expected_from_export, rtol=0.0, atol=0.0)
            expected_from_masters = strength * (config.alpha / config.rank) * (up @ down)
            torch.testing.assert_close(actual, expected_from_masters, rtol=0.0, atol=1e-7)
    finally:
        store.close()


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "int8"])
def test_h3_export_is_deterministic_and_round_trips_native_precalculation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quantized: bool
) -> None:
    monkeypatch.setattr(training_service, "minimax_h3_capability_report", _fast_report)
    monkeypatch.setattr(training_export, "assemble_minimax_h3_dit", assemble_tiny_minimax_h3_dit)
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    selected_model_factory = quantized_model_factory if quantized else model_factory
    try:
        service = MiniMaxH3LoRATrainingService(
            store,
            root,
            model_factory=selected_model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create(
            "h3-export-proof",
            serialized_config(
                alpha=3.0,
                baseDtype="bfloat16",
                quantizedBase=quantized,
            ),
        )
        final = service.advance(initial, 2, "train adapter before export").handle
        state = checkpoint_state(root, final.checkpoint_manifest_digest)
        assert state.adapter
        assert state.optimizer
        assert state.loss is not None
        first_path, first_digest = service.export_lora(final, '{"path":"first.safetensors"}')
        second_path, second_digest = service.export_lora(final, '{"path":"second.safetensors"}')
        first = Path(first_path)
        second = Path(second_path)

        assert first.read_bytes() == second.read_bytes()
        assert first_digest == second_digest == digest_bytes(first.read_bytes())
        source = load_safetensors_header(first)
        assert source.metadata() == {
            "dinkster_checkpoint_manifest_digest": final.checkpoint_manifest_digest,
            "dinkster_config_digest": final.config_digest,
            "dinkster_runtime_identity": MINIMAX_H3_TRAINING_RUNTIME_IDENTITY,
            "dinkster_session_id": final.session_id,
            "dinkster_step_cursor": "2",
        }
        assert all(str(tmp_path) not in value for value in source.metadata().values())
        assert {source.entry(key).geometry.dtype.name for key in source.keys()} == {"float16"}

        config = MiniMaxH3TrainingConfig.from_mapping(state.config)
        targets = resolve_lora_targets(
            selected_model_factory(config), config.rank, family="minimax-h3"
        )
        bf16_targets = resolve_lora_targets(model_factory(config), config.rank, family="minimax-h3")
        assert targets == bf16_targets
        expected_keys = {
            f"diffusion_model.{target.module_path}.{suffix}"
            for target in targets
            for suffix in ("lora_A.weight", "lora_B.weight", "alpha")
        }
        assert set(source.keys()) == expected_keys
        model_keys = tuple(f"diffusion_model.{key}" for key in model_factory(config).state_dict())
        decoded = decode_lora(
            {key: source.entry(key).geometry for key in source.keys()},
            native_unet_key_map(model_keys),
        )
        assert decoded.unmatched == ()
        assert decoded.diagnostics == ()
        assert len(decoded.patches) == len(targets)
        assert all(getattr(spec, "variant", None) == "peft" for spec in decoded.patches.values())

        tensors = load_tensors(first)
        routed = {
            PatchTarget(target.key.removeprefix("diffusion_model."), target.offset): spec
            for target, spec in decoded.patches.items()
        }
        strength = 0.25
        patch_set = build_patch_set(routed, tensors, strength=strength)
        application_config = replace(config, base_dtype="float32", quantized_base=False)
        applied_model = model_factory(application_config)
        applied_modules = dict(applied_model.named_modules())
        with torch.no_grad():
            for target in targets:
                cast("torch.nn.Linear", applied_modules[target.module_path]).weight.zero_()
        enrolled = enroll_assembled(
            AssembledMiniMaxH3Model(
                cast("object", applied_model),  # pyright: ignore[reportArgumentType]
                _component_compute_dtypes={"diffusion": torch.float32},
            ),
            load_device=torch.device("cpu"),
            offload_device=torch.device("cpu"),
            patch_sets={"diffusion": patch_set},
        )
        enrolled["diffusion"].partially_load(None)

        for target in targets:
            down = state.adapter[f"{target.target_id}.down"]
            up = state.adapter[f"{target.target_id}.up"]
            stem = f"diffusion_model.{target.module_path}"
            exported_down = tensors[f"{stem}.lora_A.weight"].to(torch.float32)
            exported_up = tensors[f"{stem}.lora_B.weight"].to(torch.float32)
            exported_alpha = float(tensors[f"{stem}.alpha"].item())
            expected_from_export = (
                strength * (exported_alpha / config.rank) * (exported_up @ exported_down)
            )
            actual = cast("torch.nn.Linear", applied_modules[target.module_path]).weight.detach()
            torch.testing.assert_close(actual, expected_from_export, rtol=0.0, atol=0.0)
            expected_from_masters = strength * (config.alpha / config.rank) * (up @ down)
            # Both factors and alpha use fp16. Under pinned torch 2.13, this
            # proof's maximum master-to-export drift is 6.38e-8; the limit is 1e-7.
            torch.testing.assert_close(actual, expected_from_masters, rtol=0.0, atol=1e-7)

        for dtype, expected_dtype in (("bf16", "bfloat16"), ("fp32", "float32")):
            path = root / "exports" / f"h3-{dtype}.safetensors"
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


def test_h3_export_refuses_uncommitted_foreign_and_unmatched_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_service, "minimax_h3_capability_report", _fast_report)
    monkeypatch.setattr(training_export, "assemble_minimax_h3_dit", assemble_tiny_minimax_h3_dit)
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    foreign_store = make_store(tmp_path / "foreign.sqlite")
    try:
        service = MiniMaxH3LoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create("h3-export-refusals", serialized_config())
        final = service.advance(initial, 1, "train adapter").handle

        with pytest.raises(TrainingLineageConflict, match="committed checkpoint"):
            service.export_lora(
                replace(final, journal_seq=final.journal_seq + 1),
                '{"path":"uncommitted.safetensors"}',
            )
        foreign = MiniMaxH3LoRATrainingService(
            foreign_store,
            tmp_path / "foreign-checkpoints",
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        with pytest.raises(TrainingLineageConflict, match="unknown training session"):
            foreign.export_lora(final, '{"path":"foreign.safetensors"}')
        with pytest.raises(ValueError, match="dtype must be"):
            service.export_lora(
                final,
                '{"path":"unsupported.safetensors","dtype":"float16"}',
            )

        real_key_map = training_export.native_unet_key_map  # pyright: ignore[reportPrivateImportUsage]

        def incomplete_key_map(model_keys: tuple[str, ...]) -> dict[str, str]:
            key_map = real_key_map(model_keys)
            key_map.pop("diffusion_model.audio_projection")
            return key_map

        monkeypatch.setattr(training_export, "native_unet_key_map", incomplete_key_map)
        with pytest.raises(CheckpointError, match="do not match the native DiT state dict"):
            service.export_lora(final, '{"path":"unmatched.safetensors"}')
    finally:
        store.close()
        foreign_store.close()
