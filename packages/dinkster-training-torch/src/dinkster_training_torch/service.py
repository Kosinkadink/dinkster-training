"""Durable training services over the native LoRA trainer cores."""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from dinkster_api.v1 import TrainingSessionHandle, digest_bytes
from dinkster_inference import (
    Flux2AssemblyPlan,
    FluxAssemblyPlan,
    QwenImageAssemblyPlan,
    Wan21AssemblyPlan,
    load_safetensors_header,
)
from dinkster_inference_torch import (
    Flux,
    Ideogram4DiT,
    MiniMaxMusic3DiT,
    QwenImage,
    UNetModel,
    Wan21Model,
    assemble_minimax_h3_dit,
    select_attention,
)
from dinkster_nodes_training import AdvanceOutcome
from dinkster_server import TrainingLineageConflict, TrainingOperationRecord, TrainingSessionStore

from .attachment import TargetDescriptor, resolve_lora_targets
from .checkpoint import (
    CheckpointState,
    ContentAddressedCheckpointStore,
    blake3_digest,
    canonical_json,
)
from .config import (
    Flux2TrainingConfig,
    FluxTrainingConfig,
    Ideogram4TrainingConfig,
    MiniMaxH3TrainingConfig,
    MiniMaxMusic3TrainingConfig,
    QwenImageTrainingConfig,
    TrainingConfig,
    TrainingConfigError,
    WanTrainingConfig,
)
from .container import MINIMAX_H3_CONTAINER_DECODER_IDENTITY
from .data import inspect_encoder_memory, inspect_minimax_h3_encoder_memory
from .dataset import TrainingFamily, minimax_h3_audio_sample_count
from .distributed import FileRendezvousSettings, TorchDistributedRankContext, bootstrap_rank_context
from .encoded_cache import (
    encoded_cache_state,
    flux2_encoded_cache_state,
    flux_encoded_cache_state,
    ideogram4_encoded_cache_state,
    minimax_h3_encoded_cache_state,
    qwen_image_encoded_cache_state,
    wan_encoded_cache_state,
)
from .export import (
    LoraExportSettings,
    LoraExportSource,
    export_flux2_lora,
    export_flux_lora,
    export_ideogram4_lora,
    export_intermediate_lora,
    export_kohya_lora,
    export_minimax_h3_lora,
    export_minimax_music3_lora,
    export_qwen_image_lora,
    export_wan_lora,
)
from .minimax_music3_training import (
    MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION,
    MINIMAX_MUSIC3_DAV_ENCODER_CONTRACT,
    MINIMAX_MUSIC3_DAV_SOURCE_REVISION,
    MINIMAX_MUSIC3_FLOW_OBJECTIVE,
    MINIMAX_MUSIC3_RVQ_ENCODER_CONTRACT,
    MINIMAX_MUSIC3_RVQ_SOURCE_REVISION,
    MINIMAX_MUSIC3_TEACHER_FORCING_CONTRACT,
    minimax_music3_encoded_cache_state,
)
from .optimizer import factored_state_elements
from .paging import paged_layer_indices
from .rank_worker import RankWorkerGroup, adapter_sync_digest, build_trainer
from .trainer import (
    DataSourceFactory,
    Flux2DataSourceFactory,
    Flux2LoRATrainer,
    Flux2ModelFactory,
    FluxDataSourceFactory,
    FluxLoRATrainer,
    FluxModelFactory,
    Ideogram4DataSourceFactory,
    Ideogram4LoRATrainer,
    Ideogram4ModelFactory,
    MiniMaxH3DataSourceFactory,
    MiniMaxH3LoRATrainer,
    MiniMaxH3ModelFactory,
    MiniMaxMusic3DataSourceFactory,
    MiniMaxMusic3LoRATrainer,
    MiniMaxMusic3ModelFactory,
    ModelFactory,
    QwenImageDataSourceFactory,
    QwenImageLoRATrainer,
    QwenImageModelFactory,
    SD15LoRATrainer,
    SDXLLoRATrainer,
    WanDataSourceFactory,
    WanLoRATrainer,
    WanModelFactory,
    default_data_source_factory,
    default_flux2_data_source_factory,
    default_flux2_model_factory,
    default_flux_data_source_factory,
    default_flux_model_factory,
    default_ideogram4_data_source_factory,
    default_ideogram4_model_factory,
    default_minimax_h3_data_source_factory,
    default_minimax_h3_model_factory,
    default_minimax_music3_data_source_factory,
    default_minimax_music3_model_factory,
    default_model_factory,
    default_qwen_image_data_source_factory,
    default_qwen_image_model_factory,
    default_wan_data_source_factory,
    default_wan_model_factory,
    flux2_model_assembly_plan,
    flux_model_assembly_plan,
    ideogram4_component_plans,
    merge_rank_rng_state_dicts,
    minimax_h3_time_embedding_kind,
    qwen_image_model_assembly_plan,
    wan_model_assembly_plan,
)

TRAINING_RUNTIME_IDENTITY = "sd15-lora-torch/4"
TRAINING_SNAPSHOT_DIGEST = blake3_digest(b"dinkster.sd15-lora-torch.extension-snapshot.v4")
SDXL_TRAINING_RUNTIME_IDENTITY = "sdxl-lora-torch/1"
SDXL_TRAINING_SNAPSHOT_DIGEST = blake3_digest(b"dinkster.sdxl-lora-torch.extension-snapshot.v1")
MINIMAX_H3_TRAINING_RUNTIME_IDENTITY = "minimax-h3-lora-torch/2"
MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.minimax-h3-lora-torch.extension-snapshot.v2"
)
MINIMAX_MUSIC3_TRAINING_RUNTIME_IDENTITY = "minimax-music3-lora-torch-community/1"
MINIMAX_MUSIC3_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.minimax-music3-lora-torch.community-extension-snapshot.v1"
)
WAN_TRAINING_RUNTIME_IDENTITY = "wan21-t2v-lora-torch/1"
WAN_TRAINING_SNAPSHOT_DIGEST = blake3_digest(b"dinkster.wan21-t2v-lora-torch.extension-snapshot.v1")
WAN22_TI2V_TRAINING_RUNTIME_IDENTITY = "wan22-ti2v-5b-lora-torch/1"
WAN22_TI2V_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.wan22-ti2v-5b-lora-torch.extension-snapshot.v1"
)
WAN22_T2V_TRAINING_RUNTIME_IDENTITY = "wan22-t2v-14b-lora-torch/1"
WAN22_T2V_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.wan22-t2v-14b-lora-torch.extension-snapshot.v1"
)
WAN22_I2V_TRAINING_RUNTIME_IDENTITY = "wan22-i2v-14b-lora-torch/1"
WAN22_I2V_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.wan22-i2v-14b-lora-torch.extension-snapshot.v1"
)
FLUX_DEV_TRAINING_RUNTIME_IDENTITY = "flux1-dev-lora-torch/1"
FLUX_DEV_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.flux1-dev-lora-torch.extension-snapshot.v1"
)
FLUX_SCHNELL_TRAINING_RUNTIME_IDENTITY = "flux1-schnell-lora-torch/1"
FLUX_SCHNELL_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.flux1-schnell-lora-torch.extension-snapshot.v1"
)
FLUX2_DEV_TRAINING_RUNTIME_IDENTITY = "flux2-dev-lora-torch/1"
FLUX2_DEV_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.flux2-dev-lora-torch.extension-snapshot.v1"
)
FLUX2_KLEIN_9B_TRAINING_RUNTIME_IDENTITY = "flux2-klein-9b-lora-torch/1"
FLUX2_KLEIN_9B_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.flux2-klein-9b-lora-torch.extension-snapshot.v1"
)
FLUX2_KLEIN_4B_TRAINING_RUNTIME_IDENTITY = "flux2-klein-4b-lora-torch/1"
FLUX2_KLEIN_4B_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.flux2-klein-4b-lora-torch.extension-snapshot.v1"
)
QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY = "qwen-image-lora-torch/1"
QWEN_IMAGE_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.qwen-image-lora-torch.extension-snapshot.v1"
)
IDEOGRAM4_TRAINING_RUNTIME_IDENTITY = "ideogram4-lora-torch/1"
IDEOGRAM4_TRAINING_SNAPSHOT_DIGEST = blake3_digest(
    b"dinkster.ideogram4-lora-torch.extension-snapshot.v1"
)

_WAN_RUNTIME_IDENTITIES = {
    "wan21-t2v": WAN_TRAINING_RUNTIME_IDENTITY,
    "wan22-ti2v-5b": WAN22_TI2V_TRAINING_RUNTIME_IDENTITY,
    "wan22-t2v-14b": WAN22_T2V_TRAINING_RUNTIME_IDENTITY,
    "wan22-i2v-14b": WAN22_I2V_TRAINING_RUNTIME_IDENTITY,
}
_WAN_SNAPSHOT_DIGESTS = {
    "wan21-t2v": WAN_TRAINING_SNAPSHOT_DIGEST,
    "wan22-ti2v-5b": WAN22_TI2V_TRAINING_SNAPSHOT_DIGEST,
    "wan22-t2v-14b": WAN22_T2V_TRAINING_SNAPSHOT_DIGEST,
    "wan22-i2v-14b": WAN22_I2V_TRAINING_SNAPSHOT_DIGEST,
}
_FLUX_RUNTIME_IDENTITIES = {
    "flux1-dev": FLUX_DEV_TRAINING_RUNTIME_IDENTITY,
    "flux1-schnell": FLUX_SCHNELL_TRAINING_RUNTIME_IDENTITY,
}
_FLUX_SNAPSHOT_DIGESTS = {
    "flux1-dev": FLUX_DEV_TRAINING_SNAPSHOT_DIGEST,
    "flux1-schnell": FLUX_SCHNELL_TRAINING_SNAPSHOT_DIGEST,
}
_FLUX2_RUNTIME_IDENTITIES = {
    "flux2-dev": FLUX2_DEV_TRAINING_RUNTIME_IDENTITY,
    "flux2-klein-9b": FLUX2_KLEIN_9B_TRAINING_RUNTIME_IDENTITY,
    "flux2-klein-4b": FLUX2_KLEIN_4B_TRAINING_RUNTIME_IDENTITY,
}
_FLUX2_SNAPSHOT_DIGESTS = {
    "flux2-dev": FLUX2_DEV_TRAINING_SNAPSHOT_DIGEST,
    "flux2-klein-9b": FLUX2_KLEIN_9B_TRAINING_SNAPSHOT_DIGEST,
    "flux2-klein-4b": FLUX2_KLEIN_4B_TRAINING_SNAPSHOT_DIGEST,
}

_TrainingConfiguration = (
    TrainingConfig
    | FluxTrainingConfig
    | Flux2TrainingConfig
    | MiniMaxH3TrainingConfig
    | MiniMaxMusic3TrainingConfig
    | QwenImageTrainingConfig
    | Ideogram4TrainingConfig
    | WanTrainingConfig
)
_Trainer = (
    SD15LoRATrainer
    | SDXLLoRATrainer
    | FluxLoRATrainer
    | Flux2LoRATrainer
    | MiniMaxH3LoRATrainer
    | MiniMaxMusic3LoRATrainer
    | QwenImageLoRATrainer
    | Ideogram4LoRATrainer
    | WanLoRATrainer
)


class TrainingAdvancePaused(Exception):
    """The advance stopped after publishing a complete recovery checkpoint."""


def _close_rank_group(
    workers: RankWorkerGroup | None,
    rank_context: TorchDistributedRankContext | None,
    *,
    backend: str | None,
) -> None:
    failure: BaseException | None = None
    all_workers_signaled = True
    if workers is not None:
        try:
            all_workers_signaled = workers.signal_teardown()
        except BaseException as exc:
            all_workers_signaled = False
            failure = exc

    if rank_context is not None:
        try:
            if backend == "nccl" and workers is not None and not all_workers_signaled:
                # Graceful NCCL shutdown needs every rank, while abort is local cleanup.
                rank_context.abort()
            else:
                rank_context.close()
        except BaseException as exc:
            if failure is None:
                failure = exc
            if backend == "nccl":
                try:
                    rank_context.abort()
                except BaseException as abort_exc:
                    if not isinstance(abort_exc, Exception):
                        failure = abort_exc

    if workers is not None:
        try:
            workers.close()
        except BaseException as exc:
            if failure is None:
                failure = exc

    if failure is not None:
        raise failure


@dataclass
class _HotRuntime:
    checkpoint_digest: str
    trainer: _Trainer
    config: _TrainingConfiguration
    workers: RankWorkerGroup | None = None
    rank_context: TorchDistributedRankContext | None = None

    def close(self) -> None:
        workers, self.workers = self.workers, None
        rank_context, self.rank_context = self.rank_context, None
        settings = self.config.distributed
        _close_rank_group(
            workers,
            rank_context,
            backend=None if settings is None else settings.backend,
        )


def _runtime_identity(config: _TrainingConfiguration) -> str:
    if config.family == "sd15":
        return TRAINING_RUNTIME_IDENTITY
    if config.family == "sdxl":
        return SDXL_TRAINING_RUNTIME_IDENTITY
    if config.family == "minimax-h3":
        return MINIMAX_H3_TRAINING_RUNTIME_IDENTITY
    if config.family == "minimax-music3":
        return MINIMAX_MUSIC3_TRAINING_RUNTIME_IDENTITY
    if config.family == "wan":
        return _WAN_RUNTIME_IDENTITIES[config.variant]
    if config.family == "flux":
        return _FLUX_RUNTIME_IDENTITIES[config.variant]
    if config.family == "flux2":
        return _FLUX2_RUNTIME_IDENTITIES[config.variant]
    if config.family == "qwen-image":
        return QWEN_IMAGE_TRAINING_RUNTIME_IDENTITY
    if config.family == "ideogram4":
        return IDEOGRAM4_TRAINING_RUNTIME_IDENTITY
    raise ValueError(f"unsupported training family {config.family!r}")


def _snapshot_digest(config: _TrainingConfiguration) -> str:
    if config.family == "sd15":
        return TRAINING_SNAPSHOT_DIGEST
    if config.family == "sdxl":
        return SDXL_TRAINING_SNAPSHOT_DIGEST
    if config.family == "minimax-h3":
        return MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST
    if config.family == "minimax-music3":
        return MINIMAX_MUSIC3_TRAINING_SNAPSHOT_DIGEST
    if config.family == "wan":
        return _WAN_SNAPSHOT_DIGESTS[config.variant]
    if config.family == "flux":
        return _FLUX_SNAPSHOT_DIGESTS[config.variant]
    if config.family == "flux2":
        return _FLUX2_SNAPSHOT_DIGESTS[config.variant]
    if config.family == "qwen-image":
        return QWEN_IMAGE_TRAINING_SNAPSHOT_DIGEST
    if config.family == "ideogram4":
        return IDEOGRAM4_TRAINING_SNAPSHOT_DIGEST
    raise ValueError(f"unsupported training family {config.family!r}")


def _config_digest(config: _TrainingConfiguration) -> str:
    return blake3_digest(
        canonical_json(
            ["dinkster.training.config.v1", _runtime_identity(config), config.identity_mapping()]
        )
    )


def _validate_service_world(config: _TrainingConfiguration) -> None:
    settings = config.distributed
    if settings is None or settings.world_size == 1:
        return
    if config.device == "cpu":
        return
    if config.device in ("cuda", "cuda:0"):
        device_count = torch.cuda.device_count()
        if device_count < settings.world_size:
            raise TrainingConfigError(
                f"distributed world size {settings.world_size} requires at least"
                f" {settings.world_size} CUDA devices, but only {device_count} are available"
            )
        return
    raise TrainingConfigError(
        f"multi-rank training supports device 'cpu', 'cuda', or 'cuda:0', not {config.device!r}"
    )


def _session_id(session_key: str) -> str:
    return digest_bytes(("session:" + session_key).encode("utf-8"))[7:]


def _parameter_shapes(
    targets: tuple[TargetDescriptor, ...], rank: int
) -> tuple[tuple[int, ...], ...]:
    shapes: list[tuple[int, ...]] = []
    for target in targets:
        in_features = math.prod(target.weight_shape[1:])
        shapes.extend(((rank, in_features), (target.weight_shape[0], rank)))
    return tuple(shapes)


def capability_report(config: TrainingConfig) -> dict[str, object]:
    """Resolve targets and estimate training memory categories."""
    try:
        device = torch.device(config.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid training device {config.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"the {config.family} LoRA trainer supports cpu and cuda devices, not {device.type}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    with torch.device("meta"):
        model = UNetModel(config.unet)
    targets = resolve_lora_targets(model, config.rank, family=config.family)
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(target.adapter_parameter_count for target in targets)
    shapes = _parameter_shapes(targets, config.rank)
    if config.optimizer == "adamw":
        optimizer_elements = trainable_parameters * 2
    else:
        optimizer_elements = factored_state_elements(shapes)
    base_bytes = base_parameters * (4 if config.base_dtype == "float32" else 2)
    trainable_bytes = trainable_parameters * 4
    gradient_bytes = trainable_parameters * 4
    optimizer_bytes = optimizer_elements * 4
    batch_values = math.prod(config.latent_shape) + math.prod(config.context_shape)
    if config.pooled_shape is not None:
        batch_values += math.prod(config.pooled_shape) + config.latent_shape[0] * 6
    activation_bytes = batch_values * 2 * 4
    dataset_report: dict[str, object]
    dataset_memory: dict[str, int] = {}
    dataset_store_bytes = 0
    if config.dataset is None:
        dataset_report = {
            "type": "prepared-batches" if config.prepared_batch_root is not None else "injected",
            "statisticsAvailable": False,
            "errors": [],
        }
    else:
        inspection = config.dataset.inspection
        cache_state = encoded_cache_state(
            config.dataset,
            batch_size=config.latent_shape[0],
            device=device,
        )
        if cache_state == "hit":
            encoder_names = (
                (
                    "vaeEncoderParametersTransientLowerBound",
                    "clipTextEncoderParametersTransientLowerBound",
                )
                if config.family == "sd15"
                else (
                    "vaeEncoderParametersTransientLowerBound",
                    "clipLTextEncoderParametersTransientLowerBound",
                    "clipGTextEncoderParametersTransientLowerBound",
                )
            )
            dataset_memory = {name: 0 for name in encoder_names}
            encoder_errors: tuple[str, ...] = ()
        else:
            dataset_memory, encoder_errors = inspect_encoder_memory(config.dataset)
        item_count = len(inspection.items)
        latent_store_bytes = item_count * math.prod(config.latent_shape[1:]) * 4
        text_store_bytes = item_count * math.prod(config.context_shape[1:]) * 4
        pooled_store_bytes = (
            0 if config.pooled_shape is None else item_count * (config.pooled_shape[1] + 6) * 4
        )
        dataset_store_bytes = latent_store_bytes + text_store_bytes + pooled_store_bytes
        image_batch_bytes = (
            config.latent_shape[0]
            * 3
            * config.dataset.resolution[0]
            * config.dataset.resolution[1]
            * 4
        )
        dataset_memory["encodedDatasetStoreResidentCpu"] = dataset_store_bytes
        dataset_memory["encoderInputBatchTransientLowerBound"] = (
            0 if cache_state == "hit" else image_batch_bytes
        )
        dataset_report = {
            "type": "image-caption-folder",
            "digest": config.dataset.digest,
            "encodedCache": {"state": cache_state},
            "itemCount": len(inspection.items),
            "resolution": {
                "height": config.dataset.resolution[0],
                "width": config.dataset.resolution[1],
            },
            "captionCoverage": {
                "total": len(inspection.items),
                "present": inspection.caption_files,
                "nonEmpty": inspection.nonempty_captions,
            },
            "errors": [*inspection.errors, *encoder_errors],
        }
    total_lower_bound = (
        base_bytes
        + trainable_bytes
        + gradient_bytes
        + optimizer_bytes
        + activation_bytes
        + dataset_store_bytes
    )
    memory = {
        "estimated": True,
        "categories": {
            "frozenBaseParameters": base_bytes,
            "loraMasterParameters": trainable_bytes,
            "loraGradients": gradient_bytes,
            "optimizerState": optimizer_bytes,
            "activationsLowerBound": activation_bytes,
            **dataset_memory,
        },
        "totalLowerBoundBytes": total_lower_bound,
        "activationEstimateBasis": (
            "float32 prepared and noised inputs retained at the checkpoint boundary;"
            " excludes intermediate activations and transient kernel workspaces"
        ),
        "totalLowerBoundBasis": (
            "resident training storage across the configured device and CPU; excludes"
            " transient dataset precompute categories"
        ),
    }
    if config.dataset is not None:
        encoder_names = "VAE and CLIP" if config.family == "sd15" else "VAE, CLIP-L, and CLIP-G"
        cache = cast("dict[str, str]", dataset_report["encodedCache"])
        if cache["state"] == "hit":
            memory["datasetEncoderEstimateBasis"] = (
                "float32 CPU storage for one latent and CLIP embedding per item remains resident;"
                " the verified encoded cache skips all encoder and RGB input transients"
            )
        else:
            memory["datasetEncoderEstimateBasis"] = (
                "float32 CPU storage for one latent and CLIP embedding per item remains resident;"
                f" {encoder_names} parameters plus one RGB batch are transient during precompute,"
                " with one encoder loaded at a time; excludes encoder activations, token buffers,"
                " and image decoder workspace"
            )
    precision_plan = {
        "baseStorage": config.base_dtype,
        "forwardAutocast": "bfloat16",
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
    }
    if config.dataset is not None:
        cache = cast("dict[str, str]", dataset_report["encodedCache"])
        precision_plan["datasetPrecomputeEncoders"] = (
            "not loaded on encoded cache hit"
            if cache["state"] == "hit"
            else "float32, transient, one at a time"
        )
        precision_plan["encodedDatasetStore"] = "float32, CPU resident"
    return {
        "trainer": _runtime_identity(config),
        "device": str(device),
        **(
            {"distributed": config.distributed.to_mapping()}
            if config.distributed is not None
            else {}
        ),
        "configDigest": _config_digest(config),
        "sessionExtensionSnapshotDigest": _snapshot_digest(config),
        "capabilities": {
            "autograd": True,
            "families": [config.family],
            "checkpointResume": True,
            "safePointCancellation": True,
            "gradientAccumulation": True,
            "gradientCheckpointing": True,
            "textEncoderTraining": False,
            "imageCaptionDataset": True,
            "encodedDatasetDiskCache": True,
            "optimizers": ["adamw", "factored-adamw"],
        },
        "targets": [target.to_wire() for target in targets],
        "parameterCounts": {
            "frozenBase": base_parameters,
            "trainableLora": trainable_parameters,
            "targetedBase": sum(target.base_parameter_count for target in targets),
        },
        "precisionPlan": precision_plan,
        "memoryLedger": memory,
        "dataset": dataset_report,
    }


def _wan_component_float32_bytes(
    plan: Wan21AssemblyPlan, config: WanTrainingConfig
) -> dict[str, int]:
    categories: dict[str, int] = {}
    for name, component, artifact in (
        ("vaeEncoderParametersTransientLowerBound", plan.vae, config.vae_state),
        (
            "umt5xxlTextEncoderParametersTransientLowerBound",
            plan.umt5xxl,
            config.umt5xxl_state,
        ),
    ):
        source = load_safetensors_header(
            component.path,
            asset_digest=artifact.digest,
            asset_size=artifact.size,
        )
        source_keys = set(component.keys.values())
        categories[name] = sum(source.entry(key).geometry.numel for key in source_keys) * 4
    return categories


def _flux_component_float32_bytes(
    plan: FluxAssemblyPlan, config: FluxTrainingConfig
) -> dict[str, int]:
    assert plan.clip_l is not None and plan.t5xxl is not None
    categories: dict[str, int] = {}
    for name, component, artifact in (
        ("vaeEncoderParametersTransientLowerBound", plan.vae, config.vae_state),
        ("clipLTextEncoderParametersTransientLowerBound", plan.clip_l, config.clip_l_state),
        ("t5xxlTextEncoderParametersTransientLowerBound", plan.t5xxl, config.t5xxl_state),
    ):
        source = load_safetensors_header(
            component.path,
            asset_digest=artifact.digest,
            asset_size=artifact.size,
        )
        source_keys = set(component.keys.values())
        categories[name] = sum(source.entry(key).geometry.numel for key in source_keys) * 4
    return categories


def flux_capability_report(config: FluxTrainingConfig) -> dict[str, object]:
    """Resolve classic Flux DiT targets and training memory lower bounds."""
    try:
        device = torch.device(config.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid training device {config.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"the Flux LoRA trainer supports cpu and cuda devices, not {device.type}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")

    plan = flux_model_assembly_plan(config)
    with torch.device("meta"):
        model = Flux(plan.diffusion.config)
    targets = resolve_lora_targets(model, config.rank, family="flux")
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(target.adapter_parameter_count for target in targets)
    targeted_base_parameters = sum(target.base_parameter_count for target in targets)
    shapes = _parameter_shapes(targets, config.rank)
    optimizer_elements = (
        trainable_parameters * 2 if config.optimizer == "adamw" else factored_state_elements(shapes)
    )
    base_bytes = base_parameters * 2
    trainable_bytes = trainable_parameters * 4
    gradient_bytes = trainable_parameters * 4
    optimizer_bytes = optimizer_elements * 4
    batch_values = (
        math.prod(config.latent_shape)
        + math.prod(config.context_shape)
        + math.prod(config.pooled_shape)
    )
    activation_bytes = batch_values * 2 * 4
    dataset_memory: dict[str, int] = {}
    dataset_store_bytes = 0
    cache_state = "disabled"

    if config.dataset is None:
        dataset_report: dict[str, object] = {
            "type": "injected",
            "statisticsAvailable": False,
            "errors": [],
        }
    else:
        inspection = config.dataset.inspection
        cache_state = flux_encoded_cache_state(
            config.dataset,
            latent_shape=config.latent_shape,
            context_shape=config.context_shape,
            pooled_shape=config.pooled_shape,
            device=device,
        )
        item_count = len(inspection.items)
        dataset_store_bytes = (
            item_count
            * (
                math.prod(config.latent_shape[1:])
                + math.prod(config.context_shape[1:])
                + math.prod(config.pooled_shape[1:])
            )
            * 4
        )
        if cache_state == "hit":
            dataset_memory = {
                "vaeEncoderParametersTransientLowerBound": 0,
                "clipLTextEncoderParametersTransientLowerBound": 0,
                "t5xxlTextEncoderParametersTransientLowerBound": 0,
            }
        else:
            dataset_memory = _flux_component_float32_bytes(plan, config)
        dataset_memory["encodedDatasetStoreResidentCpu"] = dataset_store_bytes
        dataset_memory["encoderInputImageTransientLowerBound"] = (
            0
            if cache_state == "hit"
            else 3 * config.dataset.resolution[0] * config.dataset.resolution[1] * 4
        )
        dataset_report = {
            "type": "flux-image-caption-folder",
            "digest": config.dataset.digest,
            "encodedCache": {"state": cache_state},
            "itemCount": item_count,
            "resolution": {
                "height": config.dataset.resolution[0],
                "width": config.dataset.resolution[1],
            },
            "contextTokens": config.dataset.context_tokens,
            "captionCoverage": {
                "total": item_count,
                "present": inspection.caption_files,
                "nonEmpty": inspection.nonempty_captions,
            },
            "errors": list(inspection.errors),
        }

    total_lower_bound = (
        base_bytes
        + trainable_bytes
        + gradient_bytes
        + optimizer_bytes
        + activation_bytes
        + dataset_store_bytes
    )
    precision_plan = {
        "baseStorage": config.base_dtype,
        "forwardCompute": config.base_dtype,
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
        "flowObjective": "float32",
    }
    if config.dataset is not None:
        precision_plan["datasetPrecomputeComponents"] = (
            "not loaded on encoded cache hit"
            if cache_state == "hit"
            else "VAE float32, then CLIP-L float32, then T5-XXL float32"
        )
        precision_plan["encodedDatasetStore"] = "float32, CPU resident"
    memory: dict[str, object] = {
        "estimated": True,
        "categories": {
            "frozenBaseParameters": base_bytes,
            "loraMasterParameters": trainable_bytes,
            "loraGradients": gradient_bytes,
            "optimizerState": optimizer_bytes,
            "preparedBatchAndNoisedStreamsLowerBound": activation_bytes,
            **dataset_memory,
        },
        "totalLowerBoundBytes": total_lower_bound,
        "deviceMemoryLowerBoundBytes": (
            base_bytes + trainable_bytes + gradient_bytes + optimizer_bytes + activation_bytes
        ),
        "hostRamLowerBoundBytes": dataset_store_bytes,
        "activationEstimateBasis": (
            "float32 prepared latents, context, pooled conditioning, noised input, and velocity"
            " target; excludes intermediate activations and transient kernel workspaces"
        ),
        "totalLowerBoundBasis": (
            "resident training storage across the configured device and CPU; excludes"
            " transient dataset precompute categories"
        ),
    }
    if config.dataset is not None:
        memory["datasetEncoderEstimateBasis"] = (
            "verified encoded cache skips all component and decoded-image transients"
            if cache_state == "hit"
            else "one float32 latent, T5 context, and CLIP-L pooled vector per item remain"
            " resident on CPU; VAE, CLIP-L, and T5-XXL storage plus one decoded image are"
            " transient, with one component loaded at a time and released before the DiT loads;"
            " excludes component activations"
        )

    return {
        "trainer": _runtime_identity(config),
        "device": str(device),
        **(
            {"distributed": config.distributed.to_mapping()}
            if config.distributed is not None
            else {}
        ),
        "variant": config.variant,
        "configDigest": _config_digest(config),
        "sessionExtensionSnapshotDigest": _snapshot_digest(config),
        "capabilities": {
            "autograd": True,
            "families": ["flux"],
            "checkpointResume": True,
            "safePointCancellation": True,
            "gradientAccumulation": True,
            "gradientCheckpointing": True,
            "textEncoderTraining": False,
            "flowMatchingObjective": True,
            "imageCaptionDataset": True,
            "encodedDatasetDiskCache": True,
            "loraExport": True,
            "optimizers": ["adamw", "factored-adamw"],
        },
        "targets": [target.to_wire() for target in targets],
        "parameterCounts": {
            "frozenBase": base_parameters,
            "trainableLora": trainable_parameters,
            "targetedBase": targeted_base_parameters,
        },
        "precisionPlan": precision_plan,
        "memoryLedger": memory,
        "dataset": dataset_report,
    }


def _flux2_component_bytes(plan: Flux2AssemblyPlan, config: Flux2TrainingConfig) -> dict[str, int]:
    categories: dict[str, int] = {}
    for name, component, artifact, element_bytes in (
        ("vaeEncoderParametersTransientLowerBound", plan.vae, config.vae_state, 4),
        (
            "textEncoderParametersTransientLowerBound",
            plan.text_encoder,
            config.text_encoder_state,
            2,
        ),
    ):
        source = load_safetensors_header(
            component.path,
            asset_digest=artifact.digest,
            asset_size=artifact.size,
        )
        source_keys = set(component.keys.values())
        categories[name] = (
            sum(source.entry(key).geometry.numel for key in source_keys) * element_bytes
        )
    return categories


def flux2_capability_report(config: Flux2TrainingConfig) -> dict[str, object]:
    """Resolve Flux2 DiT targets and training memory lower bounds."""
    try:
        device = torch.device(config.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid training device {config.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"the Flux2 LoRA trainer supports cpu and cuda devices, not {device.type}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")

    plan = flux2_model_assembly_plan(config)
    with torch.device("meta"):
        model = Flux(plan.diffusion.config)
    targets = resolve_lora_targets(model, config.rank, family="flux2")
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(target.adapter_parameter_count for target in targets)
    targeted_base_parameters = sum(target.base_parameter_count for target in targets)
    shapes = _parameter_shapes(targets, config.rank)
    optimizer_elements = (
        trainable_parameters * 2 if config.optimizer == "adamw" else factored_state_elements(shapes)
    )
    base_bytes = base_parameters * 2
    trainable_bytes = trainable_parameters * 4
    gradient_bytes = trainable_parameters * 4
    optimizer_bytes = optimizer_elements * 4
    batch_values = math.prod(config.latent_shape) + math.prod(config.context_shape)
    activation_bytes = batch_values * 2 * 4
    dataset_memory: dict[str, int] = {}
    dataset_store_bytes = 0
    cache_state = "disabled"

    if config.dataset is None:
        dataset_report: dict[str, object] = {
            "type": "injected",
            "statisticsAvailable": False,
            "errors": [],
        }
    else:
        inspection = config.dataset.inspection
        cache_state = flux2_encoded_cache_state(
            config.dataset,
            latent_shape=config.latent_shape,
            context_shape=config.context_shape,
            device=device,
        )
        item_count = len(inspection.items)
        dataset_store_bytes = (
            item_count
            * (math.prod(config.latent_shape[1:]) + math.prod(config.context_shape[1:]))
            * 4
        )
        if cache_state == "hit":
            dataset_memory = {
                "vaeEncoderParametersTransientLowerBound": 0,
                "textEncoderParametersTransientLowerBound": 0,
            }
        else:
            dataset_memory = _flux2_component_bytes(plan, config)
        dataset_memory["encodedDatasetStoreResidentCpu"] = dataset_store_bytes
        dataset_memory["encoderInputImageTransientLowerBound"] = (
            0
            if cache_state == "hit"
            else 3 * config.dataset.resolution[0] * config.dataset.resolution[1] * 4
        )
        dataset_report = {
            "type": "flux2-image-caption-folder",
            "variant": config.variant,
            "digest": config.dataset.digest,
            "encodedCache": {"state": cache_state},
            "itemCount": item_count,
            "resolution": {
                "height": config.dataset.resolution[0],
                "width": config.dataset.resolution[1],
            },
            "contextTokens": config.dataset.context_tokens,
            "captionCoverage": {
                "total": item_count,
                "present": inspection.caption_files,
                "nonEmpty": inspection.nonempty_captions,
            },
            "errors": list(inspection.errors),
        }

    total_lower_bound = (
        base_bytes
        + trainable_bytes
        + gradient_bytes
        + optimizer_bytes
        + activation_bytes
        + dataset_store_bytes
    )
    precision_plan = {
        "baseStorage": config.base_dtype,
        "forwardCompute": config.base_dtype,
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
        "flowObjective": "float32",
    }
    if config.dataset is not None:
        precision_plan["datasetPrecomputeComponents"] = (
            "not loaded on encoded cache hit"
            if cache_state == "hit"
            else "VAE float32, then variant text encoder bfloat16"
        )
        precision_plan["encodedDatasetStore"] = "float32, CPU resident"
    memory: dict[str, object] = {
        "estimated": True,
        "categories": {
            "frozenBaseParameters": base_bytes,
            "loraMasterParameters": trainable_bytes,
            "loraGradients": gradient_bytes,
            "optimizerState": optimizer_bytes,
            "preparedBatchAndNoisedStreamsLowerBound": activation_bytes,
            **dataset_memory,
        },
        "totalLowerBoundBytes": total_lower_bound,
        "deviceMemoryLowerBoundBytes": (
            base_bytes + trainable_bytes + gradient_bytes + optimizer_bytes + activation_bytes
        ),
        "hostRamLowerBoundBytes": dataset_store_bytes,
        "activationEstimateBasis": (
            "float32 prepared latents and context, noised input, and velocity target; excludes"
            " intermediate activations and transient kernel workspaces"
        ),
        "totalLowerBoundBasis": (
            "resident training storage across the configured device and CPU; excludes"
            " transient dataset precompute categories"
        ),
    }
    if config.dataset is not None:
        memory["datasetEncoderEstimateBasis"] = (
            "verified encoded cache skips all component and decoded-image transients"
            if cache_state == "hit"
            else "one float32 latent and text context per item remain resident on CPU; VAE and"
            " variant text encoder storage plus one decoded image are transient, with one"
            " component loaded at a time and released before the DiT loads; excludes component"
            " activations"
        )

    return {
        "trainer": _runtime_identity(config),
        "device": str(device),
        **(
            {"distributed": config.distributed.to_mapping()}
            if config.distributed is not None
            else {}
        ),
        "variant": config.variant,
        "configDigest": _config_digest(config),
        "sessionExtensionSnapshotDigest": _snapshot_digest(config),
        "capabilities": {
            "autograd": True,
            "families": ["flux2"],
            "checkpointResume": True,
            "safePointCancellation": True,
            "gradientAccumulation": True,
            "gradientCheckpointing": True,
            "textEncoderTraining": False,
            "flowMatchingObjective": True,
            "imageCaptionDataset": True,
            "encodedDatasetDiskCache": True,
            "loraExport": True,
            "optimizers": ["adamw", "factored-adamw"],
        },
        "targets": [target.to_wire() for target in targets],
        "parameterCounts": {
            "frozenBase": base_parameters,
            "trainableLora": trainable_parameters,
            "targetedBase": targeted_base_parameters,
        },
        "precisionPlan": precision_plan,
        "memoryLedger": memory,
        "dataset": dataset_report,
    }


def _qwen_image_component_bytes(
    plan: QwenImageAssemblyPlan, config: QwenImageTrainingConfig
) -> dict[str, int]:
    categories: dict[str, int] = {}
    for name, component, artifact, element_bytes in (
        ("vaeEncoderParametersTransientLowerBound", plan.vae, config.vae_state, 4),
        (
            "qwen25VlTextEncoderParametersTransientLowerBound",
            plan.qwen2_5_vl_7b,
            config.text_encoder_state,
            2,
        ),
    ):
        source = load_safetensors_header(
            component.path,
            asset_digest=artifact.digest,
            asset_size=artifact.size,
        )
        source_keys = set(component.keys.values())
        categories[name] = (
            sum(source.entry(key).geometry.numel for key in source_keys) * element_bytes
        )
    return categories


def qwen_image_capability_report(config: QwenImageTrainingConfig) -> dict[str, object]:
    """Resolve Qwen-Image DiT targets and training memory lower bounds."""
    try:
        device = torch.device(config.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid training device {config.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"the Qwen-Image LoRA trainer supports cpu and cuda devices, not {device.type}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    device_index = cast("int | None", device.index)
    if (
        device.type == "cuda"
        and device_index is not None
        and device_index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device {device} does not exist")

    plan = qwen_image_model_assembly_plan(config)
    with torch.device("meta"):
        model = QwenImage(plan.diffusion.config)
    targets = resolve_lora_targets(model, config.rank, family="qwen-image")
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(target.adapter_parameter_count for target in targets)
    targeted_base_parameters = sum(target.base_parameter_count for target in targets)
    shapes = _parameter_shapes(targets, config.rank)
    optimizer_elements = (
        trainable_parameters * 2 if config.optimizer == "adamw" else factored_state_elements(shapes)
    )
    base_bytes = base_parameters * (2 if config.base_dtype == "bfloat16" else 4)
    trainable_bytes = trainable_parameters * 4
    gradient_bytes = trainable_parameters * 4
    optimizer_bytes = optimizer_elements * 4
    latent_values = math.prod(config.latent_shape)
    context_values = math.prod(config.context_shape)
    mask_values = math.prod(config.attention_mask_shape)
    activation_bytes = (4 * latent_values + context_values + mask_values) * 4 + mask_values
    dataset_memory: dict[str, int] = {}
    dataset_store_bytes = 0
    cache_state = "disabled"

    if config.dataset is None:
        dataset_report: dict[str, object] = {
            "type": "injected",
            "statisticsAvailable": False,
            "errors": [],
        }
    else:
        inspection = config.dataset.inspection
        cache_state = qwen_image_encoded_cache_state(
            config.dataset,
            latent_shape=config.latent_shape,
            context_shape=config.context_shape,
            attention_mask_shape=config.attention_mask_shape,
            device=device,
        )
        item_count = len(inspection.items)
        dataset_store_bytes = (
            item_count
            * (
                math.prod(config.latent_shape[1:])
                + math.prod(config.context_shape[1:])
                + math.prod(config.attention_mask_shape[1:])
            )
            * 4
        )
        if cache_state == "hit":
            dataset_memory = {
                "vaeEncoderParametersTransientLowerBound": 0,
                "qwen25VlTextEncoderParametersTransientLowerBound": 0,
            }
        else:
            dataset_memory = _qwen_image_component_bytes(plan, config)
        dataset_memory["encodedDatasetStoreResidentCpu"] = dataset_store_bytes
        dataset_memory["encoderInputImageTransientLowerBound"] = (
            0
            if cache_state == "hit"
            else 3 * config.dataset.resolution[0] * config.dataset.resolution[1] * 4
        )
        dataset_report = {
            "type": "qwen-image-image-caption-folder",
            "variant": config.variant,
            "fixedResolution": True,
            "digest": config.dataset.digest,
            "encodedCache": {"state": cache_state},
            "itemCount": item_count,
            "resolution": {
                "height": config.dataset.resolution[0],
                "width": config.dataset.resolution[1],
            },
            "contextTokens": config.dataset.context_tokens,
            "captionCoverage": {
                "total": item_count,
                "present": inspection.caption_files,
                "nonEmpty": inspection.nonempty_captions,
            },
            "errors": list(inspection.errors),
        }

    total_lower_bound = (
        base_bytes
        + trainable_bytes
        + gradient_bytes
        + optimizer_bytes
        + activation_bytes
        + dataset_store_bytes
    )
    precision_plan = {
        "baseStorage": config.base_dtype,
        "forwardCompute": config.base_dtype,
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
        "flowObjective": "float32",
        "preparedBatch": "float32 latents, context, and attention mask",
    }
    if config.dataset is not None:
        precision_plan["datasetPrecomputeComponents"] = (
            "not loaded on encoded cache hit"
            if cache_state == "hit"
            else "Wan 2.1 VAE float32, then Qwen2.5-VL-7B bfloat16"
        )
        precision_plan["encodedDatasetStore"] = (
            "float32 latents, context, and attention mask, CPU resident"
        )
    memory: dict[str, object] = {
        "estimated": True,
        "categories": {
            "frozenBaseParameters": base_bytes,
            "loraMasterParameters": trainable_bytes,
            "loraGradients": gradient_bytes,
            "optimizerState": optimizer_bytes,
            "preparedBatchAndNoisedStreamsLowerBound": activation_bytes,
            **dataset_memory,
        },
        "totalLowerBoundBytes": total_lower_bound,
        "deviceMemoryLowerBoundBytes": (
            base_bytes + trainable_bytes + gradient_bytes + optimizer_bytes + activation_bytes
        ),
        "hostRamLowerBoundBytes": dataset_store_bytes,
        "activationEstimateBasis": (
            "float32 prepared latents, context, and attention mask, float32 noise, noised input,"
            " and velocity target, plus the boolean model attention mask; excludes intermediate"
            " activations, dtype-conversion copies, and transient kernel workspaces"
        ),
        "totalLowerBoundBasis": (
            "resident training storage across the configured device and CPU; excludes"
            " transient dataset precompute categories"
        ),
    }
    if config.dataset is not None:
        memory["datasetEncoderEstimateBasis"] = (
            "verified encoded cache skips all component and decoded-image transients"
            if cache_state == "hit"
            else "one float32 latent, text context, and attention mask per item remain resident"
            " on CPU; the Wan 2.1 VAE and Qwen2.5-VL-7B text encoder plus one decoded image"
            " are transient, with one component loaded at a time and released before the DiT"
            " loads; excludes component activations"
        )

    return {
        "trainer": _runtime_identity(config),
        "device": str(device),
        **(
            {"distributed": config.distributed.to_mapping()}
            if config.distributed is not None
            else {}
        ),
        "variant": config.variant,
        "configDigest": _config_digest(config),
        "sessionExtensionSnapshotDigest": _snapshot_digest(config),
        "capabilities": {
            "autograd": True,
            "families": ["qwen-image"],
            "checkpointResume": True,
            "safePointCancellation": True,
            "gradientAccumulation": True,
            "gradientCheckpointing": True,
            "textEncoderTraining": False,
            "flowMatchingObjective": True,
            "guidanceConditioning": False,
            "imageCaptionDataset": True,
            "datasetBucketing": False,
            "encodedDatasetDiskCache": True,
            "loraExport": True,
            "quantizedBase": False,
            "optimizers": ["adamw", "factored-adamw"],
        },
        "targets": [target.to_wire() for target in targets],
        "parameterCounts": {
            "frozenBase": base_parameters,
            "trainableLora": trainable_parameters,
            "targetedBase": targeted_base_parameters,
        },
        "precisionPlan": precision_plan,
        "memoryLedger": memory,
        "dataset": dataset_report,
    }


def ideogram4_capability_report(config: Ideogram4TrainingConfig) -> dict[str, object]:
    """Resolve role-bound Ideogram 4 targets and training resource bounds."""
    try:
        device = torch.device(config.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid training device {config.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"the Ideogram 4 LoRA trainer supports cpu and cuda devices, not {device.type}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    device_index = cast("int | None", device.index)
    if (
        device.type == "cuda"
        and device_index is not None
        and device_index >= torch.cuda.device_count()
    ):
        raise ValueError(f"CUDA device {device} does not exist")

    ideogram4_component_plans(config)
    with torch.device("meta"):
        model = Ideogram4DiT()
    targets = resolve_lora_targets(model, config.rank, family="ideogram4", role=config.role)
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(target.adapter_parameter_count for target in targets)
    targeted_base_parameters = sum(target.base_parameter_count for target in targets)
    shapes = _parameter_shapes(targets, config.rank)
    optimizer_elements = (
        trainable_parameters * 2 if config.optimizer == "adamw" else factored_state_elements(shapes)
    )
    base_bytes = base_parameters
    trainable_bytes = trainable_parameters * 4
    gradient_bytes = trainable_parameters * 4
    optimizer_bytes = optimizer_elements * 4
    latent_values = math.prod(config.latent_shape)
    context_values = 0 if config.context_shape is None else math.prod(config.context_shape)
    mask_values = (
        0 if config.attention_mask_shape is None else math.prod(config.attention_mask_shape)
    )
    activation_bytes = (4 * latent_values + context_values + mask_values) * 4 + mask_values
    cache_state = "disabled"
    dataset_store_bytes = 0
    if config.dataset is None:
        dataset_report: dict[str, object] = {
            "type": "injected",
            "role": config.role,
            "statisticsAvailable": False,
            "errors": [],
        }
    else:
        inspection = config.dataset.inspection
        cache_state = ideogram4_encoded_cache_state(
            config.dataset,
            latent_shape=config.latent_shape,
            context_shape=config.context_shape,
            attention_mask_shape=config.attention_mask_shape,
            device=device,
        )
        item_count = len(inspection.items)
        per_item_values = math.prod(config.latent_shape[1:])
        if config.context_shape is not None and config.attention_mask_shape is not None:
            per_item_values += math.prod(config.context_shape[1:])
            per_item_values += math.prod(config.attention_mask_shape[1:])
        dataset_store_bytes = item_count * per_item_values * 4
        dataset_report = {
            "type": config.dataset.to_mapping()["type"],
            "role": config.role,
            "fixedResolution": True,
            "digest": config.dataset.digest,
            "encodedCache": {"state": cache_state},
            "itemCount": item_count,
            "resolution": {
                "height": config.dataset.resolution[0],
                "width": config.dataset.resolution[1],
            },
            "contextTokens": config.dataset.context_tokens,
            "captionCoverage": {
                "total": item_count,
                "present": inspection.caption_files,
                "nonEmpty": inspection.nonempty_captions,
            },
            "errors": list(inspection.errors),
        }
    total_lower_bound = (
        base_bytes
        + trainable_bytes
        + gradient_bytes
        + optimizer_bytes
        + activation_bytes
        + dataset_store_bytes
    )
    return {
        "trainer": _runtime_identity(config),
        "device": str(device),
        **(
            {"distributed": config.distributed.to_mapping()}
            if config.distributed is not None
            else {}
        ),
        "variant": config.variant,
        "role": config.role,
        "configDigest": _config_digest(config),
        "sessionExtensionSnapshotDigest": _snapshot_digest(config),
        "capabilities": {
            "autograd": True,
            "families": ["ideogram4"],
            "roles": [config.role],
            "checkpointResume": True,
            "safePointCancellation": True,
            "gradientAccumulation": True,
            "gradientCheckpointing": True,
            "textEncoderTraining": False,
            "flowMatchingObjective": True,
            "imageCaptionDataset": config.role == "conditional",
            "imageDataset": config.role == "unconditional",
            "datasetBucketing": False,
            "encodedDatasetDiskCache": True,
            "loraExport": True,
            "quantizedBase": True,
            "optimizers": ["adamw", "factored-adamw"],
        },
        "targets": [target.to_wire() for target in targets],
        "parameterCounts": {
            "frozenBase": base_parameters,
            "trainableLora": trainable_parameters,
            "targetedBase": targeted_base_parameters,
        },
        "precisionPlan": {
            "baseStorage": config.base_storage,
            "forwardCompute": config.base_dtype,
            "int8BaseForward": config.int8_base_forward,
            "loraMasters": "float32",
            "gradients": "float32",
            "optimizerState": "float32",
            "flowObjective": "float32",
            "preparedBatch": (
                "float32 latents"
                if config.role == "unconditional"
                else "float32 latents, context, and attention mask"
            ),
        },
        "memoryLedger": {
            "estimated": True,
            "categories": {
                "frozenBaseParametersLowerBound": base_bytes,
                "loraMasterParameters": trainable_bytes,
                "loraGradients": gradient_bytes,
                "optimizerState": optimizer_bytes,
                "preparedBatchAndNoisedStreamsLowerBound": activation_bytes,
                "encodedDatasetStoreResidentCpu": dataset_store_bytes,
            },
            "totalLowerBoundBytes": total_lower_bound,
            "deviceMemoryLowerBoundBytes": total_lower_bound - dataset_store_bytes,
            "hostRamLowerBoundBytes": dataset_store_bytes,
        },
        "dataset": dataset_report,
    }


def wan_capability_report(config: WanTrainingConfig) -> dict[str, object]:
    """Resolve the selected Wan DiT targets and memory lower bounds."""
    try:
        device = torch.device(config.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid training device {config.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(f"the Wan LoRA trainer supports cpu and cuda devices, not {device.type}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")

    plan = wan_model_assembly_plan(config)
    with torch.device("meta"):
        model = Wan21Model(plan.diffusion.config)
    targets = resolve_lora_targets(model, config.rank, family="wan")
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(target.adapter_parameter_count for target in targets)
    targeted_base_parameters = sum(target.base_parameter_count for target in targets)
    shapes = _parameter_shapes(targets, config.rank)
    optimizer_elements = (
        trainable_parameters * 2 if config.optimizer == "adamw" else factored_state_elements(shapes)
    )
    base_bytes = base_parameters * 2
    trainable_bytes = trainable_parameters * 4
    gradient_bytes = trainable_parameters * 4
    optimizer_bytes = optimizer_elements * 4
    batch_values = math.prod(config.latent_shape) + math.prod(config.context_shape)
    activation_bytes = batch_values * 2 * 4
    dataset_memory: dict[str, int] = {}
    dataset_store_bytes = 0
    cache_state = "disabled"

    if config.dataset is None:
        dataset_report: dict[str, object] = {
            "type": "injected",
            "statisticsAvailable": False,
            "errors": [],
        }
    else:
        inspection = config.dataset.inspection
        cache_state = wan_encoded_cache_state(
            config.dataset,
            latent_shape=config.latent_shape,
            context_shape=config.context_shape,
            device=device,
        )
        item_count = len(inspection.items)
        dataset_store_bytes = (
            item_count
            * (
                math.prod(config.latent_shape[1:])
                + math.prod(config.context_shape[1:])
                + (
                    20 * math.prod(config.latent_shape[2:])
                    if config.dataset.conditioning_contract == "first-frame-i2v"
                    else 0
                )
            )
            * 4
        )
        if cache_state == "hit":
            dataset_memory = {
                "vaeEncoderParametersTransientLowerBound": 0,
                "umt5xxlTextEncoderParametersTransientLowerBound": 0,
            }
        else:
            dataset_memory = _wan_component_float32_bytes(plan, config)
        dataset_memory["encodedDatasetStoreResidentCpu"] = dataset_store_bytes
        dataset_memory["encoderInputVideoTransientLowerBound"] = (
            0
            if cache_state == "hit"
            else 3
            * config.dataset.frame_count
            * config.dataset.resolution[0]
            * config.dataset.resolution[1]
            * 4
        )
        dataset_report = {
            "type": (
                "wan22-video-caption-folder"
                if config.dataset.vae_contract == "wan22"
                else "wan21-video-caption-folder"
            ),
            "digest": config.dataset.digest,
            "encodedCache": {"state": cache_state},
            "itemCount": item_count,
            "resolution": {
                "height": config.dataset.resolution[0],
                "width": config.dataset.resolution[1],
            },
            "frameCount": config.dataset.frame_count,
            "captionCoverage": {
                "total": item_count,
                "present": inspection.caption_files,
                "nonEmpty": inspection.nonempty_captions,
            },
            "errors": list(inspection.errors),
        }

    total_lower_bound = (
        base_bytes
        + trainable_bytes
        + gradient_bytes
        + optimizer_bytes
        + activation_bytes
        + dataset_store_bytes
    )
    device_lower_bound = (
        base_bytes + trainable_bytes + gradient_bytes + optimizer_bytes + activation_bytes
    )
    precision_plan = {
        "baseStorage": config.base_dtype,
        "forwardCompute": config.base_dtype,
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
        "flowObjective": "float32",
    }
    if config.dataset is not None:
        precision_plan["datasetPrecomputeComponents"] = (
            "not loaded on encoded cache hit"
            if cache_state == "hit"
            else "VAE float32, then UMT5XXL float32"
        )
        precision_plan["encodedDatasetStore"] = "float32, CPU resident"
    memory: dict[str, object] = {
        "estimated": True,
        "categories": {
            "frozenBaseParameters": base_bytes,
            "loraMasterParameters": trainable_bytes,
            "loraGradients": gradient_bytes,
            "optimizerState": optimizer_bytes,
            "preparedBatchAndNoisedStreamsLowerBound": activation_bytes,
            **dataset_memory,
        },
        "totalLowerBoundBytes": total_lower_bound,
        "deviceMemoryLowerBoundBytes": device_lower_bound,
        "hostRamLowerBoundBytes": dataset_store_bytes,
        "activationEstimateBasis": (
            "float32 prepared latents, context, noised stream, and velocity target;"
            " excludes intermediate activations and transient kernel workspaces"
        ),
        "totalLowerBoundBasis": (
            "resident training storage across the configured device and CPU; excludes"
            " transient dataset precompute categories"
        ),
    }
    if config.dataset is not None:
        memory["datasetEncoderEstimateBasis"] = (
            "verified encoded cache skips all component and decoded-video transients"
            if cache_state == "hit"
            else "one float32 encoded video latent and UMT5 context per item remain resident"
            " on CPU; VAE and UMT5XXL storage plus one decoded video are transient, with"
            " one component loaded at a time and released before the DiT loads; excludes"
            " component activations"
        )

    return {
        "trainer": _runtime_identity(config),
        "device": str(device),
        **(
            {"distributed": config.distributed.to_mapping()}
            if config.distributed is not None
            else {}
        ),
        "variant": config.variant,
        "configDigest": _config_digest(config),
        "sessionExtensionSnapshotDigest": _snapshot_digest(config),
        "capabilities": {
            "autograd": True,
            "families": ["wan"],
            "checkpointResume": True,
            "safePointCancellation": True,
            "gradientAccumulation": True,
            "gradientCheckpointing": True,
            "textEncoderTraining": False,
            "flowMatchingObjective": True,
            "videoCaptionDataset": True,
            "encodedDatasetDiskCache": True,
            "loraExport": True,
            "optimizers": ["adamw", "factored-adamw"],
        },
        "targets": [target.to_wire() for target in targets],
        "parameterCounts": {
            "frozenBase": base_parameters,
            "trainableLora": trainable_parameters,
            "targetedBase": targeted_base_parameters,
        },
        "precisionPlan": precision_plan,
        "memoryLedger": memory,
        "dataset": dataset_report,
    }


def minimax_h3_capability_report(config: MiniMaxH3TrainingConfig) -> dict[str, object]:
    """Resolve the complete H3 DiT target set and memory lower bounds."""
    try:
        device = torch.device(config.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid training device {config.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"the MiniMax H3 LoRA trainer supports cpu and cuda devices, not {device.type}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    with torch.device("meta"):
        model = assemble_minimax_h3_dit(
            time_embedding_kind=minimax_h3_time_embedding_kind(config),
            attention_selection=select_attention("flux", "sdpa"),
        )
    targets = resolve_lora_targets(model, config.rank, family="minimax-h3")
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    targeted_base_parameters = sum(target.base_parameter_count for target in targets)
    blocks_value = getattr(model, "blocks", None)
    blocks = tuple(blocks_value) if isinstance(blocks_value, torch.nn.ModuleList) else ()
    if config.host_layer_paging_fraction and not blocks:
        raise ValueError("MiniMax H3 host layer paging requires a non-empty model.blocks")
    paged_indices = (
        paged_layer_indices(len(blocks), config.host_layer_paging_fraction)
        if config.host_layer_paging_fraction
        else ()
    )
    paged_prefixes = tuple(f"blocks.{index}." for index in paged_indices)
    paged_base_parameters = sum(
        parameter.numel() for index in paged_indices for parameter in blocks[index].parameters()
    )
    paged_targets = tuple(
        target for target in targets if target.module_path.startswith(paged_prefixes)
    )
    paged_targeted_parameters = sum(target.base_parameter_count for target in paged_targets)
    trainable_parameters = sum(target.adapter_parameter_count for target in targets)
    shapes = _parameter_shapes(targets, config.rank)
    optimizer_elements = (
        trainable_parameters * 2 if config.optimizer == "adamw" else factored_state_elements(shapes)
    )
    base_itemsize = 4 if config.base_dtype == "float32" else 2
    base_bytes = (
        targeted_base_parameters + (base_parameters - targeted_base_parameters) * base_itemsize
        if config.quantized_base
        else base_parameters * base_itemsize
    )
    paged_base_bytes = (
        paged_targeted_parameters
        + (paged_base_parameters - paged_targeted_parameters) * base_itemsize
        if config.quantized_base
        else paged_base_parameters * base_itemsize
    )
    quantization_scale_bytes = len(targets) * 4 if config.quantized_base else 0
    paged_quantization_scale_bytes = len(paged_targets) * 4 if config.quantized_base else 0
    trainable_bytes = trainable_parameters * 4
    gradient_bytes = trainable_parameters * 4
    optimizer_bytes = optimizer_elements * 4
    batch_values = (
        math.prod(config.video_latent_shape)
        + math.prod(config.audio_latent_shape)
        + math.prod(config.conditioner_shape)
    )
    activation_bytes = batch_values * 2 * 4
    dataset_memory: dict[str, int] = {}
    dataset_store_bytes = 0
    if config.dataset is None:
        dataset_report: dict[str, object] = {
            "type": (
                "prepared-h3-batches" if config.prepared_batch_root is not None else "injected"
            ),
            "statisticsAvailable": False,
            "errors": [],
        }
        cache_state = "disabled"
    else:
        inspection = config.dataset.inspection
        cache_state = minimax_h3_encoded_cache_state(config.dataset, device=device)
        component_names = (
            "videoVaeParametersTransientLowerBound",
            "audioVaeParametersTransientLowerBound",
            "conditionerParametersTransientLowerBound",
        )
        if cache_state == "hit":
            dataset_memory = {name: 0 for name in component_names}
            component_errors: tuple[str, ...] = ()
        else:
            dataset_memory, component_errors = inspect_minimax_h3_encoder_memory(config.dataset)
        item_count = len(inspection.items)
        dataset_store_bytes = item_count * (
            math.prod(config.video_latent_shape[1:]) * 4
            + math.prod(config.audio_latent_shape[1:]) * 4
            + math.prod(config.conditioner_shape[1:]) * 4
            + config.conditioner_shape[1] * 8
        )
        media_input_bytes = (
            3
            * config.dataset.frame_count
            * config.dataset.resolution[0]
            * config.dataset.resolution[1]
            * 4
            + 2 * minimax_h3_audio_sample_count(config.dataset.frame_count) * 4
        )
        dataset_memory["encodedDatasetStoreResidentCpu"] = dataset_store_bytes
        dataset_memory["encoderInputMediaTransientLowerBound"] = (
            0 if cache_state == "hit" else media_input_bytes
        )
        dataset_report = {
            "type": "h3-video-audio-caption-folder",
            "digest": config.dataset.digest,
            "encodedCache": {"state": cache_state},
            "itemCount": item_count,
            "resolution": {
                "height": config.dataset.resolution[0],
                "width": config.dataset.resolution[1],
            },
            "frameCount": config.dataset.frame_count,
            "itemSources": {
                "frameFolders": inspection.frame_folder_items,
                "containers": inspection.container_items,
            },
            "audio": {
                "channels": 2,
                "sampleRateHz": 32_000,
                "samplesPerItem": minimax_h3_audio_sample_count(config.dataset.frame_count),
                "embeddedContainerItems": inspection.embedded_audio_items,
                "sidecarWavItems": inspection.sidecar_audio_items,
            },
            "containerDecoder": (
                MINIMAX_H3_CONTAINER_DECODER_IDENTITY if inspection.container_items else None
            ),
            "captionCoverage": {
                "total": item_count,
                "present": inspection.caption_files,
                "nonEmpty": inspection.nonempty_captions,
            },
            "errors": [*inspection.errors, *component_errors],
        }
    total_lower_bound = (
        base_bytes
        + quantization_scale_bytes
        + trainable_bytes
        + gradient_bytes
        + optimizer_bytes
        + activation_bytes
        + dataset_store_bytes
    )
    composition = config.execution_composition
    precision_plan = {
        "baseStorage": "int8" if config.quantized_base else config.base_dtype,
        "forwardAutocast": "bfloat16",
        "loraMasters": "float32",
        "gradients": "float32",
        "optimizerState": "float32",
        "flowObjective": "float32",
    }
    if config.quantized_base:
        precision_plan["baseScales"] = "float32"
        if config.int8_base_forward == "fused":
            precision_plan["baseForward"] = "fused W8A8 per projection"
            precision_plan["baseBackwardDequantization"] = "bounded bfloat16 input-gradient chunks"
        else:
            precision_plan["baseDequantization"] = "bfloat16 per projection"
    if paged_indices:
        precision_plan["pagedBaseStorage"] = "immutable pinned CPU, streamed at source dtype"
    if config.dataset is not None:
        precision_plan["datasetPrecomputeComponents"] = (
            "not loaded on encoded cache hit"
            if cache_state == "hit"
            else "video VAE float16, then audio VAE float16, then conditioner bfloat16"
        )
        precision_plan["encodedDatasetStore"] = "float32 plus int64 carrier tags, CPU resident"
    base_categories = (
        {
            "frozenBaseParametersDevice": base_bytes - paged_base_bytes,
            "frozenBaseParametersPinnedCpu": paged_base_bytes,
        }
        if paged_indices
        else {"frozenBaseParameters": base_bytes}
    )
    scale_categories: dict[str, int] = {}
    if config.quantized_base:
        scale_categories = (
            {
                "quantizationScalesDeviceLowerBound": (
                    quantization_scale_bytes - paged_quantization_scale_bytes
                ),
                "quantizationScalesPinnedCpuLowerBound": paged_quantization_scale_bytes,
            }
            if paged_indices
            else {"quantizationScalesLowerBound": quantization_scale_bytes}
        )
    device_lower_bound = (
        base_bytes
        - paged_base_bytes
        + quantization_scale_bytes
        - paged_quantization_scale_bytes
        + trainable_bytes
        + gradient_bytes
        + optimizer_bytes
        + activation_bytes
    )
    host_lower_bound = paged_base_bytes + paged_quantization_scale_bytes + dataset_store_bytes
    memory: dict[str, object] = {
        "estimated": True,
        "categories": {
            **base_categories,
            **scale_categories,
            "loraMasterParameters": trainable_bytes,
            "loraGradients": gradient_bytes,
            "optimizerState": optimizer_bytes,
            "preparedBatchAndNoisedStreamsLowerBound": activation_bytes,
            **dataset_memory,
        },
        "totalLowerBoundBytes": total_lower_bound,
        "deviceMemoryLowerBoundBytes": device_lower_bound,
        "hostRamLowerBoundBytes": host_lower_bound,
        "activationEstimateBasis": (
            "float32 prepared video, audio, conditioner, noised streams, and targets;"
            " excludes intermediate activations and transient kernel workspaces"
        ),
    }
    if config.quantized_base:
        memory["frozenBaseEstimateBasis"] = (
            "one INT8 byte per targetable projection weight, bfloat16 remaining parameters,"
            " and at least one float32 scale per quantized projection"
        )
    if config.dataset is not None:
        memory["datasetEncoderEstimateBasis"] = (
            "verified encoded cache skips all component and decoded-media transients"
            if cache_state == "hit"
            else "one float32 encoded item per stream remains resident on CPU; video VAE,"
            " audio VAE, and conditioner storage plus one decoded media item are transient,"
            " with one component loaded at a time and released before the DiT loads;"
            " excludes component activations"
        )
    return {
        "trainer": MINIMAX_H3_TRAINING_RUNTIME_IDENTITY,
        "device": str(device),
        **(
            {"distributed": config.distributed.to_mapping()}
            if config.distributed is not None
            else {}
        ),
        "configDigest": _config_digest(config),
        "sessionExtensionSnapshotDigest": MINIMAX_H3_TRAINING_SNAPSHOT_DIGEST,
        "executionComposition": {
            "familyId": composition.family_id,
            "executionIdentity": composition.execution_identity,
            "components": [
                {"role": binding.role, "identity": binding.identity}
                for binding in composition.components
            ],
        },
        "capabilities": {
            "autograd": True,
            "families": ["minimax-h3"],
            "ditRoles": [config.dit_role],
            "checkpointResume": True,
            "safePointCancellation": True,
            "gradientAccumulation": True,
            "gradientCheckpointing": True,
            "textEncoderTraining": False,
            "imageCaptionDataset": False,
            "videoAudioCaptionDataset": True,
            "containerVideoDataset": True,
            "encodedDatasetDiskCache": True,
            "preparedMultistreamBatches": True,
            "loraExport": True,
            "quantizedBase": True,
            "int8BaseForwards": ["dequantize", "fused"],
            "hostRamLayerPaging": True,
            "optimizers": ["adamw", "factored-adamw"],
        },
        "layerPaging": {
            "enabled": bool(paged_indices),
            "configuredHostResidentFraction": config.host_layer_paging_fraction,
            "hostResidentFraction": 0.0 if not blocks else len(paged_indices) / len(blocks),
            "hostResidentLayerCount": len(paged_indices),
            "transformerLayerCount": len(blocks),
            "forwardPrefetchDistance": 1 if paged_indices else 0,
            "backwardPrefetchDistance": 1 if paged_indices else 0,
        },
        "targets": [target.to_wire() for target in targets],
        "parameterCounts": {
            "frozenBase": base_parameters,
            "trainableLora": trainable_parameters,
            "targetedBase": targeted_base_parameters,
        },
        "precisionPlan": precision_plan,
        "memoryLedger": memory,
        "dataset": dataset_report,
    }


def minimax_music3_capability_report(
    config: MiniMaxMusic3TrainingConfig,
) -> dict[str, object]:
    """Resolve the fixed Music 3 target set and conservative memory lower bounds."""
    try:
        device = torch.device(config.device)
    except RuntimeError as exc:
        raise ValueError(f"invalid training device {config.device!r}") from exc
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"the MiniMax Music 3 LoRA trainer supports cpu and cuda devices, not {device.type}"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device {device} was requested but CUDA is unavailable")
    with torch.device("meta"):
        model = MiniMaxMusic3DiT()
    targets = resolve_lora_targets(model, config.rank, family="minimax-music3")
    base_parameters = sum(parameter.numel() for parameter in model.parameters())
    targeted_base_parameters = sum(target.base_parameter_count for target in targets)
    trainable_parameters = sum(target.adapter_parameter_count for target in targets)
    optimizer_elements = (
        trainable_parameters * 2
        if config.optimizer == "adamw"
        else factored_state_elements(_parameter_shapes(targets, config.rank))
    )
    base_itemsize = 4 if config.base_dtype == "float32" else 2
    base_bytes = (
        targeted_base_parameters + (base_parameters - targeted_base_parameters) * base_itemsize
        if config.quantized_base
        else base_parameters * base_itemsize
    )
    scale_bytes = len(targets) * 4 if config.quantized_base else 0
    trainable_bytes = trainable_parameters * 4
    optimizer_bytes = optimizer_elements * 4
    item_count = len(config.dataset.inspection.items)
    dataset_bytes = (
        item_count * (math.prod(config.latent_shape[1:]) + math.prod(config.context_shape[1:])) * 4
    )
    activation_bytes = (math.prod(config.latent_shape) * 3 + math.prod(config.context_shape)) * 4
    cache_state = minimax_music3_encoded_cache_state(config.dataset, device=device)
    encoder_bytes = (
        0
        if cache_state == "hit"
        else max(
            config.dataset.dav_encoder_state.size,
            config.dataset.rvq_encoder_state.size,
            config.dataset.text_encoder_state.size,
        )
    )
    memory = {
        "estimated": True,
        "categories": {
            "frozenBaseParameters": base_bytes,
            "quantizationScalesLowerBound": scale_bytes,
            "loraMasterParameters": trainable_bytes,
            "loraGradients": trainable_bytes,
            "optimizerState": optimizer_bytes,
            "preparedBatchAndNoisedStreamsLowerBound": activation_bytes,
            "encodedDatasetStoreResidentCpu": dataset_bytes,
            "largestSequentialDatasetEncoderArtifactTransient": encoder_bytes,
        },
        "totalLowerBoundBytes": (
            base_bytes
            + scale_bytes
            + trainable_bytes * 2
            + optimizer_bytes
            + activation_bytes
            + dataset_bytes
        ),
        "datasetEncoderEstimateBasis": (
            "verified encoded cache skips DAV, RVQ, and text encoders"
            if cache_state == "hit"
            else "DAV, RVQ, and text encoders load sequentially; artifact bytes exclude"
            " activations and framework overhead"
        ),
    }
    inspection = config.dataset.inspection
    return {
        "trainer": MINIMAX_MUSIC3_TRAINING_RUNTIME_IDENTITY,
        "device": str(device),
        **(
            {"distributed": config.distributed.to_mapping()}
            if config.distributed is not None
            else {}
        ),
        "configDigest": _config_digest(config),
        "sessionExtensionSnapshotDigest": MINIMAX_MUSIC3_TRAINING_SNAPSHOT_DIGEST,
        "communityTrainingSource": {
            "authority": "engineering-source-not-official-parity-authority",
            "recipeRevision": MINIMAX_MUSIC3_COMMUNITY_RECIPE_REVISION,
            "davSourceRevision": MINIMAX_MUSIC3_DAV_SOURCE_REVISION,
            "rvqSourceRevision": MINIMAX_MUSIC3_RVQ_SOURCE_REVISION,
            "davContract": MINIMAX_MUSIC3_DAV_ENCODER_CONTRACT,
            "rvqContract": MINIMAX_MUSIC3_RVQ_ENCODER_CONTRACT,
            "teacherForcingContract": MINIMAX_MUSIC3_TEACHER_FORCING_CONTRACT,
            "flowObjective": MINIMAX_MUSIC3_FLOW_OBJECTIVE,
        },
        "capabilities": {
            "autograd": True,
            "families": ["minimax-music3"],
            "checkpointResume": True,
            "safePointCancellation": True,
            "gradientAccumulation": True,
            "gradientCheckpointing": True,
            "textEncoderTraining": False,
            "audioCaptionLyricsDataset": True,
            "encodedDatasetDiskCache": True,
            "loraExport": True,
            "quantizedBase": True,
            "optimizers": ["adamw", "factored-adamw"],
        },
        "targets": [target.to_wire() for target in targets],
        "parameterCounts": {
            "frozenBase": base_parameters,
            "trainableLora": trainable_parameters,
            "targetedBase": targeted_base_parameters,
        },
        "precisionPlan": {
            "baseStorage": "int8" if config.quantized_base else config.base_dtype,
            "baseCompute": config.base_dtype,
            "textCompute": config.text_dtype,
            "loraMasters": "float32",
            "gradients": "float32",
            "optimizerState": "float32",
            "flowObjective": "float32",
            "encodedDatasetStore": "float32 CPU",
        },
        "memoryLedger": memory,
        "dataset": {
            "type": "minimax-music3-audio-caption-lyrics-folder",
            "digest": config.dataset.digest,
            "encodedCache": {"state": cache_state},
            "itemCount": item_count,
            "sampleRateHz": 44_100,
            "audioFrames": config.dataset.audio_frames,
            "samplesPerItem": config.dataset.samples_per_item,
            "captionCoverage": {
                "total": item_count,
                "present": inspection.caption_files,
                "nonEmpty": inspection.nonempty_captions,
            },
            "lyricsCoverage": {
                "total": item_count,
                "present": inspection.lyrics_files,
                "nonEmpty": inspection.nonempty_lyrics,
            },
            "errors": list(inspection.errors),
        },
    }


class _LoRATrainingService:
    """TrainingService implementation using the durable claim/fence ledger."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        expected_family: str,
        trainer_type: (
            type[SD15LoRATrainer]
            | type[SDXLLoRATrainer]
            | type[FluxLoRATrainer]
            | type[Flux2LoRATrainer]
            | type[MiniMaxH3LoRATrainer]
            | type[MiniMaxMusic3LoRATrainer]
            | type[QwenImageLoRATrainer]
            | type[Ideogram4LoRATrainer]
            | type[WanLoRATrainer]
        ),
        scope: str = "local",
        model_factory: ModelFactory
        | FluxModelFactory
        | Flux2ModelFactory
        | MiniMaxH3ModelFactory
        | MiniMaxMusic3ModelFactory
        | QwenImageModelFactory
        | Ideogram4ModelFactory
        | WanModelFactory = default_model_factory,
        data_source_factory: (
            DataSourceFactory
            | FluxDataSourceFactory
            | Flux2DataSourceFactory
            | MiniMaxH3DataSourceFactory
            | MiniMaxMusic3DataSourceFactory
            | QwenImageDataSourceFactory
            | Ideogram4DataSourceFactory
            | WanDataSourceFactory
        ) = default_data_source_factory,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._store = store
        self._checkpoints = ContentAddressedCheckpointStore(checkpoint_root)
        self._export_root = checkpoint_root / "exports"
        self._scope = scope
        self._expected_family = expected_family
        self._trainer_type = trainer_type
        self._model_factory = model_factory
        self._data_source_factory = data_source_factory
        self._cancelled = cancelled if cancelled is not None else lambda: False
        self._fences: dict[str, int] = {}
        self._hot: dict[str, _HotRuntime] = {}
        self._lock = threading.RLock()
        self.steps_run = 0

    @staticmethod
    def session_id(session_key: str) -> str:
        return _session_id(session_key)

    def _fence(self, session_id: str) -> int:
        with self._lock:
            epoch = self._fences.get(session_id)
            if epoch is None:
                epoch = self._store.acquire_fence(session_id).fence_epoch
                self._fences[session_id] = epoch
            return epoch

    def dry_run(self, config: str) -> Mapping[str, object]:
        normalized = self._parse_config(config)
        _validate_service_world(normalized)
        if isinstance(normalized, MiniMaxH3TrainingConfig):
            return minimax_h3_capability_report(normalized)
        if isinstance(normalized, MiniMaxMusic3TrainingConfig):
            return minimax_music3_capability_report(normalized)
        if isinstance(normalized, WanTrainingConfig):
            return wan_capability_report(normalized)
        if isinstance(normalized, FluxTrainingConfig):
            return flux_capability_report(normalized)
        if isinstance(normalized, Flux2TrainingConfig):
            return flux2_capability_report(normalized)
        if isinstance(normalized, QwenImageTrainingConfig):
            return qwen_image_capability_report(normalized)
        if isinstance(normalized, Ideogram4TrainingConfig):
            return ideogram4_capability_report(normalized)
        return capability_report(normalized)

    def _parse_config(self, serialized: str) -> _TrainingConfiguration:
        if self._expected_family == "minimax-h3":
            return MiniMaxH3TrainingConfig.parse(serialized)
        if self._expected_family == "minimax-music3":
            return MiniMaxMusic3TrainingConfig.parse(serialized)
        if self._expected_family == "wan":
            return WanTrainingConfig.parse(serialized)
        if self._expected_family == "flux":
            return FluxTrainingConfig.parse(serialized)
        if self._expected_family == "flux2":
            return Flux2TrainingConfig.parse(serialized)
        if self._expected_family == "qwen-image":
            return QwenImageTrainingConfig.parse(serialized)
        if self._expected_family == "ideogram4":
            return Ideogram4TrainingConfig.parse(serialized)
        return TrainingConfig.parse(
            serialized, expected_family=cast("TrainingFamily", self._expected_family)
        )

    def _config_from_mapping(self, mapping: dict[str, object]) -> _TrainingConfiguration:
        if self._expected_family == "minimax-h3":
            return MiniMaxH3TrainingConfig.from_mapping(mapping)
        if self._expected_family == "minimax-music3":
            return MiniMaxMusic3TrainingConfig.from_mapping(mapping)
        if self._expected_family == "wan":
            return WanTrainingConfig.from_mapping(mapping)
        if self._expected_family == "flux":
            return FluxTrainingConfig.from_mapping(mapping)
        if self._expected_family == "flux2":
            return Flux2TrainingConfig.from_mapping(mapping)
        if self._expected_family == "qwen-image":
            return QwenImageTrainingConfig.from_mapping(mapping)
        if self._expected_family == "ideogram4":
            return Ideogram4TrainingConfig.from_mapping(mapping)
        return TrainingConfig.from_mapping(
            mapping, expected_family=cast("TrainingFamily", self._expected_family)
        )

    @staticmethod
    def _report(config: _TrainingConfiguration) -> dict[str, object]:
        if isinstance(config, MiniMaxH3TrainingConfig):
            return minimax_h3_capability_report(config)
        if isinstance(config, MiniMaxMusic3TrainingConfig):
            return minimax_music3_capability_report(config)
        if isinstance(config, WanTrainingConfig):
            return wan_capability_report(config)
        if isinstance(config, FluxTrainingConfig):
            return flux_capability_report(config)
        if isinstance(config, Flux2TrainingConfig):
            return flux2_capability_report(config)
        if isinstance(config, QwenImageTrainingConfig):
            return qwen_image_capability_report(config)
        if isinstance(config, Ideogram4TrainingConfig):
            return ideogram4_capability_report(config)
        return capability_report(config)

    def _write_checkpoint(
        self,
        *,
        session_id: str,
        config_digest: str,
        parent_manifest_digest: str,
        runtime: _HotRuntime,
    ) -> str:
        trainer = runtime.trainer
        rng = trainer.rng_state_dict()
        workers = runtime.workers
        if workers is not None:
            settings = runtime.config.distributed
            assert settings is not None and settings.world_size > 1
            rng = merge_rank_rng_state_dicts(
                (rng, *workers.rng_state_dicts()),
                world_size=settings.world_size,
            )
        return self._checkpoints.write(
            session_id=session_id,
            config_digest=config_digest,
            extension_snapshot_digest=_snapshot_digest(runtime.config),
            parent_manifest_digest=parent_manifest_digest,
            step_cursor=trainer.step_cursor,
            config=runtime.config.to_mapping(),
            adapter=trainer.attachment.state_dict(),
            optimizer=trainer.optimizer_state_dict(),
            rng=rng,
            data_cursor=trainer.data_cursor,
            loss=trainer.last_loss,
        )

    def _write_intermediate_lora(
        self,
        *,
        session_id: str,
        config_digest: str,
        runtime: _HotRuntime,
    ) -> tuple[Path, str]:
        trainer = runtime.trainer
        source = LoraExportSource(
            checkpoint_manifest_digest=None,
            session_id=session_id,
            config_digest=config_digest,
            step_cursor=trainer.step_cursor,
            adapter=trainer.attachment.state_dict(),
        )
        settings = LoraExportSettings(
            path=self._export_root
            / "cadence"
            / session_id
            / f"step-{trainer.step_cursor:012d}.safetensors",
            dtype="fp32",
        )
        return export_intermediate_lora(
            source,
            runtime.config,
            settings,
            trainer.attachment.targets,
            trainer.lora_export_model_state_keys,
            runtime_identity=_runtime_identity(runtime.config),
        )

    def _evict_runtime(
        self,
        session_id: str,
        runtime: _HotRuntime | None = None,
    ) -> None:
        current = self._hot.get(session_id)
        if current is None or (runtime is not None and current is not runtime):
            return
        self._hot.pop(session_id)
        current.close()

    def _evict_other_distributed_runtimes(self, session_id: str) -> None:
        for other_session_id, runtime in tuple(self._hot.items()):
            if other_session_id != session_id and runtime.workers is not None:
                self._evict_runtime(other_session_id, runtime)

    def _new_runtime(
        self,
        *,
        session_id: str,
        checkpoint_digest: str,
        config: _TrainingConfiguration,
        state: CheckpointState | None = None,
    ) -> _HotRuntime:
        settings = config.distributed
        if settings is None or settings.world_size == 1:
            trainer = build_trainer(
                config,
                trainer_type=self._trainer_type,
                model_factory=self._model_factory,
                data_source_factory=self._data_source_factory,
            )
            if state is not None:
                trainer.restore(
                    adapter=state.adapter,
                    optimizer=state.optimizer,
                    rng=state.rng,
                    step_cursor=state.step_cursor,
                    data_cursor=state.data_cursor,
                    loss=state.loss,
                )
            return _HotRuntime(checkpoint_digest, trainer, config)

        self._evict_other_distributed_runtimes(session_id)
        if isinstance(settings.rendezvous, FileRendezvousSettings):
            Path(settings.rendezvous.path).unlink(missing_ok=True)
        workers = RankWorkerGroup(
            settings.world_size,
            expected_family=self._expected_family,
            trainer_type=self._trainer_type,
            model_factory=self._model_factory,
            data_source_factory=self._data_source_factory,
        )
        rank_context: TorchDistributedRankContext | None = None
        try:
            workers.start_load(
                config.to_mapping(),
                None if state is None else checkpoint_digest,
                self._checkpoints.root,
            )
            rank_context = bootstrap_rank_context(settings, 0)
            trainer = build_trainer(
                config,
                trainer_type=self._trainer_type,
                model_factory=self._model_factory,
                data_source_factory=self._data_source_factory,
                rank_context=rank_context,
            )
            if state is not None:
                trainer.restore(
                    adapter=state.adapter,
                    optimizer=state.optimizer,
                    rng=state.rng,
                    step_cursor=state.step_cursor,
                    data_cursor=state.data_cursor,
                    loss=state.loss,
                )
            workers.finish_load()
            return _HotRuntime(
                checkpoint_digest,
                trainer,
                config,
                workers,
                rank_context,
            )
        except BaseException as exc:
            try:
                _close_rank_group(workers, rank_context, backend=settings.backend)
            except BaseException as cleanup_exc:
                exc.add_note(f"distributed runtime cleanup failed: {cleanup_exc}")
            raise

    def create(
        self, session_key: str, config: str
    ) -> tuple[TrainingSessionHandle, Mapping[str, object]]:
        normalized = self._parse_config(config)
        _validate_service_world(normalized)
        report = self._report(normalized)
        dataset_report = report["dataset"]
        assert isinstance(dataset_report, dict)
        errors = dataset_report["errors"]
        assert isinstance(errors, list)
        if errors:
            raise ValueError(
                "dataset validation failed: " + "; ".join(str(error) for error in errors)
            )
        config_digest = _config_digest(normalized)
        snapshot_digest = _snapshot_digest(normalized)
        session_id = _session_id(session_key)
        existing = self._store.get_session(session_id)
        if existing is not None:
            if (
                existing.config_digest != config_digest
                or existing.extension_snapshot_digest != snapshot_digest
            ):
                raise TrainingLineageConflict(
                    f"session {session_id!r} already exists with different identity facts"
                )
            initial = self._store.list_checkpoints(session_id)[0]
            return (
                TrainingSessionHandle(
                    session_id=session_id,
                    checkpoint_manifest_digest=initial.manifest_digest,
                    step_cursor=0,
                    config_digest=config_digest,
                    session_extension_snapshot_digest=snapshot_digest,
                    journal_seq=initial.journal_seq,
                ),
                report,
            )
        runtime = self._new_runtime(
            session_id=session_id,
            checkpoint_digest="",
            config=normalized,
        )
        try:
            initial_digest = self._write_checkpoint(
                session_id=session_id,
                config_digest=config_digest,
                parent_manifest_digest="",
                runtime=runtime,
            )
            self._store.create_session(
                session_id,
                scope=self._scope,
                config_digest=config_digest,
                extension_snapshot_digest=snapshot_digest,
                initial_manifest_digest=initial_digest,
            )
        except BaseException:
            runtime.close()
            raise
        initial = self._store.list_checkpoints(session_id)[0]
        runtime.checkpoint_digest = initial_digest
        self._hot[session_id] = runtime
        return (
            TrainingSessionHandle(
                session_id=session_id,
                checkpoint_manifest_digest=initial.manifest_digest,
                step_cursor=0,
                config_digest=config_digest,
                session_extension_snapshot_digest=snapshot_digest,
                journal_seq=initial.journal_seq,
            ),
            report,
        )

    def _load_runtime(
        self,
        digest: str,
        handle: TrainingSessionHandle,
        state: CheckpointState | None = None,
    ) -> _HotRuntime:
        if state is None:
            state = self._checkpoints.load(digest)
        if (
            state.session_id != handle.session_id
            or state.config_digest != handle.config_digest
            or state.extension_snapshot_digest != handle.session_extension_snapshot_digest
        ):
            raise TrainingLineageConflict("checkpoint identity does not match the session handle")
        config = self._config_from_mapping(state.config)
        if _config_digest(config) != handle.config_digest:
            raise TrainingLineageConflict("checkpoint config payload does not match its digest")
        _validate_service_world(config)
        return self._new_runtime(
            session_id=handle.session_id,
            checkpoint_digest=digest,
            config=config,
            state=state,
        )

    def _runtime(
        self,
        digest: str,
        handle: TrainingSessionHandle,
        state: CheckpointState | None = None,
    ) -> _HotRuntime:
        hot = self._hot.get(handle.session_id)
        if hot is None or hot.checkpoint_digest != digest:
            if hot is not None:
                self._evict_runtime(handle.session_id, hot)
            hot = self._load_runtime(digest, handle, state)
            self._hot[handle.session_id] = hot
        return hot

    def _verify_handle(self, handle: TrainingSessionHandle) -> CheckpointState:
        session = self._store.get_session(handle.session_id)
        if session is None:
            raise TrainingLineageConflict(f"unknown training session {handle.session_id!r}")
        if (
            session.config_digest != handle.config_digest
            or session.extension_snapshot_digest != handle.session_extension_snapshot_digest
        ):
            raise TrainingLineageConflict(
                "handle pins do not match the session's config/extension identity"
            )
        checkpoints = {
            checkpoint.manifest_digest: checkpoint
            for checkpoint in self._store.list_checkpoints(handle.session_id)
        }
        checkpoint = checkpoints.get(handle.checkpoint_manifest_digest)
        if (
            checkpoint is None
            or checkpoint.step_cursor != handle.step_cursor
            or checkpoint.journal_seq != handle.journal_seq
        ):
            raise TrainingLineageConflict(
                "handle does not match a committed checkpoint of this session"
            )
        state = self._checkpoints.load(handle.checkpoint_manifest_digest)
        if (
            state.session_id != handle.session_id
            or state.config_digest != handle.config_digest
            or state.extension_snapshot_digest != handle.session_extension_snapshot_digest
            or state.step_cursor != handle.step_cursor
        ):
            raise TrainingLineageConflict("handle fields do not match the checkpoint manifest")
        return state

    def _operation_id(
        self,
        handle: TrainingSessionHandle,
        steps: int,
        note: str,
        config: _TrainingConfiguration,
    ) -> str:
        value = [
            "dinkster.training.advance.v1",
            handle.session_id,
            handle.checkpoint_manifest_digest,
            handle.config_digest,
            handle.session_extension_snapshot_digest,
            steps,
            note,
            _runtime_identity(config),
        ]
        return digest_bytes(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
        )[7:]

    @staticmethod
    def _verify_sync_digests(runtime: _HotRuntime) -> None:
        trainer = runtime.trainer
        workers = runtime.workers
        if workers is None:
            return
        rank_zero_digest = adapter_sync_digest(trainer)
        worker_digests = workers.sync_digests()
        for rank, digest in zip(workers.ranks, worker_digests, strict=True):
            if digest != rank_zero_digest:
                raise RuntimeError(
                    "distributed adapter parameters diverged after optimizer step"
                    f" {trainer.step_cursor}: rank {rank} digest {digest}"
                    f" != rank 0 digest {rank_zero_digest}"
                )

    def _train_step(self, runtime: _HotRuntime, *, verify_sync_digest: bool = False) -> None:
        trainer = runtime.trainer
        workers = runtime.workers
        if workers is None:
            trainer.train_step()
            self.steps_run += 1
            return

        workers.start_step()
        trainer.train_step()
        workers.finish_step()
        if verify_sync_digest:
            self._verify_sync_digests(runtime)
        self.steps_run += 1

    def advance(self, handle: TrainingSessionHandle, steps: int, note: str) -> AdvanceOutcome:
        if steps < 1:
            raise ValueError("an advance must request at least one optimizer step")
        input_state = self._verify_handle(handle)
        operation_id = self._operation_id(
            handle,
            steps,
            note,
            self._config_from_mapping(input_state.config),
        )
        fence = self._fence(handle.session_id)
        record = self._store.claim_advance(
            handle.session_id,
            operation_id,
            input_manifest_digest=handle.checkpoint_manifest_digest,
            fence_epoch=fence,
        )
        if record.status == "committed":
            return self._committed_outcome(handle, record)
        resume_digest = record.recovery_checkpoint_digest or handle.checkpoint_manifest_digest
        runtime = self._runtime(
            resume_digest,
            handle,
            input_state if resume_digest == handle.checkpoint_manifest_digest else None,
        )
        trainer = runtime.trainer
        target_step = handle.step_cursor + steps
        try:
            if trainer.step_cursor < handle.step_cursor or trainer.step_cursor > target_step:
                raise TrainingLineageConflict(
                    f"recovery checkpoint step {trainer.step_cursor} is outside the advance"
                    f" range {handle.step_cursor}..{target_step}"
                )
            recovery_published = bool(record.recovery_checkpoint_digest)
            checkpoint_interval = runtime.config.checkpoint_interval
            lora_export_interval = runtime.config.lora_export_interval
            sync_digest_interval = runtime.config.sync_digest_interval
            checkpointed_step = trainer.step_cursor
            digest_checked_step: int | None = None

            def publish_recovery_checkpoint() -> None:
                nonlocal recovery_published, checkpointed_step, digest_checked_step
                if runtime.workers is not None and digest_checked_step != trainer.step_cursor:
                    self._verify_sync_digests(runtime)
                    digest_checked_step = trainer.step_cursor
                digest = self._write_checkpoint(
                    session_id=handle.session_id,
                    config_digest=handle.config_digest,
                    parent_manifest_digest=handle.checkpoint_manifest_digest,
                    runtime=runtime,
                )
                runtime.checkpoint_digest = digest
                self._store.set_recovery_checkpoint(
                    handle.session_id,
                    operation_id,
                    fence_epoch=fence,
                    manifest_digest=digest,
                )
                recovery_published = True
                checkpointed_step = trainer.step_cursor

            def pause_if_cancelled() -> None:
                if not self._cancelled():
                    return
                if trainer.step_cursor > checkpointed_step:
                    # A pause is a durable safe point: steps taken since the
                    # last cadence checkpoint must not be lost.
                    publish_recovery_checkpoint()
                elif not recovery_published:
                    self._store.set_recovery_checkpoint(
                        handle.session_id,
                        operation_id,
                        fence_epoch=fence,
                        manifest_digest=runtime.checkpoint_digest,
                    )
                self._store.pause_advance(
                    handle.session_id,
                    operation_id,
                    fence_epoch=fence,
                    reason="cancel requested",
                )
                raise TrainingAdvancePaused(
                    f"advance paused at safe point (step {trainer.step_cursor} of {target_step})"
                )

            while trainer.step_cursor < target_step:
                pause_if_cancelled()
                next_step = trainer.step_cursor + 1
                writes_checkpoint = next_step == target_step or (
                    checkpoint_interval
                    and (next_step - handle.step_cursor) % checkpoint_interval == 0
                )
                writes_lora_export = bool(
                    lora_export_interval
                    and (next_step - handle.step_cursor) % lora_export_interval == 0
                )
                # The digest cadence anchors to the absolute step count, unlike
                # checkpoint_interval's advance-start anchor, so periodic checks
                # land on the same steps regardless of pause/resume boundaries.
                verifies_digest = bool(
                    writes_checkpoint
                    or writes_lora_export
                    or (sync_digest_interval and next_step % sync_digest_interval == 0)
                )
                self._train_step(runtime, verify_sync_digest=verifies_digest)
                if verifies_digest and runtime.workers is not None:
                    digest_checked_step = trainer.step_cursor
                if writes_lora_export:
                    self._write_intermediate_lora(
                        session_id=handle.session_id,
                        config_digest=handle.config_digest,
                        runtime=runtime,
                    )
                if writes_checkpoint:
                    publish_recovery_checkpoint()
                pause_if_cancelled()
            covered = self._store.read_events(handle.session_id, after=0, limit=1).latest_seq
            committed = self._store.commit_advance(
                handle.session_id,
                operation_id,
                fence_epoch=fence,
                output_manifest_digest=runtime.checkpoint_digest,
                output_step_cursor=target_step,
                covered_journal_seq=covered,
            )
            return self._committed_outcome(handle, committed, replayed=False)
        except TrainingAdvancePaused:
            raise
        except BaseException as exc:
            try:
                self._evict_runtime(handle.session_id, runtime)
            except BaseException as cleanup_exc:
                exc.add_note(f"distributed runtime cleanup failed: {cleanup_exc}")
            raise

    def export_lora(self, handle: TrainingSessionHandle, settings: str) -> tuple[str, str]:
        state = self._verify_handle(handle)
        config = self._config_from_mapping(state.config)
        if _config_digest(config) != handle.config_digest:
            raise TrainingLineageConflict("checkpoint config payload does not match its digest")
        export_settings = LoraExportSettings.parse(settings, export_root=self._export_root)
        if isinstance(config, MiniMaxH3TrainingConfig):
            path, digest = export_minimax_h3_lora(
                state,
                config,
                export_settings,
                runtime_identity=_runtime_identity(config),
            )
        elif isinstance(config, MiniMaxMusic3TrainingConfig):
            path, digest = export_minimax_music3_lora(
                state,
                config,
                export_settings,
                runtime_identity=_runtime_identity(config),
            )
        elif isinstance(config, WanTrainingConfig):
            path, digest = export_wan_lora(
                state,
                config,
                export_settings,
                runtime_identity=_runtime_identity(config),
            )
        elif isinstance(config, FluxTrainingConfig):
            path, digest = export_flux_lora(
                state,
                config,
                export_settings,
                runtime_identity=_runtime_identity(config),
            )
        elif isinstance(config, Flux2TrainingConfig):
            path, digest = export_flux2_lora(
                state,
                config,
                export_settings,
                runtime_identity=_runtime_identity(config),
            )
        elif isinstance(config, QwenImageTrainingConfig):
            path, digest = export_qwen_image_lora(
                state,
                config,
                export_settings,
                runtime_identity=_runtime_identity(config),
            )
        elif isinstance(config, Ideogram4TrainingConfig):
            path, digest = export_ideogram4_lora(
                state,
                config,
                export_settings,
                runtime_identity=_runtime_identity(config),
            )
        else:
            path, digest = export_kohya_lora(
                state,
                config,
                export_settings,
                runtime_identity=_runtime_identity(config),
            )
        return str(path), digest

    def _committed_outcome(
        self,
        handle: TrainingSessionHandle,
        record: TrainingOperationRecord,
        *,
        replayed: bool = True,
    ) -> AdvanceOutcome:
        state = self._checkpoints.load(record.output_manifest_digest)
        if (
            state.session_id != handle.session_id
            or state.config_digest != handle.config_digest
            or state.extension_snapshot_digest != handle.session_extension_snapshot_digest
            or state.parent_manifest_digest != handle.checkpoint_manifest_digest
            or state.step_cursor != record.output_step_cursor
            or state.loss is None
        ):
            raise TrainingLineageConflict(
                "committed checkpoint state does not match its ledger row"
            )
        output = TrainingSessionHandle(
            session_id=handle.session_id,
            checkpoint_manifest_digest=record.output_manifest_digest,
            step_cursor=record.output_step_cursor,
            config_digest=handle.config_digest,
            session_extension_snapshot_digest=handle.session_extension_snapshot_digest,
            journal_seq=record.output_journal_seq,
        )
        return AdvanceOutcome(handle=output, loss=state.loss, replayed=replayed)

    def complete(self, handle: TrainingSessionHandle) -> TrainingSessionHandle:
        self._verify_handle(handle)
        session = self._store.get_session(handle.session_id)
        assert session is not None
        if session.handle() != handle:
            raise TrainingLineageConflict(
                "handle does not name the session's committed head; complete from the head"
            )
        record = self._store.complete_session(
            handle.session_id, fence_epoch=self._fence(handle.session_id)
        )
        self._evict_runtime(handle.session_id)
        return record.handle()


class SD15LoRATrainingService(_LoRATrainingService):
    """Durable SD1.5 LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: ModelFactory = default_model_factory,
        data_source_factory: DataSourceFactory = default_data_source_factory,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="sd15",
            trainer_type=SD15LoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )


class SDXLLoRATrainingService(_LoRATrainingService):
    """Durable SDXL base LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: ModelFactory = default_model_factory,
        data_source_factory: DataSourceFactory = default_data_source_factory,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="sdxl",
            trainer_type=SDXLLoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )


class MiniMaxH3LoRATrainingService(_LoRATrainingService):
    """Durable MiniMax H3 DiT LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: MiniMaxH3ModelFactory = default_minimax_h3_model_factory,
        data_source_factory: MiniMaxH3DataSourceFactory = default_minimax_h3_data_source_factory,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="minimax-h3",
            trainer_type=MiniMaxH3LoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )


class MiniMaxMusic3LoRATrainingService(_LoRATrainingService):
    """Durable community-derived MiniMax Music 3 DiT LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: MiniMaxMusic3ModelFactory = default_minimax_music3_model_factory,
        data_source_factory: MiniMaxMusic3DataSourceFactory = (
            default_minimax_music3_data_source_factory
        ),
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="minimax-music3",
            trainer_type=MiniMaxMusic3LoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )


class WanLoRATrainingService(_LoRATrainingService):
    """Durable Wan DiT LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: WanModelFactory = default_wan_model_factory,
        data_source_factory: WanDataSourceFactory = default_wan_data_source_factory,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="wan",
            trainer_type=WanLoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )


class FluxLoRATrainingService(_LoRATrainingService):
    """Durable classic Flux DiT LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: FluxModelFactory = default_flux_model_factory,
        data_source_factory: FluxDataSourceFactory = default_flux_data_source_factory,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="flux",
            trainer_type=FluxLoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )


class Flux2LoRATrainingService(_LoRATrainingService):
    """Durable Flux2 DiT LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: Flux2ModelFactory = default_flux2_model_factory,
        data_source_factory: Flux2DataSourceFactory = default_flux2_data_source_factory,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="flux2",
            trainer_type=Flux2LoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )


class QwenImageLoRATrainingService(_LoRATrainingService):
    """Durable base Qwen-Image DiT LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: QwenImageModelFactory = default_qwen_image_model_factory,
        data_source_factory: QwenImageDataSourceFactory = default_qwen_image_data_source_factory,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="qwen-image",
            trainer_type=QwenImageLoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )


class Ideogram4LoRATrainingService(_LoRATrainingService):
    """Durable role-bound Ideogram 4 DiT LoRA training service."""

    def __init__(
        self,
        store: TrainingSessionStore,
        checkpoint_root: Path,
        *,
        scope: str = "local",
        model_factory: Ideogram4ModelFactory = default_ideogram4_model_factory,
        data_source_factory: Ideogram4DataSourceFactory = (default_ideogram4_data_source_factory),
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        super().__init__(
            store,
            checkpoint_root,
            expected_family="ideogram4",
            trainer_type=Ideogram4LoRATrainer,
            scope=scope,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
            cancelled=cancelled,
        )
