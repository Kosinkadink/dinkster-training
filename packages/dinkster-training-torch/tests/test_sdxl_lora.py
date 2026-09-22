from __future__ import annotations

import contextlib
import gc
import json
import math
import struct
import weakref
import zlib
from collections.abc import Generator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from dinkster_inference import (
    CLIP_G_PROFILE,
    CLIP_L_PROFILE,
    FLOAT32,
    SDXL,
    SDXL_REFINER,
    SDXL_UNET_CONFIG,
    ClipTextConfig,
    ComponentPlan,
    KLConfig,
    LayerQuant,
    LinearToConv2D,
    ModelFamily,
    Parameterization,
    PromptTokenizer,
    RowChunk,
    SamplingDescriptor,
    SamplingSpace,
    SDAssemblyPlan,
    Transpose2D,
    UNetConfig,
    decode_lora,
    load_clip_bpe,
    load_safetensors_header,
    native_unet_key_map,
    pack_spans,
)
from dinkster_inference_torch import (
    SDXL_CLIP_POLICY,
    AutoencoderKL,
    ClipTextEncoder,
    ClipTextModel,
    UNetModel,
    build_patch_set,
    compose_sdxl_conditioning,
    load_tensors,
    patch_weights,
)
from dinkster_server import JournalStore, TrainingSessionStore
from dinkster_training_torch import (
    SDXL_TRAINING_RUNTIME_IDENTITY,
    SDXL_TRAINING_SNAPSHOT_DIGEST,
    CheckpointState,
    ContentAddressedCheckpointStore,
    ImageCaptionDatasetSource,
    PreparedBatch,
    SD15LoRATrainingService,
    SDXLLoRATrainer,
    SDXLLoRATrainingService,
    TrainingAdvancePaused,
    TrainingConfig,
    TrainingConfigError,
    capability_report,
)
from dinkster_training_torch.attachment import resolve_lora_targets
from dinkster_training_torch.data import load_planned_component, validate_sdxl_training_plan
from dinkster_training_torch.dataset import (
    EncoderStateSource,
    ImageCaptionDatasetSettings,
    inspect_image_caption_dataset,
)
from PIL import Image

_MASK32 = 0xFFFFFFFF
_UNET = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1,),
    channel_mult=(1,),
    transformer_depth=(1,),
    transformer_depth_output=(1, 1),
    transformer_depth_middle=1,
    context_dim=24,
    use_linear_in_transformer=True,
    adm_in_channels=1544,
    num_head_channels=8,
)
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
_CLIP_L = ClipTextConfig(
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=32,
    hidden_act="quick_gelu",
)
_CLIP_G = ClipTextConfig(
    hidden_size=8,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=16,
    hidden_act="gelu",
)


def _unet_wire(config: UNetConfig) -> dict[str, object]:
    return {
        "inChannels": config.in_channels,
        "outChannels": config.out_channels,
        "modelChannels": config.model_channels,
        "numResBlocks": list(config.num_res_blocks),
        "channelMult": list(config.channel_mult),
        "transformerDepth": list(config.transformer_depth),
        "transformerDepthOutput": list(config.transformer_depth_output),
        "transformerDepthMiddle": config.transformer_depth_middle,
        "contextDim": config.context_dim,
        "useLinearInTransformer": config.use_linear_in_transformer,
        "admInChannels": config.adm_in_channels,
        "numHeads": config.num_heads,
        "numHeadChannels": config.num_head_channels,
    }


def config_mapping(
    *,
    device: str = "cpu",
    base_dtype: str = "float32",
    optimizer: str = "adamw",
) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "family": "sdxl",
        "unet": _unet_wire(_UNET),
        "device": device,
        "baseDtype": base_dtype,
        "rank": 2,
        "alpha": 2.0,
        "learningRate": 0.0005,
        "weightDecay": 0.01,
        "optimizer": optimizer,
        "gradientAccumulationSteps": 1,
        "gradientCheckpointing": True,
        "seed": 1234,
        "latentShape": [1, 4, 2, 2],
        "contextShape": [1, 2, 24],
        "pooledShape": [1, 8],
    }


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
    values = 1.0 + (uniform - 0.5) * 0.1 if len(shape) == 1 else (uniform - 0.5) * 0.04
    return values.reshape(shape)


def _filled(module: torch.nn.Module) -> torch.nn.Module:
    module.load_state_dict(
        {
            key: _fill_value(key, tuple(value.shape)).to(dtype=value.dtype)
            for key, value in module.state_dict().items()
        },
        strict=True,
    )
    return module


def model_factory(config: TrainingConfig) -> torch.nn.Module:
    return _filled(UNetModel(config.unet))


class SyntheticBatches:
    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def batch(
        self,
        cursor: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> PreparedBatch:
        del generator, device
        assert self._config.pooled_shape is not None
        height = self._config.latent_shape[2] * 8
        width = self._config.latent_shape[3] * 8
        return PreparedBatch(
            latents=_fill_value(f"latent:{cursor}", self._config.latent_shape),
            text_embeddings=_fill_value(f"text:{cursor}", self._config.context_shape),
            pooled_embeddings=_fill_value(f"pooled:{cursor}", self._config.pooled_shape),
            time_ids=torch.tensor([[height, width, 0, 0, height, width]]),
        )


def data_source_factory(config: TrainingConfig) -> SyntheticBatches:
    return SyntheticBatches(config)


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
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            assert_tree_equal(left_item, right_item)
    else:
        assert left == right


@contextlib.contextmanager
def deterministic_algorithms() -> Generator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _component_plan(
    component: str,
    config: object,
    *,
    identity_facts: tuple[str, ...] = (),
    quant: bool = False,
) -> ComponentPlan[object]:
    keys = {"layer.weight": "layer.weight"} if quant else {}
    layer_quant = (
        {
            "layer": LayerQuant(
                layer="layer",
                format="float8_e4m3fn",
                weight="layer.weight",
                weight_scale="layer.scale_weight",
            )
        }
        if quant
        else {}
    )
    return ComponentPlan(
        component=component,
        path=Path("/unused.safetensors"),
        config=config,
        keys=keys,
        dtypes={key: FLOAT32 for key in keys},
        quant=layer_quant,
        identity_facts=identity_facts,
    )


def _sdxl_plan(
    *,
    family: ModelFamily = SDXL,
    sampling: SamplingDescriptor | None = None,
    clip_l: bool = True,
    quantized_diffusion: bool = False,
) -> SDAssemblyPlan:
    selected_sampling = family.sampling if sampling is None else sampling
    identity_facts = ()
    if selected_sampling != family.sampling:
        identity_facts = (f"parameterization={selected_sampling.parameterization.value}",)
        if selected_sampling.space is SamplingSpace.CONTINUOUS_EDM:
            identity_facts += (
                f"sampling_space={selected_sampling.space.value}",
                f"sigma_min={selected_sampling.sigma_min!r}",
                f"sigma_max={selected_sampling.sigma_max!r}",
            )
        else:
            identity_facts += (f"zsnr={selected_sampling.zsnr}",)
    return SDAssemblyPlan(
        family=family,
        diffusion=cast(
            "ComponentPlan[UNetConfig]",
            _component_plan(
                "diffusion",
                _UNET,
                identity_facts=identity_facts,
                quant=quantized_diffusion,
            ),
        ),
        clip_l=(
            cast("ComponentPlan[ClipTextConfig]", _component_plan("clip_l", _CLIP_L))
            if clip_l
            else None
        ),
        clip_g=cast("ComponentPlan[ClipTextConfig]", _component_plan("clip_g", _CLIP_G)),
        vae=cast("ComponentPlan[KLConfig]", _component_plan("vae", _TINY_KL)),
        sampling=selected_sampling,
    )


def test_sdxl_training_plan_rejects_refiner_v_prediction_missing_tower_and_quantization() -> None:
    with pytest.raises(ValueError, match="plain epsilon-prediction SDXL base"):
        validate_sdxl_training_plan(_sdxl_plan(family=SDXL_REFINER, clip_l=False))

    v_prediction = SamplingDescriptor(
        parameterization=Parameterization.V_PREDICTION,
        sigma_min=SDXL.sampling.sigma_min,
        sigma_max=SDXL.sampling.sigma_max,
    )
    edm = SamplingDescriptor(
        parameterization=Parameterization.EDM,
        sigma_min=0.002,
        sigma_max=80.0,
        space=SamplingSpace.CONTINUOUS_EDM,
    )
    for variant in (v_prediction, edm):
        with pytest.raises(ValueError, match="plain epsilon-prediction SDXL base"):
            validate_sdxl_training_plan(_sdxl_plan(sampling=variant))

    with pytest.raises(ValueError, match="requires checkpoint CLIP-L and CLIP-G"):
        validate_sdxl_training_plan(_sdxl_plan(clip_l=False))

    with pytest.raises(ValueError, match="quantized training components are not supported"):
        validate_sdxl_training_plan(_sdxl_plan(quantized_diffusion=True))


class _PlannedEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(4, 2)


class _PlannedTextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = _PlannedEmbeddings()


class _PlannedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.chunk = torch.nn.Parameter(torch.empty(2, 3))
        self.transposed = torch.nn.Parameter(torch.empty(2, 3))
        self.conv = torch.nn.Parameter(torch.empty(2, 3, 1, 1))
        self.text_model = _PlannedTextModel()
        self.text_projection = torch.nn.Linear(2, 2, bias=False)


def _write_float32_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    for key, tensor in tensors.items():
        value = tensor.detach().contiguous()
        data = bytes(value.untyped_storage())[: value.numel() * value.element_size()]
        header[key] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [len(payload), len(payload) + len(data)],
        }
        payload.extend(data)
    encoded = json.dumps(header, separators=(",", ":")).encode("ascii")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(payload))


def test_planned_component_loader_applies_all_sdxl_checkpoint_transforms(
    tmp_path: Path,
) -> None:
    tensors = {
        "fused": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "to_transpose": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "to_conv": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "tokens": torch.arange(8, dtype=torch.float32).reshape(4, 2),
    }
    path = tmp_path / "component.safetensors"
    _write_float32_safetensors(path, tensors)
    keys = {
        "chunk": "fused",
        "transposed": "to_transpose",
        "conv": "to_conv",
        "text_model.embeddings.token_embedding.weight": "tokens",
    }
    plan = ComponentPlan(
        component="test",
        path=path,
        config=None,
        keys=keys,
        dtypes={key: FLOAT32 for key in keys},
        quant={},
        absent=("text_projection.weight",),
        transforms={
            "chunk": RowChunk(parts=2, part=1),
            "transposed": Transpose2D(),
            "conv": LinearToConv2D(),
        },
    )
    loaded = load_planned_component(plan, lambda _config: _PlannedModule())
    assert torch.equal(loaded.chunk, tensors["fused"][2:])
    assert torch.equal(loaded.transposed, tensors["to_transpose"].transpose(0, 1))
    assert torch.equal(loaded.conv, tensors["to_conv"].reshape(2, 3, 1, 1))
    assert torch.equal(
        loaded.text_model.embeddings.token_embedding.weight,
        tensors["tokens"],
    )
    assert torch.equal(loaded.text_projection.weight, torch.eye(2))


def test_sdxl_capability_enumerates_native_projection_tree() -> None:
    full = TrainingConfig.from_mapping({"family": "sdxl"})
    report = capability_report(full)
    targets = cast("list[dict[str, object]]", report["targets"])

    assert full.checkpointing_mode == "blockReentrant"
    assert full.to_mapping()["checkpointingMode"] == "blockReentrant"
    assert report["trainer"] == SDXL_TRAINING_RUNTIME_IDENTITY == "sdxl-lora-torch/1"
    assert (
        report["sessionExtensionSnapshotDigest"]
        == SDXL_TRAINING_SNAPSHOT_DIGEST
        == "blake3:979d00f897d9d7d90458753d8e71a467f4460f8937c75adc2fbe5fdcedf353fc"
    )
    assert len(targets) == 722
    assert {target["operation"] for target in targets} == {"linear"}
    assert all(str(target["targetId"]).startswith("sdxl/unet/") for target in targets)
    assert any(str(target["modulePath"]).endswith("proj_in") for target in targets)
    assert any(str(target["modulePath"]).endswith("proj_out") for target in targets)
    counts = cast("dict[str, int]", report["parameterCounts"])
    assert counts["frozenBase"] == 2_567_463_684
    assert cast("dict[str, object]", report["capabilities"])["families"] == ["sdxl"]


def test_sdxl_config_refuses_wrong_adm_and_sd15_backend(tmp_path: Path) -> None:
    mapping = config_mapping()
    unet = cast("dict[str, object]", mapping["unet"])
    unet["admInChannels"] = 1543
    with pytest.raises(TrainingConfigError, match="six 256-wide"):
        TrainingConfig.from_mapping(mapping)

    mapping = config_mapping()
    mapping["baseState"] = {
        "path": "/models/sdxl.safetensors",
        "digest": "blake3:" + "2" * 64,
        "prefix": "model.diffusion_model.",
    }
    with pytest.raises(TrainingConfigError, match="prefix must be empty"):
        TrainingConfig.from_mapping(mapping)

    store = make_store(tmp_path / "training.sqlite")
    try:
        service = SD15LoRATrainingService(
            store,
            tmp_path / "checkpoints",
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        with pytest.raises(TrainingConfigError, match="family must be 'sd15'"):
            service.dry_run(serialized_config())
    finally:
        store.close()


@pytest.mark.parametrize("base_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("optimizer", ["adamw", "factored-adamw"])
def test_sdxl_resume_and_safe_point_cancellation_are_bit_exact(
    tmp_path: Path,
    optimizer: str,
    base_dtype: str,
) -> None:
    config = serialized_config(optimizer=optimizer, baseDtype=base_dtype)
    uninterrupted_root = tmp_path / "uninterrupted"
    resumed_root = tmp_path / "resumed"
    uninterrupted_store = make_store(tmp_path / "uninterrupted.sqlite")
    resumed_store = make_store(tmp_path / "resumed.sqlite")
    try:
        with deterministic_algorithms():
            uninterrupted = SDXLLoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
            )
            initial_a, _ = uninterrupted.create("sdxl-resume", config)
            final_a = uninterrupted.advance(initial_a, 2, "same operation").handle

            armed = False
            paused: SDXLLoRATrainingService

            def cancelled() -> bool:
                return armed and paused.steps_run >= 1

            paused = SDXLLoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("sdxl-resume", config)
            armed = True
            with pytest.raises(TrainingAdvancePaused, match="step 1 of 2"):
                paused.advance(initial_b, 2, "same operation")
            successor = SDXLLoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=data_source_factory,
            )
            final_b = successor.advance(initial_b, 2, "same operation").handle

        state_a = checkpoint_state(uninterrupted_root, final_a.checkpoint_manifest_digest)
        state_b = checkpoint_state(resumed_root, final_b.checkpoint_manifest_digest)
        assert state_a.loss == state_b.loss
        assert state_a.step_cursor == state_b.step_cursor == 2
        assert_tree_equal(state_a.adapter, state_b.adapter)
        assert_tree_equal(state_a.optimizer, state_b.optimizer)
        assert_tree_equal(state_a.rng, state_b.rng)
    finally:
        uninterrupted_store.close()
        resumed_store.close()


def test_sdxl_export_is_deterministic_and_round_trips_native_inference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkpoints"
    store = make_store(tmp_path / "training.sqlite")
    try:
        service = SDXLLoRATrainingService(
            store,
            root,
            model_factory=model_factory,
            data_source_factory=data_source_factory,
        )
        initial, _ = service.create("sdxl-export", serialized_config(loraExportInterval=1))
        final = service.advance(initial, 1, "train before export").handle
        cadence = (
            root / "exports" / "cadence" / initial.session_id / "step-000000000001.safetensors"
        )
        assert cadence.is_file()
        assert load_safetensors_header(cadence).metadata()["dinkster_step_cursor"] == "1"
        first_path, first_digest = service.export_lora(final, '{"path":"first.safetensors"}')
        second_path, second_digest = service.export_lora(final, '{"path":"second.safetensors"}')
        first = Path(first_path)
        second = Path(second_path)
        assert first.read_bytes() == second.read_bytes()
        assert first_digest == second_digest

        source = load_safetensors_header(first)
        assert source.metadata()["dinkster_runtime_identity"] == "sdxl-lora-torch/1"
        state = checkpoint_state(root, final.checkpoint_manifest_digest)
        config = TrainingConfig.from_mapping(state.config)
        model = model_factory(config)
        targets = resolve_lora_targets(model, config.rank, family="sdxl")
        model_keys = [f"diffusion_model.{key}" for key in model.state_dict()]
        decoded = decode_lora(
            {key: source.entry(key).geometry for key in source.keys()},
            native_unet_key_map(model_keys),
        )
        assert decoded.unmatched == () and decoded.diagnostics == ()
        assert len(decoded.patches) == len(targets)
        patch_set = build_patch_set(decoded.patches, load_tensors(first))
        weights = {
            f"diffusion_model.{target.module_path}.weight": torch.zeros(target.weight_shape)
            for target in targets
        }
        patch_weights(weights, patch_set)
        for target in targets:
            down = state.adapter[f"{target.target_id}.down"]
            up = state.adapter[f"{target.target_id}.up"]
            cast_expected = (up.half().float() @ down.half().float()).reshape(
                target.weight_shape
            ) * (torch.tensor(config.alpha).half().float() / config.rank)
            actual = weights[f"diffusion_model.{target.module_path}.weight"]
            torch.testing.assert_close(actual, cast_expected, rtol=0.0, atol=0.0)
    finally:
        store.close()


def _sdxl_dataset_settings(
    tmp_path: Path,
    *,
    item_count: int = 1,
    encoded_cache_root: Path | None = None,
) -> ImageCaptionDatasetSettings:
    root = tmp_path / "dataset"
    root.mkdir(parents=True)
    entries = (
        ("image.png", (40, 80, 160), "blue geometric shape"),
        ("alpha.png", (180, 20, 40), "red round shape"),
        ("zeta.png", (30, 160, 70), "green angular shape"),
    )
    for name, color, caption in entries[:item_count]:
        Image.new("RGB", (10, 8), color).save(root / name)
        (root / name).with_suffix(".txt").write_text(caption, encoding="utf-8")
    checkpoint = EncoderStateSource(
        path=str(tmp_path / "unused-standard-sdxl.safetensors"),
        digest="blake3:" + "1" * 64,
        prefix="",
    )
    inspection = inspect_image_caption_dataset(
        root,
        (16, 16),
        None,
        None,
        family="sdxl",
        checkpoint_state=checkpoint,
    )
    return ImageCaptionDatasetSettings(
        family="sdxl",
        root=str(root),
        resolution=(16, 16),
        vae_state=None,
        text_encoder_state=None,
        checkpoint_state=checkpoint,
        inspection=inspection,
        encoded_cache_root=(
            None if encoded_cache_root is None else str(encoded_cache_root.resolve())
        ),
    )


def tiny_vae() -> AutoencoderKL:
    return cast("AutoencoderKL", _filled(AutoencoderKL(_TINY_KL)))


def tiny_clip_l() -> ClipTextModel:
    return cast("ClipTextModel", _filled(ClipTextModel(_CLIP_L)))


def tiny_clip_g() -> ClipTextModel:
    return cast("ClipTextModel", _filled(ClipTextModel(_CLIP_G)))


def tracked_sdxl_source(
    settings: ImageCaptionDatasetSettings,
    loaded: list[str],
    *,
    batch_size: int = 1,
) -> ImageCaptionDatasetSource:
    def vae_factory() -> AutoencoderKL:
        loaded.append("vae")
        return tiny_vae()

    def clip_l_factory() -> ClipTextModel:
        loaded.append("clip_l")
        return tiny_clip_l()

    def clip_g_factory() -> ClipTextModel:
        loaded.append("clip_g")
        return tiny_clip_g()

    return ImageCaptionDatasetSource(
        settings,
        vae_factory,
        clip_l_factory,
        clip_g_factory,
        batch_size=batch_size,
        device=torch.device("cpu"),
    )


def assert_sdxl_batch_equal(left: PreparedBatch, right: PreparedBatch) -> None:
    for name in ("latents", "text_embeddings", "pooled_embeddings", "time_ids"):
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        assert isinstance(left_value, torch.Tensor)
        assert isinstance(right_value, torch.Tensor)
        assert torch.equal(left_value, right_value)


def sdxl_dataset_config(settings: ImageCaptionDatasetSettings) -> TrainingConfig:
    mapping = config_mapping()
    mapping["contextShape"] = [1, 77, 24]
    mapping["dataset"] = settings.to_mapping()
    return TrainingConfig.from_mapping(mapping)


def test_sdxl_dataset_precomputes_vae_then_clip_l_then_clip_g(
    tmp_path: Path,
) -> None:
    settings = _sdxl_dataset_settings(tmp_path)
    loaded: list[str] = []
    references: list[weakref.ReferenceType[torch.nn.Module]] = []

    def vae_factory() -> AutoencoderKL:
        model = tiny_vae()
        loaded.append("vae")
        references.append(weakref.ref(model))
        return model

    def clip_l_factory() -> ClipTextModel:
        gc.collect()
        assert references[-1]() is None
        model = tiny_clip_l()
        loaded.append("clip_l")
        references.append(weakref.ref(model))
        return model

    def clip_g_factory() -> ClipTextModel:
        gc.collect()
        assert references[-1]() is None
        model = tiny_clip_g()
        loaded.append("clip_g")
        references.append(weakref.ref(model))
        return model

    source = ImageCaptionDatasetSource(
        settings,
        vae_factory,
        clip_l_factory,
        clip_g_factory,
        batch_size=1,
        device=torch.device("cpu"),
    )
    gc.collect()
    batch = source.batch(
        0,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )

    assert loaded == ["vae", "clip_l", "clip_g"]
    assert all(reference() is None for reference in references)
    tokenizer = PromptTokenizer(encode_word=load_clip_bpe().encode, disable_weights=True)
    spans = tokenizer.tokenize("blue geometric shape")
    clip_l = ClipTextEncoder(tiny_clip_l(), policy=SDXL_CLIP_POLICY).encode_chunks(
        pack_spans(spans, CLIP_L_PROFILE)[:1]
    )
    clip_g = ClipTextEncoder(
        tiny_clip_g(), profile=CLIP_G_PROFILE, policy=SDXL_CLIP_POLICY
    ).encode_chunks(pack_spans(spans, CLIP_G_PROFILE)[:1])
    expected = compose_sdxl_conditioning(clip_l, clip_g)
    assert torch.equal(batch.text_embeddings, expected.embeddings)
    assert expected.pooled is not None
    assert batch.pooled_embeddings is not None
    pooled = batch.pooled_embeddings
    assert torch.equal(pooled, expected.pooled)
    assert batch.time_ids is not None
    assert torch.equal(batch.time_ids, torch.tensor([[16.0, 16.0, 0.0, 0.0, 16.0, 16.0]]))
    assert batch.latents.shape == (1, 4, 2, 2)


def test_sdxl_encoded_cache_matches_memory_and_skips_all_encoders(tmp_path: Path) -> None:
    settings = _sdxl_dataset_settings(tmp_path, item_count=3)
    cached_settings = replace(
        settings,
        encoded_cache_root=str((tmp_path / "encoded-cache").resolve()),
    )
    memory_loaded: list[str] = []
    miss_loaded: list[str] = []
    hit_loaded: list[str] = []
    memory = tracked_sdxl_source(settings, memory_loaded, batch_size=2)
    miss = tracked_sdxl_source(cached_settings, miss_loaded, batch_size=2)
    hit = tracked_sdxl_source(cached_settings, hit_loaded, batch_size=2)

    assert memory_loaded == miss_loaded == ["vae", "clip_l", "clip_g"]
    assert hit_loaded == []
    for cursor in (0, 1, 2, 3, 4, 7):
        generator = torch.Generator().manual_seed(cursor)
        expected = memory.batch(cursor, generator=generator, device=torch.device("cpu"))
        actual_miss = miss.batch(cursor, generator=generator, device=torch.device("cpu"))
        actual_hit = hit.batch(cursor, generator=generator, device=torch.device("cpu"))
        assert_sdxl_batch_equal(expected, actual_miss)
        assert_sdxl_batch_equal(expected, actual_hit)

    different_batch_loaded: list[str] = []
    tracked_sdxl_source(cached_settings, different_batch_loaded, batch_size=1)
    assert different_batch_loaded == ["vae", "clip_l", "clip_g"]
    cache_root = Path(cast("str", cached_settings.encoded_cache_root))
    assert len(tuple((cache_root / "v1" / "entries").glob("*.json"))) == 2

    report = capability_report(sdxl_dataset_config(cached_settings))
    dataset = cast("dict[str, object]", report["dataset"])
    assert dataset["encodedCache"] == {"state": "hit"}
    memory_report = cast("dict[str, object]", report["memoryLedger"])
    categories = cast("dict[str, int]", memory_report["categories"])
    assert categories["vaeEncoderParametersTransientLowerBound"] == 0
    assert categories["clipLTextEncoderParametersTransientLowerBound"] == 0
    assert categories["clipGTextEncoderParametersTransientLowerBound"] == 0
    assert categories["encoderInputBatchTransientLowerBound"] == 0
    precision = cast("dict[str, str]", report["precisionPlan"])
    assert precision["datasetPrecomputeEncoders"] == "not loaded on encoded cache hit"


def test_sdxl_encoded_cache_repairs_corruption_and_is_digest_scoped(tmp_path: Path) -> None:
    cache_root = tmp_path / "encoded-cache"
    settings = _sdxl_dataset_settings(
        tmp_path,
        item_count=3,
        encoded_cache_root=cache_root,
    )
    expected = tracked_sdxl_source(settings, []).batch(
        4,
        generator=torch.Generator().manual_seed(1),
        device=torch.device("cpu"),
    )
    manifest_path = next((cache_root / "v1" / "entries").glob("*.json"))
    manifest = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="ascii")))
    digest = cast("str", manifest["shardDigest"])
    shard = cache_root / "v1" / "shards" / f"{digest[7:]}.safetensors"
    full_size = shard.stat().st_size
    shard.write_bytes(shard.read_bytes()[:32])

    repaired_loaded: list[str] = []
    repaired = tracked_sdxl_source(settings, repaired_loaded).batch(
        4,
        generator=torch.Generator().manual_seed(2),
        device=torch.device("cpu"),
    )
    assert repaired_loaded == ["vae", "clip_l", "clip_g"]
    assert shard.stat().st_size == full_size
    assert_sdxl_batch_equal(expected, repaired)

    caption = Path(settings.root) / "image.txt"
    caption.write_text("changed blue shape", encoding="utf-8")
    changed_inspection = inspect_image_caption_dataset(
        Path(settings.root),
        settings.resolution,
        None,
        None,
        family="sdxl",
        checkpoint_state=settings.checkpoint_state,
    )
    changed = replace(settings, inspection=changed_inspection)
    assert changed.digest != settings.digest
    changed_loaded: list[str] = []
    tracked_sdxl_source(changed, changed_loaded)
    assert changed_loaded == ["vae", "clip_l", "clip_g"]
    assert len(tuple((cache_root / "v1" / "entries").glob("*.json"))) == 2


def test_sdxl_resume_from_persisted_encoded_cache_is_bit_exact(tmp_path: Path) -> None:
    cache_root = tmp_path / "encoded-cache"
    settings = _sdxl_dataset_settings(
        tmp_path,
        item_count=3,
        encoded_cache_root=cache_root,
    )
    with deterministic_algorithms():
        tracked_sdxl_source(settings, [])
    config = sdxl_dataset_config(settings)
    serialized = json.dumps(config.to_mapping(), sort_keys=True)
    uninterrupted_root = tmp_path / "cache-uninterrupted-checkpoints"
    resumed_root = tmp_path / "cache-resumed-checkpoints"
    uninterrupted_store = make_store(tmp_path / "cache-uninterrupted.sqlite")
    resumed_store = make_store(tmp_path / "cache-resumed.sqlite")
    loaded: list[str] = []
    resumed_roots: list[str | None] = []

    def cached_source(config: TrainingConfig) -> ImageCaptionDatasetSource:
        assert config.dataset is not None
        resumed_roots.append(config.dataset.encoded_cache_root)
        return tracked_sdxl_source(config.dataset, loaded)

    try:
        with deterministic_algorithms():
            uninterrupted = SDXLLoRATrainingService(
                uninterrupted_store,
                uninterrupted_root,
                model_factory=model_factory,
                data_source_factory=cached_source,
            )
            initial_a, _ = uninterrupted.create("sdxl-cache-resume", serialized)
            final_a = uninterrupted.advance(initial_a, 2, "same operation").handle

            armed = False
            paused: SDXLLoRATrainingService

            def cancelled() -> bool:
                return armed and paused.steps_run >= 1

            paused = SDXLLoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=cached_source,
                cancelled=cancelled,
            )
            initial_b, _ = paused.create("sdxl-cache-resume", serialized)
            initial_state = checkpoint_state(resumed_root, initial_b.checkpoint_manifest_digest)
            persisted_dataset = cast("dict[str, object]", initial_state.config["dataset"])
            assert persisted_dataset["encodedCacheRoot"] == str(cache_root.resolve())
            armed = True
            with pytest.raises(TrainingAdvancePaused, match="step 1 of 2"):
                paused.advance(initial_b, 2, "same operation")
            successor = SDXLLoRATrainingService(
                resumed_store,
                resumed_root,
                model_factory=model_factory,
                data_source_factory=cached_source,
            )
            final_b = successor.advance(initial_b, 2, "same operation").handle

        state_a = checkpoint_state(uninterrupted_root, final_a.checkpoint_manifest_digest)
        state_b = checkpoint_state(resumed_root, final_b.checkpoint_manifest_digest)
        assert loaded == []
        assert resumed_roots and set(resumed_roots) == {str(cache_root.resolve())}
        assert state_a.loss == state_b.loss
        assert_tree_equal(state_a.adapter, state_b.adapter)
        assert_tree_equal(state_a.optimizer, state_b.optimizer)
        assert_tree_equal(state_a.rng, state_b.rng)
    finally:
        uninterrupted_store.close()
        resumed_store.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("optimizer", ["adamw", "factored-adamw"])
def test_cuda_sdxl_bfloat16_step_keeps_float32_lora_masters(optimizer: str) -> None:
    config = TrainingConfig.from_mapping(
        config_mapping(device="cuda", base_dtype="bfloat16", optimizer=optimizer)
    )
    trainer = SDXLLoRATrainer(config, model_factory(config), SyntheticBatches(config))
    assert math.isfinite(trainer.train_step())
    torch.cuda.synchronize()
    assert all(
        parameter.device.type == "cuda"
        and parameter.dtype == torch.float32
        and parameter.requires_grad
        for parameter in trainer.attachment.parameters()
    )
    assert all(not parameter.requires_grad for parameter in trainer.model.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in trainer.model.parameters())


def test_standard_sdxl_profile_has_only_supported_target_classes() -> None:
    with torch.device("meta"):
        model = UNetModel(SDXL_UNET_CONFIG)
    targets = resolve_lora_targets(model, 4, family="sdxl")
    assert len(targets) == 722
    assert {target.operation for target in targets} == {"linear"}
