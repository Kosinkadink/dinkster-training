"""Execute one direct Dinkster LoRA comparison run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from dinkster_training_torch import CheckpointState, SD15LoRATrainer, TrainingConfig
from dinkster_training_torch.export import LoraExportSettings, export_lora

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nvidia_smi_process_memory_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    pid = str(os.getpid())
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == pid:
            return int(fields[1])
    return 0


def _memory_snapshot_summary() -> dict[str, object]:
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    block_states: Counter[str] = Counter()
    segment_types: Counter[str] = Counter()
    for segment in torch.cuda.memory_snapshot():
        segment_size = int(cast("int", segment["total_size"]))
        segment_types[str(segment["segment_type"])] += segment_size
        for block in cast("list[dict[str, object]]", segment["blocks"]):
            block_states[str(block["state"])] += int(cast("int", block["size"]))
    reserved = int(stats["reserved_bytes.all.current"])
    nvidia_smi_mib = _nvidia_smi_process_memory_mib()
    return {
        "active_bytes": int(stats["active_bytes.all.current"]),
        "allocated_bytes": int(stats["allocated_bytes.all.current"]),
        "block_state_bytes": dict(sorted(block_states.items())),
        "cuda_context_and_non_allocator_bytes": max(0, nvidia_smi_mib * 1024 * 1024 - reserved),
        "inactive_split_bytes": int(stats["inactive_split_bytes.all.current"]),
        "nvidia_smi_process_memory_mib": nvidia_smi_mib,
        "reserved_bytes": reserved,
        "segment_type_bytes": dict(sorted(segment_types.items())),
    }


def _tensor_bytes(tensors: list[torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _optimizer_tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_optimizer_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_optimizer_tensor_bytes(item) for item in value)
    return 0


def _source_frame(frames: object) -> str:
    if not isinstance(frames, list):
        return "<native>"
    for raw_frame in frames:
        if not isinstance(raw_frame, dict):
            continue
        filename = str(raw_frame.get("filename", ""))
        marker = "/site-packages/torch/"
        if marker in filename:
            relative = "torch/" + filename.split(marker, 1)[1]
            return f"{relative}:{raw_frame.get('line', 0)}:{raw_frame.get('name', '')}"
        try:
            relative = Path(filename).resolve().relative_to(REPOSITORY_ROOT).as_posix()
        except ValueError:
            continue
        return f"{relative}:{raw_frame.get('line', 0)}:{raw_frame.get('name', '')}"
    return "<native>"


def _allocator_peak_summary(snapshot: dict[str, object]) -> dict[str, object]:
    device_traces = cast("list[list[dict[str, object]]]", snapshot["device_traces"])
    trace = device_traces[torch.cuda.current_device()]
    live: dict[int, tuple[int, str]] = {}
    live_bytes = 0
    peak_bytes = 0
    peak_live: dict[int, tuple[int, str]] = {}
    for event in trace:
        action = event["action"]
        address = int(cast("int", event.get("addr", 0)))
        if action == "alloc":
            size = int(cast("int", event["size"]))
            live[address] = (size, _source_frame(event.get("frames")))
            live_bytes += size
        elif action == "free_requested":
            released = live.pop(address, None)
            if released is not None:
                live_bytes -= released[0]
        if live_bytes > peak_bytes:
            peak_bytes = live_bytes
            peak_live = dict(live)
    bytes_by_frame: Counter[str] = Counter()
    count_by_frame: Counter[str] = Counter()
    for size, frame in peak_live.values():
        bytes_by_frame[frame] += size
        count_by_frame[frame] += 1
    return {
        "recorded_peak_live_allocation_bytes": peak_bytes,
        "recorded_peak_live_allocation_count": len(peak_live),
        "top_source_frames": [
            {"bytes": size, "count": count_by_frame[frame], "frame": frame}
            for frame, size in bytes_by_frame.most_common(20)
        ],
        "trace_entry_count": len(trace),
    }


class _SavedTensorMeasurements:
    def __init__(self) -> None:
        self.total_bytes = 0
        self.total_count = 0
        self.by_grad_fn_bytes: Counter[str] = Counter()

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        size = tensor.numel() * tensor.element_size()
        grad_fn = type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else "leaf"
        self.total_bytes += size
        self.total_count += 1
        self.by_grad_fn_bytes[grad_fn] += size
        return tensor

    @staticmethod
    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def to_mapping(self) -> dict[str, object]:
        return {
            "by_grad_fn_bytes": dict(sorted(self.by_grad_fn_bytes.items())),
            "total_bytes": self.total_bytes,
            "total_count": self.total_count,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-attribution", action="store_true")
    parser.add_argument("--allocator-attribution", action="store_true")
    args = parser.parse_args()
    if args.allocator_attribution and not args.memory_attribution:
        raise ValueError("allocator attribution requires memory attribution")

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    family = str(comparison.get("family", "sd15"))
    if family != "sd15":
        raise ValueError("the current dinkster_comfy training layer supports SD1.5 only")
    model = args.model.resolve()
    model_digest = _sha256(model)
    if model_digest != comparison["model_sha256"]:
        raise ValueError(f"model SHA-256 mismatch: {model_digest}")
    dataset = (args.dataset.resolve() / "1_compare").resolve()
    config = TrainingConfig.from_mapping(
        {
            "schemaVersion": 1,
            "family": "sd15",
            "checkpointPath": str(model),
            "checkpointDigest": "sha256:" + model_digest,
            "dataset": {
                "root": str(dataset),
                "resolution": [comparison["resolution"], comparison["resolution"]],
                "encodedCacheRoot": str(args.output / "encoded-cache"),
            },
            "device": "cuda:0",
            "rank": comparison["rank"],
            "alpha": comparison["alpha"],
            "learningRate": comparison["learning_rate"],
            "weightDecay": comparison["weight_decay"],
            "batchSize": comparison["batch_size"],
            "gradientAccumulationSteps": comparison["gradient_accumulation_steps"],
            "seed": comparison["seed"],
            "checkpointInterval": 1,
        }
    )
    torch.use_deterministic_algorithms(True)
    memory_attribution: dict[str, object] | None = None
    peak_before_training_reset = {"allocated_bytes": 0, "reserved_bytes": 0}
    trainer = SD15LoRATrainer(config)
    if args.memory_attribution:
        memory_attribution = {"phases": {"after_trainer_load": _memory_snapshot_summary()}}
    if memory_attribution is not None:
        adapter_parameters = list(trainer.lora_program.parameters())
        adapter_ids = {id(parameter) for parameter in adapter_parameters}
        base_parameters = [
            parameter
            for parameter in trainer.model.parameters()
            if id(parameter) not in adapter_ids
        ]
        memory_attribution["resident_tensor_bytes"] = {
            "frozen_base_parameters": _tensor_bytes(base_parameters),
            "lora_master_parameters": _tensor_bytes(adapter_parameters),
            "model_buffers": _tensor_bytes(list(trainer.model.buffers())),
        }
        memory_attribution["lora_forward"] = {
            "formulation": "base-plus-low-rank-branch",
            "full_effective_weight_bytes_per_unet_forward": 0,
            "target_count": len(trainer.lora_program.targets),
        }
        phases = cast("dict[str, object]", memory_attribution["phases"])
        phases["after_unet_residency"] = _memory_snapshot_summary()
        peak_before_training_reset = {
            "allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        memory_attribution["peak_before_training_reset"] = peak_before_training_reset
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    losses: list[float] = []
    if memory_attribution is not None:
        saved_tensors = _SavedTensorMeasurements()
        record_memory_history: Callable[..., None] | None = None
        allocator_snapshot: dict[str, object] | None = None
        if args.allocator_attribution:
            record_memory_history = cast(
                "Callable[..., None]", vars(torch.cuda.memory)["_record_memory_history"]
            )
            record_memory_history(
                enabled="all",
                context="alloc",
                stacks="python",
                max_entries=200_000,
                clear_history=True,
            )
        try:
            with torch.autograd.graph.saved_tensors_hooks(saved_tensors.pack, saved_tensors.unpack):
                losses.append(trainer.train_step())
            if args.allocator_attribution:
                snapshot = cast(
                    "Callable[[], dict[str, object]]", vars(torch.cuda.memory)["_snapshot"]
                )
                allocator_snapshot = snapshot()
        finally:
            if record_memory_history is not None:
                record_memory_history(enabled=None)
        torch.cuda.synchronize()
        adapter_parameters = list(trainer.lora_program.parameters())
        latent_elements = (
            comparison["batch_size"]
            * 4
            * (comparison["resolution"] // 8)
            * (comparison["resolution"] // 8)
        )
        memory_attribution["checkpoint_boundary_input_bytes"] = (
            latent_elements * 4
            + comparison["batch_size"] * 8
            + comparison["batch_size"] * 77 * 768 * 4
        )
        memory_attribution["first_step"] = {
            "gradient_bytes": _tensor_bytes(
                [parameter.grad for parameter in adapter_parameters if parameter.grad is not None]
            ),
            "optimizer_state_bytes": _optimizer_tensor_bytes(trainer.optimizer.state_dict()),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "saved_tensors": saved_tensors.to_mapping(),
        }
        if allocator_snapshot is not None:
            memory_attribution["first_step_allocator_replay"] = _allocator_peak_summary(
                allocator_snapshot
            )
        phases = cast("dict[str, object]", memory_attribution["phases"])
        phases["after_first_step"] = _memory_snapshot_summary()
    losses.extend(trainer.train_step() for _ in range(comparison["steps"] - len(losses)))
    torch.cuda.synchronize()
    runtime_seconds = time.monotonic() - started

    args.output.mkdir(parents=True, exist_ok=True)
    export_path = args.output / "adapter.safetensors"
    state = CheckpointState(
        manifest_digest="sha256:" + "0" * 64,
        session_id="training-comparison",
        config_digest="sha256:" + "0" * 64,
        extension_snapshot_digest="sha256:" + "0" * 64,
        parent_manifest_digest="",
        step_cursor=trainer.step_cursor,
        config=config.to_mapping(),
        adapter=trainer.lora_program.state_dict(),
        optimizer={},
        rng={},
        data_cursor=trainer.data_cursor,
        loss=trainer.last_loss,
    )
    _, export_digest = export_lora(
        state,
        LoraExportSettings(path=export_path, dtype="fp32"),
        runtime_identity="dinkster-comfy-training-comparison",
    )
    result = {
        "adapter": str(export_path),
        "adapter_sha256": export_digest.removeprefix("sha256:"),
        "base_dtype": "float32",
        "deterministic_algorithms": True,
        "gradient_checkpointing": False,
        "losses": losses,
        "max_memory_allocated_bytes": max(
            peak_before_training_reset["allocated_bytes"], torch.cuda.max_memory_allocated()
        ),
        "max_memory_reserved_bytes": max(
            peak_before_training_reset["reserved_bytes"], torch.cuda.max_memory_reserved()
        ),
        "optimizer": "adamw",
        "runtime_seconds": runtime_seconds,
        "steps": trainer.step_cursor,
        "torch_version": torch.__version__,
    }
    if memory_attribution is not None:
        phases = cast("dict[str, object]", memory_attribution["phases"])
        phases["after_final_step"] = _memory_snapshot_summary()
        result["memory_attribution"] = memory_attribution
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    trainer.close()


if __name__ == "__main__":
    main()
