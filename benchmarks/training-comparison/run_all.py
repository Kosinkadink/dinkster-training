"""Orchestrate the pinned SD1.5 and SDXL LoRA comparison runs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_URLS = {
    "sd15": (
        "https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/"
        "c36740b77a55ec396ace7c8c26589cdf2b4bc3da/v1-5-pruned-emaonly.safetensors"
    ),
    "sdxl": (
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/"
        "462165984030d82259a11f4367a4eed129e94a7b/sd_xl_base_1.0.safetensors"
    ),
}
SD15_RUNS = (
    "dinkster-adamw-a",
    "dinkster-adamw-b",
    "dinkster-factored",
    "dinkster-fp32-adamw-a",
    "dinkster-fp32-adamw-b",
    "kohya-gc",
    "kohya-no-gc",
    "ai-toolkit",
)
SDXL_RUNS = (
    "sdxl-dinkster-adamw-a",
    "sdxl-dinkster-adamw-b",
    "sdxl-dinkster-factored",
    "sdxl-kohya",
    "sdxl-ai-toolkit",
)
RUNS = SD15_RUNS + SDXL_RUNS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def _prepare_run(root: Path, overwrite: bool) -> None:
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise ValueError(f"run directory is not empty: {root}; pass --overwrite to replace it")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _monitor(
    run_root: Path,
    cwd: Path,
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> None:
    merged = os.environ.copy()
    if environment:
        merged.update(environment)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "monitor.py"),
            "--output",
            str(run_root / "monitor.json"),
            "--log",
            str(run_root / "run.log"),
            "--cwd",
            str(cwd),
            "--",
            *command,
        ],
        check=True,
        env=merged,
    )


def _dinkster_command(
    comparison_path: Path,
    model: Path,
    dataset: Path,
    run_root: Path,
    optimizer: str,
    base_dtype: str,
    *,
    allocator_attribution: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(HERE / "run_dinkster.py"),
        "--comparison",
        str(comparison_path),
        "--model",
        str(model),
        "--dataset",
        str(dataset),
        "--output",
        str(run_root),
        "--optimizer",
        optimizer,
        "--base-dtype",
        base_dtype,
        "--gradient-checkpointing",
        "--memory-attribution",
    ]
    if allocator_attribution:
        command.append("--allocator-attribution")
    return command


def _kohya_command(
    repo: Path,
    model: Path,
    dataset: Path,
    run_root: Path,
    comparison: dict[str, object],
    *,
    family: str,
    checkpointing: bool,
) -> list[str]:
    command = [
        str(repo / "venv/bin/python"),
        str(repo / ("sdxl_train_network.py" if family == "sdxl" else "train_network.py")),
        "--pretrained_model_name_or_path",
        str(model),
        "--train_data_dir",
        str(dataset),
        "--output_dir",
        str(run_root),
        "--output_name",
        "adapter",
        "--save_model_as",
        "safetensors",
        "--save_precision",
        "float",
        "--resolution",
        f"{comparison['resolution']},{comparison['resolution']}",
        "--train_batch_size",
        str(comparison["batch_size"]),
        "--max_train_steps",
        str(comparison["steps"]),
        "--network_module",
        "networks.lora",
        "--network_dim",
        str(comparison["rank"]),
        "--network_alpha",
        str(comparison["alpha"]),
        "--network_train_unet_only",
        "--learning_rate",
        str(comparison["learning_rate"]),
        "--unet_lr",
        str(comparison["learning_rate"]),
        "--optimizer_type",
        "AdamW",
        "--optimizer_args",
        f"weight_decay={comparison['weight_decay']}",
        "betas=0.9,0.999",
        "eps=1e-8",
        "--lr_scheduler",
        "constant",
        "--lr_warmup_steps",
        "0",
        "--mixed_precision",
        "bf16",
        "--gradient_accumulation_steps",
        str(comparison["gradient_accumulation_steps"]),
        "--seed",
        str(comparison["seed"]),
        "--max_grad_norm",
        "0",
        "--caption_extension",
        ".txt",
        "--logging_dir",
        str(run_root / "tensorboard"),
        "--log_with",
        "tensorboard",
        "--max_data_loader_n_workers",
        "0",
        "--no_half_vae",
        "--sdpa",
    ]
    if checkpointing:
        command.append("--gradient_checkpointing")
    return command


def _ai_config(
    model: Path,
    dataset: Path,
    run_root: Path,
    comparison: dict[str, object],
    *,
    family: str,
) -> dict[str, object]:
    name = "comparison_ai_toolkit_sdxl" if family == "sdxl" else "comparison_ai_toolkit"
    return {
        "job": "extension",
        "config": {
            "name": name,
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": str(run_root / "output"),
                    "device": "cuda:0",
                    "network": {
                        "type": "lora",
                        "linear": comparison["rank"],
                        "linear_alpha": comparison["alpha"],
                        "transformer_only": True,
                    },
                    "save": {
                        "dtype": "float32",
                        "save_every": comparison["steps"],
                        "max_step_saves_to_keep": 1,
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": str(dataset / "1_compare"),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.0,
                            "shuffle_tokens": False,
                            "cache_latents": False,
                            "cache_latents_to_disk": False,
                            "resolution": comparison["resolution"],
                            "buckets": False,
                            "random_crop": False,
                            "num_workers": 0,
                        }
                    ],
                    "train": {
                        "batch_size": comparison["batch_size"],
                        "steps": comparison["steps"],
                        "gradient_accumulation_steps": comparison["gradient_accumulation_steps"],
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "ddpm",
                        "optimizer": "adamw",
                        "optimizer_params": {
                            "weight_decay": comparison["weight_decay"],
                            "betas": comparison["betas"],
                        },
                        "lr": comparison["learning_rate"],
                        "lr_scheduler": "constant",
                        # AI Toolkit always clips; zero would erase every gradient.
                        "max_grad_norm": 1.0e9,
                        "dtype": "bf16",
                        "sdp": True,
                        "disable_sampling": True,
                    },
                    "model": {
                        "name_or_path": str(model),
                        "is_xl": family == "sdxl",
                        "is_v2": False,
                        "quantize": False,
                        "low_vram": False,
                    },
                    "logging": {"log_every": 1, "use_ui_logger": True},
                }
            ],
        },
        "meta": {"name": "[name]", "version": "1.0"},
    }


def _versions(python: Path) -> dict[str, str]:
    code = (
        "import importlib.metadata,json;"
        "print(json.dumps({d.metadata['Name']:d.version for d in "
        "importlib.metadata.distributions() if d.metadata['Name']},sort_keys=True))"
    )
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def _python_version(python: Path) -> str:
    return subprocess.check_output(
        [str(python), "-c", "import platform; print(platform.python_version())"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("sd15", "sdxl"), default="sd15")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--kohya-repo", type=Path, required=True)
    parser.add_argument("--ai-toolkit-repo", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", choices=RUNS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    family_runs = SDXL_RUNS if args.family == "sdxl" else SD15_RUNS
    runs = list(family_runs) if args.runs is None else args.runs
    unknown_runs = sorted(set(runs) - set(family_runs))
    if unknown_runs:
        parser.error(f"runs do not belong to {args.family}: {', '.join(unknown_runs)}")
    comparison_name = "comparison-sdxl.json" if args.family == "sdxl" else "comparison.json"
    comparison_path = HERE / comparison_name
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    model = args.model.resolve()
    if model.stat().st_size != comparison["model_bytes"]:
        raise ValueError(f"model size mismatch: {model.stat().st_size}")
    if _sha256(model) != comparison["model_sha256"]:
        raise ValueError("model SHA-256 does not match comparison.json")
    kohya_repo = args.kohya_repo.resolve()
    ai_repo = args.ai_toolkit_repo.resolve()
    if _git_head(kohya_repo) != comparison["kohya_commit"]:
        raise ValueError("kohya checkout does not match the pinned commit")
    if _git_head(ai_repo) != comparison["ai_toolkit_commit"]:
        raise ValueError("AI Toolkit checkout does not match the pinned commit")

    scratch = args.scratch.resolve()
    dataset = scratch / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "generate_dataset.py"),
            "--output",
            str(dataset),
            "--size",
            str(comparison["resolution"]),
        ],
        check=True,
    )
    instrumentation = str(HERE / "instrumentation")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    metadata = {
        "environment": {
            "gpu_name_driver_memory_mib": gpu,
            "hostname": platform.node(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "storage_free_bytes": shutil.disk_usage(scratch).free,
        },
        "generated_dataset": dataset_manifest,
        "model": {
            "bytes": model.stat().st_size,
            "family": args.family,
            "path": str(model),
            "sha256": comparison["model_sha256"],
            "url": MODEL_URLS[args.family],
        },
        "python_versions": {
            "ai_toolkit": _python_version(ai_repo / "venv/bin/python"),
            "dinkster": platform.python_version(),
            "kohya": _python_version(kohya_repo / "venv/bin/python"),
        },
        "references": {
            "ai_toolkit": {
                "commit": comparison["ai_toolkit_commit"],
                "url": "https://github.com/ostris/ai-toolkit.git",
            },
            "kohya": {
                "commit": comparison["kohya_commit"],
                "url": "https://github.com/kohya-ss/sd-scripts.git",
            },
        },
        "versions": {
            "ai_toolkit": _versions(ai_repo / "venv/bin/python"),
            "dinkster": {
                distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            },
            "kohya": _versions(kohya_repo / "venv/bin/python"),
        },
    }
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for name in runs:
        run_root = scratch / "runs" / name
        _prepare_run(run_root, args.overwrite)
        if "dinkster-" in name:
            optimizer = "factored-adamw" if name.endswith("-factored") else "adamw"
            base_dtype = "float32" if name.startswith("dinkster-fp32-") else "bfloat16"
            _monitor(
                run_root,
                Path.cwd(),
                _dinkster_command(
                    comparison_path,
                    model,
                    dataset,
                    run_root,
                    optimizer,
                    base_dtype,
                    allocator_attribution=name
                    in ("dinkster-adamw-a", "dinkster-fp32-adamw-a", "sdxl-dinkster-adamw-a"),
                ),
            )
        elif "kohya" in name:
            environment = {
                "PYTHONPATH": instrumentation,
                "TRAINING_COMPARISON_TORCH_MEMORY": str(run_root / "torch-memory.json"),
            }
            _monitor(
                run_root,
                kohya_repo,
                _kohya_command(
                    kohya_repo,
                    model,
                    dataset,
                    run_root,
                    comparison,
                    family=args.family,
                    checkpointing=name != "kohya-no-gc",
                ),
                environment=environment,
            )
        else:
            config_path = run_root / "config.json"
            config_path.write_text(
                json.dumps(
                    _ai_config(model, dataset, run_root, comparison, family=args.family), indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            environment = {
                "PYTHONPATH": instrumentation,
                "SEED": str(comparison["seed"]),
                "TRAINING_COMPARISON_TORCH_MEMORY": str(run_root / "torch-memory.json"),
            }
            _monitor(
                run_root,
                ai_repo,
                [str(ai_repo / "venv/bin/python"), str(ai_repo / "run.py"), str(config_path)],
                environment=environment,
            )


if __name__ == "__main__":
    main()
