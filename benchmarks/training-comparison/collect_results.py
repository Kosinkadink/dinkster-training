"""Collect reference losses and compute pre-declared comparison statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sqlite3
import statistics
from pathlib import Path

from adapter_stats import summarize
from safetensors import safe_open

SD15_RUN_NAMES = (
    "dinkster-adamw-a",
    "dinkster-adamw-b",
    "dinkster-factored",
    "dinkster-fp32-adamw-a",
    "dinkster-fp32-adamw-b",
    "kohya-gc",
    "kohya-no-gc",
    "ai-toolkit",
)
SDXL_RUN_NAMES = (
    "sdxl-dinkster-adamw-a",
    "sdxl-dinkster-adamw-b",
    "sdxl-dinkster-factored",
    "sdxl-kohya",
    "sdxl-ai-toolkit",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loss_summary(losses: list[float]) -> dict[str, object]:
    if not losses or not all(math.isfinite(value) for value in losses):
        raise ValueError("losses must be non-empty and finite")
    window = min(40, len(losses) // 2)
    first = statistics.fmean(losses[:window])
    last = statistics.fmean(losses[-window:])
    normalized_change = (last - first) / first
    return {
        "count": len(losses),
        "first_window_mean": first,
        "last_window_mean": last,
        "normalized_change": normalized_change,
        "non_worsening": last <= first * 1.10,
    }


def _ai_losses(path: Path) -> tuple[list[float], dict[str, object]]:
    with sqlite3.connect(path) as connection:
        keys = [row[0] for row in connection.execute("SELECT key FROM metric_keys ORDER BY key")]
        loss_key = next((key for key in keys if key in {"loss", "loss/loss", "mse_loss"}), None)
        if loss_key is None:
            loss_key = next((key for key in keys if "loss" in key.lower()), None)
        if loss_key is None:
            raise ValueError(f"no loss metric in {path}; keys={keys}")
        rows = list(
            connection.execute(
                "SELECT step, value_real FROM metrics WHERE key = ? ORDER BY step", (loss_key,)
            )
        )
    null_count = sum(value is None for _step, value in rows)
    if null_count:
        raise ValueError(f"AI Toolkit recorded {null_count} NULL values for loss key {loss_key}")
    steps = [int(step) for step, _value in rows]
    return [float(value) for _step, value in rows], {
        "distinct_step_count": len(set(steps)),
        "key": loss_key,
        "maximum_step": max(steps) if steps else None,
        "minimum_step": min(steps) if steps else None,
        "null_value_count": null_count,
        "row_count": len(rows),
    }


def _kohya_losses(log_root: Path) -> list[float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = sorted(log_root.rglob("events.out.tfevents.*"))
    if not events:
        raise ValueError(f"no TensorBoard events under {log_root}")
    accumulator = EventAccumulator(str(events[-1]))
    accumulator.Reload()
    tags = accumulator.Tags()["scalars"]
    key = (
        "loss/current"
        if "loss/current" in tags
        else next((tag for tag in tags if "loss" in tag.lower()), None)
    )
    if key is None:
        raise ValueError(f"no loss scalar in {events[-1]}; tags={tags}")
    return [float(event.value) for event in accumulator.Scalars(key)]


def _factor(left: float, right: float) -> float | None:
    if left == 0.0 or right == 0.0:
        return None
    return max(left / right, right / left)


def _delta_comparisons(
    results: dict[str, object],
    dinkster_name: str,
    reference_names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    dinkster = results[dinkster_name]
    assert isinstance(dinkster, dict)
    dinkster_deltas = dinkster["adapter_delta_statistics"]
    assert isinstance(dinkster_deltas, dict)
    dinkster_zero_count = sum(
        int(metric["zero_count"])
        for family in dinkster_deltas.values()
        if isinstance(family, dict)
        for metric in [family["applied_delta_rms"]]
        if isinstance(metric, dict)
    )
    comparisons: dict[str, dict[str, object]] = {}
    for reference_name in reference_names:
        reference = results[reference_name]
        assert isinstance(reference, dict)
        reference_deltas = reference["adapter_delta_statistics"]
        assert isinstance(reference_deltas, dict)
        metric_factors: dict[str, object] = {}
        for metric in (
            "applied_delta_rms",
            "normalized_delta_rms",
            "down_rms",
            "up_rms",
        ):
            factors = {}
            for family in sorted(dinkster_deltas.keys() & reference_deltas.keys()):
                dinkster_family = dinkster_deltas[family]
                reference_family = reference_deltas[family]
                assert isinstance(dinkster_family, dict) and isinstance(reference_family, dict)
                dinkster_metric = dinkster_family[metric]
                reference_metric = reference_family[metric]
                assert isinstance(dinkster_metric, dict) and isinstance(reference_metric, dict)
                factors[family] = {
                    "median": _factor(
                        float(dinkster_metric["median"]), float(reference_metric["median"])
                    ),
                    "p95": _factor(float(dinkster_metric["p95"]), float(reference_metric["p95"])),
                }
            metric_factors[metric] = factors
        zero_count = sum(
            int(metric["zero_count"])
            for family in reference_deltas.values()
            if isinstance(family, dict)
            for metric in [family["applied_delta_rms"]]
            if isinstance(metric, dict)
        )
        applied_factors = metric_factors["applied_delta_rms"]
        assert isinstance(applied_factors, dict)
        comparisons[reference_name] = {
            "factors": metric_factors,
            "dinkster_zero_delta_count": dinkster_zero_count,
            "reference_zero_delta_count": zero_count,
            "within_limits": bool(applied_factors)
            and dinkster_zero_count == 0
            and zero_count == 0
            and all(
                isinstance(values, dict)
                and isinstance(values["median"], float)
                and isinstance(values["p95"], float)
                and values["median"] <= 5.0
                and values["p95"] <= 10.0
                for values in applied_factors.values()
            ),
        }
    return comparisons


def _configuration_expectations(
    results: dict[str, object],
    run_names: tuple[str, ...],
    first_name: str,
    second_name: str,
    kohya_name: str,
    reference_names: tuple[str, ...],
) -> dict[str, object]:
    first = results[first_name]
    second = results[second_name]
    kohya = results[kohya_name]
    assert isinstance(first, dict) and isinstance(second, dict) and isinstance(kohya, dict)
    first_loss = first["loss_summary"]
    kohya_loss = kohya["loss_summary"]
    assert isinstance(first_loss, dict) and isinstance(kohya_loss, dict)
    loss_change_difference = abs(
        float(first_loss["normalized_change"]) - float(kohya_loss["normalized_change"])
    )
    delta_comparisons = _delta_comparisons(results, first_name, reference_names)
    loss_summaries = []
    for name in run_names:
        run = results[name]
        assert isinstance(run, dict)
        loss_summary = run["loss_summary"]
        assert isinstance(loss_summary, dict)
        loss_summaries.append(loss_summary)
    return {
        "all_runs_have_200_finite_losses": all(
            summary["count"] == 200 for summary in loss_summaries
        ),
        "all_runs_non_worsening": all(summary["non_worsening"] for summary in loss_summaries),
        "delta_comparisons": delta_comparisons,
        "delta_scale_within_limits": all(
            comparison["within_limits"] for comparison in delta_comparisons.values()
        ),
        "determinism": {
            "losses_exact": first["losses"] == second["losses"],
            "adapter_sha256_exact": first["adapter_sha256"] == second["adapter_sha256"],
        },
        "dinkster_kohya_loss_change_difference": loss_change_difference,
        "dinkster_kohya_loss_change_within_0_25": loss_change_difference <= 0.25,
    }


def _adapter_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as adapter:
        return dict(adapter.metadata() or {})


def _sdxl_export_audit(
    name: str,
    families: dict[str, object],
    metadata: dict[str, str],
    *,
    rank: float,
    alpha: float,
    steps: int,
) -> dict[str, object]:
    family_audits: dict[str, object] = {}
    for family, raw_statistics in sorted(families.items()):
        assert isinstance(raw_statistics, dict)
        statistics_by_name = raw_statistics
        rank_statistics = statistics_by_name["rank"]
        alpha_statistics = statistics_by_name["alpha"]
        scale_statistics = statistics_by_name["effective_scale"]
        present_statistics = statistics_by_name["alpha_tensor_present"]
        assert isinstance(rank_statistics, dict)
        assert isinstance(alpha_statistics, dict)
        assert isinstance(scale_statistics, dict)
        assert isinstance(present_statistics, dict)
        matches = (
            rank_statistics["min"] == rank_statistics["max"] == rank
            and alpha_statistics["min"] == alpha_statistics["max"] == alpha
            and scale_statistics["min"] == scale_statistics["max"] == alpha / rank
            and present_statistics["min"] == present_statistics["max"] == 1.0
        )
        family_audits[family] = {
            "alpha": alpha_statistics["min"],
            "alpha_tensor_present": present_statistics["min"] == 1.0,
            "effective_scale": scale_statistics["min"],
            "matches": matches,
            "module_count": rank_statistics["count"],
            "rank": rank_statistics["min"],
        }

    if name.startswith("sdxl-dinkster-"):
        metadata_evidence = {
            "dinkster_runtime_identity": metadata.get("dinkster_runtime_identity"),
            "dinkster_step_cursor": metadata.get("dinkster_step_cursor"),
        }
        metadata_matches = metadata_evidence == {
            "dinkster_runtime_identity": "dinkster-sdxl-training-comparison",
            "dinkster_step_cursor": str(steps),
        }
    elif name == "sdxl-kohya":
        metadata_evidence = {
            "ss_base_model_version": metadata.get("ss_base_model_version"),
            "ss_network_alpha": metadata.get("ss_network_alpha"),
            "ss_network_dim": metadata.get("ss_network_dim"),
            "ss_sd_scripts_commit_hash": metadata.get("ss_sd_scripts_commit_hash"),
            "ss_steps": metadata.get("ss_steps"),
        }
        metadata_matches = metadata_evidence == {
            "ss_base_model_version": "sdxl_base_v1-0",
            "ss_network_alpha": str(alpha),
            "ss_network_dim": str(int(rank)),
            "ss_sd_scripts_commit_hash": "37a1cbbc5725ed2a3575506e7bd2001c9908ac92",
            "ss_steps": str(steps),
        }
    else:
        training_info = json.loads(metadata.get("training_info", "{}"))
        metadata_evidence = {
            "ss_base_model_version": metadata.get("ss_base_model_version"),
            "training_step": training_info.get("step"),
        }
        metadata_matches = metadata_evidence == {
            "ss_base_model_version": "sdxl_1.0",
            "training_step": steps,
        }

    family_matches = all(
        isinstance(audit, dict) and audit["matches"] for audit in family_audits.values()
    )
    return {
        "family_audits": family_audits,
        "matches": bool(family_audits) and family_matches and metadata_matches,
        "metadata": metadata_evidence,
        "metadata_matches": metadata_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("sd15", "sdxl"), default="sd15")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path)
    args = parser.parse_args()
    scratch = args.scratch.resolve()
    results: dict[str, object] = {}
    if args.family == "sdxl":
        run_names = SDXL_RUN_NAMES
        dinkster_names = SDXL_RUN_NAMES[:3]
        kohya_names = ("sdxl-kohya",)
        ai_name = "sdxl-ai-toolkit"
        ai_output_name = "comparison_ai_toolkit_sdxl"
    else:
        run_names = SD15_RUN_NAMES
        dinkster_names = SD15_RUN_NAMES[:5]
        kohya_names = ("kohya-gc", "kohya-no-gc")
        ai_name = "ai-toolkit"
        ai_output_name = "comparison_ai_toolkit"

    for name in dinkster_names:
        run = json.loads((scratch / "runs" / name / "result.json").read_text(encoding="utf-8"))
        run.setdefault("base_dtype", "float32" if name.startswith("dinkster-fp32-") else "bfloat16")
        run.setdefault("deterministic_algorithms", True)
        run["loss_summary"] = _loss_summary(run["losses"])
        results[name] = run
    for name in kohya_names:
        losses = _kohya_losses(scratch / "runs" / name / "tensorboard")
        results[name] = {"losses": losses, "loss_summary": _loss_summary(losses)}
    ai_root = scratch / "runs" / ai_name / "output" / ai_output_name
    ai_losses, ai_loss_log_audit = _ai_losses(ai_root / "loss_log.db")
    results[ai_name] = {
        "loss_log_audit": ai_loss_log_audit,
        "losses": ai_losses,
        "loss_summary": _loss_summary(ai_losses),
    }
    adapter_paths: dict[str, Path] = {
        name: scratch / "runs" / name / "adapter.safetensors"
        for name in (*dinkster_names, *kohya_names)
    }
    adapter_paths[ai_name] = ai_root / f"{ai_output_name}.safetensors"
    comparison = json.loads(
        (
            Path(__file__).resolve().parent
            / ("comparison-sdxl.json" if args.family == "sdxl" else "comparison.json")
        ).read_text(encoding="utf-8")
    )
    for name in run_names:
        value = results[name]
        assert isinstance(value, dict)
        monitor_path = scratch / "runs" / name / "monitor.json"
        torch_path = scratch / "runs" / name / "torch-memory.json"
        if monitor_path.is_file():
            monitor = json.loads(monitor_path.read_text(encoding="utf-8"))
            value["monitor"] = {
                key: monitor[key] for key in ("nvidia_smi_peak_mib", "runtime_seconds", "samples")
            }
        if torch_path.is_file():
            value["torch_memory"] = json.loads(torch_path.read_text(encoding="utf-8"))
        adapter = adapter_paths[name]
        value.pop("adapter", None)
        value["adapter_sha256"] = _sha256(adapter)
        families = summarize(adapter)["families"]
        value["adapter_delta_statistics"] = families
        if args.family == "sdxl":
            assert isinstance(families, dict)
            value["export_audit"] = _sdxl_export_audit(
                name,
                families,
                _adapter_metadata(adapter),
                rank=float(comparison["rank"]),
                alpha=float(comparison["alpha"]),
                steps=int(comparison["steps"]),
            )

    if args.family == "sdxl":
        sdxl_expectations = _configuration_expectations(
            results,
            SDXL_RUN_NAMES,
            "sdxl-dinkster-adamw-a",
            "sdxl-dinkster-adamw-b",
            "sdxl-kohya",
            ("sdxl-kohya", "sdxl-ai-toolkit"),
        )
        export_audits: dict[str, object] = {}
        for name in SDXL_RUN_NAMES:
            result = results[name]
            assert isinstance(result, dict)
            export_audits[name] = result["export_audit"]
        sdxl_expectations["export_audits"] = export_audits
        sdxl_expectations["rank_alpha_and_metadata_match"] = all(
            isinstance(audit, dict) and audit["matches"] for audit in export_audits.values()
        )
        current_all_losses = bool(sdxl_expectations["all_runs_have_200_finite_losses"])
        current_non_worsening = bool(sdxl_expectations["all_runs_non_worsening"])
        if not args.output.is_file():
            parser.error("collect SD1.5 before appending SDXL")
        output = json.loads(args.output.read_text(encoding="utf-8"))
        expectations = output["expectations"]
        assert isinstance(expectations, dict)
        configurations = expectations["configurations"]
        assert isinstance(configurations, dict)
        configurations["sdxl"] = sdxl_expectations
        sd15_summaries = []
        for name in SD15_RUN_NAMES:
            existing_result = json.loads(
                (args.output_dir / f"{name}.json").read_text(encoding="utf-8")
            )
            existing_summary = existing_result["loss_summary"]
            assert isinstance(existing_summary, dict)
            sd15_summaries.append(existing_summary)
        expectations["all_runs_have_200_finite_losses"] = (
            all(summary["count"] == 200 for summary in sd15_summaries) and current_all_losses
        )
        expectations["all_runs_non_worsening"] = (
            all(summary["non_worsening"] for summary in sd15_summaries) and current_non_worsening
        )
        output_runs = output["runs"]
        assert isinstance(output_runs, dict)
        output_runs.update({name: f"results/{name}.json" for name in run_names})
    else:
        configurations = {
            "bfloat16": _configuration_expectations(
                results,
                ("dinkster-adamw-a", "dinkster-adamw-b", "dinkster-factored"),
                "dinkster-adamw-a",
                "dinkster-adamw-b",
                "kohya-gc",
                ("kohya-gc", "ai-toolkit"),
            ),
            "float32": _configuration_expectations(
                results,
                ("dinkster-fp32-adamw-a", "dinkster-fp32-adamw-b"),
                "dinkster-fp32-adamw-a",
                "dinkster-fp32-adamw-b",
                "kohya-gc",
                ("kohya-gc", "ai-toolkit"),
            ),
        }
        output = {
            "expectations": {
                "all_runs_have_200_finite_losses": all(
                    isinstance(value, dict)
                    and isinstance(value["loss_summary"], dict)
                    and value["loss_summary"]["count"] == 200
                    for value in results.values()
                ),
                "all_runs_non_worsening": all(
                    isinstance(value, dict)
                    and isinstance(value["loss_summary"], dict)
                    and value["loss_summary"]["non_worsening"]
                    for value in results.values()
                ),
                "configurations": configurations,
            },
            "runs": {name: f"results/{name}.json" for name in run_names},
        }
        if args.output.is_file():
            existing_output = json.loads(args.output.read_text(encoding="utf-8"))
            existing_expectations = existing_output.get("expectations")
            existing_runs = existing_output.get("runs")
            if isinstance(existing_expectations, dict) and isinstance(existing_runs, dict):
                existing_configurations = existing_expectations.get("configurations")
                if isinstance(existing_configurations, dict):
                    sdxl_expectations = existing_configurations.get("sdxl")
                    if isinstance(sdxl_expectations, dict):
                        configurations["sdxl"] = sdxl_expectations
                        output["expectations"]["all_runs_have_200_finite_losses"] = bool(
                            output["expectations"]["all_runs_have_200_finite_losses"]
                        ) and bool(sdxl_expectations["all_runs_have_200_finite_losses"])
                        output["expectations"]["all_runs_non_worsening"] = bool(
                            output["expectations"]["all_runs_non_worsening"]
                        ) and bool(sdxl_expectations["all_runs_non_worsening"])
                        output["runs"].update(
                            {
                                name: existing_runs[name]
                                for name in SDXL_RUN_NAMES
                                if name in existing_runs
                            }
                        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in run_names:
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(results[name], indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    run_environment = json.loads((scratch / "run-metadata.json").read_text(encoding="utf-8"))
    auxiliary_artifacts = comparison.get("auxiliary_artifacts")
    if isinstance(auxiliary_artifacts, dict):
        run_environment["auxiliary_artifacts"] = auxiliary_artifacts
    model = run_environment.get("model")
    if isinstance(model, dict):
        model.pop("path", None)
    environment_details = run_environment.get("environment")
    if not isinstance(environment_details, dict):
        raise ValueError("run metadata has no environment object")
    environment_details["storage_free_bytes_at_collection"] = shutil.disk_usage(scratch).free
    environment_output = args.environment_output or args.output.parent / "environment.json"
    if args.family == "sdxl":
        if not environment_output.is_file():
            parser.error("collect SD1.5 before appending SDXL")
        environment = json.loads(environment_output.read_text(encoding="utf-8"))
        environment["sdxl"] = run_environment
    else:
        environment = run_environment
        if environment_output.is_file():
            existing_environment = json.loads(environment_output.read_text(encoding="utf-8"))
            if isinstance(existing_environment.get("sdxl"), dict):
                environment["sdxl"] = existing_environment["sdxl"]
    environment_output.parent.mkdir(parents=True, exist_ok=True)
    environment_output.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
