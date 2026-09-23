"""Public CI must remain independent of private infrastructure and credentials."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"
FULL_VALIDATION = ROOT / ".github/workflows/full-validation.yml"


def test_ci_uses_only_hosted_public_dependencies() -> None:
    for path in (WORKFLOW, FULL_VALIDATION):
        workflow = path.read_text(encoding="utf-8")
        assert "runs-on: ubuntu-latest" in workflow
        assert "permissions:\n  contents: read" in workflow
        for prohibited in (
            "self-hosted",
            "secrets.",
            "vars.",
            "deploy-key",
            "configure-dinkster-access",
            "CUDA_VISIBLE_DEVICES",
            "pull_request_target",
            "workflow_dispatch",
            "write-all",
        ):
            assert prohibited not in workflow


def test_ci_actions_are_pinned_to_commits() -> None:
    for path in (WORKFLOW, FULL_VALIDATION):
        workflow = path.read_text(encoding="utf-8")
        action_lines = [line.strip() for line in workflow.splitlines() if "uses:" in line]
        assert action_lines
        for line in action_lines:
            reference = line.split("@", 1)[1].split()[0]
            assert len(reference) == 40
            assert all(character in "0123456789abcdef" for character in reference)


def test_pull_requests_run_only_the_fast_tier() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "pull_request:" in workflow
    assert "push:" not in workflow
    assert "timeout-minutes: 10" in workflow
    assert "uv run pytest -q tests" in workflow
    assert "packages/dinkster-training-torch/tests" not in workflow


def test_main_runs_every_training_test_file_once() -> None:
    workflow = FULL_VALIDATION.read_text(encoding="utf-8")
    expected = sorted(
        path.name for path in (ROOT / "packages/dinkster-training-torch/tests").glob("test_*.py")
    )
    selected = sorted(
        name
        for name in expected
        if workflow.count(name) == (6 if name == "test_sd15_lora.py" else 1)
    )
    assert selected == expected
    assert "timeout-minutes: 16" in workflow
    assert "awk 'NR % 3 == 1'" in workflow
    assert "awk 'NR % 3 == 2'" in workflow
    assert "awk 'NR % 3 == 0'" in workflow


def test_main_status_propagates_every_lane_result() -> None:
    workflow = FULL_VALIDATION.read_text(encoding="utf-8")
    for lane in (
        "quality",
        "training-core",
        "training-sd15-1",
        "training-sd15-2",
        "training-sd15-3",
        "training-sdxl",
        "training-image",
        "training-media",
    ):
        assert f'"{lane}":"%s"' in workflow
        assert f"needs.{lane}.result" in workflow
