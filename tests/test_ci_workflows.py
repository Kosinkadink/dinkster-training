from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github/workflows/ci.yml"
FULL = ROOT / ".github/workflows/full-validation.yml"


def load_workflow(path: Path) -> dict[str, Any]:
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def jobs(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    loaded = workflow["jobs"]
    assert isinstance(loaded, dict)
    return loaded


def test_pull_requests_run_only_the_fast_integration_tier() -> None:
    workflow = load_workflow(CI)
    assert workflow["on"] == {"pull_request": ""}
    assert set(jobs(workflow)) == {"validate"}
    validate = jobs(workflow)["validate"]
    assert validate["timeout-minutes"] == "10"
    assert validate["runs-on"] == (
        "${{ fromJSON(vars.CI_RUNNERS)[github.event.pull_request.head.repo.full_name != "
        "github.repository && 'forkLinux' || 'linux'] }}"
    )
    commands = [step.get("run") for step in validate["steps"] if isinstance(step, dict)]
    assert any("pull request validation did not run" in (command or "") for command in commands)
    assert "uv run pytest -q tests" in commands
    assert not any(
        "packages/dinkster-training-torch/tests" in (command or "") for command in commands
    )


def test_main_runs_the_complete_cpu_validation_tier() -> None:
    workflow = load_workflow(FULL)
    assert workflow["on"]["push"] == {"branches": ["main"]}
    workflow_jobs = jobs(workflow)
    assert set(workflow_jobs) == {"integration", "main-status"}
    integration_commands = [
        step.get("run") for step in workflow_jobs["integration"]["steps"] if isinstance(step, dict)
    ]
    assert "uv run ruff format --check ." in integration_commands
    assert "uv run ruff check ." in integration_commands
    assert "uv run pyright" in integration_commands
    assert "uv run pyright -p packages/dinkster-training-torch" in integration_commands
    assert "uv run pytest -q tests" in integration_commands
    assert "gpu" not in FULL.read_text(encoding="utf-8").lower()
    assert "cuda" not in FULL.read_text(encoding="utf-8").lower()


def test_one_required_variable_controls_every_hosted_eligible_job() -> None:
    for path in (CI, FULL):
        workflow_jobs = jobs(load_workflow(path))
        for job in workflow_jobs.values():
            assert "vars.CI_RUNNERS" in job["runs-on"]
        text = path.read_text(encoding="utf-8")
        assert "DINKSTER_PR_RUNNER" not in text
        assert "ubuntu-latest" not in text


def test_main_status_reports_and_fails_for_every_lane() -> None:
    status = jobs(load_workflow(FULL))["main-status"]
    assert status["if"] == "always()"
    assert status["needs"] == ["integration"]
    text = FULL.read_text(encoding="utf-8")
    assert 'test "$INTEGRATION_RESULT" = success' in text
    assert "main-validation-status.json" in text
