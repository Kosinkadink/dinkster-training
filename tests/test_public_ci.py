"""Public CI must remain independent of private infrastructure and credentials."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def test_ci_uses_only_hosted_public_dependencies() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

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
    workflow = WORKFLOW.read_text(encoding="utf-8")

    action_lines = [line.strip() for line in workflow.splitlines() if "uses:" in line]
    assert action_lines
    for line in action_lines:
        reference = line.split("@", 1)[1].split()[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)
