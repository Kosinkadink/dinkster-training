# dinkster-training

First-party Dinkster training node schemas, isolated worker, and torch LoRA
runtime. The repository is a uv workspace containing:

- `dinkster-nodes-training`: graph-facing training nodes and service contract.
- `dinkster-training-worker`: isolated fake and torch backend selection.
- `dinkster-training-torch`: native model training, datasets, checkpoints, and
  LoRA export.
- `benchmarks/training-comparison`: retained SD1.5 and SDXL trainer comparisons,
  reports, and collection tools.

The lockfile pins every Dinkster dependency to one immutable Dinkster commit.
This repository does not require a sibling checkout. From this repository:

```sh
uv sync --locked --all-packages --all-extras
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pyright -p packages/dinkster-training-torch
uv run pytest -q tests packages/dinkster-training-torch/tests
```

Pull requests run format, lint, both type checks, and the torch-free integration
tests with a 10-minute budget. Main repeats those CPU checks with a 10-minute
budget. The
`CI_RUNNERS` repository variable is required and contains runner-label arrays:

```json
{"linux":["self-hosted","linux","x64"],"windows":["windows-latest"],"macos":["macos-latest"],"forkLinux":["ubuntu-latest"]}
```

After the repository is public, changing that one value to the following moves
every job without changing workflow source:

```json
{"linux":["ubuntu-latest"],"windows":["windows-latest"],"macos":["macos-latest"],"forkLinux":["ubuntu-latest"]}
```

The complete CUDA training runtime is an owner gate, not an Actions job. Model
family changes also require a verifier-checked ComfyUI-master output and peak
memory receipt. Each main run uploads `main-validation-status`, which records
the CPU integration lane and fails when it did not pass.
Before the private dependency becomes public, approved fork PRs use
`forkLinux` but fail with an explicit not-run reason because GitHub withholds
repository secrets. Unexecuted validation is never reported as green.

Dinkster's default installation sources the node and worker packages from this
workspace, so serving with a library root composes training automatically.

Dependency updates are sequential. Training first advances its pinned Dinkster
revision when it needs a newer public API and publishes a tested commit. Dinkster
then advances its pinned training revision when that pack version is ready.
Neither repository depends on an uncommitted checkout of the other.
