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

Dinkster's default installation sources the node and worker packages from this
workspace, so serving with a library root composes training automatically.

Dependency updates are sequential. Training first advances its pinned Dinkster
revision when it needs a newer public API and publishes a tested commit. Dinkster
then advances its pinned training revision when that pack version is ready.
Neither repository depends on an uncommitted checkout of the other.
