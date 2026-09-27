# dinkster-training

[![CI](https://github.com/Kosinkadink/dinkster-training/actions/workflows/ci.yml/badge.svg)](https://github.com/Kosinkadink/dinkster-training/actions/workflows/ci.yml)

**Status: in progress.** This repository is under active development. Its
packages are versioned `0.0.1`; interfaces and training behavior may change
without a compatibility release. The workspace packages are not published as
stable package-index releases.

This workspace provides Dinkster's first-party training node schemas, durable
training worker, and LoRA training integration with `dinkster_comfy`. Dinkster
supplies the graph, protocol, server, and worker foundations; this repository
pins immutable public Dinkster and `dinkster_comfy` revisions in `uv.lock` and
adds the training-specific packages on top. It contains:

- `dinkster-nodes-training`: graph-facing training nodes and service contract.
- `dinkster-training-worker`: isolated fake and torch backend selection.
- `dinkster-training-torch`: SD1.5 model training through `dinkster_comfy`,
  datasets, checkpoints, and LoRA export.
- `benchmarks/training-comparison`: retained SD1.5 and SDXL trainer comparisons,
  reports, and collection tools.

## Install and run the CPU example

Install [uv](https://docs.astral.sh/uv/), Git, and Python 3.12 or newer. From a
public clone, no repository credentials, model files, or accelerator are needed
for the deterministic fake-backend example:

```sh
git clone https://github.com/Kosinkadink/dinkster-training.git
cd dinkster-training
uv sync --locked --all-packages
uv run pytest -q \
  tests/test_training_worker.py::test_fold_and_while_loops_commit_across_process_boundary
```

The test runs a torch-free trainer in an isolated worker process and verifies
durable fold and while-loop commits in a temporary SQLite journal. It does not
download model weights or perform model training. After the environment is
synced, it should finish in about 15 seconds on a recent desktop CPU; timing
varies by machine.

## Packages and support status

| Path or backend | Status | Requirements and limitations |
| --- | --- | --- |
| `packages/dinkster-nodes-training` | Implemented | Training node schemas and the service boundary; no torch dependency. |
| `packages/dinkster-training-worker` with `fake` | Supported for CPU development and CI | Deterministic protocol exerciser only; it does not train a model. |
| `packages/dinkster-training-worker` with `sd15-lora` | In progress | Requires an explicit checkpoint root, user-supplied model and dataset paths, and the torch runtime package. |
| SD1.5 LoRA | In progress | Loads and encodes through `dinkster_comfy`; real training requires user-supplied model assets and suitable hardware. |

`packages/dinkster-training-torch/README.md` documents each model backend's
configuration, data contract, artifact requirements, and validation commands.
The repository does not distribute model weights, datasets, checkpoints, or
generated LoRA files.

The lockfile pins every Dinkster dependency to one immutable Dinkster commit.
This repository requires neither a sibling checkout nor private package access.
Dinkster's default installation does not install or compose these training
packs. An installation must add the node and dedicated worker packs explicitly.

Dependency updates are sequential. Training first advances its pinned Dinkster
revision when it needs a newer public API and publishes a tested commit. Dinkster
then advances its pinned training revision when that pack version is ready.
Neither repository depends on an uncommitted checkout of the other.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the complete local checks and pull
request requirements. Public CI runs only on GitHub-hosted Linux workers and
requires no repository secrets. Accelerator validation is a separate developer
activity and is not part of hosted CI.

## License

This project is licensed under GPL-3.0-or-later. See [LICENSE](LICENSE).
