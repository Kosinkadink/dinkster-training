# Contributing to dinkster-training

## Local checks

Install uv, Git, and Python 3.12 or newer, then run the repository checks from
the workspace root:

```sh
uv sync --locked --all-packages --all-extras
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pyright -p packages/dinkster-training-torch
uv run pytest -q tests
uv run pytest -q packages/dinkster-training-torch/tests
```

The complete torch suite runs its CPU coverage without model downloads. Tests
that prove accelerator behavior require an explicit opt-in, suitable hardware,
and the artifacts documented in
`packages/dinkster-training-torch/README.md`; they are not part of hosted CI.

## Pull requests

Use the public [issue tracker](https://github.com/Kosinkadink/dinkster-training/issues)
for reproducible bugs and focused feature proposals before opening a large
change.

- Keep each pull request focused and include tests for behavior changes.
- Keep Dinkster dependencies pinned to immutable 40-character commits.
- Do not put repository credentials, release tokens, model access tokens,
  private package references, or machine-specific paths in source, workflows,
  documentation, fixtures, or artifacts.
- Update the root status matrix and package documentation when support changes.
- Use ASCII in source, tests, documentation, commit messages, and pull request
  text.

GitHub may hold workflows from fork pull requests until a maintainer approves
the run. That approval is the repository's fork CI security gate.

## License

Outside contributions are accepted under GPL-3.0-or-later with a grant to
relicense under AGPL-3.0.
