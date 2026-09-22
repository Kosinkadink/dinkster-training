# dinkster-training guidance

- Keep production imports from `dinkster_inference` and
  `dinkster_inference_torch` on their documented package-root APIs.
- Do not copy inference numerical or model implementations into this repository.
- Never widen a numerical tolerance to make a test pass.
- Before every commit run locked sync, Ruff format and lint, both Pyright
  projects, the root integration tests, and the complete training-torch suite.
  On a CUDA host, run the training-torch suite in the CUDA environment too.
