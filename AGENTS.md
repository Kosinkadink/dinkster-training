# dinkster-training guidance

- Use `dinkster_inference` for model loading, encoding, forward passes, adapter
  attachment, and LoRA key maps. Do not copy inference numerical or model
  implementations into this repository.
- Never widen a numerical tolerance to make a test pass.
- Before every commit run locked sync, Ruff format and lint, both Pyright
  projects, the root integration tests, and the complete training-torch suite.
  On a CUDA host, run the training-torch suite in the CUDA environment too.
