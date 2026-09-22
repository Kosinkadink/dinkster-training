# dinkster-training-worker

This package executes the first-party training node schemas in an isolated
worker process. The worker requires two explicit environment variables:

- `DINKSTER_TRAINING_JOURNAL`: path to the server's training SQLite journal.
- `DINKSTER_TRAINING_BACKEND`: `fake` for the deterministic torch-free protocol
  exerciser, `sd15-lora` for native SD1.5 LoRA training, `sdxl-lora` for
  native SDXL base LoRA training, `flux-lora` for classic Flux DiT LoRA
  training, `flux2-lora` for Flux2 DiT LoRA training, `minimax-h3-lora` for
  MiniMax H3 DiT LoRA training, `minimax-music3-lora` for community-derived
  MiniMax Music 3 DiT LoRA training, `ideogram4-lora` for conditional or
  unconditional Ideogram 4 DiT LoRA training, `qwen-image-lora` for Qwen-Image
  DiT LoRA training, or `wan-lora` for Wan DiT LoRA training.
- `DINKSTER_TRAINING_CHECKPOINT_ROOT`: content-addressed checkpoint root required
  by each LoRA backend.

The fake backend also accepts `DINKSTER_TRAINING_FAKE_FIRST_STEP_DELAY` as a
non-negative delay in seconds after its first recovery checkpoint. This is
used to exercise worker termination at a deterministic safe point.

The LoRA backends are optional dependencies so the root workspace environment
does not install torch. Install `dinkster-training-worker[sd15-lora]` or
`dinkster-training-worker[sdxl-lora]` for the SD backends, or
`dinkster-training-worker[flux-lora]` for classic Flux or Flux2, or
`dinkster-training-worker[minimax-h3-lora]` for H3, or
`dinkster-training-worker[minimax-music3-lora]` for Music 3, or
`dinkster-training-worker[ideogram4-lora]` for Ideogram 4, or
`dinkster-training-worker[qwen-image-lora]` for Qwen-Image, or
`dinkster-training-worker[wan-lora]` for Wan, in the dedicated torch worker
environment. The backend import remains lazy: selecting `fake` does not import
torch or `dinkster-training-torch`.

The LoRA backends poll the isolated invocation's cooperative cancellation token.
They finish the current optimizer-safe unit, publish recovery state through the
shared ledger, and pause rather than treating task cancellation as a commit.
