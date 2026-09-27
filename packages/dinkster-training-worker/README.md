# dinkster-training-worker

This package executes Dinkster training nodes in a dedicated isolated process.
It retains the durable session handle, claim/fence/commit ledger, recovery
checkpoints, cancellation pauses, replay, and completion behavior.

Required environment variables:

- `DINKSTER_TRAINING_JOURNAL`: training SQLite journal path.
- `DINKSTER_TRAINING_BACKEND`: `fake` or `sd15-lora`.
- `DINKSTER_TRAINING_CHECKPOINT_ROOT`: checkpoint root for `sd15-lora`.

The SD1.5 backend also accepts:

- `DINKSTER_TRAINING_DEVICE`: governed process device, default `cuda:0`.
- `DINKSTER_TRAINING_RAM_BYTES`: model-operation RAM reservation, default 8 GiB.
- `DINKSTER_TRAINING_VRAM_BYTES`: CUDA reservation, default 8 GiB.

The model backend is an optional dependency. Install
`dinkster-training-worker[sd15-lora]` only in the training worker environment.
Selecting `fake` remains torch-free.

Model residency exists only during a reserved create or advance invocation.
The worker unloads `dinkster_comfy` models before releasing that invocation's
reservation; it never reuses an inference worker or retains an unaccounted hot
model between calls.
