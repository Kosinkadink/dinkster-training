# dinkster-training-torch

This package provides SD1.5 LoRA training through `dinkster_inference`. It does
not contain a second model implementation. The fork owns checkpoint loading,
VAE and CLIP encoding, the UNet forward pass, bypass attachment, and LoRA key
mapping.

Training runs only in the isolated `dinkster-training-worker` process. The
worker holds governor reservations for every invocation that loads a model,
then unloads all fork model residency before returning. It is not an inference
worker switched into a training mode.

## Configuration

Select the backend and process device:

```text
DINKSTER_TRAINING_BACKEND=sd15-lora
DINKSTER_TRAINING_JOURNAL=/path/to/training.sqlite
DINKSTER_TRAINING_CHECKPOINT_ROOT=/path/to/training-checkpoints
DINKSTER_TRAINING_DEVICE=cuda:0
```

`DINKSTER_TRAINING_RAM_BYTES` and `DINKSTER_TRAINING_VRAM_BYTES` may override
the default 8 GiB reservations. A training config's `device` must match the
worker device.

The node config is JSON:

```json
{
  "schemaVersion": 1,
  "family": "sd15",
  "checkpointPath": "/models/v1-5-pruned-emaonly.safetensors",
  "checkpointDigest": "sha256:<64 lowercase hex characters>",
  "device": "cuda:0",
  "seed": 1234,
  "rank": 4,
  "alpha": 4.0,
  "learningRate": 0.0001,
  "batchSize": 1,
  "gradientAccumulationSteps": 1,
  "checkpointInterval": 1,
  "dataset": {
    "root": "/data/images",
    "resolution": [512, 512],
    "encodedCacheRoot": "/data/encoded"
  }
}
```

The dataset is a recursively sorted set of image files with same-basename
UTF-8 `.txt` captions. VAE latents and CLIP embeddings are produced under
`torch.inference_mode()` and stored in a digest-verified safetensors cache.
The trainable forward is not inference-mode wrapped.

Checkpoints store adapter, optimizer, named RNG streams, data cursor, and loss
as content-addressed shards. The shared `TrainingSessionStore` ledger remains
the authority for claims, recovery checkpoints, commits, replay, fencing, and
completion.

`training.export_lora` accepts a path relative to the checkpoint root's
`exports` directory and `dtype` of `fp16`, `bf16`, or `fp32`. Tensor names come
from the fork's SD1.5 LoRA key map and are serialized in sorted order.

Training is not installed by Dinkster's default composition. Install and
configure this repository's schema pack and isolated worker explicitly.
