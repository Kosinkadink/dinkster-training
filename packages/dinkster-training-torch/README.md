# dinkster-training-torch

This package owns the torch runtime for SD1.5, SDXL base, Wan 2.1 T2V, and
MiniMax H3 and Music 3 DiT LoRA training. It keeps the base model frozen,
installs float32 LoRA masters on native projections, and applies them as
low-rank forward branches in each model operation's output dtype. SD training
uses bfloat16 autocast and the linear-beta epsilon-prediction MSE objective. H3
training uses eager model compute, its fixed sigma table, and a multistream
velocity flow-matching objective. Music 3 uses the explicitly community-derived
flow recipe documented below. Optimizer checkpoints remain float32.

The trainer supports AdamW and `factored-adamw`. The factored option is a
self-contained AdamW variant that keeps a full first moment and row/column
second moments for each LoRA matrix. It avoids a bitsandbytes runtime and cuts
second-moment storage from one value per trainable parameter to rows plus
columns.

## Data boundary

`PreparedBatchSource.batch(cursor, generator, device)` is the data pipeline
boundary. It returns prepared latents and text embeddings for an absolute data
cursor, with optional paired timestep/noise tensors. If those tensors are
absent, the trainer samples both from its checkpointed timestep/noise RNG. A
folder dataset implements the same protocol from image files and same-basename
UTF-8 `.txt` captions. It recursively sorts relative image paths, deterministically
shuffles each epoch from the dataset digest, center-crops and resizes RGB images,
and precomputes them under `torch.no_grad`. The VAE encodes every image and is
released before CLIP-L encodes every caption. Training retains one float32
latent and text embedding per item in CPU memory and transfers each selected
batch to the training device. SDXL precomputation uses VAE, then CLIP-L, then
CLIP-G, releasing each before loading the next. It retains the concatenated
CLIP-L/CLIP-G penultimate states, projected CLIP-G pooled output, and fixed
size conditioning.

SDXL conditioning follows Dinkster's native inference recipe and kohya sd-scripts
commit `37a1cbbc5725ed2a3575506e7bd2001c9908ac92`: CLIP-L precedes CLIP-G in the
2048-wide context, the projected CLIP-G EOS output precedes six 256-wide size
embeddings, and the size order is height, width, crop top, crop left, target
height, target width. The fixed-size dataset contract treats the normalized
training canvas as both original and target size and uses a zero top-left crop.
This keeps every precomputed item at the configured resolution without
aspect-ratio buckets.

Folder datasets can set `encodedCacheRoot` to an absolute or resolvable path.
On a miss, the normal sequential precomputation writes one deterministic
safetensors shard and an atomic manifest under that root. A verified hit loads
the CPU tensor store without loading a VAE, text encoder, or H3 conditioner.
Cache entries are keyed by the dataset digest and execution identity, including
batch size for SD because VAE kernels can change with batch size or device.
Shards and manifests are digest-verified; stale or corrupt entries are
recomputed and replaced. Dataset or component changes naturally select another
entry. Old entries are not removed automatically, so cache eviction is manual.

The built-in filesystem source reads `000000000000.pt`, `000000000001.pt`, and
so on. Each file is loaded with `weights_only=True` and contains `latents` and
`text_embeddings`; SDXL files also contain paired `pooled_embeddings` and
`time_ids`. `timesteps` and `noise` are optional but must be supplied together.

H3 uses a separate cursor-keyed prepared-batch protocol. Each file contains
`video_latents`, `audio_latents`, `conditioner_embeddings`, and an optional
plain mapping for the complete DiT conditioning carrier. `sigma_indices`,
`video_noise`, and `audio_noise` may be supplied only as a complete set. The
trainer scales the clean audio stream into H3's shared video-sigma coordinate,
samples from the fixed H3 sigma table when no index is supplied, and sends the
ordered video/audio streams, sigma, context, carrier, and H3 sigma descriptor
through the same DiT call contract used by inference.

H3 FL2VA training can instead use video/audio/caption media items. The existing
item directory form contains a direct `frames/` folder, exactly one uncompressed
stereo 32 kHz PCM `.wav`, and a same-basename UTF-8 `.txt` caption. Frame image
names use equal-width consecutive decimal stems; their lexical order defines
time. An item can alternatively be a container video with a same-basename
caption. Common AVI, M4V, Matroska, MOV, MP4, MPEG, and WebM suffixes are
discovered and accepted when the pinned `av==16.0.1` decoder can read exactly
one video stream and the selected audio source satisfies the requirements below.
A same-basename WAV takes audio precedence. Without one, the container must have
exactly one audio stream with a sample-accurate timeline that starts at zero and
is contiguous. Its declared duration or decoded length must match the target;
PyAV deterministically converts it to signed 16-bit stereo 32 kHz PCM. Containers
whose audio timeline cannot meet this strict profile are refused. A frame-folder
item and container with the same relative basename are ambiguous and refused.

Every item has the configured resolution and a frame count of `17k+5`; its PCM
audio has `round(frameCount/24*40)*800` samples so the encoded stream exactly
matches the H3 target grid. Precomputation uses the native deterministic video
and audio VAE posterior means, then Qwen3-VL conditioning, releasing each
component before loading the next. The resident CPU store and optional cache
contain float32 video latents, audio latents, conditioner embeddings, and int64
text carrier tags. Cache hits rehash source bytes but do not load PyAV or any
model component. H3 REF2VA training continues to require prepared batches.

Wan 2.1 T2V training reads container videos with same-basename UTF-8 captions.
Every video has the configured `4k+1` frame count and fixed resolution. Offline
precomputation runs the digest-pinned Wan VAE and UMT5-XXL components with
float32 compute independent of checkpoint storage dtype, stores normalized
16-channel latents and 4096-wide context in float32, and releases each component
before training starts. The optional encoded cache uses the same
locked, atomic, digest-verified shard and manifest contract as the other media
datasets. Its identity includes source bytes, preprocessing, component pins,
tensor contracts, and execution hardware.

The media dataset config pins the dataset root, `[height, width]` resolution,
and digest-addressed encoder state. SD1.5 uses separate VAE and CLIP-L sources;
SDXL uses its standard single-file checkpoint. The content digest covers the
sorted image/caption names, every file digest, and the preprocessing and encoder
identities. Missing, empty, or non-UTF-8 captions are reported by dry-run and
prevent session creation.

## Checkpoints

A checkpoint writes content-addressed adapter, optimizer, RNG, config, and
trainer-state shards. The digest-addressed manifest is recorded in the shared
`TrainingSessionStore` operation ledger. The ledger commit, not the presence
of files, selects a checkpoint as the committed session head.

By default a checkpoint is written only at advance boundaries: session
creation, the final step of each `training.advance` call, and a
cancellation pause (which publishes a checkpoint on demand when steps ran
since the last one, so a pause never loses work). Set `checkpointInterval`
to an integer N >= 1 to additionally checkpoint every N optimizer steps
within an advance; `checkpointInterval: 1` restores per-step checkpoints.
Every written checkpoint is a full durable recovery point, so crash
recovery resumes from the most recent one. The setting is runtime-only:
changing it does not change the config digest or session lineage, while
checkpoints retain it so a fresh service resumes with the same cadence.

Set `loraExportInterval` to an integer N >= 1 to additionally export the
float32 adapter masters every N optimizer steps within an advance. These
lightweight files use the same standard layout as `training.export_lora`:
kohya for SD1.5, SDXL, and Wan, or PEFT for MiniMax H3 and Music 3. They are
written to `exports/cadence/<session-id>/step-<12-digit-step>.safetensors`
under the checkpoint root. The `cadence` directory is reserved from manual
exports. The interval is independent of `checkpointInterval`, so it can add
adapter snapshots alongside intermediate full checkpoints or provide the only
intermediate cadence while advance-boundary, pause, and creation recovery
checkpoints remain unchanged. The default `0` writes no cadence exports. This
setting is runtime-only and is retained in full checkpoints across restarts.

Cross-rank adapter digest checks run at every advance boundary and checkpoint
write. Set `syncDigestInterval` to an integer N >= 1 to additionally check every
N optimizer steps. The default `0` avoids extra checks between durable safe
points. This setting is runtime-only and checkpoints retain it across restarts.

`training.export_lora` writes the committed checkpoint named by its session
handle as a deterministic LoRA safetensors file: kohya layout for SD1.5 and
SDXL UNets, or PEFT layout for MiniMax H3 and Music 3 DiTs. Its JSON settings
require a `path` relative to the checkpoint root's `exports` directory and
accept `dtype` as `fp16` (the default), `bf16`, or `fp32`. The output includes
the resolved path and a `blake3:` content digest.

## Worker configuration

Select the backend with:

```text
DINKSTER_TRAINING_BACKEND=sd15-lora
DINKSTER_TRAINING_JOURNAL=/path/to/training.sqlite
DINKSTER_TRAINING_CHECKPOINT_ROOT=/path/to/training-checkpoints
```

Use `DINKSTER_TRAINING_BACKEND=sdxl-lora` for SDXL base training.
Use `DINKSTER_TRAINING_BACKEND=minimax-h3-lora` for MiniMax H3 DiT training.
Use `DINKSTER_TRAINING_BACKEND=minimax-music3-lora` for MiniMax Music 3 DiT
training.

### Distributed training

The JSON node config can include an optional distributed section:

```json
{
  "rngPolicy": "counter",
  "distributed": {
    "worldSize": 2,
    "backend": "nccl",
    "rendezvous": {"method": "tcp", "host": "127.0.0.1", "port": 29500}
  }
}
```

`rngPolicy` is independent of the distributed section. It defaults to `sequential`,
which preserves sequential streams for one rank and rank-specific streams for
multiple ranks. The `counter` policy instead derives fresh data and timestep/noise
streams from each absolute data cursor, so a sample receives the same draws at every
world size. `worldSize` still changes the training identity because it changes the
global batch. `backend` and `rendezvous` are runtime-only, so they can change when a
committed session resumes. CPU training keeps every rank on CPU. For CUDA training,
the shared `cuda` or `cuda:0` selector maps rank `r` to `cuda:r`; a world size greater
than one therefore requires at least that many visible CUDA devices. Other explicit
CUDA indices are not supported for multi-rank training.

TCP rendezvous uses a host and port reachable by every local rank. File rendezvous
uses `{"method": "file", "path": "/absolute/path/to/rendezvous"}`. The service
removes its stale file before bootstrap, so the configured path must not be shared
with unrelated processes or sessions. Process-group initialization and collectives
use a 120-second timeout.

Single-rank checkpoints preserve the existing bare `data` and `timestep-noise` RNG
keys. Multi-rank checkpoints contain a complete matrix with `rankN:data` and
`rankN:timestep-noise` for every rank. Each written checkpoint gathers the full
matrix before publishing the recovery checkpoint, so a resumed run continues every
rank's data and random streams exactly. Counter-policy checkpoints carry one
stateless policy marker instead of generator state. The marker has no world-size
constraint, so a host can restore the checkpoint with a different rank count.

The JSON node config accepts a native UNet geometry, an exact safetensors base
state path, required `blake3:` digest, and optional key prefix, a prepared-batch
root or image/caption dataset, device, base dtype, LoRA rank/alpha, optimizer
settings, accumulation, checkpointing, and seed. A folder dataset has this
shape:

SD1.5 and SDXL activation checkpointing defaults to `checkpointingMode` set to
`blockReentrant`, which checkpoints each UNet input, middle, and output block.
Set it to `wholeModel` to preserve the original bit-exact training trajectory.
The SD mode is persisted in session identity and is consulted only when
`gradientCheckpointing` is true. MiniMax H3 accepts only its implicit
`wholeModel` default until it has a block checkpointing path.

```json
{
  "dataset": {
    "type": "image-caption-folder",
    "root": "/data/animals",
    "encodedCacheRoot": "/data/dinkster-encoded-cache",
    "resolution": [512, 512],
    "vaeState": {"path": "/models/vae.safetensors", "digest": "blake3:...", "prefix": ""},
    "textEncoderState": {"path": "/models/clip.safetensors", "digest": "blake3:...", "prefix": ""}
  }
}
```

`encodedCacheRoot` is optional and runtime-only: changing it does not change
the training config digest or session lineage. The complete setting is still
stored in checkpoints so a fresh service can use the same cache while
resuming. Without it, precomputation and the in-memory tensor store behave as
before. For an existing session, the value stored in its checkpoints wins;
changing the submitted root affects only new sessions.

Wan training uses the shifted 1000-step `WAN21_SIGMAS` flow space used by
inference. The DiT receives `sigma * 1000` timesteps and predicts velocity
against `noise - latent` from `(1 - sigma) * latent + sigma * noise`. Wan
activation checkpointing defaults to `wholeModel`. Set `checkpointingMode` to
`blockNonReentrant` to checkpoint each DiT block independently while leaving
the embeddings and output head outside checkpoint regions. The frozen DiT
remains in the selected float16 or bfloat16 dtype while LoRA masters and loss
math remain float32.

`baseState` and both data-source forms can be omitted only when an embedding
host injects model and data-source factories.

SDXL uses one standard single-file base checkpoint for `baseState` and the
dataset's `checkpointState`. Both sources must have the same empty-prefix path
and digest. The loader rejects refiners and checkpoints marked for v-prediction
or EDM sampling.

H3 config selects `fl2va-dit` or `ref2va-dit` and pins the corresponding
official DiT asset by path, BLAKE3 digest, byte size, and native component
identity. It also pins the conditioner component identity used to produce the
prepared embeddings and carrier. These identities compose into the session's
H3 execution identity. The DiT identity must be derived for the configured
`baseDtype`. The loader verifies the immutable artifact role, plans the
component, checks the expected identity, and strictly loads the DiT. Source
storage dtypes remain intact. Forward inputs and LoRA views use the configured
model dtype without autocast, while clean latents, velocity targets, loss math,
LoRA masters, and optimizer state remain float32. Set `quantizedBase` to `true`
with `baseDtype` set to `bfloat16` to select an INT8 DiT. INT8 weights remain
frozen in their source representation. By default, each projection
deterministically dequantizes to bfloat16 for the standard linear forward so
input and LoRA gradients use ordinary autograd. Set `int8BaseForward` to
`fused` on CUDA to use comfy-kitchen's W8A8 forward instead. That opt-in route
computes frozen-base input gradients by dequantizing bounded bfloat16 feature
chunks; LoRA gradients remain ordinary autograd. ConvRot weights use the pinned
comfy-kitchen operations on both routes.

Set `hostLayerPagingFraction` to a value in `(0, 1]` on a CUDA device to keep
that tail fraction of frozen DiT transformer layers in pinned host RAM. The
trainer streams source-dtype layer tensors one layer ahead on a dedicated CUDA
transfer stream in forward order and reverse order during backward. LoRA
masters, gradients, and optimizer state remain on the training device. The
setting is runtime-only: changing it does not change the config digest or
session handle, while checkpoints retain it so a fresh service reconstructs
the same paging policy. The default `0` keeps the complete frozen DiT on the
training device.

An FL2VA folder dataset also pins the official video VAE, audio VAE, and
Qwen3-VL conditioner artifacts. For example:

```json
{
  "ditRole": "fl2va-dit",
  "conditionerIdentity": "native:dinkster.minimax_h3:...",
  "videoLatentShape": [1, 24, 2, 64, 64],
  "audioLatentShape": [1, 32, 2, 8],
  "conditionerShape": [1, 12, 5120],
  "dataset": {
    "type": "h3-video-audio-caption-folder",
    "root": "/data/h3-clips",
    "encodedCacheRoot": "/data/dinkster-encoded-cache",
    "resolution": [1024, 1024],
    "frameCount": 5,
    "videoVaeState": {"path": "/models/video-vae.safetensors", "digest": "blake3:...", "size": 1, "identity": "native:dinkster.minimax_h3:..."},
    "audioVaeState": {"path": "/models/audio-vae.safetensors", "digest": "blake3:...", "size": 1, "identity": "native:dinkster.minimax_h3:..."},
    "conditionerState": {"path": "/models/qwen3vl-32b.safetensors", "digest": "blake3:...", "size": 1, "identity": "native:dinkster.minimax_h3:..."}
  }
}
```

Use each artifact's exact byte size in place of the abbreviated example
values. Dataset content, preprocessing, and all three component pins compose
the dataset digest. Container-item identity additionally covers the complete
container bytes and pinned decoder semantics. Frame-folder datasets retain
their existing identity. `encodedCacheRoot` remains runtime-only and does not
change session lineage.

### MiniMax Music 3 community-derived training

MiniMax Music 3 training is an explicitly community-derived path, not an
official MiniMax training recipe or source-parity claim. The engineering recipe
is pinned to SimpleTuner commit
`def7bbc065e5f15d9e551827e247ba046fe36eb6`. The DAV conversion and RVQ
approximation are pinned to these immutable sources:

- `SimpleTuner/MiniMax-Music-3-Encoder@fce0d00b1ae42ee47874babb8c06fb859eb01443`,
  `audio_vae/diffusion_pytorch_model.safetensors`, 306,466,152 bytes,
  SHA-256 `ea6d2458de8d71e3d8b8210362ab31c547ac3c99bafa53ba004f3751acb5428e`,
  BLAKE3 `ab71ad1a4706c5532ad34e2ea7af1027357e54ef1c8e2b1dfdf8d7d08fbd9df7`.
- `SimpleTuner/open-rvq-encoder-minimax-music3@326964c2f4edcc642c1ea116274dd2dd94081713`,
  `encoders/minimax_music3_rvq_encoder_v4_169m_autoregressive_depth_recommended.safetensors`,
  676,055,232 bytes,
  SHA-256 `e8fa93a7db2e5a090442d9492ea3f9f07664ce5bdb2ea7890559a4408f049744`,
  BLAKE3 `c0547f9be17f60341d71a8df04029679759ea0b140cdd37b4b0ab55abf0b57e0`.

Each recursive dataset item is a fixed-duration uncompressed signed 16-bit PCM
`.wav` at 44.1 kHz with one or two channels plus nonempty, same-basename UTF-8
`.caption.txt` and `.lyrics.txt` files. `audioFrames` is in `[1, 128]` at 25 Hz,
so every item has exactly `audioFrames * 1764` samples. The dataset digest covers
all source bytes, names, preprocessing contracts, and DAV, RVQ, and text-model
artifact pins.

The DAV path duplicates mono, right-pads to a 512-sample hop, independently
encodes both channels, and concatenates posterior means as
`[batch, 128, ceil(samples / 512)]`. The community RVQ v4 approximation maps
those latents to eight 25 Hz codebooks using the released open-RVQ reference
adapter's `floor(frame * 441 / 128)` boundaries, which intentionally exclude
DAV positions containing only right-padding. The pinned SimpleTuner trainer's
legacy cache uses different duration-local, ceiling-rounded boundaries; this
path follows the released encoder artifact contract and does not claim parity
with that legacy cache. The native autoregressive text model teacher-forces
caption, lyrics, C0, and seven depth codes to produce one 32,768-wide
conditioning row per audio frame. DAV, RVQ, and text components load
sequentially under `torch.no_grad` and are released before the DiT loads. An
optional `encodedCacheRoot` stores float32 latents and conditioning in a locked,
atomic, digest-verified safetensors cache. A verified hit loads no encoder.

The source objective draws `t = sigmoid(N(0, 1))`, forms
`x_t = t * data + (1 - t) * noise`, sends data-time `t` to the DiT, and minimizes
ordinary float32 mean squared error against `data - noise`. Training targets
the global `project_in` and `project_out` plus each block's fused
`self_attn.to_qkv`, `self_attn.to_out`, `ff.ff.0.proj`, and `ff.ff.2`: 146
stable linear targets. Export maps every target to
`diffusion_model.<module>.lora_A.weight`, `.lora_B.weight`, and `.alpha`, which
the native inference LoRA decoder consumes directly.

Example config (all digests, sizes, and native identities must match the exact
local artifacts):

```json
{
  "schemaVersion": 1,
  "family": "minimax-music3",
  "communityRecipeRevision": "SimpleTuner/def7bbc065e5f15d9e551827e247ba046fe36eb6",
  "flowObjective": "community-minimax-music3-logistic-normal-flow-v1",
  "diffusionState": {"path": "/models/minimax_music3_dit_fp16.safetensors", "digest": "blake3:...", "size": 1, "identity": "native:dinkster.minimax_music3:e..."},
  "dataset": {
    "type": "minimax-music3-audio-caption-lyrics-folder",
    "root": "/data/music3",
    "audioFrames": 128,
    "encodedCacheRoot": "/data/dinkster-encoded-cache",
    "davEncoderState": {"path": "/models/music3-dav.safetensors", "digest": "blake3:...", "size": 306466152},
    "rvqEncoderState": {"path": "/models/music3-rvq.safetensors", "digest": "blake3:...", "size": 676055232},
    "textEncoderState": {"path": "/models/music3-text.safetensors", "digest": "blake3:...", "size": 1, "identity": "native:dinkster.minimax_music3:e..."}
  },
  "device": "cuda:0",
  "baseDtype": "float16",
  "textDtype": "bfloat16",
  "rank": 4,
  "alpha": 4.0,
  "learningRate": 0.0001,
  "gradientCheckpointing": true,
  "seed": 1176
}
```

The native FP16 and FP32 diffusion artifacts use matching `baseDtype` values.
The INT8 ConvRot artifact requires `baseDtype: "bfloat16"` and
`quantizedBase: true`; it remains frozen and has no optimizer state or base
gradients. LoRA masters, gradients, optimizer state, objective inputs, and loss
math remain float32 for every base. There is no automatic precision downgrade,
base-model CPU offload, retry, or OOM fallback. An OOM propagates without
changing the selected artifact or recipe.

The DAV conversion reports derivation from the official `dav.pth`, but the
adapter has not been validated against an official MiniMax training release.
The pinned RVQ approximation measured 0.8748 cosine replay on its synthetic
benchmark; real-audio code accuracy and training quality are not established.
Training is limited to at most 5.12-second fixed-duration items and does not
train the text, DAV, or RVQ components. The recipe revision and versioned DAV,
RVQ, teacher-forcing, objective, dataset, cache, and runtime identities prevent
this path from silently sharing sessions or cache entries with a future
official recipe. An official recipe can replace it under new versioned
identities while retaining the same worker and service seam.

## Environment and validation

The root environment is torch-free. `scripts/setup_envs.sh` on Linux/macOS and
`scripts/setup_envs.ps1` on Windows install this package into `.venv-torch`
and, on a CUDA host, `.venv-gpu`.

```bash
.venv/bin/ruff check packages/dinkster-training-torch
.venv/bin/ruff format --check packages/dinkster-training-torch
.venv/bin/pyright -p packages/dinkster-training-torch
.venv-torch/bin/python -m pytest -q packages/dinkster-training-torch/tests
```

On a CUDA host, run the same test suite with `.venv-gpu/bin/python`; the CUDA
training test must execute rather than skip.

The MiniMax Music 3 physical-GPU proof needs the pinned community DAV and RVQ
artifacts plus each inference-supported diffusion artifact and a supported text
artifact. Set `DINKSTER_MINIMAX_MUSIC3_MODELS` and
`DINKSTER_MINIMAX_MUSIC3_TRAINING_ARTIFACTS` when they are not in the test's
documented shared artifact roots, then run:

```bash
DINKSTER_MINIMAX_MUSIC3_TRAINING_PREFLIGHT=1 \
  .venv-gpu/bin/python -m pytest -q -s \
  packages/dinkster-training-torch/tests/test_minimax_music3_training_gpu.py
```

The SDXL end-to-end GPU validation uses the official SDXL 1.0 base checkpoint:

- URL: `https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/sd_xl_base_1.0.safetensors`
- Bytes: 6,938,078,334
- SHA-256: `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b`

On an RTX 5060 Ti 16 GB with torch 2.13.0+cu130, a 512x512 bfloat16 rank-2
AdamW run with activation checkpointing completed two optimizer steps with a
service restart before the second step. It used a 7,234.1 MiB PyTorch allocated
peak, 7,310 MiB reserved peak, and 7,500 MiB sampled process peak. The exported
722-target fp16 kohya LoRA decoded through the native SDXL inference loader and
every applied patch exactly matched the trainer masters after fp16 factor casts.
