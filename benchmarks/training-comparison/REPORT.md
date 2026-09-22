# SD1.5 LoRA short-run comparison

Run date: 2026-08-22

## Verdict

Dinkster's 200-step AdamW loss trajectories are directionally consistent with the
reference trainers and each same-seed rerun is exactly deterministic. The
bfloat16-base loss change differs from kohya by 0.214 and the float32-base
change also differs by 0.215, both inside the pre-declared 0.25 limit.

The exported-adapter scale check passes for both base-storage dtypes. Across
every shared layer family, the largest median factor is 1.66x and the largest
p95 factor is 1.33x, well inside the declared 5x and 10x limits. Every file
contains rank 4, alpha 4, and effective scale 1 for all 192 modules.

The low-rank forward branch lowers Dinkster's bfloat16 sampled peak from 3,762 MiB
to 3,204 MiB and its PyTorch allocated peak from 3,411.6 MiB to 2,951.9 MiB.
The new point is 396 MiB below checkpointed kohya, 612 MiB below uncheckpointed
kohya, and 238 MiB (8.0%) above AI Toolkit. The float32-base control peaks at
6,456 MiB, so bfloat16 saves 3,252 MiB (50.4%) of the sampled peak and remains
the memory-competitive Dinkster configuration.

Allocator replay confirms that the previous 472.7 MiB live effective-weight
category is gone. The remaining peak is composed of low-rank branch outputs,
native activations, normalization, convolution, residual, and attention
allocations rather than full target-shaped LoRA weights.

The collected machine-readable evidence is in [summary.json](summary.json),
[results/](results/), and [environment.json](environment.json). The AdamW A
rows include allocator-event replay for the live-allocation breakdown.

## Declared checks

The limits in [README.md](README.md) were written before the runs.

| Check | Result | Evidence |
| --- | --- | --- |
| 200 finite recorded losses per run | **Fail** | AI Toolkit's `loss/loss` key has 199 non-NULL rows at distinct steps 1-199 despite completing 200 optimizer steps; all other runs recorded 200 finite values. |
| Last-40 mean no more than 110% of first-40 | Pass | All eight trajectories improve. |
| Dinkster versus kohya normalized loss-change difference at most 0.25 | Pass | Bfloat16 base: 0.2143; float32 base: 0.2146. |
| Applied-delta median within 5x and p95 within 10x | Pass | Bfloat16 max: 1.66x median / 1.33x p95; float32 max: 1.65x / 1.33x. |
| Same-seed Dinkster exact loss and export | Pass | Both dtype pairs have exact loss arrays and exports. Bfloat16 SHA-256: `56b96823186b5d6053088845c8e41064ed3c7ec7767efcaff3daaf94151e297e`; float32: `2e23a045ceed06a473095ede11234e6389bfe8fa6b9915bbc859d7ca647dc28b`. |

Memory was pre-declared as comparative evidence rather than a pass/fail gate.

## Environment and provenance

| Item | Value |
| --- | --- |
| Host | `5800XT1L`, Linux `7.0.0-29-generic`, x86-64 |
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB, driver 595.84 |
| Python | 3.12.3 in all three environments |
| PyTorch | 2.13.0+cu130 in all three environments |
| Dinkster source | This report's change, based on `59745db01ad2fe69be6a4a5487d2f73f34d12349` |
| kohya | `https://github.com/kohya-ss/sd-scripts.git` at `37a1cbbc5725ed2a3575506e7bd2001c9908ac92` |
| AI Toolkit | `https://github.com/ostris/ai-toolkit.git` at `27a03a91f23eb1b757d5ec2e80ee3e129cfcc350` |
| Storage free | 840,918,523,904 bytes before the final Dinkster rerun; 840,918,278,144 bytes at final evidence collection |

Each reference was cloned outside the Dinkster repository and installed in its
own virtual environment. The exact installed package inventories are recorded
in [environment.json](environment.json).

The base is `v1-5-pruned-emaonly.safetensors` from Comfy-Org's immutable
Stable Diffusion 1.5 archive revision:

- URL: `https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/c36740b77a55ec396ace7c8c26589cdf2b4bc3da/v1-5-pruned-emaonly.safetensors`
- Bytes: 4,265,146,304
- SHA-256: `6ce0161689b3853acaa03779ec93eafe75a02f4ced659bee03f50797806fa2fa`

The committed generator created 16 fixed 512x512 synthetic images and matching
captions. All trainers consumed those same files. The generator version and
all 32 file hashes are in [environment.json](environment.json).

### Standard checkpoint path

Dinkster and both reference trainers read the unmodified hash-pinned checkpoint.
The Dinkster loader discards the inert
`cond_stage_model.transformer.text_model.embeddings.position_ids` buffer while
retaining strict rejection for other unexpected CLIP keys. No derived CLIP
artifact or checkpoint workaround was used.

AI Toolkit's single-file conversion also populated the Hugging Face cache with
Stable Diffusion configuration/tokenizer revision
`451f4fe16113bff5a5d2269ed5ad43b0592e9a14` and safety-checker revision
`cb41f3a270d63d454d385fc2e4f571c487c253c5`. Its downloaded
`pytorch_model.bin` was 1,216,067,303 bytes with SHA-256
`64b8393f1afd5a0c1ed2aa5f341fa7c08286839a48f3743162a76a2835c808bd`.
Sampling and safety checking were disabled; this is a setup reproducibility
caveat rather than a training input.

## Configuration

| Setting | Dinkster | kohya | AI Toolkit |
| --- | --- | --- | --- |
| Model | Same SD1.5 checkpoint | Same checkpoint | Same checkpoint |
| Target | UNet only | UNet only | UNet only |
| Rank / alpha | 4 / 4 | 4 / 4 | 4 / 4 |
| Resolution | 512x512 | 512x512 | 512x512 |
| Batch / accumulation | 1 / 1 | 1 / 1 | 1 / 1 |
| Optimizer steps | 200 | 200 | 200 |
| Optimizer | AdamW | AdamW | AdamW |
| LR / scheduler | 1e-4 / constant | 1e-4 / constant | 1e-4 / constant |
| Betas / weight decay | 0.9, 0.999 / 0.01 | 0.9, 0.999 / 0.01 | 0.9, 0.999 / 0.01 |
| Adam epsilon | 1e-8 | 1e-8 | 1e-6 fixed by the optimizer factory |
| Frozen base storage | Bfloat16 and float32 measured | Bfloat16 | Bfloat16 |
| Forward dtype | bf16 autocast | bf16 mixed precision | bf16 |
| Attention | PyTorch SDPA | PyTorch SDPA | PyTorch SDP |
| Gradient checkpointing | Enabled | Measured enabled and disabled | Enabled |
| Gradient clipping | Disabled | Disabled | `max_grad_norm=1e9`, effectively inactive |
| Dataset encoding cache | CPU latents and CLIP embeddings | Disabled | Disabled |
| Seed | 1234 | 1234 | `SEED=1234` |

The tools do not share RNG implementation, data-loader ordering, image
preprocessing code, timestep/noise stream partitioning, or model wrappers.
Per-step equality across tools is neither expected nor claimed. AI Toolkit
does not expose AdamW epsilon through this config path. It also always invokes
gradient clipping for AdamW, so 1e9 is used as the closest non-clipping value.

## Loss trajectories

The statistic uses the first and last 40 recorded losses. Negative normalized
change means improvement.

| Run | Recorded losses | First-40 mean | Last-40 mean | Normalized change | Trend check |
| --- | ---: | ---: | ---: | ---: | --- |
| Dinkster bfloat16-base AdamW A | 200 | 0.022606 | 0.017466 | -0.2274 | Pass |
| Dinkster bfloat16-base AdamW B | 200 | 0.022606 | 0.017466 | -0.2274 | Pass |
| Dinkster bfloat16-base FactoredAdamW | 200 | 0.023646 | 0.016955 | -0.2830 | Pass |
| Dinkster float32-base AdamW A | 200 | 0.022601 | 0.017467 | -0.2272 | Pass |
| Dinkster float32-base AdamW B | 200 | 0.022601 | 0.017467 | -0.2272 | Pass |
| kohya, checkpointing | 200 | 0.021074 | 0.011765 | -0.4417 | Pass |
| kohya, no checkpointing | 200 | 0.021076 | 0.011765 | -0.4418 | Pass |
| AI Toolkit | 199 | 0.020454 | 0.011786 | -0.4238 | Pass; count fails |

Dinkster improves less than either reference over this short run, but the
bfloat16-base difference of 0.2143 and float32-base difference of 0.2146 from
the pre-declared kohya comparison stay inside 0.25. AI Toolkit's progress log
reaches 200/200 and saves an adapter,
while its SQLite metrics database has exactly 199 non-NULL `loss/loss` rows at
distinct steps 1 through 199. Step 0 is absent. The cause was not determined,
so the report does not attribute a logger mechanism. The statistics use the
199 values actually present and do not fill the absent step.

## Adapter behavior

`adapter_stats.py` is the neutral applier. For every module it reads the
exported `lora_down`, `lora_up`, and actual per-module `alpha` tensor, then
computes `lora_up @ lora_down * (alpha / rank)`. It uses rank only as the
fallback when an alpha tensor is absent. All measured files contain alpha, so
the fallback was not used. Adding these deltas to the same base weights would
not change the delta distributions and therefore does not require loading the
4.3 GB base for analysis.

The 192 modules have identical family coverage in all three files: 64
cross-attention, 32 feed-forward, 64 self-attention, and 32 spatial-projection
modules.

### Alpha and effective scale audit

| Export | Alpha tensors | Rank, every family | Alpha, every family | Effective alpha/rank |
| --- | ---: | ---: | ---: | ---: |
| Dinkster AdamW, both base dtypes | 192 / 192 | 4 | 4 | 1.0 |
| kohya, checkpointing | 192 / 192 | 4 | 4 | 1.0 |
| AI Toolkit | 192 / 192 | 4 | 4 | 1.0 |

There is no alpha/rank convention mismatch in these files.

### Applied-delta gate

Values are the factor between the Dinkster and reference RMS distributions for
the median and p95. The declared limits are 5x median and 10x p95.

| Dinkster base | Reference | Family | Median factor | p95 factor | Result |
| --- | --- | --- | ---: | ---: | --- |
| Bfloat16 | kohya | Cross-attention | 1.23x | 1.29x | Pass |
| Bfloat16 | kohya | Feed-forward | 1.11x | 1.24x | Pass |
| Bfloat16 | kohya | Self-attention | 1.28x | 1.33x | Pass |
| Bfloat16 | kohya | Spatial projection | 1.22x | 1.20x | Pass |
| Bfloat16 | AI Toolkit | Cross-attention | 1.66x | 1.26x | Pass |
| Bfloat16 | AI Toolkit | Feed-forward | 1.30x | 1.17x | Pass |
| Bfloat16 | AI Toolkit | Self-attention | 1.24x | 1.13x | Pass |
| Bfloat16 | AI Toolkit | Spatial projection | 1.09x | 1.09x | Pass |
| Float32 | kohya | Cross-attention | 1.23x | 1.29x | Pass |
| Float32 | kohya | Feed-forward | 1.11x | 1.24x | Pass |
| Float32 | kohya | Self-attention | 1.28x | 1.33x | Pass |
| Float32 | kohya | Spatial projection | 1.23x | 1.20x | Pass |
| Float32 | AI Toolkit | Cross-attention | 1.65x | 1.26x | Pass |
| Float32 | AI Toolkit | Feed-forward | 1.29x | 1.17x | Pass |
| Float32 | AI Toolkit | Self-attention | 1.24x | 1.13x | Pass |
| Float32 | AI Toolkit | Spatial projection | 1.09x | 1.09x | Pass |

The secondary normalized-scale comparison divides each applied delta by its
effective alpha/rank. It is exactly the same table because all effective
scales are 1.0. The primary gate passes.

### Raw-factor audit

The table reports the bfloat16-base factor between raw RMS distributions. Down
factors are within 1.03x in every comparison. Up-factor directions vary, but
their median magnitudes remain within 1.64x. The float32-base distributions are
recorded in [results/dinkster-fp32-adamw-a.json](results/dinkster-fp32-adamw-a.json).

| Reference | Family | Down median factor | Down p95 factor | Up median factor | Up p95 factor |
| --- | --- | ---: | ---: | ---: | ---: |
| kohya | Cross-attention | 1.00x | 1.01x | 1.21x | 1.25x |
| kohya | Feed-forward | 1.00x | 1.00x | 1.11x | 1.11x |
| kohya | Self-attention | 1.00x | 1.01x | 1.25x | 1.20x |
| kohya | Spatial projection | 1.01x | 1.02x | 1.19x | 1.13x |
| AI Toolkit | Cross-attention | 1.02x | 1.01x | 1.61x | 1.19x |
| AI Toolkit | Feed-forward | 1.02x | 1.01x | 1.18x | 1.17x |
| AI Toolkit | Self-attention | 1.01x | 1.01x | 1.17x | 1.13x |
| AI Toolkit | Spatial projection | 1.01x | 1.02x | 1.08x | 1.07x |

All three trainers initialize `lora_down` with Kaiming uniform for
`a=sqrt(5)` and initialize `lora_up` to zero. Dinkster computes the equivalent
bound from fan-in, including the convolutional receptive field, and draws with
its checkpointed generator
([attachment.py](../../packages/dinkster-training-torch/src/dinkster_training_torch/attachment.py#L101-L111)).
kohya applies the same recipe to both linear and convolutional down factors
([networks/lora.py](https://github.com/kohya-ss/sd-scripts/blob/37a1cbbc5725ed2a3575506e7bd2001c9908ac92/networks/lora.py#L65-L77)),
and AI Toolkit uses the same construction
([toolkit/kohya_lora.py](https://github.com/ostris/ai-toolkit/blob/27a03a91f23eb1b757d5ec2e80ee3e129cfcc350/toolkit/kohya_lora.py#L60-L73)).
The measured median down RMS is 0.017-0.023 for feed-forward and attention
families and 0.023-0.024 for spatial projections across all trainers.

FactoredAdamW was included as a memory data point rather than the cross-tool
correctness comparator. Its median applied deltas are 0.00116-0.00166, larger
than Dinkster AdamW's 0.000093-0.000121, so its adapter behavior should not be
treated as interchangeable based on this short run.

### AI Toolkit zero-adapter triage

An initial run copied kohya's `max_grad_norm=0` setting into AI Toolkit. Its
saved adapter had all 192 `lora_up` tensors literally zero and all 192 down
tensors nonzero; its optimizer state had all 768 moment tensors zero. AI
Toolkit unconditionally calls `clip_grad_norm_` for AdamW
([SDTrainer.py](https://github.com/ostris/ai-toolkit/blob/27a03a91f23eb1b757d5ec2e80ee3e129cfcc350/extensions_built_in/sd_trainer/SDTrainer.py#L2273-L2286)),
so zero erased every gradient. This was a harness configuration semantic
mismatch, not an adapter-key mapping problem; the invalid adapter had SHA-256
`c3cd07f7018e2f3874cefa7a26592205bbfad05952066bd859509266dd6bab03`.

The committed config uses `max_grad_norm: 1e9` as effectively unclipped. In
the corrected run, all 192 up tensors and all 768 optimizer moments are
nonzero. The loss, memory, and adapter results elsewhere in this report are
from that corrected run.

## Determinism control

Each pair of same-seed Dinkster AdamW runs produced exactly equal arrays of 200
losses and byte-identical fp32 exports. The bfloat16-base export hash is
`56b96823186b5d6053088845c8e41064ed3c7ec7767efcaff3daaf94151e297e`; the
float32-base hash is
`2e23a045ceed06a473095ede11234e6389bfe8fa6b9915bbc859d7ca647dc28b`.
Both pass the declared control. This establishes reproducibility within this
machine and software environment, not cross-hardware determinism.

## Memory

`monitor.py` samples the complete process tree's `nvidia-smi` memory every 0.1
seconds. External trainers also load `instrumentation/sitecustomize.py`, which
polls PyTorch peaks every 0.05 seconds and merges maxima across child Python
processes under a file lock. Dinkster reports the same PyTorch counters directly.

| Run | PyTorch allocated peak | PyTorch reserved peak | `nvidia-smi` peak | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Dinkster bfloat16-base AdamW A | 2,951.9 MiB | 3,014.0 MiB | 3,204 MiB | 73.90 s |
| Dinkster bfloat16-base AdamW B | 2,951.9 MiB | 3,014.0 MiB | 3,204 MiB | 73.43 s |
| Dinkster bfloat16-base FactoredAdamW | 2,947.3 MiB | 3,010.0 MiB | 3,200 MiB | 85.54 s |
| Dinkster float32-base AdamW A | 6,170.2 MiB | 6,266.0 MiB | 6,456 MiB | 74.80 s |
| Dinkster float32-base AdamW B | 6,170.2 MiB | 6,266.0 MiB | 6,456 MiB | 76.75 s |
| kohya, checkpointing | 2,942.9 MiB | 3,410.0 MiB | 3,600 MiB | 78.96 s |
| kohya, no checkpointing | 3,594.1 MiB | 3,626.0 MiB | 3,816 MiB | 58.20 s |
| AI Toolkit | 2,557.9 MiB | 2,776.0 MiB | 2,966 MiB | 64.62 s |

Bfloat16 base storage saves 3,218.3 MiB of allocated peak and 3,252 MiB of
sampled peak versus float32 base storage. The sampled peak falls 50.4% and is
11.0% below checkpointed kohya, 16.0% below uncheckpointed kohya, and 8.0%
above AI Toolkit. Its
whole-UNet checkpointing path is visible in
[trainer.py](../../packages/dinkster-training-torch/src/dinkster_training_torch/trainer.py#L193-L208),
and its dataset source now precomputes every latent, releases the VAE, then
precomputes every CLIP embedding and releases CLIP
([data.py](../../packages/dinkster-training-torch/src/dinkster_training_torch/data.py#L264-L397)).
Only encoded CPU tensors remain during training. A direct timing of the 16-item
precompute took 6.26 seconds; the complete monitored AdamW run improved from
89.35 to 71.74 seconds when encoder precomputation landed. The new low-rank
branch runs took 73.43-73.90 seconds, about 2 seconds longer than the previous
full-weight rows while using 558 MiB less sampled GPU memory.

The in-memory store costs 295 KiB per 512x512 SD1.5 item: 64 KiB for the
float32 latent and 231 KiB for the float32 CLIP embedding. The 16-item harness
store is 4.61 MiB. This comparison uses no disk cache. FactoredAdamW reduces
allocated memory by another 4.6 MiB because LoRA optimizer state is small
relative to the frozen base and activations.

All five Dinkster rows were collected fresh. Both AdamW A rows include allocator
replay; their matching B rows reproduce every loss, export byte, and headline
memory peak without replay. Float32 sampled memory rises 22 MiB because its
peak is dominated by native float32-base operation allocations under autocast,
not by the removed effective LoRA weights.

### Dinkster phase attribution

The harness records `torch.cuda.memory_stats`, allocator block snapshots, and
process memory after data precompute, after UNet residency, after the first
step, and after the final step. The figures are from the two 200-step AdamW A
runs.

| Component or phase | Bfloat16 base | Float32 base | Interpretation |
| --- | ---: | ---: | --- |
| Encoded dataset after VAE/CLIP release | 32.0 MiB allocated | 32.0 MiB allocated | Persistent PyTorch allocation; all 4.61 MiB of prepared data is CPU-resident. |
| Frozen UNet parameter tensors | 1,639.4 MiB | 3,278.8 MiB | Exact tensor bytes. |
| LoRA masters | 6.5 MiB | 6.5 MiB | Float32 in both modes. |
| Allocated after UNet residency | 1,690.5 MiB | 3,335.1 MiB | Base, LoRA masters, and allocator bookkeeping. |
| LoRA gradients after step | 6.5 MiB | 6.5 MiB | Float32 in both modes. |
| AdamW state after step | 12.9 MiB | 12.9 MiB | Float32 first and second moments. |
| Autograd-saved tensors in checkpointed step | 0.4 MiB | 0.4 MiB | Five tensors; explicit model inputs account for 0.29 MiB. |
| First-step allocated peak | 2,939.3 MiB | 6,156.5 MiB | Forward/backward transient peak. |
| Allocated after first step | 1,741.9 MiB | 3,386.5 MiB | Resident model plus gradients and optimizer state. |
| Inactive allocator blocks after first step | 1,260.1 MiB | 2,867.5 MiB | Cached consequence of the transient peak, not live tensors. |
| CUDA context and non-allocator memory | 190 MiB | 190 MiB | `nvidia-smi` process bytes minus PyTorch reserved bytes. |

The checkpoint boundary, retained prepared/noised inputs, gradients, optimizer
state, allocator fragmentation, and CUDA context are all too small or shared
between the compared modes to explain the large step peak. The bfloat16 step
temporarily allocates 1,248.7 MiB above post-load residency, 461.2 MiB less
than the full-weight path. The float32 step allocates 2,821.4 MiB above
residency.

Dinkster now applies each adapter as a low-rank branch beside the frozen operation
([attachment.py](../../packages/dinkster-training-torch/src/dinkster_training_torch/attachment.py#L135-L171)).
The float32 master shapes, seeds, initialization, checkpoint keys, and export
path remain unchanged. There is no full target-shaped delta or effective
weight in the runtime path.

Replaying allocator events at the bfloat16 first-step peak records 1,235.5 MiB
of live allocations. The previous 472.7 MiB effective-weight category is
absent. Group normalization accounts for 264.9 MiB, base-plus-branch output
addition for 248.0 MiB, one UNet residual/skip site for 156.2 MiB, layer
normalization for 138.1 MiB, low-rank linear down and up outputs for 116.6 MiB
and 41.6 MiB, native convolutions for 87.5 MiB, native linear outputs for 52.0
MiB, attention for 45.7 MiB, and low-rank convolution outputs for 22.6 MiB.
These are live-at-peak allocations rather than cumulative operator traffic.

AI Toolkit instead runs the frozen operation and adds the low-rank
`up(down(input))` branch without constructing a full effective weight
([network_mixins.py](https://github.com/ostris/ai-toolkit/blob/27a03a91f23eb1b757d5ec2e80ee3e129cfcc350/toolkit/network_mixins.py#L304-L348)).
It enables Diffusers block-level checkpointing
([BaseSDTrainProcess.py](https://github.com/ostris/ai-toolkit/blob/27a03a91f23eb1b757d5ec2e80ee3e129cfcc350/jobs/process/BaseSDTrainProcess.py#L1863-L1904)).
Because this comparison disables latent and text-embedding caches, its VAE is
moved back to CUDA and its text encoder is not unloaded
([SDTrainer.py](https://github.com/ostris/ai-toolkit/blob/27a03a91f23eb1b757d5ec2e80ee3e129cfcc350/extensions_built_in/sd_trainer/SDTrainer.py#L294-L383)).
Those extra frozen encoders contain 394.3 MiB of bfloat16 parameters, yet AI
Toolkit still peaks 394.0 MiB lower in PyTorch allocations. Its measured peak
leaves at most about 500 MiB beyond the exact UNet, VAE, CLIP-L, LoRA, gradient,
and AdamW tensor sizes, versus about 1,210 MiB beyond Dinkster's post-step live
allocation. Subtracting AI Toolkit's extra frozen encoders leaves about 788 MiB
of like-for-like peak difference, now localized to Dinkster's remaining step-time
activation and operator temporaries.

A focused comparison of Dinkster's whole-UNet checkpoint against AI Toolkit's
Diffusers block checkpoints can attribute the remaining normalization,
residual, convolution, linear, and attention categories. The float32 allocator
replay is instead dominated by 1,683.7 MiB of native convolution and linear
allocations under autocast, which explains why removing full LoRA weights does
not reduce that control's peak.

The pinned kohya source ships these relevant SD1.5 controls:

- gradient checkpointing and optional CPU-offloaded checkpointing
  ([train_network.py](https://github.com/kohya-ss/sd-scripts/blob/37a1cbbc5725ed2a3575506e7bd2001c9908ac92/train_network.py#L1123-L1134));
- SDPA, xformers, and memory-efficient attention replacement
  ([train_network.py](https://github.com/kohya-ss/sd-scripts/blob/37a1cbbc5725ed2a3575506e7bd2001c9908ac92/train_network.py#L172-L180));
- RAM or disk latent caching
  ([train_network.py](https://github.com/kohya-ss/sd-scripts/blob/37a1cbbc5725ed2a3575506e7bd2001c9908ac92/train_network.py#L191-L195));
- fp8 UNet base storage
  ([train_network.py](https://github.com/kohya-ss/sd-scripts/blob/37a1cbbc5725ed2a3575506e7bd2001c9908ac92/train_network.py#L1234-L1254));
- AdamW8bit and paged optimizer choices in `library/optimizer.py`.

This harness measured kohya gradient checkpointing on and off with SDPA. It
did not measure xformers, latent caching, fp8, CPU offload, or 8-bit optimizer
variants.

AI Toolkit ships gradient-checkpoint configuration and exercised it here. Its
configuration source labels `quantize`, text-encoder quantization, and
`low_vram` as "only for flux for now"
([config_modules.py](https://github.com/ostris/ai-toolkit/blob/27a03a91f23eb1b757d5ec2e80ee3e129cfcc350/toolkit/config_modules.py#L675-L689)).
Those fields were disabled and are not counted as available SD1.5
quantized-base or low-VRAM modes. The measured AI Toolkit point is therefore
its ordinary SD1.5 bf16, checkpointed, SDP configuration.

## Conclusions

1. **Loss correctness passes its comparison gate.** Dinkster learns in the same
   direction over 200 steps and remains within the pre-declared normalized
   loss-change tolerance relative to kohya.
2. **Applied adapter behavior passes its scale gate.** Matching the references'
   Kaiming-uniform down-factor initialization brings every family inside 1.66x
   at the median and 1.33x at p95.
3. **Determinism passes.** Same seed, data, software, and hardware reproduce
   both every loss and every export byte.
4. **Memory competitiveness is achieved against the measured kohya band.**
   Dinkster is 11.0% below kohya with checkpointing and 16.0% below kohya without
   checkpointing. The low-rank branch removes 459.7 MiB from the allocated peak
   and 558 MiB from the sampled peak. Dinkster remains 8.0% above AI Toolkit in
   sampled process memory, with the remaining difference localized to native
   activation and operator temporaries rather than effective LoRA weights.
5. **AI Toolkit coverage is complete with a documented compatibility knob.**
   The final AI run trains nonzero adapter factors. Its 199-row metrics log
   still fails the declared 200-record check despite a completed 200-step run.

These are short synthetic-data runs, not quality evaluations. They test
training dynamics, export behavior, determinism, and memory under pinned
configurations. Different seeds, longer runs, real datasets, or inference
image judgments could produce different relative outcomes.

# SDXL LoRA short-run comparison

Run date: 2026-08-22

## SDXL verdict

Dinkster's SDXL backend passes the loss-change, adapter-scale, determinism, trend,
and export-audit gates. Its normalized loss change differs from kohya by
0.2477, inside the pre-declared 0.25 limit. Every shared adapter family is
inside 4.36x at the median and 1.64x at p95, below the 5x and 10x limits.

The 200-loss completeness gate fails because AI Toolkit again records only 199
non-NULL loss rows despite completing 200 optimizer steps and exporting step
200. Dinkster, Dinkster FactoredAdamW, and kohya each record 200 finite improving
losses. No tolerance, hyperparameter, or trainer code was changed after this
finding.

Dinkster is memory-competitive under the D-53 rule. Its 7,584 MiB sampled peak is
1,072 MiB (12.4%) below kohya and 516 MiB (6.4%) below AI Toolkit. Its 7,307.3
MiB PyTorch allocated peak is 401.0 MiB (5.2%) below kohya and only 19.6 MiB
(0.27%) above AI Toolkit, which is not a material regression.

| Check | Result | Evidence |
| --- | --- | --- |
| 200 finite recorded losses per run | **Fail** | AI Toolkit records 199 non-NULL rows at steps 1-199; the other four runs record 200. |
| Last-40 mean no more than 110% of first-40 | Pass | All five trajectories improve. |
| Dinkster versus kohya normalized loss-change difference at most 0.25 | Pass | 0.247692. |
| Applied-delta median within 5x and p95 within 10x | Pass | Largest factor: 4.3548x median / 1.6369x p95. |
| Same-seed Dinkster exact loss and export | Pass | Loss arrays are equal and both exports have SHA-256 `6a726680f4209d21dfd889b4bfdf594317cd0b602500346e1f1d29a877867eb3`. |
| Rank, alpha, scale, and metadata audit | Pass | All 722 modules use rank 4, alpha 4, scale 1; family and step metadata match. |

## SDXL configuration and provenance

All three tools use the same standard SDXL 1.0 single-file checkpoint, the
same 16 generated images and captions, and the same settings below. SDXL is
conventionally trained at 1024x1024. The fixed 512x512 resolution lets all
three tools fit on a 16 GB GPU and makes this a correctness and memory
comparison, not a quality evaluation.

| Setting | Dinkster | kohya | AI Toolkit |
| --- | --- | --- | --- |
| Model | Same SDXL 1.0 checkpoint | Same checkpoint | Same checkpoint |
| Entry point | `sdxl-lora` backend | `sdxl_train_network.py` | `is_xl: true` |
| Target | UNet only, 722 linear modules | UNet only, 722 linear modules | UNet only, 722 linear modules |
| Text encoders | Frozen | Frozen | Frozen; export log reports 0 modules |
| Rank / alpha | 4 / 4 | 4 / 4 | 4 / 4 |
| Resolution | 512x512 | 512x512 | 512x512 |
| Batch / accumulation | 1 / 1 | 1 / 1 | 1 / 1 |
| Optimizer steps | 200 | 200 | 200 |
| Optimizer | AdamW | AdamW | AdamW |
| LR / scheduler | 1e-4 / constant | 1e-4 / constant | 1e-4 / constant |
| Betas / weight decay | 0.9, 0.999 / 0.01 | 0.9, 0.999 / 0.01 | 0.9, 0.999 / 0.01 |
| Adam epsilon | 1e-8 | 1e-8 | 1e-6 fixed by its optimizer factory |
| Dtype | Bfloat16 base and forward | Bfloat16 mixed precision | Bfloat16 |
| Attention | PyTorch SDPA | SDPA | SDP |
| Gradient checkpointing | Enabled | Enabled | Enabled |
| Gradient clipping | Disabled | Disabled | `max_grad_norm=1e9`, effectively inactive |
| Dataset encoding cache | CPU latents and both text embeddings | Disabled | Disabled |
| Seed | 1234 | 1234 | `SEED=1234` |

AI Toolkit always invokes gradient clipping for AdamW, so the established
`max_grad_norm=1e9` compatibility setting is reused. A value of zero erases
every gradient in this pinned source. Its Adam epsilon and data pipeline are
not configurable to match the other two paths. The trainers also have
different RNG partitioning, image preprocessing, loader order, and model
wrappers, so per-step cross-tool equality is not expected.

The pinned AI Toolkit source explicitly labels `quantize`, text-encoder
quantization, and `low_vram` as "only for flux for now"
([config_modules.py](https://github.com/ostris/ai-toolkit/blob/27a03a91f23eb1b757d5ec2e80ee3e129cfcc350/toolkit/config_modules.py#L684-L689)).
Its SDXL loading branch constructs a standard `StableDiffusionXLPipeline` and
does not apply those fields
([stable_diffusion_model.py](https://github.com/ostris/ai-toolkit/blob/27a03a91f23eb1b757d5ec2e80ee3e129cfcc350/toolkit/stable_diffusion_model.py#L335-L386)).
They were disabled and are not claimed as SDXL memory options.

| Item | Value |
| --- | --- |
| Host | `5800XT1L`, Linux `7.0.0-29-generic`, x86-64 |
| GPU | NVIDIA GeForce RTX 5060 Ti 16 GB, driver 595.84 |
| Python | 3.12.3 in all three environments |
| PyTorch | 2.13.0+cu130 in all three environments |
| Dinkster source | This report's change, based on `4b4f401bce1cee101771c9be31f8f30a9facdef2` |
| kohya | `https://github.com/kohya-ss/sd-scripts.git` at `37a1cbbc5725ed2a3575506e7bd2001c9908ac92` |
| AI Toolkit | `https://github.com/ostris/ai-toolkit.git` at `27a03a91f23eb1b757d5ec2e80ee3e129cfcc350` |
| Storage free | 833,941,807,104 bytes before the runs; 833,456,316,416 bytes at collection |

The base checkpoint provenance is:

- URL: `https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/462165984030d82259a11f4367a4eed129e94a7b/sd_xl_base_1.0.safetensors`
- Bytes: 6,938,078,334
- SHA-256: `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b`

The checkpoint was reused from the shared artifact root. kohya additionally
sourced the OpenCLIP bigG tokenizer at revision
`743c27bd53dfe508a0ade0f50698f99b39d03bec`. AI Toolkit's single-file loader
sourced 17 SDXL pipeline configuration and tokenizer files from the checkpoint
revision above. `comparison-sdxl.json` and the SDXL object in
`environment.json` pin the exact immutable URL prefix, relative path, byte
size, and SHA-256 of every newly sourced file.

The committed generator created 16 fixed 512x512 images and matching captions.
All 32 hashes are recorded under the SDXL object in `environment.json`.

## SDXL loss trajectories

The statistic uses the first and last 40 recorded losses. Negative normalized
change means improvement.

| Run | Recorded losses | First-40 mean | Last-40 mean | Normalized change | Trend check |
| --- | ---: | ---: | ---: | ---: | --- |
| Dinkster AdamW A | 200 | 0.029074 | 0.020006 | -0.3119 | Pass |
| Dinkster AdamW B | 200 | 0.029074 | 0.020006 | -0.3119 | Pass |
| Dinkster FactoredAdamW | 200 | 0.032056 | 0.029787 | -0.0708 | Pass |
| kohya | 200 | 0.022041 | 0.020626 | -0.0642 | Pass |
| AI Toolkit | 199 | 0.026424 | 0.018817 | -0.2879 | Pass; count fails |

The Dinkster-to-kohya normalized-change difference is 0.247692, so it passes
without tolerance headroom beyond 0.002308. AI Toolkit's progress output
reaches 200/200, its adapter metadata reports step 200, and it saves both the
adapter and optimizer. Its `loss/loss` database key contains 199 non-NULL rows
at distinct steps 1 through 199. The statistics use only those observed rows.

## SDXL adapter behavior

The collector maps all three kohya-layout exports into the same SDXL families:
280 cross-attention, 140 feed-forward, 280 self-attention, and 22 spatial
projection modules. It applies each adapter as
`lora_up @ lora_down * (alpha / rank)` before comparing RMS distributions.

| Reference | Family | Median factor | p95 factor | Result |
| --- | --- | ---: | ---: | --- |
| kohya | Cross-attention | 1.0365x | 1.0716x | Pass |
| kohya | Feed-forward | 1.0374x | 1.0453x | Pass |
| kohya | Self-attention | 1.0068x | 1.0293x | Pass |
| kohya | Spatial projection | 1.0081x | 1.0568x | Pass |
| AI Toolkit | Cross-attention | 4.3548x | 1.6369x | Pass |
| AI Toolkit | Feed-forward | 1.6755x | 1.0268x | Pass |
| AI Toolkit | Self-attention | 1.4343x | 1.0070x | Pass |
| AI Toolkit | Spatial projection | 1.1886x | 1.3237x | Pass |

All 722 modules in every export contain an alpha tensor, rank 4, alpha 4, and
effective scale 1. Dinkster metadata identifies the SDXL comparison runtime and
step cursor 200. kohya identifies `sdxl_base_v1-0`, its pinned commit, rank,
alpha, and 200 steps. AI Toolkit identifies `sdxl_1.0` and training step 200.
The audit passes for all five exports.

## SDXL determinism and memory

The same-seed Dinkster AdamW pair has equal arrays of 200 losses and byte-identical
fp32 exports. Both adapters have SHA-256
`6a726680f4209d21dfd889b4bfdf594317cd0b602500346e1f1d29a877867eb3`.

| Run | PyTorch allocated peak | PyTorch reserved peak | `nvidia-smi` peak | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Dinkster AdamW A | 7,307.3 MiB | 7,394.0 MiB | 7,584 MiB | 248.63 s |
| Dinkster AdamW B | 7,307.3 MiB | 7,394.0 MiB | 7,584 MiB | 245.02 s |
| Dinkster FactoredAdamW | 7,277.5 MiB | 7,366.0 MiB | 7,556 MiB | 289.50 s |
| kohya | 7,708.3 MiB | 8,466.0 MiB | 8,656 MiB | 147.88 s |
| AI Toolkit | 7,287.6 MiB | 7,910.0 MiB | 8,100 MiB | 168.89 s |

Dinkster's sampled process peak is below both references. Against AI Toolkit, the
only higher headline is 19.6 MiB of PyTorch allocated memory, while Dinkster uses
516 MiB less process memory and 516 MiB less reserved memory. This is an
effective tie in allocator peak and a clear sampled-memory advantage, so the
D-53 memory-competitiveness requirement passes.

### SDXL Dinkster phase attribution

The AdamW A run attributes the 7,307.3 MiB allocated peak as follows:

| Component or phase | Measurement |
| --- | ---: |
| Encoded dataset after VAE and text-encoder release | 32.0 MiB allocated |
| Frozen SDXL UNet parameters | 4,897.0 MiB |
| LoRA master parameters | 40.6 MiB |
| Allocated after UNet residency | 5,069.9 MiB |
| Checkpoint-boundary explicit inputs | 0.67 MiB |
| Autograd-saved tensors in the first checkpointed step | 0.80 MiB |
| LoRA gradients | 40.6 MiB |
| AdamW state | 81.2 MiB |
| First-step allocated peak | 7,226.1 MiB |
| Allocated after first step | 5,223.6 MiB |
| CUDA context and non-allocator memory | 190 MiB |

Allocator replay records 2,122.3 MiB live at the first-step peak. The largest
sources are adapter-output addition (624.4 MiB), a UNet residual site (400.0
MiB), low-rank branch outputs (313.2 MiB), layer normalization (302.5 MiB),
group normalization (225.2 MiB), attention (103.1 MiB), and native convolution
(62.2 MiB). The frozen 4,897.0 MiB SDXL UNet dominates residency; the remaining
peak is pre-attributed to normal low-rank, residual, normalization, attention,
and convolution activity rather than a full target-shaped adapter weight.

These are short synthetic-data measurements. They establish reproducibility,
training dynamics, export scale, and memory behavior for the pinned 512x512
configuration, not SDXL image quality at its conventional resolution.
