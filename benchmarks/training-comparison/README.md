# SD1.5 and SDXL LoRA training comparison

This harness compares short SD1.5 and SDXL UNet LoRA runs in Dinkster,
`kohya-ss/sd-scripts`, and `ostris/ai-toolkit`. It generates a fixed synthetic
image/caption dataset, runs each trainer with matched settings, records loss and
CUDA memory, and summarizes the exported adapters without committing model or
dataset artifacts. Dinkster is measured with both bfloat16 and float32 frozen-base
storage for SD1.5 and bfloat16 storage for SDXL, while its LoRA masters,
gradients, optimizer state, and exports remain float32.

The runs use 200 optimizer steps, 512x512 images, batch size 1, no gradient
accumulation, rank/alpha 4, AdamW at 1e-4 with a constant learning rate, bf16
forward computation, and UNet-only training. Reference-specific differences
must be recorded in `REPORT.md`; they must not be hidden by changing the
expectations after a run.

SDXL also runs at 512x512 so all three tools fit on a 16 GB GPU under the same
settings. SDXL is conventionally trained at 1024x1024; this harness measures
short-run correctness and memory, not training quality.

## Pre-declared expectations

These checks were fixed before collecting results:

1. Every recorded loss must be finite and each run must contain 200 losses.
2. The mean of the last 40 losses must be no more than 110% of the mean of the
   first 40 losses. This tolerates short-run timestep noise while rejecting a
   clearly worsening run.
3. Dinkster's normalized loss change, `(last_mean - first_mean) / first_mean`,
   must be within 0.25 of kohya's matched run. AI Toolkit is reported using the
   same statistic but is not a hard gate because its fixed AdamW epsilon and
   data pipeline cannot be made identical.
4. For layer families present in both exports, the median applied-weight delta
   RMS must be within a factor of 5 and the 95th percentile within a factor of
   10. Zero or non-finite deltas fail.
5. Two same-seed Dinkster AdamW runs must have exactly equal per-step losses and
   byte-identical fp32 LoRA exports.
6. Memory is comparative evidence, not a correctness gate. Both PyTorch peak
   allocated bytes and sampled `nvidia-smi` process memory are reported.

## Setup

Clone each reference outside the Dinkster checkout, pin it to the commit recorded
in `REPORT.md`, and create its own virtual environment according to its
installation documentation. Download the exact checkpoint URL and verify the
byte size and SHA-256 listed in the report.

The complete run is:

```bash
.venv-gpu/bin/python benchmarks/training-comparison/run_all.py \
  --model /scratch/v1-5-pruned-emaonly.safetensors \
  --scratch /scratch/dinkster-training-comparison \
  --kohya-repo /scratch/sd-scripts \
  --ai-toolkit-repo /scratch/ai-toolkit
```

The matched SDXL run uses the same pinned references and a separate scratch
root:

```bash
.venv-gpu/bin/python benchmarks/training-comparison/run_all.py \
  --family sdxl \
  --model /scratch/sd_xl_base_1.0.safetensors \
  --scratch /scratch/dinkster-training-comparison-sdxl \
  --kohya-repo /scratch/sd-scripts \
  --ai-toolkit-repo /scratch/ai-toolkit
```

To add or replace only the two Dinkster float32-base controls while preserving the
existing reference and bfloat16 run directories, pass:

```bash
.venv-gpu/bin/python benchmarks/training-comparison/run_all.py \
  --model /scratch/v1-5-pruned-emaonly.safetensors \
  --scratch /scratch/dinkster-training-comparison \
  --kohya-repo /scratch/sd-scripts \
  --ai-toolkit-repo /scratch/ai-toolkit \
  --runs dinkster-fp32-adamw-a dinkster-fp32-adamw-b --overwrite
```

`run_all.py` refuses unpinned reference commits and a model whose SHA-256 does
not match the selected comparison JSON. Dinkster reads all required components
directly from the standard checkpoint. The scratch directory receives the
generated dataset, logs, reference configs, exports, and intermediate results.
Use `collect_results.py` after the runs to produce compact JSON and evaluate
the pre-declared expectations:

```bash
/scratch/sd-scripts/venv/bin/python \
  benchmarks/training-comparison/collect_results.py \
  --scratch /scratch/dinkster-training-comparison \
  --output benchmarks/training-comparison/summary.json \
  --output-dir benchmarks/training-comparison/results \
  --environment-output benchmarks/training-comparison/environment.json
```

After collecting SD1.5, append the SDXL rows with:

```bash
/scratch/sd-scripts/venv/bin/python \
  benchmarks/training-comparison/collect_results.py \
  --family sdxl \
  --scratch /scratch/dinkster-training-comparison-sdxl \
  --output benchmarks/training-comparison/summary.json \
  --output-dir benchmarks/training-comparison/results \
  --environment-output benchmarks/training-comparison/environment.json
```

Collection reads each exported module's actual alpha tensor and rank. It
reports applied and alpha-normalized deltas plus the raw up/down factor
distributions. This distinguishes export scaling differences from training
differences. AI Toolkit always invokes gradient clipping for AdamW, so its
generated config uses `max_grad_norm: 1e9` as the closest non-clipping value;
zero would erase every gradient in that trainer. Collection also audits the
AI Toolkit loss-key row range and NULL count, and converts the scratch run
metadata into a path-sanitized environment record. SDXL collection additionally
audits rank, alpha, effective scale, step metadata, and model-family metadata.
Exact URLs, byte sizes, and SHA-256 values for the reference tools' auxiliary
tokenizer and pipeline files are pinned in `comparison-sdxl.json` and copied to
`environment.json`.

New Dinkster runs also include phase-level `torch.cuda.memory_stats` and allocator
snapshot summaries, known resident tensor sizes, checkpoint-boundary saved
tensors, and the LoRA forward formulation. The report uses those measurements
for attribution. The AdamW A runs also replay allocator events from the first
step and group the live-at-peak allocations by source frame. The snapshot and
allocator-replay data are not correctness gates.
