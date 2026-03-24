# Parameter Golf — Autoresearch Program (H100 CUDA)

This is an autonomous research loop for OpenAI's Parameter Golf challenge on a single H100 GPU.

## Setup

1. **Create a branch**: `git checkout -b autoresearch/h100-<tag>` from current `autoresearch/mar19`.
2. **Read the in-scope files**:
   - `train_gpt.py` — the file you modify. Model architecture, optimizer, training loop, hyperparameters.
   - This file (`program_h100.md`) — your instructions.
3. **Verify data exists**: Check `./data/datasets/fineweb10B_sp1024/` has train shards and val shard. If not: `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10`
4. **Initialize results.tsv**: Create `results.tsv` with header row.
5. **Confirm and go**.

## Environment

- **Platform**: Single H100 SXM 80GB (RunPod)
- **Framework**: PyTorch + CUDA
- **Training command**: `python train_gpt.py > run.log 2>&1`
- **Time budget**: 600 seconds (10 min wallclock, controlled by `MAX_WALLCLOCK_SECONDS` env var)
- **Steps per run**: ~6,000-10,000 steps at ~60-100ms/step
- **Total per experiment**: ~12-15 minutes (training + eval)

## What you CAN modify

Only `train_gpt.py`. Everything is fair game:
- Model architecture (layers, width, heads, skip connections, activations, normalization)
- Optimizer (learning rates, momentum, schedules, weight decay)
- Hyperparameters (batch size, grad accumulation, MLP expansion)
- Training loop structure
- Quantization strategy (within the int8+zlib framework)

## What you CANNOT modify

- Data loading pipeline and tokenizer
- The evaluation harness (eval_val function for standard BPB)
- Do NOT install new packages beyond what's in `requirements.txt`
- Do NOT change the shard format or tokenizer

## Hard Constraints

1. **Compressed artifact < 16,000,000 bytes** (int8 quantized + zlib). The script checks this.
2. **Metric to minimize: `val_bpb`** (bits per byte) — the `sliding_window val_bpb` printed at the end is the primary metric. The standard `val_bpb` is also printed for reference.
3. **Training must complete within MAX_WALLCLOCK_SECONDS** (default 600s).
4. **Must use PyTorch + CUDA**.

## Output Format

The script prints results at the end:

```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:XXXms
final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
sliding_window val_bpb:X.XXXXXX eval_time:XXXms
```

Extract key metrics:
```bash
grep "sliding_window val_bpb\|final_int8_zlib_roundtrip_exact\|Serialized model int8" run.log
```

## Logging Results

Log every experiment to `results.tsv` (tab-separated). Header and 5 columns:

```
commit	val_bpb	sw_bpb	compressed_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb from standard eval (final_int8_zlib_roundtrip)
3. sw_bpb from sliding window eval (the primary metric)
4. compressed size in MB (divide int8+zlib bytes by 1048576)
5. status: `keep`, `discard`, or `crash`
6. short text description

Do NOT commit results.tsv — leave it untracked.

## The Experiment Loop

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Edit `train_gpt.py` with ONE experimental idea.
3. `git commit -am "description of change"`
4. Run: `python train_gpt.py > run.log 2>&1`
5. Read results: `grep "sliding_window val_bpb\|final_int8_zlib_roundtrip_exact\|Serialized model int8" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback.
7. Record results in results.tsv.
8. If sliding_window val_bpb improved AND compressed size < 16,000,000: **keep**.
9. If worse or over size limit: **discard** (`git reset --hard HEAD~1`).

**Timeout**: Each experiment should take ~12-15 minutes. If a run exceeds 20 minutes, kill it and treat as failure.

**Crashes**: Fix typos/easy bugs and re-run. If fundamentally broken, skip and move on.

**NEVER STOP**: Do NOT pause to ask the human. Continue experimenting indefinitely until manually stopped.

## Current Best Configuration

These optimizations have been validated through 78 experiments on M4 Pro:

- **Architecture**: 9L 512D 8H 4KV, 2x MLP, ReLU^2, U-Net skips, resid_mix, attn/mlp scales
- **Value Residual (ResFormer)**: Blends first-block V into later blocks via learned lambda
- **Matrix LR**: 0.015 (Muon) — note: optimal LR depends on step count. At ~10K steps, 0.015-0.04 range.
- **Muon Weight Decay**: 0.06
- **q_gain init**: 2.5
- **FP16 Embedding Export**: tok_emb.weight kept in fp16 during quantization
- **Sliding Window Eval**: stride=64, scores every token with 960+ context

## Research Directions (ordered by expected impact for H100)

### 1. AdamW TTT (Test-Time Training) — HIGH PRIORITY
The current SOTA (1.0891 BPB) uses AdamW TTT. This is a training technique where the model is fine-tuned on the test data at evaluation time. This is the single biggest gap between our approach and the leaderboard.

### 2. EMA (Exponential Moving Average) — HIGH PRIORITY
Maintain an EMA of model weights during training, use EMA weights for evaluation. The SOTA uses EMA. Typical decay: 0.9999 for ~10K steps. **NOTE**: We tried EMA with decay=0.995 on M4 Pro and it was too aggressive. Use 0.9999 or higher.

### 3. 11 Layers — MEDIUM PRIORITY
The SOTA uses 11 layers. On H100 with ~60ms/step, an extra layer adds minimal time. Try 10 or 11 layers.

### 4. Learning Rate Tuning — MEDIUM PRIORITY
Optimal LR depends on step count. At 250 steps we needed 0.24, at 3K steps 0.03, at 12K steps 0.015. For H100 with ~6K-10K steps, sweep 0.02-0.06 range.

### 5. Gated Attention — LOW PRIORITY
Per-head sigmoid gate on attention output. The SOTA uses this. Only +8 params/layer. We tested on M4 Pro and saw negligible improvement, but with more steps it might help.

### 6. Better Quantization — MEDIUM PRIORITY
- Mixed precision: keep critical layers in FP16
- We already keep tok_emb.weight in FP16. Try keeping more.
- Explore LZMA compression instead of zlib

### 7. Larger Batch Size — LOW PRIORITY
H100 can handle much larger batches. The default 524K tokens/step with 8 grad accum steps is tuned for multi-GPU. On single GPU, try different batch/accum ratios.

### 8. Vocabulary Experiments — LOW PRIORITY
Try sp2048 or sp4096 variants. Requires re-downloading data:
`python data/cached_challenge_fineweb.py --variant sp2048 --train-shards 10`

## Rules for the Agent

1. **ONE change per experiment.** Never change multiple things at once.
2. **Always log results** to results.tsv after every experiment.
3. **If an experiment crashes**, try to fix once. If it crashes again, revert and move on.
4. **If BPB improves AND size under 16 MB**, keep it.
5. **If BPB is worse OR size exceeds 16 MB**, revert.
6. **After every 10 experiments**, write a brief summary in results.tsv.
7. **Prioritize by expected impact**: AdamW TTT and EMA first, then architecture, then tuning.
8. **The baseline sliding_window val_bpb should be ~1.35-1.40** based on our M4 Pro experiments scaled to H100 step counts. If you see much worse, something is wrong.

## H100-Specific Notes

- `torch.compile` is used and will take ~30s to compile on first run. The warmup phase handles this.
- Flash Attention is enabled by default via PyTorch's SDPA.
- Single GPU: no DDP needed. Just `python train_gpt.py`.
- For multi-GPU (if available): `torchrun --standalone --nproc_per_node=N train_gpt.py`
- `MAX_WALLCLOCK_SECONDS=600` (10 min) is the default. Adjust if needed.
- The H100 processes ~500K tokens/step in ~60ms. You'll see ~10K steps in 10 min.
