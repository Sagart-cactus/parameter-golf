# Parameter Golf — Autoresearch Program

This is an autonomous research loop for OpenAI's Parameter Golf challenge. The goal: train the best language model that fits in a **16 MB compressed artifact** (int8+zlib), evaluated by **bits-per-byte (BPB)** on the FineWeb validation set — lower is better.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git init && git add -A && git commit -m "initial" && git checkout -b autoresearch/<tag>` (if not already a git repo, otherwise just branch).
3. **Read the in-scope files**:
   - `prepare.py` — fixed data pipeline, evaluation, quantization. **DO NOT MODIFY.**
   - `train.py` — the file you modify. Model architecture, optimizer, training loop, hyperparameters.
   - This file (`program.md`) — your instructions.
4. **Verify data exists**: Run `python prepare.py` and confirm it says "Ready to train!"
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good, then start experimenting.

## Environment

- **Platform**: Apple M4 Pro, 48 GB unified RAM
- **Framework**: MLX (Apple Silicon ML framework) — NOT PyTorch/CUDA
- **Python venv**: `source .venv/bin/activate` before running anything
- **Training command**: `python -u train.py > run.log 2>&1`
- **Time budget**: 5 minutes training (300s wallclock), controlled by `TIME_BUDGET` env var
- **Eval**: ~3-5 min for int8+zlib quantized roundtrip BPB on 10M val token subset
- **Total per experiment**: ~8-10 minutes

## What you CAN modify

Only `train.py`. Everything is fair game:
- Model architecture (layers, width, heads, skip connections, activation functions, normalization)
- Optimizer (learning rates, momentum, schedules, warmup/warmdown)
- Hyperparameters (batch size, grad accumulation, MLP expansion ratio)
- Training loop structure

## What you CANNOT modify

- `prepare.py` — fixed evaluation harness, data loading, quantization pipeline
- Do NOT install new packages. Only use what's in `.venv/`.
- Do NOT modify the evaluation or quantization logic.

## Hard Constraints (must not violate)

1. **Compressed artifact < 16,000,000 bytes** (int8 quantized + zlib). The script checks this.
2. **Metric to minimize: `val_bpb`** (bits per byte) — printed after `---` in the output.
3. **Training must complete within the time budget** (default 300s).
4. **Must use MLX framework** (no PyTorch/CUDA).

## Output Format

The script prints a summary block at the end:

```
---
val_bpb:          2.194762
val_loss:         3.662264
compressed_bytes: 7751179
training_seconds: 301.1
num_steps:        244
num_params:       17059912
architecture:     9L 512D 8H 4KV
OK: compressed size 7751179 under limit 16000000
```

Extract the key metric: `grep "^val_bpb:" run.log`

If you see `FAIL:` at the end, the compressed size exceeded 16 MB — that experiment must be discarded.

## Logging Results

Log every experiment to `results.tsv` (tab-separated). Header and 5 columns:

```
commit	val_bpb	compressed_mb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 2.1948) — use 0.0000 for crashes
3. compressed size in MB, round to .1f (e.g. 7.4 — divide compressed_bytes by 1048576) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Do NOT commit results.tsv — leave it untracked.

## The Experiment Loop

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Edit `train.py` with ONE experimental idea.
3. `git commit -am "description of change"`
4. Run: `source .venv/bin/activate && python -u train.py > run.log 2>&1`
5. Read results: `grep "^val_bpb:\|^compressed_bytes:\|^FAIL\|^OK" run.log`
6. If grep is empty, the run crashed. Run `tail -n 50 run.log` to read the traceback.
7. Record results in results.tsv.
8. If val_bpb improved AND compressed_bytes < 16,000,000: **keep** (advance the branch).
9. If val_bpb is equal or worse, or size exceeds limit: **discard** (`git reset --hard HEAD~1`).

**Timeout**: Each experiment takes ~8-10 minutes total. If a run exceeds 15 minutes, kill it (`kill $(pgrep -f train.py)`) and treat it as a failure.

**Crashes**: If it's a typo or easy fix, fix and re-run. If fundamentally broken, skip it and move on.

**NEVER STOP**: Do NOT pause to ask the human. The human may be away. Continue experimenting indefinitely until manually stopped. If you run out of ideas, think harder — re-read the code, try combining near-misses, try more radical changes.

## Research Directions (ordered by expected impact)

### 1. Depth Recurrence / Layer Looping (HIGH PRIORITY)

Instead of N unique layers, use K unique layers looped N/K times. This drastically reduces parameter count while keeping compute constant. The freed parameter budget can increase model width.

**Try**:
- 3 unique layers looped 3× (= 9 effective layers) with increased width
- 4 unique layers looped 2× (= 8 effective layers) with increased width
- 5 unique layers looped 2× (= 10 effective layers)

**Implementation**: In the GPT `__call__`, loop over `self.blocks` multiple times instead of once. Need separate skip connection logic for looped architecture. Try both with and without U-Net skips.

**Why this is promising**: The baseline has 9 unique layers at 512 width = ~17M params compressing to 7.7 MB. With 3 unique layers, you could go to ~768 width within the same compressed budget, getting much more compute per parameter.

### 2. Width/Depth/Head Search (HIGH PRIORITY)

The baseline uses 9 layers, 512 width, 8 heads, 4 KV heads, 2× MLP. Systematically try:

- **Fewer layers, wider**: 6×640, 5×720, 7×576
- **More KV head sharing**: 8 heads with 2 KV heads, or 1 KV head (MQA)
- **Different MLP ratios**: 3× or 4× instead of 2× (more MLP capacity relative to attention)
- **Different head dimensions**: Currently 512/8=64. Try 32 or 128 head dim.

**Key insight**: The 16 MB compressed size limit means there's an optimal width-depth tradeoff. Wider models compress slightly worse (more unique weights) but have more capacity per layer. Find the sweet spot.

### 3. Vocabulary Size Tuning (MEDIUM PRIORITY)

Baseline uses 1024 tokens. Larger vocab = more embedding params but fewer tokens per document.

**IMPORTANT**: Different vocab sizes require re-downloading tokenized data. To change vocab:
1. Edit the `VOCAB_SIZE` constant in train.py
2. Update `DATA_PATH` and `TOKENIZER_PATH` in prepare.py constants to match
3. Run: `python data/cached_challenge_fineweb.py --variant sp<SIZE> --train-shards 1`

**Try**: 512, 2048, 4096. Since BPB is tokenizer-agnostic, this is a pure efficiency tradeoff.

**CAUTION**: Changing vocab size is expensive (data re-download). Only try this after exhausting architecture changes with the current vocab.

### 4. Better Quantization Awareness (MEDIUM PRIORITY)

The baseline does uniform int8. The model doesn't know it will be quantized during training. Try:
- **Quantization-aware training**: Add noise during training that simulates int8 quantization
- **Straight-through estimator**: Quantize activations/weights in forward, use full precision in backward
- **Mixed precision layers**: Some layers may tolerate more aggressive compression

### 5. Skip Connection Variants (MEDIUM PRIORITY)

The baseline uses U-Net skip connections (encoder half stores, decoder half consumes). Try:
- **No skip connections** (ablation — is U-Net helping at this scale?)
- **Dense connections** (every layer connects to every later layer)
- **Different skip weighting schemes**
- **Residual mixing** (the `resid_mix` parameter blends current hidden state with initial embedding)

### 6. Optimizer Tuning (LOW PRIORITY)

Muon + Adam optimizer with current settings was tuned for CUDA H100s. MLX on M4 Pro may have different optimal settings. Try:
- **Learning rate sweeps**: Matrix LR (0.02-0.08), Embed LR (0.02-0.1), Scalar LR (0.02-0.08)
- **Momentum schedules**: MUON_MOMENTUM (0.9-0.98), warmup steps
- **Gradient clipping**: Currently disabled (0.0). Try 1.0 or 0.5.
- **Different warmdown schedules**: More or less warmdown iterations

### 7. Activation Functions (LOW PRIORITY)

Baseline uses relu^2 (squared ReLU). Try:
- **SwiGLU / GeGLU**: Gated activations (need to adjust MLP structure)
- **GELU**: Standard transformer activation
- **Just ReLU**: Ablation — is the squaring helping?

### 8. Training Dynamics (LOW PRIORITY)

- **Sequence length**: Currently 1024. Try 512 (faster steps, less context) or 2048 (more context, slower steps). NOTE: MAX_SEQ_LEN in prepare.py is fixed at 1024, but you could try shorter sequences.
- **Batch size**: Currently 32K tokens. Larger batches (64K, 128K) give fewer but higher-quality steps. Smaller batches (16K, 8K) give more steps with more noise.

## Rules for the Agent

1. **ONE change per experiment.** Never change multiple things at once. This is critical for understanding what works.
2. **Always log results** to results.tsv after every experiment.
3. **If an experiment crashes**, try to fix it once. If it crashes again, revert and move on.
4. **If BPB improves AND size is under 16 MB**, keep it.
5. **If BPB is worse OR size exceeds 16 MB**, revert.
6. **After every 10 experiments**, write a brief summary comment in results.tsv of what worked and what didn't.
7. **Prioritize by expected impact**: depth recurrence first, then width search, then other directions.
8. **Be bold with architecture changes, conservative with hyperparameters.** A 10% width increase is more likely to help than a 10% LR change.
9. **Watch the compressed size.** If you're near 16 MB, you need to find ways to reduce parameters without hurting BPB. If you're well under (like the baseline at 7.7 MB), you have room to add parameters.
10. **The baseline val_bpb is ~2.19** (5 min training, 10M val subset, 32K batch). Your first run should establish this exact number as the baseline.

## MLX-Specific Notes

- MLX uses lazy evaluation. Large computation graphs without `mx.eval()` can cause silent crashes (SIGKILL). The training loop already handles this with periodic `mx.eval()` calls.
- `mx.compile()` is used for the loss function. If you change the model signature, you may need to update the compiled functions.
- `mx.fast.scaled_dot_product_attention` is the optimized attention kernel. Use it.
- Memory is unified (CPU+GPU share 48 GB). No separate VRAM limit.
- Compilation happens on first call. The warmup phase handles this.
