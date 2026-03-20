#!/usr/bin/env python3
"""
Parameter Golf training script (MLX, Apple Silicon).
This is the file the autoresearch agent modifies.

Usage: python train.py
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable

import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from prepare import (
    MAX_SEQ_LEN, TIME_BUDGET, MAX_COMPRESSED_BYTES, VOCAB_SIZE,
    COMPUTE_DTYPE,
    get_train_loader, load_validation_tokens,
    get_tokenizer, build_sentencepiece_luts,
    evaluate_bpb, evaluate_bpb_sliding_window,
    measure_compressed_size, evaluate_quantized_bpb,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

# Model architecture
NUM_LAYERS = 9
MODEL_DIM = 512
NUM_HEADS = 8
NUM_KV_HEADS = 4
MLP_MULT = 2
LOGIT_SOFTCAP = 30.0
ROPE_BASE = 10000.0
QK_GAIN_INIT = 2.5
TIED_EMBED_INIT_STD = 0.005

# Training — tuned for M4 Pro (48GB). Each step processes TRAIN_BATCH_TOKENS tokens.
# For fast local iteration: smaller batch = faster steps but noisier gradients.
TRAIN_BATCH_TOKENS = int(os.environ.get("TRAIN_BATCH_TOKENS", 32_768))
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 4))
TRAIN_SEQ_LEN = MAX_SEQ_LEN
MLX_MAX_MICROBATCH_TOKENS = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", 5))
WARMDOWN_ITERS = 1200

# Optimizer
BETA1 = 0.9
BETA2 = 0.95
ADAM_EPS = 1e-8
TIED_EMBED_LR = 0.05
MATRIX_LR = 0.24
SCALAR_LR = 0.04
MUON_MOMENTUM = 0.95
MUON_BACKEND_STEPS = 5
MUON_MOMENTUM_WARMUP_START = 0.85
MUON_MOMENTUM_WARMUP_STEPS = 500
GRAD_CLIP_NORM = 0.0

# Eval — larger batch = faster eval. Also control how much of val set to use.
VAL_BATCH_TOKENS = int(os.environ.get("VAL_BATCH_TOKENS", 524_288))
# Max val tokens to evaluate (0 = full val set). Set lower for faster iteration.
MAX_VAL_TOKENS = int(os.environ.get("MAX_VAL_TOKENS", 2_000_000))
# Sliding window eval: stride controls overlap (smaller = better BPB, slower eval)
USE_SLIDING_WINDOW = int(os.environ.get("USE_SLIDING_WINDOW", 1))
SW_STRIDE = int(os.environ.get("SW_STRIDE", 512))
SW_BATCH_SIZE = int(os.environ.get("SW_BATCH_SIZE", 8))

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

MICROBATCH_TOKENS = TRAIN_BATCH_TOKENS // GRAD_ACCUM_STEPS

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)

# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class RMSNormNoWeight(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5

    def __call__(self, x: mx.array) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(rms_norm(q).astype(COMPUTE_DTYPE))
        k = self.rope(rms_norm(k).astype(COMPUTE_DTYPE))
        q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int):
        super().__init__()
        hidden = dim * mlp_mult
        self.fc = CastedLinear(dim, hidden)
        self.proj = CastedLinear(hidden, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(nn.relu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, mlp_mult: int, rope_base: float, qk_gain_init: float):
        super().__init__()
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, dim: int, num_heads: int, num_kv_heads: int,
                 mlp_mult: int, logit_softcap: float, rope_base: float, tied_embed_init_std: float,
                 qk_gain_init: float):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_softcap = logit_softcap

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.blocks = [
            Block(dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        for i in range(self.num_encoder_layers + self.num_decoder_layers):
            x = self.blocks[i](x)
        return self.final_norm(x)

    def forward_logits(self, input_ids: mx.array) -> mx.array:
        """Returns logits [B, T, V] for sliding window evaluation."""
        x = self(input_ids)  # [B, T, D]
        logits = x @ self.tok_emb.weight.astype(x.dtype).T  # [B, T, V]
        return self.softcap(logits)

    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        logits_proj = x @ self.tok_emb.weight.astype(x.dtype).T
        logits = self.softcap(logits_proj)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")


# ---------------------------------------------------------------------------
# Optimizer (Muon + Adam split)
# ---------------------------------------------------------------------------

class Muon:
    def __init__(self, keys: list[str], params: dict[str, mx.array]):
        self.keys = keys
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array],
             step: int, lr_mul: float) -> dict[str, mx.array]:
        if MUON_MOMENTUM_WARMUP_STEPS:
            t = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1.0)
            momentum = (1.0 - t) * MUON_MOMENTUM_WARMUP_START + t * MUON_MOMENTUM
        else:
            momentum = MUON_MOMENTUM
        lr = MATRIX_LR * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, MUON_BACKEND_STEPS)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    def __init__(self, model: GPT):
        params = dict(tree_flatten(model.parameters()))
        self.embed_key = "tok_emb.weight"
        self.matrix_keys = [
            k for k, p in params.items()
            if k.startswith("blocks.") and p.ndim == 2
            and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k == "skip_weights" or (
                k.startswith("blocks.") and (
                    p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
                )
            )
        ]
        self.muon = Muon(self.matrix_keys, params)
        self.adam_embed = optim.Adam(
            learning_rate=TIED_EMBED_LR, betas=[BETA1, BETA2], eps=ADAM_EPS, bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=SCALAR_LR, betas=[BETA1, BETA2], eps=ADAM_EPS, bias_correction=True,
        )

    def step(self, model: GPT, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)
        updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))
        self.adam_embed.learning_rate = TIED_EMBED_LR * lr_mul
        updated.update(
            self.adam_embed.apply_gradients(
                {self.embed_key: grads[self.embed_key]},
                {self.embed_key: params[self.embed_key]},
            )
        )
        self.adam_scalar.learning_rate = SCALAR_LR * lr_mul
        scalar_grads = {k: grads[k] for k in self.scalar_keys}
        scalar_params = {k: params[k] for k in self.scalar_keys}
        updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))
        model.update(tree_unflatten(list(updated.items())))


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def lr_mul(step: int, elapsed_ms: float) -> float:
    if WARMDOWN_ITERS <= 0:
        return 1.0
    if TIME_BUDGET <= 0:
        return 1.0
    step_ms = elapsed_ms / max(step, 1)
    warmdown_ms = WARMDOWN_ITERS * step_ms
    remaining_ms = max(1000.0 * TIME_BUDGET - elapsed_ms, 0.0)
    return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = sum(
        float(np.sum(np.square(np.array(g.astype(mx.float32), dtype=np.float32, copy=False)), dtype=np.float64))
        for g in flat.values()
    )
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def loss_and_grad_chunked(train_loader, compiled_loss_and_grad):
    chunk_sizes = token_chunks(MICROBATCH_TOKENS, TRAIN_SEQ_LEN, MLX_MAX_MICROBATCH_TOKENS)
    total_tokens = float(sum(chunk_sizes))
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum = None
    for chunk_tokens in chunk_sizes:
        x, y = train_loader.next_batch(chunk_tokens, TRAIN_SEQ_LEN)
        loss, grads = compiled_loss_and_grad(x, y)
        scale = float(y.size) / total_tokens
        loss_value = loss_value + loss.astype(mx.float32) * scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, scale)
    return loss_value, tree_unflatten(list(grad_accum.items()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Parameter Golf MLX Training")
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Max compressed bytes: {MAX_COMPRESSED_BYTES}")

    mx.random.seed(1337)

    # Setup tokenizer + BPB lookup tables
    sp = get_tokenizer()
    if int(sp.vocab_size()) != VOCAB_SIZE:
        raise ValueError(f"VOCAB_SIZE={VOCAB_SIZE} != tokenizer vocab_size={sp.vocab_size()}")
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, VOCAB_SIZE)

    # Load validation tokens
    val_tokens = load_validation_tokens()

    # Data loader
    train_loader = get_train_loader()

    # Model
    model = GPT(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, dim=MODEL_DIM,
        num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, mlp_mult=MLP_MULT,
        logit_softcap=LOGIT_SOFTCAP, rope_base=ROPE_BASE,
        tied_embed_init_std=TIED_EMBED_INIT_STD, qk_gain_init=QK_GAIN_INIT,
    )
    opt = SplitOptimizers(model)

    n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
    print(f"Parameters: {n_params:,}")
    print(f"Architecture: {NUM_LAYERS}L {MODEL_DIM}D {NUM_HEADS}H {NUM_KV_HEADS}KV {MLP_MULT}x MLP")

    # Compiled functions
    compiled_loss = mx.compile(
        lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state
    )
    compiled_loss_and_grad = mx.compile(
        nn.value_and_grad(model, lambda x, y: model.loss(x, y)),
        inputs=model.state, outputs=model.state,
    )

    # Warmup (compile + allocate) — use small batches to avoid huge lazy graphs
    if WARMUP_STEPS > 0:
        for warmup_step in range(WARMUP_STEPS):
            x, y = train_loader.next_batch(MLX_MAX_MICROBATCH_TOKENS, TRAIN_SEQ_LEN)
            warmup_loss, grads = compiled_loss_and_grad(x, y)
            mx.eval(warmup_loss)
            mx.synchronize()
            if warmup_step + 1 == WARMUP_STEPS or (warmup_step + 1) % 10 == 0:
                print(f"warmup: {warmup_step + 1}/{WARMUP_STEPS}")

        # Prime eval graph
        val_batch_seqs = min(
            VAL_BATCH_TOKENS // TRAIN_SEQ_LEN,
            (val_tokens.size - 1) // TRAIN_SEQ_LEN,
        )
        warm_chunk = val_tokens[: val_batch_seqs * TRAIN_SEQ_LEN + 1]
        x_val = mx.array(warm_chunk[:-1].reshape(-1, TRAIN_SEQ_LEN), dtype=mx.int32)
        y_val = mx.array(warm_chunk[1:].reshape(-1, TRAIN_SEQ_LEN), dtype=mx.int32)
        warm_val_loss = compiled_loss(x_val, y_val)
        mx.eval(warm_val_loss)
        mx.synchronize()

        # Reset data loader
        train_loader = get_train_loader()

    # Training loop
    train_time_ms = 0.0
    max_wallclock_ms = 1000.0 * TIME_BUDGET
    stop_after_step = None
    t0 = time.perf_counter()
    step = 0
    total_iterations = 100_000  # effectively infinite; wallclock is the real limit

    while True:
        last_step = (stop_after_step is not None and step >= stop_after_step)

        if last_step:
            train_time_ms += 1000.0 * (time.perf_counter() - t0)
            break

        lr_m = lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
        step_t0 = time.perf_counter()

        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / GRAD_ACCUM_STEPS
        for ga_step in range(GRAD_ACCUM_STEPS):
            loss, grads = loss_and_grad_chunked(train_loader, compiled_loss_and_grad)
            accum = accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            # Evaluate periodically to avoid huge lazy graphs
            if (ga_step + 1) % 2 == 0 or ga_step + 1 == GRAD_ACCUM_STEPS:
                mx.eval(train_loss, accum)

        grads = tree_unflatten(list(accum.items()))
        grads = clip_grad_tree(grads, GRAD_CLIP_NORM)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_m)
        mx.synchronize()

        step_ms = 1000.0 * (time.perf_counter() - step_t0)
        approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
        tok_s = TRAIN_BATCH_TOKENS / (step_ms / 1000.0)
        step += 1

        if step <= 10 or step % 200 == 0 or stop_after_step is not None:
            print(
                f"step:{step} train_loss:{train_loss_value:.4f} "
                f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms "
                f"tok/s:{tok_s:.0f}"
            )

        if stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
            stop_after_step = step
            print(f"wallclock cap reached at step {step}, will stop after this step")

    total_training_seconds = train_time_ms / 1000.0
    print(f"\nTraining done: {step} steps in {total_training_seconds:.1f}s")

    # Trim val tokens for faster eval if MAX_VAL_TOKENS is set
    eval_val_tokens = val_tokens
    if MAX_VAL_TOKENS > 0 and val_tokens.size > MAX_VAL_TOKENS:
        usable = ((MAX_VAL_TOKENS - 1) // TRAIN_SEQ_LEN) * TRAIN_SEQ_LEN
        eval_val_tokens = val_tokens[: usable + 1]
        print(f"Using {eval_val_tokens.size} of {val_tokens.size} val tokens for eval")

    # Quantized roundtrip evaluation (the metric that matters for submission)
    eval_mode = "sliding_window" if USE_SLIDING_WINDOW else "standard"
    print(f"Evaluating int8+zlib roundtrip ({eval_mode})...")
    q_val_loss, q_val_bpb, q_compressed_bytes = evaluate_quantized_bpb(
        model, eval_val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        seq_len=TRAIN_SEQ_LEN, val_batch_tokens=VAL_BATCH_TOKENS,
        sliding_window=bool(USE_SLIDING_WINDOW),
        sw_stride=SW_STRIDE, sw_batch_size=SW_BATCH_SIZE,
    )

    # Final summary (autoresearch-compatible output format)
    print("---")
    print(f"val_bpb:          {q_val_bpb:.6f}")
    print(f"val_loss:         {q_val_loss:.6f}")
    print(f"compressed_bytes: {q_compressed_bytes}")
    print(f"training_seconds: {total_training_seconds:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params:       {n_params}")
    print(f"architecture:     {NUM_LAYERS}L {MODEL_DIM}D {NUM_HEADS}H {NUM_KV_HEADS}KV")

    # Hard constraint check
    if q_compressed_bytes > MAX_COMPRESSED_BYTES:
        print(f"FAIL: compressed size {q_compressed_bytes} exceeds limit {MAX_COMPRESSED_BYTES}")
    else:
        print(f"OK: compressed size {q_compressed_bytes} under limit {MAX_COMPRESSED_BYTES}")


if __name__ == "__main__":
    main()
