"""
Fixed data pipeline and evaluation for Parameter Golf autoresearch.
DO NOT MODIFY — this is the ground truth evaluation harness.

Provides:
- Constants (MAX_SEQ_LEN, TIME_BUDGET, MAX_COMPRESSED_BYTES)
- Data loading from pre-tokenized binary shards
- SentencePiece tokenizer wrapper
- BPB evaluation (bits per byte)
- Int8+zlib quantization and compressed size measurement

Usage:
    python prepare.py                  # verify data exists
    python prepare.py --download       # download dataset (sp1024, 1 shard)
"""

from __future__ import annotations

import glob
import math
import os
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 1024          # context length (matches baseline)
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", 3600))  # training time budget in seconds (60 min)
MAX_COMPRESSED_BYTES = 16_000_000  # 16 MB compressed artifact limit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GOLF_DIR = Path(__file__).resolve().parent
DATA_PATH = str(GOLF_DIR / "data" / "datasets" / "fineweb10B_sp1024")
TOKENIZER_PATH = str(GOLF_DIR / "data" / "tokenizers" / "fineweb_1024_bpe.model")
VOCAB_SIZE = 1024

# ---------------------------------------------------------------------------
# Data loading (binary shards)
# ---------------------------------------------------------------------------

def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens.astype(np.int32, copy=False)


class TokenStream:
    def __init__(self, pattern: str, dataset_name: str = ""):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.dataset_name = dataset_name
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def next_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        if self.file_idx == 0:
            self.epoch += 1
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


class TokenLoader:
    def __init__(self, pattern: str, dataset_name: str = ""):
        self.stream = TokenStream(pattern, dataset_name=dataset_name)

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)


def get_train_loader() -> TokenLoader:
    pattern = f"{DATA_PATH}/fineweb_train_*.bin"
    return TokenLoader(pattern, dataset_name="fineweb10B_sp1024")


def load_validation_tokens() -> np.ndarray:
    pattern = f"{DATA_PATH}/fineweb_val_*.bin"
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No validation files found: {pattern}")
    tokens = np.ascontiguousarray(
        np.concatenate([load_data_shard(f) for f in files], axis=0)
    )
    usable = ((tokens.size - 1) // MAX_SEQ_LEN) * MAX_SEQ_LEN
    if usable <= 0:
        raise ValueError(f"Validation split too short for MAX_SEQ_LEN={MAX_SEQ_LEN}")
    return tokens[: usable + 1]


# ---------------------------------------------------------------------------
# Tokenizer + BPB lookup tables
# ---------------------------------------------------------------------------

def get_tokenizer() -> spm.SentencePieceProcessor:
    return spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_bpb(
    model,
    compiled_loss_fn,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    seq_len: int = MAX_SEQ_LEN,
    val_batch_tokens: int = 524_288,
) -> tuple[float, float]:
    """
    Bits per byte (BPB): tokenizer-agnostic evaluation metric.
    Returns (val_loss, val_bpb).
    """
    val_batch_seqs = val_batch_tokens // seq_len
    total_seqs = (val_tokens.size - 1) // seq_len
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0

    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * seq_len
        raw_end = batch_seq_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss_fn(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())

    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def evaluate_bpb_sliding_window(
    model,
    compiled_forward_logits_fn,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    seq_len: int = MAX_SEQ_LEN,
    stride: int = 64,
    batch_size: int = 4,
) -> tuple[float, float]:
    """
    Sliding window BPB evaluation. Every scored token gets at least
    (seq_len - stride) context tokens, dramatically improving BPB.

    compiled_forward_logits_fn: takes input_ids [B, T] and returns logits [B, T, V]
    """
    total_tokens_in_val = val_tokens.size - 1
    # Generate window start positions
    starts = list(range(0, total_tokens_in_val - seq_len + 1, stride))
    if not starts:
        starts = [0]

    total_nats = 0.0
    total_bytes = 0.0
    n_windows = len(starts)

    for batch_start in range(0, n_windows, batch_size):
        batch_positions = starts[batch_start : batch_start + batch_size]
        B = len(batch_positions)

        # Build input batch [B, seq_len]
        x_np = np.stack([val_tokens[s : s + seq_len] for s in batch_positions])
        y_np = np.stack([val_tokens[s + 1 : s + seq_len + 1] for s in batch_positions])

        x = mx.array(x_np, dtype=mx.int32)

        # Forward pass to get logits [B, T, V]
        logits = compiled_forward_logits_fn(x)
        mx.eval(logits)

        # Compute per-token NLL using cross_entropy with reduction='none'
        logits_f32 = np.array(logits.astype(mx.float32), dtype=np.float32)

        # Vectorized log-softmax + gather
        log_sum_exp = np.log(np.sum(np.exp(logits_f32 - logits_f32.max(axis=-1, keepdims=True)), axis=-1, keepdims=True)) + logits_f32.max(axis=-1, keepdims=True)
        # per_token_nll[b, t] = -logits_f32[b, t, y_np[b, t]] + log_sum_exp[b, t, 0]
        gathered = logits_f32[np.arange(B)[:, None], np.arange(seq_len)[None, :], y_np]
        per_token_nll = -gathered + log_sum_exp[:, :, 0]  # [B, T]

        # Score only the last `stride` tokens per window (full context)
        # First window: score all tokens
        for i, s in enumerate(batch_positions):
            score_start = 0 if s == 0 else seq_len - stride
            # Vectorized byte counting for scored range
            scored_x = x_np[i, score_start:]
            scored_y = y_np[i, score_start:]
            scored_nll = per_token_nll[i, score_start:]

            nbytes = base_bytes_lut[scored_y].astype(np.float64)
            nbytes += (has_leading_space_lut[scored_y] & ~is_boundary_token_lut[scored_x]).astype(np.float64)
            mask = nbytes > 0
            total_nats += float(np.sum(scored_nll[mask]))
            total_bytes += float(np.sum(nbytes[mask]))

        if (batch_start // batch_size) % 500 == 0:
            done = min(batch_start + batch_size, n_windows)
            print(f"  sliding_window_eval: {done}/{n_windows} windows", flush=True)

    if total_bytes == 0:
        return 0.0, 0.0
    val_bpb = total_nats / (math.log(2.0) * total_bytes)
    # val_loss approximation (nats per token, using bytes as proxy for token count)
    return float(val_bpb * math.log(2.0)), float(val_bpb)


# ---------------------------------------------------------------------------
# Quantization (int8 + zlib) — fixed compression pipeline
# ---------------------------------------------------------------------------

COMPUTE_DTYPE = mx.bfloat16

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name, arr, passthrough_orig_dtypes):
    if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(
            np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False)
        )
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(arr):
    f32 = _np_float32(arr)
    if f32.ndim == 2:
        clip_abs = (
            np.quantile(np.abs(f32), INT8_CLIP_Q, axis=1)
            if f32.size
            else np.empty((f32.shape[0],), dtype=np.float32)
        )
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(
            scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False)
        )
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), INT8_CLIP_Q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(
        np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127
    ).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


# Tensor names to always keep in fp16 (never int8 quantize).
# tok_emb.weight is tied (input + output) so int8 errors compound twice.
FP16_KEEP_PATTERNS = tuple(
    p for p in os.environ.get("FP16_KEEP_PATTERNS", "tok_emb.weight").split(",") if p
)


def quantize_state_dict_int8(flat_state):
    quantized = {}
    scales = {}
    dtypes = {}
    passthrough = {}
    passthrough_orig_dtypes = {}
    qmeta = {}
    total_bytes = 0
    for name, arr in flat_state.items():
        if not mx.issubdtype(arr.dtype, mx.floating):
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            total_bytes += passthrough[name].nbytes
            continue
        # Force FP16 for specified tensors (e.g., tied embeddings)
        force_fp16 = any(p in name for p in FP16_KEEP_PATTERNS)
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL or force_fp16:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            total_bytes += kept.nbytes
            continue
        q, s = quantize_float_array(arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        total_bytes += q.nbytes + s.nbytes
    obj = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj


def dequantize_state_dict_int8(quant_obj):
    out = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            out_arr = q_np.astype(np.float32) * scale.reshape(
                (q_np.shape[0],) + (1,) * (q_np.ndim - 1)
            )
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def measure_compressed_size(model) -> int:
    """Quantize model to int8 and compress with zlib. Returns compressed byte count."""
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    quant_obj = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    return len(quant_blob)


def evaluate_quantized_bpb(
    model,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    seq_len: int = MAX_SEQ_LEN,
    val_batch_tokens: int = 524_288,
    sliding_window: bool = False,
    sw_stride: int = 64,
    sw_batch_size: int = 4,
) -> tuple[float, float, int]:
    """
    Quantize the model, load it back, and evaluate BPB.
    Returns (val_loss, val_bpb, compressed_bytes).
    """
    flat_state = {k: v for k, v in tree_flatten(model.state)}
    quant_obj = quantize_state_dict_int8(flat_state)
    quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
    quant_blob = zlib.compress(quant_raw, level=9)
    compressed_bytes = len(quant_blob)

    # Load quantized weights back
    quant_flat = dequantize_state_dict_int8(pickle.loads(zlib.decompress(quant_blob)))
    model.update(tree_unflatten(list(quant_flat.items())))

    if sliding_window:
        compiled_forward_logits = mx.compile(
            lambda x: model.forward_logits(x), inputs=model.state, outputs=model.state
        )
        val_loss, val_bpb = evaluate_bpb_sliding_window(
            model, compiled_forward_logits, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            seq_len=seq_len, stride=sw_stride, batch_size=sw_batch_size,
        )
    else:
        compiled_loss = mx.compile(
            lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state
        )
        val_loss, val_bpb = evaluate_bpb(
            model, compiled_loss, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            seq_len=seq_len, val_batch_tokens=val_batch_tokens,
        )
    return val_loss, val_bpb, compressed_bytes


# ---------------------------------------------------------------------------
# Main (verification / download)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify/download data for Parameter Golf")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    args = parser.parse_args()

    if args.download:
        import subprocess
        subprocess.run(
            [sys.executable, str(GOLF_DIR / "data" / "cached_challenge_fineweb.py"),
             "--variant", "sp1024", "--train-shards", "1"],
            check=True,
        )

    # Verify
    train_files = sorted(glob.glob(f"{DATA_PATH}/fineweb_train_*.bin"))
    val_files = sorted(glob.glob(f"{DATA_PATH}/fineweb_val_*.bin"))
    print(f"Data path: {DATA_PATH}")
    print(f"Train shards: {len(train_files)}")
    print(f"Val shards: {len(val_files)}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    if train_files and val_files:
        sp = get_tokenizer()
        print(f"Vocab size: {sp.vocab_size()}")
        print("Ready to train!")
    else:
        print("Missing data. Run: python prepare.py --download")
