#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


PATTERNS = {
    "train_stop": re.compile(
        r"step:(?P<steps>\d+)/\d+\s+val_loss:(?P<pre_val_loss>[0-9.]+)\s+val_bpb:(?P<pre_val_bpb>[0-9.]+)\s+train_time:(?P<train_time_ms>\d+)ms"
    ),
    "compressed_bytes": re.compile(r"Serialized model int8\+zlib:\s+(?P<compressed_bytes>\d+)\s+bytes"),
    "roundtrip_exact": re.compile(
        r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
    ),
    "sliding_window": re.compile(
        r"sliding_window val_bpb:(?P<sw_bpb>[0-9.]+) eval_time:(?P<sw_eval_ms>\d+)ms"
    ),
}


def parse_log(text: str) -> dict[str, object]:
    metrics: dict[str, object] = {
        "steps": None,
        "pre_val_loss": None,
        "pre_val_bpb": None,
        "train_time_ms": None,
        "compressed_bytes": None,
        "compressed_mb": None,
        "val_loss": None,
        "val_bpb": None,
        "sw_bpb": None,
        "sw_eval_ms": None,
        "complete": False,
    }

    train_matches = list(PATTERNS["train_stop"].finditer(text))
    if train_matches:
        match = train_matches[-1]
        metrics["steps"] = int(match.group("steps"))
        metrics["pre_val_loss"] = float(match.group("pre_val_loss"))
        metrics["pre_val_bpb"] = float(match.group("pre_val_bpb"))
        metrics["train_time_ms"] = int(match.group("train_time_ms"))

    compressed_matches = list(PATTERNS["compressed_bytes"].finditer(text))
    if compressed_matches:
        compressed_bytes = int(compressed_matches[-1].group("compressed_bytes"))
        metrics["compressed_bytes"] = compressed_bytes
        metrics["compressed_mb"] = compressed_bytes / (1024 * 1024)

    roundtrip_matches = list(PATTERNS["roundtrip_exact"].finditer(text))
    if roundtrip_matches:
        match = roundtrip_matches[-1]
        metrics["val_loss"] = float(match.group("val_loss"))
        metrics["val_bpb"] = float(match.group("val_bpb"))

    sw_matches = list(PATTERNS["sliding_window"].finditer(text))
    if sw_matches:
        match = sw_matches[-1]
        metrics["sw_bpb"] = float(match.group("sw_bpb"))
        metrics["sw_eval_ms"] = int(match.group("sw_eval_ms"))
        metrics["complete"] = True

    return metrics


def format_tsv_row(metrics: dict[str, object], commit: str, status: str, description: str) -> str:
    val_bpb = float(metrics["val_bpb"]) if metrics["val_bpb"] is not None else 0.0
    sw_bpb = float(metrics["sw_bpb"]) if metrics["sw_bpb"] is not None else 0.0
    compressed_mb = float(metrics["compressed_mb"]) if metrics["compressed_mb"] is not None else 0.0
    return f"{commit}\t{val_bpb:.8f}\t{sw_bpb:.6f}\t{compressed_mb:.1f}\t{status}\t{description}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse train_gpt.py H100 logs.")
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--summary", action="store_true", help="Print a human-readable summary.")
    parser.add_argument("--json", action="store_true", help="Print parsed metrics as JSON.")
    parser.add_argument("--tsv-row", action="store_true", help="Print a results_h100.tsv row.")
    parser.add_argument("--commit", default="UNKNOWN")
    parser.add_argument("--status", default="keep")
    parser.add_argument("--description", default="manual entry")
    args = parser.parse_args()

    text = args.log_path.read_text(encoding="utf-8", errors="replace")
    metrics = parse_log(text)

    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))

    if args.summary:
        for key in (
            "steps",
            "train_time_ms",
            "compressed_bytes",
            "compressed_mb",
            "val_loss",
            "val_bpb",
            "sw_bpb",
            "sw_eval_ms",
            "complete",
        ):
            print(f"{key}={metrics[key]}")

    if args.tsv_row:
        print(format_tsv_row(metrics, args.commit, args.status, args.description))

    if not args.summary and not args.json and not args.tsv_row:
        print(json.dumps(metrics, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
