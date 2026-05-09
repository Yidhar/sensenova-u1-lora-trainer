"""Verify the freeze regexes against the *real* parameter names from
the model.safetensors.index.json without loading any weights.

This catches naming surprises (e.g., new module added in a future revision)
before we burn GPU minutes loading the model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.model.params import (
    FREEZE_REGEX_AUX_NO_HEAD,
    FREEZE_REGEX_BALANCED,
    FREEZE_REGEX_GEN_VISION,
    FREEZE_REGEX_MVP,
    TRAINABLE_REGEX_AUX_NO_HEAD,
    TRAINABLE_REGEX_BALANCED,
    TRAINABLE_REGEX_GEN_VISION,
    TRAINABLE_REGEX_MVP,
    TRAINABLE_REGEX_MVP_AUX,
    classify_param,
    format_param_table,
)


def find_index(cache_dir: str, model_id: str = MODEL_ID, sha: str = MODEL_SHA) -> Path:
    safe_id = model_id.replace("/", "--")
    snap = Path(cache_dir) / f"models--{safe_id}" / "snapshots" / sha
    idx = snap / "model.safetensors.index.json"
    if not idx.exists():
        raise FileNotFoundError(f"weight index not found at {idx}")
    return idx


def evaluate_policy(names: list[str], freeze: tuple[str, ...], train: tuple[str, ...]):
    train_re = [re.compile(p) for p in train]
    freeze_re = [re.compile(p) for p in freeze]
    n_train = n_frozen = 0
    bucket_train = defaultdict(int)
    bucket_frozen = defaultdict(int)
    unmatched: list[str] = []
    for name in names:
        bucket = classify_param(name)
        if any(r.search(name) for r in train_re):
            n_train += 1
            bucket_train[bucket] += 1
        elif any(r.search(name) for r in freeze_re):
            n_frozen += 1
            bucket_frozen[bucket] += 1
        else:
            unmatched.append(name)
    return {
        "n_train": n_train,
        "n_frozen": n_frozen,
        "n_unmatched": len(unmatched),
        "bucket_train": dict(bucket_train),
        "bucket_frozen": dict(bucket_frozen),
        "unmatched": unmatched,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument(
        "--scenario",
        default="all",
        choices=("mvp", "mvp_aux", "balanced", "gen_vision", "aux_no_head", "all"),
    )
    args = ap.parse_args()

    idx = find_index(args.cache_dir)
    with open(idx) as f:
        weight_map = json.load(f)["weight_map"]
    names = sorted(weight_map.keys())

    print(f"[index] {idx}")
    print(f"[index] {len(names)} parameter names\n")

    # bucket histogram (param-count by classify)
    bucket_counts = defaultdict(int)
    for n in names:
        bucket_counts[classify_param(n)] += 1
    print("[bucket counts (number of param tensors)]")
    for k, v in sorted(bucket_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<24s} {v:>6d}")
    print()

    all_scenarios = [
        ("mvp", FREEZE_REGEX_MVP, TRAINABLE_REGEX_MVP),
        ("mvp_aux", FREEZE_REGEX_MVP, TRAINABLE_REGEX_MVP_AUX),
        ("balanced", FREEZE_REGEX_BALANCED, TRAINABLE_REGEX_BALANCED),
        ("gen_vision", FREEZE_REGEX_GEN_VISION, TRAINABLE_REGEX_GEN_VISION),
        ("aux_no_head", FREEZE_REGEX_AUX_NO_HEAD, TRAINABLE_REGEX_AUX_NO_HEAD),
    ]
    scenarios = (
        all_scenarios
        if args.scenario == "all"
        else [s for s in all_scenarios if s[0] == args.scenario]
    )

    rc = 0
    for label, freeze, train in scenarios:
        rep = evaluate_policy(names, freeze, train)
        print(f"\n========== scenario: {label} ==========")
        print(f"trainable tensors: {rep['n_train']}")
        print(f"frozen tensors:    {rep['n_frozen']}")
        print(f"unmatched tensors: {rep['n_unmatched']}")
        if rep["unmatched"]:
            print("  first 10 unmatched:")
            for n in rep["unmatched"][:10]:
                print(f"    {n}")
            rc = 1
        print("  trainable buckets:")
        for k, v in sorted(rep["bucket_train"].items(), key=lambda kv: -kv[1]):
            print(f"    {k:<24s} {v}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
