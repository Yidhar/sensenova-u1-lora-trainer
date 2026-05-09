"""SFT-vs-final per-parameter delta probe (report §5.5 / §13.2).

Walks the safetensors shards directly via the index.json — never loads
the model object — and aggregates per-bucket × per-layer relative deltas:

    delta_rel(name) = ||W_final[name] - W_sft[name]|| / max(||W_sft[name]||, eps)

Output: a JSON heatmap of `(bucket, layer_idx?) -> delta_rel` plus a
flat list of the top-k drifting parameters. Helps answer "where does
final RL/SFT actually move weights?" — which then informs the trainable
subset for low-VRAM PEFT.

Memory: O(largest_tensor) — loads one parameter at a time from each
side, computes the diff, drops it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

from train_u1.constants import MODEL_ID, MODEL_SHA, SFT_MODEL_ID, SFT_MODEL_SHA
from train_u1.model.params import classify_param

LAYER_RE = re.compile(r"language_model\.model\.layers\.(\d+)\.")


def _snapshot_dir(cache_dir: str, model_id: str, sha: str) -> Path:
    safe = model_id.replace("/", "--")
    return Path(cache_dir) / f"models--{safe}" / "snapshots" / sha


def _load_index(snap: Path) -> dict[str, str]:
    with open(snap / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _open_handles(snap: Path, weight_map: dict[str, str]) -> dict[str, "safe_open"]:
    handles: dict[str, "safe_open"] = {}
    files = sorted(set(weight_map.values()))
    for fn in files:
        handles[fn] = safe_open(str(snap / fn), framework="pt", device="cpu").__enter__()
    return handles


def _close_handles(handles: dict[str, "safe_open"]) -> None:
    for h in handles.values():
        try:
            h.__exit__(None, None, None)
        except Exception:
            pass


def diff_walk(
    final_snap: Path, sft_snap: Path, eps: float = 1e-12
) -> dict[str, float]:
    """Per-parameter ||W_f - W_s|| / ||W_s|| via safetensors mmap, no model load."""
    wm_f = _load_index(final_snap)
    wm_s = _load_index(sft_snap)
    keys = sorted(set(wm_f) & set(wm_s))
    only_f = set(wm_f) - set(wm_s)
    only_s = set(wm_s) - set(wm_f)
    if only_f or only_s:
        print(f"[warn] only_in_final={len(only_f)} only_in_sft={len(only_s)}", file=sys.stderr)

    h_f = _open_handles(final_snap, wm_f)
    h_s = _open_handles(sft_snap, wm_s)
    out: dict[str, float] = {}
    try:
        for i, k in enumerate(keys):
            wf = h_f[wm_f[k]].get_tensor(k).to(torch.float32)
            ws = h_s[wm_s[k]].get_tensor(k).to(torch.float32)
            if wf.shape != ws.shape:
                continue
            denom = ws.norm().item()
            out[k] = (wf - ws).norm().item() / max(denom, eps)
            if i % 100 == 0:
                print(f"[walk] {i}/{len(keys)}  last={k}  rel={out[k]:.4f}", flush=True)
    finally:
        _close_handles(h_f)
        _close_handles(h_s)
    return out


def aggregate(per_param: dict[str, float]) -> dict[str, dict[str, float]]:
    sums: dict[tuple[str, str], float] = defaultdict(float)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for name, d in per_param.items():
        bucket = classify_param(name)
        m = LAYER_RE.search(name)
        layer_key = m.group(1) if m else "_global"
        sums[(bucket, layer_key)] += d
        counts[(bucket, layer_key)] += 1
    rolled: dict[str, dict[str, float]] = defaultdict(dict)
    for (bucket, layer), s in sums.items():
        rolled[bucket][layer] = s / counts[(bucket, layer)]
    return rolled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--final-id", default=MODEL_ID)
    ap.add_argument("--final-sha", default=MODEL_SHA)
    ap.add_argument("--sft-id", default=SFT_MODEL_ID)
    ap.add_argument("--sft-sha", default=SFT_MODEL_SHA)
    ap.add_argument("--out", default="artifacts/sft_final_diff.json")
    ap.add_argument("--top-k", type=int, default=30)
    args = ap.parse_args()

    final_snap = _snapshot_dir(args.cache_dir, args.final_id, args.final_sha)
    sft_snap = _snapshot_dir(args.cache_dir, args.sft_id, args.sft_sha)
    if not (final_snap / "model.safetensors.index.json").exists():
        raise SystemExit(f"no index in {final_snap}")
    if not (sft_snap / "model.safetensors.index.json").exists():
        raise SystemExit(f"no index in {sft_snap}")

    print(f"[diff] final  {final_snap}")
    print(f"[diff] sft    {sft_snap}")
    per_param = diff_walk(final_snap, sft_snap)
    rolled = aggregate(per_param)

    top = sorted(per_param.items(), key=lambda kv: -kv[1])[: args.top_k]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "final": {"id": args.final_id, "sha": args.final_sha},
                "sft": {"id": args.sft_id, "sha": args.sft_sha},
                "n_params_compared": len(per_param),
                "per_bucket_layer_avg_rel_delta": rolled,
                "top_k_drifting": [{"name": n, "rel_delta": d} for n, d in top],
            },
            indent=2,
        )
    )
    print(f"\n[diff] wrote {out_path}")

    print("\n[bucket × layer averaged relative delta]")
    for bucket, layers in sorted(rolled.items()):
        if "_global" in layers and len(layers) == 1:
            print(f"  {bucket:<24s} {layers['_global']:.4f}")
        else:
            ranked = sorted(layers.items(), key=lambda kv: -kv[1])[:5]
            avg = sum(layers.values()) / max(len(layers), 1)
            top_str = ", ".join(f"L{lk}={lv:.3f}" for lk, lv in ranked)
            print(f"  {bucket:<24s} avg={avg:.4f}  top: {top_str}")

    print("\n[top-k drifting params]")
    for n, d in top[:10]:
        print(f"  {d:.4f}  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
