"""Download SenseNova-U1-8B-MoT to a local HF cache (no model load).

Usage:
    HF_HOME=/workspace/senesNovenove/hf_cache \
        .venv/bin/python -m train_u1.scripts.download_model

We use `snapshot_download` so we never construct the model in RAM during
the bytes transfer. Downstream loaders pin the same revision via
`MODEL_SHA`.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from train_u1.constants import MODEL_ID, MODEL_SHA, SFT_MODEL_ID, SFT_MODEL_SHA


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", action="store_true", help="also download the SFT checkpoint")
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    targets = [(MODEL_ID, MODEL_SHA, "final")]
    if args.sft:
        targets.append((SFT_MODEL_ID, SFT_MODEL_SHA, "sft"))

    for repo, rev, label in targets:
        print(f"[{label}] downloading {repo} @ {rev[:12]}...", flush=True)
        t0 = time.time()
        path = snapshot_download(
            repo_id=repo,
            revision=rev,
            cache_dir=args.cache_dir,
            local_dir_use_symlinks=False,
        )
        print(f"[{label}] -> {path} (took {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
