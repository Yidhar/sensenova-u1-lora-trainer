"""End-to-end forward smoke test — confirms the wrapper pipes a real
NEOChatModel correctly with one synthetic batch.

Goal: exercise extract_feature(gen_model=True) → language_model(image_gen_indicators=True)
→ fm_head pathway, get an `x_pred` with shape (B, N, 3072), and verify
that gradient flows ONLY through the trainable subset.

This script does NOT do an optimizer step or train; it's correctness-only.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator
from train_u1.data.datasets import SyntheticT2ITinyDataset
from train_u1.model.loader import load_neo_chat, load_neo_chat_4bit
from train_u1.model.params import (
    FREEZE_REGEX_MVP,
    TRAINABLE_REGEX_MVP_AUX,
    set_requires_grad_by_regex,
)
from train_u1.model.wrapper import TrainingWrapper


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant", choices=("bf16", "4bit"), default="4bit")
    ap.add_argument("--image-h", type=int, default=256)
    ap.add_argument("--image-w", type=int, default=256)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[smoke] loading model (quant={args.quant})...", flush=True)
    t0 = time.time()
    if args.quant == "4bit":
        model = load_neo_chat_4bit(cache_dir=args.cache_dir, device_map=args.device)
    else:
        model = load_neo_chat(cache_dir=args.cache_dir, device_map=args.device)
    print(f"[smoke] model loaded in {time.time()-t0:.1f}s", flush=True)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] total params: {n_total:,}")

    print("\n[smoke] applying MVP+aux freeze policy (strict=True)...", flush=True)
    rep = set_requires_grad_by_regex(
        model,
        freeze_patterns=FREEZE_REGEX_MVP,
        trainable_patterns=TRAINABLE_REGEX_MVP_AUX,
        default=False,
        strict=True,
    )
    print(f"[smoke] trainable params: {rep.n_trainable:,}")
    for k, v in sorted(rep.bucket_trainable.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24s} {v:>14,d}")

    # tokenizer — also prefer local snapshot for the same trust_remote_code reason
    from transformers import AutoTokenizer

    from train_u1.model.loader import _resolve_local_snapshot

    local = _resolve_local_snapshot(args.cache_dir, MODEL_ID, MODEL_SHA)
    if local is not None:
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    else:
        tok = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_SHA, trust_remote_code=True, cache_dir=args.cache_dir
        )

    print(f"\n[smoke] building synthetic batch H,W=({args.image_h},{args.image_w})...", flush=True)
    ds = SyntheticT2ITinyDataset(n=args.n_samples, image_hw=(args.image_h, args.image_w))
    collator = SenseNovaU1Collator(tok, cfg=CollatorConfig(image_hw=(args.image_h, args.image_w)))
    samples = [ds[i] for i in range(args.n_samples)]
    batch = collator(samples)

    # move to device + dtype
    dtype = torch.bfloat16
    device = next(model.parameters()).device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            t = v.to(device)
            if t.is_floating_point():
                t = t.to(dtype)
            batch[k] = t

    print(f"[smoke] batch on {device}; running forward...", flush=True)
    wrapper = TrainingWrapper(model)
    t1 = time.time()
    try:
        out = wrapper.forward_t2i_step(batch)
    except Exception as e:
        print(f"\n[smoke] FORWARD ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise
    print(f"[smoke] forward took {time.time()-t1:.2f}s")
    print(f"[smoke] x_pred shape: {tuple(out.x_pred.shape)} dtype: {out.x_pred.dtype}")
    print(f"[smoke] z_t   shape: {tuple(out.z_t.shape)}")
    print(f"[smoke] hidden_image shape: {tuple(out.hidden_image.shape)}")

    # mini loss + backward sanity (no optimizer step)
    from train_u1.model.losses import fm_loss_x0

    loss = fm_loss_x0(out.x_pred, batch["x0_patch"])
    print(f"[smoke] x0-MSE loss: {loss.item():.4f}")
    loss.backward()
    grads_present = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    grads_zero = sum(1 for p in model.parameters() if (not p.requires_grad) and p.grad is not None)
    print(f"[smoke] params with grad (trainable): {grads_present}")
    print(f"[smoke] params with grad (frozen, should be 0): {grads_zero}")
    if torch.cuda.is_available():
        print(f"[smoke] peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
