"""Regression suite — fixed prompts × fixed seeds, before/after a training run.

Currently exercises the *forward* path only (no actual sampling loop yet);
this is enough to detect:
- understanding regressions (text-only prefix forward → next-token logits diff)
- gen-side regressions (one-step FM x_pred drift on a fixed (prompt, seed))

Future-work knobs: hook in `t2i_generate` once it's wired through this
pipeline, plus CLIPScore / OCR / ImageReward as the report's §5.4 calls for.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from train_u1.constants import MODEL_ID, MODEL_SHA
from train_u1.data.collators import CollatorConfig, SenseNovaU1Collator, to_device
from train_u1.data.datasets import SyntheticT2ITinyDataset
from train_u1.model.loader import _resolve_local_snapshot, load_neo_chat_4bit
from train_u1.model.wrapper import TrainingWrapper

REGRESSION_PROMPTS: dict[str, str] = {
    "vqa": "Describe the contents of this image in one sentence.",
    "t2i": "A photograph of a red apple on a wooden table, soft natural light.",
    "edit": "Repaint the wall in the picture to a deep matte navy blue.",
    "interleave": "Continue the story by generating a follow-up image of the same scene at sunset.",
}


@dataclass
class RegressionRow:
    name: str
    prompt: str
    image_hw: tuple[int, int]
    seed: int
    text_logits_first_l1: float
    fm_x_pred_l1: float
    fm_x_pred_l2: float


def _text_only_logits(model, tok, prompt: str, device) -> torch.Tensor:
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    text_indexes = torch.stack(
        [
            torch.arange(ids.shape[1], device=device, dtype=torch.long),
            torch.zeros(ids.shape[1], device=device, dtype=torch.long),
            torch.zeros(ids.shape[1], device=device, dtype=torch.long),
        ],
        dim=0,
    )
    from train_u1.model.masking import create_block_causal_mask

    attn_mask = create_block_causal_mask(text_indexes[0]).to(device)
    with torch.no_grad():
        out = model.language_model.model(
            input_ids=ids,
            indexes=text_indexes,
            attention_mask={"full_attention": attn_mask},
            use_cache=False,
        )
    last_h = out.last_hidden_state[:, -1, :]
    if hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head(last_h).float()
    return last_h.float()


def _run_regression(
    wrapper: TrainingWrapper,
    tok,
    *,
    image_hw: tuple[int, int],
    seed: int,
    device: torch.device | str,
) -> list[RegressionRow]:
    rows: list[RegressionRow] = []
    for name, prompt in REGRESSION_PROMPTS.items():
        # text-only logits
        logits = _text_only_logits(wrapper.model, tok, prompt, device)

        # one fm step from a fixed seed image + fixed prompt
        ds = SyntheticT2ITinyDataset(
            n=1, image_hw=image_hw, prompt_template=prompt, base_seed=seed
        )
        cfg = CollatorConfig(image_hw=image_hw, seed=seed)
        collator = SenseNovaU1Collator(tok, cfg=cfg)
        batch = to_device(collator([ds[0]]), device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = wrapper.forward_t2i_step(batch)
        diff = (out.x_pred.float() - batch["x0_patch"].float())
        rows.append(
            RegressionRow(
                name=name,
                prompt=prompt,
                image_hw=image_hw,
                seed=seed,
                text_logits_first_l1=float(logits.abs().mean()),
                fm_x_pred_l1=float(out.x_pred.abs().mean()),
                fm_x_pred_l2=float(diff.pow(2).mean().sqrt()),
            )
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image-h", type=int, default=256)
    ap.add_argument("--image-w", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/regression_run.json")
    args = ap.parse_args()

    print("[reg] loading model 4bit...", flush=True)
    model = load_neo_chat_4bit(cache_dir=args.cache_dir, device_map=args.device)

    from transformers import AutoTokenizer

    local = _resolve_local_snapshot(args.cache_dir, MODEL_ID, MODEL_SHA)
    tok = AutoTokenizer.from_pretrained(
        local or MODEL_ID, revision=None if local else MODEL_SHA, trust_remote_code=True
    )

    wrapper = TrainingWrapper(model)
    rows = _run_regression(
        wrapper, tok, image_hw=(args.image_h, args.image_w), seed=args.seed, device=args.device
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))

    print("\n[reg] regression rows:")
    for r in rows:
        print(
            f"  {r.name:<10s}  text|h|.l1={r.text_logits_first_l1:.4f}  "
            f"fm.x.l1={r.fm_x_pred_l1:.4f}  fm.diff.l2={r.fm_x_pred_l2:.4f}"
        )
    print(f"\n[reg] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
