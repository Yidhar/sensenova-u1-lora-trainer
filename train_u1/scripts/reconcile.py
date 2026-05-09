"""Print parameter buckets + freeze plan summary.

Two modes:
  --formula : reuse the formula-only breakdown from train_u1.scripts.param_breakdown
  --load-hf : actually load the model with trust_remote_code and print buckets

Without args, prints the static formula table only (no GPU/HF needed).
"""
from __future__ import annotations

import argparse
import sys

from train_u1 import constants as C
from train_u1.model.params import (
    FREEZE_REGEX_BALANCED,
    FREEZE_REGEX_MVP,
    TRAINABLE_REGEX_BALANCED,
    TRAINABLE_REGEX_MVP,
    TRAINABLE_REGEX_MVP_AUX,
    format_param_table,
    set_requires_grad_by_regex,
    summarize_named_parameters,
)


def print_formula_table() -> None:
    print(format_param_table(C.PARAM_COUNTS))
    print()
    for k, v in (
        ("MVP freeze patterns", FREEZE_REGEX_MVP),
        ("MVP trainable patterns", TRAINABLE_REGEX_MVP),
        ("MVP+aux trainable patterns", TRAINABLE_REGEX_MVP_AUX),
        ("Balanced freeze patterns", FREEZE_REGEX_BALANCED),
        ("Balanced trainable patterns", TRAINABLE_REGEX_BALANCED),
    ):
        print(f"\n# {k}")
        for r in v:
            print(f"  {r}")


def load_and_apply(model_id: str, scenario: str, revision: str | None) -> int:
    from transformers import AutoModel  # lazy import

    if revision is None:
        # Default to the project-pinned revision so this script can never
        # silently drift to whatever HF's `main` is.
        revision = C.MODEL_SHA if model_id == C.MODEL_ID else None
    print(
        f"loading {model_id} @ {revision or '<unpinned>'} (trust_remote_code=True)...",
        flush=True,
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype="auto",
    )

    stats = summarize_named_parameters(model)
    print("\n[bucket totals from named_parameters()]")
    print(format_param_table(stats))

    if scenario == "mvp":
        freeze_pats = FREEZE_REGEX_MVP
        train_pats = TRAINABLE_REGEX_MVP
    elif scenario == "mvp_aux":
        freeze_pats = FREEZE_REGEX_MVP
        train_pats = TRAINABLE_REGEX_MVP_AUX
    elif scenario == "balanced":
        freeze_pats = FREEZE_REGEX_BALANCED
        train_pats = TRAINABLE_REGEX_BALANCED
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    rep = set_requires_grad_by_regex(
        model,
        freeze_patterns=freeze_pats,
        trainable_patterns=train_pats,
        default=False,
        strict=False,
    )
    print(f"\n[grad mask] scenario={scenario}")
    print(f"  trainable params : {rep.n_trainable:,}")
    print(f"  frozen params    : {rep.n_frozen:,}")
    print(f"  unmatched        : {len(rep.unmatched)}")
    if rep.unmatched:
        print("  first 5 unmatched:")
        for n in rep.unmatched[:5]:
            print(f"    {n}")
    print("\n  trainable by bucket:")
    for k, v in sorted(rep.bucket_trainable.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<24s} {v:>14,d}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--formula", action="store_true", help="static formula table only")
    ap.add_argument("--load-hf", action="store_true", help="actually load model & report")
    ap.add_argument("--model", default=C.MODEL_ID)
    ap.add_argument(
        "--revision",
        default=None,
        help=f"HF revision to pin; defaults to MODEL_SHA={C.MODEL_SHA[:12]}… for the canonical model id.",
    )
    ap.add_argument(
        "--scenario",
        default="mvp",
        choices=("mvp", "mvp_aux", "balanced"),
    )
    args = ap.parse_args()

    if args.load_hf:
        return load_and_apply(args.model, args.scenario, args.revision)

    print_formula_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
