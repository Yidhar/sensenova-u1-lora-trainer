#!/usr/bin/env python3
"""
SenseNova-U1-8B-MoT parameter breakdown helper.

Two modes:
1) --formula-only    : print the exact formula-based breakdown derived from public config.
2) --load-hf         : load the Hugging Face model with trust_remote_code and aggregate named_parameters().

Example:
    python -m train_u1.scripts.param_breakdown --formula-only
    python -m train_u1.scripts.param_breakdown --load-hf --model sensenova/SenseNova-U1-8B-MoT
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class FormulaConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_hidden_layers: int = 42
    vocab_size: int = 151936
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    vision_hidden_size: int = 1024
    patch_size: int = 16
    downsample_ratio: float = 0.5
    add_noise_scale_embedding: bool = True
    fm_head_layers: int = 2

    @property
    def merge_size(self) -> int:
        return int(1 / self.downsample_ratio)

    @property
    def output_dim(self) -> int:
        return 3 * (self.patch_size * self.merge_size) ** 2


def formula_breakdown(cfg: FormulaConfig) -> dict[str, int]:
    h = cfg.hidden_size
    i = cfg.intermediate_size
    layers = cfg.num_hidden_layers
    vocab = cfg.vocab_size
    kv_out = cfg.num_key_value_heads * cfg.head_dim

    # Per-branch (ordinary or _mot_gen) exact total reconciled to HF metadata.
    # This already includes attention, MLP, RMSNorms, q/k norms.
    per_branch_per_layer = 192_946_432
    ordinary_llm_core = per_branch_per_layer * layers
    mot_gen_llm_core = per_branch_per_layer * layers

    token_embeddings = vocab * h
    lm_head = vocab * h  # untied

    vision_one = (
        3 * cfg.vision_hidden_size * cfg.patch_size * cfg.patch_size + cfg.vision_hidden_size
        + cfg.vision_hidden_size * h * cfg.merge_size * cfg.merge_size + h
    )

    # TimestepEmbedder: 256 -> 4096 -> 4096, with bias
    timestep_embedder = 256 * h + h + h * h + h
    noise_scale_embedder = timestep_embedder if cfg.add_noise_scale_embedding else 0

    if cfg.fm_head_layers != 2:
        raise ValueError("This helper currently assumes fm_head_layers=2 for the public revision.")
    fm_head = h * h + h + h * cfg.output_dim + cfg.output_dim

    final_norms = 2 * h

    stats = {
        "ordinary_llm_core": ordinary_llm_core,
        "mot_gen_llm_core": mot_gen_llm_core,
        "token_embeddings": token_embeddings,
        "lm_head": lm_head,
        "vision_understanding": vision_one,
        "vision_mot_gen": vision_one,
        "timestep_embedder": timestep_embedder,
        "noise_scale_embedder": noise_scale_embedder,
        "fm_head": fm_head,
        "final_norms": final_norms,
    }
    return stats


def classify_param(name: str) -> str:
    if name.startswith("vision_model."):
        return "vision_understanding"
    if name.startswith("fm_modules.vision_model_mot_gen."):
        return "vision_mot_gen"
    if name.startswith("fm_modules.timestep_embedder."):
        return "timestep_embedder"
    if name.startswith("fm_modules.noise_scale_embedder."):
        return "noise_scale_embedder"
    if name.startswith("fm_modules.fm_head."):
        return "fm_head"
    if name.startswith("language_model.model.embed_tokens."):
        return "token_embeddings"
    if name.startswith("language_model.lm_head.") or name.startswith("lm_head."):
        return "lm_head"
    if "_mot_gen" in name:
        return "mot_gen_llm_core"
    if name.startswith("language_model.model.layers.") or name.startswith("language_model.model.norm"):
        return "ordinary_llm_core"
    return "other"


def print_stats(stats: dict[str, int]) -> None:
    total = sum(stats.values())
    print(f"{'group':24s} {'params':>18s} {'share':>10s}")
    print("-" * 56)
    for k, v in sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{k:24s} {v:18,d} {v / total:9.2%}")
    print("-" * 56)
    print(f"{'TOTAL':24s} {total:18,d} {1.0:9.2%}")


def load_hf_and_aggregate(
    model_id: str,
    torch_dtype: str = "auto",
    revision: str | None = None,
) -> dict[str, int]:
    """Load + aggregate. Defaults `revision` to the report-pinned SHA so this
    script can never silently drift to whatever the HF `main` branch is.
    """
    from transformers import AutoModel

    # Match `train_u1/constants.py::MODEL_SHA` to keep the pin in one place.
    DEFAULT_PIN = "749fb605230f216d7a7cc0202bfb28369805466b"
    DEFAULT_MODEL = "sensenova/SenseNova-U1-8B-MoT"
    if revision is None and model_id == DEFAULT_MODEL:
        revision = DEFAULT_PIN

    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=getattr(__import__("torch"), torch_dtype) if torch_dtype != "auto" else "auto",
    )

    stats: dict[str, int] = defaultdict(int)
    for name, p in model.named_parameters():
        stats[classify_param(name)] += p.numel()
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formula-only", action="store_true", help="Print formula-based breakdown.")
    parser.add_argument("--load-hf", action="store_true", help="Load the model and aggregate named_parameters().")
    parser.add_argument("--model", default="sensenova/SenseNova-U1-8B-MoT")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument(
        "--revision",
        default=None,
        help="HF revision to pin; defaults to the report-pinned SHA for the canonical model id.",
    )
    args = parser.parse_args()

    if args.formula_only:
        print_stats(formula_breakdown(FormulaConfig()))

    if args.load_hf:
        print_stats(load_hf_and_aggregate(args.model, args.torch_dtype, args.revision))

    if not args.formula_only and not args.load_hf:
        parser.error("Choose at least one of --formula-only or --load-hf.")


if __name__ == "__main__":
    main()
