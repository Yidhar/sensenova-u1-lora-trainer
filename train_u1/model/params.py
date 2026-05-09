"""Parameter classification, freeze regexes, and trainable-param utilities.

Extends `train_u1.scripts.param_breakdown` with the operational
helpers we actually need at training time:

- `classify_param(name)` — same buckets as the report's Table §2.2.
- `FREEZE_REGEX_*` — reusable patterns for the three recommended scenarios.
- `set_requires_grad_by_regex(...)` — apply freeze/trainable masks safely.
- `summarize_trainable(model)` — print trainable param count + per-bucket.

公开证据显示 + 公式复核: bucket assignment matches the param reconcile table.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch.nn as nn


# --------------------------------------------------------------------------- #
# 1) Parameter bucket classification                                          #
# --------------------------------------------------------------------------- #

BUCKETS = (
    "ordinary_llm_core",
    "mot_gen_llm_core",
    "token_embeddings",
    "lm_head",
    "vision_understanding",
    "vision_mot_gen",
    "timestep_embedder",
    "noise_scale_embedder",
    "fm_head",
    "final_norms",
    "other",
)


def classify_param(name: str) -> str:
    """Map a fully-qualified parameter name to one of the report's buckets.

    Order matters — `_mot_gen` test must run before the ordinary LLM-core test
    so that `language_model.model.layers.X.<...>_mot_gen` lands in mot_gen,
    not ordinary.
    """
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
    if name.endswith("language_model.model.norm.weight") or name.endswith(
        "language_model.model.norm_mot_gen.weight"
    ):
        return "final_norms"
    if "_mot_gen" in name:
        return "mot_gen_llm_core"
    if name.startswith("language_model.model.layers.") or name.startswith(
        "language_model.model.norm"
    ):
        return "ordinary_llm_core"
    return "other"


# --------------------------------------------------------------------------- #
# 2) Reusable freeze / trainable regex sets                                   #
# --------------------------------------------------------------------------- #

# Scenario A: 24GB MVP — freeze everything except fm_head (and optionally
# the gen-side norms / timestep / noise-scale embedders if `--with_gen_aux`).
FREEZE_REGEX_MVP: tuple[str, ...] = (
    r"^vision_model\.",
    r"^fm_modules\.vision_model_mot_gen\.",
    r"^language_model\.model\.embed_tokens\b",
    r"^language_model\.lm_head\b",
    # final norms — match `language_model.model.norm.<param>` and
    # `language_model.model.norm_mot_gen.<param>` (e.g. `.weight`).
    r"^language_model\.model\.norm(?:_mot_gen)?\.",
    # ordinary path: any layer parameter without _mot_gen in the name.
    r"^language_model\.model\.layers\.\d+\.(?!.*_mot_gen).*$",
    # full _mot_gen LLM core stays frozen in MVP scenario.
    r"^language_model\.model\.layers\.\d+\..*_mot_gen.*$",
    # MVP also freezes timestep / noise-scale embedders by default;
    # MVP+aux below overrides them via TRAINABLE_REGEX_MVP_AUX.
    r"^fm_modules\.timestep_embedder\.",
    r"^fm_modules\.noise_scale_embedder\.",
)
TRAINABLE_REGEX_MVP: tuple[str, ...] = (
    r"^fm_modules\.fm_head\.",
)
TRAINABLE_REGEX_MVP_AUX: tuple[str, ...] = TRAINABLE_REGEX_MVP + (
    r"^fm_modules\.timestep_embedder\.",
    r"^fm_modules\.noise_scale_embedder\.",
    # 公开证据显示: per-layer norm_mot_gen is part of the gen path; safe
    # to add as cheap trainable params.
    r"^language_model\.model\.layers\.\d+\.input_layernorm_mot_gen\.",
    r"^language_model\.model\.layers\.\d+\.post_attention_layernorm_mot_gen\.",
    r"^language_model\.model\.norm_mot_gen\.",
)

# Scenario B: 48GB balanced — fm_head + vision_model_mot_gen full + top-N
# `_mot_gen` q/k/v/o LoRA. Build the LoRA target list separately via PEFT;
# the regex below covers the *non-LoRA* trainable modules.
FREEZE_REGEX_BALANCED: tuple[str, ...] = (
    r"^vision_model\.",
    r"^language_model\.model\.embed_tokens\b",
    r"^language_model\.lm_head\b",
    # ordinary final norm only; norm_mot_gen is left to TRAINABLE below.
    r"^language_model\.model\.norm\.",
    # ordinary core stays frozen.
    r"^language_model\.model\.layers\.\d+\.(?!.*_mot_gen).*$",
    # `_mot_gen` LLM core: stays frozen at base-weight level; LoRA adapters
    # are added separately via PEFT and become trainable independently.
    r"^language_model\.model\.layers\.\d+\.(?!input_layernorm_mot_gen|post_attention_layernorm_mot_gen).*_mot_gen.*$",
)
TRAINABLE_REGEX_BALANCED: tuple[str, ...] = (
    r"^fm_modules\.fm_head\.",
    r"^fm_modules\.vision_model_mot_gen\.",
    r"^fm_modules\.timestep_embedder\.",
    r"^fm_modules\.noise_scale_embedder\.",
    r"^language_model\.model\.layers\.\d+\.input_layernorm_mot_gen\.",
    r"^language_model\.model\.layers\.\d+\.post_attention_layernorm_mot_gen\.",
    r"^language_model\.model\.norm_mot_gen\.",
)

# Scenario "gen_vision": **deliberately no fm_head**. Trains the modules that
# the SFT→final diff probe (artifacts/sft_final_diff.json @ 2026-05-01)
# identified as the actual RL-stage drift surface:
#   vision_model_mot_gen (avg rel Δ 0.42%, top patch_embedding 1.33%)
#   timestep_embedder    (0.15%)
#   noise_scale_embedder (0.12%)
#   mot_gen layer norms  (mostly 0 but cheap to learn)
# Used by experiment C to test whether mimicking the RL-stage trainable set
# produces faster/cleaner one-step reconstruction than the default mvp_aux
# (which trains fm_head — a module the RL stage left untouched at 0.0% Δ).
FREEZE_REGEX_GEN_VISION: tuple[str, ...] = FREEZE_REGEX_MVP + (
    # MVP already freezes vision_model_mot_gen; we OVERRIDE that via
    # TRAINABLE_REGEX_GEN_VISION below. The freeze list still lists it
    # because trainable patterns take precedence in `set_requires_grad_by_regex`.
    # fm_head is *trainable* in MVP scenarios but *frozen* here (the whole
    # point of the gen_vision arm is to leave fm_head untouched, mirroring
    # the SFT→final 0% delta on fm_head).
    r"^fm_modules\.fm_head\.",
)
TRAINABLE_REGEX_GEN_VISION: tuple[str, ...] = (
    r"^fm_modules\.vision_model_mot_gen\.",
    r"^fm_modules\.timestep_embedder\.",
    r"^fm_modules\.noise_scale_embedder\.",
    r"^language_model\.model\.layers\.\d+\.input_layernorm_mot_gen\.",
    r"^language_model\.model\.layers\.\d+\.post_attention_layernorm_mot_gen\.",
    r"^language_model\.model\.norm_mot_gen\.",
)

# Scenario "aux_no_head": diagnostic — train ONLY the timestep / noise-scale
# embedders + mot_gen layer norms. NO fm_head, NO vision_model_mot_gen, NO
# `_mot_gen` LLM core. Used to test whether the textured "halftone" artifact
# in 2048²+8bit sampling after `mvp_aux` training is caused by the trainable
# fm_head (its hidden→3072 RGB-patch projection forces patchwise discontinuity
# under x0-MSE) or by something else (e.g., t-distribution mismatch).
# If after-images here are clean → fm_head was the noise source.
# If after-images are still grainy → t-distribution / loss-function level issue.
FREEZE_REGEX_AUX_NO_HEAD: tuple[str, ...] = FREEZE_REGEX_MVP + (
    # FREEZE_REGEX_MVP already covers fm_head's siblings; here we explicitly
    # also freeze fm_head and the gen vision model (otherwise they'd be
    # trainable from MVP's defaults).
    r"^fm_modules\.fm_head\.",
)
TRAINABLE_REGEX_AUX_NO_HEAD: tuple[str, ...] = (
    r"^fm_modules\.timestep_embedder\.",
    r"^fm_modules\.noise_scale_embedder\.",
    r"^language_model\.model\.layers\.\d+\.input_layernorm_mot_gen\.",
    r"^language_model\.model\.layers\.\d+\.post_attention_layernorm_mot_gen\.",
    r"^language_model\.model\.norm_mot_gen\.",
)


# --------------------------------------------------------------------------- #
# 3) Apply requires_grad masks                                                #
# --------------------------------------------------------------------------- #


@dataclass
class GradMaskReport:
    n_trainable: int
    n_frozen: int
    bucket_trainable: dict[str, int]
    bucket_frozen: dict[str, int]
    unmatched: list[str]


def _compile(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def set_requires_grad_by_regex(
    model: nn.Module,
    *,
    freeze_patterns: Iterable[str] = (),
    trainable_patterns: Iterable[str] = (),
    default: bool = False,
    strict: bool = True,
) -> GradMaskReport:
    """Set `requires_grad` on every named parameter using regex policies.

    Resolution order per parameter name:
        1. matches *any* trainable_pattern → True
        2. matches *any* freeze_pattern → False
        3. otherwise → `default`

    If `strict=True`, every parameter must match at least one pattern
    (excluding the default fallback) — useful as a guardrail when promoting
    a freeze policy to a new model revision.
    """
    train_re = _compile(trainable_patterns)
    freeze_re = _compile(freeze_patterns)

    n_trainable = 0
    n_frozen = 0
    b_train: dict[str, int] = defaultdict(int)
    b_frozen: dict[str, int] = defaultdict(int)
    unmatched: list[str] = []

    for name, p in model.named_parameters():
        bucket = classify_param(name)
        if any(r.search(name) for r in train_re):
            p.requires_grad_(True)
            n_trainable += p.numel()
            b_train[bucket] += p.numel()
        elif any(r.search(name) for r in freeze_re):
            p.requires_grad_(False)
            n_frozen += p.numel()
            b_frozen[bucket] += p.numel()
        else:
            p.requires_grad_(default)
            if default:
                n_trainable += p.numel()
                b_train[bucket] += p.numel()
            else:
                n_frozen += p.numel()
                b_frozen[bucket] += p.numel()
            unmatched.append(name)

    if strict and unmatched:
        raise RuntimeError(
            f"{len(unmatched)} params matched no pattern "
            f"(strict=True). first 5: {unmatched[:5]}"
        )

    return GradMaskReport(
        n_trainable=n_trainable,
        n_frozen=n_frozen,
        bucket_trainable=dict(b_train),
        bucket_frozen=dict(b_frozen),
        unmatched=unmatched,
    )


# --------------------------------------------------------------------------- #
# 4) Reporting helpers                                                        #
# --------------------------------------------------------------------------- #


def summarize_named_parameters(model: nn.Module) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for name, p in model.named_parameters():
        stats[classify_param(name)] += p.numel()
    return dict(stats)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def format_param_table(stats: dict[str, int]) -> str:
    total = sum(stats.values())
    lines = [f"{'group':<24s} {'params':>18s} {'share':>9s}", "-" * 56]
    for k, v in sorted(stats.items(), key=lambda kv: (-kv[1], kv[0])):
        share = (v / total) if total else 0.0
        lines.append(f"{k:<24s} {v:>18,d} {share:>8.2%}")
    lines.append("-" * 56)
    lines.append(f"{'TOTAL':<24s} {total:>18,d}")
    return "\n".join(lines)
