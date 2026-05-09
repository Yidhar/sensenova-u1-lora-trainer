"""Offline tests for param classification + freeze regex on a mock module.

Builds a stub `nn.Module` whose parameter names mirror the U1 module tree
(see report §2.1) so we can exercise the regex policy without downloading
~35 GB of weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from train_u1 import constants as C
from train_u1.model.params import (
    BUCKETS,
    FREEZE_REGEX_BALANCED,
    FREEZE_REGEX_MVP,
    TRAINABLE_REGEX_BALANCED,
    TRAINABLE_REGEX_MVP,
    TRAINABLE_REGEX_MVP_AUX,
    classify_param,
    set_requires_grad_by_regex,
    summarize_named_parameters,
)


def _p(numel: int = 1) -> nn.Parameter:
    """Tiny parameter so 42 * lots-of-stubs stays cheap (KB, not GB)."""
    return nn.Parameter(torch.zeros(numel))


def build_stub_u1_module() -> nn.Module:
    """Mock NEOChatModel-shaped parameter tree for unit tests."""
    m = nn.Module()
    # vision (understanding)
    m.register_parameter("vision_model__embeddings__patch_embedding__weight", _p(8))
    # fm_modules
    m.register_parameter("fm_modules__vision_model_mot_gen__embeddings__weight", _p(8))
    m.register_parameter("fm_modules__timestep_embedder__mlp__0__weight", _p(8))
    m.register_parameter("fm_modules__noise_scale_embedder__mlp__0__weight", _p(8))
    m.register_parameter("fm_modules__fm_head__layers__0__weight", _p(8))
    # language_model.model.embed_tokens
    m.register_parameter("language_model__model__embed_tokens__weight", _p(16))
    # language_model.lm_head
    m.register_parameter("language_model__lm_head__weight", _p(16))
    # language_model.model.norm + norm_mot_gen
    m.register_parameter("language_model__model__norm__weight", _p(4))
    m.register_parameter("language_model__model__norm_mot_gen__weight", _p(4))

    # 42 layers — ordinary + _mot_gen variants
    for i in range(C.NUM_HIDDEN_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m.register_parameter(f"language_model__model__layers__{i}__self_attn__{proj}__weight", _p(8))
            m.register_parameter(
                f"language_model__model__layers__{i}__self_attn__{proj}_mot_gen__weight",
                _p(8),
            )
        for ln in ("input_layernorm", "post_attention_layernorm"):
            m.register_parameter(f"language_model__model__layers__{i}__{ln}__weight", _p(2))
            m.register_parameter(f"language_model__model__layers__{i}__{ln}_mot_gen__weight", _p(2))
        for mlp_w in ("gate_proj", "up_proj", "down_proj"):
            m.register_parameter(f"language_model__model__layers__{i}__mlp__{mlp_w}__weight", _p(8))
            m.register_parameter(
                f"language_model__model__layers__{i}__mlp_mot_gen__{mlp_w}__weight",
                _p(8),
            )
    # `register_parameter` uses single-underscore segments; convert to dotted
    # naming to match the real model.
    return _DottedView(m)


class _DottedView(nn.Module):
    """Wrap a stub module so named_parameters() yields dotted names."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self._base = base
        # Store params as plain attributes with dotted keys for iteration.
        self._dotted: dict[str, nn.Parameter] = {}
        for name, p in base._parameters.items():  # type: ignore[attr-defined]
            self._dotted[name.replace("__", ".")] = p

    def named_parameters(self, *args, **kwargs):
        for k, v in self._dotted.items():
            yield k, v

    def parameters(self, *args, **kwargs):
        return iter(self._dotted.values())


def test_classify_examples():
    cases = {
        "vision_model.embeddings.patch_embedding.weight": "vision_understanding",
        "fm_modules.vision_model_mot_gen.embeddings.weight": "vision_mot_gen",
        "fm_modules.timestep_embedder.mlp.0.weight": "timestep_embedder",
        "fm_modules.noise_scale_embedder.mlp.0.weight": "noise_scale_embedder",
        "fm_modules.fm_head.layers.0.weight": "fm_head",
        "language_model.model.embed_tokens.weight": "token_embeddings",
        "language_model.lm_head.weight": "lm_head",
        "language_model.model.norm.weight": "final_norms",
        "language_model.model.norm_mot_gen.weight": "final_norms",
        "language_model.model.layers.7.self_attn.q_proj.weight": "ordinary_llm_core",
        "language_model.model.layers.7.self_attn.q_proj_mot_gen.weight": "mot_gen_llm_core",
        "language_model.model.layers.7.input_layernorm.weight": "ordinary_llm_core",
        "language_model.model.layers.7.input_layernorm_mot_gen.weight": "mot_gen_llm_core",
        "language_model.model.layers.7.mlp.gate_proj.weight": "ordinary_llm_core",
        "language_model.model.layers.7.mlp_mot_gen.gate_proj.weight": "mot_gen_llm_core",
    }
    for name, expected in cases.items():
        got = classify_param(name)
        assert got == expected, f"{name!r} -> {got!r} (expected {expected!r})"


def test_all_buckets_visited():
    stub = build_stub_u1_module()
    stats = summarize_named_parameters(stub)
    seen = set(stats)
    expected_present = set(BUCKETS) - {"other"}
    missing = expected_present - seen
    assert not missing, f"buckets missing in stub: {missing}"


def test_mvp_freeze_only_fm_head_trainable():
    stub = build_stub_u1_module()
    rep = set_requires_grad_by_regex(
        stub,
        freeze_patterns=FREEZE_REGEX_MVP,
        trainable_patterns=TRAINABLE_REGEX_MVP,
        default=False,
        strict=True,
    )
    # only fm_head is trainable
    assert set(rep.bucket_trainable) == {"fm_head"}
    # every other bucket is fully frozen
    frozen_buckets = set(rep.bucket_frozen)
    assert "ordinary_llm_core" in frozen_buckets
    assert "mot_gen_llm_core" in frozen_buckets
    assert "vision_understanding" in frozen_buckets


def test_mvp_aux_adds_norms_and_embedders():
    stub = build_stub_u1_module()
    rep = set_requires_grad_by_regex(
        stub,
        freeze_patterns=FREEZE_REGEX_MVP,
        trainable_patterns=TRAINABLE_REGEX_MVP_AUX,
        default=False,
        strict=True,
    )
    train_keys = set(rep.bucket_trainable)
    assert {"fm_head", "timestep_embedder", "noise_scale_embedder"} <= train_keys
    # mot_gen norms (per-layer) live in the mot_gen_llm_core bucket per
    # classify_param; so we should see *some* mot_gen_llm_core params trainable.
    assert rep.bucket_trainable.get("mot_gen_llm_core", 0) > 0
    # ordinary_llm_core stays frozen entirely
    assert rep.bucket_trainable.get("ordinary_llm_core", 0) == 0


def test_balanced_keeps_ordinary_frozen():
    stub = build_stub_u1_module()
    rep = set_requires_grad_by_regex(
        stub,
        freeze_patterns=FREEZE_REGEX_BALANCED,
        trainable_patterns=TRAINABLE_REGEX_BALANCED,
        default=False,
        strict=True,
    )
    assert rep.bucket_trainable.get("ordinary_llm_core", 0) == 0
    # vision_mot_gen is fully trainable in balanced scenario
    assert rep.bucket_trainable.get("vision_mot_gen", 0) > 0
    # token_embeddings stay frozen in balanced too
    assert rep.bucket_trainable.get("token_embeddings", 0) == 0


def test_strict_flag_raises_on_unknown_param():
    stub = build_stub_u1_module()
    # add a bogus param that nothing matches
    stub._dotted["totally.unrelated.param"] = nn.Parameter(torch.zeros(1))
    try:
        set_requires_grad_by_regex(
            stub,
            freeze_patterns=FREEZE_REGEX_MVP,
            trainable_patterns=TRAINABLE_REGEX_MVP,
            default=False,
            strict=True,
        )
    except RuntimeError as e:
        assert "totally.unrelated.param" in str(e) or "unmatched" in str(e).lower()
    else:
        raise AssertionError("strict=True should have raised on unmatched param")
