"""Metadata-only LoRA target estimator for dense 8B and A3B/MoE configs.

This script intentionally reads only ``config.json``. It does not instantiate
the model and does not download safetensors shards, so it is safe to run against
large A3B repositories before the public MoE runtime is usable locally.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from train_u1.model.lora import parse_lora_spec_str


_MOE_EXPERT_TARGET_RE = re.compile(
    r"^mlp_mot_gen\.experts\.(?P<expert>\*|\d+)\."
    r"(?P<leaf>gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class ShapeEstimate:
    modules: int
    params: int
    note: str = ""


def _load_config(model: str, cache_dir: str | None = None) -> dict:
    path = Path(model)
    if path.is_dir():
        return json.loads((path / "config.json").read_text())
    if path.is_file():
        return json.loads(path.read_text())

    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(model, "config.json", cache_dir=cache_dir)
    return json.loads(Path(cfg_path).read_text())


def _linear_lora_params(in_features: int, out_features: int, rank: int) -> int:
    return rank * (in_features + out_features)


def _llm_config(cfg: dict) -> dict:
    return cfg.get("llm_config", cfg)


def _fm_output_dim(cfg: dict) -> int:
    patch_size = int(cfg.get("patch_size", 16))
    downsample_ratio = float(cfg.get("downsample_ratio", 0.5))
    merge_size = int(1 / downsample_ratio)
    return 3 * (patch_size * merge_size) ** 2


def _estimate_target(cfg: dict, target: str, rank: int) -> ShapeEstimate:
    llm = _llm_config(cfg)
    layers = int(llm.get("num_hidden_layers", 0))
    hidden = int(llm.get("hidden_size", 0))
    head_dim = int(llm.get("head_dim", hidden // max(int(llm.get("num_attention_heads", 1)), 1)))
    n_heads = int(llm.get("num_attention_heads", 0))
    n_kv = int(llm.get("num_key_value_heads", n_heads))
    q_out = n_heads * head_dim
    kv_out = n_kv * head_dim
    intermediate = int(llm.get("intermediate_size", 0))
    moe_intermediate = int(llm.get("moe_intermediate_size", 0))
    gen_experts = int(llm.get("gen_num_experts", 0) or 0)

    if target == "q_proj_mot_gen":
        return ShapeEstimate(layers, layers * _linear_lora_params(hidden, q_out, rank))
    if target in {"k_proj_mot_gen", "v_proj_mot_gen"}:
        return ShapeEstimate(layers, layers * _linear_lora_params(hidden, kv_out, rank))
    if target == "o_proj_mot_gen":
        return ShapeEstimate(layers, layers * _linear_lora_params(q_out, hidden, rank))

    if target in {"mlp_mot_gen.gate_proj", "mlp_mot_gen.up_proj"}:
        return ShapeEstimate(layers, layers * _linear_lora_params(hidden, intermediate, rank))
    if target == "mlp_mot_gen.down_proj":
        return ShapeEstimate(layers, layers * _linear_lora_params(intermediate, hidden, rank))

    m = _MOE_EXPERT_TARGET_RE.match(target)
    if m is not None:
        if gen_experts <= 0 or moe_intermediate <= 0:
            return ShapeEstimate(0, 0, "config has no generation MoE experts")
        expert_selector = m.group("expert")
        if expert_selector == "*":
            n_experts = gen_experts
        else:
            expert_idx = int(expert_selector)
            n_experts = 1 if expert_idx < gen_experts else 0
        if n_experts == 0:
            return ShapeEstimate(0, 0, "selected expert is outside gen_num_experts")
        leaf = m.group("leaf")
        if leaf in {"gate_proj", "up_proj"}:
            per_module = _linear_lora_params(hidden, moe_intermediate, rank)
        else:
            per_module = _linear_lora_params(moe_intermediate, hidden, rank)
        modules = layers * n_experts
        return ShapeEstimate(modules, modules * per_module)

    if target == "mlp_mot_gen.gate":
        if gen_experts <= 0:
            return ShapeEstimate(0, 0, "config has no generation MoE router")
        return ShapeEstimate(layers, layers * _linear_lora_params(hidden, gen_experts, rank))

    if target == "fm_modules.fm_head.0":
        return ShapeEstimate(1, _linear_lora_params(hidden, 4096, rank))
    if target == "fm_modules.fm_head.2":
        return ShapeEstimate(1, _linear_lora_params(4096, _fm_output_dim(cfg), rank))

    return ShapeEstimate(0, 0, "unknown estimator target")


def _model_kind(cfg: dict) -> str:
    llm = _llm_config(cfg)
    arch = ",".join(llm.get("architectures") or [])
    if "Moe" in arch or llm.get("gen_num_experts"):
        return "a3b_moe"
    return "dense"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF repo id, local model dir, or config.json path")
    ap.add_argument("--spec", required=True, help="LoRA spec string, e.g. 'attn=r8a8;gen_moe_mlp=r8a8'")
    ap.add_argument("--cache-dir", default="hf_cache")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = ap.parse_args()

    cfg = _load_config(args.model, cache_dir=args.cache_dir)
    specs = parse_lora_spec_str(args.spec)

    rows = []
    total_modules = 0
    total_params = 0
    for spec in specs:
        if not spec.enabled:
            continue
        est = _estimate_target(cfg, spec.target, spec.r)
        row = {
            "target": spec.target,
            "rank": spec.r,
            "alpha": spec.alpha,
            "modules": est.modules,
            "lora_params": est.params,
            "note": est.note,
        }
        rows.append(row)
        total_modules += est.modules
        total_params += est.params

    out = {
        "model": args.model,
        "model_kind": _model_kind(cfg),
        "llm_architectures": _llm_config(cfg).get("architectures"),
        "spec": args.spec,
        "targets": rows,
        "total_modules": total_modules,
        "total_lora_params": total_params,
        "approx_checkpoint_mb_fp32": total_params * 4 / 1e6,
        "runtime_note": (
            "metadata estimate only; end-to-end A3B requires a runtime with "
            "mlp_mot_gen.experts.* modules"
        ),
    }

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print(f"model: {out['model']}")
    print(f"kind:  {out['model_kind']}  llm_arch={out['llm_architectures']}")
    print(f"spec:  {out['spec']}")
    print()
    print(f"{'target':45s} {'r':>4s} {'modules':>8s} {'lora params':>14s}  note")
    print("-" * 90)
    for row in rows:
        print(
            f"{row['target']:45s} {row['rank']:4d} {row['modules']:8d} "
            f"{row['lora_params']:14,d}  {row['note']}"
        )
    print("-" * 90)
    print(f"{'total':45s} {'':4s} {total_modules:8d} {total_params:14,d}")
    print(f"approx checkpoint size if saved fp32: {out['approx_checkpoint_mb_fp32']:.1f} MB")
    print(out["runtime_note"])


if __name__ == "__main__":
    main()
