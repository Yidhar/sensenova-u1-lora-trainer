"""Training-run config schema.

A single YAML file describes a complete LoRA training run. Example::

    # configs/default.yaml
    run_name: my_style

    data:
      data_dir: dataset/my_style
      cap_max_pixels: 4194304
      snap_bucket: true

    style:
      trigger: "my style"
      prompt_template: official        # or 'plain'

    lora:
      preset: attn_mlp_no_head         # small-data baseline: LoRA attn+mlp only
      # spec: "attn=r64a64;mlp=r64a64;mlp_mot_gen.down_proj=off"
      dropout: 0.0

    unfreeze:                          # full-finetune (non-LoRA) regex patterns
      - '^fm_modules\\.timestep_embedder\\.'
      - '^fm_modules\\.noise_scale_embedder\\.'
      - '^fm_modules\\.vision_model_mot_gen\\.'
      - '^fm_modules\\.fm_head\\.'

    train:
      steps: 6000
      lr: 5.0e-5
      seed: 0
      shuffle: true
      grad_accum: 1
      checkpoint_every: 600
      loss_type: x0
      t_dist: uniform
      cond_dropout_text: 0.0
      cond_dropout_both: 0.0

    runtime:
      keep_kvs_on_gpu: true
      gc_skip_last: 6
      device: cuda
      cpu_device: cpu

`run_name` is interpolated into the default `checkpoint_dir`
(`artifacts/{run_name}/checkpoints`). Override at the CLI level with
`--checkpoint-dir`.

Loader resolves precedence:
  1. CLI flag if explicitly given (`argparse` default sentinel = `None`).
  2. YAML config value.
  3. Hardcoded default below.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from train_u1.model.lora import (
    LORA_PRESETS,
    LoRASpec,
    parse_lora_spec_str,
    resolve_preset,
)


@dataclass
class DataConfig:
    data_dir: str = "dataset/my_style"
    cap_max_pixels: int = 4_194_304
    snap_bucket: bool = True
    n_samples: int | None = None  # default: use entire dataset
    sample_buckets_file: str | None = None
    use_think_labels: bool = False


@dataclass
class StyleConfig:
    trigger: str = ""
    prompt_template: str = "official"   # 'official' | 'plain'


@dataclass
class LoRAConfig:
    preset: str | None = "attn_mlp_no_head"   # one of LORA_PRESETS
    spec: str | None = None          # overrides preset if set
    dropout: float = 0.0

    def resolved_specs(self) -> list[LoRASpec]:
        if self.spec:
            specs = parse_lora_spec_str(self.spec)
        elif self.preset:
            specs = resolve_preset(self.preset)
        else:
            specs = []
        if self.dropout > 0:
            specs = [
                LoRASpec(target=s.target, r=s.r, alpha=s.alpha,
                         dropout=self.dropout, enabled=s.enabled)
                for s in specs
            ]
        return specs


@dataclass
class TrainConfig:
    steps: int = 6000
    lr: float = 5.0e-5
    seed: int = 0
    shuffle: bool = True
    grad_accum: int = 1
    checkpoint_every: int = 600
    checkpoint_dir: str | None = None  # default: artifacts/{run_name}/checkpoints
    # FM loss objective. Default is the local small-data baseline (`x0`) because
    # the ablation study showed that official-style v-loss is not a good
    # small-data style-training default. `v` remains available for explicit
    # official alignment experiments.
    # Choose one of `x0` | `v` | `x0_huber` | `v_huber`.
    loss_type: str = "x0"
    huber_delta: float = 1.0
    # FM `t`-sampling distribution. Default is uniform for the same local
    # baseline reason. `logit_normal` is kept for report-alignment ablations.
    t_dist: str = "uniform"
    t_logit_mean: float = -0.8
    t_logit_std: float = 0.8
    # CFG / condition dropout. `cond_dropout_text` drops text condition only;
    # `cond_dropout_both` is the additional unconditional bucket from the
    # report. In the current pure-T2I trainer there is no separate reference
    # image condition, so both modes use the sampler's unconditional prompt
    # prefix while preserving separate log labels.
    cond_dropout_text: float = 0.0
    cond_dropout_both: float = 0.0


@dataclass
class RuntimeConfig:
    keep_kvs_on_gpu: bool = True
    gc_skip_last: int = 6
    device: str = "cuda"
    cpu_device: str = "cpu"
    upstream_lora_path: str | None = None
    upstream_lora_skip: tuple[str, ...] = ()


def _default_unfreeze_patterns() -> list[str]:
    return [
        r"^fm_modules\.timestep_embedder\.",
        r"^fm_modules\.noise_scale_embedder\.",
        r"^fm_modules\.vision_model_mot_gen\.",
        r"^fm_modules\.fm_head\.",
    ]


@dataclass
class TrainRunConfig:
    run_name: str = "my_run"
    data: DataConfig = field(default_factory=DataConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    unfreeze: list[str] = field(default_factory=_default_unfreeze_patterns)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def checkpoint_dir(self) -> str:
        if self.train.checkpoint_dir:
            return self.train.checkpoint_dir
        return f"artifacts/{self.run_name}/checkpoints"


def _coerce(target_cls, raw: dict[str, Any] | None):
    if raw is None:
        return target_cls()
    raw = dict(raw)
    # Filter to known fields (ignore unknowns rather than fail loudly so
    # comments/extra keys in user YAML don't break loading).
    valid_keys = {f.name for f in target_cls.__dataclass_fields__.values()}
    extra = set(raw) - valid_keys
    if extra:
        print(f"[config] {target_cls.__name__}: ignoring unknown keys {sorted(extra)}")
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    # Tuple coercion for runtime.upstream_lora_skip
    if "upstream_lora_skip" in filtered and isinstance(filtered["upstream_lora_skip"], list):
        filtered["upstream_lora_skip"] = tuple(filtered["upstream_lora_skip"])
    return target_cls(**filtered)


def load_train_config(path: str | Path) -> TrainRunConfig:
    """Parse a training-run YAML into a `TrainRunConfig`.

    Missing top-level sections fall back to defaults. Unknown keys are
    warned about but not fatal.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    data_cfg = _coerce(DataConfig, raw.get("data"))
    style_cfg = _coerce(StyleConfig, raw.get("style"))
    lora_cfg = _coerce(LoRAConfig, raw.get("lora"))
    train_cfg = _coerce(TrainConfig, raw.get("train"))
    runtime_cfg = _coerce(RuntimeConfig, raw.get("runtime"))

    cfg = TrainRunConfig(
        run_name=raw.get("run_name", "my_run"),
        data=data_cfg,
        style=style_cfg,
        lora=lora_cfg,
        unfreeze=list(raw.get("unfreeze") or []),
        train=train_cfg,
        runtime=runtime_cfg,
    )

    # Validate preset choice early.
    if cfg.lora.preset is not None and cfg.lora.preset not in LORA_PRESETS:
        raise ValueError(
            f"unknown lora.preset {cfg.lora.preset!r}; "
            f"valid: {list(LORA_PRESETS)}"
        )

    return cfg


def dump_train_config(cfg: TrainRunConfig, path: str | Path) -> None:
    """Round-trip helper — write a TrainRunConfig back as YAML."""
    raw = {
        "run_name": cfg.run_name,
        "data": cfg.data.__dict__,
        "style": cfg.style.__dict__,
        "lora": cfg.lora.__dict__,
        "unfreeze": cfg.unfreeze,
        "train": cfg.train.__dict__,
        "runtime": {
            **cfg.runtime.__dict__,
            "upstream_lora_skip": list(cfg.runtime.upstream_lora_skip),
        },
    }
    Path(path).write_text(yaml.safe_dump(raw, sort_keys=False))
