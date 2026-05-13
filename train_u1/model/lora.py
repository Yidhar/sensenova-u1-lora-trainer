"""LoRA adapters for SenseNova-U1 `_mot_gen` modules.

Naming and storage convention follows the upstream U1 release
(`sensenova/SenseNova-U1-8B-MoT-LoRAs`, repo `OpenSenseNova/SenseNova-U1`
post commit `8b9220e`):

    <module_path>.lora_down.weight    # shape (r, in_features), fp32 on save
    <module_path>.lora_up.weight      # shape (out_features, r),  fp32 on save
    <module_path>.alpha               # scalar buffer; int32 on save

Wrapped modules supported (per-module rank/alpha/enable independently):

    Attention (per layer × 4):
      q_proj_mot_gen   k_proj_mot_gen   v_proj_mot_gen   o_proj_mot_gen
    MLP (per layer × 3):
      mlp_mot_gen.gate_proj   mlp_mot_gen.up_proj   mlp_mot_gen.down_proj
    Patch decoder (×2):
      fm_modules.fm_head.0    fm_modules.fm_head.2

Experimental A3B/MoE target grammar (requires an A3B runtime whose modules
match the public checkpoint names):

    Generation MoE experts:
      mlp_mot_gen.experts.*.gate_proj
      mlp_mot_gen.experts.*.up_proj
      mlp_mot_gen.experts.*.down_proj
    Generation MoE router:
      mlp_mot_gen.gate

The adapter is implemented as `y = base(x) + scaling * lora_up(lora_down(x))`
with `scaling = alpha / r`. Initial state: `lora_down` kaiming uniform,
`lora_up` zeros — so the wrapped module starts at exactly the base output.

Compatibility:
- bnb `Linear4bit` / `Linear8bitLt` (legacy paths): we call `self.base(x)`,
  no special handling needed beyond reading `in_features` / `out_features`.
- `torch.utils.checkpoint`: the GC monkey-patch wraps whole decoder layers;
  per-layer forward including LoRA gets recomputed in backward correctly.
- `nn.Conv2d` (for `fm_head.0` etc. it is actually `nn.Linear`; we keep the
  Conv2d branch for future patch-encoder LoRA extension).

Legacy `lora_A` / `lora_B` parameter names produced by older checkpoints
(pre-2026-05-09) are auto-translated by `train_u1.model.lora_io.load_lora_state`.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# Target taxonomy                                                             #
# --------------------------------------------------------------------------- #

# Canonical target identifiers used in CLI specs and YAML configs. Each one
# resolves to a fixed list of nn.Module instances inside the loaded model.
ATTN_TARGETS = ("q_proj_mot_gen", "k_proj_mot_gen", "v_proj_mot_gen", "o_proj_mot_gen")
MLP_TARGETS = ("mlp_mot_gen.gate_proj", "mlp_mot_gen.up_proj", "mlp_mot_gen.down_proj")
FM_HEAD_TARGETS = ("fm_modules.fm_head.0", "fm_modules.fm_head.2")
GEN_MOE_MLP_TARGETS = (
    "mlp_mot_gen.experts.*.gate_proj",
    "mlp_mot_gen.experts.*.up_proj",
    "mlp_mot_gen.experts.*.down_proj",
)
GEN_MOE_ROUTER_TARGETS = ("mlp_mot_gen.gate",)
GEN_MOE_TARGETS = GEN_MOE_MLP_TARGETS + GEN_MOE_ROUTER_TARGETS

DENSE_KNOWN_TARGETS = ATTN_TARGETS + MLP_TARGETS + FM_HEAD_TARGETS
ALL_KNOWN_TARGETS = DENSE_KNOWN_TARGETS + GEN_MOE_TARGETS

# Convenience expansions used by the CLI parser (`attn`, `mlp`, `fm_head`).
TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    "attn": ATTN_TARGETS,
    "mlp": MLP_TARGETS,
    "fm_head": FM_HEAD_TARGETS,
    # A3B generation-side MoE aliases. These are deliberately separate from
    # `mlp`/`all` so existing 8B configs remain byte-for-byte semantic matches.
    "gen_moe_mlp": GEN_MOE_MLP_TARGETS,
    "moe_mlp": GEN_MOE_MLP_TARGETS,
    "gen_moe_router": GEN_MOE_ROUTER_TARGETS,
    "moe_router": GEN_MOE_ROUTER_TARGETS,
    "gen_moe_all": ATTN_TARGETS + GEN_MOE_TARGETS + FM_HEAD_TARGETS,
    "moe_all": ATTN_TARGETS + GEN_MOE_TARGETS + FM_HEAD_TARGETS,
    "all": DENSE_KNOWN_TARGETS,
}

_GEN_MOE_EXPERT_TARGET_RE = re.compile(
    r"^mlp_mot_gen\.experts\.(?P<expert>\*|\d+)\."
    r"(?P<leaf>gate_proj|up_proj|down_proj)$"
)


def _is_known_target(target: str) -> bool:
    return target in ALL_KNOWN_TARGETS or _GEN_MOE_EXPERT_TARGET_RE.match(target) is not None


# --------------------------------------------------------------------------- #
# Spec types                                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoRASpec:
    """Per-target LoRA configuration.

    `target` is one of `ALL_KNOWN_TARGETS` (verbatim module-name suffix) or
    an A3B MoE expert target like `mlp_mot_gen.experts.0.gate_proj`.
    `r` is the LoRA rank. `alpha` is the LoRA alpha; `scaling = alpha / r`.
    `dropout` applies to the input before `lora_down`.
    `enabled=False` lets a preset entry be turned off without removing it.
    """

    target: str
    r: int = 64
    alpha: float = 64.0
    dropout: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if not _is_known_target(self.target):
            raise ValueError(
                f"unknown LoRA target {self.target!r}. "
                f"valid: {ALL_KNOWN_TARGETS} or groups {list(TARGET_GROUPS)}"
            )
        if self.enabled and self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got r={self.r} for {self.target}")


@dataclass
class LoRAReport:
    """Summary returned by `apply_lora_specs`."""

    n_wrapped: int = 0
    n_params: int = 0
    per_target: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"wrapped {self.n_wrapped} modules ({self.n_params:,} LoRA params)"]
        for t, n in sorted(self.per_target.items(), key=lambda kv: -kv[1]):
            parts.append(f"  {t:<30s} {n:>10,d}")
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Adapter                                                                     #
# --------------------------------------------------------------------------- #


class LoraAdapter(nn.Module):
    """LoRA wrapper over a frozen base linear (or conv) module.

    Produces parameters at::

        <wrapper>.lora_down.weight   (r, in_features)
        <wrapper>.lora_up.weight     (out_features, r)
        <wrapper>.alpha              () — registered buffer, int

    Forward: `y = base(x) + scaling * lora_up(lora_down(dropout(x)))`,
    with `scaling = alpha / r`.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        r: int = 16,
        alpha: int | float = 32,
        dropout: float = 0.0,
        adapter_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {r}")
        in_features = getattr(base, "in_features", None)
        out_features = getattr(base, "out_features", None)
        if in_features is None or out_features is None:
            raise ValueError(
                f"Base module {type(base).__name__} has no in_features/out_features; "
                "LoRA wrap requires a Linear-like layer."
            )
        self.base = base
        self.r = int(r)
        self.alpha_value = float(alpha)
        self.scaling = self.alpha_value / float(r)
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.lora_down = nn.Linear(self.in_features, r, bias=False, dtype=adapter_dtype)
        self.lora_up = nn.Linear(r, self.out_features, bias=False, dtype=adapter_dtype)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # Stored as a registered buffer so it's part of state_dict — matches
        # upstream's safetensors layout (`.alpha` int32 scalar). We keep it
        # as float for precision; saver casts to int32 when emitting.
        self.register_buffer("alpha", torch.tensor(self.alpha_value, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Freeze base permanently.
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(self.dropout(x.to(self.lora_down.weight.dtype))))
        return base_out + lora_out.to(base_out.dtype) * self.scaling

    @property
    def weight(self):  # pragma: no cover — pass-through for legacy attribute reads
        return self.base.weight if hasattr(self.base, "weight") else None

    # Legacy attribute aliases (read-only). Old code that still references
    # `module.lora_A.weight` / `module.lora_B.weight` keeps working — but
    # parameter NAMES are no longer `lora_A.weight`/`lora_B.weight` (they
    # are `lora_down.weight`/`lora_up.weight`). For checkpoint loads that
    # still carry the legacy names, see `train_u1.model.lora_io.load_lora_state`.
    @property
    def lora_A(self) -> nn.Linear:  # pragma: no cover
        return self.lora_down

    @property
    def lora_B(self) -> nn.Linear:  # pragma: no cover
        return self.lora_up


# --------------------------------------------------------------------------- #
# Module resolution                                                           #
# --------------------------------------------------------------------------- #


def _walk_attn_targets(model: nn.Module, target_name: str):
    """Yield `(parent, attr, layer_idx)` for each attn projection matching target_name."""
    layers = model.language_model.model.layers
    for idx, layer in enumerate(layers):
        attn = layer.self_attn
        if hasattr(attn, target_name):
            yield attn, target_name, idx


def _walk_mlp_targets(model: nn.Module, target_name: str):
    """Yield `(parent, attr, layer_idx)` for each MLP projection matching target_name.

    `target_name` is e.g. `mlp_mot_gen.gate_proj` — split on the first dot.
    """
    sub_attr, leaf = target_name.split(".", 1)
    layers = model.language_model.model.layers
    for idx, layer in enumerate(layers):
        sub = getattr(layer, sub_attr, None)
        if sub is None or not hasattr(sub, leaf):
            continue
        yield sub, leaf, idx


def _walk_moe_mlp_targets(model: nn.Module, target_name: str):
    """Yield generation-side MoE expert projections for A3B-style modules.

    `target_name` is `mlp_mot_gen.experts.*.gate_proj` or a single expert
    target such as `mlp_mot_gen.experts.7.down_proj`. The walker is intentionally
    shape/runtime agnostic: if the loaded model has no `experts` ModuleList, it
    yields nothing so 8B dense configs are unaffected.
    """
    m = _GEN_MOE_EXPERT_TARGET_RE.match(target_name)
    if m is None:
        raise ValueError(f"invalid MoE expert target {target_name!r}")
    expert_selector = m.group("expert")
    leaf = m.group("leaf")

    layers = model.language_model.model.layers
    for layer_idx, layer in enumerate(layers):
        sub = getattr(layer, "mlp_mot_gen", None)
        experts = getattr(sub, "experts", None)
        if experts is None:
            continue
        if expert_selector == "*":
            expert_indices = range(len(experts))
        else:
            expert_idx = int(expert_selector)
            if expert_idx >= len(experts):
                continue
            expert_indices = (expert_idx,)
        for expert_idx in expert_indices:
            expert = experts[expert_idx]
            if hasattr(expert, leaf):
                yield expert, leaf, layer_idx


def _walk_moe_router_targets(model: nn.Module, target_name: str):
    """Yield generation-side MoE router gates (`mlp_mot_gen.gate`) per layer."""
    if target_name != "mlp_mot_gen.gate":
        raise ValueError(f"invalid MoE router target {target_name!r}")
    layers = model.language_model.model.layers
    for idx, layer in enumerate(layers):
        sub = getattr(layer, "mlp_mot_gen", None)
        if sub is not None and hasattr(sub, "gate"):
            yield sub, "gate", idx


def _walk_fm_head_targets(model: nn.Module, target_name: str):
    """Yield `(parent, attr, idx)` for each fm_head linear matching target_name.

    `target_name` is e.g. `fm_modules.fm_head.0` — drop `fm_modules.fm_head.`,
    treat trailing token as the attribute index on `model.fm_modules.fm_head`.
    """
    fm_modules = getattr(model, "fm_modules", None)
    if fm_modules is None:
        return
    fm_head = getattr(fm_modules, "fm_head", None)
    if fm_head is None:
        return
    # `fm_head` is an nn.Sequential or ModuleList — leaf is the integer index.
    leaf_idx = target_name.rsplit(".", 1)[-1]
    if not leaf_idx.isdigit():
        return
    leaf_idx_int = int(leaf_idx)
    if leaf_idx_int >= len(fm_head):
        return
    base = fm_head[leaf_idx_int]
    if isinstance(base, nn.Linear):
        # Use a tuple-yielding facade so the wrap loop can treat it like the
        # other walks. The "attr" is the integer cast to str so setattr-style
        # replacement falls back to indexed assignment below.
        yield fm_head, leaf_idx_int, 0


def _resolve_target_walker(target: str):
    if target in ATTN_TARGETS:
        return _walk_attn_targets
    if target in MLP_TARGETS:
        return _walk_mlp_targets
    if _GEN_MOE_EXPERT_TARGET_RE.match(target):
        return _walk_moe_mlp_targets
    if target in GEN_MOE_ROUTER_TARGETS:
        return _walk_moe_router_targets
    if target in FM_HEAD_TARGETS:
        return _walk_fm_head_targets
    raise ValueError(f"no walker for target {target!r}")


def _replace_child(parent, attr, new_module):
    """Set parent.attr = new_module, handling both attribute and indexed access."""
    if isinstance(attr, int):
        parent[attr] = new_module
    else:
        setattr(parent, attr, new_module)


def _get_child(parent, attr):
    if isinstance(attr, int):
        return parent[attr]
    return getattr(parent, attr)


# --------------------------------------------------------------------------- #
# Apply specs                                                                 #
# --------------------------------------------------------------------------- #


def apply_lora_specs(
    model: nn.Module,
    specs: list[LoRASpec],
    *,
    adapter_dtype: torch.dtype = torch.bfloat16,
    layer_filter=None,
) -> LoRAReport:
    """In-place: wrap every module identified by `specs` with a LoraAdapter.

    `layer_filter` (optional `int -> bool`) restricts attn/MLP wraps to specific
    layer indices; ignored for fm_head (no concept of layer there).

    Returns a `LoRAReport` summarising the wraps applied.
    """
    report = LoRAReport()
    for spec in specs:
        if not spec.enabled:
            continue
        walker = _resolve_target_walker(spec.target)
        n_for_target = 0
        params_for_target = 0
        for parent, attr, layer_idx in walker(model, spec.target):
            if layer_filter is not None and spec.target not in FM_HEAD_TARGETS:
                if not layer_filter(layer_idx):
                    continue
            base = _get_child(parent, attr)
            if isinstance(base, LoraAdapter):
                continue  # already wrapped
            adapter = LoraAdapter(
                base, r=spec.r, alpha=spec.alpha,
                dropout=spec.dropout, adapter_dtype=adapter_dtype,
            )
            try:
                base_device = next(base.parameters()).device
                adapter = adapter.to(base_device)
            except StopIteration:
                pass
            _replace_child(parent, attr, adapter)
            n_for_target += 1
            params_for_target += adapter.lora_down.weight.numel() + adapter.lora_up.weight.numel()
        if n_for_target:
            report.per_target[spec.target] = params_for_target
            report.n_wrapped += n_for_target
            report.n_params += params_for_target
    return report


# --------------------------------------------------------------------------- #
# CLI spec parser                                                             #
# --------------------------------------------------------------------------- #


_SPEC_TOK_RE = re.compile(r"^(?P<target>[A-Za-z0-9_.*]+)(?:=(?P<body>.+))?$")
_RA_RE = re.compile(r"^r(?P<r>\d+)(?:a(?P<alpha>\d+(?:\.\d+)?))?$")


def parse_lora_spec_str(s: str) -> list[LoRASpec]:
    """Parse a CLI-friendly LoRA spec string.

    Syntax: `target=BODY` entries separated by `;` (whitespace ignored). BODY is:
        - `rNaM`     enable target with rank=N, alpha=M  (alpha defaults to N)
        - `rN`       enable with rank=N, alpha=N
        - `off`      disable a target (overrides earlier entries)
        - `r=N,a=M`  alternative comma form (more readable)

    Group expansions: `attn`, `mlp`, `fm_head`, `gen_moe_mlp`,
    `gen_moe_router`, `gen_moe_all`, `all` expand to their member targets,
    all sharing the same body.

    Examples::

        attn=r64a64;mlp=r64a64
        q_proj_mot_gen=r128a128; k_proj_mot_gen=r128a128
        all=r64a64; mlp_mot_gen.down_proj=off
        gen_moe_mlp=r8a8; gen_moe_router=r4a4
        fm_head=r=128,a=128
    """
    specs: dict[str, LoRASpec] = {}
    for raw in s.split(";"):
        tok = raw.strip()
        if not tok:
            continue
        m = _SPEC_TOK_RE.match(tok)
        if not m:
            raise ValueError(f"cannot parse LoRA spec token: {tok!r}")
        target = m.group("target")
        body = (m.group("body") or "").strip()

        targets = TARGET_GROUPS.get(target, (target,))
        for t in targets:
            if not _is_known_target(t):
                raise ValueError(
                    f"unknown LoRA target {t!r}. "
                    f"valid: {ALL_KNOWN_TARGETS} or groups {list(TARGET_GROUPS)}"
                )
            if body in ("", "on", "enable"):
                # Default config when only the target is named.
                specs[t] = LoRASpec(target=t)
                continue
            if body in ("off", "disable"):
                specs[t] = LoRASpec(target=t, r=1, enabled=False)
                continue
            r, alpha = _parse_body(body)
            specs[t] = LoRASpec(target=t, r=r, alpha=alpha)

    return list(specs.values())


def _parse_body(body: str) -> tuple[int, float]:
    """Return (r, alpha) parsed from one of `rNaM` / `rN` / `r=N,a=M`."""
    body = body.replace(" ", "")
    # Comma-form `r=N,a=M`
    if "," in body or "=" in body:
        r = None
        alpha = None
        for part in body.split(","):
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"cannot parse spec body {body!r}: token {part!r}")
            k, v = part.split("=", 1)
            k = k.strip().lower()
            if k == "r":
                r = int(v)
            elif k in ("a", "alpha"):
                alpha = float(v)
            else:
                raise ValueError(f"unknown key {k!r} in spec body {body!r}")
        if r is None:
            raise ValueError(f"missing rank in spec body {body!r}")
        return r, float(alpha if alpha is not None else r)
    # Compact form `rNaM` / `rN`
    m = _RA_RE.match(body)
    if not m:
        raise ValueError(f"cannot parse spec body {body!r}; expected rNaM or r=N,a=M")
    r = int(m.group("r"))
    alpha = m.group("alpha")
    return r, float(alpha) if alpha is not None else float(r)


# --------------------------------------------------------------------------- #
# Presets                                                                     #
# --------------------------------------------------------------------------- #

# Named presets that resolve to a list of LoRASpec entries via spec-string.
LORA_PRESETS: dict[str, str] = {
    # **Default**: matches the official 8-step LoRA's module coverage
    # (296 wraps = 168 attn + 126 mlp + 2 fm_head) but at rank 64 instead
    # of upstream's rank 128. Halves trainable LoRA params (~149 M → ~75 M)
    # and halves on-disk size while keeping the same surface.
    "default": "attn=r64a64;mlp=r64a64;fm_head=r64a64",

    # Attention-only LoRA, ablation baseline.
    "attn_only": "attn=r64a64",

    # Attn + MLP only (no fm_head); equivalent to our pre-v16c v15a recipe.
    "attn_mlp": "attn=r64a64;mlp=r64a64",

    # **Safe presets** that explicitly drop fm_head from the trained surface.
    # The technical report's grid-artifact discussion attributes artifacts to
    # the final FFN + MLP head independently modelling disjoint 32×32 patches,
    # and notes that the official T2I RL stage freezes the generation-branch
    # MLP head and the last three transformer layers for exactly this reason.
    # Use these when you want to avoid touching the head at all.
    "attn_only_no_head": "attn=r64a64",
    "attn_mlp_no_head": "attn=r64a64;mlp=r64a64",

    # Exact upstream 8-step distill LoRA shape (rank 128 alpha 128).
    "official_r128": "attn=r128a128;mlp=r128a128;fm_head=r128a128",

    # Experimental A3B/MoE coverage. Small ranks are intentional: covering all
    # 48 layers × 32 gen experts × 3 projections gets large quickly.
    "a3b_moe_r8": "attn=r8a8;gen_moe_mlp=r8a8;fm_head=r8a8",
    "a3b_moe_router_r8": "gen_moe_router=r8a8",
}


def resolve_preset(name: str) -> list[LoRASpec]:
    if name not in LORA_PRESETS:
        raise ValueError(
            f"unknown preset {name!r}. valid presets: {list(LORA_PRESETS)}"
        )
    return parse_lora_spec_str(LORA_PRESETS[name])


# --------------------------------------------------------------------------- #
# Counting                                                                    #
# --------------------------------------------------------------------------- #


def lora_param_count(model: nn.Module) -> int:
    """Total LoRA-adapter parameters in the model (lora_down + lora_up)."""
    n = 0
    for module in model.modules():
        if isinstance(module, LoraAdapter):
            n += module.lora_down.weight.numel() + module.lora_up.weight.numel()
    return n


def list_wrapped_targets(model: nn.Module) -> list[tuple[str, int, int]]:
    """Return `(qualified_name, r, alpha)` for every LoraAdapter in the model.

    Useful for debugging or for re-emitting the spec from a wrapped model.
    """
    out: list[tuple[str, int, int]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoraAdapter):
            out.append((name, module.r, int(module.alpha_value)))
    return out


# --------------------------------------------------------------------------- #
# Backwards-compatibility wrappers                                            #
# --------------------------------------------------------------------------- #


def wrap_mot_gen_attention(
    model,
    *,
    targets: tuple[str, ...] = ATTN_TARGETS,
    r: int = 16,
    alpha: int | float = 32,
    dropout: float = 0.0,
    adapter_dtype: torch.dtype = torch.bfloat16,
    layer_filter=None,
) -> int:
    """Legacy uniform-rank attention wrapper. Prefer `apply_lora_specs`."""
    specs = [LoRASpec(target=t, r=r, alpha=alpha, dropout=dropout) for t in targets]
    return apply_lora_specs(
        model, specs, adapter_dtype=adapter_dtype, layer_filter=layer_filter
    ).n_wrapped


def wrap_mot_gen_mlp(
    model,
    *,
    targets: tuple[str, ...] = MLP_TARGETS,
    r: int = 16,
    alpha: int | float = 32,
    dropout: float = 0.0,
    adapter_dtype: torch.dtype = torch.bfloat16,
    layer_filter=None,
) -> int:
    """Legacy uniform-rank MLP wrapper. Prefer `apply_lora_specs`.

    Accepts either bare leaf names like `gate_proj` (for back-compat) or full
    `mlp_mot_gen.gate_proj` paths.
    """
    full_targets: list[str] = []
    for t in targets:
        if t.startswith("mlp_mot_gen."):
            full_targets.append(t)
        else:
            full_targets.append(f"mlp_mot_gen.{t}")
    specs = [LoRASpec(target=t, r=r, alpha=alpha, dropout=dropout) for t in full_targets]
    return apply_lora_specs(
        model, specs, adapter_dtype=adapter_dtype, layer_filter=layer_filter
    ).n_wrapped


def wrap_fm_head(
    model,
    *,
    r: int = 128,
    alpha: int | float = 128,
    dropout: float = 0.0,
    adapter_dtype: torch.dtype = torch.bfloat16,
) -> int:
    """LoRA-wrap the two `fm_modules.fm_head.{0,2}` patch-decoder linears."""
    specs = [LoRASpec(target=t, r=r, alpha=alpha, dropout=dropout) for t in FM_HEAD_TARGETS]
    return apply_lora_specs(model, specs, adapter_dtype=adapter_dtype).n_wrapped
