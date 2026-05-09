"""TrainingWrapper.forward_t2i_step — single-step FM forward.

The wrapper exists because `NEOChatModel.forward()` upstream is
`raise NotImplementedError("forward")`. We build the smallest training
forward we can directly from the *inference* building blocks that are
already public:

- `extract_feature(..., gen_model=True)` — applies `vision_model_mot_gen`
  + timestep/noise-scale embedders to noisy patches.
- `language_model.model(... image_gen_indicators=True)` — runs the
  `_mot_gen` path through Qwen3 with token-type routing.
- `fm_head(hidden)` — projects hidden states to RGB-patch space.
- `_t2i_predict_v` semantics — public formula `v = (x_pred - z_t)/(1-t)`.

References (commit df86ca90):
- modeling_neo_chat.py L304-314, L562-600, L752-770, L1578-1805, L1847-1862
- modeling_qwen3.py L152-164, L739-1001

公开证据显示 — every step references public source.
合理推断 — only the `MSE(x_pred, x0)` loss head, per report §0.1 (5).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from train_u1.constants import FM_OUTPUT_DIM, NOISE_SCALE_MAX, PATCH_SIZE, T_EPS_DEFAULT
from train_u1.model.patching import linear_z_t, patchify, predict_v_from_x


@dataclass
class T2IStepOutput:
    x_pred: torch.Tensor          # (B, N, 3072) predicted clean patches
    v_pred: torch.Tensor          # (B, N, 3072) velocity (only for compat)
    z_t: torch.Tensor             # (B, N, 3072) noisy patches at t
    hidden_image: torch.Tensor    # (B, N, hidden) image-span hidden states
    text_len: int
    image_len: int


class TrainingWrapper(nn.Module):
    """Wrap a loaded `NEOChatModel` and expose a single-step T2I forward.

    The wrapper does **not** own any new trainable params — it only
    routes tensors through the public sub-modules of the base model.
    Freeze policy is applied externally via `set_requires_grad_by_regex`.
    """

    def __init__(self, base_model: nn.Module, t_eps: float = T_EPS_DEFAULT):
        super().__init__()
        self.model = base_model
        self.t_eps = t_eps

    # ------------------------------------------------------------------ #
    # Sub-module accessors — fail fast if upstream renames them          #
    # ------------------------------------------------------------------ #
    @property
    def fm_modules(self) -> nn.Module:
        return self.model.fm_modules

    @property
    def fm_head(self) -> nn.Module:
        return self.model.fm_modules["fm_head"]

    @property
    def vision_model_mot_gen(self) -> nn.Module:
        return self.model.fm_modules["vision_model_mot_gen"]

    @property
    def language_model(self) -> nn.Module:
        return self.model.language_model

    # ------------------------------------------------------------------ #
    # Forward                                                             #
    # ------------------------------------------------------------------ #
    def forward_t2i_step(
        self,
        batch: dict[str, Any],
        *,
        prefix_grad: bool = False,
        prefix_kv=None,
    ) -> T2IStepOutput:
        """One Euler-equivalent FM training step.

        Mirrors the upstream two-stage pattern (公开证据显示 —
        `_t2i_prefix_forward` + `_t2i_predict_v` in modeling_neo_chat.py):
        1. Prefix forward over text-only ids (forward_und) → `past_key_values`.
        2. Image-only forward over noisy-image embeddings (forward_gen) using
           `past_key_values=prefix_kv, update_cache=False`.

        This deliberately avoids the mixed-routing branch in Qwen3Attention.forward,
        which has shape-incompatible q_norm calls (its q_norm weights are sized
        for head_dim, but the mixed-routing path applies them on flat hidden
        chunks — see modeling_qwen3.py L758-781).

        `prefix_grad=False` (default) wraps the prefix forward in `no_grad`,
        which is correct only when the ordinary path is fully frozen (the
        report's MVP / Balanced scenarios).

        Required `batch` fields (see `data/collators.py`):
            input_ids            : (B, L_text)              int64
            text_indexes         : (3, L_text)              int64
            image_indexes        : (3, N)                   int64
            attn_mask_prefix     : (1, 1, L_text, L_text)   float
            x0_patch             : (B, N, FM_OUTPUT_DIM)    bf16/fp32
            eps                  : (B, N, FM_OUTPUT_DIM)    same
            t                    : (B,)                     in (t_eps, 1]
            noise_scale          : (B,) or None
            noisy_pixel_values   : (B, 3, H_img, W_img)
            noisy_grid_hw        : (B, 2)
        """
        x0 = batch["x0_patch"]
        eps = batch["eps"]
        t = batch["t"].to(x0.device)
        if x0.shape[-1] != FM_OUTPUT_DIM:
            raise ValueError(f"x0 dim {x0.shape[-1]} != FM_OUTPUT_DIM {FM_OUTPUT_DIM}")

        # 1) z_t in fm_head output space (patch=32, dim=3072).
        z_t = linear_z_t(x0, eps, t)

        # 2) gen-side visual features. Upstream `extract_feature(...,
        #    gen_model=True)` consumes flat `(B*grid_h*grid_w, c*p*p)` patches
        #    and returns `(B * token_h * token_w, hidden)` after the internal
        #    2×2 dense merge (公开证据显示 — modeling_neo_chat.py L1470).
        noisy_pixel_values = batch["noisy_pixel_values"]
        Bv, _, Himg, Wimg = noisy_pixel_values.shape
        grid_h = Himg // PATCH_SIZE
        grid_w = Wimg // PATCH_SIZE
        image_input = patchify(noisy_pixel_values, PATCH_SIZE, channel_first=True)
        image_input_flat = image_input.reshape(Bv * grid_h * grid_w, -1)
        img_embeds = self.model.extract_feature(
            image_input_flat,
            gen_model=True,
            grid_hw=batch["noisy_grid_hw"],
        )
        N = batch["x0_patch"].shape[1]
        img_embeds = img_embeds.view(Bv, N, -1)

        # 3) timestep + (optional) noise-scale conditioning.
        img_embeds = img_embeds + self.fm_modules["timestep_embedder"](t).unsqueeze(1)
        if "noise_scale_embedder" in self.fm_modules and batch.get("noise_scale") is not None:
            ns = batch["noise_scale"].to(img_embeds.dtype) / NOISE_SCALE_MAX
            img_embeds = img_embeds + self.fm_modules["noise_scale_embedder"](ns).unsqueeze(1)

        # 4) prefix forward (text-only, `forward_und` path). When the ordinary
        #    path is frozen we don't need gradients here — wrap in no_grad
        #    so prefix activations aren't kept on the autograd tape.
        # If `prefix_kv` is provided (bf16-offload static prefix cache mode),
        # skip the prefix forward entirely — caller has pre-computed KV.
        input_ids = batch["input_ids"]
        if prefix_kv is None:
            text_indexes = batch["text_indexes"]
            attn_mask_prefix = batch["attn_mask_prefix"]
            prefix_ctx = torch.enable_grad() if prefix_grad else torch.no_grad()
            with prefix_ctx:
                prefix_out = self.language_model.model(
                    input_ids=input_ids,
                    indexes=text_indexes,
                    attention_mask={"full_attention": attn_mask_prefix},
                    use_cache=True,
                )
            prefix_kv = prefix_out.past_key_values

        # 5) image-only forward (`forward_gen` path). Mirrors `_t2i_predict_v`
        #    exactly: image_gen_indicators = ones, attention_mask =
        #    {"full_attention": None}, past_key_values = prefix_kv,
        #    update_cache=False.
        image_gen_indicators = torch.ones(
            (img_embeds.shape[0], img_embeds.shape[1]),
            dtype=torch.bool,
            device=img_embeds.device,
        )
        gen_out = self.language_model.model(
            inputs_embeds=img_embeds,
            indexes=batch["image_indexes"],
            image_gen_indicators=image_gen_indicators,
            attention_mask={"full_attention": None},
            past_key_values=prefix_kv,
            use_cache=True,
            update_cache=False,
        )
        hidden_image = gen_out.last_hidden_state  # (B, N, hidden)

        # 6) fm_head → x_pred → v_pred.
        x_pred = self.fm_head(hidden_image)
        v_pred = predict_v_from_x(x_pred, z_t, t, t_eps=self.t_eps)

        return T2IStepOutput(
            x_pred=x_pred,
            v_pred=v_pred,
            z_t=z_t,
            hidden_image=hidden_image,
            text_len=int(input_ids.shape[1]),
            image_len=int(N),
        )
