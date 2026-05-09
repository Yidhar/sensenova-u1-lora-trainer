"""Collator that turns `T2ISample`s into the dict consumed by `forward_t2i_step`.

Responsibilities:
1. tokenize the prompt with the same tokenizer revision as the model SHA;
2. build text_indexes / image_indexes via `masking.py`;
3. concatenate them, derive the full block-causal attention mask;
4. patchify the target image to the fm_head output space (`x0_patch`);
5. sample `t` and pre-sample `eps`, build `z_t`, then *render* the noisy
   pixel-space image needed by `vision_model_mot_gen` (since the gen
   vision encoder consumes 16×16 patches, not 32×32 ones).

The pixel-rendering step in (5) is a deliberate choice: report §3.2 notes
that we cannot statically cache T2I noisy-image features; constructing
them on the fly inside the collator keeps the training loop simple.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch

import math

from train_u1.constants import (
    FM_OUTPUT_DIM,
    NOISE_SCALE_BASE_IMAGE_SEQ_LEN,
    NOISE_SCALE_DEFAULT,
    NOISE_SCALE_MAX,
    PATCH32,
    PATCH_SIZE,
    T_EPS_DEFAULT,
)
from train_u1.data.datasets import T2ISample
from train_u1.model.masking import (
    build_t2i_image_indexes,
    build_t2i_text_indexes,
    concat_text_image_indexes,
    create_block_causal_mask,
)
from train_u1.model.patching import linear_z_t, patchify_x0, unpatchify


@dataclass
class CollatorConfig:
    # If `image_hw` is None, the collator reads the H,W from the first sample's
    # image tensor — meaning each sample's resolution wins. This is the natural
    # "native-resolution" mode for real datasets (run smart_resize per sample,
    # then batch=1).
    image_hw: tuple[int, int] | None = (512, 512)
    t_eps: float = T_EPS_DEFAULT
    t_dist: str = "uniform"      # uniform on (t_eps, 1] for MVP
    add_noise_scale: bool = True
    # Base noise_scale value (config.noise_scale = 1.0). The *effective* per-sample
    # noise_scale is computed at collator runtime as
    #   eff = min(NOISE_SCALE_MAX, sqrt(image_token_num / NOISE_SCALE_BASE) × value)
    # which matches `t2i_generate` L1656 exactly. The value passed to
    # noise_scale_embedder is `eff / NOISE_SCALE_MAX` ∈ [0, 1].
    noise_scale_value: float = NOISE_SCALE_DEFAULT
    seed: int = 0
    # Native-resolution path requires batch_size == 1 because each image has
    # a different H,W; right-padding text to the longest in batch also breaks
    # the upstream block-causal mask, which treats every text token as having
    # a unique `t` index. Lift this only after batched packing is implemented.
    enforce_batch_one: bool = True
    # Prompt formatting:
    #   "raw"      → tokenize the caption directly (matches `_build_t2i_text_inputs`
    #                in upstream; what Phase 4 has been using).
    #   "official" → wrap caption in the official chat template via the
    #                upstream `_build_t2i_query` (system message + roles +
    #                `<think>\n\n</think>\n\n<img>`-style preamble).
    # If "official", the caller must pass `model` (a NEOChatModel) at
    # collator construction so we can reuse `model._build_t2i_query`.
    prompt_template: str = "raw"
    # Style trigger: prepended to every caption before chat-template wrap.
    # Standard SDXL/Flux LoRA style-training practice (DreamBooth-style "sks"
    # token analogue). Both train and inference must use identical trigger
    # text to enable the LoRA-adapted style to fire. Empty string → disabled.
    # Format used: f"{style_trigger}, {original_caption}".
    style_trigger: str = ""


class SenseNovaU1Collator:
    """Stateful collator: holds tokenizer + config, callable on a list of `T2ISample`.

    The tokenizer must come from the same HF revision as the model so the
    `<IMG_CONTEXT>` / image start/end tokens line up. Callers usually grab
    `model.processor.tokenizer` or `AutoTokenizer.from_pretrained(MODEL_ID,
    revision=MODEL_SHA, trust_remote_code=True)`.
    """

    def __init__(self, tokenizer, cfg: CollatorConfig | None = None, *, model=None):
        self.tok = tokenizer
        self.cfg = cfg or CollatorConfig()
        self._gen = torch.Generator().manual_seed(self.cfg.seed)
        if self.cfg.prompt_template == "official":
            if model is None or not hasattr(model, "_build_t2i_query"):
                raise ValueError(
                    "prompt_template='official' requires `model=<NEOChatModel>` "
                    "with `_build_t2i_query` (passed via SenseNovaU1Collator(..., model=...))"
                )
            # Pull SYSTEM_MESSAGE_FOR_GEN from upstream utils so we match
            # `t2i_generate` L1593 *exactly*:
            #   _build_t2i_query(prompt, system_message=SYSTEM_MESSAGE_FOR_GEN,
            #                    append_text='<think>\n\n</think>\n\n<img>')
            # The append_text closes a (forced-empty) think section and opens
            # the image span; the model's prefix forward then feeds image tokens
            # right after the trailing <img>.
            try:
                # The utils module gets registered under transformers' dynamic
                # module cache when the model is loaded with trust_remote_code.
                # Pull from there to guarantee bit-faithful constants.
                from importlib import import_module
                _utils = import_module(
                    f"{model.__class__.__module__.rsplit('.', 1)[0]}.utils"
                )
                self._sys_msg_for_gen = _utils.SYSTEM_MESSAGE_FOR_GEN
            except Exception:
                # Fallback: hardcode from the pinned commit's utils.py
                self._sys_msg_for_gen = (
                    "You are an excellent painter. The user will give a description, "
                    "and you should generate an image based on it.\n"
                    "General Rules:\n"
                    "- For any visible text in the image, follow the language specified for "
                    "the rendered text in the user's description, not the language of the "
                    "prompt. If no language is specified, use the user's input language."
                )
            self._build_t2i_query = model._build_t2i_query
            self._gen_append = "<think>\n\n</think>\n\n<img>"
        else:
            self._build_t2i_query = None
            self._sys_msg_for_gen = None
            self._gen_append = None

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _tokenize(self, prompts: list[str]) -> tuple[torch.Tensor, list[int]]:
        out = self.tok(prompts, return_tensors="pt", padding=True)
        ids = out["input_ids"]
        # Trim padded EOS off lengths; we use right-padding by default.
        lens = (ids != self.tok.pad_token_id).sum(dim=1).tolist() if self.tok.pad_token_id is not None else [ids.shape[1]] * ids.shape[0]
        return ids, lens

    def _sample_t(self, batch_size: int) -> torch.Tensor:
        if self.cfg.t_dist == "uniform":
            t = torch.rand(batch_size, generator=self._gen)
            t = t * (1.0 - self.cfg.t_eps) + self.cfg.t_eps
        else:
            raise NotImplementedError(f"t_dist={self.cfg.t_dist}")
        return t

    @staticmethod
    def _check_image_hw(image_hw: tuple[int, int]) -> None:
        H, W = image_hw
        if H % PATCH32 or W % PATCH32:
            raise ValueError(
                f"image_hw=({H},{W}) must be divisible by PATCH32={PATCH32}; "
                "use smart_resize or align manually before training."
            )

    # ------------------------------------------------------------------ #
    # Main entry                                                          #
    # ------------------------------------------------------------------ #
    def __call__(self, samples: list[T2ISample]) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        if cfg.enforce_batch_one and len(samples) != 1:
            raise ValueError(
                f"native-resolution collator requires batch_size=1, got {len(samples)}. "
                "Set CollatorConfig(enforce_batch_one=False) only after you've added "
                "batched packing or per-sample mask gathering."
            )

        # Native-resolution path: take H,W from the first sample's image tensor
        # if cfg.image_hw is None. All samples in the (size-1) batch must share
        # this H,W. We rely on dataset-side smart_resize to guarantee H,W are
        # multiples of PATCH32.
        if cfg.image_hw is None:
            _, Himg, Wimg = samples[0].image.shape
            H, W = int(Himg), int(Wimg)
        else:
            H, W = cfg.image_hw
        self._check_image_hw((H, W))
        for s in samples:
            _, Hs, Ws = s.image.shape
            if (Hs, Ws) != (H, W):
                raise ValueError(
                    f"sample {s.sample_id} image hw=({Hs},{Ws}) does not match batch hw=({H},{W}). "
                    "When mixing aspect ratios, run dataset with batch_size=1."
                )

        token_h, token_w = H // PATCH32, W // PATCH32
        N = token_h * token_w

        # 1) text → ids + per-sample lengths. With enforce_batch_one we know
        #    `len(samples) == 1` so no batch padding is applied — `L_text` is
        #    exactly this prompt's length (matches upstream `_build_t2i_text_inputs`).
        # Apply style trigger BEFORE chat-template wrap so the trigger lives
        # inside the user-message portion of the chat (not in system or
        # assistant). Identical formatting must be replicated at sample time.
        if cfg.style_trigger:
            raw_prompts = [f"{cfg.style_trigger}, {s.prompt}" for s in samples]
        else:
            raw_prompts = [s.prompt for s in samples]
        if self._build_t2i_query is not None:
            prompts = [
                self._build_t2i_query(
                    rp,
                    system_message=self._sys_msg_for_gen,
                    append_text=self._gen_append,
                )
                for rp in raw_prompts
            ]
        else:
            prompts = list(raw_prompts)
        input_ids, text_lens = self._tokenize(prompts)
        B = input_ids.shape[0]
        L_text = input_ids.shape[1]

        # For the MVP we treat all text as one prefix segment (no system
        # tokens, no chat template) — matches `_build_t2i_text_inputs`.
        text_indexes = build_t2i_text_indexes(text_len=L_text, device="cpu")     # (3, L_text)
        image_indexes = build_t2i_image_indexes(
            token_h=token_h, token_w=token_w, text_len=L_text, device="cpu"
        )                                                                         # (3, N)
        position_indexes = concat_text_image_indexes(text_indexes, image_indexes)  # (3, L_text + N)
        # Two block-causal masks:
        #   `attn_mask_prefix` for the text-only prefix forward (forward_und path)
        #   `attn_mask` for the full text+image sequence (kept for diagnostics)
        # The image-only forward (forward_gen) takes `attention_mask = {"full_attention": None}`
        # per upstream _t2i_predict_v (modeling_neo_chat.py L562-600).
        attn_mask_prefix = create_block_causal_mask(text_indexes[0])              # (1, 1, L_text, L_text)
        attn_mask = create_block_causal_mask(position_indexes[0])                  # (1, 1, L, L)

        # 2) image → patches (in fm_head output space).
        images = torch.stack([s.image for s in samples], dim=0)  # (B, 3, H, W)
        x0_patch = patchify_x0(images)                            # (B, N, 3072)

        # 3) compute resolution-dependent noise_scale FIRST (we need it to
        #    scale eps below).
        if cfg.add_noise_scale:
            base = float(NOISE_SCALE_BASE_IMAGE_SEQ_LEN)
            scale = math.sqrt(N / base)
            eff_noise_scale = min(NOISE_SCALE_MAX, scale * cfg.noise_scale_value)
        else:
            eff_noise_scale = 1.0

        # 4) sample t + eps; build z_t. eps is scaled by eff_noise_scale so
        #    the training-time distribution at t→0 matches inference's
        #    initial state `image_prediction = noise_scale × randn` (公开
        #    证据显示 — t2i_generate L1665).
        t = self._sample_t(B)
        eps_raw = torch.randn(B, N, FM_OUTPUT_DIM, generator=self._gen)
        eps = eps_raw * eff_noise_scale
        z_t = linear_z_t(x0_patch, eps, t)

        # 4) render noisy *pixel-space* images for vision_model_mot_gen.
        #    z_t lives in patch-32 space; unpatchify -> (B, 3, H, W). The
        #    gen vision encoder will then patchify at patch=16 internally.
        noisy_pixel_values = unpatchify(z_t, grid_hw=(token_h, token_w), patch_size=PATCH32)
        # 16-patch grid for the gen vision model:
        noisy_grid_hw = torch.tensor([[H // PATCH_SIZE, W // PATCH_SIZE]] * B, dtype=torch.long)

        # 5) emit eff_noise_scale (already computed in step 3) for the wrapper
        #    to feed into noise_scale_embedder. Wrapper divides by NOISE_SCALE_MAX
        #    before embedder (matching `t2i_generate` L1683's
        #    `noise_scale / noise_scale_max_value`).
        if cfg.add_noise_scale:
            noise_scale = torch.full((B,), float(eff_noise_scale), dtype=torch.float32)
        else:
            noise_scale = None

        return {
            "input_ids": input_ids,
            "text_indexes": text_indexes,             # (3, L_text)
            "image_indexes": image_indexes,           # (3, N)
            "position_indexes": position_indexes,     # (3, L_text + N)
            "attn_mask_prefix": attn_mask_prefix,     # (1, 1, L_text, L_text)
            "attn_mask": attn_mask,                   # (1, 1, L, L) — diagnostics
            "x0_patch": x0_patch,                     # (B, N, 3072)
            "eps": eps,                               # (B, N, 3072)
            "t": t,                                   # (B,)
            "noisy_pixel_values": noisy_pixel_values, # (B, 3, H, W)
            "noisy_grid_hw": noisy_grid_hw,           # (B, 2)
            "noise_scale": noise_scale,               # (B,) or None
            "sample_ids": [s.sample_id for s in samples],
            "text_lens": text_lens,
            "token_hw": (token_h, token_w),
        }


def to_device(batch: dict, device: torch.device | str, dtype: torch.dtype | None = None) -> dict:
    """Move tensor entries of `batch` to (device, dtype). Leaves non-tensors alone."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device)
            if dtype is not None and t.is_floating_point():
                t = t.to(dtype)
            out[k] = t
        else:
            out[k] = v
    return out
