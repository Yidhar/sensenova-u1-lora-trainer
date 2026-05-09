"""train_u1 — low-VRAM training scaffold for SenseNova-U1-8B-MoT.

Layout follows the 2026-04-30 research report (§13.7). All claims here are
graded as in the report: 公开证据显示 / 合理推断 / 待验证.
"""

from train_u1.constants import CODE_COMMIT, MODEL_ID, MODEL_SHA, SFT_MODEL_ID, SFT_MODEL_SHA

__all__ = ["MODEL_ID", "MODEL_SHA", "SFT_MODEL_ID", "SFT_MODEL_SHA", "CODE_COMMIT"]
