"""Pinned upstream revisions and architectural constants.

Source of truth: the 2026-04-30 research report (§1.1) plus the public
config.json of the HF model. These constants must match what is loaded at
runtime; if a future revision drifts, freeze new SHAs here rather than
silently inheriting them.
"""
from __future__ import annotations

# ---- Upstream version anchors ---------------------------------------------
MODEL_ID = "sensenova/SenseNova-U1-8B-MoT"
MODEL_SHA = "749fb605230f216d7a7cc0202bfb28369805466b"

SFT_MODEL_ID = "sensenova/SenseNova-U1-8B-MoT-SFT"
SFT_MODEL_SHA = "9581d526c2c811098735f310222bebc06fa99f1f"

CODE_REPO = "https://github.com/OpenSenseNova/SenseNova-U1"
CODE_COMMIT = "df86ca90bfcd95fbdd1e2b3a590822721dba8cd1"

# ---- Architectural constants (from public config.json) --------------------
# 公开证据显示 — derived from sensenova/SenseNova-U1-8B-MoT/config.json.
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
NUM_HIDDEN_LAYERS = 42
VOCAB_SIZE = 151_936
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128

VISION_HIDDEN_SIZE = 1024
PATCH_SIZE = 16
DOWNSAMPLE_RATIO = 0.5  # 2x2 dense merge in NEOVisionModel
MERGE_SIZE = int(1 / DOWNSAMPLE_RATIO)  # = 2
PATCH32 = PATCH_SIZE * MERGE_SIZE  # logical "image-token pixel size" = 32

# fm_head 输出维度 = 3 * (16*2)^2 = 3072 (RGB patch) — 公开证据显示.
FM_OUTPUT_DIM = 3 * (PATCH_SIZE * MERGE_SIZE) ** 2  # = 3072
FM_HEAD_LAYERS = 2

ADD_NOISE_SCALE_EMBEDDING = True

# 公开证据显示 — fields read directly from config.json of the pinned revision.
T_EPS_DEFAULT = 0.05            # config.t_eps
NOISE_SCALE_DEFAULT = 1.0       # config.noise_scale (the *unscaled* base value)
NOISE_SCALE_MAX = 8.0           # config.noise_scale_max_value
NOISE_SCALE_MODE = "resolution"  # config.noise_scale_mode
# Resolution-dependent scaling base: at training image_token_num =
# `noise_scale_base_image_seq_len` the effective scale equals NOISE_SCALE_DEFAULT;
# above it scales as `sqrt(image_token_num / base) × NOISE_SCALE_DEFAULT`,
# clamped to NOISE_SCALE_MAX. (公开证据显示: t2i_generate L1656.)
NOISE_SCALE_BASE_IMAGE_SEQ_LEN = 64  # config.noise_scale_base_image_seq_len
# `P_mean` / `P_std` are present in config but no public inference call uses
# them in the pinned revision (report §3.3). Recorded here only for reference.
P_MEAN_CONFIG = -0.8
P_STD_CONFIG = 0.8

# Image / vision special tokens (公开证据显示 — added_tokens.json @ MODEL_SHA)
IMG_CONTEXT_TOKEN_ID = 151_669    # "<IMG_CONTEXT>"
IMG_START_TOKEN_ID = 151_670       # "<img>"
IMG_END_TOKEN_ID = 151_671         # "</img>"
VISION_START_TOKEN_ID = 151_652
VISION_END_TOKEN_ID = 151_653
PAD_TOKEN_ID = 151_643             # config.pad_token_id
EOS_TOKEN_ID = 151_645             # config.eos_token_id

# smart_resize bounds (vision_config.min_pixels / max_pixels)
SMART_RESIZE_MIN_PIXELS = 65_536
SMART_RESIZE_MAX_PIXELS = 16_777_216

# 官方支持的训练分辨率 bucket（~2K pixel total，H/W 32 整除）
# 公开证据显示: https://github.com/OpenSenseNova/SenseNova-U1/blob/main/examples/README.md#supported-resolution-buckets
# 训练应将每张图 smart_resize 后 snap 到最接近的 bucket（按 aspect ratio 选）；
# 评测/采样应当用与训练相同 bucket 形状（避免 in/out-of-distribution mismatch）。
OFFICIAL_BUCKETS_HW: tuple[tuple[int, int], ...] = (
    (2048, 2048),  # 1:1
    (1536, 2720),  # 9:16
    (2720, 1536),  # 16:9
    (1664, 2496),  # 2:3
    (2496, 1664),  # 3:2
    (1760, 2368),  # 3:4
    (2368, 1760),  # 4:3
    (1440, 2880),  # 1:2
    (2880, 1440),  # 2:1
    (1152, 3456),  # 1:3
    (3456, 1152),  # 3:1
)

# ---- Reconciled parameter counts (公开证据显示 + 公式复核) -----------------
PARAM_COUNTS = {
    "ordinary_llm_core": 8_103_750_144,
    "mot_gen_llm_core": 8_103_750_144,
    "token_embeddings": 622_329_856,
    "lm_head": 622_329_856,
    "vision_understanding": 17_568_768,
    "vision_mot_gen": 17_568_768,
    "timestep_embedder": 17_833_984,
    "noise_scale_embedder": 17_833_984,
    "fm_head": 29_367_296,
    "final_norms": 8_192,
}
PARAM_TOTAL = 17_552_340_992
assert sum(PARAM_COUNTS.values()) == PARAM_TOTAL, "Parameter reconcile broke."

# ---- Cache schema version -------------------------------------------------
CACHE_VERSION = "sensenova-u1-cache-v1"
PREPROCESS_VERSION = "u1-preprocess-v1"
