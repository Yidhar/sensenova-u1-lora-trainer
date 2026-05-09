# train_u1 — SenseNova-U1-8B-MoT 低显存训练 / 反推骨架

包内说明（与发布到 git 的 trainer 同步）。仓库级入口请看根目录 `README.md`，
配方/实验细节（per-experiment diagnostics、checkpoints、sweeps）在维护者的
本地 `artifacts/` 下，未随 trainer 公开。

## 目录布局（与代码同步，2026-05-01）

```
train_u1/
├─ constants.py            # 固定 HF SHA / GitHub commit / 架构常数 / 参数 reconcile / 真实 config 字段
├─ upstream_pinned_sha256.json   # commit df86ca9 的 9 个 modeling .py 哈希（守卫 trust_remote_code）
├─ requirements.txt        # 含 transformers==4.57.1 / bitsandbytes / peft 等（venv 装）
├─ data/
│  ├─ datasets.py          # SyntheticT2ITinyDataset / FilesystemT2ITinyDataset / PairedFolderT2IDataset
│  ├─ collators.py         # SenseNovaU1Collator（native-resolution + batch=1 守卫 + 可选 official template）
│  ├─ cache_io.py          # safetensors blob + JSONL manifest + duplicate-id 守卫
│  └─ u1_preprocess.py     # 上游 smart_resize + ImageNet normalize
├─ model/
│  ├─ loader.py            # 4-bit base + bf16 trainable + local-snapshot 优先
│  ├─ wrapper.py           # TrainingWrapper.forward_t2i_step（prefix→gen 两段）
│  ├─ losses.py            # fm_loss_x0 / fm_loss_v / Huber / text_ce_guardrail
│  ├─ masking.py           # block-causal mask / THW index helpers
│  ├─ patching.py          # patchify(16/32) / unpatchify / linear_z_t / predict_v_from_x
│  └─ params.py            # 分类 + freeze regex（MVP / MVP+aux / Balanced）+ set_requires_grad
├─ scripts/
│  ├─ download_model.py                     # HF + SFT 固定 SHA 下载
│  ├─ install_modeling_into_snapshot.py     # 9 个 .py 经 sha256 守卫装入 snapshot
│  ├─ verify_freeze_regex_against_index.py  # strict=True 真模型对齐
│  ├─ reconcile.py                          # 公式表 + freeze 计划（已 pin revision）
│  ├─ forward_smoke.py                      # 单步 forward 烟雾测试
│  ├─ cache_equiv_smoke.py                  # online-vs-cached vit_embeds 等价
│  ├─ train_fm_mvp.py                       # 24GB MVP；含 --eval-panel-out 钩子
│  ├─ eval_one_step.py                      # 实验 A：一步去噪 4-panel 可视化 + MSE 表
│  ├─ regression_suite.py                   # 4 类 prompt 固定回归（forward 级别）
│  ├─ sample_t2i.py                         # 实验 D：完整 t2i_generate 采样验证（pending wrapper hooks）
│  └─ sft_final_diff.py                     # safetensors mmap walk → SFT vs final delta heatmap
└─ tests/
   ├─ test_constants.py     # 4 ✓
   ├─ test_params.py        # 6 ✓
   ├─ test_patchify.py      # 8 ✓
   ├─ test_mask.py          # 5 ✓
   ├─ test_collator.py      # 5 ✓ (含 batch>1 守卫 + native HW)
   ├─ test_cache_io.py      # 8 ✓ (含 duplicate-id error/replace)
   ├─ test_u1_preprocess.py # 4 ✓ (smart_resize + paired folder)
   ├─ test_overfit_smoke.py # 1 (heavy, RUN_HEAVY_TESTS=1)
   ├─ test_repro_seed.py    # 1 (heavy)
   └─ test_grad_flow.py     # 1 (heavy)
```

**未实现 / pending**：
- `model/peft_targets.py` — LoRA target 工具（实验 C 时落地）
- `scripts/train_balanced.py` — 48GB 平衡场景（实验 C 后视效果决定）
- `scripts/sample_t2i.py` 完整管线（依赖 wrapper 的 `t2i_generate` 钩子）

## 阶段计划（与 TaskList 对齐）

- Phase 0  目录骨架 + 版本锚点 — 当前任务
- Phase 1  参数 reconcile / freeze 工具 / 静态测试（无需下载权重，纯代码）
- Phase 2  TrainingWrapper + losses + collator（开始触碰 NEOChatModel）
- Phase 3  cache_io + online-vs-cached 等价（要求模型已 load 一次）
- Phase 4  train_fm_mvp 跑通 1→4→16 样本 overfit
- Phase 5  regression suite + SFT-final diff probe

## 评级口径（沿用研究报告政策）

代码注释里只写四档：
- `公开证据显示` — config/源码可直接复核
- `公式复核` — 用公开常数推出，且与 HF metadata 等量对齐
- `合理推断` — 从推理代码反推训练用法
- `待验证` — 实验未完成，禁止当事实使用

不要在代码或文档里把 "待验证" 写成 "事实"。
