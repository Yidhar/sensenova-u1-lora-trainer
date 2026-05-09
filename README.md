---
base_model:
- sensenova/SenseNova-U1-8B-MoT
---
# train_u1 — simple LoRA trainer for SenseNova-U1-8B-MoT

A single-GPU LoRA / partial-finetune trainer for
[SenseNova-U1-8B-MoT](https://huggingface.co/sensenova/SenseNova-U1-8B-MoT).
Drives the entire run from one YAML file. Save format follows the upstream
LoRA convention (`<key>.lora_down.weight` / `lora_up.weight` / `.alpha`) so
checkpoints drop straight into the official inference scripts.

```
./train.sh configs/default.yaml                                  # train
./sample.sh configs/default.yaml output.safetensors --prompt …  # sample
```

Fits on a 32 GB GPU (RTX 5090 / A100-40 / RTX 6000 Ada). Peak VRAM ~20 GB
on the maintainer's 56-image hayateluc dataset at 2048².

---

## What you get out of the box

- **Config-first**: every run is one YAML file (`configs/default.yaml`).
- **Per-module rank + enable**: each LoRA target (`q_proj_mot_gen`, `mlp_mot_gen.down_proj`,
  `fm_modules.fm_head.0`, …) takes its own rank / alpha / on-off independently.
- **Default = official coverage at rank 64**: the same 296 module wraps as
  upstream's 8-step distill LoRA (168 attn + 126 mlp + 2 fm_head), but at
  rank 64 instead of 128 — half the trainable params, half the on-disk size,
  retains full module surface.
- **Upstream-format save**: load straight into `examples/t2i/inference.py`
  via `--lora_path`, or stack with the official 8-step LoRA.
- **bf16 training, not 4/8-bit**. Earlier 4-bit nf4 LoRA training produced
  grid artefacts and limb collapse on the gen tower; switching the base to
  bf16 (with offload + static prefix-KV cache) eliminated both.

---

## Hardware

| Resource | Required |
|---|---|
| GPU | 32 GB CUDA-12-class card (sm_90 / sm_100 / sm_120). 24 GB cards can sample but cannot train at default bucket. |
| CPU RAM | ≥ 64 GB (bf16 base lives on CPU; one prefetched batch is staged in pinned memory) |
| Disk | ≥ 80 GB (HF snapshot ~17 GB + checkpoints) |

`bitsandbytes>=0.45` and `torch>=2.9` must be linked against your CUDA
runtime. On RTX 5090 (sm_120) you'll likely need the cu128 torch wheel.

---

## Install

```bash
git clone <this-repo-url> sensenovenove
cd sensenovenove

python -m venv .venv --system-site-packages
source .venv/bin/activate
pip install -r requirements.txt

# Drop in the pinned upstream modeling .py files (sha256-guarded):
git clone https://github.com/OpenSenseNova/SenseNova-U1 u1_src
git -C u1_src checkout df86ca90bfcd95fbdd1e2b3a590822721dba8cd1

# One-time HF snapshot download (~17 GB):
HF_HOME=$PWD/hf_cache python -m train_u1.scripts.download_model
HF_HOME=$PWD/hf_cache python -m train_u1.scripts.install_modeling_into_snapshot \
    --src u1_src/src/sensenova_u1/models/neo_unify
```

---

## Train

1. Lay out your data:

   ```
   dataset/my_style/
   ├── 001.jpg     ├── 001.txt
   ├── 002.jpg     ├── 002.txt
   └── …           └── …
   ```

   Each `.txt` is a plain-text caption. The style trigger string (`my style`,
   `hayateluc style`, …) is **prepended** at training time — don't bake it
   into the captions yourself.

2. Edit `configs/default.yaml`. The only fields you must touch:

   ```yaml
   run_name: my_run
   data:
     data_dir: dataset/my_style
   style:
     trigger: "my style"
   ```

3. Launch:

   ```bash
   ./train.sh configs/default.yaml
   ```

   Long-running tip — for 2 h+ runs use `setsid + disown` so an SSH/IDE
   disconnect can't SIGHUP-kill the process:

   ```bash
   setsid nohup ./train.sh configs/default.yaml </dev/null >run.log 2>&1 &
   disown
   ```

Output:

```
artifacts/my_run/
├── checkpoints/
│   ├── step_000600.safetensors
│   └── …
├── trainable_state.safetensors      # final
└── train_log.jsonl
```

Each `.safetensors` is in upstream format (`<key>.lora_down.weight` /
`.lora_up.weight` / `.alpha`).

---

## Sample

```bash
./sample.sh configs/default.yaml \
    artifacts/my_run/trainable_state.safetensors \
    --prompt "anime girl in dark kimono on a veranda…" \
    --image-h 1024 --image-w 1024 \
    --num-steps 50 --cfg-scale 4.0 --timestep-shift 3.0 \
    --out preview.png
```

Optional `--think-mode --think-max-tokens 1024` adds a chain-of-thought
window before image generation for prompt-fidelity boosts (+~95 s/sample).

---

## Configuration

Everything below the data path is opinionated but tunable. The full schema
lives in [`train_u1/config.py`](train_u1/config.py).

```yaml
run_name: my_run

data:
  data_dir: dataset/my_style
  cap_max_pixels: 4194304          # 2048² hard cap per image
  snap_bucket: true                 # snap to upstream bucket grid
  # n_samples: 56                   # cap dataset size; default = use everything

style:
  trigger: "my style"              # prepended to every caption
  prompt_template: official        # 'official' (recommended) | 'plain'

lora:
  preset: default                  # = attn+mlp+fm_head, all r=64 a=64
  # spec: "attn=r64a64;mlp=r64a64;mlp_mot_gen.down_proj=off;fm_head=r128a128"
  dropout: 0.0

unfreeze:                          # full-FT (non-LoRA) regex patterns
  - '^fm_modules\.timestep_embedder\.'
  - '^fm_modules\.noise_scale_embedder\.'

train:
  steps: 6000
  lr: 5.0e-5
  seed: 0
  shuffle: true
  grad_accum: 1
  checkpoint_every: 600

runtime:
  keep_kvs_on_gpu: true
  gc_skip_last: 6
  device: cuda
  cpu_device: cpu
  # upstream_lora_path: SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors
  # upstream_lora_skip: ['fm_modules.fm_head']
```

### LoRA spec mini-language

`lora.spec` (or `--lora-spec` on the CLI) is a `;`-separated list of
`target=BODY` entries. Targets are specific modules **or** group aliases:

| Target | Resolves to |
|---|---|
| `q_proj_mot_gen` `k_proj_mot_gen` `v_proj_mot_gen` `o_proj_mot_gen` | one attn projection × 42 layers |
| `mlp_mot_gen.gate_proj` `mlp_mot_gen.up_proj` `mlp_mot_gen.down_proj` | one mlp projection × 42 layers |
| `fm_modules.fm_head.0` `fm_modules.fm_head.2` | one of the two patch-decoder linears |
| `attn` | all four attn projections |
| `mlp` | all three mlp projections |
| `fm_head` | both fm_head linears |
| `all` | every supported target |

`BODY` is one of:

- `r64a64` — enable with rank 64, alpha 64
- `r128` — enable with rank 128, alpha defaults to rank
- `r=64,a=32` — comma form, more readable
- `off` / `disable` — turn that target off

Examples:

```
all=r64a64                                          # = the 'default' preset
attn=r128a128;mlp=r128a128;fm_head=r128a128         # exact upstream 8-step shape
attn=r64;mlp=r64;mlp_mot_gen.down_proj=off          # ablate one MLP projection
q_proj_mot_gen=r=128,a=64;k_proj_mot_gen=r=64,a=64  # asymmetric ranks
```

### Built-in presets

| Preset | Coverage | Trainable LoRA params | Use when |
|---|---|---|---|
| `default` | 168 attn + 126 mlp + 2 fm_head, all r=64 | ~75 M | first try / production |
| `attn_only` | 168 attn, r=64 | ~50 M | ablation |
| `attn_mlp` | attn + mlp (no fm_head), r=64 | ~75 M | when fm_head is full-FT'd separately |
| `official_r128` | exact upstream shape (r=128 across all 296 wraps) | ~298 M | parameter-matching upstream's 8-step LoRA |

---

## Stack with the official 8-step distill LoRA

Upstream released a step-distillation LoRA that brings inference down to 8
NFE at `cfg_scale=1.0`. You can train your own style LoRA **on top** of it.

```yaml
# configs/stack_8step.yaml (already in this repo)
runtime:
  upstream_lora_path: hf_cache/.../SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors
  upstream_lora_skip: ['fm_modules.fm_head']   # don't clobber our fm_head LoRA
```

At sample time, also pass the same upstream LoRA:

```bash
./sample.sh configs/stack_8step.yaml \
    artifacts/my_style_8step/trainable_state.safetensors \
    --prompt "…" \
    --upstream-lora-path SenseNova-U1-8B-MoT-LoRA-8step-V1.0.safetensors \
    --upstream-lora-skip fm_modules.fm_head \
    --num-steps 8 --cfg-scale 1.0 --timestep-shift 3.0 \
    --out preview_8step.png
```

---

## Layout

```
.
├── train.sh                       # quick-launch wrapper (calls train_bf16_offload)
├── sample.sh                      # quick-launch wrapper (calls sample_t2i_offload)
├── requirements.txt               # pip dependencies
├── pyproject.toml                 # package metadata
├── LICENSE                        # Apache-2.0
├── configs/
│   ├── default.yaml               # opinionated starting point
│   ├── v16c.yaml                  # production recipe (LoRA + ts/ns/vision/fm_head full-FT)
│   └── stack_8step.yaml           # train on top of 8-step distill LoRA
├── train_u1/                      # importable package
│   ├── config.py                  # YAML config schema
│   ├── constants.py               # pinned MODEL_SHA / CODE_COMMIT / arch constants
│   ├── data/                      # collator / dataset / cache I/O
│   ├── model/
│   │   ├── lora.py                # LoraAdapter + per-spec apply
│   │   ├── lora_io.py             # save/load + upstream merge
│   │   ├── loader.py              # bf16 base load + tower offload
│   │   ├── wrapper.py             # forward_t2i_step
│   │   ├── losses.py              # fm_loss_x0
│   │   ├── patching.py            # patchify/unpatchify
│   │   └── …
│   ├── scripts/
│   │   ├── train_bf16_offload.py  # main training entry
│   │   ├── sample_t2i_offload.py  # bf16 sampler
│   │   ├── sample_t2i_offload_batch.py  # (state × prompt) sweep
│   │   ├── download_model.py      # HF snapshot download
│   │   └── install_modeling_into_snapshot.py
│   └── tests/
├── docs/
│   └── SETUP.md                   # data layout, design rationale, pinned-upstream details
├── artifacts/                     # local-only: checkpoints + sweeps (gitignored)
├── dataset/                       # local-only: image+caption pairs (gitignored)
├── hf_cache/                      # local-only: HF snapshot (gitignored)
└── u1_src/                        # local-only: upstream clone (gitignored)
```

---

## Acknowledgements & license

- **Upstream**: [`OpenSenseNova/SenseNova-U1`](https://github.com/OpenSenseNova/SenseNova-U1) (Apache-2.0).
  We pin commit `df86ca90` of the modeling code and load it via
  `trust_remote_code` after sha256 verification. The training stack here is
  independent of upstream training code (none was released).
- **Model weights**: `sensenova/SenseNova-U1-8B-MoT` (post-RL) and
  `sensenova/SenseNova-U1-8B-MoT-SFT`. Use according to their model card.
- **8-step distill LoRA**: `sensenova/SenseNova-U1-8B-MoT-LoRAs` — public
  release; consumed via the `upstream_lora_path` mechanism.
- **This trainer** is licensed under Apache-2.0 (see `LICENSE`).

**Thanks to comfy.org for the GPU power support. The open-source community will not forget.**