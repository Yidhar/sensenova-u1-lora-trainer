# Setup notes

## Dataset layout

The trainer expects a flat directory with paired files:

```
dataset/<name>/
├─ <id1>.jpg     # or .png / .jpeg / .webp
├─ <id1>.txt     # plain-text caption, single paragraph
├─ <id2>.jpg
├─ <id2>.txt
└─ ...
```

The dataset class (`train_u1.data.datasets.PairedFolderT2IDataset`) discovers
images by extension and pairs them with the matching `.txt` sidecar. 16–256
images is the working range for style transfer; smaller sets risk overfitting
to specific compositions, larger sets blur the style toward dataset-average.

Captions should describe the **content** of the image, not the style. The
style-trigger string is prepended automatically at training and sample time
via `--style-trigger`.

## Pinned upstream

`train_u1/constants.py` pins:

- `MODEL_ID` / `MODEL_SHA` — the post-RL SenseNova-U1-8B-MoT release.
- `SFT_MODEL_ID` / `SFT_MODEL_SHA` — the public SFT checkpoint (used as a
  baseline for evaluation only; do not train on top of it without re-deriving
  the freeze sets).
- `CODE_COMMIT` — pinned upstream commit (`df86ca90...`) for the 9
  `modeling_*.py` files we copy into the HF snapshot.

`train_u1/upstream_pinned_sha256.json` records the sha256 of each modeling
file at that commit. The installer script
(`train_u1/scripts/install_modeling_into_snapshot.py`) verifies these before
overwriting anything in the snapshot — a defence-in-depth around
`trust_remote_code=True`.

## Why bf16 base on CPU + tower offload?

The model has two parallel forward paths inside each transformer block:

- the **prefix tower** (`*_mot` modules) — used to encode text prompt + the
  image conditioning prefix;
- the **gen tower** (`*_mot_gen` modules) — used during the flow-matching
  diffusion loop, with LoRA adapters wrapped on a subset.

Only one tower is active per phase. The trainer keeps the bf16 base on CPU,
swaps the prefix tower to GPU for one prefix-forward, evicts it back to CPU,
then pins the gen tower on GPU permanently for the diffusion loop.

A static **prefix-KV cache** is precomputed once for the whole training set
(56 samples × ~59 MB ≈ 3.3 GB on GPU when `--keep-kvs-on-gpu` is set). This
removes the prefix forward from the per-step cost.

The combination — bf16 base + prefix-KV cache + LoRA in bf16 + paged
AdamW8bit + partial gradient checkpointing — peaks at ~20 GB on 32 GB cards.

## Why no 4/8-bit base for training?

We tried 4-bit nf4 and 8-bit base for LoRA training. Both produced visible
artefacts on the gen tower (grid patterns, scanlines, occasional limb
collapse) that did not appear in inference under the same quantisation. The
hypothesis is that 4/8-bit quantisation is non-linear w.r.t. the
ts/ns conditioning embedders, so the LoRA adapters overshoot during training
in directions that look fine in eval but bad in render.

Switching the **training** base to bf16 (with offload) eliminated all of
these artefacts. The trainer now requires bf16 base for training. Sampling
is also bf16 only; the legacy `sample_t2i.py` 8-bit path is kept for
reference but its output should not be trusted for style evaluation.

## Long-running training

Use `setsid + disown` for runs longer than the SSH/IDE session may live:

```bash
setsid nohup bash -c '
  HF_HOME=$PWD/hf_cache PYTHONPATH=$PWD .venv/bin/python -m \
    train_u1.scripts.train_bf16_offload <args...>
' </dev/null >run.log 2>&1 &
disown
```


