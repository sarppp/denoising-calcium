



















Here's how to use the models in `@/app/workspace/src/cidc`. All five architectures ([deepinterp](cci:9://file:///app/workspace/src/cidc/models/deepinterp:0:0-0:0), [deepcad](cci:9://file:///app/workspace/src/cidc/models/deepcad:0:0-0:0), [mamba3d](cci:9://file:///app/workspace/src/cidc/models/mamba3d:0:0-0:0), [n2v3d](cci:9://file:///app/workspace/src/cidc/models/n2v3d:0:0-0:0), [pinn](cci:9://file:///app/workspace/src/cidc/models/pinn:0:0-0:0)) share the same CLI and config schema — only the YAML changes.

# Entry point

The package exposes a single console script (`pyproject.toml:37`):

```@/app/pyproject.toml:36-37
[project.scripts]
cidc = "cidc.cli:main"
```

Invoke as `uv run cidc <subcommand>`. Subcommands are `show`, `train`, `infer`, `score`, `bench`, `eval-all` (see `@/app/workspace/src/cidc/cli.py:7-14`).

# Configs

One YAML per model in `@/app/workspace/configs/`:

- **deepinterp** — [configs/deepinterp.yaml](cci:7://file:///app/workspace/configs/deepinterp.yaml:0:0-0:0)
- **deepcad** — [configs/deepcad.yaml](cci:7://file:///app/workspace/configs/deepcad.yaml:0:0-0:0)
- **mamba3d** — [configs/mamba3d.yaml](cci:7://file:///app/workspace/configs/mamba3d.yaml:0:0-0:0)
- **n2v3d** — [configs/n2v3d.yaml](cci:7://file:///app/workspace/configs/n2v3d.yaml:0:0-0:0)
- **pinn** — [configs/pinn.yaml](cci:7://file:///app/workspace/configs/pinn.yaml:0:0-0:0)
- **quick_6gb** — small-VRAM smoke-test preset

Inspect the resolved config (applies defaults) before running:

```bash
uv run cidc show workspace/configs/deepinterp.yaml
```

# Train

```bash
uv run cidc train workspace/configs/deepinterp.yaml \
    --data workspace/data \
    --out runs/deepinterp_v1
```

- `--data` points to a directory containing `train/` and `val/` subdirs with the TIFFs (`A1.tif`, …, `F0.tif`, …). The stack names to use are read from `data.train_stacks` / `data.val_stacks` in the YAML.
- `--out` gets checkpoints (`ckpt_*.pt`, `best.pt`), a `.log` and a `.jsonl` via `RunLogger`.
- Optional overrides without editing YAML: `--override-lr`, `--override-epochs`, `--override-batch`, `--override-grad-accum` (`@/app/workspace/src/cidc/cli.py:105-112`).

Swap models by swapping the config path — every other flag stays the same. For a small GPU try [configs/quick_6gb.yaml](cci:7://file:///app/workspace/configs/quick_6gb.yaml:0:0-0:0) first.

# Inference on one stack

```bash
uv run cidc infer workspace/configs/deepinterp.yaml \
    --ckpt runs/deepinterp_v1/best.pt \
    --noisy workspace/data/val/F1.tif \
    --out   runs/deepinterp_v1/preds/F1_denoised.tif \
    --ref   workspace/data/val/F0.tif        # optional, computes stSNR
```

Notes (`@/app/workspace/src/cidc/cli.py:122-166`):

- Noise level is auto-detected from the filename (`F1`→level 1, `F2`→2, `F3`→3, training stacks likewise) via `identify_noise_level`. Unknown names fall back to level 1.
- Tiling uses `inference.tile` / `inference.overlap` from the YAML.
- EMA weights are used by default; pass `--no-ema` for raw weights.
- `--device auto|cpu|cuda|cuda:0`.

# Score only

Compare a pre-computed prediction to ground truth:

```bash
uv run cidc score preds/F1_denoised.tif workspace/data/val/F0.tif
```

Or run inference + score in one shot:

```bash
uv run cidc score \
    --config workspace/configs/deepinterp.yaml \
    --ckpt   runs/deepinterp_v1/best.pt \
    --noisy  workspace/data/val/F1.tif \
    --ref    workspace/data/val/F0.tif
```

# Benchmark (60-min / 16 GB budget check)

```bash
uv run cidc bench workspace/configs/deepinterp.yaml \
    --ckpt runs/deepinterp_v1/best.pt \
    --noisy workspace/data/val/F1.tif \
    --ref   workspace/data/val/F0.tif \
    --out   runs/deepinterp_v1/bench
```

Logs wall-clock, per-frame time, peak VRAM, and `budget_60m_ok`.

# Compare many models

```bash
uv run cidc eval-all \
    --configs workspace/configs/deepinterp.yaml workspace/configs/deepcad.yaml workspace/configs/mamba3d.yaml \
    --ckpts   runs/deepinterp_v1/best.pt runs/deepcad_v1/best.pt runs/mamba3d_v1/best.pt \
    --noisy   workspace/data/val/F1.tif workspace/data/val/F2.tif workspace/data/val/F3.tif \
    --ref     workspace/data/val/F0.tif \
    --out     runs/eval_all
```

Prints a Markdown table of sSNR/tSNR/stSNR + wall + VRAM per (model × stack) and writes `eval_all.jsonl`.

# Caveats per model

- **mamba3d** needs the Mamba CUDA kernels built; see `md files/mamba_topic.md` for the install recipe. The CLI dispatch itself is unchanged.
- **pinn** returns a dict with a `"denoised"` key internally; this is handled transparently in `cli.py:233`.
- **n2v3d** is self-supervised — no clean target needed, but `val_stacks` in the YAML still drives validation metrics vs F0.








# PINN is a wrapper, not a standalone model

Looking at `@/app/workspace/src/cidc/models/pinn/model.py:56-88`, [PINNWrapper](cci:2://file:///app/workspace/src/cidc/models/pinn/model.py:55:0-156:57) is a **composable head that wraps one of the 3-D backbones**: [n2v3d](cci:9://file:///app/workspace/src/cidc/models/n2v3d:0:0-0:0), [deepcad](cci:9://file:///app/workspace/src/cidc/models/deepcad:0:0-0:0), or [mamba3d](cci:9://file:///app/workspace/src/cidc/models/mamba3d:0:0-0:0). The 2-D [deepinterp](cci:9://file:///app/workspace/src/cidc/models/deepinterp:0:0-0:0) backbone is explicitly rejected because the PINN head needs a T axis.

So there are really **4 denoiser backbones** plus an **optional PINN head** that bolts onto any of the 3-D ones.

# How it's configured

In `@/app/workspace/configs/pinn.yaml:6-24`:

```yaml
model:
  name: pinn
  kwargs:
    backbone:
      name: mamba3d        # or n2v3d / deepcad
      kwargs: { ... backbone's own kwargs ... }
    tau_range: [5.0, 200.0]
    baseline_from: head
```

The shipped [configs/pinn.yaml](cci:7://file:///app/workspace/configs/pinn.yaml:0:0-0:0) defaults to **mamba3d + PINN head** (hence `name: pinn_mamba3d_v1`). To get PINN-on-deepcad or PINN-on-n2v3d, copy the file and change `model.kwargs.backbone.name` + its `kwargs`.

# What PINN adds on top

[forward()](cci:1://file:///app/workspace/src/cidc/models/pinn/model.py:120:4-153:9) returns a [PINNOutput](cci:2://file:///app/workspace/src/cidc/models/pinn/model.py:37:0-44:42) dict with five tensors (`@/app/workspace/src/cidc/models/pinn/model.py:121-154`):

- `denoised` — same as the bare backbone would produce
- `tau`, `baseline`, `source` — per-pixel calcium-kinetics parameters
- `reconstruction` — ODE forward-Euler rollout of those parameters

Training uses the normal Poisson-Gaussian NLL on `denoised` **plus** an auxiliary kinetics regulariser enabled by `loss.aux.pinn` in the YAML.

# CLI usage is identical

```bash
uv run cidc train workspace/configs/pinn.yaml --data workspace/data --out runs/pinn_mamba3d_v1
uv run cidc infer workspace/configs/pinn.yaml --ckpt runs/pinn_mamba3d_v1/best.pt --noisy ... --out ...
```

At inference the CLI already handles the dict return — see `@/app/workspace/src/cidc/cli.py:233`: it extracts `["denoised"]` when `cfg.model.name == "pinn"`.

# Short answer

PINN ≠ a separate architecture. It's a **physics-informed head** that sits on top of [mamba3d](cci:9://file:///app/workspace/src/cidc/models/mamba3d:0:0-0:0) (recommended), [deepcad](cci:9://file:///app/workspace/src/cidc/models/deepcad:0:0-0:0), or [n2v3d](cci:9://file:///app/workspace/src/cidc/models/n2v3d:0:0-0:0). The default config pairs it with [mamba3d](cci:9://file:///app/workspace/src/cidc/models/mamba3d:0:0-0:0), but you can swap the backbone freely.