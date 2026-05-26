# Modal GPU Training

Trains CIDC25 on a cloud GPU. Data lives in a Modal Volume; outputs download automatically when training finishes.

## Setup (once)

```bash
uv sync    # picks up modal from pyproject.toml
```

Credentials are in `modal/.env` (already set).

## Workflow

```bash
# 1. Upload data to Modal (once — 5.5 GB, ~5 min)
uv run --env-file modal/.env modal run modal/upload_data.py

# 2. Launch training — outputs download automatically when done
uv run --env-file modal/.env modal run modal/app.py
```

Outputs land in `runs/full_training/` (best.pt, last.pt, train_n2v3d_large.jsonl).

## GPU options

Edit `GPU_TYPE` in `modal/app.py`:

| GPU | $/h | Est. time | Total |
|---|---|---|---|
| `"t4"` | $0.59 | ~12h | $7.08 |
| `"l40s"` | $1.95 | ~5h | $9.75 |
| `"a100-80gb"` | $2.50 | ~3h | **$7.50** ← default |
| `"h100"` | $3.95 | ~2h | $7.90 |
| `"h200"` | $4.54 | ~1.5h | $6.81 |

> If you switch to a GPU other than L40S or A100-80GB, you may need to adjust
> `batch_size` / `grad_accum` in `configs/n2v3d_large.yaml` — see NEXT_STEPS.md GPU guide.

## Commands

```bash
# Quick pipeline check — 4 batches only (~2 min, free sanity check before paying for full run)
uv run --env-file modal/.env modal run modal/app.py --probe

# Start fresh (ignore existing last.pt checkpoint)
uv run --env-file modal/.env modal run modal/app.py --no-resume

# Different config or run name
uv run --env-file modal/.env modal run modal/app.py --config n2v3d.yaml --run-name test_base

# Re-upload data (if TIF files changed)
uv run --env-file modal/.env modal run modal/upload_data.py --force
```

## Monitor a running job

If you need to detach and re-attach to logs:

```bash
uv run --env-file modal/.env modal app logs cidc25-training
```

## Volume layout

| Volume | Mount in container | Contents |
|---|---|---|
| `cidc-data` | `/data` | `train/` A1–D2.tif, `val/` F0–F3.tif |
| `cidc-runs` | `/runs` | `full_training/` best.pt, last.pt, *.jsonl |

## Crash / resume

If the job crashes, re-run the same command. Training auto-resumes from `last.pt`:

```bash
modal run modal/app.py
```
