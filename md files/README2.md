# AI4Life-CIDC25 — Workspace

Workspace for the [AI4Life Calcium Imaging Denoising Challenge 2025](https://ai4life-cidc25.grand-challenge.org/).

## Layout
```
workspace/
  docs/
    concepts.md           # challenge summary + method notes
  scripts/
    download_data.py      # fetch train + val TIFFs from Zenodo (md5-verified)
  data/
    train/                # A1.tif B1.tif C2.tif D2.tif  (~2.88 GB)
    val/                  # F0.tif F1.tif F2.tif F3.tif  (~2.88 GB, F0 = clean)
  src/                    # (empty) place for denoising models
```

## Quickstart
```bash
# from repo root (/app)
uv sync

# dry-run (just prints the plan)
uv run python workspace/scripts/download_data.py --dry-run

# grab validation only (~2.88 GB) — useful first sanity check
uv run python workspace/scripts/download_data.py --split val

# full download (~5.76 GB)
uv run python workspace/scripts/download_data.py
```

Re-running is idempotent: existing files are md5-checked and skipped.

## Data naming
| File    | Split | Role                |
|---------|-------|---------------------|
| `A1.tif`| train | sample A, noise lvl 1 |
| `B1.tif`| train | sample B, noise lvl 1 |
| `C2.tif`| train | sample C, noise lvl 2 |
| `D2.tif`| train | sample D, noise lvl 2 |
| `F0.tif`| val   | sample F, **clean** |
| `F1.tif`| val   | sample F, noise lvl 1 |
| `F2.tif`| val   | sample F, noise lvl 2 |
| `F3.tif`| val   | sample F, noise lvl 3 (OOD, Task 2) |

All stacks are `[1500, 490, 490]` (FxHxW).

## Rules reminder
- Validation data is for **model selection only** — never train on it.
- No external clean data.
- Submission container: T4 16 GB, 32 GB RAM, 60 min/video.

See `docs/concepts.md` for the full write-up.
