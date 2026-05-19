# Training Pipeline

Self-contained 3D denoising pipeline. All hyperparameters locked from notebook measurements (01–10). Every run is fully observable, resumable, and LLM-readable.

## Directory structure

```
training/
├── config.py           Locked parameters — 3 sections (data / model / train)
├── model.py            3D U-Net
├── loss.py             Poisson-Gaussian NLL (reduction='mean' and 'none')
├── dataset.py          PatchDataset with LogUniform gain augmentation
├── evaluate.py         Vectorised stSNR — fast and full modes
├── training_utils.py   ExperimentDir, MetricsLogger, EarlyStopping, probe
├── train.py            Main training script
├── quick_eval.py       Fast sanity-check evaluation (F1 only, subsampled)
└── runs/               All experiment outputs (auto-created)
    └── run_YYYYMMDD_HHMMSS_<name>/
        ├── config.json     full parameter snapshot at launch
        ├── metrics.jsonl   one JSON record per epoch
        ├── train.log       full text log
        └── checkpoints/
            ├── model_epoch_NNN.pt
            ├── model_best.pt   best combined val score
            └── model_final.pt  last epoch
```

## Quick start

```bash
cd training

# 1. Sanity check — 4 batches, no training (< 30 s)
python train.py --probe-only

# 2. End-to-end test — 2 epochs, tiny patches (< 5 min)
python train.py --quick

# 3. Full training
python train.py --run-name baseline

# 4. Resume after interruption
python train.py --resume runs/run_20260519_143022_baseline/

# 5. Check a checkpoint quickly (F1 only, fast mode)
python quick_eval.py --model runs/<run>/checkpoints/model_best.pt

# 6. Full evaluation for submission
python evaluate.py --model runs/<run>/checkpoints/model_best.pt
```

## Config sections

`config.py` is divided into three sections with explicit purpose:

| Section | Contents | Should you change it? |
|---------|----------|-----------------------|
| **DATA** | Noise params, baselines, patch size, gain aug range | No — measured from notebooks |
| **MODEL** | Channels, in/out | Yes — for ablations |
| **TRAIN** | LR, epochs, schedule, batch size | Yes — for tuning |

## What was fixed vs. the original

| Bug | Location | Fix |
|-----|----------|-----|
| Linear uniform gain aug | `dataset.py` | LogUniform — equal probability per decade |
| Noise map used pre-rescaled signal | `dataset.py` | Now uses gain-rescaled signal |
| tSNR was O(H×W) Python loop | `evaluate.py` | Fully vectorised `np.var` over axis=0 |
| Hardcoded `g=100` at inference | `evaluate.py` | Uses `G_INFER = sqrt(G_MIN × G_MAX) ≈ 150` |
| `Adam` → no weight decay | `train.py` | `AdamW` with `WEIGHT_DECAY=1e-4` |
| Constant LR | `train.py` | Linear warmup + cosine decay to `LR_MIN` |
| No per-stack loss visibility | `train.py` | `loss_A1`, `loss_B1`, ... logged each epoch |
| No structured metrics | `train.py` | `metrics.jsonl` — one JSON per epoch |
| No experiment isolation | `train.py` | `runs/run_<ts>_<name>/` per run |
| No probe gate | `train.py` | 4-batch probe before full training |

## Reading metrics

```python
import json, pandas as pd

records = [json.loads(l) for l in open("runs/<run>/metrics.jsonl")]
df = pd.DataFrame(records)
df[['epoch', 'loss', 'loss_A1', 'loss_C2', 'val_combined']].tail(20)
```

## Locked parameters (from notebooks)

| Parameter | Value | Source |
|-----------|-------|--------|
| Patch depth T | 128 | NB01: τ₀.₅=46 → 2×46=92 → next power of 2 |
| Patch H, W | 128 | NB09: patch sampling experiments |
| Mask ratio | 0.5 % | NB06: single-voxel blind-spot |
| Gain aug | LogUniform[15, 1500] | NB10: 3× safety margin for OOD |
| Loss | Poisson-Gaussian NLL | NB03/NB04 |
| Sampling | Random | NB09: 100 % active patches |
| Architecture | 3D voxel-level | NB02: temporal smoothing destroys tSNR |

## Noise parameters (nb03)

| Stack | g | σ_r² | R² | Method |
|-------|---|------|----|--------|
| A1 | 27.6 | 2490 | 0.70 | frame-differencing |
| B1 | 27.7 | 2490 | 0.73 | frame-differencing |
| C2 | 250.96 | 2700 | 0.97 | mean-intensity |
| D2 | 258.9 | 2700 | 0.91 | mean-intensity |

## Baseline floors (nb01) — model must beat all three

| Stack | stSNR floor | Notes |
|-------|-------------|-------|
| F1 | +7.27 dB | Same noise level as training |
| F2 | −0.79 dB | 2× harder — OOD noise level |
| F3 | −6.64 dB | 3× harder — extreme noise, 13.91 dB gap vs F1 |

## Hardware

- VRAM: 16 GB minimum for 128³ patches, batch=4
- Mixed precision: disabled (BatchNorm + AMP unstable; revisit with GroupNorm)
- Estimated runtime: ~20–25 h on a single A100 for 100 epochs
