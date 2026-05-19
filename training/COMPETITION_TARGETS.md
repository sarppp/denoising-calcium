# CIDC25 Challenge Targets

## Leaderboard Scores to Beat

| Task | Current Leader | Target stSNR |
|------|---|---|
| Content Generalization (F1) | 22.14 dB | **> 22.14 dB** |
| Noise Level Generalization (F2+F3) | 16.75 dB | **> 16.75 dB** |

## Your Baseline (No Denoising)

| Stack | Level | Baseline stSNR | Notes |
|-------|-------|---|---|
| F1 | 1 | 7.27 dB | Similar noise to training (A1, B1) |
| F2 | 2 | -0.79 dB | 2× harder (OOD noise level) |
| F3 | 3 | -6.64 dB | 3× harder (OOD noise level, 13.91 dB gap) |

**Goal**: Improve F1 by +14.87 dB, F2/F3 by ~24 dB via:
1. Self-supervised denoising (N2V3D masking)
2. Poisson-Gaussian NLL loss
3. Gain augmentation (LogUniform[15, 1500]) for noise generalization

## Next Steps

### 1. Test Mode (5 epochs, ~1-2 hours)
```bash
cd training
jupyter notebook train.ipynb
# Set TEST_MODE = True, run all cells, check for errors
```

### 2. Full Training (100 epochs, ~20-25 hours)
```bash
# Once TEST_MODE passes, set TEST_MODE = False and re-run
# Or use standalone script:
python train.py --epochs 100
```

### 3. Evaluate on Validation Data
```bash
python evaluate.py --model checkpoints/model_final.pt
# Output: stSNR scores for F1, F2, F3
```

### 4. Submit to Leaderboard
- Container: Dockerfile wrapping inference on (noisy_stack) → (denoised_stack)
- Test on both "Content Generalization" (F1 variant) and "Noise Level Generalization" (F2/F3 variants)

## Configuration Locked (From Notebooks 01-10)

**Noise Parameters:**
- A1: g=27.6, σ_r²=2490
- B1: g=27.7, σ_r²=2490
- C2: g=254.9, σ_r²=2700
- D2: g=258.9, σ_r²=2700

**Architecture & Training:**
- Patch size: 128×128×128
- Batch size: 4
- LR: 1e-4
- Epochs: 100
- Loss: Poisson-Gaussian NLL
- Masking: N2V3D (0.5%)
- Gain aug: LogUniform(15, 1500)
- Mixed precision: Enabled

## Metrics

- **stSNR** (Spatio-Temporal SNR): Primary metric
  - α=0.5 × sSNR + (1-α) × tSNR
  - sSNR: SNR per frame, averaged over time
  - tSNR: SNR per voxel, averaged over space
