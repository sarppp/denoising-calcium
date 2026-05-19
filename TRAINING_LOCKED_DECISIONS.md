# Training Locked Decisions

**Status:** Ready after Quick Fixes A & B  
**Updated:** 2026-04-25  
**Basis:** Notebooks 01-10 + user diagnosis

## Architecture & Sampling

| Decision | Value | Status | Source |
|----------|-------|--------|--------|
| **Patch depth T** | 128 frames | ✅ Locked | NB01: τ₀.₅=46 frames → T=2×τ |
| **Patch size** | [128, 128, 128] | ✅ Locked | T×H×W coverage |
| **Sampling method** | Random | ✅ Locked | NB09: 100% patches contain signal |
| **Masking** | N2V3D blind-spot | ✅ Locked | NB02, NB06: temporal > spatial |

## Noise & Calibration

| Decision | Value | Status | Notes |
|----------|-------|--------|-------|
| **Loss function** | Poisson-Gaussian NLL | ✅ Locked | Theory correct, empirical test in NB04 was setup issue |
| **Noise model (Levels 1-2)** | variance = g × mean + σ² | 🔧 Fix required | Use **frame differencing** (Quick Fix A) |
| **Background selection** | Bottom 10% mean intensity | 🔧 Fix required | Use **mean-based** not variance-based (Quick Fix B) |

## Training Augmentation

| Decision | Value | Status | Notes |
|----------|-------|--------|-------|
| **Gain augmentation range** | LogUniform(15, 1500) | ✅ Locked | Covers ±3× safety margin around observed gains |
| **Gain sampling** | Per-patch during training | ✅ Locked | NB05: 14+ dB loss if gain mismatched |
| **Gradient clipping** | max_norm=1.0 | ✅ Locked | Stabilizes loss landscape |
| **Noise as input** | Noise map channel | ✅ Locked | Informs decoder about local SNR |

## Performance Baseline (No Denoising)

| Stack | Level | Gain | stSNR (dB) | sSNR (dB) | tSNR (dB) | Notes |
|-------|-------|------|-----------|-----------|-----------|-------|
| F1 | 1 | 28.4 | **7.27** | 8.10 | 6.45 | Distribution match |
| F2 | 2 | 248.7 | **-0.79** | -0.07 | -1.52 | 2× harder |
| F3 | 3 | 990.5 | **-6.64** | -5.94 | -7.34 | OOD, 4× harder |

- Model must beat F1: +7.27 dB floor
- F3 is **13.91 dB harder** than F1 (generalization test)
- Temporal metric (tSNR) is **1-2 dB harder** than spatial (sSNR)

## Memory Implications (T=128)

**Per-patch memory:**
- Forward: 2 × 128³ × 4 bytes = 536 MB (single sample)
- With activations (backprop): ~2 GB per sample

**Recommended by VRAM:**
- 6 GB: Batch=1 only (risky), must use mixed precision + gradient checkpointing
- 16 GB: Batch=4-6 (comfortable), must use mixed precision
- 40 GB: Batch=16+ (ideal), standard training

**Required for 16 GB:**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In encoder: wrap each ConvNeXt3D block with gradient checkpointing
from torch.utils.checkpoint import checkpoint
output = checkpoint(block, input, use_reentrant=False)
# ~40% memory savings, ~30% speed cost
```

## Quick Fixes Required (Before Training)

### Fix A: Level 1 Noise Model (Frame Differencing)
**File:** `03_noise_model/quick_fix_A_frame_diff.py`  
**Issue:** R²=0.23-0.30 at low gain (signal contaminates variance estimate)  
**Solution:** Frame differencing removes signal: `Var[y_t - y_{t-1}] = 2 × noise_var`  
**Expected:** R² > 0.85 on A1 and B1  
**Time:** ~30 min

### Fix B: Validation Calibration (Mean-Intensity Background)
**File:** `10_noise_model_calibration/quick_fix_B_mean_intensity_bg.py`  
**Issue:** Temporal variance selects saturated pixels on F0, gives negative gains  
**Solution:** Use low mean intensity (dark pixels = background), not low variance  
**Expected:** All F0-F3 have positive gains, R² > 0.5  
**Time:** ~30 min

**Total time for fixes:** 1-2 hours

---

## Summary Before Training

✅ **Measurement-grounded:**
- Baseline floors from NB01 (τ₀.₅, stSNR)
- Temporal penalty from NB02 (6 dB for spatial-only)
- Patch depth from NB01 (T=128)
- Gain robustness from NB05 (14+ dB loss if wrong)
- Sampling validity from NB09 (100% signal)

✅ **Theory-backed:**
- Poisson-Gaussian NLL (physics of sensor noise)
- Frame differencing (removes signal, isolates noise)
- Mean-intensity background (works on clean & noisy)

🔧 **Awaiting fixes:**
- Level 1 noise model (frame differencing)
- Validation calibration (mean-based background)

**Next:** Run Quick Fixes A & B today → Start training tomorrow.
