# CIDC25 — Findings Summary

A self-contained snapshot of everything we've learned about the dataset
and everything that follows from it. This is the document to read after
`cidc25_context.md` (what the challenge asks) and `concepts.md`
(imaging / noise / denoising vocabulary).

---

## 1. Data layout

8 TIFF stacks, all `[1500, 490, 490]` `int16`, ≈ 720 MB each.

| split | files | role |
|---|---|---|
| **train** | `A1, B1` (level 1)  ·  `C2, D2` (level 2) | weight updates only |
| **val**   | `F0` (clean GT) · `F1, F2, F3` (levels 1, 2, 3) | inspection + early stopping only |

`F1, F2, F3` are **three independent noisy realisations of the same
clean scene `F0`** (star topology, not chain — proven via cross-frame
Pearson correlation; see finding 3 below).
Training stacks are *different* scenes from F0.

We **don't know the true `fps`** from the files. 1500 frames implies:

| if fps = | total duration |
|----------|----------------|
| 10 Hz    | 150 s (~2.5 min) |
| 15 Hz    | 100 s (~1.7 min) |
| 30 Hz    | 50 s   |
| 60 Hz    | 25 s   |
| 150 Hz   | 10 s   |

Any "seconds" numbers below assume **30 Hz** as a reasonable default
for two-photon calcium imaging.

---

## 2. Noise model (verified)

Pure **Poisson-Gaussian** (photon shot noise + Gaussian read noise):

```
y = g · Poisson(λ) + N(0, σ_r²)        with  λ = clean / g
Var[y] = g · E[y] + σ_r²
```

Three discrete levels, geometric ladder:

| level | gain `g` | σ_r² | files | SNR (from ACF[1]) |
|---|---|---|---|---|
| 1 | ≈ 28  | ≈ 2 500 | A1, B1, F1 | ~ −14 dB |
| 2 | ≈ 249 | ≈ 2 700 | C2, D2, F2 | ~ −21 dB |
| **3 — OOD** | **≈ 991** | ≈ 3 700 | **F3 only (Task 2)** | **~ −24 dB** |

**Evidence:**

- `R² ≥ 0.92` on the `Var = g · Mean + σ_r²` line fit for every noisy
  stack (R² ≈ 0.09 on clean F0 — the fit is nonsense, which is itself
  the confirmation F0 is clean).
- `read_var ≈ 2 500` ADU² is nearly constant across stacks → single
  physical sensor.
- `mean(F_k − F_0) ≈ 0` for every k → additive, zero-mean noise. No
  offset to fit, no bias to worry about.
- `cidc.noise.sample_poisson_gaussian` reproduces real `F_k − F_0`
  variance-vs-intensity slopes to within 4 % on all three levels.

---

## 3. Structural finding — F0 is the clean of *all* F_k

Computed per-frame Pearson correlation between pairs of validation stacks:

```
corr(F0, F1) = 0.74     corr(F1, F2) = 0.30
corr(F0, F2) = 0.40     corr(F1, F3) = 0.16
corr(F0, F3) = 0.22     corr(F2, F3) = 0.09
```

**Key tell:** `corr(F0, F2) = 0.40 > corr(F1, F2) = 0.30`.
If the structure were a chain (`F0 → F1 → F2 → F3`), then `F1` would
be *closer* to `F2` than `F0` is. The data shows the opposite: `F0`
is always the best predictor. Together with zero-mean residuals, this
confirms the **star**:

```
       F0 (clean scene)
            │
  ┌─────────┼─────────┐
  ▼         ▼         ▼
 F1        F2        F3
 +L1       +L2       +L3 (OOD)
```

---

## 4. Signal structure

### Spatial

- Bright pixel fraction ≈ 0.3–0.9 %.
- ≈ 120–270 neurons per 490×490 frame.
- Neuron radius ≈ 2–3 px (diameter 4–6 px).
- F3 has ~20 % smaller apparent blob radius than F0 — noise is already
  eating into fine spatial detail; some small neurons may be
  genuinely unrecoverable.

### Temporal

- F0 (clean): `ACF[1] = 0.995`, `ACF[10] = 0.917`, `ACF[30] = 0.665`,
  half-decay `τ(0.5) = 45 frames` (≈ 1.5 s at 30 Hz → calcium
  transient kinetics).
- F1/F2/F3: `ACF[k] ≈ 0` for all `k ≥ 1`. **Not** because there's no
  signal — because noise variance dominates total variance by
  ~25–250×.
- SNR estimate from `ACF[1] / (1 − ACF[1])`: level 1 ≈ −14 dB, level 2
  ≈ −21 dB, level 3 ≈ −24 dB. Signal at level 3 is 0.4 % of total power.

---

## 5. Anscombe variance-stabilising transform

### Why we need it

Poisson-Gaussian noise has *intensity-dependent* variance: bright
pixels have ~250× more noise variance than dim pixels at level 2.
MSE on raw ADU therefore weights bright-pixel errors ~250× more →
dim neurons become invisible to the loss.

### Forward

```
z(y) = (2/g) · sqrt( g·y + (3/8)·g² + σ_r² )
```

Makes `Var[z] ≈ 1` independent of intensity. Any Gaussian denoiser
applies.

### Inverse (Mäkitalo-Foi, 2011)

```
ŷ = g [ (z/2)² + √(3/2)/(4z) - 11/(8z²) + 5√(3/2)/(8z³) - 1/8 - σ_r²/g² ]
```

**Bug history:** we initially coded the `1/z` coefficient as
`√(3/2)/2` (off by 2×) — round-trip showed a bias of `~gain × 0.3`
ADU. After fixing to `√(3/2)/4`, residual bias dropped to ~7 / 38 / 46
ADU at levels 1/2/3.

### Caveat

At level 3, per-pixel Poisson rate is < 1 photon and the VST can't
fully stabilise. Observed `Var[z]`:

| level | bright-pixel Var[z] | dim-pixel Var[z] |
|---|---|---|
| 1 | ~1.20 | ~1.31 |
| 2 | ~1.01 | ~0.81 |
| 3 | ~0.60 | ~0.28 |

Good preprocessing for levels 1 and 2, partial for level 3; a learned
denoiser adapts to the residual heteroscedasticity.

---

## 6. Modelling plan that falls out of the data

1. **Normalise** `int16 → float32 / 10 000`. Background ≈ 0.
2. **Apply Anscombe VST** with measured `(g, σ_r²)`. Train in z-space.
3. **Architecture:** temporal U-Net (DeepInterpolation / DeepCAD
   style). Predict frame `t` from `{t±1, …, t±k}`. Self-supervised —
   no clean target needed. Works because signal is temporally
   correlated and noise is independent across frames: the network
   cannot reconstruct the noise from neighbours, so MSE forces it to
   emit the denoised signal.
4. **Task 1 (levels 1, 2):** train on A1/B1/C2/D2 together.
5. **Task 2 (OOD level 3):** augment by resampling Poisson-Gaussian
   noise on an approximately-clean estimate of training data, with
   log-uniform `g ∈ [20, 2000]`. Never train on F3 itself. Level 3
   becomes in-distribution by construction.
6. **Inverse Anscombe** on output → ADU.
7. **Metrics:** PSNR, SSIM, Pearson trace correlation vs F0.
8. **Deployment:** tiled spatial inference (128×128 with 16 px
   overlap) + temporal windowing. Target: T4 16 GB / 60 min.

---

## 7. What would fail and why

| Approach | Why it fails |
|---|---|
| Single-frame CNN (BM3D, DnCNN, plain U-Net) | Throws away the strongest signal we have — temporal correlation. At −24 dB per-frame SNR, one frame is not enough. |
| Train without Anscombe | Bright-pixel error dominates loss by ~250× at level 2; dim neurons never learn. |
| Use F0 (val) as a training target | Disqualified by challenge rule 3. Also overfits one specific scene. |
| Train only at observed gains (28, 249) | Catastrophic on F3 (g ≈ 991, ~4× OOD). |
| Oversized receptive field (256+ px) | Wastes capacity; neurons are 4–6 px. Risks blurring small neurons. |
| Ignore inference budget | T4 16 GB / 60 min → ~2.4 s/frame. Naive 3-D U-Net at full res OOMs. Need tiling. |
| Skip star-vs-chain check | Would have sampled augmentation wrong and validated wrong. |
| Trust F3 for noise fit during training | Violates the no-val-for-weights line. Continuous gain augmentation is the legal route. |

---

## 8. The genuine hard parts

1. **−24 dB SNR on F3.** Signal = 0.4 % of total power. There is an
   information-theoretic floor.
2. **OOD gain factor of 4×.** Augmentation should cover it, but
   extrapolation carries risk; the real-world physical noise at
   g ≈ 991 has never been *generated* during training, only
   *simulated*.
3. **F3 spatial detail is already gone.** Blob radius shrinks ~20 %.
   Fine structure that's below the noise floor cannot be recovered.
4. **Inference budget.** 60 min on T4 is tight for any serious 3-D
   temporal model. Profile early.
5. **No clean pairs.** Only self-supervised training is legal
   (DeepInterpolation / Noise2Noise / Noise2Void family).

---

## 9. Next tasks

- `cidc.eval` — PSNR, SSIM, Pearson trace-correlation against F0.
- `cidc.baselines` — temporal mean, rolling median (floor the learned
  model must beat).
- `cidc.models.deepinterp` — first temporal U-Net + training script.
- Gain-augmentation pipeline for Task 2.
- Inference profiling on a fake `[1500, 490, 490]` stack before
  architecture grows.
