# CIDC25 — project context & measured findings

Tags: `cidc25`, `calcium_imaging`, `denoising`, `poisson_gaussian`,
`dataset`, `project_context`.

**Project:** AI4Life Calcium Imaging Denoising Challenge 2025 (CIDC25).
**URL:** <https://ai4life-cidc25.grand-challenge.org/>

This file is the compact "cheat-sheet" version of `concepts.md` — keep it
up to date so anyone (or any LLM) jumping into the project has the exact
measured parameters and ground rules on hand.

## Data

- **Training** (Zenodo 15799507): `A1.tif`, `B1.tif` (noise level 1),
  `C2.tif`, `D2.tif` (noise level 2). Shape `[1500, 490, 490]`, `int16`.
- **Validation** (Zenodo 15807610): `F0.tif` (**CLEAN GT**), `F1.tif`
  (level 1), `F2.tif` (level 2), `F3.tif` (**OOD** noise level 3 —
  Task 2 only).
- All 8 stacks same shape, same structure, `int16`, ~720 MB each.

## Measured noise model

From `workspace/scripts/eda_numbers.py`.
Poisson-Gaussian: `Var[y] = gain · Mean[y] + read_var`.

Three discrete noise levels:

| level | gain | read_var | files |
|-------|------|----------|-------|
| 1 | ≈ 28.4  | ≈ 2 490 | A1, B1, F1 |
| 2 | ≈ 248.7 | ≈ 2 700 | C2, D2, F2 |
| 3 | ≈ 990.5 | ≈ 3 730 | F3 only (OOD, Task 2) |

Geometric ladder, factor 9× then 4×. R² ≥ 0.92 on all noisy fits.
`mean(F_k − F_0) ≈ 0` → purely additive zero-mean noise. `F0` is
genuinely clean (`min = 0` exactly).

## F0-Fk correspondence (verified)

Per-frame Pearson correlation
(`workspace/scripts/verify_pairs.py`):

```
corr(F0, F1) = 0.74     corr(F1, F2) = 0.30
corr(F0, F2) = 0.40     corr(F1, F3) = 0.16
corr(F0, F3) = 0.22     corr(F2, F3) = 0.09
```

Monotone decrease with noise level, contrasted against
`corr(F0, {A1,B1,C2,D2}) ≤ 0.20`, confirms that `F0` is the clean
underlying signal of `F1/F2/F3` and is unrelated to the training
samples.

## Signal-to-noise regime

From `ACF[1] / (1 − ACF[1])` on raw noisy stacks:

- Level 1: ≈ −14 dB
- Level 2: ≈ −21 dB
- Level 3: ≈ −24 dB (signal is 0.4 % of total power — extremely hard)

## Temporal structure (from F0)

Calcium transients persist ~45 frames at half-amplitude
(`ACF[1]=0.995`, `ACF[30]=0.665`, `τ(0.5)=45`). Strong temporal
correlation in the signal; noise is independent between frames. This
justifies temporal-prediction denoisers (DeepInterpolation, DeepCAD).

## Spatial structure

Neurons are sparse (0.3–0.9 % bright fraction), small (radius 2–3 px,
diameter 4–6 px). Receptive field 20–40 px sufficient.

## Modelling strategy

1. Normalise `int16` → `float32 / 10000`.
2. Anscombe VST on input using measured `(gain, read_var)` per stack.
3. DeepInterpolation / DeepCAD-style temporal U-Net.
4. **Task 2 (OOD)**: bootstrap an approximate-clean estimate, then
   resample Poisson-Gaussian noise at log-uniform gain in `[20, 2000]`
   for augmentation. Implemented in `cidc.noise.sample_poisson_gaussian`.
5. Metrics vs `F0`: PSNR, SSIM, Pearson trace correlation.

## Challenge-rule compliance

1. Train **only** on `A1, B1, C2, D2`. No external clean data.
2. Validation (`F0, F1, F2, F3`) is for visual inspection + model
   selection (early stopping, HP tuning) — **never weight updates**.
3. No data that would leak a clean ground truth may be used for
   training, including `F0`.
4. Any augmentation / processing / unsupervised technique allowed.
5. Code must be public + reproducible at deadline; declare exact data
   and pretrained weights used.

### How we use `F0`

We used `F0` only in `scripts/eda_numbers.py` and
`scripts/verify_noise_model.py` to *measure* noise parameters. No model
weights see `F0`. The training-time noise model is derivable from
training data alone (self-fit on A1/B1/C2/D2 gives the same level-1/2
gains within ~1 %). **We do not hard-code F3's gain (990) into the
training loop** — instead, log-uniform gain augmentation over
`[20, 2000]` covers F3 without "knowing" it.

## Constraints

- Submission: Docker container on grand-challenge.
- Runtime: 60 min per video max on **T4 16 GB** / 32 GB RAM ⇒ ~2.4 s/frame.
- Implement tiled spatial inference (e.g. 128×128 with overlap) +
  temporal windowing.

## Code layout

```
/app/
  pyproject.toml                     # hatchling build, installs `cidc`
  workspace/
    src/cidc/                        # importable package
      io.py       stats.py  noise.py
    notebooks/
      01_eda.py                      # marimo EDA
    scripts/
      download_data.py               # Zenodo fetch, md5-verified
      eda_numbers.py                 # decision-driving scalars
      verify_noise_model.py          # Anscombe round-trip + sampler check
      verify_pairs.py                # F0-Fk correspondence check
    docs/
      concepts.md                    # full writeup
      cidc25_context.md              # (this file)
      marimo_conventions.md
    data/
      train/ A1.tif B1.tif C2.tif D2.tif
      val/   F0.tif F1.tif F2.tif F3.tif
```

## Anscombe inverse caveat

The Mäkitalo-Foi exact inverse has residual bias of ~+7 / +38 / +46 ADU
at levels 1 / 2 / 3 due to the low-count regime. **Not a bug** — inherent
to the transform. Learned denoisers absorb it; for classical baselines,
apply a constant offset correction.
