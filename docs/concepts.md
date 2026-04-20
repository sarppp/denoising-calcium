# AI4Life-CIDC25 — Concepts

Source: https://ai4life-cidc25.grand-challenge.org/tasks-and-datasets/

## Challenge in one line
Unsupervised denoising of calcium imaging microscopy videos — recover the clean
calcium signal while preserving both **spatial** and **temporal** dynamics.

## Why calcium imaging?
Calcium imaging records fluorescence changes that proxy neuronal activity over
time. Data are 3D stacks `[T, H, W]` (time × height × width). Signal is sparse,
temporally correlated (calcium transients decay slowly), and typically drowned
in shot/read noise, especially at high frame rates or low laser power.

## Why unsupervised?
Clean ground truth is (almost) never available for real recordings. Methods
must learn from noisy data alone. External clean data is forbidden for
training.

## Tasks
- **Task 1 — Content generalisation:** test on *unseen samples*, noise levels
  similar to training. Two video stacks.
- **Task 2 — Noise-level generalisation:** test on unseen samples **and** an
  out-of-distribution (lower SNR) noise level. Separate leaderboard.

## Data
Shape: `[1500, 490, 490]` per TIFF, `float`/`uint16` (to verify on load).

### Training (Zenodo 15799507)
- `A1.tif`, `B1.tif` — noise level 1
- `C2.tif`, `D2.tif` — noise level 2
- 4 distinct in-silico samples × 2 noise intensities.

### Validation (Zenodo 15807610)
- `F0.tif` — **clean** reference (ground truth)
- `F1.tif`, `F2.tif`, `F3.tif` — noisy versions at different levels
- Validation set is **only** for visual inspection / model selection
  (early stopping, HP tuning). **Not** for weight updates.

### Test
Hidden. Evaluation runs on the platform.

## Constraints
- Container runtime per video: **60 min max**.
- Hardware: 32 GB RAM, **nVidia T4 16 GB**.
- Implement **tiled / batched** inference to fit memory.
- Must submit a Docker container with evaluation code.
- Submissions require a (manually) verified grand-challenge account.
- Preliminary phase: 5 submissions, smaller subset. Final: 2 submissions.
- Code must be public and reproducible; declare training data + any
  pretrained weights.
- Ideally submit the same method to both tasks (for joint leaderboard).

## Families of unsupervised denoisers to consider
1. **Noise2Void / Noise2Self** — blind-spot masking, per-frame 2D.
2. **Noise2Noise** — pairs of independent noisy realisations; we only have
   one noisy view per sample, so not directly applicable unless we synthesise
   pairs (e.g. temporal splits).
3. **DeepInterpolation (Lecoq et al.)** — predict frame *t* from neighbouring
   frames `t-k..t-1, t+1..t+k`; exploits temporal redundancy of calcium.
   Likely a strong baseline here.
4. **DeepCAD / DeepCAD-RT (Li et al.)** — 3D U-Net trained on self-supervised
   sub-sampled stacks. The training data author (Li, Xinyang) is the DeepCAD
   author — so this is the implicit reference baseline.
5. **Probabilistic / diffusion-based** (e.g. HDN, DivNoising) — model noise
   explicitly, useful when noise model is known.
6. **Classical baselines for sanity checks:** temporal mean, rolling mean,
   PMD (penalised matrix decomposition), NL-means, BM4D.

## Evaluation (expected)
Typical metrics for denoising challenges: PSNR, SSIM against clean GT,
plus task-specific temporal metrics (correlation of inferred traces,
preservation of transient shapes). Exact metric per leaderboard TBD —
check the challenge "Evaluation" page when the leaderboard opens.

## Measured properties of the dataset

These numbers come from running `workspace/scripts/eda_numbers.py` on all 8
stacks (A1, B1, C2, D2, F0, F1, F2, F3). They drive every downstream choice.

### Storage & dynamic range

All stacks are `int16`, shape `[1500, 490, 490]`.

| file | dtype | min | mean | max | role |
|------|-------|-----|------|-----|------|
| A1   | int16 | −217  | 164.1 | 11 400 | train, noise level 1 |
| B1   | int16 | −213  | 196.2 |  9 925 | train, noise level 1 |
| C2   | int16 | −258  | 168.2 |  7 610 | train, noise level 2 |
| D2   | int16 | −242  | 196.0 | 19 262 | train, noise level 2 |
| F0   | int16 |    0  | 204.4 | 15 076 | val, **clean GT** |
| F1   | int16 | −205  | 204.4 | 14 979 | val, noise level 1 |
| F2   | int16 | −256  | 204.3 | 17 321 | val, noise level 2 |
| F3   | int16 | −265  | 204.4 | 17 039 | val, **OOD noise (Task 2)** |

Key observations:

- Signal lives in **0..~20 000**, i.e. < 1 % of the `int16` dynamic range.
  Cast to `float32`, normalise by a fixed scale (~10 000), no information lost.
- `F0.min == 0` exactly; every noisy stack dips negative by ~200 ADU. That
  confirms the noisy data has a subtracted pedestal and dithers below zero
  on pure background — a sanity check that `F0` is genuinely the clean
  reference (no residual baseline offset).
- `mean(F_k − F_0) ≈ 0` to within 0.2 ADU for k=1,2,3 → the noise is
  **additive and zero-mean**. No gain or bias correction to worry about.

### Noise model: Poisson-Gaussian

We fit `Var[pixel] = gain · Mean[pixel] + read_var` per stack using
background-dominated pixels, then cross-validated against the
`Var(F_k − F_0)` residual method (only possible on the val split):

| file | gain (self-fit) | gain (vs F0) | read_var | R² | noise level |
|------|-----------------|--------------|----------|----|-------------|
| A1   |  28.4 | —     | 2 495 | 0.92 | **1** |
| B1   |  28.3 | —     | 2 494 | 0.94 | **1** |
| C2   | 248.8 | —     | 2 656 | 0.98 | **2** |
| D2   | 248.7 | —     | 2 743 | 0.96 | **2** |
| F0   |   1.1 | —     |   −63 | 0.09 | — (clean) |
| F1   |  28.6 |  27.1 | 2 481 | 0.92 | **1** (matches A1/B1) |
| F2   | 248.5 | 266.9 | 2 773 | 0.97 | **2** (matches C2/D2) |
| F3   | 990.5 | 1062  | 3 732 | 0.98 | **3 — OOD**, ~4× level 2 |

The two fit methods agree to within ~7 %, and `R² ≥ 0.92` on every noisy
stack ⇒ Poisson-Gaussian is **the** noise model, not an approximation.

### Three discrete noise levels

```
level 1:  gain ≈  28   (training: A1, B1 ;  val: F1)
level 2:  gain ≈ 249   (training: C2, D2 ;  val: F2)
level 3:  gain ≈ 991   (val only: F3)    ← OOD, Task 2
```

Geometric ladder, factor ~9× between levels 1 and 2, factor ~4× between
2 and 3. Read-noise variance is ~2 500 ADU² and roughly constant, consistent
with a single physical sensor.

### Signal-to-noise regime

From `ACF[1] / (1 − ACF[1])` on the raw noisy stacks (signal-power vs total):

```
level 1:  SNR ≈ −14 dB   (signal is ~3.5 % of total power)
level 2:  SNR ≈ −21 dB
level 3:  SNR ≈ −24 dB
```

Signal is 250× weaker than noise at level 3. Task 2 is *very* hard.

### Temporal structure

Temporal autocorrelation of active pixels on the **clean** `F0`:

```
acf[1]  = 0.995
acf[5]  = 0.968
acf[10] = 0.917
acf[30] = 0.665
τ(0.5)  = 45 frames
```

Calcium transients persist ~45 frames at half-amplitude, which is typical
(τ_decay ≈ 1.5 s at 30 Hz). On raw noisy stacks the ACF looks flat at 0,
but that's a noise-dominates-variance illusion — the *true* signal has rich
temporal structure. Noise is independent across frames. This is the entire
justification for temporal-prediction denoisers (DeepInterpolation, DeepCAD):
predicting frame *t* from frames `t±k` forces the network to output the
denoised signal because noise can't be predicted from neighbouring frames.

### Spatial structure

From temporal-mean blob analysis (MAD-based 5σ threshold):

```
bright fraction  ≈ 0.3 – 0.9 % of pixels
neuron count     ≈ 120 – 270 per frame (490×490)
neuron radius    ≈ 2 – 3 pixels  (so diameter 4–6 px, ~20–30 px² each)
```

Neurons are small and sparse. A denoiser receptive field of ~20–40 px is
plenty; over-sizing wastes capacity. `F3` has ~20 % smaller blob radius than
`F0`, i.e. noise is already eating into fine spatial detail.

## Challenge-rule compliance (important)

Paraphrasing the challenge page:

1. Train **only** on `A1, B1, C2, D2`. No external clean data.
2. Validation (`F0, F1, F2, F3`) is for **visual inspection** and
   **model selection** (early stopping, hyper-parameter tuning) — never
   for weight updates.
3. No data that would leak a clean ground truth may be used for training,
   including `F0`.
4. Any augmentation / processing / unsupervised technique is allowed.
5. Code must be public + reproducible at deadline; declare exact data and
   pretrained weights used.

### How we interpret "used F0 for noise measurement"

We used `F0` only in `scripts/eda_numbers.py` and `scripts/verify_noise_model.py`
to *measure* noise parameters. No model weights see `F0`. Moreover, the
self-fit on training data alone gives essentially the same level-1 and
level-2 gains (28 vs 28, 249 vs 249). So the training-time noise model is
derivable from training data only. **We do not hard-code F3's gain (990)
into the training loop** — instead, for Task 2 we use continuous
log-uniform gain augmentation over `[20, 2000]`, which covers F3 without
ever "knowing" it. This keeps us on the safe side of the rules.

### F0-Fk correspondence — verified

Per-frame Pearson correlations:
```
corr(F0, F1) = 0.74     corr(F1, F2) = 0.30
corr(F0, F2) = 0.40     corr(F1, F3) = 0.16
corr(F0, F3) = 0.22     corr(F2, F3) = 0.09
```
Monotone decay with noise level, consistent with `F_k = F0 + noise_k`.
Contrast: `corr(F0, {A1,B1,C2,D2}) ≤ 0.20`, confirming F0 is *not* the
clean of any training sample but *is* of F1/F2/F3.
Script: `workspace/scripts/verify_pairs.py`.

## Modelling decisions driven by the data

1. **Normalisation.** Read as `int16`, cast to `float32`, divide by a fixed
   scale (e.g. 10 000). Keep background near 0.
2. **Anscombe variance stabilisation.** With known per-stack `(gain, read_var)`,
   apply the generalised Anscombe transform on input. Output of the network
   is the denoised Anscombe-domain signal; invert at the end. After Anscombe
   the noise is ~`N(0, 1)`, so any plain Gaussian denoiser applies.
3. **Baseline family.** DeepInterpolation / DeepCAD — temporal self-supervision.
   Train on `A1, B1, C2, D2` jointly. Loss is MSE in Anscombe space on held-out
   central frames.
4. **Task 2 (OOD noise) strategy.** Bootstrap: run a first-pass denoiser on
   training data, treat its output as an approximate clean estimate, then
   *resample* Poisson-Gaussian noise at a continuous range of gains (e.g.
   `20..2000`, log-uniform) to generate augmented training pairs. Optionally
   condition the network on the target gain. This turns F3 into an
   in-distribution case by construction.
5. **Metrics.** PSNR + SSIM vs `F0` on `F1, F2, F3` independently. Track
   trace correlation (Pearson on pixel time-series) and MSE in Anscombe space
   as sanity metrics. Challenge metric is TBD on the evaluation page —
   revisit when it opens.
6. **Inference budget.** Container limit is 60 min on T4 16 GB for a single
   `[1500, 490, 490]` stack. Budget: ~2.4 s per frame. Tile spatially
   (e.g. 128×128 with 16 px overlap) and stream temporal windows.

## Local project plan

1. ✅ Download train + val into `workspace/data/` (md5-verified).
2. ✅ EDA: dtype, noise-level ladder, Poisson-Gaussian fit, ACF, spatial
   stats, `F_k - F_0` consistency check. Findings above.
3. ⏭ `src/cidc/noise.py`: constants, Anscombe forward/inverse, PG sampler
   for augmentation.
4. ⏭ `src/cidc/baselines.py`: temporal-mean baseline, rolling median.
5. ⏭ `src/cidc/models/deepinterp.py`: DeepInterpolation-style U-Net + trainer.
6. ⏭ `src/cidc/eval.py`: PSNR, SSIM, Pearson trace correlation on `F0` GT.
7. ⏭ `scripts/train.py`, `scripts/predict.py`.
8. ⏭ Grand-challenge Docker wrapper with tiled inference.
