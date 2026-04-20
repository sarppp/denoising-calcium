# What changed and why

## Before — what we assumed the evaluation was

Working assumption going into the architecture discussion:

- **PSNR** (dB of per-pixel reconstruction) as the primary score.
- **SSIM** as a secondary quality metric.
- **Pearson trace correlation** as an extra "biology-aware" check.

With that assumption, the natural model choice was:

- Frame-level denoising (DeepInterpolation-style).
- Loss: MSE in Anscombe space.
- Success = good-looking frames; temporal quality treated as a
  nice-to-have check, not as a co-equal objective.

## After — what the evaluation actually is

From the challenge page:

$$
\mathrm{stSNR} = \alpha \cdot \overline{\mathrm{sSNR}} + (1 - \alpha) \cdot \overline{\mathrm{tSNR}}
$$

- **sSNR** = SNR computed *per frame* (spatial fidelity), averaged over T.
- **tSNR** = SNR computed *per pixel trace* (temporal fidelity), averaged over H×W.
- **Final score** = mean stSNR across files.

Two concrete differences from what I assumed:

1. **It's SNR (`Σ y² / Σ (y−x)²`), not PSNR (`MAX² / MSE`).**
   Direct from the challenge page formula. The denominator is the
   same residual energy, but the numerator is signal *energy*, not a
   fixed `MAX²`. Absolute numbers are not comparable to PSNR values
   you see elsewhere, so don't anchor on "30 dB is good" intuitions
   from other denoising work.
2. **Temporal quality is literally half the score.** tSNR isn't a bonus; it's weighted equally to sSNR. The `(1−α)` term means a model with perfect frames and flat traces can score at most `α · large + (1−α) · 0 = α · large` — so ~50 % of the achievable score is locked behind getting traces right.

## Why this matters — the three things it breaks in the old plan

### 1. "Optimise PSNR, check traces later" is now wrong

Under stSNR, an over-smoothing model that kills calcium transients
**loses half the score instantly**. The failure mode isn't "slightly
worse" — it's catastrophic. Any denoiser that smooths out a real
`ΔF/F = 0.3` transient over 45 frames will have tSNR that's closer to
0 than to the sSNR it achieves on background pixels.

*Consequence:* the temporal ACF measurements in
`@/app/workspace/notebooks/06_proofs.py:6` aren't just "interesting
facts about the data" — they're the **thing the metric punishes you
for getting wrong**. Preserving `τ(0.5) = 45 frames` structure in the
output is half of winning.

### 2. Architecture priority flips

**Old (PSNR-implicit) priority:**

1. Backbone (ConvNeXt / Mamba / 3-D conv)
2. Spatial loss
3. Training signal
4. Data recipe

Architecture-first. Backbone choice matters a lot for PSNR because a
~2 dB PSNR gain is hard-won on clean pairs.

**New (stSNR-actual) priority:**

1. **Data recipe** — Anscombe + log-uniform gain augmentation.
2. **Training signal** — self-supervised masking that preserves temporal
   structure (DeepInterp or 3-D N2V).
3. **Loss** — must include a temporal term, or the gradient never
   teaches the model to preserve traces.
4. **Backbone** — last ~1–2 dB.

Under stSNR, the top lever is "use temporal context at all" (why the
leaderboard shows 2-D at 6.35 and 3-D at ≥14 — an 8 dB gap just from
dimensionality). That's not true under PSNR.

### 3. The leader's choice (N2V 3D) suddenly makes sense

I initially steered you toward DeepInterpolation because I was
implicitly optimising for frame reconstruction. The leaderboard
champion uses **N2V 3D with 64³ patches, no conv bias** — a masking
scheme that treats time as just another axis of the patch. Under
stSNR, that's what you'd expect to win: dense voxel-level supervision
that forces the model to reconstruct pixels in temporal context, not
reconstruct whole frames in one shot.

DeepInterp is not *inherently* bad for tSNR — at inference it still
emits one frame at a time from `±K` temporal neighbours, and the
per-pixel output sequence does contain temporal information. But
the **training signal** is sparser along the temporal axis: one
predicted frame per window, no supervision on how individual voxel
traces should look. N2V 3D supervises hundreds of voxels per patch,
scattered through the volume, each explicitly reconstructed from a
temporal neighbourhood. That density is the edge, not a fundamental
flaw in DeepInterp. The honest framing: **N2V 3D ≥ DeepInterp on
tSNR in expectation; DeepInterp is still a useful auxiliary loss**.

## So what did our plan change to?

Practical diffs in `@/app/workspace/docs/strategy.md`:

| Plan element             | Old                                | New                                                   |
|--------------------------|------------------------------------|-------------------------------------------------------|
| Primary metric           | PSNR                               | **stSNR** (our own implementation must match)         |
| Early-stopping signal    | Validation PSNR on F1/F2           | **Validation stSNR on F1/F2**                         |
| Self-supervision scheme  | DeepInterp (drop centre frame)     | **N2V 3-D sparse masking** (primary) + DeepInterp (secondary) |
| Input shape              | `(B, 2K, H, W)` stacked channels   | **`(B, 1, D, H, W)` 3-D patch**                       |
| Loss                     | MSE (Anscombe) on whole frame      | **Masked-voxel MSE** in raw ADU (after inverse Anscombe) |
| Architecture attention   | "ConvNeXt vs Mamba vs 3-D U-Net?"  | **Data recipe first; backbone is the last 1 dB**      |
| Success criterion        | "Good-looking denoised frames"     | **Trace τ preserved AND frames clean** (both, jointly) |

## One-line version

We thought we were optimising picture quality; we're actually
optimising *picture quality + trace quality in equal measure*. That
inverts which tricks matter most and tells us to copy a voxel-masking
scheme instead of a frame-masking scheme. Everything downstream of
that follows.