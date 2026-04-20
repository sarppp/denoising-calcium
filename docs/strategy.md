# Strategy — evaluation metric, leaderboard, and architecture plan

Companion to `findings_summary.md` (data) and `concepts.md` (vocabulary).
This doc is **how we plan to win**, not what we measured. If a claim
here is numeric and not backed by a cell in `notebooks/06_proofs.py`,
treat it as an *estimate* and mark it as such.

---

## 1. The evaluation metric is stSNR, not PSNR/SSIM

Source: <https://ai4life-cidc25.grand-challenge.org/cidc25-evaluation-metrics/>

For a denoised stack `x` vs clean stack `y`, both shape `[T, H, W]`:

- **sSNR** — spatial SNR, per frame:

$$
\mathrm{sSNR}_t = 10\,\log_{10}\!\left(
  \frac{\sum_{h,w} y_{t,h,w}^2}
       {\sum_{h,w} (y_{t,h,w} - x_{t,h,w})^2}
\right),
\qquad \overline{\mathrm{sSNR}} = \frac{1}{T}\sum_t \mathrm{sSNR}_t
$$

- **tSNR** — temporal SNR, per pixel trace:

$$
\mathrm{tSNR}_{h,w} = 10\,\log_{10}\!\left(
  \frac{\sum_t y_{t,h,w}^2}
       {\sum_t (y_{t,h,w} - x_{t,h,w})^2}
\right),
\qquad \overline{\mathrm{tSNR}} = \frac{1}{HW}\sum_{h,w} \mathrm{tSNR}_{h,w}
$$

- **stSNR (the final score)** — convex combination:

$$
\mathrm{stSNR} = \alpha \cdot \overline{\mathrm{sSNR}} + (1 - \alpha) \cdot \overline{\mathrm{tSNR}}
$$

The challenge page doesn't pin `α` on this version of the page; assume
`α = 0.5` until confirmed, and design the model to win under either
extreme.

**Leaderboard score** = mean stSNR across all files in the task.

### What this metric implies for the model

- **SNR, not PSNR.** Numerator is `Σ y²` (signal energy), not `MAX²`.
  Absolute values are *smaller* than typical PSNR numbers; scores of
  6–17 on the leaderboard are normal.
- **Temporal fidelity is weighted equally to spatial fidelity.** A
  model that over-smooths transients (good-looking frames, flat
  traces) will tank tSNR and therefore the final score. Trace
  correlation is no longer a nice-to-have; it's half the metric.
- **Anything 2-D-only collapses.** Empirically (see §2), 2-D methods
  score ~6, 3-D methods score ≥14. Temporal context is the dominant
  lever on this dataset under this metric.

### Implementation note

Our `src/cidc/eval.py` (to be written) must match the challenge code
bit-for-bit. Scaffold:

```python
def stsnr(pred, ref, alpha=0.5):
    ssnr = _snr(ref, pred, axis=(-2, -1)).mean()        # per-frame, avg over T
    tsnr = _snr(ref, pred, axis=0).mean()               # per-pixel, avg over HW
    return alpha * ssnr + (1 - alpha) * tsnr
```

Score yourself on `F0` vs a trivial baseline (e.g. the temporal mean,
or simple Gaussian blur on `F1`) *before* training anything, so the
numbers are meaningful from day one.

---

## 2. Leaderboard snapshot (Task 1, as of 2026-04-20)

| Rank | Algorithm                       | Score | Notes                         |
|-----:|---------------------------------|------:|-------------------------------|
|    1 | **N2V 3D, patch 64³, no bias**  | 16.75 | Winner. 3-D Noise2Void.       |
|    2 | AI4Life CIDC25 Submission       | 15.59 | Unknown details.              |
|    3 | N2N 2D windowed                 | 15.10 | Noise2Noise, 2-D, temporal win. |
|    4 | 3D U-Net + N2V + FM2S           | 14.51 | Same family as #1.            |
|    5 | NafNet2Void 2D                  |  6.35 | 2-D only — collapses.         |

Four observations that drive strategy:

1. **3-D wins.** 5th (pure 2-D) scores 6.35; 4th (3-D) scores 14.51. A
   ~8 dB gap purely from using temporal context.
2. **The top-4 are all variants of self-supervised blind-spot / N2V /
   N2N.** Architecture is vanilla U-Net in every case.
3. **"No conv bias"** is called out by the winner. This is standard
   in N2V — biases let the network shortcut the blind-spot invariance.
   We must copy this.
4. **None of the top 4 publicly mention Anscombe VST or log-uniform
   gain augmentation.** These are the two dB leaks we expect to win on.

---

## 3. Our target: stSNR ≥ 25 on Task 1

Current leader is 16.75. Going for 25+ is **ambitious but not
irrational**; the metric has no explicit ceiling and the leader isn't
using known-good tricks. Honest gain budget:

| dB (opt.) | Source                                            | Status              |
|----------:|---------------------------------------------------|---------------------|
|     ~+2   | Anscombe VST + inverse on train & loss            | Not public          |
|     ~+2   | Log-uniform gain augmentation for Task 2 / F3     | Not public          |
|     ~+2   | ConvNeXt spatial encoder (vs plain Conv)          | Not public          |
|     ~+1   | Mamba temporal block at bottleneck only           | Not public          |
|     ~+1   | Dual loss (N2V mask + DeepInterp frame-drop)      | Not public          |
|     ~+1   | 8× TTA (4 rot × 2 flip) + EMA weights             | Not public          |
|  **~+9**  | **Stacked best-case**                             | Won't stack linearly |

Realistic expectation is **22–25**. If we land at 20 we are top-3; at
23+ we lead; at 25+ the approach is genuinely novel for this dataset.

### Why not just copy the winner and tune?

Because the *marginal* gains on a saturated 3-D N2V are small, and the
four levers above are *orthogonal* to what the winner is doing. Copying
the winner and tuning gets us to ~17–18. Stacking orthogonal levers is
the only path to ≥25.

---

## 4. Architecture — the honest version

### Things the other LLM got right

- Separating spatial-per-frame from temporal-across-frames is correct.
- ConvNeXt is a solid spatial encoder for this problem.
- ViT at 490² is a bad idea (small data × huge token count).

### Things the other LLM got wrong (measured in our proofs notebook)

- **"Long-range temporal dependencies across 1500 frames."** Our
  `06_proofs.py §6` shows `ACF[1]=0.995`, `ACF[30]=0.665`,
  `τ(0.5)=45 frames`. A spike at frame 200 has decayed to ~3.5 % by
  frame 350. The *physically relevant* temporal window is **60–100
  frames, not 1500.** Mamba's long-sequence selling point is wasted at
  full temporal extent; it earns its keep only at a U-Net bottleneck
  where features are already pooled.
- **"Only 4 training images."** We have 4 stacks × 1500 frames × many
  spatial patches ≈ **>1 M self-supervised windows**. Data scarcity is
  not the constraint; *clean-target* scarcity is (hence self-supervision).
- **PINN as the primary output head.** τ is pixel-dependent (cell-
  dependent GCaMP expression), the spike source `s(t)` needs a
  sparsity prior that can eat real fast transients, and coupling
  denoising to spike inference degrades both. **PINN only as an
  auxiliary loss with a small weight, and only after a strong denoiser
  exists.**

### Priority order (the critical point)

The biggest levers on this challenge, in order:

1. **Data recipe** — Anscombe VST + log-uniform gain augmentation. Fixes
   loss behaviour across intensities *and* brings F3's gain=991 noise
   in-distribution. Biggest single jump.
2. **Training signal** — self-supervised masking (N2V-3D and/or
   DeepInterp frame-drop). Decides whether you can train at all
   without paired clean data.
3. **Loss** — Anscombe-space MSE or explicit Poisson-Gaussian NLL.
   Decides whether dim neurons are recovered or smoothed out.
4. **Backbone** — U-Net vs ConvNeXt vs ConvNeXt + Mamba. Last 1–2 dB.

The other LLM told an *architecture-first* story. The correct order is
**data → signal → loss → architecture**, and the stSNR metric
reinforces that because temporal structure (1 & 2) dominates over
backbone capacity (4).

---

## 5. Recommended model (target: stSNR ≥ 25)

Hybrid 3-D denoiser, staged v1 → v2 → v3:

```
Input:  (B, 1, D, H, W) patch in Anscombe space
            ↓
   Per-frame ConvNeXt encoder (shared across D)
   3 stages @ [64, 128, 256] ch, depthwise-7x7, LN, GELU, stride-2 down
            ↓
   Bi-directional Mamba SSM at the bottleneck only
   Operates on (B, D, 256, H/8, W/8) — cheap at this resolution
            ↓
   Mirror decoder with skip connections
            ↓
   1×1×1 head → Anscombe-space prediction
            ↓
   Inverse Anscombe (Mäkitalo-Foi) → raw ADU output
```

**Training:**

- **Masking:** N2V-3D sparse voxel masking (~0.5 % per patch) + occasional
  DeepInterp-style full-centre-frame masking (~10 % of batches). Dual
  signal. Decouple by warmup.
- **Loss:** MSE on masked positions in raw ADU *after* inverse Anscombe.
  (Loss in raw space = what stSNR is computed in. Don't be clever.)
- **Aux loss (v3 only):** PINN regulariser on decoded traces, weight
  λ ∈ {0, 0.01, 0.1}, ablated.
- **Augmentation:** flips, 90° rotations, temporal reversal,
  log-uniform gain resampling `g ∈ [20, 2000]` on 50 % of batches.
- **Never train on F3.** F3 is validation-only for measuring Task 2
  generalisation honesty.
- **Optimiser:** AdamW, lr 3e-4, cosine w/ 3 restarts, EMA decay 0.999.
- **Early-stop:** on held-out `F1`/`F2` stSNR.

**Inference:**

- Overlap-tile with 16 px spatial + 8 frame temporal overlap.
- 8× TTA (4 rotations × 2 flips), averaged in **Anscombe space** before
  the inverse transform.
- Optional: ensemble 2–3 checkpoints selected by different tSNR/sSNR
  trade-offs.

### Staged delivery

| Stage | What lands                                           | Risk | Expected stSNR |
|-------|------------------------------------------------------|------|---------------:|
| v1    | 3-D U-Net (no ConvNeXt, no Mamba) + Anscombe + gain aug | low   | 14–17           |
| v1.5  | Swap encoder → ConvNeXt blocks                        | low   | 16–19           |
| v2    | Add Mamba at U-Net bottleneck                         | med   | 18–22           |
| v3    | PINN auxiliary loss + 8× TTA + EMA + ensembling       | high  | 20–25+          |

Every stage must produce a scored submission before the next starts. No
"finish v3 and submit once".

### Inference budget sanity (T4, 16 GB, 60 min / video)

- v1 3-D U-Net: ~3 M params, ~0.5 s/frame tiled → ~12 min per stack. ✓
- v2 + Mamba at bottleneck only: ~6–8 M params, ~1 s/frame → ~25 min. ✓
- v3 with 8× TTA: 8× v2 ≈ 200 min. ✗. Cut to 4× rotation-only TTA (~100 min) and time-budget it by dropping to 2× TTA on F3 only if the whole stack risks going over 60 min. Profile before committing.

---

## 6. Anti-patterns (what we will explicitly not do)

- **Training on F3.** F3 is Task 2 test; touching it leaks information.
- **Dropping Anscombe "because MSE on raw ADU is simpler".** It costs
  you the bright-pixel-dominance ~2 dB.
- **Per-level separate models.** Log-uniform gain augmentation
  replaces this; one model should handle all levels at inference.
- **ViT or pure Mamba as a full-resolution backbone.** Memory and
  convergence issues, and wasted on a problem with 60-frame effective
  context.
- **Optimising PSNR or SSIM.** The scorer is stSNR; early-stop on stSNR.
- **"Finish everything, submit once" mentality.** Every stage ships a
  scored submission. Measurement discipline beats architecture taste.

---

## 7. Reading order for a new contributor/LLM

1. `docs/ONBOARDING.md` — what this project is.
2. `docs/findings_summary.md` — what we measured.
3. `notebooks/06_proofs.py` — how we measured it (open this in marimo).
4. **This file (`docs/strategy.md`)** — what we plan to do with it.
5. `docs/cidc25_context.md` — challenge cheat-sheet.
6. `docs/concepts.md` — vocabulary.
7. `src/cidc/models/deepinterp/unet.py` — current reference backbone.

Next engineering step after reading: implement `src/cidc/eval.py` with
the stSNR metric and produce a baseline score using the F0 temporal
mean as a "denoiser". That number is our floor; anything below it is
a bug, not a model.
