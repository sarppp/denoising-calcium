# Onboarding — read this first

You are picking up work on the **CIDC25 calcium-imaging denoising
challenge** (AI4Life). This file tells you where everything lives and
in what order to read it.

---

## TL;DR in 5 lines

1. 8 TIFF stacks of calcium imaging: 4 noisy train, 1 clean val, 3
   noisy val at noise levels 1/2/3.
2. Noise is **Poisson-Gaussian** with three known gain levels
   (28 / 249 / 991). F0 is the shared clean source of F1/F2/F3.
3. Two tasks: content generalisation (levels 1–2) and OOD
   generalisation (level 3, **never seen in training**).
4. Plan: temporal U-Net (DeepInterpolation-style) in Anscombe space
   with continuous-gain augmentation for Task 2.
5. No clean training pairs are allowed; F0 may never touch the weights.

---

## Read in this order

### Step 1 — Challenge rules and context

`@/app/workspace/docs/cidc25_context.md`
What the challenge asks, the two tasks, hardware/time budget, what's
forbidden.

### Step 2 — Vocabulary and math

`@/app/workspace/docs/concepts.md`
Poisson-Gaussian noise, Anscombe VST, ACF, SNR, PSNR/SSIM, U-Net
receptive fields, self-supervised denoising (Noise2Noise /
DeepInterpolation / DeepCAD).

### Step 3 — What we actually measured

`@/app/workspace/docs/findings_summary.md`
Self-contained report of every finding with numbers: noise constants,
star-vs-chain verification, ACF/SNR, Anscombe bug history, what would
fail and why, the modelling plan that follows.

### Step 4 — Marimo editing rules (if touching notebooks)

`@/app/workspace/docs/marimo_conventions.md`
Single-owner rule for variables, cell-private `_prefix` convention,
why LaTeX / markdown rendering broke for us.

---

## Code layout

```
workspace/
├── src/cidc/              ← the library (imports: `from cidc import ...`)
│   ├── io.py              ← load_stack() memmap TIFF loader
│   ├── stats.py           ← stack_info, temporal_autocorr, blob stats
│   ├── noise.py           ← NOISE_LEVELS, FILE_NOISE, anscombe,
│   │                        inverse_anscombe, sample_poisson_gaussian
│   └── __init__.py        ← public API
├── notebooks/             ← marimo notebooks for exploration
│   ├── 01_basic_look.py   ← load + one frame + temporal mean
│   ├── 02_noise.py        ← bg histograms, var-vs-mean (8-panel grid)
│   ├── 03_temporal.py     ← per-pixel traces, ACF (val + train)
│   ├── 04_compare_levels.py ← F0 vs Fk frames/traces/residuals
│   ├── 05_findings.py     ← guided tour, math + code + GIF + MP4
│   └── 06_proofs.py       ← every claim in findings_summary.md
│                             mapped to the line of code that proves it
├── scripts/               ← one-shot EDA / noise-fitting scripts
├── data/
│   ├── train/A1.tif B1.tif C2.tif D2.tif
│   └── val/  F0.tif F1.tif F2.tif F3.tif
└── docs/                  ← this directory
```

---

## Key facts to carry in your head

- **Dtype.** Raw files are `int16`. Always cast to `float32` before
  arithmetic.
- **Noise constants** live in `cidc.noise.NOISE_LEVELS` and
  `cidc.noise.FILE_NOISE`. Use them; do not re-fit on validation.
- **F0 is val-only.** Never fit weights to it. Use only for metrics.
- **Anscombe has a 2× coefficient bug history.** The correct
  coefficient on `1/z` is `√(3/2)/4`. Already fixed in the code.
- **Memmap everything.** `load_stack` returns a tifffile memmap, not
  an in-memory array. Slice before `np.asarray()` to avoid loading
  720 MB.
- **Inference budget: T4 16 GB / 60 min** for `[1500, 490, 490]`.
  Plan for tiled spatial inference + temporal windowing.

---

## What's been built, what's pending

### Done

- EDA pipeline (`cidc.stats`, `scripts/`).
- Noise-model fit per stack, confirmed Poisson-Gaussian, gains
  measured.
- Anscombe forward + Mäkitalo-Foi inverse, sampler verified to within
  4 % on real data.
- Star-vs-chain structural check.
- Full guided notebook (`05_findings.py`).

### Pending (suggested order)

1. `cidc.eval` — PSNR / SSIM / Pearson trace-correlation vs F0.
2. `cidc.baselines` — temporal mean, rolling median. Lower bound the
   learned model must beat.
3. `cidc.models.deepinterp` — first temporal U-Net + training script.
4. Gain-augmentation pipeline for Task 2 (log-uniform `g ∈ [20, 2000]`).
5. Inference profiler on a fake `[1500, 490, 490]` stack → lock tile
   size before architecture grows.

---

## Style / behaviour rules the user has set

- **Minimal upstream fixes.** Don't pile on workarounds; fix the root
  cause with the smallest possible change.
- **Never delete tests without explicit permission.** Write tests
  first for major changes.
- **Don't spam the user.** Short, concrete, Markdown-formatted
  answers. No acknowledgement phrases ("Great!", "You're right!").
- **Don't add emojis unless asked.**
- **Don't create new `.md` files or helper scripts unless necessary.**
- **Marimo notebooks are for exploration.** One topic per notebook.
  Don't cram features into an existing notebook — split instead.

---

## Quick commands

```bash
# Launch a notebook
uv run --with="marimo[mcp]" marimo edit workspace/notebooks/05_findings.py \
    --mcp --no-token --host 0.0.0.0 --port 2718 --watch

# Run all tests
uv run pytest

# Quick EDA script (numbers only, no plots)
uv run python workspace/scripts/eda_numbers.py
```
