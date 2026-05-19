"""05 — Findings, with code and explanations.

A guided tour of everything we learned about the CIDC25 data, with the
actual code that produced each finding and short explanations of the
technical terms. The goal: you end this notebook knowing exactly what
the data looks like and why the modelling plan is what it is.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # CIDC25 — what the data told us

        Eight TIFF stacks, all `[1500, 490, 490]` `int16`:

        - **Training** (used for weight updates): `A1, B1` (noise level 1)
          and `C2, C2` (noise level 2).
        - **Validation** (used only for inspection + model selection):
          `F0` is the *clean* ground-truth signal; `F1, F2, F3` are
          *three independent noisy realisations of `F0`* at noise
          levels 1, 2, and 3 respectively. Level 3 is OOD (Task 2).

        Below, each finding is shown with the exact code that produced
        it and a short glossary of the terms involved.
        """
    )
    return (mo,)


@app.cell
def _setup():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from cidc import (
        FILE_NOISE,
        NOISE_LEVELS,
        anscombe,
        inverse_anscombe,
        load_stack,
        sample_poisson_gaussian,
        stack_info,
        temporal_autocorr,
    )

    DATA = Path("/app/workspace/data")
    return (
        DATA,
        NOISE_LEVELS,
        anscombe,
        inverse_anscombe,
        load_stack,
        np,
        plt,
        sample_poisson_gaussian,
        stack_info,
        temporal_autocorr,
    )


@app.cell
def _finding1_intro(mo):
    mo.md("""
    ## Finding 1 — everything is `int16`, signal is sparse

    - Values are stored as 16-bit signed integers, range −32 768 to
      32 767, but **actual signal lives in 0–20 000** — we use
      less than 1 % of the dynamic range. Safe to cast to
      `float32` and divide by ~10 000.
    - `F0.min == 0` exactly. Every noisy stack dips negative by
      ~200 ADU. That's a subtracted pedestal plus noise dithering
      below zero on pure background. **Sanity check:** this confirms
      `F0` is genuinely the clean reference.
    - All `F0..F3` have the same mean (~204). *Noise changes
      variance, not mean.*

    **Term — ADU:** analog-to-digital unit. The number stored in
    the sensor after digitisation. Not photons; related to photons
    by `ADU = gain × photons + read_noise`.
    """)
    return


@app.cell
def _finding1_code(DATA, stack_info):
    for _p in sorted(DATA.glob("*/*.tif")):
        _i = stack_info(_p)
        print(f"{_p.parent.name}/{_p.name:8s}  shape={_i.shape}  "
              f"dtype={str(_i.dtype):6s}  min={_i.min:>6.0f}  "
              f"mean={_i.mean:>6.1f}  max={_i.max:>6.0f}")
    return


@app.cell
def _pg_background(mo):
    mo.md(r"""
    ---
    ### Background — what *is* Poisson-Gaussian noise?

    Two physical sources of randomness in an imaging sensor, one on top
    of the other.

    **1. Poisson noise (shot noise)** — photon counting.
    Light arrives in discrete packets (photons). Even with a perfectly
    constant light source, the *number of photons* you count in a fixed
    time window fluctuates. If the true rate is `λ` photons per pixel
    per frame, the actual count `N` is a random draw from the Poisson
    distribution:

    $$
    P(N = k) = \frac{\lambda^k e^{-\lambda}}{k!}
    $$

    Key property: **mean = variance = λ**. Brighter pixels have *more*
    absolute noise, but *less relative* noise (`σ/μ = 1/√λ`).

    **2. Gaussian noise (read noise)** — sensor electronics.
    When the sensor reads out pixel values, amplifiers and the ADC add
    a small independent *zero-mean Gaussian* error:

    $$
    r \sim \mathcal{N}(0, \sigma_r^2)
    $$

    This doesn't depend on how bright the pixel is.

    **Putting them together.** A real calcium imaging pixel stores

    $$
    y = g \cdot N + r,\qquad N \sim \text{Poisson}(\lambda),\ r \sim \mathcal{N}(0, \sigma_r^2)
    $$

    where `g` is the camera **gain** (ADU per photon). The mean and
    variance work out to:

    $$
    \mathbb{E}[y] = g \lambda,\qquad
    \operatorname{Var}[y] = g^2 \lambda + \sigma_r^2
    $$

    Eliminating `λ` using `λ = E[y] / g`:

    $$
    \boxed{\operatorname{Var}[y] \;=\; g \cdot \mathbb{E}[y] \;+\; \sigma_r^2}
    $$

    **This is the key equation.** A straight line with slope = gain and
    intercept = read variance. To test whether noise in a stack is
    Poisson-Gaussian, you just compute `(mean, variance)` per pixel over
    time and check if they lie on a line.

    **How we verified it on this dataset.**

    1. Pick ~100 000 random pixels.
    2. For each pixel, compute its temporal mean and variance across
       the 1 500 frames.
    3. Scatter-plot `(mean, variance)`.
    4. Fit a line `v = g·m + c`. Report `R²` (fraction of variance
       explained by the line — 1 is perfect).
    5. Result: `R² ≥ 0.92` on every noisy stack. On clean `F0`,
       `R² ≈ 0.09` — *nonsense fit, which is itself the confirmation
       that `F0` carries no Poisson-Gaussian noise to fit*.

    The scatter plots live in `02_noise.py` (the `var_vs_intensity_all`
    cell). Below, Finding 2 runs the same fit numerically for each
    stack and prints the numbers.
    ---
    """)
    return


@app.cell
def _finding2_intro(mo):
    mo.md(r"""
    ## Finding 2 — noise is Poisson-Gaussian, in three discrete levels

    The calcium imaging forward model: photons arrive as a Poisson
    process, the sensor multiplies by a *gain* to get ADU, and adds
    independent Gaussian *read noise*. That gives

    $$
    \operatorname{Var}[y] \;=\; g \cdot \mathbb{E}[y] \;+\; \sigma_r^2
    $$

    - **gain (`g`)** — ADU per photon. Bigger gain = noisier image.
    - **read variance (`σ_r²`)** — constant per-pixel electronic
      noise, independent of signal.

    Fit `Var = g · Mean + σ_r²` to per-pixel `(mean, var)` over
    time. Linear slope = gain, intercept = read-noise variance.
    `R²` is the fraction of variance explained by the line (1 =
    perfect fit).

    The measured values form a **geometric ladder**:

    | level | gain | σ_r²  | files |
    |-------|------|-------|-------|
    | 1 | ≈ 28  | ≈ 2 500 | A1, B1, F1 |
    | 2 | ≈ 249 | ≈ 2 700 | C2, D2, F2 |
    | 3 | ≈ 991 | ≈ 3 700 | **F3 only (OOD, Task 2)** |

    Level 2 ≈ 9× noisier than level 1. Level 3 ≈ 4× noisier than
    level 2. `R² ≥ 0.92` everywhere ⇒ this is *the* model, not an
    approximation. The constant `σ_r²` ≈ 2 500 is consistent with
    a single physical sensor.
    """)
    return


@app.cell
def _finding2_code(DATA, load_stack, np):
    """Fit Var = g·Mean + σ_r² on 100k random pixels per stack."""
    def _fit(path):
        s = load_stack(path)
        rng = np.random.default_rng(0)
        idx = rng.choice(490 * 490, size=100_000, replace=False)
        y, x = np.divmod(idx, 490)
        tr = np.asarray(s[:, y, x], dtype=np.float64)
        m, v = tr.mean(axis=0), tr.var(axis=0)
        # Use only background-dominated pixels (bottom 80% by intensity).
        q = np.quantile(m, 0.8)
        mask = m < q
        A = np.vstack([m[mask], np.ones(mask.sum())]).T
        (g, c), *_ = np.linalg.lstsq(A, v[mask], rcond=None)
        r2 = 1 - ((v[mask] - A @ [g, c]) ** 2).sum() / ((v[mask] - v[mask].mean()) ** 2).sum()
        return float(g), float(c), float(r2)

    print(f"{'file':22s}  {'gain':>8s}  {'σ_r²':>8s}  {'R²':>6s}")
    for _p in sorted(DATA.glob("*/*.tif")):
        _g, _c, _r2 = _fit(_p)
        print(f"{_p.parent.name+'/'+_p.name:22s}  {_g:8.1f}  {_c:8.0f}  {_r2:6.2f}")
    return


@app.cell
def _finding3_intro(mo):
    mo.md(r"""
    ## Finding 3 — `F0` is the clean of *all* `F1/F2/F3` (not a chain)

    Two plausible structures for the validation set:

    | Hypothesis | Means |
    |---|---|
    | **Star:** `F0` is clean, `F1/F2/F3` are three *independent* noisy realisations | each `F_k = F_0 + noise_k` |
    | **Chain:** `F_{k+1}` is a noisy version of `F_k` | `F_2 = F_1 + noise`, `F_3 = F_2 + noise` |

    Distinguish by computing the per-frame Pearson correlation of
    pixel values:

    $$
    \operatorname{corr}(A, B)
    = \frac{\sum_i (A_i - \bar{A})(B_i - \bar{B})}
           {\sqrt{\sum_i (A_i - \bar{A})^2 \cdot \sum_i (B_i - \bar{B})^2}}
    $$

    Measured (averaged over 30 frames):

    ```
    corr(F0, F1) = 0.74     corr(F1, F2) = 0.30
    corr(F0, F2) = 0.40     corr(F1, F3) = 0.16
    corr(F0, F3) = 0.22     corr(F2, F3) = 0.09
    ```

    **Key tell:** `corr(F0, F2) = 0.40 > corr(F1, F2) = 0.30`.
    If the chain hypothesis were true, `F1` would be *closer* to
    `F2` than `F0` is, because `F1 → F2` adds only one round of
    noise and `F0 → F2` adds two. We see the opposite: `F0` is the
    best predictor of every `F_k`. Combined with
    `mean(F_k − F_0) ≈ 0` for all `k`, this confirms the **star**.

    ```
          F0 (clean scene)
               │
    ┌──────────┼──────────┐
    ▼          ▼          ▼
    F1         F2         F3
    +noise L1  +noise L2  +noise L3 (OOD)
    ```
    """)
    return


@app.cell
def _finding3_code(DATA, load_stack, np):
    def _corr(A, B):
        A = A.astype(np.float64).ravel()
        B = B.astype(np.float64).ravel()
        A, B = A - A.mean(), B - B.mean()
        return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))

    def _frame_corr(p1, p2, n=30):
        a = load_stack(p1)
        b = load_stack(p2)
        idx = np.linspace(0, a.shape[0] - 1, n, dtype=int)
        return np.mean([_corr(np.asarray(a[t]), np.asarray(b[t])) for t in idx])

    _pairs = [
        ("F0", "F1"), ("F0", "F2"), ("F0", "F3"),
        ("F1", "F2"), ("F1", "F3"), ("F2", "F3"),
    ]
    for _a, _b in _pairs:
        _c = _frame_corr(DATA / "val" / f"{_a}.tif", DATA / "val" / f"{_b}.tif")
        print(f"corr({_a}, {_b}) = {_c:+.3f}")
    return


@app.cell
def _finding4_intro(mo):
    mo.md(r"""
    ## Finding 4 — signal has strong temporal structure; noise does not

    **Term — autocorrelation function (ACF).** For a 1D time series
    `x[t]`, the ACF at lag `k` is `corr(x[t], x[t+k])`. It measures
    how much a value predicts a value `k` frames later.

    - `ACF[0] = 1` always.
    - A pure-noise signal has `ACF[k] ≈ 0` for `k ≥ 1`.
    - A signal with slow dynamics has `ACF[k]` decaying gradually.

    **Measured** (averaged over active pixels):

    | stack | `ACF[1]` | `ACF[10]` | `ACF[30]` | τ(0.5) |
    |---|---|---|---|---|
    | `F0` (clean) | 0.995 | 0.917 | 0.665 | **45 frames** |
    | `F1` (level 1) | ≈ 0.03 | ≈ 0 | ≈ 0 | 1 |
    | `F2` (level 2) | ≈ 0.008 | ≈ 0 | ≈ 0 | 1 |
    | `F3` (level 3) | ≈ 0.004 | ≈ 0 | ≈ 0 | 1 |

    `τ(0.5) = 45` on `F0` is the calcium-transient decay: a spike
    takes ~1.5 s (at 30 Hz) to drop to half.

    **Term — SNR in dB.** Given signal fraction `s = ACF[1] / (1 − ACF[1])`
    of the noisy stack (because `Var(signal) ≈ ACF[1] · Var(total)`
    for a white-noise + signal mixture), the SNR in decibels is
    `10 · log₁₀(s)`:

    ```
    level 1:  ≈ −14 dB   (signal ~4 % of total power)
    level 2:  ≈ −21 dB
    level 3:  ≈ −24 dB   (signal ~0.4 % of total power)
    ```

    **Implication — temporal denoisers (DeepInterpolation / DeepCAD).**
    The signal is temporally correlated; the noise is not. Train a
    network to predict frame `t` from its neighbours `{t±1, t±2, …}`
    using MSE loss on the noisy target. Noise is independent across
    frames, so the network cannot predict it from neighbours; the
    best it can do is emit the denoised signal. This is the
    theoretical reason to prefer that architecture family here.
    """)
    return


@app.cell
def _finding4_code(DATA, load_stack, plt, temporal_autocorr):
    _fig, _ax = plt.subplots(figsize=(9, 4))
    for _name, _folder in [
        ("F0", "val"), ("F1", "val"), ("F2", "val"), ("F3", "val"),
    ]:
        _acf = temporal_autocorr(load_stack(DATA / _folder / f"{_name}.tif"),
                                 max_lag=60)
        _ax.plot(_acf, label=_name)
    _ax.set_xlabel("lag (frames)")
    _ax.set_ylabel("ACF")
    _ax.set_title("Temporal autocorrelation — validation stacks")
    _ax.axhline(0, c="k", lw=0.5)
    _ax.legend()
    _fig
    return


@app.cell
def _finding5_intro(mo):
    mo.md(r"""
    ## Finding 5 — our noise sampler matches reality

    Once we have `(gain, read_var)` per level, we can *generate*
    noise from the clean `F0` and compare the result to the real
    `F1/F2/F3`. If the model is right, the simulated and real
    residuals should have the same variance-vs-intensity slope.

    **Result — simulated/real gain ratio:**

    ```
    F1:  sim / real = 0.984   (level 1)
    F2:  sim / real = 1.002   (level 2)
    F3:  sim / real = 0.960   (level 3)
    ```

    All within ~4 %. This means: **given any clean-signal estimate,
    we can synthesise arbitrary-noise-level training data**. That's
    exactly what the Task 2 augmentation strategy needs — sample a
    log-uniform `gain ∈ [20, 2000]`, run the sampler, pair input-output.

    The sampler implementation (`cidc.noise.sample_poisson_gaussian`):

    ```python
    λ = clean / gain                  # Poisson rate
    shots = Poisson(λ) * gain         # photon fluctuations, scaled back
    read  = Normal(0, sqrt(read_var)) # sensor noise
    y = shots + read
    # -> E[y] = clean ; Var[y] = gain·clean + read_var ✓
    ```
    """)
    return


@app.cell
def _finding5_code(
    DATA,
    NOISE_LEVELS,
    load_stack,
    np,
    sample_poisson_gaussian,
):
    """Compare simulated Fk (from F0) to real Fk in residual-variance slope."""
    def _slope(clean, noisy):
        a, b = clean.astype(np.float64), noisy.astype(np.float64)
        r = b - a
        bins = np.linspace(a.min(), a.max(), 30)
        which = np.digitize(a.ravel(), bins)
        m, v = [], []
        for i in range(1, len(bins)):
            mask = which == i
            if mask.sum() > 200:
                m.append(a.ravel()[mask].mean())
                v.append(r.ravel()[mask].var())
        A = np.vstack([m, np.ones_like(m)]).T
        return float(np.linalg.lstsq(A, v, rcond=None)[0][0])

    _F0 = load_stack(DATA / "val" / "F0.tif")
    _t = np.linspace(0, _F0.shape[0] - 1, 150, dtype=int)
    _clean = np.asarray(_F0[_t, :200, :200], dtype=np.float64)

    for _lvl, _name in [(1, "F1"), (2, "F2"), (3, "F3")]:
        _params = NOISE_LEVELS[_lvl]
        _real = np.asarray(load_stack(DATA / "val" / f"{_name}.tif")[_t, :200, :200])
        _sim = sample_poisson_gaussian(_clean, _params,
                                        rng=np.random.default_rng(0))
        _gr = _slope(_clean, _real)
        _gs = _slope(_clean, _sim)
        print(f"{_name}:  real gain={_gr:7.1f}   sim gain={_gs:7.1f}   "
              f"ratio={_gs/_gr:.3f}")
    return


@app.cell
def _finding6_intro(mo):
    mo.md(r"""
    ## Finding 6 — Anscombe variance-stabilising transform

    **Problem:** Poisson-Gaussian noise has *intensity-dependent*
    variance. A network trained with MSE loss weights bright-pixel
    errors more than dim-pixel errors — biased and hard to learn.

    **Solution — generalised Anscombe VST** (Foi et al. 2008). Apply

    $$
    z(y) = \frac{2}{g} \sqrt{ g \cdot y + \frac{3}{8} g^2 + \sigma_r^2 }
    $$

    and the transformed variable `z` has `Var[z] ≈ 1` *independent
    of intensity*. Now any Gaussian denoiser applies directly. After
    denoising in the `z` domain, invert the transform to get back
    ADU units.

    **Inverse transform** — Mäkitalo-Foi (2011) closed-form
    unbiased inverse:

    $$
    \hat{y} = g \left[ \left( \frac{z}{2} \right)^2 + \frac{\sqrt{3/2}}{4 z} - \frac{11}{8 z^2} + \frac{5 \sqrt{3/2}}{8 z^3} - \frac{1}{8} - \frac{\sigma_r^2}{g^2} \right]
    $$

    **Bug we fixed:** I initially coded the `1/z` coefficient as
    `sqrt(3/2)/2` (off by 2×) — round-trip showed a bias of
    `~gain × 0.3` ADU. After fixing to the correct `sqrt(3/2)/4`,
    residual bias dropped to ~7 / 38 / 46 ADU at levels 1/2/3.

    **Remaining caveat:** at very high gain (level 3), the per-pixel
    Poisson rate is < 1 photon, and the VST can't fully stabilise
    variance at such low counts. Observed `Var[z]`:

    | level | bright-pixel `Var[z]` | dim-pixel `Var[z]` |
    |---|---|---|
    | 1 | ~ 1.20 | ~ 1.31 |
    | 2 | ~ 1.01 | ~ 0.81 |
    | 3 | ~ 0.60 | ~ 0.28 |

    ⇒ Anscombe is a good preprocessing for levels 1 and 2 but only
    partially stabilises level 3. A learned denoiser adapts.

    **Term — variance-stabilising transform (VST).** Any monotone
    `f` such that `Var[f(Y)]` is approximately constant across
    intensities. Anscombe is the standard VST for Poisson data.
    """)
    return


@app.cell
def _finding6_code(
    NOISE_LEVELS,
    anscombe,
    inverse_anscombe,
    np,
    sample_poisson_gaussian,
):
    """Show that Var(z) ≈ 1 after Anscombe, across intensities."""
    _rng = np.random.default_rng(0)
    _clean = np.linspace(0, 3000, 2000).repeat(200).reshape(2000, 200)

    print(f"{'level':>7s}  {'bin_mean':>10s}  {'Var[z]':>8s}  {'target':>8s}")
    for _lvl, _params in NOISE_LEVELS.items():
        _y = sample_poisson_gaussian(_clean, _params, rng=_rng)
        _z = anscombe(_y, _params)
        for _q in (0.1, 0.5, 0.9):
            _cut = np.quantile(_clean, _q)
            _mask = (_clean > _cut - 100) & (_clean < _cut + 100)
            print(f"  level {_lvl}  {_clean[_mask].mean():10.1f}  "
                  f"{_z[_mask].var():8.3f}  {1.0:>8.1f}")

    # Round-trip bias
    print()
    print(f"{'level':>7s}  {'bias(y_back - clean)':>22s}")
    for _lvl, _params in NOISE_LEVELS.items():
        _y = sample_poisson_gaussian(_clean, _params, rng=_rng)
        _y_back = inverse_anscombe(anscombe(_y, _params), _params)
        print(f"  level {_lvl}  {float((_y_back - _clean).mean()):+22.2f}")
    return


@app.cell
def _finding7_intro(mo):
    mo.md(r"""
    ## Finding 7 — spatial structure: neurons are tiny and sparse

    Using MAD-based thresholding on the temporal mean image (see
    `scripts/eda_numbers.py`):

    - **Bright pixel fraction** ≈ 0.3–0.9 % of the FOV.
    - **Neuron count** ≈ 120–270 per 490×490 frame.
    - **Neuron radius** ≈ 2–3 px (so diameter 4–6 px).

    **Term — MAD (median absolute deviation).** A robust alternative
    to standard deviation: `MAD = median(|x - median(x)|)`. For
    Gaussian data, `σ ≈ 1.4826 · MAD`. Used for thresholding because
    a few bright neurons would blow up a non-robust `mean + k·std`
    estimate, but can't skew the median.

    **Implications:**

    - A U-Net with **receptive field 20–40 px** covers a neuron and
      its neighbourhood comfortably. Don't oversize.
    - `F3` has ~20 % smaller blob radius than `F0` — noise is
      already eating into fine spatial detail.
    """)
    return


@app.cell
def _summary(mo):
    mo.md(r"""
    ## Summary — the plan that falls out of the findings

    1. **Normalise** `int16 → float32 / 10_000`. Background ≈ 0.
    2. **Apply Anscombe VST** with the measured `(gain, read_var)`
       for each stack. Input to the network is the `z`-transformed
       stack; train with plain MSE because `Var[z] ≈ 1`.
    3. **Architecture — temporal U-Net** (DeepInterpolation /
       DeepCAD style). Predict frame `t` from `{t±1, …, t±k}`.
       Self-supervised: no clean target needed.
    4. **Task 1 (noise levels 1, 2)** — train on A1/B1/C2/D2
       together. Validate on F1/F2 against F0.
    5. **Task 2 (OOD, level 3)** — augment by resampling
       Poisson-Gaussian noise on an approximate clean estimate of
       the training set, with log-uniform gain in `[20, 2000]`.
       Never train on F3 itself. Validate on F3 vs F0.
    6. **Metrics** — PSNR, SSIM, and Pearson trace correlation of
       denoised vs F0.
    7. **Deployment** — tiled spatial inference
       (128×128 with 16 px overlap) + temporal windowing, to fit
       T4 16 GB / 60-min budget.

    ### Mini-glossary

    - **Poisson-Gaussian noise**: `y = Poisson(clean / g) · g + N(0, σ_r²)`.
    - **gain (g)**: ADU per photon. We measured 28, 249, 991.
    - **read_var (σ_r²)**: constant electronic variance, ~2 500 ADU².
    - **ACF[k]**: `corr(x[t], x[t+k])`. Signal has slow decay; noise = 0.
    - **τ(0.5)**: lag at which ACF crosses 0.5. Measures signal timescale.
    - **SNR in dB**: `10 · log₁₀(signal_power / total_power)`.
      Negative dB means signal < total (noisy).
    - **VST / Anscombe**: transform that makes Poisson-Gaussian noise
      look like `N(·, 1)`, intensity-independent.
    - **Inverse Anscombe (Mäkitalo-Foi)**: closed-form unbiased
      inverse, with small residual bias at low photon counts.
    - **PSNR**: `10·log₁₀(max_value² / MSE)` — higher is better.
    - **SSIM**: structural similarity index (0 to 1, 1 = identical).
      Penalises blurring more than MSE does.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### VIDEO PARTS

    ---

    **Note on frame rate.** The stacks have **1500 frames** but the
    TIFFs we received contain no timing metadata we extracted. So
    the *duration in seconds* depends on the (currently unknown)
    acquisition frame rate `fps`:

    | if fps = | total duration |
    |----------|----------------|
    | 10 Hz    | 150 s (~2.5 min) |
    | 15 Hz    | 100 s (~1.7 min) |
    | 30 Hz    | 50 s   |
    | 60 Hz    | 25 s   |
    | 150 Hz   | 10 s   |

    Wherever you see "~1.5 s" or similar in the temporal-decay
    discussion, that assumes **30 Hz** as a reasonable default for
    two-photon calcium imaging. The real `fps` should be confirmed
    from the challenge / Zenodo dataset description.
    """)
    return


@app.cell
def _frame_view(DATA, frame_slider, load_stack, np, plt):
    """Side-by-side F0 vs F1 at the selected frame."""
    _F0 = load_stack(DATA / "val" / "F0.tif")
    _F1 = load_stack(DATA / "val" / "F1.tif")
    _t = int(frame_slider.value)
    _vmax = float(np.percentile(np.asarray(_F0[::300]), 99))

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 5.5))
    _axes[0].imshow(np.asarray(_F0[_t]), cmap="gray", vmin=0, vmax=_vmax)
    _axes[0].set_title(f"F0 (clean)  t={_t}")
    _axes[0].axis("off")
    _axes[1].imshow(np.asarray(_F1[_t]), cmap="gray", vmin=0, vmax=_vmax)
    _axes[1].set_title(f"F1 (noisy L1)  t={_t}")
    _axes[1].axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    """Scrub through frames of F0 (clean) and F1 (noisy level 1) side by side."""
    frame_slider = mo.ui.slider(0, 1499, value=750, step=1,
                                 label="frame t", show_value=True)
    frame_slider
    return (frame_slider,)


@app.cell
def _animation(DATA, load_stack, mo, np):
    """Auto-playing GIF of F0 (clean) vs F1 (noisy L1) side by side.

    Written as a real .gif (loops automatically, no JS needed).
    60 frames, 192x192 per panel."""
    from PIL import Image

    _out = "/tmp/cidc_f0_vs_f1.gif"
    _F0 = load_stack(DATA / "val" / "F0.tif")
    _F1 = load_stack(DATA / "val" / "F1.tif")

    _stride = 25                            # 60 frames total
    _crop = slice(150, 342)                 # 192x192 ROI
    _f0 = np.asarray(_F0[::_stride, _crop, _crop], dtype=np.float32)
    _f1 = np.asarray(_F1[::_stride, _crop, _crop], dtype=np.float32)

    # Map to 0..255 with shared vmax so brightness matches.
    _vmax = float(np.percentile(_f0, 99))
    def _norm(arr):
        return np.clip(arr / _vmax * 255.0, 0, 255).astype(np.uint8)

    _frames = []
    for _i in range(_f0.shape[0]):
        _row = np.concatenate([_norm(_f0[_i]), _norm(_f1[_i])], axis=1)
        _frames.append(Image.fromarray(_row, mode="L"))

    _frames[0].save(
        _out, save_all=True, append_images=_frames[1:],
        duration=80, loop=0, optimize=True,
    )
    mo.image(str(_out), caption="left: F0 (clean)   right: F1 (noisy L1)")
    return


@app.cell
def _video_mp4(DATA, load_stack, mo, np):
    """Same content as the GIF, but as a real MP4 with play/pause/seek/speed
    controls. Needs imageio + imageio-ffmpeg."""
    import imageio.v3 as iio

    _out = "/tmp/cidc_f0_vs_f1.mp4"
    _F0 = load_stack(DATA / "val" / "F0.tif")
    _F1 = load_stack(DATA / "val" / "F1.tif")

    _stride = 10                            # 150 frames (mp4 compresses well)
    _crop = slice(120, 376)                 # 256x256 ROI
    _f0 = np.asarray(_F0[::_stride, _crop, _crop], dtype=np.float32)
    _f1 = np.asarray(_F1[::_stride, _crop, _crop], dtype=np.float32)

    _vmax = float(np.percentile(_f0, 99))
    _f0_u8 = np.clip(_f0 / _vmax * 255.0, 0, 255).astype(np.uint8)
    _f1_u8 = np.clip(_f1 / _vmax * 255.0, 0, 255).astype(np.uint8)
    # Side-by-side, replicate to RGB.
    _frames = np.concatenate([_f0_u8, _f1_u8], axis=2)
    _frames = np.repeat(_frames[..., None], 3, axis=-1)  # (T, H, 2W, 3)

    iio.imwrite(_out, _frames, fps=15, codec="libx264",
                macro_block_size=1, output_params=["-pix_fmt", "yuv420p"])
    mo.video(src=_out, controls=True, autoplay=False, loop=True,
             muted=True, width="640px")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
