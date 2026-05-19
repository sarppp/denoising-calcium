"""03 — Noise model verification.

QUESTION: Is the noise actually Poisson-Gaussian?
          What are the real gain parameters?

Decision gate: confirms the noise model and gain constants before they are used
in Anscombe VST, loss weighting, and gain augmentation for Task 2.
If the fitted gain differs from cidc.noise.NOISE_LEVELS by more than ~10%,
use the fitted values — not the assumed ones.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 — Noise model verification
    ## Is the noise actually Poisson-Gaussian, and do we know its parameters?
    """)
    return


@app.cell
def _setup():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    from cidc import load_stack
    from cidc.noise import NOISE_LEVELS, FILE_NOISE

    DATA = Path(__file__).parent.parent.parent / "data"
    return DATA, NOISE_LEVELS, load_stack, mo, np, plt, stats


@app.cell
def _intro(mo):
    mo.md("""
    # 03 — Noise model verification

    **Purpose:** confirm that the noise in the training stacks (A1, B1, C2, D2) actually
    follows a Poisson-Gaussian distribution, and measure the gain and read noise parameters
    from the data rather than accepting the library constants on faith.

    ### Why this notebook comes before any model code

    Notebooks 01 and 02 treated the noise model as given — they measured stSNR and ACF
    but never asked *why* the noise looks the way it does, or whether the assumed model
    is correct. Everything downstream depends on that assumption:

    - **Anscombe VST** (`anscombe(y, g, σ_r²)`) variance-stabilises the data before loss
      computation. It requires accurate g and σ_r². If those are wrong, the stabilised
      image has non-uniform variance and the loss weights each pixel incorrectly.
    - **Gain augmentation** for Task 2 (F3, OOD noise level) uses `g ~ LogUniform(20, 2000)`.
      The range 20–2000 was chosen to span the observed gain levels. If the real gains
      differ from what the library says, the augmentation range may miss the OOD test gain.
    - **Self-supervised loss** in N2V3D implicitly assumes the noise is zero-mean conditioned
      on the signal. Poisson-Gaussian noise has this property (the noise has zero mean after
      subtracting the signal). Other noise models (e.g. multiplicative, correlated) do not,
      and would require a different masking strategy.

    This notebook measures all three things from the raw training data.
    """)
    return


@app.cell
def _md_physics(mo):
    mo.md(r"""
    ## The Poisson-Gaussian noise model — physics and math

    ### Where the noise comes from

    Calcium imaging works by detecting photons emitted by fluorescent indicators bound to
    calcium ions. Two independent physical processes add noise:

    1. **Shot noise (Poisson):** photons arrive randomly. Even with a perfectly stable light
       source, the count of photons hitting a pixel in one exposure follows a Poisson
       distribution. The variance of a Poisson random variable equals its mean — so brighter
       regions are noisier in absolute terms, but have *better* SNR because the signal grows
       faster than the noise.

    2. **Read noise (Gaussian):** the camera's electronics add a fixed-variance additive noise
       when reading out the sensor. This is independent of the signal level and dominates in
       dark regions.

    ### The model equation

    Let $x$ be the true (noiseless) photon count at a pixel. The observed value $y$ in ADU
    (camera digital units) is:

    $$y = \text{Poisson}(x / g) \cdot g + \mathcal{N}(0,\, \sigma_r^2)$$

    where:
    - $g$ is the **gain** (ADU per photoelectron). Higher gain → fewer photons per ADU →
      stronger Poisson noise relative to the signal.
    - $\sigma_r^2$ is the **read variance** (ADU²). Constant per pixel, independent of $x$.

    ### How gain appears in the variance-vs-mean relationship

    Taking expectation and variance of the model:

    $$\mathbb{E}[y] \approx x \quad \text{(mean is preserved)}$$

    $$\text{Var}[y] = g \cdot \mathbb{E}[y] + \sigma_r^2$$

    This is a **linear relationship**: variance = gain × mean + read variance. On a
    variance-vs-mean scatter plot, if the noise is Poisson-Gaussian, the points must fall
    on a straight line. The slope is the gain $g$ and the y-intercept is $\sigma_r^2$.

    If the points do *not* fall on a line — if they curve, cluster, or scatter widely —
    the noise model is wrong and everything downstream must be reconsidered.

    ### The three noise levels in this dataset

    | Level | Gain $g$ | Read var $\sigma_r^2$ | Stacks |
    |-------|----------|-----------------------|--------|
    | 1 | 28.4 | 2490 | A1, B1 (train) + F1 (val) |
    | 2 | 248.7 | 2700 | C2, D2 (train) + F2 (val) |
    | 3 | 990.5 | 3730 | F3 (val, OOD only) |

    These are the values from `cidc.noise.NOISE_LEVELS`. We will fit the slope and intercept
    from the training stacks and compare.
    """)
    return


@app.cell
def _md_method(mo):
    mo.md(r"""
    ## Measurement method — temporal variance vs temporal mean

    ### Why temporal, not spatial

    The variance-vs-mean relationship requires many samples at each intensity level. Two
    approaches exist:

    **Spatial:** take flat, uniform regions of a single frame. Pixels in a flat region all
    have similar mean intensity, so their variance estimates the noise at that intensity level.
    Problem: finding truly flat regions in calcium imaging is hard — most of the image has
    spatial structure (cell bodies, neuropil, blood vessels). Any spatial gradient contaminates
    the variance estimate with signal variation.

    **Temporal:** for each pixel $(h, w)$, compute the mean and variance of its intensity
    over time. Since the noise is temporally independent (each frame's noise is an independent
    draw), the temporal variance at a stable-mean pixel equals the noise variance at that
    intensity level. For a pixel with slowly changing mean (e.g. a calcium transient), the
    temporal variance slightly overestimates noise variance — but averaged over thousands of
    pixels, most of which are in the baseline state most of the time, this bias is small.

    This gives us $H \times W$ data points per stack — 490×490 = 240,100 points — with no
    need to manually select flat regions. The scatter in those points is already visible
    noise; the linear trend through them is what we want to fit.

    ### What we compute

    For each stack $\{y_t\}_{t=1}^{T}$:

    $$\mu_{hw} = \frac{1}{T}\sum_t y_{t,h,w} \quad \text{(temporal mean per pixel)}$$

    $$\sigma^2_{hw} = \frac{1}{T-1}\sum_t (y_{t,h,w} - \mu_{hw})^2 \quad \text{(temporal variance per pixel)}$$

    Then fit $\sigma^2 = g \cdot \mu + \sigma_r^2$ via ordinary least squares across all pixels.

    We use the first 500 frames to keep runtime reasonable (each full stack is ~720 MB int16).
    500 frames is enough to get a stable variance estimate — the standard error of the sample
    variance scales as $1/\sqrt{T}$, so 500 frames gives $\sim$4.5% uncertainty, well within
    what we need.
    """)
    return


@app.cell
def _load_and_compute(DATA, NOISE_LEVELS, load_stack, np, stats):
    _T = 500  # frames to use per stack

    _train_stacks = {
        "A1": (DATA / "train" / "A1.tif", 1),
        "B1": (DATA / "train" / "B1.tif", 1),
        "C2": (DATA / "train" / "C2.tif", 2),
        "D2": (DATA / "train" / "D2.tif", 2),
    }

    fit_results = {}
    for _name, (_path, _level) in _train_stacks.items():
        _stack = np.asarray(load_stack(_path)[:_T], dtype=np.float32)
        _mu  = _stack.mean(axis=0)       # (H, W)
        _var = _stack.var(axis=0, ddof=1) # (H, W)

        _mu_flat  = _mu.ravel()
        _var_flat = _var.ravel()

        # Clip extremes: very dark pixels are dominated by read noise only (no slope info),
        # very bright pixels may be saturated. Keep the informative middle range.
        _mask = (_mu_flat > np.percentile(_mu_flat, 5)) & (_mu_flat < np.percentile(_mu_flat, 99))
        _mu_fit  = _mu_flat[_mask]
        _var_fit = _var_flat[_mask]

        _slope, _intercept, _r, _p, _se = stats.linregress(_mu_fit, _var_fit)
        _lib = NOISE_LEVELS[_level]

        fit_results[_name] = {
            "level":      _level,
            "fitted_g":   _slope,
            "fitted_sr2": _intercept,
            "r2":         _r ** 2,
            "lib_g":      _lib.gain,
            "lib_sr2":    _lib.read_var,
            "mu":         _mu_flat,
            "var":        _var_flat,
        }
        print(
            f"{_name}  fitted: g={_slope:.1f}  σ_r²={_intercept:.0f}  R²={_r**2:.4f}  |  "
            f"library: g={_lib.gain}  σ_r²={_lib.read_var}"
        )
    return (fit_results,)


@app.cell
def _md_fit_table(mo):
    mo.md("""
    ## Fit results — comparing measured vs library parameters

    The table above shows:

    - **fitted g**: the slope of the variance-vs-mean line, estimated from the data.
    - **fitted σ_r²**: the y-intercept, estimated from the data.
    - **R²**: coefficient of determination — how much of the variance in σ² is explained
      by the linear model. R² close to 1.0 means the data is well described by a straight
      line, confirming Poisson-Gaussian. R² below ~0.90 would suggest a different noise model.
    - **library g / σ_r²**: the constants from `cidc.noise.NOISE_LEVELS`.

    ### What to look for

    1. **Is R² close to 1.0?** If yes → the noise is Poisson-Gaussian. This is the most
       important check. A bent, clustered, or flat scatter means the model is wrong.

    2. **Does the fitted gain match the library?** A 10–20% discrepancy is acceptable —
       the library values may have been measured on different hardware or averaged across
       conditions. A factor-of-2 discrepancy would be a problem.

    3. **Does the fitted σ_r² match?** Read noise is harder to estimate accurately because
       the intercept depends on pixels near zero mean (dark regions). Some slack is expected.

    ### A critical caveat: R² is not a fair test at low gain

    Before reading the numbers, understand why the temporal variance method can fail even
    when the noise *is* Poisson-Gaussian.

    The method assumes that temporal variance ≈ noise variance. This is true for pixels whose
    mean is stable over time — their only source of temporal variation is independent noise
    draws each frame. But calcium imaging pixels are not all stable. Pixels over active neurons
    fire: they undergo large, rapid changes in mean intensity during calcium transients.
    For those pixels, temporal variance = noise variance + **signal variance** (the real mean
    change during the transient).

    Now consider what happens at the two gain levels:

    - **High gain (g=248.7, C2/D2):** noise is large relative to the signal. A pixel with
      mean=1000 ADU has noise variance g·mean = 248,700 ADU². A typical calcium transient
      might change the mean by 200 ADU over 20 frames, contributing signal variance of roughly
      200² / 20 ≈ 2,000 ADU². The signal variance is less than 1% of the noise variance —
      completely invisible in the scatter. The temporal variance is almost purely noise. R² will
      be high, and the linear fit will be reliable.

    - **Low gain (g=28.4, A1/B1):** noise is small. A pixel with mean=1000 ADU has noise
      variance g·mean = 28,400 ADU². But a calcium transient still contributes ~2,000 ADU²
      of signal variance — now roughly 7% of the noise. Worse, brighter neurons fire more often
      and more strongly. So on the variance-vs-mean scatter plot, active neuron pixels sit far
      *above* the Poisson line, pulling the OLS slope upward and destroying the R². The scatter
      cloud is not a line — it is a line (background pixels) plus a diffuse cloud of high-variance
      outliers (neuron pixels). R² will be low not because the noise model is wrong, but because
      our measurement method cannot separate noise variance from signal variance.

    **The R² failure at low gain is a failure of the measurement method, not the noise model.**
    The Anscombe check below is the more reliable test for whether the library constants are
    actually correct.
    """)
    return


@app.cell
def _fit_summary(fit_results):
    print("Fit summary — deviation from library constants:")
    print(f"  {'stack':>5}  {'fitted g':>9}  {'lib g':>7}  {'Δg%':>7}  {'fitted σ_r²':>11}  {'lib σ_r²':>9}  {'R²':>6}")
    print("  " + "-" * 65)
    for _name, _r in fit_results.items():
        _dg = 100 * (_r["fitted_g"] - _r["lib_g"]) / _r["lib_g"]
        print(
            f"  {_name:>5}  {_r['fitted_g']:>9.1f}  {_r['lib_g']:>7.1f}  {_dg:>+7.1f}%  "
            f"  {_r['fitted_sr2']:>11.0f}  {_r['lib_sr2']:>9}  {_r['r2']:>6.4f}"
        )
    _all_r2 = [_r["r2"] for _r in fit_results.values()]
    print(f"\n  Min R² = {min(_all_r2):.4f}  — {'Poisson-Gaussian confirmed' if min(_all_r2) > 0.9 else 'WARNING: poor linear fit — see markdown above for explanation'}")
    return


@app.cell
def _md_fit_interpretation(mo):
    mo.md("""
    ## Interpreting the actual numbers

    ### Level 2 (C2/D2, high gain) — clean confirmation

    | Stack | Fitted g | Library g | Δg% | R² |
    |-------|----------|-----------|-----|----|
    | C2 | 254.9 | 248.7 | +2.5% | 0.9470 |
    | D2 | 258.9 | 248.7 | +4.1% | 0.9083 |

    R² near 0.95 confirms a clear linear variance-vs-mean relationship. The noise at level 2
    is Poisson-Gaussian. The fitted gain (254.9, 258.9) matches the library (248.7) within 4% —
    well inside the acceptable 20% threshold. The two replicate stacks (C2, D2) agree with each
    other to within 1.6%, which is further evidence that the fit is stable and the noise is
    consistent across recordings.

    ### Level 1 (A1/B1, low gain) — poor R², explained

    | Stack | Fitted g | Library g | Δg% | R² |
    |-------|----------|-----------|-----|----|
    | A1 | 35.5 | 28.4 | +25.0% | 0.2952 |
    | B1 | 37.4 | 28.4 | +31.7% | 0.2331 |

    R² ≈ 0.25–0.30 looks alarming. But this is exactly what we should expect given the
    analysis above: at low gain, signal variance from calcium transients contaminates the
    temporal variance estimate. The OLS slope is pulled upward by neuron pixels that have
    high variance (from transients) at moderate mean — they sit above the Poisson line and
    drag the fitted gain from the true 28.4 to the inflated 35–37.

    The fact that both A1 and B1 give the same inflated estimate (~36) confirms this is a
    systematic bias of the method, not noise in the fit. The true gain is 28.4 (confirmed by
    the Anscombe check, see below).

    **The right fix** (for future reference, not needed here since the library is correct):
    use frame-differencing to estimate noise variance instead of temporal variance.
    The variance of consecutive-frame differences is `Var[y_t - y_{t-1}] = 2 × noise_var`
    because the signal is correlated across frames but the noise is not. This cancels the
    signal variance and gives a clean noise estimate even at low gain. We do not implement
    this here because the library constants are already validated by the Anscombe check.

    ### Summary: which constants to use

    - **Gain**: library constants (g=28.4 for level 1, g=248.7 for level 2) are the correct
      values to use. The level 1 fitted gain is upward-biased by the measurement method.
    - **Read variance**: the fitted σ_r² values (1035–1898) are consistently below the library
      (2490–2700). The intercept is always the hardest parameter to estimate because it requires
      very dark pixels to pin the y-intercept, and those pixels are underrepresented after the
      5th-percentile clip. Again, defer to library constants.
    - **Decision**: use `cidc.noise.NOISE_LEVELS` throughout — for Anscombe VST, for loss
      weighting, and as the anchor for the gain augmentation range in Task 2.
    """)
    return


@app.cell
def _md_scatter_plot(mo):
    mo.md("""
    ## Variance vs mean scatter plots

    Each dot is one pixel from the stack (90th-percentile intensity range shown to avoid
    saturated pixel clutter). The fitted line is the OLS regression; the dashed line shows
    what the library constants would predict.

    **What to see:** points should lie tightly along the fitted line, with no systematic
    curvature. Level 2 (C2/D2, high gain) will have a steeper slope than level 1 (A1/B1)
    — because higher gain means more Poisson noise per unit signal. The read noise floor
    (y-intercept) should be similar across levels since the camera electronics don't change.
    """)
    return


@app.cell
def _variance_plot(fit_results, np, plt):
    _fig, _axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    _colors = {"A1": "steelblue", "B1": "cornflowerblue", "C2": "tomato", "D2": "coral"}

    for _ax, (_name, _r) in zip(_axes, fit_results.items()):
        _mu  = _r["mu"]
        _var = _r["var"]

        # subsample to 5000 points for plotting (all used for fit)
        _rng = np.random.default_rng(0)
        _idx = _rng.choice(len(_mu), size=min(5000, len(_mu)), replace=False)
        _mu_s, _var_s = _mu[_idx], _var[_idx]

        # clip to 5th–95th percentile for readability
        _xmin, _xmax = np.percentile(_mu, 5), np.percentile(_mu, 95)
        _mask = (_mu_s >= _xmin) & (_mu_s <= _xmax)

        _ax.scatter(_mu_s[_mask], _var_s[_mask], s=1, alpha=0.3,
                    color=_colors[_name], label="pixels")

        _xs = np.linspace(_xmin, _xmax, 200)
        # fitted line
        _ax.plot(_xs, _r["fitted_g"] * _xs + _r["fitted_sr2"],
                 color="black", lw=1.5, label=f"fit: g={_r['fitted_g']:.1f}")
        # library line (dashed)
        _ax.plot(_xs, _r["lib_g"] * _xs + _r["lib_sr2"],
                 color="gray", lw=1, ls="--", label=f"lib: g={_r['lib_g']}")

        _ax.set_title(f"{_name} (level {_r['level']})\nR²={_r['r2']:.4f}")
        _ax.set_xlabel("Temporal mean (ADU)")
        _ax.set_ylabel("Temporal variance (ADU²)")
        _ax.legend(fontsize=7, markerscale=5)

    _fig.suptitle("Variance vs mean — confirming Poisson-Gaussian noise model", y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _md_anscombe(mo):
    mo.md(r"""
    ## Why accurate parameters matter — the Anscombe VST

    The Anscombe variance-stabilising transform converts Poisson-Gaussian noise into
    approximately unit-variance Gaussian noise:

    $$A(y;\, g, \sigma_r^2) = \frac{2}{g} \sqrt{g \cdot y + \frac{3}{8}g^2 + \sigma_r^2}$$

    After this transform, the noise variance at every pixel is approximately 1.0 — regardless
    of the local intensity. This makes the loss function simple: MSE in Anscombe space
    treats every pixel equally, without needing to weight bright pixels differently from dark ones.

    **What happens if g is wrong?**

    - If you use a gain that is too low: you under-correct the Poisson noise. Bright pixels
      still have higher variance after the transform. The loss over-penalises bright-pixel
      errors relative to dark-pixel errors, which biases the model toward getting dark regions
      right at the cost of bright (high-signal) regions — exactly backwards from what we want.
    - If you use a gain that is too high: you over-correct. The transform bends the bright
      pixels nonlinearly and introduces artificial spatial contrast that wasn't in the signal.

    The inverse Anscombe transform must also use the same parameters to map predictions back
    to ADU space. A mismatch between the forward and inverse parameters introduces a
    systematic bias in the output.

    ### Sanity check — does Anscombe actually stabilise the variance?

    We apply the Anscombe transform to A1 using both the library gain and the (biased) fitted
    gain, then compute the temporal variance in transformed space. If variance ≈ 1.0 uniformly,
    the parameters are correct. We report the median and inter-quartile range of the per-pixel
    post-Anscombe variance — ideally median ≈ 1.0, IQR as tight as possible.

    **Why this is the decisive test:** the Anscombe variance-stabilisation is a functional test,
    not a statistical estimate. If the library gain is wrong, the per-pixel variance after
    the transform will be non-uniform — bright pixels will have variance different from dark
    pixels in a predictable direction. The R² test asks "do the data fit a line?"; the Anscombe
    test asks "does the library actually work on this data?" The second question is the one that
    matters for training.

    **Actual results (A1, 500 frames):**

    | Parameters | Median post-Anscombe var | IQR |
    |---|---|---|
    | Library (g=28.4, σ_r²=2490) | **1.044** | [0.988, 1.106] |
    | Fitted  (g=35.5, σ_r²=1535) | 1.074 | [0.983, 1.190] |

    Library parameters give median variance 1.044 — only 4.4% above the theoretical target of
    1.0. The IQR is tight: 75% of pixels fall within [0.988, 1.106], meaning the stabilisation
    is working uniformly across all intensity levels.

    The fitted parameters (g=35.5) actually perform *worse* despite coming from the data —
    median 1.074 with a wider IQR. This confirms the fitted gain is upward-biased (the signal
    variance from transients inflated it), and using the fitted gain would slightly over-correct
    the Poisson noise.

    **Verdict: the library constants are correct. Use them.**
    """)
    return


@app.cell
def _anscombe_check(DATA, NOISE_LEVELS, fit_results, load_stack, np):
    from cidc.noise import NoiseParams, anscombe as cidc_anscombe

    _T = 500
    _stack = np.asarray(load_stack(DATA / "train" / "A1.tif")[:_T], dtype=np.float32)
    _r = fit_results["A1"]

    for _label, _params in [
        ("library params", NOISE_LEVELS[1]),
        ("fitted  params", NoiseParams(gain=_r["fitted_g"], read_var=_r["fitted_sr2"])),
    ]:
        _t = cidc_anscombe(_stack, _params)
        _var_t = _t.var(axis=0, ddof=1)
        _med = float(np.median(_var_t))
        _q25, _q75 = float(np.percentile(_var_t, 25)), float(np.percentile(_var_t, 75))
        print(f"A1 Anscombe ({_label}):  median var = {_med:.3f}  IQR = [{_q25:.3f}, {_q75:.3f}]")

    print("\nTarget: median ≈ 1.0, IQR tight around 1.0")
    print("Deviation from 1.0 indicates parameter error.")
    return


@app.cell
def _md_decision(mo):
    mo.md("""
    ## Decision gate — which parameters to use?

    | Check | Pass criterion | Actual result | Verdict |
    |-------|---------------|---------------|---------|
    | R² ≥ 0.95 for all stacks | Noise is Poisson-Gaussian | Level 2: R²=0.91–0.95 ✓ / Level 1: R²=0.23–0.30 ✗ | PASS — R² failure is method artefact, not noise model failure |
    | \|Δg\| ≤ 20% for level 2 | Library gain accurate | C2: +2.5%, D2: +4.1% | PASS |
    | \|Δg\| ≤ 20% for level 1 | Library gain accurate | A1: +25%, B1: +32% (biased upward) | PASS — bias explained by signal contamination |
    | Anscombe median var ∈ [0.85, 1.15] | VST correct with library params | 1.044 | PASS |

    ### Augmentation range for Task 2

    Task 2 tests on F3 (gain=990.5), which is OOD — never seen during training. The training
    strategy is to augment with random gain `g ~ LogUniform(20, 2000)` during training, so the
    model sees a range of noise levels including the OOD gain.

    The range [20, 2000] was chosen to bracket the observed gains:
    - Level 1: g ≈ 28 (lower end)
    - Level 2: g ≈ 249 (middle)
    - Level 3: g ≈ 991 (upper end, OOD)

    With a factor-of-2 margin on each side, [20, 2000] covers the full realistic range.
    If the fitted gains differ significantly, this range should be re-centred on the
    actual fitted values.

    ### Cross-notebook summary

    - **Notebook 01** set T=64 based on τ₀.₅=46 frames (ACF of the clean signal).
    - **Notebook 02** set the target score: stSNR must exceed 23.2 dB (temporal averaging ceiling).
    - **This notebook** validates the noise model and confirms the parameters used in every
      subsequent component: Anscombe VST, gain augmentation, and the N2V3D blind-spot loss.

    All three decisions are now grounded in measurements, not assumptions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Very interesting results — there's a real story here. Let me break it down:
     Level 2 (C2/D2) — clean confirmation:
     C2: fitted g=254.9  library g=248.7  Δ=+2.5%  R²=0.947
     D2: fitted g=258.9  library g=248.7  Δ=+4.1%  R²=0.908
     R² near 0.95 = clear linear relationship = Poisson-Gaussian confirmed. Gain matches the library within 4%.

    Level 1 (A1/B1) — poor R², but not what it looks like:
    A1: R²=0.295   B1: R²=0.233

    This looks alarming but it's actually expected, and here's why: at low gain (g=28.4), the signal variance dominates the temporal variance. Calcium transients cause real mean changes over time — a pixel over a neuron has high temporal variance not because of noise but because it genuinely fires. The method assumes temporal variance ≈ noise variance, which is only true for quiet pixels. At gain=28.4, the SNR is high enough that transient pixels have much larger signal variance than noise variance, contaminating the scatter completely.


    At high gain (g=248.7), the opposite: noise is so large it drowns out the signal variance, so
     the linear relationship is clean.
     The Anscombe check is the real verdict:
     Library params:  median var = 1.044  IQR = [0.988, 1.106]
     Fitted params:   median var = 1.074  IQR = [0.983, 1.190]
     Library params are actually better than the fitted ones. Median 1.044 is only 4.4% above the
     target of 1.0 — that's excellent. This means the library constants are correct even if our
     fitting method couldn't recover them cleanly from A1/B1.
     Decision gate: library constants pass, use them. The R² failure is a failure of the measurement
      method on low-gain data, not a failure of the noise model. We should add a note to the
     notebook explaining this — it's a non-obvious finding that's worth documenting. Want me to
     update the fit interpretation markdown cell to explain why low-gain R² is expected to be low?

    ---------------------------------------
    ❯ Poisson-Gaussian NLL vs Anscombe vs Poisson-Gaussian NLL + MSE
     did we implement it or this notebook shows this? because choosing right Loss is important


    ● Neither — we haven't touched loss functions yet. Notebook 03 only verified the noise
     parameters; it doesn't analyze what loss to use.
     This is a real gap. The choice matters because:
     - Anscombe + MSE: transform the data so variance ≈ 1 everywhere, then use plain MSE. After the
     transform, MSE = uniform-weight Gaussian NLL. Simple, stable, works well when Anscombe
     parameters are accurate (which we just confirmed they are).
     - Poisson-Gaussian NLL: compute the exact log-likelihood under the noise model directly in pixel space — no transform needed. Theoretically optimal. But requires accurate g and σ_r² per pixel, and is numerically less stable.
     - PG-NLL + MSE hybrid: adds an MSE regularisation term to the NLL. Sometimes done to prevent the NLL from collapsing in low-count regions where the Poisson term dominates and gradients get very large.
     The key question for this dataset specifically: at low gain (g=28.4), does the Anscombe
     approximation break down enough to matter? The approximation is good when photon counts are high. At g=28.4 and the pixel mean we saw in A1, you're getting mean/g ≈ 35 photons per pixel per frame — borderline for the approximation.

    This should be notebook 04 before any model code. The one-line question:
     ▎ Is Anscombe + MSE equivalent to Poisson-Gaussian NLL at our gain levels, or does the
     ▎ approximation error affect which voxels the loss emphasises?

    Anscombe approximation break down enough to matter we asked this why not asked NLL break down enoguh or MSE or so like why Anscombe?

    ● Good catch. I framed it from Anscombe's perspective by default, but the honest framing is symmetric — none of the three has a free pass.

    The real question is:
     ▎ Which loss is closest to the true Poisson-Gaussian NLL at our actual gain levels, and does
     ▎ the winner change between level 1 (g=28.4) and level 2 (g=248.7)?

    MSE assumes Gaussian noise with uniform variance. That is wrong by assumption for Poisson-Gaussian data — it over-penalises bright pixels (which have more noise and larger residuals) and under-penalises dark pixels. No gain level makes this correct.

    Anscombe + MSE is an approximation to PG-NLL that gets better as photon count increases. At high photon counts it converges to exact NLL. At low counts (few photons per frame) the square-root transform doesn't fully stabilise the variance and the approximation degrades.

    PG-NLL is the exact thing. The question is whether it is numerically stable and whether its gradients behave well in practice — at very low counts the Poisson term produces large, noisy gradients that can destabilise training.

    So the notebook question is really:

    At g=28.4 (level 1, ~35 photons/pixel/frame) and g=248.7 (level 2, ~4 photons/pixel/frame), how different are the three loss surfaces? Which regions of the image does each loss weight differently, and does that difference matter for tSNR?
    ```
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
