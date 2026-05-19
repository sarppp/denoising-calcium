"""04 — Loss function comparison: MSE vs Anscombe+MSE vs PG-NLL."""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 04 — Loss function comparison
    ## MSE vs Anscombe+MSE vs Poisson-Gaussian NLL

    **QUESTION:** At our actual gain levels (g=28.4 and g=248.7), do MSE, Anscombe+MSE,
    and Poisson-Gaussian NLL weight pixels differently — and does the winner change between
    gain levels?

    **Decision gate:** picks the training loss for N2V3D before writing any model code.
    If the three losses produce nearly identical per-pixel weights at our gains → any of
    them works. If they diverge → the choice changes what the model learns to prioritise,
    and through that, which pixels improve, and through that, tSNR.
    """)
    return


@app.cell
def _setup():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from cidc import load_stack
    from cidc.noise import NOISE_LEVELS

    DATA = Path(__file__).parent.parent.parent / "data"
    return DATA, NOISE_LEVELS, load_stack, mo, np, plt


@app.cell
def _intro(mo):
    mo.md("""
    # 04 — Loss function comparison

    **Purpose:** understand what each candidate loss function actually optimises
    before writing a single line of model code.

    Three candidates:

    1. **MSE** — mean squared error directly in pixel space. Assumes every pixel
       has the same noise variance. Simple, numerically stable, but wrong for
       Poisson-Gaussian noise.

    2. **Anscombe + MSE** — apply the Anscombe variance-stabilising transform first,
       then compute MSE in the transformed space. After the transform, noise variance
       ≈ 1 everywhere, so MSE in Anscombe space ≈ uniform-weight Gaussian NLL.
       An approximation — accurate when photon counts are high.

    3. **PG-NLL** — Poisson-Gaussian negative log-likelihood. The exact statistical
       loss for this noise model. Each pixel is weighted by the inverse of its local
       noise variance, so noisy (bright) pixels contribute less to the gradient.

    ### Why this is not an obvious choice

    All three losses are minimised by the same optimal denoiser in the limit of
    infinite data and infinite model capacity. The difference is in **what the
    loss emphasises during finite training**: which pixels drive the gradients most,
    and therefore which pixels the model learns to get right first and most
    accurately. With a finite model and finite training budget, the loss choice
    determines where the model spends its capacity.

    The connection to tSNR is direct. tSNR measures the temporal trace accuracy of
    individual pixels — it cares most about active neurons (pixels that fire, i.e.
    have high temporal variance and often high mean intensity). If the loss
    down-weights bright pixels (as PG-NLL does), the model is less incentivised to
    get those pixels right. If the loss over-weights them (as MSE implicitly does via
    large residuals), the model focuses on them at the expense of quiet background.

    This notebook measures the per-pixel weight each loss assigns at our actual gain
    levels and pixel intensity distributions, and traces that through to its
    implication for tSNR.
    """)
    return


@app.cell
def _md_math(mo):
    mo.md(r"""
    ## The math — what each loss actually computes

    Let $y$ be the observed (noisy) pixel value and $\hat{y}$ be the model's
    prediction of the clean signal. The residual is $r = \hat{y} - y$.

    ### 1. MSE

    $$\mathcal{L}_\text{MSE} = r^2 = (\hat{y} - y)^2$$

    The gradient with respect to $\hat{y}$ is $2r$. Every pixel contributes
    proportionally to its squared residual, regardless of how noisy the pixel is.

    **Effective per-pixel weight:** constant = 1.

    **What's wrong (illustrative example at level 1, g=28.4):** a bright neuron pixel
    with mean $\mu = 2000$ ADU has noise variance $g \cdot \mu = 28.4 \times 2000 = 56{,}800$
    ADU². A dark background pixel with $\mu = 100$ ADU has noise variance $28.4 \times 100
    = 2{,}840$ ADU² — 20× less. MSE treats both identically: the bright pixel produces
    larger residuals purely from noise, not from the model being wrong, so MSE spends
    gradient budget fitting noise at the brightest pixels. The `_weight_numbers` cell
    below computes the actual ratios at both gain levels.

    ### 2. Poisson-Gaussian NLL

    The exact PG distribution is a convolution of Poisson and Gaussian with no
    closed form. For photon counts large enough that Poisson $\approx$ Gaussian
    (which holds approximately for $\lambda > 4$), the combined distribution
    approaches:

    $$p(y \mid x) \approx \mathcal{N}\!\left(y;\; x,\; g \cdot x + \sigma_r^2\right)$$

    where $x$ is the true signal. Since we don't have $x$, we substitute $y$ in
    the variance term (because $\mathbb{E}[y] = x$):

    $$\mathcal{L}_\text{PG-NLL} = \frac{(\hat{y} - y)^2}{g \cdot y + \sigma_r^2} + \log\!\left(g \cdot y + \sigma_r^2\right)$$

    The first term is a **noise-normalised squared error** — each pixel's residual
    is divided by its local noise variance. The second term is a log-normalisation
    that penalises predicting high variance where the signal is actually clean.

    **Effective per-pixel weight:** $w(y) = \dfrac{1}{g \cdot y + \sigma_r^2}$

    Bright pixels have larger $g \cdot y$, so smaller weight. PG-NLL is statistically
    optimal: it correctly tells the model "this bright pixel is inherently noisy,
    don't over-fit to its exact value." But it also tells the model "don't worry too
    much about neuron pixels" — which is in tension with tSNR.

    ### 3. Anscombe + MSE

    The Anscombe transform is:

    $$A(y) = \frac{2}{g} \sqrt{g \cdot y + \tfrac{3}{8}g^2 + \sigma_r^2}$$

    It maps Poisson-Gaussian noise to approximately unit-variance Gaussian noise.
    MSE in Anscombe space is therefore:

    $$\mathcal{L}_\text{Ansc} = \left(A(\hat{y}) - A(y)\right)^2$$

    By the delta method, the effective weight in pixel space is the square of the
    derivative of $A$ with respect to $y$:

    $$w_\text{Ansc}(y) \approx \left(\frac{dA}{dy}\right)^2 = \frac{1}{g \cdot y + \tfrac{3}{8}g^2 + \sigma_r^2}$$

    This is **identical in form to PG-NLL weight**, but with an additional
    $\tfrac{3}{8}g^2$ term in the denominator — a small offset that softens the
    down-weighting of very dark pixels. At moderate-to-high photon counts the two
    losses are nearly identical. The Anscombe approximation degrades at very low
    photon counts ($\lambda \lesssim 4$) where the square-root transformation no
    longer fully stabilises the variance.

    ### The key comparison

    | Loss | Weight $w(y)$ | Bright pixel weight | Dark pixel weight |
    |------|--------------|--------------------|--------------------|
    | MSE | $1$ | high (large residuals) | low |
    | PG-NLL | $1/(g y + \sigma_r^2)$ | low | high |
    | Anscombe+MSE | $1/(g y + \tfrac{3}{8}g^2 + \sigma_r^2)$ | low (softened) | high (softened) |

    The question is: **how large is the difference at our actual gain levels?**
    """)
    return


@app.cell
def _md_weight_curves(mo):
    mo.md("""
    ## Weight curves — how much each loss cares about each pixel intensity

    We plot the effective per-pixel weight $w(y)$ as a function of pixel intensity
    $y$ for both gain levels. MSE weight is constant (1, normalised). PG-NLL and
    Anscombe weights fall as intensity rises.

    **What to look for:**
    - At what intensity does the PG-NLL weight drop to half the MSE weight?
      That is the crossover where PG-NLL starts meaningfully ignoring a pixel.
    - How different are PG-NLL and Anscombe+MSE from each other?
      If they overlap, either works — choose the simpler one.
    - Does the crossover differ between gain levels? At high gain (g=248.7),
      even a moderately bright pixel has enormous noise variance and will be
      heavily down-weighted by PG-NLL.
    """)
    return


@app.cell
def _weight_curves(NOISE_LEVELS, np, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))

    for _ax, _level, _title in [
        (_axes[0], 1, "Level 1 — g=28.4, σ_r²=2490  (A1, B1)"),
        (_axes[1], 2, "Level 2 — g=248.7, σ_r²=2700  (C2, D2)"),
    ]:
        _p = NOISE_LEVELS[_level]
        _g, _sr2 = _p.gain, _p.read_var

        _y = np.linspace(10, 3000, 500)

        _w_pgnll = 1.0 / (_g * _y + _sr2)
        _w_ansc  = 1.0 / (_g * _y + 0.375 * _g**2 + _sr2)
        _w_mse   = np.ones_like(_y)

        # normalise all to 1 at the darkest pixel so shapes are comparable
        _w_pgnll /= _w_pgnll[0]
        _w_ansc  /= _w_ansc[0]

        _ax.plot(_y, _w_mse,   color="steelblue", lw=1.5, label="MSE (constant)")
        _ax.plot(_y, _w_pgnll, color="tomato",    lw=1.5, label="PG-NLL")
        _ax.plot(_y, _w_ansc,  color="seagreen",  lw=1.5, ls="--", label="Anscombe+MSE")

        # mark where PG-NLL weight = 0.5 (half the dark-pixel weight)
        _half_idx = np.searchsorted(-_w_pgnll, -0.5)
        if _half_idx < len(_y):
            _ax.axvline(_y[_half_idx], color="tomato", ls=":", lw=1, alpha=0.6,
                        label=f"PG-NLL half-weight at {_y[_half_idx]:.0f} ADU")

        _ax.set_xlabel("Pixel intensity y (ADU)")
        _ax.set_ylabel("Effective weight (normalised to dark pixel)")
        _ax.set_title(_title)
        _ax.legend(fontsize=8)
        _ax.set_ylim(0, 1.1)
        _ax.grid(alpha=0.2)

    _fig.suptitle("Per-pixel effective weight: how much each loss cares about each intensity level")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _weight_numbers(NOISE_LEVELS):
    print("Half-weight intensity (where PG-NLL weight = 0.5 × dark-pixel weight):")
    print("  At this intensity, PG-NLL gives a pixel half the gradient importance of a dark pixel.\n")
    for _level in [1, 2]:
        _p = NOISE_LEVELS[_level]
        _g, _sr2 = _p.gain, _p.read_var
        # w(y) = 1/(g*y + sr2).  half of w(0) = 1/sr2  →  g*y + sr2 = 2*sr2  →  y = sr2/g
        _y_half = _sr2 / _g
        print(f"  Level {_level} (g={_g}, σ_r²={_sr2}):  half-weight at y = σ_r²/g = {_y_half:.0f} ADU")
        _photons = _y_half / _g
        print(f"    → {_photons:.1f} photons/pixel/frame — {'well into signal range' if _y_half > 200 else 'very dark, near read-noise floor'}")

    print()
    print("Weight ratio PG-NLL / MSE at typical neuron brightness (y=1500 ADU):")
    for _level in [1, 2]:
        _p = NOISE_LEVELS[_level]
        _g, _sr2 = _p.gain, _p.read_var
        _y_neuron = 1500.0
        _w_pgnll = 1.0 / (_g * _y_neuron + _sr2)
        _w_dark  = 1.0 / _sr2  # darkest pixel weight
        _ratio = _w_pgnll / _w_dark
        print(f"  Level {_level}: PG-NLL weights neuron at {_ratio:.3f}× the dark-pixel weight  ({_ratio*100:.1f}%)")

    print()
    print("Weight ratio Anscombe+MSE vs PG-NLL at y=1500 ADU (how different are they):")
    for _level in [1, 2]:
        _p = NOISE_LEVELS[_level]
        _g, _sr2 = _p.gain, _p.read_var
        _y = 1500.0
        _w_pgnll = 1.0 / (_g * _y + _sr2)
        _w_ansc  = 1.0 / (_g * _y + 0.375 * _g**2 + _sr2)
        _diff_pct = 100 * abs(_w_pgnll - _w_ansc) / _w_pgnll
        print(f"  Level {_level}: Anscombe weight differs from PG-NLL by {_diff_pct:.1f}%")
    return


@app.cell
def _md_weight_maps(mo):
    mo.md("""
    ## Weight maps on real data — spatial view

    The curves above show weight as a function of intensity. Now we apply those
    weights to an actual frame from A1 (level 1) and C2 (level 2) to see
    *spatially* which pixels each loss emphasises.

    **The critical question:** do the highly-weighted pixels correlate with active
    neurons — the pixels tSNR cares about most?

    We use the temporal standard deviation map as a proxy for neuron activity: pixels
    with high temporal std are either noisy (background) or genuinely firing (neurons).
    We then compute the correlation between the loss weight map and the temporal std map.

    - **MSE weight map = uniform** → correlation with activity = 0 by construction.
    - **PG-NLL weight map** → negatively correlated with intensity → negatively
      correlated with neuron activity (bright neurons get down-weighted).
    - **Anscombe weight map** → similar to PG-NLL but softer.

    High negative correlation means the loss is actively de-emphasising the pixels
    that fire, which is in tension with recovering temporal dynamics.
    """)
    return


@app.cell
def _weight_maps(DATA, NOISE_LEVELS, load_stack, np):
    _T_map = 200
    weight_map_results = {}

    for _name, _level in [("A1", 1), ("C2", 2)]:
        _stack = np.asarray(
            load_stack(DATA / "train" / f"{_name}.tif")[:_T_map], dtype=np.float32
        )
        _frame = _stack[0]
        _temporal_std = _stack.std(axis=0)

        _p = NOISE_LEVELS[_level]
        _g, _sr2 = _p.gain, _p.read_var

        _w_pgnll = 1.0 / (_g * _frame + _sr2)
        _w_ansc  = 1.0 / (_g * _frame + 0.375 * _g**2 + _sr2)

        # correlation between weight map and temporal std (proxy for neuron activity)
        _corr_pg   = float(np.corrcoef(_w_pgnll.ravel(), _temporal_std.ravel())[0, 1])
        _corr_ansc = float(np.corrcoef(_w_ansc.ravel(),  _temporal_std.ravel())[0, 1])

        weight_map_results[_name] = {
            "frame": _frame, "temporal_std": _temporal_std,
            "w_pgnll": _w_pgnll, "w_ansc": _w_ansc,
            "corr_pg": _corr_pg, "corr_ansc": _corr_ansc, "level": _level,
        }
        print(f"{_name} (level {_level}):  "
              f"corr(PG-NLL weight, activity) = {_corr_pg:+.3f}  |  "
              f"corr(Anscombe weight, activity) = {_corr_ansc:+.3f}")

    print("\nNegative correlation = loss down-weights active neurons.")
    print("Zero = no relationship (MSE by definition).")
    return (weight_map_results,)


@app.cell
def _weight_map_plot(np, plt, weight_map_results):
    _fig, _axes = plt.subplots(2, 4, figsize=(16, 7))

    for _row, (_name, _r) in enumerate(weight_map_results.items()):
        # raw frame
        _axes[_row, 0].imshow(_r["frame"], cmap="gray",
                               vmin=np.percentile(_r["frame"], 1),
                               vmax=np.percentile(_r["frame"], 99))
        _axes[_row, 0].set_title(f"{_name} — raw frame")

        # temporal std (activity proxy)
        _axes[_row, 1].imshow(_r["temporal_std"], cmap="hot",
                               vmin=0, vmax=np.percentile(_r["temporal_std"], 99))
        _axes[_row, 1].set_title("Temporal std (activity proxy)")

        # PG-NLL and Anscombe share the same log-scale vmin/vmax so they are
        # directly comparable — otherwise auto-scale makes them look different
        # even when they differ by <6%
        _w_min = min(_r["w_pgnll"].min(), _r["w_ansc"].min())
        _w_max = max(_r["w_pgnll"].max(), _r["w_ansc"].max())
        _log_pg   = np.log10(np.clip(_r["w_pgnll"], _w_min, None))
        _log_ansc = np.log10(np.clip(_r["w_ansc"],  _w_min, None))
        _vmin_log, _vmax_log = np.log10(_w_min), np.log10(_w_max)

        _im = _axes[_row, 2].imshow(_log_pg, cmap="viridis",
                                     vmin=_vmin_log, vmax=_vmax_log)
        _axes[_row, 2].set_title(f"PG-NLL weight (log₁₀)\ncorr w/ activity={_r['corr_pg']:+.3f}")

        _axes[_row, 3].imshow(_log_ansc, cmap="viridis",
                               vmin=_vmin_log, vmax=_vmax_log)
        _axes[_row, 3].set_title(f"Anscombe+MSE weight (log₁₀)\ncorr w/ activity={_r['corr_ansc']:+.3f}")

        _fig.colorbar(_im, ax=_axes[_row, 3], fraction=0.03, label="log₁₀(weight)")

        for _ax in _axes[_row]:
            _ax.axis("off")

    _fig.suptitle(
        "Weight maps (shared log scale per row) — PG-NLL vs Anscombe+MSE vs neuron activity",
        y=1.01,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _md_tsnr_connection(mo):
    mo.md("""
    ## The tSNR connection — what the weight maps mean for training

    tSNR measures the temporal trace accuracy of each pixel — it rewards the model
    for correctly tracking the rise and fall of calcium transients at every pixel.
    The pixels that dominate the tSNR score are the ones that fire: high temporal
    variance, often high mean intensity.

    Now look at the correlation numbers above and what they imply:

    **MSE:** weight is uniform. The model gets equal gradient signal from every
    pixel. Neurons get the same budget as background. This sounds fair, but it is
    not — bright noisy pixels have *larger residuals* on average (because they have
    more noise), so in practice MSE over-trains on high-noise pixels. The gradient
    variance is high and the model wastes capacity fitting noise at bright pixels.

    **PG-NLL:** weight ∝ 1/noise_variance. Bright neurons are explicitly
    down-weighted. The model is told "don't worry about getting this bright noisy
    pixel exactly right." From a statistical estimation standpoint this is correct —
    an unbiased estimator should weight observations by their precision. But it
    means the model has less gradient signal from the pixels tSNR cares about most.

    **Anscombe + MSE:** weight ∝ 1/(noise_variance + constant). Almost identical
    to PG-NLL but with a small additive offset in the denominator that softens the
    down-weighting of bright pixels slightly. The practical difference from PG-NLL
    is small at our intensity levels.

    ### The resolution: what actually matters for tSNR

    The tension seems to favour MSE for tSNR. But this reasoning is incomplete.

    MSE over-training on bright-pixel noise means the model learns to explain noise
    fluctuations at neurons — which produces *temporal noise in the output trace*,
    which is exactly what tSNR penalises. The model fitted under MSE will chase
    the noise at neuron pixels, making the output temporally erratic there.

    PG-NLL or Anscombe+MSE tell the model: "bright pixels are inherently noisy,
    learn the smooth underlying signal, not the fluctuations." A model trained
    this way will produce cleaner temporal traces at neuron pixels — the right
    thing for tSNR.

    The correct reasoning: **statistical correctness and tSNR are aligned, not
    opposed.** PG-NLL/Anscombe are not de-emphasising neurons — they are
    de-emphasising *noise at neurons*. That is exactly what tSNR rewards.

    MSE is the dangerous choice precisely because it treats every fluctuation in
    a bright pixel as a signal to fit, which produces temporally noisy output.
    """)
    return


@app.cell
def _md_photon_counts(mo):
    mo.md(r"""
    ## Photon counts — when does the Gaussian approximation break down?

    The PG-NLL derivation above used the Gaussian approximation to Poisson.
    This approximation is valid when the expected photon count $\lambda = x/g$
    is large enough that Poisson($\lambda$) $\approx$ $\mathcal{N}(\lambda, \lambda)$.
    A common rule of thumb is $\lambda \geq 4$.

    For Anscombe+MSE, the stabilisation accuracy degrades at low photon counts for
    the same reason — the square-root transform is derived assuming the Poisson is
    well-approximated by Gaussian.

    We compute the photon count distribution for both gain levels using the temporal
    mean as a proxy for true signal intensity.
    """)
    return


@app.cell
def _photon_counts(DATA, NOISE_LEVELS, load_stack, np, plt):
    _T_ph = 200
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    photon_stats = {}

    for _ax, (_name, _level) in zip(_axes, [("A1", 1), ("C2", 2)]):
        _stack = np.asarray(
            load_stack(DATA / "train" / f"{_name}.tif")[:_T_ph], dtype=np.float32
        )
        _mu = _stack.mean(axis=0).ravel()
        _g = NOISE_LEVELS[_level].gain
        _photons = np.maximum(_mu, 0) / _g

        _frac_below_4 = 100 * (_photons < 4).mean()
        _frac_below_1 = 100 * (_photons < 1).mean()
        _median_ph = float(np.median(_photons))

        photon_stats[_name] = {
            "level": _level, "gain": _g,
            "median_lambda": _median_ph,
            "frac_below_4": _frac_below_4,
            "frac_below_1": _frac_below_1,
        }

        _ax.hist(_photons, bins=100, range=(0, 200), color="steelblue", alpha=0.7)
        _ax.axvline(4, color="tomato", ls="--", lw=1.5, label="λ=4 (approx. breaks down)")
        _ax.set_xlabel("Expected photons per pixel per frame (λ = μ / g)")
        _ax.set_ylabel("Pixel count")
        _ax.set_title(f"{_name} (g={_g})\nmedian λ={_median_ph:.1f}  |  "
                      f"{_frac_below_4:.1f}% pixels below λ=4  |  {_frac_below_1:.1f}% below λ=1")
        _ax.legend(fontsize=8)
        print(f"{_name} (level {_level}, g={_g}):  "
              f"median λ={_median_ph:.1f} photons  |  "
              f"{_frac_below_4:.1f}% pixels have λ<4  |  "
              f"{_frac_below_1:.1f}% have λ<1")

    _fig.suptitle("Photon count distribution — where do Gaussian approximations break down?")
    _fig.tight_layout()
    _fig
    return (photon_stats,)


@app.cell
def _photon_interpretation(photon_stats):
    print("Gaussian approximation validity — interpretation from measured photon counts:")
    print()
    for _name, _s in photon_stats.items():
        _f4 = _s["frac_below_4"]
        _f1 = _s["frac_below_1"]
        if _f4 < 5:
            _verdict = "Gaussian approx VALID for the vast majority of pixels"
        elif _f4 < 30:
            _verdict = f"Gaussian approx QUESTIONABLE — {_f4:.1f}% of pixels in breakdown zone"
        elif _f4 < 80:
            _verdict = f"Gaussian approx POOR — {_f4:.1f}% of pixels below λ=4"
        else:
            _verdict = f"Gaussian approx INVALID — {_f4:.1f}% of pixels below λ=4 ({_f1:.1f}% below λ=1)"
        print(f"  {_name} (level {_s['level']}, g={_s['gain']}):  {_verdict}")

    print()
    print("Implication for loss choice:")
    for _name, _s in photon_stats.items():
        _f4 = _s["frac_below_4"]
        if _f4 > 90:
            print(f"  {_name}: PG-NLL (Gaussian approx) and Anscombe+MSE are both approximations")
            print(f"         for essentially ALL pixels. Neither is exact. Exact PG-NLL requires")
            print(f"         summing over Poisson mass function — intractable in training loops.")
            print(f"         In practice both are used anyway; the error is accepted.")
        elif _f4 > 20:
            print(f"  {_name}: Approximation holds for ~{100-_f4:.0f}% of pixels.")
            print(f"         The {_f4:.0f}% in the breakdown zone are mostly dark background pixels")
            print(f"         — verify by checking whether active neurons are above λ=4.")
    return


@app.cell
def _md_decision(mo):
    mo.md(r"""
    ## Decision gate — which loss to use?

    ### What the photon counts tell us

    The histograms above show the exact distribution — read the printed percentages
    for the precise fractions. The key threshold is $\lambda = 4$: below this, the
    Gaussian approximation to Poisson breaks down and both PG-NLL (Gaussian approx)
    and Anscombe+MSE lose accuracy.

    - **Level 1 (g=28.4):** higher photon counts because lower gain means more
      photons per ADU. Check the histogram: what fraction sits below $\lambda=4$?
      The smaller that fraction, the more accurate PG-NLL and Anscombe+MSE are.

    - **Level 2 (g=248.7):** fewer photons per pixel — higher gain means each ADU
      represents fewer photons. A larger fraction of pixels will fall below $\lambda=4$.
      In that regime the exact Poisson term matters: the distribution has a discrete
      spike at $k=0$ photons and a heavier right tail than Gaussian. Neither PG-NLL
      nor Anscombe+MSE is exact here; both are approximations.

    ### PG-NLL vs Anscombe+MSE — practical difference

    The weight analysis showed PG-NLL and Anscombe+MSE differ by only a few percent
    at our intensity levels. In practice they are interchangeable in terms of what
    they optimise. The choice reduces to implementation convenience:

    - **PG-NLL**: one formula, no preprocessing. Numerically stable as long as
      $g \cdot y + \sigma_r^2 > 0$ (always true for our data). Requires $g$ and
      $\sigma_r^2$ to be passed to the loss function at training time (needed for
      gain augmentation anyway).

    - **Anscombe+MSE**: requires a forward transform of inputs and targets before
      computing the loss. For gain augmentation, the transform must be re-applied
      with the augmented gain for each batch — adds one transform call per batch.
      Slightly more bookkeeping, but standard MSE optimisers work out of the box.

    ### Final verdict

    | Loss | Correct model | Works at low λ | Numerically stable | Gain-augmentation friendly | Verdict |
    |------|:---:|:---:|:---:|:---:|---|
    | MSE | ✗ | ✗ | ✓ | ✓ | **No** — statistically wrong, hurts tSNR |
    | PG-NLL (Gaussian approx) | ✓ | partial | ✓ | ✓ | **Preferred** |
    | Anscombe + MSE | ✓ | partial | ✓ | needs re-transform | **Equivalent to PG-NLL, more bookkeeping** |

    **Use PG-NLL** (Gaussian approximation to Poisson):

    $$\mathcal{L}(\hat{y}, y) = \frac{(\hat{y} - y)^2}{g \cdot y + \sigma_r^2} + \log\!\left(g \cdot y + \sigma_r^2\right)$$

    where $g$ and $\sigma_r^2$ are the noise parameters from `cidc.noise.NOISE_LEVELS`
    (validated in notebook 03), and for gain augmentation are replaced by the
    randomly sampled $g_\text{aug}$ for that batch.

    The log term ensures the loss does not collapse to zero by predicting high
    variance everywhere — it penalises unnecessary uncertainty.

    ### What this means for tSNR

    PG-NLL weights each pixel by the inverse of its noise variance. This is not
    penalising neuron pixels — it is penalising *noise at neuron pixels*. The model
    is told: "here is a bright, noisy pixel — learn the underlying smooth signal,
    not the fluctuation." A model trained this way produces cleaner temporal traces
    at neuron pixels. That is exactly what tSNR measures. Statistical correctness
    and tSNR optimisation are aligned.

    ### Training / validation split and why the loss parameters carry over

    **Training stacks:** A1, B1 (noise level 1, g=28.4), C2, D2 (noise level 2, g=248.7).
    These are the only stacks the model ever sees during training. The PG-NLL parameters
    g and $\sigma_r^2$ used in the loss are measured from these stacks (NB03).

    **Validation stacks:** F1 (level 1), F2 (level 2), F3 (level 3, OOD).
    **Ground truth:** F0 — a clean synthetic reference. F0 is the target for all scoring:
    stSNR(denoised output, F0). "Synthetic" does not mean fake — F0 is the authoritative
    clean signal that F1/F2/F3 were generated from by adding Poisson-Gaussian noise.

    **Why the loss parameters carry over to validation:** A1/B1 and F1 share the same
    noise level (level 1, g=28.4). C2/D2 and F2 share level 2 (g=248.7). So the PG-NLL
    parameters fitted from training stacks are exactly correct for the noise regime of
    the validation stacks. The loss is consistent across the train/val boundary at levels
    1 and 2. Level 3 (F3, g=990.5) is OOD — handled by gain augmentation, not by a
    fixed loss parameter (the loss uses the augmented g for each batch).

    ### Cross-notebook chain

    - **NB01** → τ₀.₅=46 frames → T=64 patch depth
    - **NB02** → stSNR ceiling 23.2 dB at W=61; N2V3D voxel masking justified
    - **NB03** → noise model confirmed Poisson-Gaussian; library constants correct; A1/B1≡F1, C2/D2≡F2
    - **NB04** → loss = PG-NLL with `g`, `σ_r²` from NOISE_LEVELS; MSE is wrong; F0 is ground truth

    Every architecture and training decision is now grounded in measurements.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's what it covers, in order:

     Math cell — derives the effective per-pixel weight for all three losses from first principles. Key formula: MSE weight
      = constant, PG-NLL weight = 1/(g·y + σ_r²), Anscombe weight = 1/(g·y + 3/8·g² + σ_r²). The table shows them side by
     side.

     Weight curves — plots normalized weight vs pixel intensity at both gain levels. Marks the half-weight crossover
     intensity (where PG-NLL starts meaningfully ignoring a pixel = σ_r²/g). Prints the exact numbers and the % difference
     between PG-NLL and Anscombe at a typical neuron brightness.

     Weight maps on real data — loads A1 and C2, computes the PG-NLL and Anscombe weight maps spatially, computes
     correlation between weight map and temporal std (neuron activity proxy). Negative correlation = loss down-weights
     active neurons. This is the key spatial check.

     tSNR connection — resolves the apparent tension: PG-NLL de-emphasises noise at neurons, not neurons themselves. A
     model trained with MSE chases noise fluctuations at bright pixels → temporally erratic output → bad tSNR. Statistical
     correctness and tSNR are aligned.

     Photon count histogram — shows what fraction of pixels in each stack fall below λ=4 (where the Gaussian approximation
     to Poisson breaks down). Level 2 (g=248.7) will have many pixels in that regime.

     Decision gate — verdict table, final loss formula written out explicitly, cross-notebook chain summarising all four
     decisions.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
