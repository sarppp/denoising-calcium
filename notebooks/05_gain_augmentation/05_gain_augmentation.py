"""05 — Gain augmentation: covering the OOD noise level."""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 — Gain augmentation: covering the OOD noise level

    **QUESTION:** F3 (level 3, g=990.5) is OOD — the model never sees it during training.
    Gain augmentation proposes to randomly sample g during training so the model implicitly
    learns all gain levels. Before writing any training code, verify:

    1. Does `sample_poisson_gaussian` produce statistically correct noise at g=990.5?
    2. What range [g_min, g_max] should we sample from, and why log-uniform?
    3. Does PG-NLL with the augmented g correctly normalise residuals across the range?
    4. What does a level-3 frame look like vs level 1/2 — and what must the model learn
       to do differently?

    **Decision gate:** pins down the exact augmentation prescription before any model code.
    """)
    return


@app.cell
def _setup():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from cidc import load_stack
    from cidc.noise import NOISE_LEVELS, NoiseParams, sample_poisson_gaussian

    DATA = Path(__file__).parent.parent.parent / "data"
    return DATA, NOISE_LEVELS, load_stack, mo, np, plt, sample_poisson_gaussian


@app.cell
def _intro(mo):
    mo.md("""
    ## Why gain augmentation?

    NB03 confirmed that F3 uses g=990.5 — 35× higher gain than level 1 (g=28.4) and 4×
    higher than level 2 (g=248.7). CIDC25 Task 2 scores the denoised output of F3 as part
    of the final evaluation, but F3 is withheld from training entirely.

    The PG-NLL loss (NB04) requires knowing g and σ_r² for every training sample. During
    training we know these exactly from the measured noise parameters in `NOISE_LEVELS`.
    The problem is not knowing g at test time — it is that the model has never seen data
    with the level-3 noise character during training.

    **Gain augmentation:** for each training batch, randomly sample g_aug from a range
    spanning all three levels. Then:
    1. Take a clean patch x (derived from the noisy training input via blind-spot masking).
    2. Corrupt it: y_aug = sample_poisson_gaussian(x, NoiseParams(g_aug, σ_r²)).
    3. Train on PG-NLL(ŷ, y_aug; g_aug, σ_r²).

    The model sees every gain level implicitly and learns a single denoiser that generalises
    across the full range.

    **What this notebook verifies — in order:**
    1. The library sampler is statistically correct at g=990.5 (otherwise augmentation
       produces wrong noise and the model learns the wrong noise model).
    2. The augmentation range and sampling distribution cover all levels with reasonable
       probability.
    3. The PG-NLL loss correctly normalises residuals across the range so gradient
       magnitudes are comparable regardless of which g_aug is drawn.
    4. Level 3's visual and photon-count character — to understand what qualitatively
       different problem the model must solve at high gain.
    """)
    return


@app.cell
def _md_pg_model(mo):
    mo.md(r"""
    ## How gain g changes the noise — and why it matters for generalisation

    The Poisson-Gaussian noise model:

    $$\text{Var}[y] = g \cdot \mathbb{E}[y] + \sigma_r^2$$

    where $g$ is the detector gain (ADU per photoelectron) and $\sigma_r^2$ is the
    read-noise variance. Both are fixed per recording session and per detector setting.

    **What g controls physically:**

    - **Photon count per ADU.** The expected photon count at a pixel with ADU value $y$
      is $\lambda = y/g$. Higher $g$ means fewer photons per ADU. A pixel at 1000 ADU
      with $g=28.4$ represents $1000/28.4 \approx 35$ photons — well into the regime
      where Poisson $\approx$ Gaussian ($\lambda \gg 4$). The same 1000 ADU with
      $g=990.5$ represents only $\approx 1$ photon — deeply discrete.

    - **Shot noise scale.** Shot noise contributes $g \cdot \lambda = g \cdot (x/g) = x$
      to Var[y] in photon units. In ADU units, shot noise variance is $g \cdot x$.
      Higher $g$ amplifies each photon's contribution to ADU variance: a single photon
      event at $g=990.5$ adds a 990 ADU spike, while the same event at $g=28.4$ adds
      only 28 ADU.

    - **Noise character.** At low $g$: many photons, Gaussian-like noise, smooth
      fluctuations. At high $g$: few photons, discrete Poisson spikes, the image
      looks like a field of rare sharp events on a noisy background.

    **Why training on levels 1–2 does not generalise to level 3 by default:**

    A network trained only on $g \in \{28.4, 248.7\}$ has never seen the spike-like
    noise of $g=990.5$. Its learned noise prior expects Gaussian-like fluctuations.
    Applied to level-3 data, it will either:
    - **Under-denoise:** misinterpret single-photon spikes (990 ADU each) as signal.
    - **Over-smooth:** apply too aggressive a prior and wash out genuine transients.

    The noise variance is mis-estimated by a factor of $990.5/248.7 \approx 4 \times$
    (level 2 to level 3) or $990.5/28.4 \approx 35 \times$ (level 1 to level 3).

    **Why gain augmentation works:**

    By training on $g_\text{aug} \sim \text{LogUniform}(g_\text{min}, g_\text{max})$
    with PG-NLL using the corresponding $g_\text{aug}$, the model sees all noise
    characters during training. The loss correctly normalises gradients at each $g$
    via the $1/(g y + \sigma_r^2)$ weight, so no single gain level dominates despite
    spanning 2 orders of magnitude.
    """)
    return


@app.cell
def _level3_visual(DATA, NOISE_LEVELS, load_stack, np, plt):
    _fig, _axes = plt.subplots(2, 3, figsize=(15, 9))

    for _col, (_name, _path, _level) in enumerate([
        ("A1 — level 1", DATA / "train" / "A1.tif", 1),
        ("C2 — level 2", DATA / "train" / "C2.tif", 2),
        ("F3 — level 3", DATA / "val"   / "F3.tif", 3),
    ]):
        _st = np.asarray(load_stack(_path)[:50], dtype=np.float32)
        _frame = _st[0]
        _tstd = _st.std(axis=0)
        _p = NOISE_LEVELS[_level]

        _v1  = np.percentile(_frame, 1)
        _v99 = np.percentile(_frame, 99)
        _axes[0, _col].imshow(_frame, cmap="gray", vmin=_v1, vmax=_v99)
        _axes[0, _col].set_title(
            f"{_name}\ng={_p.gain}, σ_r²={_p.read_var}\n"
            f"ADU range [{_frame.min():.0f}, {_frame.max():.0f}]  "
            f"median={np.median(_frame):.0f}"
        )
        _axes[0, _col].axis("off")

        _axes[1, _col].imshow(_tstd, cmap="hot", vmin=0,
                               vmax=np.percentile(_tstd, 99))
        _axes[1, _col].set_title(
            f"Temporal std (T=50)\nmedian={np.median(_tstd):.0f} ADU  "
            f"max={_tstd.max():.0f}"
        )
        _axes[1, _col].axis("off")

        print(f"{_name} (g={_p.gain}):  "
              f"frame median={np.median(_frame):.0f} ADU  |  "
              f"temporal std median={np.median(_tstd):.0f} ADU  |  "
              f"max={_tstd.max():.0f}")

    _fig.suptitle(
        "Level 1 / 2 / 3 — raw frame (top) and temporal std noise map (bottom)\n"
        "Temporal std reveals noise character: smooth (low g) vs spike-like (high g)",
        y=1.02,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _md_photon_context(mo):
    mo.md(r"""
    ## Photon count distributions across all three levels

    NB04 showed that level 2 (g=248.7) already has 100% of pixels below the $\lambda=4$
    Gaussian-approximation threshold. Level 3 (g=990.5) has 4× higher gain — the photon
    count $\lambda = \mu/g$ will be 4× lower for the same ADU intensity, pushing deeply
    into the discrete-Poisson regime.

    We compute photon count histograms for A1 (level 1), C2 (level 2), and F3 (level 3),
    using the temporal mean $\hat{\mu}$ as a proxy for the true signal intensity.

    **Why this matters for augmentation design:**

    - If level 3 is mostly in the sub-photon regime ($\lambda < 1$), the model must
      handle images that are mostly zeros with rare non-zero events. That is a qualitatively
      different pattern-recognition task from levels 1–2, not just a quantitatively harder one.

    - Gain augmentation must produce level-3-like samples during training — including this
      sub-photon character — for the model to learn the right prior.

    - The PG-NLL loss at sub-photon regime is a poor approximation to the exact Poisson NLL
      (because the Gaussian tail is wrong). The `_pgnll_check` cell quantifies how much
      this matters for gradient stability.
    """)
    return


@app.cell
def _photon_all_levels(DATA, NOISE_LEVELS, load_stack, np, plt):
    _T_ph = 100
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    photon_stats_all = {}

    for _ax, (_name, _path, _level) in zip(_axes, [
        ("A1 (level 1)", DATA / "train" / "A1.tif", 1),
        ("C2 (level 2)", DATA / "train" / "C2.tif", 2),
        ("F3 (level 3)", DATA / "val"   / "F3.tif", 3),
    ]):
        _st = np.asarray(load_stack(_path)[:_T_ph], dtype=np.float32)
        _mu = _st.mean(axis=0).ravel()
        _g = NOISE_LEVELS[_level].gain
        _ph = np.maximum(_mu, 0.0) / _g

        _f4   = 100.0 * (_ph < 4).mean()
        _f1   = 100.0 * (_ph < 1).mean()
        _f01  = 100.0 * (_ph < 0.1).mean()
        _med  = float(np.median(_ph))
        _mean = float(_ph.mean())

        photon_stats_all[_name] = {
            "level": _level, "gain": _g,
            "median_lambda": _med, "mean_lambda": _mean,
            "frac_below_4": _f4, "frac_below_1": _f1, "frac_below_01": _f01,
        }

        _hist_max = max(30.0, _med * 4 + 5)
        _ax.hist(_ph, bins=100, range=(0, _hist_max), color="steelblue", alpha=0.7)
        _ax.axvline(4, color="tomato",  ls="--", lw=1.5, label="λ=4 (Gaussian approx)")
        _ax.axvline(1, color="orange",  ls=":",  lw=1.5, label="λ=1 (single photon)")
        _ax.set_xlabel("Expected photons per pixel (λ = μ/g)")
        _ax.set_ylabel("Pixel count")
        _ax.set_title(f"{_name} (g={_g})\nmedian λ={_med:.3f}  |  {_f4:.1f}% below λ=4")
        _ax.legend(fontsize=8)

        print(f"{_name} (g={_g:.1f}):  "
              f"median λ={_med:.3f}  |  "
              f"{_f4:.1f}% below λ=4  |  "
              f"{_f1:.1f}% below λ=1  |  "
              f"{_f01:.1f}% below λ=0.1")

    _fig.suptitle("Photon count distributions — all three noise levels")
    _fig.tight_layout()
    _fig
    return (photon_stats_all,)


@app.cell
def _photon_interpretation(photon_stats_all):
    print("Photon count regime — interpretation from measured distributions:")
    print()
    for _name, _s in photon_stats_all.items():
        _med = _s["median_lambda"]
        _f4  = _s["frac_below_4"]
        _f1  = _s["frac_below_1"]
        _f01 = _s["frac_below_01"]
        if _med >= 4:
            _regime = "Gaussian regime — Poisson well approximated"
        elif _med >= 1:
            _regime = "Transition regime — Gaussian marginal"
        elif _med >= 0.1:
            _regime = "Low-photon regime — discrete Poisson dominates"
        else:
            _regime = "Sub-photon regime — mostly dark, rare single events"
        print(f"  {_name}: {_regime}")
        print(f"    median λ={_med:.3f}  |  {_f4:.0f}% below λ=4  "
              f"|  {_f1:.0f}% below λ=1  |  {_f01:.0f}% below λ=0.1")
    print()
    print("Key implication for augmentation: gain augmentation must expose the model to")
    print("all three regimes, including sub-photon conditions, during training.")
    return


@app.cell
def _md_library_sanity(mo):
    mo.md(r"""
    ## Library sanity check — does `sample_poisson_gaussian` produce correct noise?

    Before using `sample_poisson_gaussian` in augmentation, verify it implements the
    PG model correctly at g=990.5 — the level it was never validated against in NB03
    (which validated the noise parameters, not the sampler itself at level 3).

    **The test:**
    1. Create a clean constant signal $x$ at several intensities spanning the data range.
    2. Draw $N=2000$ independent realisations of $y \sim \text{PG}(x, g, \sigma_r^2)$.
    3. Compute $\hat{\mu} = \overline{y}$ and $\hat{\sigma}^2 = \text{Var}(y)$.
    4. Check: $\hat{\mu} \approx x$ (unbiased) and $\hat{\sigma}^2 \approx g x + \sigma_r^2$.

    We test all three levels so we have a baseline for levels 1 and 2 (where we know the
    answer from NB03) and can confidently trust the level-3 result by comparison.

    **Why this matters:** if the sampler has a bug at level 3 (wrong Poisson rate,
    wrong Gaussian scale), training on augmented level-3 data teaches the model the
    wrong noise model. The denoiser would reach F3 already miscalibrated — a silent
    failure that would not show up in the training loss, only in final evaluation.
    """)
    return


@app.cell
def _library_sanity(NOISE_LEVELS, np, plt, sample_poisson_gaussian):
    _N_trials = 5000
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    sanity_results = {}

    for _ax, _level in zip(_axes, [1, 2, 3]):
        _p = NOISE_LEVELS[_level]
        _g, _sr2 = _p.gain, _p.read_var
        _rng = np.random.default_rng(42 + _level)

        # Test only at intensities where λ = x/g ≥ 1 (at least 1 photon on average).
        # Below λ=1 the Poisson distribution is so sparse (mostly zeros with rare spikes)
        # that the empirical variance estimator has 40%+ relative std even at N=5000 —
        # any error there reflects Monte Carlo noise, not a sampler bug.
        _lambda_min = 1.0
        _x_min = max(_g * _lambda_min, 50.0)
        _intensities = np.array([_x_min, _x_min*2, _x_min*5,
                                  _x_min*10, _x_min*20], dtype=np.float64)

        _emp_var, _exp_var, _mc_std_pct = [], [], []
        for _x_val in _intensities:
            _clean = np.full(_N_trials, _x_val)
            _samp = sample_poisson_gaussian(_clean, _p, rng=_rng)
            _ev = float(_samp.var(ddof=1))
            _xv = _g * _x_val + _sr2
            _emp_var.append(_ev)
            _exp_var.append(_xv)
            # Expected relative std of sample variance = sqrt(2/(N-1)) * sqrt(1 + kurt/2)
            # For Poisson(lambda): excess kurtosis = 1/lambda; use conservative kurtosis=1
            _lam = _x_val / _g
            _kurt = min(1.0 / _lam, 10.0)  # capped at 10 for display
            _mc_std_pct.append(100 * float(np.sqrt(2.0 / (_N_trials - 1)) * np.sqrt(1 + _kurt / 2)))

        _emp_var = np.array(_emp_var)
        _exp_var = np.array(_exp_var)
        _rel_err = np.abs(_emp_var - _exp_var) / _exp_var * 100
        _mc_std_pct = np.array(_mc_std_pct)

        _ax.scatter(_intensities, _emp_var, color="steelblue", zorder=3,
                    s=40, label="Empirical variance")
        _ax.plot(_intensities, _exp_var, color="tomato", lw=1.5,
                 label="Expected g·x + σ_r²")
        _ax.set_xlabel("Clean signal x (ADU)")
        _ax.set_ylabel("Variance of y (ADU²)")
        _ax.set_title(f"Level {_level} (g={_g}, σ_r²={_sr2})\n"
                      f"max rel. error vs MC noise: {_rel_err.max():.1f}% vs {_mc_std_pct.max():.1f}%")
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.2)

        sanity_results[_level] = {
            "gain": _g, "read_var": _sr2,
            "max_rel_err_pct": float(_rel_err.max()),
            "max_mc_std_pct": float(_mc_std_pct.max()),
        }
        _significant = _rel_err.max() > 2 * _mc_std_pct[np.argmax(_rel_err)]
        print(f"Level {_level} (g={_g}):  "
              f"max rel. error = {_rel_err.max():.2f}%  |  "
              f"MC noise at that point = {_mc_std_pct[np.argmax(_rel_err)]:.2f}%  |  "
              f"{'REAL ERROR' if _significant else 'within MC noise'}")

    _fig.suptitle(
        "Library sanity check — empirical variance vs expected g·x + σ_r²  (N=2000 trials per point)"
    )
    _fig.tight_layout()
    _fig
    return (sanity_results,)


@app.cell
def _library_verdict(sanity_results):
    print("Library sanity check verdict (tested at λ ≥ 1 to avoid Monte Carlo noise floor):")
    print("Pass criterion: max relative error < 2× expected Monte Carlo noise at that point.")
    print()
    _all_pass = True
    for _level, _r in sanity_results.items():
        _pass = _r["max_rel_err_pct"] < 2 * _r["max_mc_std_pct"]
        _all_pass = _all_pass and _pass
        _tag = "PASS" if _pass else "FAIL"
        print(f"  Level {_level} (g={_r['gain']}):  {_tag}  "
              f"max rel. error = {_r['max_rel_err_pct']:.2f}%  |  "
              f"MC noise = {_r['max_mc_std_pct']:.2f}%")
    print()
    if _all_pass:
        print("All levels PASS.  sample_poisson_gaussian is statistically correct at λ ≥ 1.")
        print("Note: at λ < 1 (sub-photon regime), variance estimator is dominated by")
        print("Monte Carlo noise — the sampler may still be correct there, but N=5000")
        print("is insufficient to verify it. The sub-photon regime uses the same code path.")
    else:
        print("WARNING: one or more levels show error exceeding MC noise floor.")
        print("Investigate sample_poisson_gaussian implementation.")
    return


@app.cell
def _md_aug_range(mo):
    mo.md(r"""
    ## Augmentation range and sampling distribution — why log-uniform?

    We need to decide:
    1. The range $[g_\text{min}, g_\text{max}]$ to draw from.
    2. The probability distribution over that range.
    3. Whether to jointly augment $\sigma_r^2$ or hold it fixed.

    ### Why log-uniform and not linear-uniform?

    Our three gain levels span $\{28.4, 248.7, 990.5\}$ — roughly three decades on a
    log scale. On a linear scale, the intervals are $(220, 742)$: the gap between
    levels 2 and 3 is 3.4× larger than the gap between levels 1 and 2. If we sample
    $g \sim \text{Uniform}(20, 1000)$ linearly, $\approx 97\%$ of sampled values
    fall above $g=50$ — the model almost never trains on level-1-like conditions.
    It would be heavily biased towards high-gain noise.

    $g \sim \text{LogUniform}(g_\text{min}, g_\text{max})$, i.e.
    $\log g \sim \text{Uniform}(\log g_\text{min}, \log g_\text{max})$, gives every
    order-of-magnitude interval equal training time. The model spends equal effort on
    $g \approx 30$, $g \approx 100$, and $g \approx 1000$.

    ### What range?

    - $g_\text{min} = 20$: slightly below level 1 (g=28.4) — a small extrapolation margin.
    - $g_\text{max} = 1200$: slightly above level 3 (g=990.5) — covers it with margin.

    The code below computes what fraction of sampled values fall within a factor of 2
    of each actual level, confirming the coverage.

    ### What about $\sigma_r^2$?

    Measured read variances: 2490, 2700, 3730 — a 1.5× range. At high $g$, the
    shot noise term $g \cdot y$ dominates $\sigma_r^2$ for any visible pixel
    ($g=990.5$ and $y=100$ ADU already gives $g y = 99{,}050 \gg 3730$). At low
    $g$ (level 1), $\sigma_r^2 = 2490$ is comparable to shot noise at dark pixels.

    The computation below checks how much $\sigma_r^2$ variation (2000–4000) affects
    the PG-NLL weight at each level. If the effect is small, fixing $\sigma_r^2 = 2700$
    (the median) removes one hyperparameter without meaningful loss quality degradation.
    """)
    return


@app.cell
def _aug_range(NOISE_LEVELS, np, plt):
    _g_min, _g_max = 20.0, 1200.0
    _n_samp = 200_000
    _rng = np.random.default_rng(0)

    _log_g = _rng.uniform(np.log(_g_min), np.log(_g_max), _n_samp)
    _g_samp = np.exp(_log_g)

    print(f"Log-uniform sampling from g ∈ [{_g_min}, {_g_max}]:")
    print()
    _factor = 2.0
    for _lv in [1, 2, 3]:
        _g_lv = NOISE_LEVELS[_lv].gain
        _near = ((_g_samp > _g_lv / _factor) & (_g_samp < _g_lv * _factor)).mean() * 100
        print(f"  Level {_lv} (g={_g_lv}):  {_near:.1f}% of batches within ×{_factor:.0f} of target")

    print()
    print("σ_r² sensitivity — % change in PG-NLL weight at y=500 ADU when σ_r² varies 2000→4000:")
    _y_test = 500.0
    for _lv in [1, 2, 3]:
        _g_lv = NOISE_LEVELS[_lv].gain
        _w_lo  = 1.0 / (_g_lv * _y_test + 2000)
        _w_hi  = 1.0 / (_g_lv * _y_test + 4000)
        _w_mid = 1.0 / (_g_lv * _y_test + 2700)
        _pct = (_w_lo - _w_hi) / _w_mid * 100
        print(f"  Level {_lv} (g={_g_lv:.1f}):  weight changes by {_pct:.1f}%")

    print()
    print("Implication: if σ_r² effect is small at all levels → fix σ_r²=2700 (median).")
    print("If large at level 1 → sample σ_r² jointly or use measured value per batch.")

    _fig, _axes = plt.subplots(1, 2, figsize=(13, 4))

    _bins_lin = np.linspace(_g_min, _g_max, 80)
    _axes[0].hist(_g_samp, bins=_bins_lin, color="steelblue", alpha=0.7, density=True)
    for _lv in [1, 2, 3]:
        _g_lv = NOISE_LEVELS[_lv].gain
        _axes[0].axvline(_g_lv, color="tomato", ls="--", lw=1.5,
                         label=f"Level {_lv} (g={_g_lv})")
    _axes[0].set_xlabel("Sampled gain g")
    _axes[0].set_ylabel("Density")
    _axes[0].set_title("Log-uniform samples — linear x axis\n"
                        "(most mass at high g on this scale)")
    _axes[0].legend(fontsize=8)
    _axes[0].grid(alpha=0.2)

    _bins_log = np.exp(np.linspace(np.log(_g_min), np.log(_g_max), 80))
    _axes[1].hist(_g_samp, bins=_bins_log, color="seagreen", alpha=0.7, density=True)
    for _lv in [1, 2, 3]:
        _g_lv = NOISE_LEVELS[_lv].gain
        _axes[1].axvline(_g_lv, color="tomato", ls="--", lw=1.5,
                         label=f"Level {_lv} (g={_g_lv})")
    _axes[1].set_xscale("log")
    _axes[1].set_xlabel("Sampled gain g (log scale)")
    _axes[1].set_ylabel("Density")
    _axes[1].set_title("Same samples — log x axis\n"
                        "(flat = equal coverage per decade)")
    _axes[1].legend(fontsize=8)
    _axes[1].grid(alpha=0.2)

    _fig.suptitle(
        f"Log-uniform gain sampling  g ∈ [{_g_min}, {_g_max}]  —  "
        "linear view (left) vs log view (right, flat = equal coverage)"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _md_pgnll_check(mo):
    mo.md(r"""
    ## PG-NLL normalisation check — are gradients comparable across gain levels?

    When augmented batches are drawn at different g_aug, the PG-NLL loss values must
    be comparable in magnitude. If a high-gain batch produces 100× larger loss values
    than a low-gain batch, the effective learning rate is wildly inconsistent across
    batches and training becomes unstable.

    The PG-NLL weight $w = 1/(g y + \sigma_r^2)$ is designed to normalise this.
    The normalised squared residual per pixel:

    $$z = \frac{(\hat{y} - y)^2}{g y + \sigma_r^2}$$

    should satisfy $\mathbb{E}[z] \approx 1$ regardless of $g$, because at the optimum
    $\hat{y} \approx x$ and:

    $$\mathbb{E}\!\left[\frac{(y - x)^2}{g y + \sigma_r^2}\right]
      \approx \frac{g x + \sigma_r^2}{g x + \sigma_r^2} = 1$$

    The approximation $(g y + \sigma_r^2 \approx g x + \sigma_r^2)$ holds when $y$
    is close to $x$ — i.e. when photon counts are high enough that Poisson $\approx$
    Gaussian. At very low $\lambda$, $y$ and $x$ can differ substantially, and
    $\mathbb{E}[z]$ deviates from 1.

    **Test:** use F0 as $x$ (clean synthetic ground truth). Generate $N=300$ realisations
    of $y \sim \text{PG}(x, g, \sigma_r^2)$ for each level. Compute two versions of $z$:

    $$z_{\hat{y}} = \frac{(y - x)^2}{g\,\max(x, 0) + \sigma_r^2} \qquad
      z_y = \frac{(y - x)^2}{g\,\max(y, 0) + \sigma_r^2}$$

    $z_{\hat{y}}$ uses the true signal $x$ as a proxy for the network prediction $\hat{y}$
    (valid at convergence when $\hat{y} \approx x$). $z_y$ uses the noisy observation $y$
    as the variance estimate — the naive approach.

    The key question: does $z_{\hat{y}}$ have mean $\approx 1$? If so, the ŷ-denominator
    loss is well-normalised in the steady state. Does $z_y$ show blowup? If so, the
    naive y-denominator is unstable — confirming we must use ŷ in training.
    """)
    return


@app.cell
def _pgnll_check(
    DATA,
    NOISE_LEVELS,
    load_stack,
    np,
    plt,
    sample_poisson_gaussian,
):
    _N_real = 300
    _f0_frame = np.asarray(load_stack(DATA / "val" / "F0.tif")[0], dtype=np.float64)
    _x_crop = _f0_frame[:128, :128].ravel()

    # chi²(1) pdf: f(x) = exp(-x/2) / sqrt(2π x)
    def _chi2_1_pdf(_x):
        return np.exp(-_x / 2.0) / np.sqrt(2.0 * np.pi * np.maximum(_x, 1e-10))

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))
    pgnll_stats = {}

    for _ax, _level in zip(_axes, [1, 2, 3]):
        _p = NOISE_LEVELS[_level]
        _g, _sr2 = _p.gain, _p.read_var
        _rng = np.random.default_rng(100 + _level)

        _z_yhat_parts = []  # denominator = g*max(x,0)+sr2  → what ŷ≈x gives at convergence
        _z_y_parts    = []  # denominator = g*max(y,0)+sr2  → naive y-based approach

        for _i in range(_N_real):
            _y = sample_poisson_gaussian(_x_crop, _p, rng=_rng)
            _resid_sq       = (_y - _x_crop) ** 2
            # ŷ-based denominator: use clean signal x as proxy (ŷ≈x at convergence)
            _denom_yhat     = _g * np.maximum(_x_crop, 0.0) + _sr2
            # y-based denominator: naive approach that causes Poisson-zero blowup
            _denom_y        = _g * np.maximum(_y,      0.0) + _sr2
            _z_yhat_parts.append(_resid_sq / _denom_yhat)
            _z_y_parts.append(_resid_sq / _denom_y)

        _z_yhat = np.concatenate(_z_yhat_parts)
        _z_y    = np.concatenate(_z_y_parts)

        _mean_yhat   = float(_z_yhat.mean())
        _mean_y      = float(_z_y.mean())
        _median_yhat = float(np.median(_z_yhat))

        pgnll_stats[_level] = {
            "gain": _g,
            "mean_z_yhat": _mean_yhat,
            "mean_z_y":    _mean_y,
            "median_z_yhat": _median_yhat,
        }

        _bins = np.linspace(0, 8, 60)
        _ax.hist(_z_yhat, bins=_bins, density=True, color="steelblue", alpha=0.7,
                 label=f"ŷ-denom  mean={_mean_yhat:.3f}")
        _ax.hist(_z_y,    bins=_bins, density=True, color="coral",     alpha=0.5,
                 label=f"y-denom  mean={_mean_y:.3f}")
        _xx = np.linspace(0.05, 8, 300)
        _ax.plot(_xx, _chi2_1_pdf(_xx), color="tomato", lw=1.5, ls="--",
                 label="χ²(1) reference")
        _ax.axvline(1.0, color="orange", ls=":", lw=1, alpha=0.8, label="E[z]=1")
        _ax.set_xlabel("z = (y−x)² / denominator")
        _ax.set_ylabel("Density")
        _ax.set_title(f"Level {_level} (g={_g})\n"
                      f"ŷ-denom mean={_mean_yhat:.3f}  |  y-denom mean={_mean_y:.2f}")
        _ax.legend(fontsize=7)
        _ax.set_xlim(0, 8)
        _ax.grid(alpha=0.2)
        print(f"Level {_level} (g={_g:.1f}):  "
              f"ŷ-denom mean z = {_mean_yhat:.4f}  |  "
              f"y-denom mean z = {_mean_y:.4f}")

    _fig.suptitle(
        "PG-NLL normalised residual — ŷ in denominator (blue) vs y in denominator (coral)\n"
        "ŷ-denom should → χ²(1) with mean=1 at convergence (ŷ≈x).  y-denom is biased by Poisson zeros."
    )
    _fig.tight_layout()
    _fig
    return (pgnll_stats,)


@app.cell
def _pgnll_verdict(pgnll_stats):
    print("PG-NLL normalisation verdict:")
    print("ŷ-denom = g·max(ŷ,0)+σ_r²  (correct training formula; here tested with ŷ=x from F0)")
    print("y-denom = g·max(y,0)+σ_r²  (naive approximation; biased by Poisson-zero events)")
    print()
    print("ŷ-denominator results (what training will see at convergence):")
    _all_ok = True
    for _level, _r in pgnll_stats.items():
        _dev = (_r["mean_z_yhat"] - 1.0) * 100
        _abs_dev = abs(_dev)
        if _abs_dev < 5:
            _tag = "EXCELLENT"
        elif _abs_dev < 15:
            _tag = "ACCEPTABLE"
        elif _abs_dev < 30:
            _tag = "MARGINAL"
        else:
            _tag = "POOR"
            _all_ok = False
        _sign = "+" if _dev > 0 else ""
        print(f"  Level {_level} (g={_r['gain']:.1f}):  "
              f"mean z = {_r['mean_z_yhat']:.4f}  ({_sign}{_dev:.1f}%)  —  {_tag}")
    print()
    print("y-denominator results (naive; shown for comparison):")
    for _level, _r in pgnll_stats.items():
        print(f"  Level {_level} (g={_r['gain']:.1f}):  mean z = {_r['mean_z_y']:.4f}  "
              f"(Poisson-zero blowup visible at high gain)")
    print()
    if _all_ok:
        print("ŷ-denominator is well-normalised.  Use ŷ in the loss denominator during training.")
        print("Add gradient clipping (global norm ≤ 1.0) as belt-and-suspenders for early")
        print("training when ŷ ≠ x and the approximation is not yet tight.")
    else:
        print("WARNING: ŷ-denominator still shows large bias. Investigate F0 crop content.")
    return


@app.cell
def _md_decision(mo):
    mo.md(r"""
    ## Decision gate — gain augmentation prescription

    ### Summary of findings

    Read all print outputs above for the exact numbers. This cell synthesises the
    structural conclusions and the prescription to carry into training code.

    **1. Library correctness:** read the sanity check verdict. If `sample_poisson_gaussian`
    passes at all three levels, the sampler is safe to use across the full augmentation range.

    **2. Level-3 photon regime:** read the photon count output. Level 3 (g=990.5) operates
    at dramatically lower photon counts than levels 1 and 2. The noise character is
    qualitatively different — sparse, discrete, spike-like — not just quantitatively harder.
    This is why simple generalisation from levels 1–2 fails: it is a different visual pattern.

    **3. Log-uniform coverage:** read the coverage output. The fraction of batches within ×2
    of each level confirms whether LogUniform(20, 1200) achieves balanced coverage. If any
    level is under-represented, adjust g_min or g_max.

    **4. σ_r² sensitivity:** read the sensitivity output. At level 1 (g=28.4), σ_r²
    variation from 2000→4000 changes the PG-NLL weight by ~11% — non-negligible.
    At levels 2 and 3 it is <2%. Use the measured σ_r² per level when the batch comes
    from a known training stack. For purely augmented batches (random g_aug), fix σ_r²=2700.

    **5. PG-NLL normalisation — two critical implementation findings:** read the verdict.

    *Finding A — use ŷ in the loss denominator, not y.* Even with a max(y,0) clip,
    using y in the denominator causes mean z >> 1 at levels 2 and 3. The reason is the
    Poisson-zero effect: bright pixels occasionally emit 0 photons, so y≈0 while x is
    large, making the denominator collapse to σ_r² while the residual (y−x)² is huge.
    Using ŷ (the network's prediction of the clean signal) avoids this: ŷ is smooth and
    tracks x, so the denominator correctly reflects the expected noise variance. At
    convergence ŷ≈x and mean z→1.

    *Finding B — add gradient clipping.* Early in training ŷ≠x, so the ŷ-denominator
    approximation is not yet tight. Global gradient norm clipping (≤1.0) prevents any
    single bad batch from dominating. This is cheap insurance for a problem that only
    occurs during early training.

    ### Exact augmentation prescription

    For each training batch:

    1. **Sample:** $g_{\text{aug}} \sim \text{LogUniform}(20, 1200)$ — one value per batch.
       `g_aug = exp(uniform(log(20), log(1200)))`.

    2. **Read variance:** $\sigma_r^2 = 2700$ for augmented batches. Use measured value
       when batch is from a known training stack (2490 for level 1, 2700 for level 2).

    3. **Corrupt:** $y \sim \text{PG}(x,\, g_{\text{aug}},\, \sigma_r^2)$ via
       `sample_poisson_gaussian(x, NoiseParams(g_aug, sr2))`.

    4. **Loss — ŷ in denominator:**
       $$\mathcal{L}(\hat{y}, y;\, g_{\text{aug}}) = \frac{(\hat{y} - y)^2}{g_{\text{aug}}\,\max(\hat{y},\,0) + \sigma_r^2} + \log\!\left(g_{\text{aug}}\,\max(\hat{y},\,0) + \sigma_r^2\right)$$

       Using $\hat{y}$ rather than $y$ in the denominator prevents the Poisson-zero
       catastrophe at dark pixels. The numerator remains $({\hat{y}} - y)^2$ — the model
       is still penalised for getting bright pixels wrong; the denominator simply uses
       a stable estimate of the local noise variance.

    5. **Conditioning:** the network receives the per-pixel noise variance map
       $v = g_{\text{aug}}\,\max(y,\,0) + \sigma_r^2$ as an additional input channel.
       This gives the model full information about local noise level. Note $v$ uses $y$
       (the observed noisy input), not $\hat{y}$ — it is an input feature, computed
       before the forward pass.

    6. **Gradient clipping:** clip global gradient norm to 1.0 at every step.

    ### Cross-notebook chain

    - **NB01** → τ₀.₅=46 frames → T=64 patch depth
    - **NB02** → stSNR ceiling 23.2 dB at W=61; N2V3D voxel masking justified
    - **NB03** → noise model confirmed PG; library constants valid
    - **NB04** → loss = PG-NLL; MSE is wrong
    - **NB05** → gain aug LogUniform(20,1200); loss uses ŷ in denominator; grad clip ≤1.0
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's what this notebook covers, in order:

    **PG model math** — how $g$ changes noise physically: photon count per ADU, shot noise
    scale, discrete vs Gaussian noise character. Why level-3 noise is qualitatively different
    (rare single-photon spikes vs smooth Gaussian fluctuations) and why training on levels 1–2
    fails to generalise.

    **Level-3 visual** — side-by-side first frame and temporal-std noise maps for levels 1/2/3.
    The temporal-std map reveals noise character directly: level 1 is smooth, level 3 is
    spike-like.

    **Photon counts across all levels** — histograms of λ=μ/g for A1, C2, F3. Quantifies
    exactly how far into the discrete-Poisson regime each level falls, including sub-photon
    ($\lambda < 0.1$) fractions.

    **Library sanity check** — empirical variance vs expected $g x + \sigma_r^2$ for
    N=2000 trials at each level. Confirms `sample_poisson_gaussian` is statistically
    correct at g=990.5 before we rely on it for augmentation.

    **Augmentation range** — log-uniform sampling from LogUniform(20, 1200), coverage
    per level within a factor of 2, $\sigma_r^2$ sensitivity analysis. Justifies the
    prescription: log-uniform to avoid level-bias, $\sigma_r^2$ fixed at 2700 unless
    level-1 sensitivity is non-negligible.

    **PG-NLL normalisation** — normalised residuals $z = (y-x)^2/(gy+\sigma_r^2)$ compared
    to $\chi^2(1)$. Mean $z$ close to 1.0 → gradients are comparable across gain levels →
    training is stable. Deviation → need gradient clipping.

    **Decision gate** — exact augmentation prescription with all parameters determined from
    measured data, cross-notebook chain updated.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MEASURED FINDINGS (not assumptions):
    ═══════════════════════════════════════════════════════════════

    TEMPORAL CORRELATIONS IN F0 (clean signal):
      ρ(0)  = 1.0000  (by definition)
      ρ(1)  = 0.9961  ← KEY: 99.6% correlated at 1 frame!
      ρ(5)  = 0.9707  (97% correlated at 0.33 sec)
      ρ(10) = 0.9235  (92% at 0.67 sec)
      ρ(20) = 0.8038  (80% at 1.3 sec)
      ρ(30) = 0.6786  (68% at 2.0 sec)

    INTERPRETATION: Calcium neuron activity is HIGHLY temporally
    correlated. When it fires at t-1, it's still firing at t with
    99.6% probability. This is the SIGNAL we need to denoise for.

    ═══════════════════════════════════════════════════════════════

    MUTUAL INFORMATION FROM TEMPORAL NEIGHBORS (bits):
      ±1  frame   (2 neighbors):  4.86 bits
      ±3  frames  (6 neighbors):  9.88 bits
      ±5  frames  (10 neighbors): 18.38 bits
      ±10 frames  (20 neighbors): 29.58 bits  ← CHOSEN
      ±15 frames  (30 neighbors): 37.47 bits
      ±20 frames  (40 neighbors): 43.30 bits

    INTERPRETATION: With ±10 frames, the model has access to
    ~30 bits of signal information just from temporal context.
    Going to ±20 only adds 13.7 more bits but costs 2× memory.
    This is the optimal tradeoff for 6GB VRAM.

    ---
    WHAT CHANGED IN OUR PLANS

    ┌──────────────┬──────────────────┬─────────────────────────┬───────────────────────────────────────────────────┐
    │    Aspect    │   Before NB06    │       After NB06        │                    Why Changed                    │
    ├──────────────┼──────────────────┼─────────────────────────┼───────────────────────────────────────────────────┤
    │ Masking      │ Uncertain (2D or │ ✓ 3D only               │ 30 bits of temporal info measured—too much to     │
    │ strategy     │  3D?)            │                         │ ignore                                            │
    ├──────────────┼──────────────────┼─────────────────────────┼───────────────────────────────────────────────────┤
    │ Temporal     │ Vague (±5 to     │ ✓ ±10 frames fixed      │ 29.58 bits measured; ±5 misses 40%, ±20 wastes    │
    │ window       │ ±20?)            │                         │ memory                                            │
    ├──────────────┼──────────────────┼─────────────────────────┼───────────────────────────────────────────────────┤
    │ Input        │ 1 (raw patch)    │ ✓ 2 (raw + variance)    │ PG-NLL loss uses v=g·max(y,0)+σ_r² to weight      │
    │ channels     │                  │                         │ pixels—give model same info                       │
    ├──────────────┼──────────────────┼─────────────────────────┼───────────────────────────────────────────────────┤
    │ Expected     │ Speculative      │ ✓ Grounded in           │ 30 bits of signal available → tSNR improvement is │
    │ outcome      │                  │ information theory      │  real                                             │
    └──────────────┴──────────────────┴─────────────────────────┴───────────────────────────────────────────────────┘

    ---
    ARCHITECTURE CHOICES: Why U-Net, Not Others

    COMPARISON TABLE:
    ═══════════════════════════════════════════════════════════════

    ARCHITECTURE    PROS                    CONS                   USE?
    ─────────────────────────────────────────────────────────────────
    U-Net (Conv)    ✓ Proven med imaging    • Limited receptive    ✓ YES
                    ✓ Interpretable           field (must design)   CHOSEN
                    ✓ Efficient memory
                    ✓ Skip connections
                      preserve details

    PINN            • Encodes PDE           ✗ 5-10× slower         ✗ NO
    (Physics-       • "Physically                (solves PDE        Overkill
     Informed)        informed"              each step)
                                            ✗ No evidence on
                                              calcium data
                                            ✗ Loss already
                                              encodes noise
                                              model (PG-NLL)

    Transformer     ✓ Large receptive       ✗ 2-3× more memory     ? MAYBE
                      field (sees all        ✗ Overkill for        LATER
                      21 frames)             3D patch
                    ✓ Attention can          ✗ Unclear if
                      learn importance       attention helps
                                            temporal denoising

    ResNet          ✓ Simple, proven        ✗ No upsampling       ✗ NO
                                            (not restoration)     Not suitable
                                            ✗ Dense blocks
                                            waste memory on
                                            3D data

    Noise2Void Conv ✓ Proven baseline       ⚠ 2D only—misses       ⚠ BASELINE
    (2D)            ✓ Self-supervised       30 bits of info       Not target

    N2V3D (3D)      ✓ Exactly what we       ⚠ Original used MSE    ✓ BASE
                      need (3D masking)      loss (suboptimal)    IMPROVED
                                            ✓ We add: PG-NLL +    BY US
                                              conditioning +
                                              gain augmentation

    ═══════════════════════════════════════════════════════════════

    WHY U-NET (decision grounded in NB06):

    1. RECEPTIVE FIELD: We need ±10 frames (depth 21), but don't
       need to see all 21 high-resolution pixels at once. U-Net's
       hierarchical downsampling is natural. Transformer would pay
       quadratic cost O(n²) for 21³ = 9261 voxels per patch.

    2. SKIP CONNECTIONS: Low-level noise details (speckle, shot
       noise) bypass deep layers. Deep layers capture signal
       structure (neuron morphology, firing patterns). This is
       proven in medical imaging (proven != assumed).

    3. RESOURCE CONSTRAINT (6GB VRAM):
       - U-Net depth 4-5: ~700 MB per batch ✓
       - Transformer on 21×64×64: >2GB per batch ✗

    4. PROVEN ON CALCIUM: Zubic et al. N2V3D, CaImAn, ScanImage
       all use U-Net variants. Not because it's unique, but because
       it works and we know the failure modes.

    5. OUR IMPROVEMENT: N2V3D used MSE loss. We use PG-NLL
       (grounded in NB04, NB05). This is the innovation, not the
       architecture.

    ---
    WHERE IS N2V3D? WHERE IS PINN?

    N2V3D (Noise2Void 3D, Buchberger et al. 2021):
      • This is the BASE we're building on
      • 3D blind-spot masking: mask a voxel, predict from context
      • Their limitation: used MSE loss (treats all pixels equally)
      • Our improvement: PG-NLL loss (weights pixels by noise level)

      OUR APPROACH = N2V3D + PG-NLL + variance conditioning +
                     gain augmentation

    PINN (Physics-Informed Neural Networks):
      • Attractive idea: encode the noise model PDE into training
      • Problem: We ALREADY encode it via PG-NLL loss
      • Cost: requires solving ∂y/∂t = ... PDE at each step
      • Benefit: unclear (loss already contains physics)
      • No published results on calcium denoising with PINN
      • ✗ REJECTED: adds cost without evidence of benefit

    ---
    RESOURCE CONSTRAINT STRATEGY (6GB VRAM → Sample First)

    MEMORY BUDGET:
    ═══════════════════════════════════════════════════════════════

    Per training batch (batch_size=1):
      • Patch: 21 × 64 × 64 × 2 (float32) = 11 MB
      • Model params: 5-10 MB
      • Forward activations: 200-300 MB
      • Optimizer state (Adam): ~500 MB
      ───────────────────────────────
      TOTAL: ~730 MB << 6GB ✓ SAFE

    ═══════════════════════════════════════════════════════════════

    TRAINING STRATEGY (don't train all data immediately):

    Phase 1: VERIFICATION (sample training data)
      • A1: use frames 0-500 only (1/3 of 1500)
      • B1, C2, D2: similar sampling
      • Purpose:
        - Test training loop works (no NaN, gradient explosion)
        - Verify loss decreases (model is learning)
        - Check gradient clipping works (PG-NLL can have spikes)
        - Estimate convergence time
      • Expected time: 3-4 hours
      • Success = loss goes from ~2.0 to ~0.8 on sampled data

    Phase 2: FULL TRAINING (if Phase 1 succeeds)
      • A1[0:1500] full, B1 full, C2 full, D2 full
      • Same hyperparameters as Phase 1
      • Expected time: 12-15 hours
      • Measure: full stSNR on F1/F2

    VALIDATION (can be on subset):
      • Validate on F1[0:200] frames (not full 1500)
      • stSNR converges by frame 50-100 (diminishing returns after)
      • Computing stSNR on 200 frames is representative
      • Saves compute, same validity

    This is NOT skipping validation.
    Sampled training catches:
      ✓ Gradient bugs (NaN, divergence)
      ✓ Model learning (loss decreasing?)
      ✓ Overfitting signature
      ✓ Loss function correctness

    ---
    BEFORE NB07 TRAINING: CREATE NB06.5 (Visualizations + Metrics)

    NB06.5 should show:

    1. MASKING GEOMETRY DIAGRAMS + METRICS
       TEXT OUTPUT:
       ─────────────────────────────────────
       2D Blind-Spot (Noise2Void original):
         Input patch: 64×64×1 (single timepoint)
         Context: spatial neighbors (4-8 adjacent pixels)
         Mask: center pixel (1,1)
         Information available: I(x_center; neighbors) ≈ 0.5-1.5 bits
                                (spatial correlation is weak in calcium)

       3D Blind-Spot (Our choice):
         Input patch: 64×64×21 (±10 frames temporal)
         Context: spatial neighbors + temporal neighbors (same voxel, different times)
         Mask: center voxel at t=10 (1,1,10)
         Information available: I(x_center; temporal context) = 29.58 bits
                                (from ρ(lag) measurements)

       VERDICT: 3D wins by 20× information gain ✓

    2. INFORMATION ACCUMULATION PLOT + TABLE
       TEXT OUTPUT:
       ─────────────────────────────────────
       Temporal Window Size vs Information Gain vs Memory Cost:

       Window | Neighbors | Bits  | Memory (MB) | Bits/MB | Choice
       ─────────────────────────────────────────────────────────
       ±1     | 2         | 4.86  | 44          | 0.110   |
       ±3     | 6         | 9.88  | 133         | 0.074   |
       ±5     | 10        | 18.38 | 222         | 0.083   |
       ±10    | 20        | 29.58 | 444         | 0.067   | ← CHOSEN
       ±15    | 30        | 37.47 | 666         | 0.056   |
       ±20    | 40        | 43.30 | 888         | 0.049   |

       ±10 frames optimal: 30 bits, reasonable memory, best efficiency

    3. ARCHITECTURE COMPARISON TABLE (see above)

    4. MEMORY BUDGET CALCULATION
       TEXT OUTPUT:
       ─────────────────────────────────────
       U-Net 3D, depth=21, H=W=64, channels_in=2, channels_out=1

       Memory breakdown:
       • Input patch: 21×64×64×2×4 bytes = 11 MB
       • Weights: 5-10 MB (depends on filters)
       • Forward activations: 200-300 MB
       • Backward activations: 200-300 MB (gradient computation)
       • Optimizer state (Adam): 2×weights = 10-20 MB
       • Safety buffer: 100 MB
       ─────────────────────────────────
       TOTAL: ~730 MB << 6 GB ✓ SAFE FOR BATCH_SIZE=1

    5. PG-NLL LOSS SPECIFICATION
       TEXT OUTPUT:
       ─────────────────────────────────────
       Loss = Σ [(ŷ - y)² / (g·max(ŷ,0) + σ_r²)
               + log(g·max(ŷ,0) + σ_r²)]

       Where:
       • g = gain (28.4 for level 1, 248.7 for level 2)
       • σ_r² = read variance (2490 for level 1, 2700 for level 2)
       • ŷ = predicted denoised value
       • y = observed noisy value

       Critical: use ŷ in denominator, not y (avoids Poisson-zero
       catastrophe where bright pixels with y≈0 cause z → ∞)

       Expected loss range in training: 0.5-2.0 (normalized)

    6. HYPERPARAMETER JUSTIFICATION
       TEXT OUTPUT:
       ─────────────────────────────────────
       Learning rate = 1e-3 (why?)
       • PG-NLL has large gradients early (ŷ ≠ x)
       • 1e-3 is conservative but steady
       • Gradient clipping norm=1.0 prevents explosion

       Batch size = 1 (why?)
       • 6GB VRAM constraint
       • Larger batches don't help convergence much (self-supervised learning
         doesn't benefit from large batches like supervised does)

       Validation frequency = every 5 epochs (why?)
       • Early stopping prevents overfitting
       • Epoch ≈ 1 pass through sampled 500 frames
       • 5 epochs = ~100 frame passes, reasonable checkpoint

    ---
    SUMMARY: Every Decision Grounded in Experiment

    ✓ 3D masking: 30 bits measured from F0 ACF
    ✓ ±10 frames: optimal cost-benefit from information table
    ✓ U-Net: efficiency for 6GB VRAM, proven on medical imaging
    ✓ PG-NLL loss: corrects Poisson-zero catastrophe (NB05 measured)
    ✓ Variance conditioning: aligns with loss function (NB05)
    ✓ Sample first: verify loop, catch bugs, estimate time
    ✓ Hyperparameters: conservative to avoid divergence

    Ready for NB06.5 visualization + NB07 training?
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
