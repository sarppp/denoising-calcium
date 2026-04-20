"""06 — Proofs: derivations for docs/findings_summary.md."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _header():
    import marimo as mo
    mo.md(
        r"""
        # Proofs — every claim, with the code

        This notebook is the *derivation ledger* for `docs/findings_summary.md`.
        For each numerical claim in the summary, it shows:

        1. **What** — the exact claim.
        2. **Terms** — the technical vocabulary in one sentence each.
        3. **Code** — the minimum code that produces the number.
        4. **Result** — what we actually read off.

        Read `docs/findings_summary.md` for the narrative. Read this for the
        receipts.
        """
    )
    return (mo,)


@app.cell
def _setup():
    from pathlib import Path
    import numpy as np
    from cidc import (
        NOISE_LEVELS,
        anscombe,
        inverse_anscombe,
        load_stack,
        sample_poisson_gaussian,
        stack_info,
        temporal_autocorr,
    )

    DATA = Path("/app/workspace/data")
    TRAIN = sorted((DATA / "train").glob("*.tif"))
    VAL = sorted((DATA / "val").glob("*.tif"))
    return (
        DATA,
        NOISE_LEVELS,
        anscombe,
        inverse_anscombe,
        load_stack,
        np,
        sample_poisson_gaussian,
        stack_info,
        temporal_autocorr,
    )


@app.cell
def _c1_intro(mo):
    mo.md(r"""
    ## 1 — 8 stacks, `[1500, 490, 490]` `int16`, ≈720 MB each

    **Terms.**
    - **`int16`** — 16-bit signed integer, range −32 768 to 32 767.
    - **ADU (analog-to-digital unit)** — the number stored in each
      pixel after the sensor digitises charge. Not a photon count;
      related by `ADU = gain × photons + read_noise`.
    - **`F0.min == 0`** is our "is this really clean?" test:
      clean data never dips below the subtracted pedestal, noisy
      data always dithers below.
    """)
    return


@app.cell
def _c1_code(DATA, stack_info):
    for _p in sorted(DATA.glob("*/*.tif")):
        _i = stack_info(_p)
        _size_mb = 2 * _i.shape[0] * _i.shape[1] * _i.shape[2] / (1024 ** 2)
        print(f"{_p.parent.name}/{_p.name:8s}  shape={_i.shape}  "
              f"dtype={str(_i.dtype):6s}  min={_i.min:>6.0f}  "
              f"mean={_i.mean:>6.1f}  max={_i.max:>6.0f}  "
              f"size={_size_mb:.0f}MB")
    return


@app.cell
def _c2_intro(mo):
    mo.md(r"""
    ## 2 — `mean(F_k − F_0) ≈ 0` for every noise level

    **Claim in the summary:** noise is *additive* and *zero-mean*
    (no offset to fit). If we subtract the clean image from the
    noisy one, we should be left with pure noise centred on zero.

    **Why it matters:** if `mean(F_k − F_0) ≠ 0`, there would be a
    signal-dependent or signal-independent *bias* that any
    denoiser would have to learn separately.

    **Terms.**
    - **Residual**: `F_k − F_0`. Whatever the noise added.
    """)
    return


@app.cell
def _c2_code(DATA, load_stack, np):
    _F0 = load_stack(DATA / "val" / "F0.tif")
    _t = np.linspace(0, _F0.shape[0] - 1, 150, dtype=int)
    _clean = np.asarray(_F0[_t], dtype=np.float64)

    for _name in ["F1", "F2", "F3"]:
        _Fk = load_stack(DATA / "val" / f"{_name}.tif")
        _noisy = np.asarray(_Fk[_t], dtype=np.float64)
        _r = _noisy - _clean
        print(f"mean({_name} - F0) = {_r.mean():+.3f} ADU   "
              f"std = {_r.std():.1f} ADU")
    return


@app.cell
def _c3_intro(mo):
    mo.md(r"""
    ## 3 — Var = g·Mean + σ_r² (Poisson-Gaussian), with R² ≥ 0.92

    **Terms.**
    - **Poisson noise (shot noise):** `N ~ Poisson(λ)` with
      `mean = variance = λ`. Inevitable photon-counting fluctuation.
    - **Gaussian read noise:** `r ~ N(0, σ_r²)`, independent of
      signal, from sensor electronics.
    - **Gain `g`:** ADU per photon.
    - **Combined model:** `y = g·Poisson(λ) + r`, giving the key
      identity

    $$
    \operatorname{Var}[y] = g \cdot \mathbb{E}[y] + \sigma_r^2.
    $$

    - **R² (coefficient of determination):** fraction of the
      variance of `y` explained by the line. `1` is perfect;
      `0` means the line is no better than the mean. `R² ≥ 0.92`
      on the noisy stacks is our "yes, this IS the noise model"
      evidence.

    **Procedure.**
    1. Sample 100 000 random pixels.
    2. For each, compute `(mean, variance)` across 1500 frames.
    3. Use only background-dominated pixels (bottom 80 % of mean)
       so a few bright active neurons don't distort the slope.
    4. Least-squares fit `v = g·m + c`. Report `R²`.
    """)
    return


@app.cell
def _c3_code(DATA, load_stack, np):
    def _fit_pg(path):
        s = load_stack(path)
        rng = np.random.default_rng(0)
        idx = rng.choice(490 * 490, size=100_000, replace=False)
        y, x = np.divmod(idx, 490)
        tr = np.asarray(s[:, y, x], dtype=np.float64)
        m, v = tr.mean(axis=0), tr.var(axis=0)
        q = np.quantile(m, 0.8)
        mask = m < q
        A = np.vstack([m[mask], np.ones(mask.sum())]).T
        (g, c), *_ = np.linalg.lstsq(A, v[mask], rcond=None)
        pred = A @ [g, c]
        ss_res = ((v[mask] - pred) ** 2).sum()
        ss_tot = ((v[mask] - v[mask].mean()) ** 2).sum()
        return float(g), float(c), 1.0 - ss_res / ss_tot

    print(f"{'file':20s}  {'gain g':>8s}  {'σ_r²':>8s}  {'R²':>6s}")
    for _p in sorted(DATA.glob("*/*.tif")):
        _g, _c, _r2 = _fit_pg(_p)
        print(f"{_p.parent.name+'/'+_p.name:20s}  {_g:8.1f}  {_c:8.0f}  {_r2:6.2f}")
    return


@app.cell
def _c3_note(mo):
    mo.md(r"""
    **Reading the output.**
    - A1, B1, F1 → `g ≈ 28` (level 1).
    - C2, D2, F2 → `g ≈ 249` (level 2).
    - F3 → `g ≈ 991` (level 3, OOD for Task 2).
    - `σ_r² ≈ 2 500` ± some inflation → single physical sensor.
    - F0 → `R² ≈ 0.09` (nonsense line fit) → confirms F0 is clean.
    - Every noisy stack → `R² ≥ 0.92` → model *is* Poisson-Gaussian.
    """)
    return


@app.cell
def _c4_intro(mo):
    mo.md(r"""
    ## 4 — Star topology: F0 is the clean of F1, F2, F3 independently

    **Competing hypotheses.**
    - **Star:** F0 is clean; F1/F2/F3 are *independent* noisy
      realisations of F0.
    - **Chain:** F1 is noisy F0; F2 is noisy F1; F3 is noisy F2.

    **The discriminator.** Under the chain hypothesis, F1 is closer
    to F2 than F0 is (because F1→F2 adds one noise; F0→F2 adds two).
    Under the star, F0 is the common ancestor and therefore the
    best predictor of every F_k.

    **Terms.**
    - **Pearson correlation `corr(A, B)`** — cosine of the angle
      between mean-centred A and mean-centred B. Ranges −1 to +1.
      +1 = identical up to scale+offset; 0 = no linear relation.

    **Measured (averaged over 30 frames).**

    | pair | corr |
    |---|---|
    | F0 ↔ F1 | **0.74** |
    | F0 ↔ F2 | **0.40** |
    | F0 ↔ F3 | **0.22** |
    | F1 ↔ F2 | 0.30 |
    | F1 ↔ F3 | 0.16 |
    | F2 ↔ F3 | 0.09 |

    `corr(F0, F2) = 0.40 > corr(F1, F2) = 0.30` ⇒ **star wins**.
    """)
    return


@app.cell
def _c4_code(DATA, load_stack, np):
    def _pearson(A, B):
        A = A.astype(np.float64).ravel()
        B = B.astype(np.float64).ravel()
        A, B = A - A.mean(), B - B.mean()
        return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))

    def _avg_corr(p1, p2, n=30):
        a, b = load_stack(p1), load_stack(p2)
        idx = np.linspace(0, a.shape[0] - 1, n, dtype=int)
        return np.mean([_pearson(np.asarray(a[t]), np.asarray(b[t])) for t in idx])

    _pairs = [("F0", "F1"), ("F0", "F2"), ("F0", "F3"),
              ("F1", "F2"), ("F1", "F3"), ("F2", "F3")]
    for _a, _b in _pairs:
        _c = _avg_corr(DATA / "val" / f"{_a}.tif", DATA / "val" / f"{_b}.tif")
        print(f"corr({_a}, {_b}) = {_c:+.3f}")
    return


@app.cell
def _c5_intro(mo):
    mo.md(r"""
    ## 5 — Spatial: neurons are tiny and sparse

    **Claim.** On the temporal-mean image (`F0.mean(axis=0)`):
    bright fraction 0.3–0.9 %, 120–270 neurons per frame, neuron
    radius 2–3 px.

    **Terms.**
    - **Temporal mean image** — `mean over time` of every pixel.
      A cheap denoiser: averaging 1500 frames drops noise by √1500.
      What's left is the *structural* image — where the neurons
      live regardless of when they spiked.
    - **MAD (median absolute deviation)** — `median(|x − median(x)|)`.
      Robust substitute for `std`. `σ ≈ 1.4826 · MAD` for Gaussian
      data. Used here because a few active neurons would blow up
      a non-robust `mean + k·std` threshold, but can't skew the
      median.
    - **Threshold** — any pixel `> median + 5·1.4826·MAD` is called
      "bright" (5-sigma, robustly).
    - **Connected components** — contiguous bright-pixel blobs.
      Each is one candidate neuron. `radius ≈ √(area/π)`.
    """)
    return


@app.cell
def _c5_code(DATA, load_stack, np):
    from scipy import ndimage

    for _p in sorted(DATA.glob("*/*.tif")):
        _s = load_stack(_p)
        _img = np.asarray(_s[::10], dtype=np.float32).mean(axis=0)
        _med = np.median(_img)
        _mad = np.median(np.abs(_img - _med))
        _thr = _med + 5 * 1.4826 * _mad
        _bright = _img > _thr
        _frac = _bright.mean() * 100
        _lbl, _n = ndimage.label(_bright)
        _areas = ndimage.sum(_bright, _lbl, range(1, _n + 1))
        _areas = _areas[_areas >= 3]  # drop 1-2 pixel specks
        _radii = np.sqrt(_areas / np.pi) if _areas.size else np.array([0.0])
        print(f"{_p.parent.name}/{_p.name:8s}  bright={_frac:4.2f}%  "
              f"neurons={len(_areas):4d}  r_mean={_radii.mean():4.2f}px  "
              f"r_max={_radii.max():4.1f}px")
    return


@app.cell
def _c6_intro(mo):
    mo.md(r"""
    ## 6 — ACF[1] on F0 is 0.995; on F1/F2/F3 it collapses

    **Terms.**
    - **Autocorrelation function `ACF[k]`** —
      `corr(x[t], x[t+k])`, pixel trace vs itself shifted by `k`
      frames. `ACF[0] = 1` always.
    - **White noise** — `ACF[k] = 0` for all `k ≥ 1`.
    - **Slow signal** — `ACF[k]` decays gradually. The lag at
      which `ACF` crosses 0.5 is **τ(0.5)**, a natural timescale.

    **On F0:** `ACF[1] = 0.995`, `τ(0.5) = 45 frames`. At 30 Hz
    that's ~1.5 s — consistent with the decay of a calcium
    transient (GCaMP kinetics).

    **On F1/F2/F3:** ACF collapses to near zero at lag 1. Not
    because signal is gone — because noise variance dominates.
    Next cell makes that quantitative.
    """)
    return


@app.cell
def _c6_code(DATA, load_stack, np, temporal_autocorr):
    for _name in ["F0", "F1", "F2", "F3"]:
        _acf = temporal_autocorr(load_stack(DATA / "val" / f"{_name}.tif"),
                                 max_lag=60)
        # τ(0.5) = first lag where ACF crosses 0.5
        _tau = int(np.argmax(_acf < 0.5)) if (_acf < 0.5).any() else len(_acf)
        print(f"{_name}:  ACF[1]={_acf[1]:.4f}  ACF[10]={_acf[10]:.4f}  "
              f"ACF[30]={_acf[30]:.4f}  τ(0.5)={_tau}")
    return


@app.cell
def _c7_intro(mo):
    mo.md(r"""
    ## 7 — SNR in dB derived from ACF[1]

    **The trick.** For a stack modelled as `total = signal + noise`
    with white noise independent across frames and slow signal,
    lag-1 autocorrelation satisfies

    $$
    \mathrm{ACF}_{\text{total}}(1) \approx
      \frac{\mathrm{Var}(\text{signal})}
           {\mathrm{Var}(\text{signal}) + \mathrm{Var}(\text{noise})}.
    $$

    So **signal power fraction ≈ ACF(1)** (at lag 1, because noise
    has zero autocorrelation there and signal has ~ACF(1) of F0).

    **SNR in dB.**

    $$
    \mathrm{SNR}_{\mathrm{dB}} = 10\,\log_{10}\!\left(\frac{\mathrm{ACF}(1)}{1 - \mathrm{ACF}(1)}\right).
    $$

    **Terms.**
    - **Decibel (dB)** — a log-scaled ratio. `10 dB` = 10× more
      power. Negative dB = signal weaker than noise.
    """)
    return


@app.cell
def _c7_code(DATA, load_stack, np, temporal_autocorr):
    for _name in ["F1", "F2", "F3"]:
        _acf = temporal_autocorr(load_stack(DATA / "val" / f"{_name}.tif"),
                                 max_lag=2)
        _ratio = _acf[1] / (1 - _acf[1])
        _snr_db = 10 * np.log10(max(_ratio, 1e-9))
        print(f"{_name}:  ACF[1]={_acf[1]:.4f}  "
              f"signal/noise={_ratio:.4f}  SNR={_snr_db:+.1f} dB")
    return


@app.cell
def _c8_intro(mo):
    mo.md(r"""
    ## 8 — Our noise sampler matches reality within ~4 %

    **Test.** Given our measured `(g, σ_r²)` per level, simulate
    `F_k_sim = F_0 + noise(level_k)`. The *residual-variance slope*
    of the simulated stack vs F0 should match the *residual-variance
    slope* of the real `F_k` vs F0.

    **Terms.**
    - **Residual-variance slope** — fit `Var(F_k − F_0)` vs `F_0`
      intensity. Under Poisson-Gaussian, this slope *is the gain*.
      So we're comparing measured gain (from real residuals) to
      measured gain (from simulated residuals).

    **Why this matters for Task 2.** If the sampler is right, we
    can generate training data at any gain — including gains near
    F3's 991 — from any "approximately clean" estimate. That's
    how we bridge the OOD gap without ever training on F3.
    """)
    return


@app.cell
def _c8_code(DATA, NOISE_LEVELS, load_stack, np, sample_poisson_gaussian):
    def _slope_from_residual(clean, noisy):
        a = clean.astype(np.float64)
        b = noisy.astype(np.float64)
        r = b - a
        bins = np.linspace(a.min(), a.max(), 30)
        w = np.digitize(a.ravel(), bins)
        m, v = [], []
        for i in range(1, len(bins)):
            mask = w == i
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
        _gr = _slope_from_residual(_clean, _real)
        _gs = _slope_from_residual(_clean, _sim)
        print(f"{_name}:  real gain={_gr:7.1f}   sim gain={_gs:7.1f}   "
              f"sim/real={_gs / _gr:.3f}")
    return


@app.cell
def _c9_intro(mo):
    mo.md(r"""
    ## 9 — The Anscombe inverse bug and its fix

    **Terms.**
    - **Variance-stabilising transform (VST)** — a monotone `f`
      such that `Var[f(Y)]` is approximately constant across
      intensities. Anscombe is the canonical VST for Poisson data.
    - **Round-trip bias** — `mean( inverse_anscombe(anscombe(y)) − y )`.
      A correct inverse has bias ≈ 0; a wrong inverse has a bias
      proportional to gain.

    **The Mäkitalo-Foi (2011) formula** has the `1/z` coefficient
    as `√(3/2) / 4`. I initially coded it as `√(3/2) / 2` — a 2×
    error. The round-trip bias was exactly `~0.3 × gain`, which is
    what a 2× error on that one term predicts.

    Below we verify the fix on synthetic data.
    """)
    return


@app.cell
def _c9_code(
    NOISE_LEVELS,
    anscombe,
    inverse_anscombe,
    np,
    sample_poisson_gaussian,
):
    _rng = np.random.default_rng(0)
    _clean = np.linspace(0, 3000, 2000).repeat(200).reshape(2000, 200)

    print(f"{'level':>7s}  {'gain':>6s}  {'round-trip bias (ADU)':>24s}")
    for _lvl, _params in NOISE_LEVELS.items():
        _y = sample_poisson_gaussian(_clean, _params, rng=_rng)
        _y_back = inverse_anscombe(anscombe(_y, _params), _params)
        print(f"  level {_lvl}  {_params.gain:6.1f}  "
              f"{float((_y_back - _clean).mean()):+24.2f}")
    return


@app.cell
def _c10_intro(mo):
    mo.md(r"""
    ## 10 — Success signal: point to a real calcium transient

    If you've followed this notebook, you should now be able to
    **look at `F0`, find one neuron, isolate one spike, and describe
    it quantitatively**. Let's do it.

    **What a calcium transient is.** When a neuron fires, Ca²⁺
    floods into the cell. GCaMP (a fluorescent calcium sensor)
    binds Ca²⁺ and lights up. Result: a sharp *rise* in fluorescence
    over a few frames, then a slower *exponential decay* as calcium
    is pumped back out.

    **Terms we'll use.**
    - **Baseline (`F_0`, unfortunate symbol collision — here it
      means the resting fluorescence of the pixel, not the file
      F0).** Median intensity of the trace. The "off" level.
    - **Peak amplitude (`ΔF`).** Peak − baseline.
    - **`ΔF/F`.** Fractional change, `(peak − baseline) / baseline`.
      Standard dimensionless measure of how much the neuron lit up.
    - **Time-to-peak.** Frames from onset (first frame above a
      threshold) to peak.
    - **FWHM (full-width at half-maximum).** How many frames the
      transient spends above `baseline + ΔF/2`. A simple shape
      descriptor.
    - **Exponential decay constant τ.** Fit `A·exp(-t/τ) + baseline`
      to the *falling* side. τ = number of frames for the transient
      to drop to `1/e` (~37 %) of its peak. Related to τ(0.5) by
      `τ(0.5) = τ · ln(2) ≈ 0.69 · τ`.

    **Procedure.**
    1. Find the brightest pixel in the `F0` temporal mean.
    2. Pull its full 1500-frame trace.
    3. Estimate baseline = median of the trace (robust to spikes).
    4. Find the tallest peak.
    5. Measure amplitude, ΔF/F, time-to-peak, FWHM.
    6. Fit an exponential to the decay side and read off τ.
    7. Plot the trace with annotations so you can *see* it.
    """)
    return


@app.cell
def _c10_code(DATA, load_stack, np):
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    # 1. brightest pixel in F0 temporal mean
    _F0 = load_stack(DATA / "val" / "F0.tif")
    _tmean = np.asarray(_F0[::10], dtype=np.float32).mean(axis=0)
    _y, _x = np.unravel_index(int(_tmean.argmax()), _tmean.shape)
    print(f"brightest pixel: (y={_y}, x={_x})")

    # 2. full trace
    _trace = np.asarray(_F0[:, _y, _x], dtype=np.float32)

    # 3. baseline (median)
    _baseline = float(np.median(_trace))
    print(f"baseline = {_baseline:.1f} ADU")

    # 4. tallest peak
    _peak_i = int(_trace.argmax())
    _peak_v = float(_trace[_peak_i])
    _dF = _peak_v - _baseline
    _dFoF = _dF / _baseline
    print(f"peak at frame {_peak_i}: F={_peak_v:.1f} ADU, "
          f"ΔF={_dF:.1f} ADU, ΔF/F={_dFoF:.2f}")

    # 5. find transient window (above baseline + ΔF/2 around peak)
    _half = _baseline + _dF / 2
    _above = _trace > _half
    # Expand from peak outward until we leave the above-half region
    _left = _peak_i
    while _left > 0 and _above[_left - 1]:
        _left -= 1
    _right = _peak_i
    while _right < len(_trace) - 1 and _above[_right + 1]:
        _right += 1
    _fwhm = _right - _left
    _time_to_peak = _peak_i - _left
    print(f"onset (half-max) = frame {_left}")
    print(f"offset (half-max) = frame {_right}")
    print(f"time-to-peak = {_time_to_peak} frames")
    print(f"FWHM = {_fwhm} frames")

    # 6. exponential fit to decay side
    def _exp_decay(t, A, tau, c):
        return A * np.exp(-t / tau) + c

    _decay_end = min(_peak_i + 200, len(_trace))
    _t_decay = np.arange(_decay_end - _peak_i)
    _y_decay = _trace[_peak_i:_decay_end]
    try:
        _popt, _ = curve_fit(_exp_decay, _t_decay, _y_decay,
                             p0=[_dF, 30.0, _baseline], maxfev=2000)
        _A_fit, _tau_fit, _c_fit = _popt
        _tau_half = _tau_fit * np.log(2)
        print(f"exp-decay fit: A={_A_fit:.1f}  τ={_tau_fit:.1f} frames  "
              f"c={_c_fit:.1f}")
        print(f"half-decay τ(0.5) = τ·ln(2) = {_tau_half:.1f} frames")
    except RuntimeError:
        _A_fit = _tau_fit = _c_fit = None
        print("exp-decay fit failed")

    # 7. plot
    _win_left = max(_left - 40, 0)
    _win_right = min(_right + 100, len(_trace))
    _fig, _ax = plt.subplots(figsize=(11, 4))
    _ax.plot(_trace, lw=0.6, color="0.7", label="full trace")
    _ax.plot(range(_win_left, _win_right),
             _trace[_win_left:_win_right], lw=1.3, color="C0",
             label="transient window")
    _ax.axhline(_baseline, color="k", lw=0.5, ls="--", label="baseline")
    _ax.axhline(_half, color="C3", lw=0.5, ls=":", label="half-max")
    _ax.scatter([_peak_i], [_peak_v], s=50, color="C3", zorder=5,
                label=f"peak  ΔF/F={_dFoF:.2f}")
    if _tau_fit is not None:
        _xs = np.arange(_peak_i, _decay_end)
        _ax.plot(_xs, _exp_decay(_xs - _peak_i, _A_fit, _tau_fit, _c_fit),
                 color="C2", lw=1.2, label=f"exp fit  τ={_tau_fit:.1f} f")
    _ax.set_xlim(_win_left, _win_right)
    _ax.set_xlabel("frame")
    _ax.set_ylabel("intensity (ADU)")
    _ax.set_title(f"F0 pixel ({_y}, {_x}) — calcium transient "
                  f"(ΔF/F={_dFoF:.2f}, FWHM={_fwhm} f, τ={_tau_fit:.1f} f)")
    _ax.legend(loc="upper right", fontsize=8)
    _fig
    return


@app.cell
def _c10_reading(mo):
    mo.md(r"""
    **What you just measured.**

    - A real event: *this* neuron fired at *this* frame and the
      fluorescence rose by some `ΔF/F`.
    - A *shape*: a few frames of rise, then a roughly exponential
      decay with time constant τ.
    - The τ you fit should be consistent with the bulk
      `τ(0.5) = 45 frames` we reported for F0 in §6 — single-
      transient τ is usually in the same ballpark as the
      population ACF half-life, since both reflect GCaMP kinetics.

    **At 30 Hz**, a τ of ~30–60 frames means 1–2 s decay. That's
    textbook GCaMP6f/s kinetics — a sanity check that we're
    actually looking at biological calcium, not some artefact.

    **Now you can:**

    - Identify a neuron spatially (we did: `(y, x) = argmax` of the
      temporal mean).
    - Isolate a single event temporally (we did: peak frame, FWHM
      window).
    - Describe its shape with three numbers: amplitude (ΔF/F),
      width (FWHM), decay (τ).

    That is the **success signal** for phase-1 data understanding.
    When your denoiser runs on `F1`, `F2`, or `F3`, these are the
    three numbers you'll compare against the `F0` ground truth to
    decide whether it actually recovered the signal or just
    produced a smooth-looking blur.
    """)
    return


@app.cell
def _final_table(mo):
    mo.md(r"""
    ## Where each claim was proved

    | Summary claim | Cell |
    |---|---|
    | Shape, dtype, 720 MB, $F_0.\min = 0$ | §1 |
    | $\operatorname{mean}(F_k - F_0) \approx 0$ | §2 |
    | $\operatorname{Var} = g \cdot \operatorname{Mean} + \sigma_r^2$, $R^2 \geq 0.92$ | §3 |
    | Gain ladder $28 / 249 / 991$ | §3 |
    | $\sigma_r^2$ constant $\Rightarrow$ one sensor | §3 |
    | Star not chain (correlations) | §4 |
    | Neurons 2–3 px, 120–270 per frame | §5 |
    | $\mathrm{ACF}(1) = 0.995$ on $F_0$, collapses on $F_k$ | §6 |
    | $\tau(0.5) = 45$ frames on $F_0$ | §6 |
    | $\mathrm{SNR} = -14 / -21 / -24$ dB | §7 |
    | Sampler matches within 4 % | §8 |
    | Anscombe coefficient bug fixed | §9 |
    | Point to a real transient, describe $\Delta F / F$, FWHM, $\tau$ | §10 |

    This is also the order you'd re-verify them if anything
    breaks. Every number in `docs/findings_summary.md` traces
    back to one cell in this notebook.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
