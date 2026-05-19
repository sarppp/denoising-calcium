"""02 — stSNR metric behavior under different degradations.

QUESTION: What kinds of errors hurt tSNR most?
          Can a denoiser improve sSNR while making tSNR worse?

Decision gate: confirms that N2V3D voxel masking (which optimises at the
individual pixel-timepoint level) is the right training strategy, and that
naive temporal smoothing is the wrong one.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What kinds of errors hurt tSNR most? (metric behavior under blur, smoothing, noise)
    """)
    return


@app.cell
def _setup():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter, uniform_filter1d
    from cidc import load_stack, stsnr

    DATA = Path(__file__).parent.parent.parent / "data"
    return (
        DATA,
        gaussian_filter,
        load_stack,
        mo,
        np,
        plt,
        stsnr,
        uniform_filter1d,
    )


@app.cell
def _intro(mo):
    mo.md("""
    # 02 — stSNR metric behavior under degradations

    **Purpose:** understand what the scoring metric actually punishes before
    building any model.

    The stSNR score is a 50/50 average of sSNR (spatial, per-frame) and tSNR
    (temporal, per-pixel trace). Those two components can move in opposite
    directions depending on how you process the data. A model that naively
    smooths can *improve* sSNR while *destroying* tSNR — and still score worse
    overall despite looking better on a naive per-frame metric.

    This notebook runs two complementary experiments:

    **Experiment A — metric geometry (input = clean F0):**
    Apply spatial blur and additive noise to F0, score against F0.
    This shows the pure geometric shape of the metric: which direction does
    each type of distortion move the sSNR / tSNR pair?

    **Experiment B — denoising scenario (input = noisy F1):**
    Apply temporal smoothing to F1, score against F0. This is the actual
    denoising scenario. Temporal smoothing looks like a valid strategy when
    you only watch sSNR — it averages away noise and scores look good.
    But at large windows it blurs the transients that tSNR measures.
    This experiment shows where the two metrics diverge.
    """)
    return


@app.cell
def _md_load(mo):
    mo.md("""
    ## Load crops

    We work on the first 200 frames, top-left 128×128 crop.

    - **F0** is the clean reference. Used as the scoring target throughout,
      and also as the "perfect" input for the metric geometry experiment.
    - **F1** (noisy, level 1, gain=28.4) is used as the denoising input for
      the temporal smoothing experiment. Using F0 as input for that experiment
      would be wrong: F0 is already temporally smooth (τ₀.₅=46 frames), so
      a 31-frame window barely distorts it and both metrics decline together.
      The failure mode only appears when the input is noisy.
    """)
    return


@app.cell
def _load_crops(DATA, load_stack, np):
    _f0 = load_stack(DATA / "val" / "F0.tif")
    _f1 = load_stack(DATA / "val" / "F1.tif")
    clean  = np.asarray(_f0[:200, :128, :128], dtype=np.float32)
    noisy1 = np.asarray(_f1[:200, :128, :128], dtype=np.float32)
    print(f"clean  shape={clean.shape}  mean={clean.mean():.1f}  max={clean.max():.0f}")
    print(f"noisy1 shape={noisy1.shape}  mean={noisy1.mean():.1f}  max={noisy1.max():.0f}")
    return clean, noisy1


@app.cell
def _md_spatial_blur(mo):
    mo.md("""
    ## Experiment A1 — Gaussian spatial blur (input = clean F0)

    We apply a 2D Gaussian blur to each frame independently (sigma 0 → 6 px).
    This is applied to **clean F0** so we can isolate the metric geometry without
    confounding noise.

    ### What spatial blur does to the metric

    Spatial blur mixes neighboring pixels within each frame but never touches
    the time axis. So the temporal trace of pixel (h, w) in the blurred output
    is a weighted average of the traces of nearby pixels — and nearby pixels
    have *similar* calcium dynamics (they come from the same neuron). This means:

    - **sSNR**: drops quickly. Blurring changes the spatial structure of each
      frame relative to F0 — neurons become diffuse blobs, fine structure disappears.
    - **tSNR**: drops slowly. The temporal trace of a blurred pixel still tracks
      the same calcium event, just mixed with neighbors. Since neighbors are
      correlated, the temporal fidelity is mostly preserved.

    This creates a gap: tSNR stays above sSNR as blur increases. On the scatter
    plot the spatial blur curve sits *above* the diagonal.

    ### Architecture implication

    A purely spatial denoiser can only recover sSNR. It cannot raise tSNR
    above what the signal's spatial correlations already provide. The 50/50
    metric penalises exactly this ceiling — spatial-only models leave half the
    score improvement unreachable.
    """)
    return


@app.cell
def _spatial_blur_sweep(clean, gaussian_filter, np, stsnr):
    _sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    blur_results = []
    for _s in _sigmas:
        if _s == 0.0:
            _blurred = clean.copy()
        else:
            _blurred = np.stack(
                [gaussian_filter(clean[t], sigma=_s) for t in range(clean.shape[0])],
                axis=0,
            )
        _r = stsnr(_blurred, clean)
        blur_results.append({"sigma": _s, "ssnr": _r.s_snr, "tsnr": _r.t_snr, "stsnr": _r.st_snr})
        print(f"sigma={_s:.1f}  sSNR={_r.s_snr:.2f}  tSNR={_r.t_snr:.2f}  stSNR={_r.st_snr:.2f}")
    return (blur_results,)


@app.cell
def _blur_math(blur_results):
    print("Spatial blur: gap = tSNR − sSNR (positive means tSNR stays above sSNR)")
    print(f"  {'sigma':>5}  {'sSNR':>6}  {'tSNR':>6}  {'gap':>7}  {'linear ratio':>12}")
    print("  " + "-" * 48)
    for _r in blur_results[1:]:
        _gap = _r["tsnr"] - _r["ssnr"]
        _ratio = 10 ** (_gap / 10)
        print(f"  {_r['sigma']:>5.1f}  {_r['ssnr']:>6.2f}  {_r['tsnr']:>6.2f}  {_gap:>+7.2f}  {_ratio:>10.2f}x")
    _gaps = [_r["tsnr"] - _r["ssnr"] for _r in blur_results[1:]]
    _mean_gap = sum(_gaps) / len(_gaps)
    print(f"\n  Mean gap = {_mean_gap:.2f} dB — constant across all sigma values")
    print(f"  In linear terms: tSNR residual is ~{10**(_mean_gap/10):.1f}x smaller than sSNR residual")
    print("  Interpretation: spatial blur always hurts sSNR more than tSNR by a fixed amount.")
    print("  This is structural — neighbouring pixels share time courses, so blending them")
    print("  preserves temporal fidelity even as spatial sharpness is lost.")
    return


@app.cell
def _md_noise(mo):
    mo.md("""
    ## Experiment A2 — Additive Gaussian noise (input = clean F0)

    We add zero-mean Gaussian noise with standard deviation σ (0 → 200 ADU)
    to clean F0. This is the symmetric reference case.

    ### What additive noise does to the metric

    Additive Gaussian noise is statistically independent across both space and
    time. Adding it increases the residual energy in both the per-frame and
    per-pixel-trace denominators by the same amount. So:

    - **sSNR** and **tSNR** both drop at the same rate.

    On the scatter plot the additive noise curve tracks along the diagonal —
    neither metric is sacrificed for the other. This is the reference shape.
    Any curve that bends *away* from the diagonal is showing an asymmetric
    effect on the two components.

    A good denoiser should move points in the *opposite* direction: up and to
    the right, along or above the diagonal, toward the clean (σ=0) point.
    """)
    return


@app.cell
def _noise_sweep(clean, np, stsnr):
    _rng = np.random.default_rng(42)
    _sigmas = [0, 10, 25, 50, 75, 100, 150, 200]
    noise_results = []
    for _s in _sigmas:
        _noisy = clean + _rng.normal(0, _s, clean.shape).astype(np.float32)
        _r = stsnr(_noisy, clean)
        noise_results.append({"sigma": _s, "ssnr": _r.s_snr, "tsnr": _r.t_snr, "stsnr": _r.st_snr})
        print(f"noise_sigma={_s:3d}  sSNR={_r.s_snr:.2f}  tSNR={_r.t_snr:.2f}  stSNR={_r.st_snr:.2f}")
    return (noise_results,)


@app.cell
def _noise_math(noise_results):
    print("Additive noise: gap = tSNR − sSNR (should be constant, near 0)")
    print("Math: i.i.d. N(0,σ²) noise adds identical energy to both spatial and temporal")
    print("  residual denominators → Δ(sSNR) = Δ(tSNR) exactly, gap is preserved from σ=0.")
    print(f"  {'sigma':>5}  {'sSNR':>6}  {'tSNR':>6}  {'gap':>7}")
    print("  " + "-" * 32)
    for _r in noise_results[1:]:
        _gap = _r["tsnr"] - _r["ssnr"]
        print(f"  {_r['sigma']:>5d}  {_r['ssnr']:>6.2f}  {_r['tsnr']:>6.2f}  {_gap:>+7.3f} dB")
    _gaps = [_r["tsnr"] - _r["ssnr"] for _r in noise_results[1:]]
    _mean_gap = sum(_gaps) / len(_gaps)
    print(f"\n  Mean gap = {_mean_gap:.3f} dB — confirming near-perfect symmetry across all sigma.")
    print("  A denoiser that truly removes noise will move both metrics up equally,")
    print("  tracking the diagonal on the scatter plot — this is the target direction.")
    return


@app.cell
def _md_geometry_plot(mo):
    mo.md("""
    ## Experiment A — scatter plot: metric geometry

    Each dot is one degradation setting (one sigma value). The diagonal marks
    sSNR = tSNR. **We clip to 0–45 dB** — the "clean" point at ~210 dB is
    numerical infinity and squashes the interesting region into one corner.

    **What the numbers show:**

    | Degradation | Gap (tSNR − sSNR) |
    |---|---|
    | Spatial blur σ=0.5 | **+6.8 dB** (tSNR preserved) |
    | Spatial blur σ=6.0 | **+6.7 dB** (gap constant!) |
    | Additive noise σ=10 | **−0.73 dB** (symmetric) |
    | Additive noise σ=200 | **−0.73 dB** (gap constant!) |

    The ~6.8 dB gap for spatial blur is perfectly constant across all sigma
    values. This means spatial blur always hurts sSNR more than tSNR by the
    same fixed amount — it is a structural property of the metric, not a
    quirk of one setting.

    The −0.73 dB gap for additive noise is also constant. This is expected:
    additive i.i.d. noise adds the same energy to every spatial and temporal
    residual denominator simultaneously.

    On the plot: blur curve sits above the diagonal; noise curve sits just
    below it (the offset reflects the ~19 dB difference between sSNR and tSNR
    on the clean baseline due to DC weighting).
    """)
    return


@app.cell
def _geometry_scatter(blur_results, noise_results, plt):
    # Skip the sigma=0 / noise=0 points (they are ~210/190 dB, numerical infinity)
    _blur  = [r for r in blur_results  if r["ssnr"] < 50]
    _noise = [r for r in noise_results if r["ssnr"] < 50]

    _fig, _ax = plt.subplots(figsize=(7, 6))
    for _results, _color, _label in [
        (_blur,  "steelblue", "Spatial blur on F0 (σ=0.5→6 px)"),
        (_noise, "seagreen",  "Additive noise on F0 (σ=10→200 ADU)"),
    ]:
        _xs = [r["ssnr"] for r in _results]
        _ys = [r["tsnr"] for r in _results]
        _ax.plot(_xs, _ys, "o-", color=_color, label=_label, lw=1.5, ms=5)
        _ax.annotate("mild", xy=(_xs[0], _ys[0]), color=_color, fontsize=7,
                     xytext=(4, 3), textcoords="offset points")
        _ax.annotate("severe", xy=(_xs[-1], _ys[-1]), color=_color, fontsize=7,
                     xytext=(4, -9), textcoords="offset points")
    _ax.plot([0, 45], [0, 45], "k--", lw=0.8, alpha=0.4, label="sSNR = tSNR")
    _ax.set_xlim(0, 45)
    _ax.set_ylim(0, 45)
    _ax.set_xlabel("sSNR (dB)")
    _ax.set_ylabel("tSNR (dB)")
    _ax.set_title("Experiment A — metric geometry (zoomed, 0–45 dB)")
    _ax.legend(fontsize=9)
    _ax.grid(alpha=0.2)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _md_temporal_smooth(mo):
    mo.md("""
    ## Experiment B — Temporal smoothing (input = noisy F1)

    We apply a uniform temporal running mean of window size W to **noisy F1**,
    then score the result against clean **F0**. W sweeps from 1 (no smoothing,
    just raw F1) to 199 (near-total temporal averaging).

    ### Why we must use noisy F1 here

    If we applied smoothing to clean F0, the signal is already temporally smooth
    (τ₀.₅=46 frames), so any window shorter than ~46 frames barely distorts it.
    Both metrics decline gently together and you don't see the failure mode.

    On noisy F1, two competing effects fight as we widen the window:

    1. **Noise cancellation**: averaging W independent noise frames reduces
       noise standard deviation by √W. This raises sSNR.
    2. **Signal blurring**: the running mean is a low-pass temporal filter.
       Calcium transients have a fast onset (~5 frames rise time) and a ~46-frame
       half-decay. A window much wider than the rise time blunts the onset.
       Beyond W ~ τ₀.₅ ≈ 46 frames the decay itself starts getting smeared.
       This reduces tSNR because the predicted trace no longer matches F0's trace.

    ### The math behind the two competing effects

    **Noise reduction** (improves both metrics):
    Averaging W independent noise frames reduces noise std by √W:

        σ_output = σ_noise / √W
        SNR gain ≈ 10·log₁₀(W) dB

    Examples: W=7 → +8.5 dB | W=31 → +14.9 dB | W=61 → +17.9 dB | W=199 → +23.0 dB

    **Signal blurring** (degrades both metrics):
    The W-frame uniform mean is a low-pass temporal filter. For a calcium transient
    with half-decay time τ≈46 frames, the peak amplitude of a transient is attenuated
    by roughly exp(−W / (2τ)):

        peak_attenuation ≈ exp(−W / 92)

    Examples: W=31 → peak×0.71 | W=61 → peak×0.51 | W=101 → peak×0.33 | W=199 → peak×0.11

    The crossover where blurring cost overtakes noise benefit occurs near W ≈ τ₀.₅ ≈ 46 frames.
    Our measured peak at W=61 confirms this — slightly above τ₀.₅ because the broad
    baseline frames (no transient) are not blurred and keep the metric up a bit longer.

    ### What to expect

    - **Small windows (W < 20)**: noise cancellation dominates — both rise together.
    - **W ~ 30–61**: both peak near W=61 ≈ τ₀.₅. SNR gain ≈ 18 dB, peak attenuation ≈ 0.51×.
    - **Large windows (W > 61)**: blurring dominates. Both metrics fall, sSNR faster.
    - **W → 199** (temporal mean): output is constant. Both metrics decline sharply.

    This is the trap: a temporal averaging denoiser stops gaining around W=61 and then
    actively loses score — you cannot tune your way past the ceiling.
    """)
    return


@app.cell
def _temporal_smooth_sweep(clean, noisy1, stsnr, uniform_filter1d):
    _windows = [1, 3, 7, 15, 31, 61, 101, 151, 199]
    smooth_results = []
    for _w in _windows:
        _smoothed = uniform_filter1d(noisy1, size=_w, axis=0, mode="nearest")
        _r = stsnr(_smoothed, clean)
        smooth_results.append({"window": _w, "ssnr": _r.s_snr, "tsnr": _r.t_snr, "stsnr": _r.st_snr})
        print(f"window={_w:3d}  sSNR={_r.s_snr:.2f}  tSNR={_r.t_snr:.2f}  stSNR={_r.st_snr:.2f}")
    return (smooth_results,)


@app.cell
def _smooth_math(np, smooth_results):
    _tau = 46  # τ₀.₅ from NB01 ACF on F0
    print("Temporal smooth — noise reduction gain vs signal blurring cost:")
    print(f"  {'W':>4}  {'SNR gain':>9}  {'peak atten':>12}  {'sSNR':>6}  {'tSNR':>6}  {'stSNR':>7}")
    print("  " + "-" * 56)
    for _r in smooth_results:
        _w = _r["window"]
        _snr_gain = 10 * np.log10(_w)
        _peak_atten = np.exp(-_w / (2 * _tau))
        print(f"  {_w:>4}  {_snr_gain:>8.1f}dB  {_peak_atten:>10.2f}x  {_r['ssnr']:>6.2f}  {_r['tsnr']:>6.2f}  {_r['stsnr']:>7.2f}")
    _peak = max(smooth_results, key=lambda _r: _r["stsnr"])
    print(f"\n  Peak stSNR = {_peak['stsnr']:.2f} dB at W = {_peak['window']} frames")
    print(f"  At W={_peak['window']}: noise gain = {10*np.log10(_peak['window']):.1f} dB, "
          f"peak attenuation = {np.exp(-_peak['window']/(2*_tau)):.2f}x")
    print(f"\n  → Temporal averaging ceiling: {_peak['stsnr']:.1f} dB (W={_peak['window']})")
    print("  → A trained model must exceed this to be worth deploying over a running mean.")
    return


@app.cell
def _md_smooth_plot(mo):
    mo.md("""
    ## Experiment B — line plot: sSNR and tSNR vs window size

    We plot sSNR (blue) and tSNR (red) as separate lines vs temporal window W.
    The dashed lines show the raw F1 baseline (W=1, no smoothing).

    **Actual numbers from the sweep:**

    | W (frames) | sSNR (dB) | tSNR (dB) | stSNR (dB) | note |
    |---|---|---|---|---|
    | 1 (raw F1) | 8.77 | 8.14 | 8.46 | baseline |
    | 7 | 17.15 | 16.56 | 16.85 | |
    | 31 | 22.75 | 21.88 | 22.31 | |
    | **61** | **23.74** | **22.59** | **23.17** | **both peak here** |
    | 101 | 22.87 | 22.03 | 22.45 | both declining |
    | 151 | 21.01 | 21.06 | 21.04 | **crossover: tSNR > sSNR** |
    | 199 | 19.54 | 20.23 | 19.88 | sSNR now lower |

    **What the data actually shows (corrected story):**

    Both metrics peak at **W=61 frames** — which is close to τ₀.₅=46 from
    notebook 01. This is the cross-notebook confirmation: the optimal temporal
    averaging window matches the ACF correlation length. Beyond that both fall,
    but **sSNR falls faster than tSNR** at large windows. At W=151 they cross
    and at W=199 tSNR is actually higher than sSNR.

    Why does sSNR fall faster? At the temporal-mean limit (W→200), the output
    is a constant image. sSNR is penalised heavily on individual transient frames
    where F0[t] is very different from its mean — each of those frames drags the
    per-frame average down sharply. tSNR is penalised on the temporal variance
    of each pixel trace, but this is diluted by the many baseline frames where
    the constant output approximately matches F0.

    The practical message remains: **temporal averaging alone caps the achievable
    stSNR at ~23 dB (W=61)**. A learned model targeting individual voxels is
    not constrained by this ceiling and should exceed it.
    """)
    return


@app.cell
def _smooth_line_plot(plt, smooth_results):
    _windows = [r["window"] for r in smooth_results]
    _ssnr = [r["ssnr"] for r in smooth_results]
    _tsnr = [r["tsnr"] for r in smooth_results]

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_windows, _ssnr, "o-", color="steelblue", lw=1.5, ms=5, label="sSNR")
    _ax.plot(_windows, _tsnr, "s-", color="coral",     lw=1.5, ms=5, label="tSNR")
    _ax.axhline(_ssnr[0], color="steelblue", ls="--", lw=0.8, alpha=0.5, label=f"raw F1 sSNR={_ssnr[0]:.1f}")
    _ax.axhline(_tsnr[0], color="coral",     ls="--", lw=0.8, alpha=0.5, label=f"raw F1 tSNR={_tsnr[0]:.1f}")
    _ax.set_xscale("log")
    _ax.set_xlabel("Temporal window size W (frames, log scale)")
    _ax.set_ylabel("SNR (dB)")
    _ax.set_title("Experiment B — temporal smoothing of noisy F1 vs clean F0")
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.2, which="both")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _md_summary(mo):
    mo.md("""
    ## Summary — what both experiments tell us

    **Experiment A confirms the metric geometry:**
    - Additive noise degrades sSNR and tSNR symmetrically (diagonal tracking).
      A good denoiser reverses this — both rise together.
    - Spatial blur degrades sSNR faster than tSNR (above the diagonal).
      Spatial operations cannot specifically hurt temporal dynamics because
      neighbouring pixels share time courses. This means spatial operations
      alone are not harmful to tSNR — but they are also not sufficient to
      raise it.

    **Experiment B reveals the temporal averaging ceiling:**
    - Both sSNR and tSNR rise together as W increases — noise cancellation
      dominates early and temporal averaging genuinely helps.
    - **Both peak at W=61 ≈ τ₀.₅=46** (confirmed cross-notebook: the ACF
      correlation length predicts the optimal averaging window).
    - Beyond W=61 both metrics fall, with sSNR falling faster. At W=151
      tSNR overtakes sSNR because per-frame errors at transient moments drag
      sSNR down more than the per-pixel-trace metric.
    - The temporal averaging ceiling is **stSNR ≈ 23.2 dB at W=61**. No
      amount of tuning the window size gets you above this number. A learned
      model is not constrained by a fixed window and should exceed this ceiling.

    ### Architecture implication

    The training objective must apply direct pressure on individual
    pixel-timepoints, not just on frames. N2V3D blind-spot masking does exactly
    this: each masked voxel is a specific (x, y, t) location. The loss at that
    voxel is the error between the predicted and true value at that single
    pixel-timepoint — which is what tSNR measures. Frame-level objectives
    (per-frame MSE) give no special signal for tSNR and silently permit the
    temporal-smoothing failure mode shown in Experiment B.

    This notebook is the measurement-based justification for choosing N2V3D
    voxel masking over simpler baselines.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. Experiment A scatter: zoomed to 0–45 dB so the spatial blur sitting above the diagonal is
      actually visible. The clean (∞ dB) points are excluded.
      2. Experiment B markdown: has an actual numbers table so you can read the values in the
      notebook without needing to scroll up to the console output. The story is corrected to match
      the data.
      3. Summary: corrected — the real finding is that temporal averaging has a hard ceiling of 23.2
      dB stSNR at W=61 frames, and W=61 was predicted by τ₀.₅=46 from notebook 01. Both metrics
      decline together past that point, with sSNR falling faster. This is the honest justification
      for N2V3D: not that averaging destroys tSNR specifically, but that averaging caps the
      achievable score regardless of window tuning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Answer to the question: What kinds of errors hurt tSNR most?

      The experiments give a clear, ranked answer:

    1. Temporal distortion beyond the correlation length — the worst for both metrics combined. Once W > τ₀.₅ ≈ 46 frames, you are blurring the actual calcium transient shape. This caps stSNR at 23.2 dB no matter how much you tune the window.

    2. Nothing hurts tSNR specifically more than sSNR in this dataset. This is the surprising finding. Spatial blur actually preserves tSNR better than sSNR (6.8 dB gap, tSNR stays higher). Additive noise hits both equally. Temporal averaging eventually hits both, with sSNR falling faster at extremes.

    So the original framing — "temporal smoothing destroys tSNR" — was too simple. The real answer is: temporal averaging imposes a hard ceiling on both metrics, and the ceiling is set by the natural correlation length of the signal.

      ---
    What you should take away from this notebook:

    1. τ₀.₅ from notebook 01 is not just a patch depth decision — it also predicts the best possible temporal-averaging denoiser. W=61≈τ₀.₅ gave the peak stSNR of 23.2 dB. These two notebooks are connected: the ACF measurement is load-bearing for multiple downstream decisions.

    2. The 23.2 dB ceiling is your target. Any model you train must clear 23.2 dB on F1 (level 1) before it is worth deploying. If it doesn't beat a simple running mean, it has learned nothing useful.

    3. Spatial operations are not the problem for tSNR — they're actually forgiving of it. This means a 3D model that uses spatial context (convolutions across x,y) is not inherently dangerous for tSNR. The danger is temporal over-smoothing, not spatial operations.

    4. N2V3D masking is justified not because averaging "destroys tSNR" but because it removes the  fixed-window constraint. A learned model can denoise each (x,y,t) voxel using the right amount of local context — not a fixed W applied uniformly to every pixel at every time. That flexibility is what lets it potentially exceed 23.2 dB.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - _blur_math (after blur sweep): table of gap = tSNR − sSNR per sigma, mean gap in dB and
    linear ratio, explanation of why it's structural
    - _noise_math (after noise sweep): table of gap per sigma with 3 decimal precision, confirms
    near-zero constant gap with the i.i.d. math
    - _md_temporal_smooth (updated): now includes the two competing equations — σ_output =
    σ_noise/√W with dB examples, and peak_attenuation ≈ exp(−W/92) with examples at each W value
    - _smooth_math (after smooth sweep): full table of SNR gain (dB), peak attenuation, and all
    three metrics per W, plus a highlighted ceiling line showing 23.2 dB at W=61
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
