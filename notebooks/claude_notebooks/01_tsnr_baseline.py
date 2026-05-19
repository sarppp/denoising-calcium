"""01 — tSNR baseline on raw noisy data.

QUESTION: What does stSNR look like on raw (undenoised) data?
          What is the effective temporal window (τ₀.₅)?

Decision gate: sets the 3D patch depth T for training.
If τ₀.₅ > 40 frames → T=64; if 20-40 → T=32; if <20 → T=16.
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _setup():



    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo
    from cidc import load_stack, temporal_autocorr, stsnr

    DATA = Path(__file__).parent.parent.parent / "data"
    return DATA, load_stack, mo, np, plt, stsnr, temporal_autocorr


@app.cell
def _intro(mo):
    mo.md("""
    # 01 — tSNR baseline on raw noisy data

    **Purpose:** measure the two things that drive every architecture decision.

    1. **Baseline stSNR** — what score does the raw noisy input get against F0?
       This is the floor the model must exceed.
    2. **τ₀.₅** — the frame lag where the temporal ACF drops to 0.5.
       This is the effective temporal window; it sets the 3D patch depth T.

    Nothing here touches model weights. All cells are pure measurement.
    """)
    return


@app.cell
def _md_load(mo):
    mo.md("""
    ## Load val stacks

    F0 is the clean reference (never used for training).
    F1 / F2 / F3 are noisy at noise levels 1 / 2 / 3.
    Level 3 (F3) is the OOD stack for Task 2.
    """)
    return


@app.cell
def _load_stacks(DATA, load_stack):
    f0 = load_stack(DATA / "val" / "F0.tif")
    f1 = load_stack(DATA / "val" / "F1.tif")
    f2 = load_stack(DATA / "val" / "F2.tif")
    f3 = load_stack(DATA / "val" / "F3.tif")
    print(f"F0 shape={f0.shape}  F1 shape={f1.shape}")
    return f0, f1, f2, f3


@app.cell
def _md_acf(mo):
    mo.md("""
    ## Temporal ACF — finding τ₀.₅

    We compute the normalised temporal autocorrelation function (ACF) on 2 000
    random pixels from **F0** (the clean stack) up to lag 100 frames.

    ### Why F0, not F1?

    Poisson-Gaussian noise is temporally independent — each frame's noise is
    drawn fresh, with no memory of the previous frame. Mathematically this means
    noise adds a spike exactly at lag 0 and dilutes (shrinks) the ACF at every
    other lag. The result: τ₀.₅ computed on a noisy stack looks shorter than the
    true signal τ₀.₅. We would underestimate the useful temporal context and
    choose a patch depth T that is too small — the model would never see a full
    calcium transient in one patch.

    F0 is the clean ground-truth stack. Its ACF reflects genuine calcium
    transient dynamics without noise contaminating the decay. Using F0 here is
    measurement, not training — F0 never touches any gradient or weight update.
    The challenge rules prohibit using validation data "to train the algorithm";
    computing an ACF to set a hyperparameter is not training.

    ### What τ₀.₅ tells you

    τ₀.₅ is the first lag where ACF drops below 0.5 — the point where
    consecutive frames are more independent than correlated. A 3D patch of depth
    T ≈ 2×τ₀.₅ captures one full correlation length on each side of the target
    frame. Going deeper adds compute cost without adding useful signal. Going
    shallower means the model cannot see the rising and falling edges of a
    calcium transient in a single patch.
    """)
    return


@app.cell
def _acf(f0, np, temporal_autocorr):
    acf = temporal_autocorr(f0, max_lag=100, max_pixels=2000)
    _below = np.where(acf < 0.5)[0]
    tau_half = int(_below[0]) if len(_below) else None
    print(f"τ₀.₅ (ACF=0.5 crossing) = {tau_half} frames")
    return acf, tau_half


@app.cell
def _acf_plot(acf, np, plt, tau_half):
    _lags = np.arange(len(acf))
    _fig, _ax = plt.subplots(figsize=(7, 3))
    _ax.plot(_lags, acf, color="steelblue", lw=1.5)
    _ax.axhline(0.5, color="tomato", ls="--", lw=1, label="ACF = 0.5")
    if tau_half is not None:
        _ax.axvline(tau_half, color="tomato", ls=":", lw=1,
                    label=f"τ₀.₅ = {tau_half} frames")
    _ax.set_xlabel("Lag (frames)")
    _ax.set_ylabel("Normalised ACF")
    _ax.set_title("Temporal ACF — F0.tif (clean reference)")
    _ax.legend()
    _ax.set_ylim(-0.1, 1.05)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _md_baseline(mo):
    mo.md("""
    ## Baseline stSNR — raw noisy input vs F0

    We pass each noisy stack directly to `stsnr()` with F0 as the reference,
    **without any denoising**. This gives us the floor score — the number a
    model must beat just to be worth deploying.

    ### Why compare noisy to clean at all?

    The challenge scores your submission on stSNR(denoised, F0). To know
    whether a model is actually helping, you need to know what score the
    unmodified noisy input already achieves. If a model scores 5 dB and the
    raw noisy input scores 7 dB, the model made things worse.

    ### What negative stSNR means

    stSNR is computed as `10 * log10(signal_energy / residual_energy)`.
    When stSNR is negative, the residual (noise) energy is larger than the
    signal energy — the noisy image is further from F0 than the zero image
    would be. This is expected for F2 and F3: at high gain, noise dominates
    the pixel values. It means the model has a lot of room to gain points.

    ### Why tSNR is always lower than sSNR

    You will see tSNR consistently ~1–2 dB below sSNR across all noise levels.
    This is the core problem the 50/50 metric exposes. Spatial noise blurs out
    individual frames but neighbouring frames still look correlated. Temporal
    noise is independent frame-to-frame, so per-pixel traces look more erratic
    than per-frame images. A denoiser that only smooths spatially can recover
    sSNR while making tSNR worse — the metric punishes exactly that.

    Only the first 200 frames are used here to keep runtime under ~30 s.
    """)
    return


@app.cell
def _baseline_results(f0, f1, f2, f3, np, stsnr):
    _T = 200
    _ref = np.asarray(f0[:_T], dtype=np.float32)
    baseline_results = {}
    for _name, _stack in [("F1 (level 1)", f1), ("F2 (level 2)", f2), ("F3 (level 3, OOD)", f3)]:
        _pred = np.asarray(_stack[:_T], dtype=np.float32)
        baseline_results[_name] = stsnr(_pred, _ref)
    return (baseline_results,)


@app.cell
def _print_baseline(baseline_results):
    for _name, _r in baseline_results.items():
        print(f"{_name:24s}  sSNR={_r.s_snr:.2f} dB  tSNR={_r.t_snr:.2f} dB  stSNR={_r.st_snr:.2f} dB")
    return


@app.cell
def _snr_breakdown_plot(baseline_results, plt):
    _names = list(baseline_results.keys())
    _ssnr = [baseline_results[n].s_snr for n in _names]
    _tsnr = [baseline_results[n].t_snr for n in _names]
    _x = range(len(_names))

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _w = 0.35
    _ax.bar([i - _w/2 for i in _x], _ssnr, width=_w, label="sSNR", color="steelblue")
    _ax.bar([i + _w/2 for i in _x], _tsnr, width=_w, label="tSNR", color="coral")
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels(_names, rotation=10, ha="right")
    _ax.set_ylabel("SNR (dB)")
    _ax.set_title("Baseline stSNR components — raw noisy input vs F0")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _md_decision(mo):
    mo.md("""
    ## Decision gate — recommended patch depth T

    Based on τ₀.₅ above:

    | τ₀.₅ | Recommended T |
    |------|---------------|
    | > 40 frames | T = 64 |
    | 20–40 frames | T = 32 |
    | < 20 frames | T = 16 |
    """)
    return


@app.cell
def _tau_decision(tau_half):
    if tau_half is None:
        _rec = "T=64 (ACF stays high beyond lag 100)"
    elif tau_half > 40:
        _rec = "T=64"
    elif tau_half > 20:
        _rec = "T=32"
    else:
        _rec = "T=16"
    print(f"Recommended 3D patch depth: {_rec}  (τ₀.₅ = {tau_half} frames)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What notebook 01 showed

    Decision locked: T=64

    The ACF on F0 (clean signal) crosses 0.5 at lag 46 frames. That means calcium transients stay meaningfully correlated for ~46 frames. A 3D patch of T=64 covers the full rising and falling edge of a typical transient. T=32 would be too shallow — it only sees ~16 frames on each side of the target.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline stSNR — the floor to beat

    | Stack | sSNR | tSNR | stSNR |
    | :--- | :---: | :---: | :---: |
    | **F1** (level 1) | 7.73 dB | 6.20 dB | 6.97 dB |
    | **F2** (level 2) | −0.38 dB | −1.71 dB | −1.05 dB |
    | **F3** (level 3, OOD) | −6.25 dB | −7.48 dB | −6.86 dB |

    > **Note:** F2 and F3 are negative — noise energy exceeds signal energy before any denoising. The model has a lot of room to gain points on those.

    The **tSNR gap** is real and consistent: tSNR runs ~1.5 dB below sSNR at every noise level. This is not an artifact — it's the structural challenge. Temporal dynamics are harder to preserve than spatial structure even in the raw data. Any model that only smooths spatially will widen this gap.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
