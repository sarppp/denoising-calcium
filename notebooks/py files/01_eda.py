"""CIDC25 -- EDA (marimo).

Run with:
    uv run marimo edit workspace/notebooks/01_eda.py --mcp --no-token
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _intro():
    import marimo as mo
    mo.md(
        """
        # CIDC25 -- Exploratory Data Analysis

        Six questions to answer *before* picking a model:

        1. Dtype & intensity range per file.
        2. Noise model: is `Var(pixel)` affine in `Mean(pixel)`? (Poisson-Gaussian)
        3. How different are noise levels across train (`*1` vs `*2`) and val (`F1/F2/F3`)?
        4. Temporal autocorrelation -- do transients decay over many frames?
        5. Spatial stats -- single frame vs temporal mean baseline.
        6. `F0` (clean) vs `F1/F2/F3`: purely additive noise? Any scale/offset?
        """
    )
    return (mo,)


@app.cell
def _imports():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from cidc import (
        estimate_poisson_gaussian,
        load_stack,
        mean_var_per_pixel,
        stack_info,
        temporal_autocorr,
    )

    _here = Path(__file__).resolve() if "__file__" in globals() else Path.cwd()
    for _parent in [_here, *_here.parents]:
        if (_parent / "data").is_dir():
            DATA = _parent / "data"
            break
    else:
        DATA = Path("/app/workspace/data")

    TRAIN = sorted((DATA / "train").glob("*.tif"))
    VAL = sorted((DATA / "val").glob("*.tif"))
    print("data dir:", DATA)
    print("train:", [p.name for p in TRAIN])
    print("val  :", [p.name for p in VAL])
    return (
        DATA,
        TRAIN,
        VAL,
        estimate_poisson_gaussian,
        load_stack,
        mean_var_per_pixel,
        np,
        plt,
        stack_info,
        temporal_autocorr,
    )


@app.cell
def _basic_info(TRAIN, VAL, stack_info):
    _rows = []
    for _p in TRAIN + VAL:
        _info = stack_info(_p)
        _rows.append(
            f"{_p.name:8s}  shape={_info.shape}  dtype={_info.dtype}  "
            f"min={_info.min:.1f}  mean={_info.mean:.1f}  max={_info.max:.1f}"
        )
    print("\n".join(_rows))
    return


@app.cell
def _noise_fit(
    TRAIN,
    VAL,
    estimate_poisson_gaussian,
    load_stack,
    mean_var_per_pixel,
    np,
    plt,
):
    """Per-stack Poisson-Gaussian fit: Var = gain * Mean + read_var."""
    fits = {}
    noise_fig, _axes = plt.subplots(2, 4, figsize=(16, 7))
    for _p, _ax in zip(TRAIN + VAL, _axes.ravel()):
        _arr = load_stack(_p)
        _m, _v = mean_var_per_pixel(_arr, max_pixels=150_000)
        _bg = _m < np.median(_m)
        _fit = estimate_poisson_gaussian(_m[_bg], _v[_bg])
        fits[_p.name] = _fit
        _ax.scatter(_m, _v, s=1, alpha=0.15)
        _xs = np.linspace(_m.min(), _m.max(), 50)
        _ax.plot(
            _xs,
            _fit.gain * _xs + _fit.read_var,
            "r-",
            lw=1.5,
            label=f"g={_fit.gain:.3f}  r0={_fit.read_var:.1f}  R2={_fit.r2:.2f}",
        )
        _ax.set_title(_p.name)
        _ax.set_xlabel("mean")
        _ax.set_ylabel("var")
        _ax.legend(fontsize=8)
    plt.tight_layout()
    print("\nGain / read_var per file:")
    for _k, _f in fits.items():
        print(
            f"  {_k:8s}  gain={_f.gain:+.3f}  "
            f"read_var={_f.read_var:+.1f}  R2={_f.r2:.2f}"
        )
    return (fits,)


@app.cell
def _noise_level_bar(fits, plt):
    _names = list(fits)
    _gains = [fits[_n].gain for _n in _names]
    gain_fig, _ax = plt.subplots(figsize=(8, 3))
    _ax.bar(_names, _gains)
    _ax.set_ylabel("fitted gain")
    _ax.set_title("Gain per stack (proxy for noise level)")
    for _t in _ax.get_xticklabels():
        _t.set_rotation(30)
    plt.tight_layout()
    return


@app.cell
def _autocorr(TRAIN, VAL, load_stack, plt, temporal_autocorr):
    acf_fig, _ax = plt.subplots(figsize=(7, 4))
    for _p in TRAIN + VAL:
        _acf = temporal_autocorr(load_stack(_p), max_lag=60, max_pixels=1500)
        _ax.plot(_acf, label=_p.name)
    _ax.axhline(0, c="k", lw=0.5)
    _ax.set_xlabel("lag (frames)")
    _ax.set_ylabel("ACF")
    _ax.set_title("Temporal autocorrelation")
    _ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    return


@app.cell
def _spatial(TRAIN, VAL, load_stack, np, plt):
    _sample = VAL[0] if VAL else TRAIN[0]
    _arr = load_stack(_sample)
    _frame = np.asarray(_arr[_arr.shape[0] // 2])
    _tmean = np.asarray(_arr[::10]).mean(axis=0)
    spatial_fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].imshow(_frame, cmap="gray")
    _axes[0].set_title(f"{_sample.name}: single frame")
    _axes[0].axis("off")
    _axes[1].imshow(_tmean, cmap="gray")
    _axes[1].set_title(f"{_sample.name}: temporal mean")
    _axes[1].axis("off")
    plt.tight_layout()
    return


@app.cell
def _clean_vs_noisy(DATA, load_stack, mo, np, plt):
    """F0 (clean) vs F1/F2/F3 (noisy). Build one row per noise level, then
    hand marimo a single stacked figure as the cell's last expression."""
    _f0_path = DATA / "val" / "F0.tif"
    _figs = []
    if not _f0_path.exists():
        _out = mo.md("**F0.tif not downloaded yet — skipping.**")
    else:
        _F0 = load_stack(_f0_path)
        _t_idx = np.linspace(0, _F0.shape[0] - 1, 60, dtype=int)
        _a = np.asarray(_F0[_t_idx, :128, :128], dtype=np.float64)
        for _name in ["F1.tif", "F2.tif", "F3.tif"]:
            _p = DATA / "val" / _name
            if not _p.exists():
                continue
            _Fk = load_stack(_p)
            _b = np.asarray(_Fk[_t_idx, :128, :128], dtype=np.float64)
            _resid = _b - _a
            _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.2))
            _axes[0].scatter(_a.ravel()[::50], _b.ravel()[::50], s=1, alpha=0.2)
            _lo, _hi = float(_a.min()), float(_a.max())
            _axes[0].plot([_lo, _hi], [_lo, _hi], "r--", lw=1)
            _axes[0].set_title(f"F0 vs {_name}")
            _axes[0].set_xlabel("F0")
            _axes[0].set_ylabel(_name)
            _axes[1].hist(_resid.ravel(), bins=120)
            _axes[1].set_title(
                f"{_name} - F0  "
                f"(mean={_resid.mean():.2f}, std={_resid.std():.2f})"
            )
            _bins = np.linspace(_a.min(), _a.max(), 40)
            _which = np.digitize(_a.ravel(), _bins)
            _m_bin, _v_bin = [], []
            for _i in range(1, len(_bins)):
                _mask = _which == _i
                if _mask.sum() > 100:
                    _m_bin.append(_a.ravel()[_mask].mean())
                    _v_bin.append(_resid.ravel()[_mask].var())
            _axes[2].plot(_m_bin, _v_bin, "o-")
            _axes[2].set_xlabel("F0 intensity")
            _axes[2].set_ylabel(f"Var({_name} - F0)")
            _axes[2].set_title("Noise-vs-intensity (residual)")
            _fig.tight_layout()
            _figs.append(_fig)
        _out = mo.vstack(_figs) if _figs else mo.md("No F1/F2/F3 files found.")
    _out
    return


@app.cell
def _pick_stack(TRAIN, VAL, mo):
    """Pick a single stack to inspect. No mixing, one stack at a time."""
    _names = [_p.name for _p in TRAIN + VAL]
    stack_picker = mo.ui.dropdown(
        options=_names,
        value="F0.tif" if "F0.tif" in _names else _names[0],
        label="Stack to inspect",
    )
    stack_picker
    return (stack_picker,)


@app.cell
def _inspect_one_stack(DATA, load_stack, mo, np, plt, stack_picker):
    """Three *separate* diagnostic figures for the selected stack.

    Each figure answers exactly one question, so nothing overlaps.

    - Figure A (spatial)  -- 'What does one frame look like, and where are
                              the bright structures?' Shows a single frame
                              and the temporal mean image side by side.
    - Figure B (temporal) -- 'How does a bright vs background pixel
                              evolve over time?' Two separate axes, one for
                              the bright pixel, one for the background.
    - Figure C (noise)    -- 'What does the noise distribution look like
                              in a true-background region?' Log-y histogram
                              with fitted mean and std.
    """
    _name = stack_picker.value  # plain filename string now
    # Find it in either train/ or val/.
    _path = DATA / "train" / _name
    if not _path.exists():
        _path = DATA / "val" / _name
    _stack = load_stack(_path)
    _T, _H, _W = _stack.shape

    # Cheap temporal mean for locating bright pixels.
    _tmean = np.asarray(_stack[::10]).mean(axis=0)
    _y_bright, _x_bright = np.unravel_index(int(_tmean.argmax()), _tmean.shape)

    # ---- Figure A: spatial ----
    _midframe = np.asarray(_stack[_T // 2])
    _vmax_frame = float(np.percentile(_midframe, 99))
    _vmax_mean = float(np.percentile(_tmean, 99))

    _figA, _axA = plt.subplots(1, 2, figsize=(11, 4.5))
    _axA[0].imshow(_midframe, cmap="gray", vmin=0, vmax=_vmax_frame)
    _axA[0].set_title(f"frame t={_T // 2}  (99th-pct scaled)")
    _axA[0].axis("off")
    _axA[1].imshow(_tmean, cmap="gray", vmin=0, vmax=_vmax_mean)
    _axA[1].plot(_x_bright, _y_bright, "r+", ms=10, mew=2)
    _axA[1].set_title("temporal mean  (red + = brightest pixel)")
    _axA[1].axis("off")
    _figA.suptitle(f"A — Spatial view of {_path.name}", fontsize=11)
    _figA.tight_layout()

    # ---- Figure B: temporal ----
    _bright_trace = np.asarray(_stack[:, _y_bright, _x_bright], dtype=np.float32)
    _bg_trace = np.asarray(_stack[:, 10, 10], dtype=np.float32)  # corner bg

    _figB, _axB = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    _axB[0].plot(_bright_trace, lw=0.6, color="C0")
    _axB[0].set_ylabel("intensity (ADU)")
    _axB[0].set_title(
        f"bright pixel ({_y_bright}, {_x_bright})  "
        f"(μ={_bright_trace.mean():.1f}, σ={_bright_trace.std():.1f})"
    )
    _axB[1].plot(_bg_trace, lw=0.6, color="C1")
    _axB[1].set_ylabel("intensity (ADU)")
    _axB[1].set_xlabel("frame")
    _axB[1].set_title(
        f"background pixel (10, 10)  "
        f"(μ={_bg_trace.mean():.1f}, σ={_bg_trace.std():.1f})"
    )
    _figB.suptitle(f"B — Temporal traces from {_path.name}", fontsize=11)
    _figB.tight_layout()

    # ---- Figure C: noise distribution in background region ----
    # Take the top-left 20x20 patch sampled every 20 frames — this is almost
    # certainly pure background (see EDA: ~0.5% bright fraction).
    _bg_region = np.asarray(_stack[::20, :20, :20], dtype=np.float32).ravel()
    _mu, _sigma = float(_bg_region.mean()), float(_bg_region.std())

    _figC, _axC = plt.subplots(figsize=(11, 3.6))
    _axC.hist(_bg_region, bins=80, color="C2", alpha=0.8)
    _axC.axvline(_mu, color="k", lw=1, label=f"mean = {_mu:.1f}")
    _axC.axvline(_mu - _sigma, color="k", ls="--", lw=0.8,
                 label=f"±1σ  (σ = {_sigma:.1f})")
    _axC.axvline(_mu + _sigma, color="k", ls="--", lw=0.8)
    _axC.set_yscale("log")
    _axC.set_xlabel("intensity (ADU)")
    _axC.set_ylabel("count (log)")
    _axC.legend(fontsize=9)
    _axC.set_title(
        f"C — Background noise distribution from {_path.name} "
        f"(top-left 20×20 patch)"
    )
    _figC.tight_layout()

    mo.vstack([
        mo.md(f"### Inspecting `{_path.name}` — shape {_stack.shape}"),
        _figA, _figB, _figC,
    ])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
