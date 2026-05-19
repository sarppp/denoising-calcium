"""03 — Temporal. Traces over time and autocorrelation."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _setup():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from cidc import load_stack, temporal_autocorr

    DATA = Path("/app/workspace/data")
    return DATA, load_stack, np, plt, temporal_autocorr


@app.cell
def _bright_trace(DATA, load_stack, np, plt):
    """Trace at the brightest pixel of F0.tif over all 1500 frames."""
    _s = load_stack(DATA / "val" / "F0.tif")
    _tmean = np.asarray(_s[::10]).mean(axis=0)
    _y, _x = np.unravel_index(int(_tmean.argmax()), _tmean.shape)
    _trace = np.asarray(_s[:, _y, _x], dtype=np.float32)

    _fig, _ax = plt.subplots(figsize=(11, 3))
    _ax.plot(_trace, lw=0.8)
    _ax.set_xlabel("frame")
    _ax.set_ylabel("intensity (ADU)")
    _ax.set_title(f"F0.tif  bright pixel ({_y}, {_x})")
    _fig
    return


@app.cell
def _bg_trace(DATA, load_stack, np, plt):
    """Trace at a corner background pixel (should be flat-ish on F0)."""
    _s = load_stack(DATA / "val" / "F0.tif")
    _trace = np.asarray(_s[:, 10, 10], dtype=np.float32)

    _fig, _ax = plt.subplots(figsize=(11, 3))
    _ax.plot(_trace, lw=0.8, color="C1")
    _ax.set_xlabel("frame")
    _ax.set_ylabel("intensity (ADU)")
    _ax.set_title("F0.tif  background pixel (10, 10)")
    _fig
    return


@app.cell
def _autocorrelation_val(DATA, load_stack, plt, temporal_autocorr):
    """Temporal ACF for validation stacks. F0 is clean -> high ACF.
    F1/F2/F3 are noisy -> ACF collapses to near 0 at lag 1."""
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _name in ["F0.tif", "F1.tif", "F2.tif", "F3.tif"]:
        _acf = temporal_autocorr(load_stack(DATA / "val" / _name), max_lag=60)
        _ax.plot(_acf, label=f"val/{_name}")
    _ax.axhline(0, c="k", lw=0.5)
    _ax.set_xlabel("lag (frames)")
    _ax.set_ylabel("autocorrelation")
    _ax.set_title("ACF — validation")
    _ax.legend()
    _fig
    return


@app.cell
def _autocorrelation_train(DATA, load_stack, plt, temporal_autocorr):
    """Temporal ACF for training stacks. All noisy -> all flat.
    That is NOT because there is no signal; noise dominates the variance."""
    _fig, _ax = plt.subplots(figsize=(8, 4))
    for _name in ["A1.tif", "B1.tif", "C2.tif", "D2.tif"]:
        _acf = temporal_autocorr(load_stack(DATA / "train" / _name), max_lag=60)
        _ax.plot(_acf, label=f"train/{_name}")
    _ax.axhline(0, c="k", lw=0.5)
    _ax.set_xlabel("lag (frames)")
    _ax.set_ylabel("autocorrelation")
    _ax.set_title("ACF — training")
    _ax.legend()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
