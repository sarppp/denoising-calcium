"""04 — Compare noise levels. F0 (clean) vs F1, F2, F3 at the same pixel/frame."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _setup():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from cidc import load_stack

    DATA = Path("/app/workspace/data")
    return DATA, load_stack, np, plt


@app.cell
def _same_frame_four_levels(DATA, load_stack, np, plt):
    """Same frame (t=750) shown for F0, F1, F2, F3 side by side."""
    _names = ["F0.tif", "F1.tif", "F2.tif", "F3.tif"]
    _frames = [np.asarray(load_stack(DATA / "val" / _n)[750]) for _n in _names]
    _vmax = max(np.percentile(_f, 99) for _f in _frames)

    _fig, _axes = plt.subplots(1, 4, figsize=(14, 4))
    for _ax, _n, _f in zip(_axes, _names, _frames):
        _ax.imshow(_f, cmap="gray", vmin=0, vmax=_vmax)
        _ax.set_title(_n)
        _ax.axis("off")
    _fig.suptitle("Same frame (t=750), same scene, increasing noise")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _same_trace_four_levels(DATA, load_stack, np, plt):
    """Same pixel trace at F0 vs F1/F2/F3. Find the bright pixel from F0."""
    _F0 = load_stack(DATA / "val" / "F0.tif")
    _tmean = np.asarray(_F0[::10]).mean(axis=0)
    _y, _x = np.unravel_index(int(_tmean.argmax()), _tmean.shape)

    _fig, _axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    for _ax, _name in zip(_axes, ["F0.tif", "F1.tif", "F2.tif", "F3.tif"]):
        _s = load_stack(DATA / "val" / _name)
        _trace = np.asarray(_s[:, _y, _x], dtype=np.float32)
        _ax.plot(_trace, lw=0.6)
        _ax.set_ylabel(_name)
    _axes[-1].set_xlabel("frame")
    _fig.suptitle(f"Same pixel ({_y}, {_x}) at 4 noise levels")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _all_eight_frames(DATA, load_stack, np, plt):
    """All 8 stacks side by side at the same frame. Training stacks are
    DIFFERENT scenes from F0, so they don't need to match it — but A1/B1
    should LOOK as noisy as F1, and C2/D2 as noisy as F2."""
    _pairs = [
        ("train", "A1.tif"), ("train", "B1.tif"),
        ("train", "C2.tif"), ("train", "D2.tif"),
        ("val", "F0.tif"),  ("val", "F1.tif"),
        ("val", "F2.tif"),  ("val", "F3.tif"),
    ]
    _frames = [np.asarray(load_stack(DATA / _d / _n)[750]) for _d, _n in _pairs]
    _vmax = max(np.percentile(_f, 99) for _f in _frames)

    _fig, _axes = plt.subplots(2, 4, figsize=(14, 7))
    for _ax, (_d, _n), _f in zip(_axes.ravel(), _pairs, _frames):
        _ax.imshow(_f, cmap="gray", vmin=0, vmax=_vmax)
        _ax.set_title(f"{_d}/{_n}")
        _ax.axis("off")
    _fig.suptitle("Frame 750 across all 8 stacks  (top: train, bottom: val)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _residuals(DATA, load_stack, np, plt):
    """Fk - F0 histograms. Zero-mean => additive noise model confirmed."""
    _F0 = load_stack(DATA / "val" / "F0.tif")
    _t = np.linspace(0, _F0.shape[0] - 1, 60, dtype=int)
    _a = np.asarray(_F0[_t, :128, :128], dtype=np.float64)

    _fig, _axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for _ax, _name in zip(_axes, ["F1.tif", "F2.tif", "F3.tif"]):
        _b = np.asarray(load_stack(DATA / "val" / _name)[_t, :128, :128],
                        dtype=np.float64)
        _r = _b - _a
        _ax.hist(_r.ravel(), bins=120)
        _ax.set_title(f"{_name} - F0   μ={_r.mean():+.2f}  σ={_r.std():.1f}")
        _ax.set_xlabel("residual (ADU)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
