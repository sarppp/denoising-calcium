"""02 — Noise. What does the noise distribution look like?"""

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
def _bg_histograms_val(DATA, load_stack, np, plt):
    """VAL background histograms (F0..F3). 20x20 corner patch = ~pure noise."""
    _files = ["F0.tif", "F1.tif", "F2.tif", "F3.tif"]
    _fig, _axes = plt.subplots(1, len(_files), figsize=(14, 3.2), sharey=True)
    for _ax, _name in zip(_axes, _files):
        _s = load_stack(DATA / "val" / _name)
        _bg = np.asarray(_s[::20, :20, :20], dtype=np.float32).ravel()
        _ax.hist(_bg, bins=80)
        _ax.set_yscale("log")
        _ax.set_title(f"val/{_name}\nμ={_bg.mean():.1f}  σ={_bg.std():.1f}")
        _ax.set_xlabel("ADU")
    _axes[0].set_ylabel("count (log)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _bg_histograms_train(DATA, load_stack, np, plt):
    """TRAIN background histograms (A1, B1, C2, D2). Compare widths to val —
    A1/B1 should match F1; C2/D2 should match F2."""
    _files = ["A1.tif", "B1.tif", "C2.tif", "D2.tif"]
    _fig, _axes = plt.subplots(1, len(_files), figsize=(14, 3.2), sharey=True)
    for _ax, _name in zip(_axes, _files):
        _s = load_stack(DATA / "train" / _name)
        _bg = np.asarray(_s[::20, :20, :20], dtype=np.float32).ravel()
        _ax.hist(_bg, bins=80)
        _ax.set_yscale("log")
        _ax.set_title(f"train/{_name}\nμ={_bg.mean():.1f}  σ={_bg.std():.1f}")
        _ax.set_xlabel("ADU")
    _axes[0].set_ylabel("count (log)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _var_vs_intensity_all(DATA, load_stack, np, plt):
    """Variance vs mean intensity for all 8 stacks.
    Poisson-Gaussian => slope = gain, intercept = read_var.
    F0 (clean) should be almost flat (slope ~0)."""
    _pairs = [
        ("train", "A1.tif"), ("train", "B1.tif"),
        ("train", "C2.tif"), ("train", "D2.tif"),
        ("val", "F0.tif"),  ("val", "F1.tif"),
        ("val", "F2.tif"),  ("val", "F3.tif"),
    ]

    _fig, _axes = plt.subplots(2, 4, figsize=(15, 7.5))
    _rng = np.random.default_rng(0)
    _idx = _rng.choice(490 * 490, size=50_000, replace=False)
    _y, _x = np.divmod(_idx, 490)

    for _ax, (_folder, _name) in zip(_axes.ravel(), _pairs):
        _s = load_stack(DATA / _folder / _name)
        _tr = np.asarray(_s[:, _y, _x], dtype=np.float64)
        _m = _tr.mean(axis=0)
        _v = _tr.var(axis=0)

        # Linear fit on background-dominated pixels (bottom 80% of intensity).
        _q = np.quantile(_m, 0.8)
        _mask = _m < _q
        _A = np.vstack([_m[_mask], np.ones(_mask.sum())]).T
        (_g, _c), *_ = np.linalg.lstsq(_A, _v[_mask], rcond=None)

        _ax.scatter(_m, _v, s=1, alpha=0.2)
        _xs = np.array([_m.min(), _m.max()])
        _ax.plot(_xs, _g * _xs + _c, "r-", lw=1)
        _ax.set_title(f"{_folder}/{_name}\ng={_g:.1f}  σ_r²={_c:.0f}")
        _ax.set_xlabel("mean")
        _ax.set_ylabel("variance")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
