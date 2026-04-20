"""Headless EDA: prints the numbers behind the plots in 01_eda.py.

The marimo MCP surface doesn't ship matplotlib images back, so this script
computes the decision-critical scalars and prints them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cidc import (
    estimate_poisson_gaussian,
    load_stack,
    mean_var_per_pixel,
    temporal_autocorr,
)

DATA = Path("/app/workspace/data")
TRAIN = sorted((DATA / "train").glob("*.tif"))
VAL = sorted((DATA / "val").glob("*.tif"))


def acf_summary(name: str, arr, max_lag: int = 60) -> None:
    acf = temporal_autocorr(arr, max_lag=max_lag, max_pixels=2000)
    # Time (frames) until ACF drops below these thresholds.
    def hit(t: float) -> int | str:
        below = np.where(acf < t)[0]
        return int(below[0]) if len(below) else ">60"

    print(
        f"  {name:8s}  acf[1]={acf[1]:+.3f}  acf[5]={acf[5]:+.3f}  "
        f"acf[10]={acf[10]:+.3f}  acf[30]={acf[30]:+.3f}  "
        f"τ(0.5)={hit(0.5)}  τ(0.2)={hit(0.2)}  τ(0.05)={hit(0.05)}"
    )


def spatial_stats(name: str, arr) -> None:
    # Temporal mean as a proxy for structure map.
    tmean = np.asarray(arr[::10], dtype=np.float32).mean(axis=0)
    # Sparsity: fraction of pixels above 2x the median (bright structures).
    med = np.median(tmean)
    mad = np.median(np.abs(tmean - med))
    thr = med + 5 * 1.4826 * mad  # 5-sigma-equivalent via MAD
    frac_bright = float(np.mean(tmean > thr))
    # Rough neuron size: average of connected-component sizes > threshold,
    # approximated by computing the characteristic length from the bright fraction
    # and the number of local maxima (very cheap heuristic).
    from scipy import ndimage

    mask = tmean > thr
    labelled, n = ndimage.label(mask)
    if n > 0:
        sizes = np.bincount(labelled.ravel())[1:]
        med_size = float(np.median(sizes))
        big_sizes = sizes[sizes > np.percentile(sizes, 75)]
        eq_radius = float(np.sqrt(big_sizes.mean() / np.pi)) if len(big_sizes) else 0.0
    else:
        med_size, eq_radius = 0.0, 0.0
    print(
        f"  {name:8s}  bright_frac={frac_bright:.3%}  n_blobs={n:4d}  "
        f"median_blob_size={med_size:5.1f}px  big_blob_radius≈{eq_radius:4.1f}px"
    )


def f0_fk_consistency() -> None:
    """For each F_k, fit gain from residual variance vs F0 intensity and
    compare to the gain fitted from (mean,var) of F_k itself."""
    f0 = load_stack(DATA / "val" / "F0.tif")
    # Small spatial subset, many frames.
    T_idx = np.linspace(0, f0.shape[0] - 1, 150, dtype=int)
    a = np.asarray(f0[T_idx, :200, :200], dtype=np.float64)
    for name in ["F1.tif", "F2.tif", "F3.tif"]:
        p = DATA / "val" / name
        fk = load_stack(p)
        b = np.asarray(fk[T_idx, :200, :200], dtype=np.float64)
        resid = b - a
        # Bin by F0 intensity; compute per-bin variance of residual.
        bins = np.linspace(a.min(), a.max(), 50)
        which = np.digitize(a.ravel(), bins)
        m_arr, v_arr = [], []
        for i in range(1, len(bins)):
            m = which == i
            if m.sum() > 200:
                m_arr.append(a.ravel()[m].mean())
                v_arr.append(resid.ravel()[m].var())
        m_arr = np.asarray(m_arr)
        v_arr = np.asarray(v_arr)
        fit = estimate_poisson_gaussian(m_arr, v_arr, trim=0.0)
        print(
            f"  {name}  residual-gain={fit.gain:+8.2f}  read_var={fit.read_var:+8.1f}  "
            f"R²={fit.r2:.3f}   mean(Fk-F0)={resid.mean():+.2f}   "
            f"std(Fk-F0)={resid.std():.2f}"
        )


def main() -> None:
    print("=" * 70)
    print("Temporal autocorrelation (how many frames do signals persist?)")
    print("=" * 70)
    for p in TRAIN + VAL:
        acf_summary(p.name, load_stack(p))

    print()
    print("=" * 70)
    print("Spatial structure (bright neuron fraction, rough blob size)")
    print("=" * 70)
    for p in TRAIN + VAL:
        spatial_stats(p.name, load_stack(p))

    print()
    print("=" * 70)
    print("F_k - F_0 residual analysis (validates Poisson-Gaussian forward model)")
    print("=" * 70)
    f0_fk_consistency()


if __name__ == "__main__":
    main()
