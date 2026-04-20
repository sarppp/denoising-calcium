"""Statistical helpers for CIDC25 stacks.

Three public helpers:

- ``mean_var_per_pixel``   — per-pixel temporal mean and variance.
- ``temporal_autocorr``    — sample-averaged temporal autocorrelation.
- ``estimate_poisson_gaussian`` — linear fit ``Var = gain * Mean + read_var``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "mean_var_per_pixel",
    "temporal_autocorr",
    "estimate_poisson_gaussian",
    "PGFit",
]


# ---------------------------------------------------------------------- #
# Per-pixel temporal mean/variance on a random subset of pixels.          #
# ---------------------------------------------------------------------- #


def mean_var_per_pixel(
    stack,
    n_pixels: int = 100_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n_pixels`` random spatial locations and compute (mean, var)
    over time at each of them.

    Works on tifffile memmaps without loading the full stack.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    T, H, W = stack.shape
    idx = rng.choice(H * W, size=int(n_pixels), replace=False)
    y, x = np.divmod(idx, W)
    # Fancy-index a memmap along the spatial axes -> (T, n_pixels) array.
    tr = np.asarray(stack[:, y, x], dtype=np.float64)
    means = tr.mean(axis=0)
    vars_ = tr.var(axis=0)
    return means, vars_


# ---------------------------------------------------------------------- #
# Temporal autocorrelation.                                               #
# ---------------------------------------------------------------------- #


def temporal_autocorr(
    stack,
    max_lag: int = 60,
    max_pixels: int = 2000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return the temporal ACF averaged over ``max_pixels`` random pixels.

    Each pixel's trace is mean-subtracted and scaled by its variance, so
    the returned array has ACF[0] == 1 and ACF[k] the normalised
    correlation at lag k frames, for k = 0 .. max_lag.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    T, H, W = stack.shape
    n = min(int(max_pixels), H * W)
    idx = rng.choice(H * W, size=n, replace=False)
    y, x = np.divmod(idx, W)
    tr = np.asarray(stack[:, y, x], dtype=np.float64)  # (T, n)
    tr -= tr.mean(axis=0, keepdims=True)
    var = (tr * tr).mean(axis=0) + 1e-12
    L = int(max_lag) + 1
    acf = np.zeros(L, dtype=np.float64)
    acf[0] = 1.0
    for k in range(1, L):
        # <x_t x_{t+k}> / <x^2>, averaged across pixels.
        num = (tr[:-k] * tr[k:]).mean(axis=0)
        acf[k] = float((num / var).mean())
    return acf


# ---------------------------------------------------------------------- #
# Poisson-Gaussian linear fit.                                            #
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class PGFit:
    """Result of fitting ``Var = gain * Mean + read_var``."""

    gain: float
    read_var: float
    r2: float


def estimate_poisson_gaussian(
    means: np.ndarray,
    variances: np.ndarray,
    trim: float = 0.01,
) -> PGFit:
    """Least-squares fit of Var vs Mean across pixels / bins.

    Parameters
    ----------
    means, variances
        1-D arrays of matched per-pixel (or per-bin) means and variances.
    trim
        Symmetric quantile trim applied to the mean axis before fitting,
        to drop outlier saturated / blank pixels. Set to 0 to disable.
    """
    m = np.asarray(means, dtype=np.float64).ravel()
    v = np.asarray(variances, dtype=np.float64).ravel()
    if trim > 0:
        lo, hi = np.quantile(m, [trim, 1.0 - trim])
        keep = (m >= lo) & (m <= hi)
        m, v = m[keep], v[keep]

    A = np.vstack([m, np.ones_like(m)]).T
    (gain, read_var), *_ = np.linalg.lstsq(A, v, rcond=None)
    v_hat = gain * m + read_var
    ss_res = float(((v - v_hat) ** 2).sum())
    ss_tot = float(((v - v.mean()) ** 2).sum()) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return PGFit(gain=float(gain), read_var=float(read_var), r2=float(r2))
