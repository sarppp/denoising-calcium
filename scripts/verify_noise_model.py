"""Sanity-check the noise model on real CIDC25 data.

For each (noise-level, file) pair:

1. Load the measured noisy stack and estimate its variance map vs F0.
2. Simulate a noisy stack from F0 using our sampler at the same gain.
3. Compare their residual variance vs F0 — they should match within a few %.

Also verifies that `inverse_anscombe(anscombe(y)) ≈ y` to ~0.1 ADU on real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cidc import (
    FILE_NOISE,
    NOISE_LEVELS,
    NoiseParams,
    anscombe,
    inverse_anscombe,
    load_stack,
    sample_poisson_gaussian,
)

DATA = Path("/app/workspace/data")


def _binned_var_vs_F0(fk: np.ndarray, f0: np.ndarray, n_bins: int = 40):
    """Return (mean_F0_per_bin, var(Fk-F0)_per_bin) over a spatial subset."""
    t_idx = np.linspace(0, f0.shape[0] - 1, 150, dtype=int)
    a = np.asarray(f0[t_idx, :200, :200], dtype=np.float64)
    b = np.asarray(fk[t_idx, :200, :200], dtype=np.float64)
    resid = b - a
    bins = np.linspace(a.min(), a.max(), n_bins)
    which = np.digitize(a.ravel(), bins)
    means, vars_ = [], []
    for i in range(1, len(bins)):
        m = which == i
        if m.sum() > 200:
            means.append(a.ravel()[m].mean())
            vars_.append(resid.ravel()[m].var())
    return np.asarray(means), np.asarray(vars_)


def check_simulated_matches_real(f0: np.ndarray, name: str, params: NoiseParams):
    fk = load_stack(DATA / "val" / name)
    # Real variance vs intensity.
    m_real, v_real = _binned_var_vs_F0(fk, f0)
    # Simulated: run our sampler on F0 and compare residuals.
    rng = np.random.default_rng(0)
    # Work on the same sub-volume to keep it cheap.
    t_idx = np.linspace(0, f0.shape[0] - 1, 150, dtype=int)
    sub = np.asarray(f0[t_idx, :200, :200], dtype=np.float64)
    sim = sample_poisson_gaussian(sub, params, rng=rng)
    resid_sim = sim - sub
    bins = np.linspace(sub.min(), sub.max(), 40)
    which = np.digitize(sub.ravel(), bins)
    m_sim, v_sim = [], []
    for i in range(1, len(bins)):
        mask = which == i
        if mask.sum() > 200:
            m_sim.append(sub.ravel()[mask].mean())
            v_sim.append(resid_sim.ravel()[mask].var())
    m_sim = np.asarray(m_sim); v_sim = np.asarray(v_sim)

    # Fit line slope for both and compare.
    def slope(m, v):
        A = np.vstack([m, np.ones_like(m)]).T
        (g, c), *_ = np.linalg.lstsq(A, v, rcond=None)
        return float(g), float(c)

    g_real, c_real = slope(m_real, v_real)
    g_sim, c_sim = slope(m_sim, v_sim)

    ratio = g_sim / g_real if g_real else float("nan")
    print(
        f"  {name}  real gain={g_real:8.2f}  sim gain={g_sim:8.2f}  "
        f"ratio={ratio:.3f}  (target 1.0)    "
        f"real read_var={c_real:+8.0f}  sim read_var={c_sim:+8.0f}"
    )


def check_anscombe_roundtrip():
    """For each noise level, verify inverse_anscombe(anscombe(y)) ≈ y."""
    f0 = load_stack(DATA / "val" / "F0.tif")
    # Small sub-volume.
    sub = np.asarray(f0[::50, :128, :128], dtype=np.float64)
    for name, params in FILE_NOISE.items():
        if name == "F0.tif":
            continue
        rng = np.random.default_rng(42)
        y = sample_poisson_gaussian(sub, params, rng=rng)
        z = anscombe(y, params)
        y_back = inverse_anscombe(z, params)
        # After VST -> inverse on a noisy realisation, we expect mean(y_back - y) ~ 0
        # and std small but non-zero (the VST isn't invertible sample-by-sample
        # noiselessly; the inverse un-biases the mean).
        bias = float((y_back - y).mean())
        rmse = float(np.sqrt(np.mean((y_back - y) ** 2)))
        # More meaningful: does it un-bias E[y] given the clean signal?
        bias_vs_clean = float((y_back - sub).mean())
        print(
            f"  {name}  round-trip bias={bias:+7.3f}  rmse={rmse:7.2f}  "
            f"bias(y_back - clean)={bias_vs_clean:+7.3f}  "
            f"(ideal: ~0)"
        )


def check_anscombe_stabilises_variance():
    """After Anscombe, variance across pixels of the same clean intensity
    should be ~1, independent of intensity and noise level."""
    f0 = load_stack(DATA / "val" / "F0.tif")
    sub = np.asarray(f0[::50, :128, :128], dtype=np.float64)
    print(
        f"  {'level':>10s}  {'bin_mean':>10s}  {'var(z)':>10s}  "
        f"{'target':>8s}"
    )
    for lvl, params in NOISE_LEVELS.items():
        rng = np.random.default_rng(7)
        y = sample_poisson_gaussian(sub, params, rng=rng)
        z = anscombe(y, params)
        # Bin by clean intensity; report var(z) per bin.
        bins = np.linspace(sub.min(), sub.max(), 8)
        which = np.digitize(sub.ravel(), bins)
        for i in range(1, len(bins)):
            m = which == i
            if m.sum() > 1000:
                print(
                    f"  level {lvl}     {sub.ravel()[m].mean():10.1f}  "
                    f"{z.ravel()[m].var():10.3f}   {1.0:>8.1f}"
                )


def main():
    f0 = load_stack(DATA / "val" / "F0.tif")

    print("=" * 78)
    print("1. Does sample_poisson_gaussian(F0, params) match real F_k - F_0?")
    print("=" * 78)
    for name in ("F1.tif", "F2.tif", "F3.tif"):
        check_simulated_matches_real(f0, name, FILE_NOISE[name])

    print()
    print("=" * 78)
    print("2. Anscombe round-trip: inverse_anscombe(anscombe(y)) ≈ y")
    print("=" * 78)
    check_anscombe_roundtrip()

    print()
    print("=" * 78)
    print("3. Anscombe variance stabilisation: var(z) should be ≈ 1")
    print("=" * 78)
    check_anscombe_stabilises_variance()


if __name__ == "__main__":
    main()
