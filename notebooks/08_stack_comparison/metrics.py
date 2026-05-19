"""Compute per-stack metrics: variance, stSNR, gain, active pixels, temporal ACF."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import stsnr, mean_var_per_pixel, temporal_autocorr


def compute_results(stacks: dict[str, np.ndarray], f0_ref: np.ndarray) -> dict:
    """Compute all metrics for each stack."""
    results = {}

    for stack_name in ["F0", "F1", "F2", "F3"]:
        st = stacks[stack_name]

        # 1. Mean intensity
        mean_int = float(np.mean(st))

        # 2. Noise variance
        _, spatial_var = mean_var_per_pixel(st)
        mean_var = float(np.mean(spatial_var))

        # 3. Active pixels (top 25% by variance)
        sv2d = np.var(st, axis=0)
        threshold = np.percentile(sv2d, 75)
        active_ratio = float(np.mean(sv2d >= threshold) * 100)

        # 4. stSNR
        res = stsnr(st, f0_ref)
        stsn = float(res.st_snr)
        ssnr = float(res.s_snr)
        tsnr = float(res.t_snr)

        # 5. ACF tau(0.5)
        taus = temporal_autocorr(st)
        valid_taus = taus[~np.isnan(taus) & (taus > 0)]
        tau = float(np.median(valid_taus)) if len(valid_taus) > 0 else np.nan

        results[stack_name] = {
            "mean_int": mean_int,
            "mean_var": mean_var,
            "active_ratio": active_ratio,
            "stsn": stsn,
            "ssnr": ssnr,
            "tsnr": tsnr,
            "tau": tau,
        }

        print(f"\n{stack_name}:")
        print(f"  Mean intensity: {mean_int:.1f}")
        print(f"  Mean variance: {mean_var:.1f}")
        print(f"  Active pixels: {active_ratio:.2f}%")
        print(f"  stSNR: {stsn:.3f}")
        print(f"  τ(0.5): {tau:.1f} frames")

    return results


def compute_gains(stacks: dict[str, np.ndarray]) -> dict:
    """Compute linear gain (slope of variance vs intensity) per stack."""
    gains = {}

    for stack_name in ["F0", "F1", "F2", "F3"]:
        st = stacks[stack_name]
        spatial_mean, spatial_var = mean_var_per_pixel(st)

        mask = (spatial_mean > 0) & (spatial_var > 0)
        x = spatial_mean[mask]
        y = spatial_var[mask]

        if len(x) > 10:
            z = np.polyfit(x, y, 1)
            gain = float(z[0])
            gains[stack_name] = {
                "gain": gain,
                "offset": float(z[1]),
                "x": x,
                "y": y,
            }
            print(f"{stack_name}: gain = {gain:.4f}")

    return gains
