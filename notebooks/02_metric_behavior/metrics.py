"""Compute metric behavior under different degradations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d
from cidc import stsnr


def spatial_blur_sweep(clean: np.ndarray, sigmas: list = None) -> list:
    """Sweep over spatial blur sigma values and measure stSNR."""
    if sigmas is None:
        sigmas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    blur_results = []
    print("Spatial blur sweep:")
    for s in sigmas:
        if s == 0.0:
            blurred = clean.copy()
        else:
            blurred = np.stack(
                [gaussian_filter(clean[t], sigma=s) for t in range(clean.shape[0])],
                axis=0,
            )
        r = stsnr(blurred, clean)
        blur_results.append({
            "sigma": s,
            "ssnr": float(r.s_snr),
            "tsnr": float(r.t_snr),
            "stsnr": float(r.st_snr)
        })
        print(f"  sigma={s:.1f}  sSNR={r.s_snr:.2f}  tSNR={r.t_snr:.2f}  stSNR={r.st_snr:.2f}")
    return blur_results


def additive_noise_sweep(clean: np.ndarray, sigmas: list = None) -> list:
    """Sweep over additive Gaussian noise sigma and measure stSNR."""
    if sigmas is None:
        sigmas = [0, 10, 20, 40, 80, 150, 200]

    noise_results = []
    print("\nAdditive noise sweep:")
    for s in sigmas:
        if s == 0:
            noisy = clean.copy()
        else:
            noisy = clean + np.random.normal(0, s, clean.shape).astype(np.float32)
        r = stsnr(noisy, clean)
        noise_results.append({
            "sigma": s,
            "ssnr": float(r.s_snr),
            "tsnr": float(r.t_snr),
            "stsnr": float(r.st_snr)
        })
        print(f"  sigma={s:3.0f}  sSNR={r.s_snr:.2f}  tSNR={r.t_snr:.2f}  stSNR={r.st_snr:.2f}")
    return noise_results


def temporal_smooth_sweep(noisy: np.ndarray, clean_ref: np.ndarray, windows: list = None) -> list:
    """Sweep over temporal smoothing window sizes on noisy data."""
    if windows is None:
        windows = [1, 3, 5, 7, 11, 17, 31, 63, 101]

    smooth_results = []
    print("\nTemporal smoothing sweep (noisy input → F0 reference):")
    for w in windows:
        if w == 1:
            smoothed = noisy.copy()
        else:
            smoothed = np.stack(
                [uniform_filter1d(noisy[:, h, ww], size=w, mode='nearest')
                 for h in range(noisy.shape[1]) for ww in range(noisy.shape[2])],
                axis=1
            ).reshape(noisy.shape)
        r = stsnr(smoothed, clean_ref)
        smooth_results.append({
            "window": w,
            "ssnr": float(r.s_snr),
            "tsnr": float(r.t_snr),
            "stsnr": float(r.st_snr)
        })
        print(f"  window={w:3d}  sSNR={r.s_snr:.2f}  tSNR={r.t_snr:.2f}  stSNR={r.st_snr:.2f}")
    return smooth_results
