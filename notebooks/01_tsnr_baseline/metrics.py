"""Compute temporal ACF and baseline stSNR metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import temporal_autocorr, stsnr


def compute_acf_and_tau(f0: np.ndarray, max_lag: int = 100, max_pixels: int = 2000) -> tuple[np.ndarray, int]:
    """Compute ACF on F0 (clean reference) and find τ₀.₅ crossing."""
    acf = temporal_autocorr(f0, max_lag=max_lag, max_pixels=max_pixels)
    below = np.where(acf < 0.5)[0]
    tau_half = int(below[0]) if len(below) else None

    print(f"τ₀.₅ (ACF=0.5 crossing) = {tau_half} frames")
    print(f"Recommended patch depth T:")
    if tau_half:
        t_rec = 2 * tau_half
        if t_rec <= 40:
            t_val = 32
        elif t_rec <= 80:
            t_val = 64
        else:
            t_val = 128
        print(f"  τ₀.₅ = {tau_half} → 2×τ = {t_rec} → T = {t_val}")
    else:
        print(f"  Could not determine τ₀.₅")

    return acf, tau_half


def compute_baseline_stsnr(stacks: dict[str, np.ndarray], f0: np.ndarray) -> dict:
    """Compute baseline stSNR, sSNR, tSNR for raw noisy stacks."""
    print(f"\n{'='*80}")
    print("BASELINE stSNR — raw noisy input vs F0 (undenoised)")
    print(f"{'='*80}\n")

    results = {}
    for stack_name in ["F1", "F2", "F3"]:
        st = stacks[stack_name]
        res = stsnr(st, f0)
        results[stack_name] = {
            "st_snr": float(res.st_snr),
            "s_snr": float(res.s_snr),
            "t_snr": float(res.t_snr),
        }
        print(f"{stack_name}:")
        print(f"  stSNR = {res.st_snr:.3f} dB (blend of sSNR and tSNR)")
        print(f"  sSNR  = {res.s_snr:.3f} dB (spatial signal-to-noise)")
        print(f"  tSNR  = {res.t_snr:.3f} dB (temporal signal-to-noise)")
        print()

    print(f"{'='*80}")
    return results
