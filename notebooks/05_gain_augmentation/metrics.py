"""Compute metrics for gain augmentation analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import stsnr


def simulate_gain_variation(stack: np.ndarray, reference: np.ndarray, gain_factors: list = None) -> list:
    """Simulate effect of gain variations on stSNR."""
    if gain_factors is None:
        gain_factors = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    results = []
    print("Gain variation effect on stSNR:")
    for g in gain_factors:
        scaled = stack * g
        res = stsnr(scaled, reference)
        results.append({
            "gain_factor": g,
            "ssnr": float(res.s_snr),
            "tsnr": float(res.t_snr),
            "stsnr": float(res.st_snr),
        })
        print(f"  g={g:.1f}x: stSNR={res.st_snr:.2f} dB")
    
    return results
