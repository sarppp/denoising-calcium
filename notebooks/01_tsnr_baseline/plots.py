"""Plotting functions for tSNR baseline analysis."""

import numpy as np
import matplotlib.pyplot as plt


def plot_acf(acf: np.ndarray, tau_half: int = None) -> None:
    """Plot temporal ACF with τ₀.₅ marker."""
    lags = np.arange(len(acf))
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lags, acf, color="steelblue", lw=1.5)
    ax.axhline(0.5, color="tomato", ls="--", lw=1, label="ACF = 0.5")
    if tau_half is not None:
        ax.axvline(tau_half, color="tomato", ls=":", lw=1,
                    label=f"τ₀.₅ = {tau_half} frames")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("Normalised ACF")
    ax.set_title("Temporal ACF — F0.tif (clean reference)")
    ax.legend()
    ax.set_ylim(-0.1, 1.05)
    fig.tight_layout()
    plt.show()


def plot_baseline_stsnr(baseline_results: dict) -> None:
    """Plot baseline stSNR across noise levels."""
    stacks = list(baseline_results.keys())
    stsn_vals = [baseline_results[s]["st_snr"] for s in stacks]
    tsnr_vals = [baseline_results[s]["t_snr"] for s in stacks]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(stacks))
    width = 0.35

    ax.bar(x - width/2, stsn_vals, width, label="stSNR", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, tsnr_vals, width, label="tSNR", color="orange", alpha=0.8)

    ax.set_xlabel("Stack")
    ax.set_ylabel("SNR (dB)")
    ax.set_title("Baseline SNR — Raw Noisy Input vs F0")
    ax.set_xticks(x)
    ax.set_xticklabels(stacks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    plt.show()
