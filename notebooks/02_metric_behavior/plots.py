"""Plotting functions for metric behavior analysis."""

import matplotlib.pyplot as plt


def plot_metric_geometry(blur_results: list, noise_results: list) -> None:
    """Plot sSNR vs tSNR for blur and noise perturbations."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Blur curve
    ssnr_blur = [r["ssnr"] for r in blur_results]
    tsnr_blur = [r["tsnr"] for r in blur_results]
    ax.plot(ssnr_blur, tsnr_blur, marker='o', label='Spatial blur', linewidth=2, markersize=6, color='steelblue')

    # Noise curve
    ssnr_noise = [r["ssnr"] for r in noise_results]
    tsnr_noise = [r["tsnr"] for r in noise_results]
    ax.plot(ssnr_noise, tsnr_noise, marker='s', label='Additive noise', linewidth=2, markersize=6, color='orange')

    # Diagonal
    min_val = min(min(ssnr_blur), min(ssnr_noise))
    max_val = max(max(ssnr_blur), max(ssnr_noise))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Identity (equal impact)')

    ax.set_xlabel('sSNR (dB)', fontsize=12)
    ax.set_ylabel('tSNR (dB)', fontsize=12)
    ax.set_title('Metric Geometry: How blur and noise move the (sSNR, tSNR) point', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_temporal_smoothing(smooth_results: list) -> None:
    """Plot sSNR and tSNR vs temporal smoothing window size."""
    fig, ax = plt.subplots(figsize=(10, 6))

    windows = [r["window"] for r in smooth_results]
    ssnr_vals = [r["ssnr"] for r in smooth_results]
    tsnr_vals = [r["tsnr"] for r in smooth_results]

    ax.plot(windows, ssnr_vals, marker='o', label='sSNR (spatial)', linewidth=2.5, markersize=7, color='steelblue')
    ax.plot(windows, tsnr_vals, marker='s', label='tSNR (temporal)', linewidth=2.5, markersize=7, color='orange')

    ax.set_xlabel('Temporal smoothing window (frames)', fontsize=12)
    ax.set_ylabel('SNR (dB)', fontsize=12)
    ax.set_title('Temporal Smoothing on Noisy Input (F1 → F0)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    plt.show()
