"""Plotting for gain augmentation."""

import matplotlib.pyplot as plt


def plot_gain_robustness(results: list) -> None:
    """Plot stSNR vs gain factor."""
    gains = [r['gain_factor'] for r in results]
    stsn = [r['stsnr'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(gains, stsn, marker='o', linewidth=2, markersize=8)
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.5, label='Nominal gain')
    ax.set_xlabel('Gain factor (×)', fontsize=12)
    ax.set_ylabel('stSNR (dB)', fontsize=12)
    ax.set_title('Model Robustness to Gain Variation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
