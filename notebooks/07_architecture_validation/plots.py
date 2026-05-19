"""Plotting for architecture validation."""

import matplotlib.pyplot as plt
import numpy as np


def plot_validation_results(val_results: dict) -> None:
    """Plot validation metrics across noise levels."""
    stacks = ["F1", "F2", "F3"]
    ssnr_vals = [val_results[s]['ssnr'] for s in stacks if s in val_results]
    tsnr_vals = [val_results[s]['tsnr'] for s in stacks if s in val_results]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(stacks))
    width = 0.35
    
    ax.bar(x - width/2, ssnr_vals, width, label='sSNR', alpha=0.8)
    ax.bar(x + width/2, tsnr_vals, width, label='tSNR', alpha=0.8)
    
    ax.set_ylabel('SNR (dB)')
    ax.set_title('Architecture Validation: SNR across Noise Levels')
    ax.set_xticks(x)
    ax.set_xticklabels(stacks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    plt.show()
