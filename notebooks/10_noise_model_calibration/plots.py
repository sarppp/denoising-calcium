"""Plotting functions for noise model calibration."""

import numpy as np
import matplotlib.pyplot as plt


def plot_calibration(calibration_results: dict, save_path: str = None) -> None:
    """Plot variance vs mean for all 4 stacks with fitted lines."""
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {"F0": "blue", "F1": "green", "F2": "orange", "F3": "red"}
    markers = {"F0": "o", "F1": "s", "F2": "^", "F3": "D"}

    for stack_name in ["F0", "F1", "F2", "F3"]:
        r = calibration_results[stack_name]

        # Scatter plot of bins
        ax.scatter(
            r["bin_means"],
            r["bin_variances"],
            label=f"{stack_name} (g={r['gain']:.4f}, σ={r['read_noise']:.4f}, R²={r['r_squared']:.3f})",
            color=colors[stack_name],
            marker=markers[stack_name],
            s=100,
            alpha=0.7,
            edgecolors='k',
            linewidth=0.5
        )

        # Fitted line
        x_fit = np.array([r["bin_means"].min(), r["bin_means"].max()])
        y_fit = r["gain"] * x_fit + (r["read_noise"] ** 2)
        ax.plot(
            x_fit,
            y_fit,
            color=colors[stack_name],
            linestyle="--",
            linewidth=2,
            alpha=0.7
        )

    ax.set_xlabel("Mean Intensity (background pixels)", fontsize=12)
    ax.set_ylabel("Variance", fontsize=12)
    ax.set_title("Noise Model Calibration: variance = g × mean + σ²", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"✓ Calibration plot saved to {save_path}")
    plt.show()
