"""Visualization functions for stack comparison."""

import numpy as np
import matplotlib.pyplot as plt


def plot_stacks_side_by_side(stacks: dict[str, np.ndarray], frame_idx: int = 100, save_path: str = None):
    """Plot all 4 stacks at the same frame and colorscale."""
    f0_frame = stacks["F0"][frame_idx]
    vmin, vmax = float(f0_frame.min()), float(f0_frame.max())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, sn in zip(axes, ["F0", "F1", "F2", "F3"]):
        frame = stacks[sn][frame_idx]
        im = ax.imshow(frame, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"{sn} (frame {frame_idx})")
        ax.axis("off")

    fig.colorbar(im, ax=axes, label="Intensity")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()

    print("✓ All 4 stacks plotted at same frame and colorscale")


def plot_gain_slopes(gains: dict, save_path: str = None):
    """Plot variance vs mean intensity per stack with fitted gain lines."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for ax, sn in zip(axes, ["F0", "F1", "F2", "F3"]):
        if sn in gains:
            g = gains[sn]
            ax.scatter(g["x"], g["y"], alpha=0.3, s=10)

            x_fit = np.array([g["x"].min(), g["x"].max()])
            y_fit = g["gain"] * x_fit + g["offset"]
            ax.plot(x_fit, y_fit, "r-", lw=2, label=f"gain={g['gain']:.4f}")

            ax.set_xlabel("Mean Intensity (per-pixel)")
            ax.set_ylabel("Variance (per-pixel)")
            ax.set_title(f"{sn}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()

    print("✓ Gain slopes plotted separately per stack")
