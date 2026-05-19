"""Visualization functions for patch sampling analysis."""

import numpy as np
import matplotlib.pyplot as plt


def plot_activity_histogram(activity: np.ndarray, threshold: float = None):
    """Plot histogram of pixel activity with threshold line."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(activity.flatten(), bins=100, log=True, edgecolor="k")
    ax.set_xlabel("Temporal Variance (per pixel)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Pixel Activity Distribution — Look for gap")

    if threshold is not None:
        ax.axvline(threshold, color="r", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.2f}")
        ax.legend()

    plt.tight_layout()
    plt.show()


def plot_sample_patches(
    patches: list,
    patch_max_variances: np.ndarray,
    active_mask: np.ndarray,
    threshold: float,
    n_viz: int = 10,
    save_path: str = None,
):
    """Visualize random patches with activity and temporal traces."""
    viz_indices = np.random.choice(len(patches), n_viz, replace=False)

    fig, axes = plt.subplots(n_viz, 3, figsize=(12, 3 * n_viz))

    for row, idx in enumerate(viz_indices):
        patch = patches[idx]
        max_var = patch_max_variances[idx]
        is_active = "ACTIVE" if active_mask[idx] else "background"

        # Center frame
        center_frame = patch[patch.shape[0] // 2, :, :]
        ax = axes[row, 0]
        im = ax.imshow(center_frame, cmap="magma")
        ax.set_title(f"Patch {idx}: {is_active} (max_var={max_var:.2f})")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

        # Find brightest pixel
        patch_activity = np.var(patch, axis=0)
        h_brightest, w_brightest = np.unravel_index(
            np.argmax(patch_activity), patch_activity.shape
        )

        # Temporal trace at brightest pixel
        temporal_trace = patch[:, h_brightest, w_brightest]
        ax = axes[row, 1]
        ax.plot(temporal_trace, linewidth=1)
        ax.set_title("Brightest pixel temporal trace")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)

        # Activity map in patch
        ax = axes[row, 2]
        im = ax.imshow(patch_activity, cmap="hot")
        ax.plot(w_brightest, h_brightest, "c+", markersize=10, markeredgewidth=2)
        ax.set_title("Activity map (variance)")
        ax.axis("off")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()

    print(f"✓ Visualized {n_viz} random patches")
