"""Plotting for masking geometry."""

import numpy as np
import matplotlib.pyplot as plt


def plot_patch_visualization(stack: np.ndarray, patch_size: int = 64) -> None:
    """Visualize a sample 3D patch from the stack."""
    t_mid = stack.shape[0] // 2
    h_mid = stack.shape[1] // 2
    w_mid = stack.shape[2] // 2
    
    half = patch_size // 2
    patch = stack[t_mid, h_mid-half:h_mid+half, w_mid-half:w_mid+half]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(patch, cmap='viridis')
    ax.set_title(f'Patch at center (T={t_mid}, H={h_mid}, W={w_mid})')
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
