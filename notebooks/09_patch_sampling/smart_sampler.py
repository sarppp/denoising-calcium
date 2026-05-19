"""Smart patch sampler that biases toward active regions."""

import numpy as np


def smart_patch_sampler(
    stack: np.ndarray,
    n_patches: int,
    activity_threshold: float,
    active_bias: float = 0.8,
    patch_size: tuple = (64, 128, 128),
) -> tuple[list, list]:
    """
    Sample patches with bias toward active regions.

    Parameters
    ----------
    stack : ndarray [T, H, W]
        Input volume
    n_patches : int
        Number of patches to sample
    activity_threshold : float
        Threshold to define "active" pixels
    active_bias : float in [0, 1]
        Fraction of patches from active regions (default 0.8 = 80%)
    patch_size : tuple
        (time, height, width) of patches

    Returns
    -------
    patches : list of ndarray
    centers : list of (t, h, w) tuples
    """
    T, H, W = stack.shape

    # Compute activity map
    activity = np.var(stack, axis=0)  # [H, W]

    # Get coordinates of active and background pixels
    active_coords = np.argwhere(activity > activity_threshold)
    background_coords = np.argwhere(activity <= activity_threshold)

    print(f"Active coordinates: {len(active_coords)}")
    print(f"Background coordinates: {len(background_coords)}")

    patches = []
    centers = []

    for i in range(n_patches):
        # Decide: active or background patch?
        if np.random.random() < active_bias and len(active_coords) > 0:
            # Sample from active region
            h_center, w_center = active_coords[np.random.randint(len(active_coords))]
        elif len(background_coords) > 0:
            # Sample from background region
            h_center, w_center = background_coords[np.random.randint(len(background_coords))]
        else:
            # Fallback to random
            h_center = np.random.randint(0, H)
            w_center = np.random.randint(0, W)

        # Extract patch around this center
        h_start = max(0, min(h_center - patch_size[1] // 2, H - patch_size[1]))
        w_start = max(0, min(w_center - patch_size[2] // 2, W - patch_size[2]))
        t_start = np.random.randint(0, T - patch_size[0])

        patch = stack[
            t_start : t_start + patch_size[0],
            h_start : h_start + patch_size[1],
            w_start : w_start + patch_size[2],
        ]

        patches.append(patch)
        centers.append((t_start, h_start, w_start))

    return patches, centers


def evaluate_sampler(
    patches: list, threshold: float
) -> tuple[int, float]:
    """Evaluate active patch ratio for a set of patches."""
    max_vars = np.array([np.max(np.var(p, axis=0)) for p in patches])
    n_active = (max_vars > threshold).sum()
    active_ratio = (max_vars > threshold).mean() * 100
    return n_active, active_ratio
