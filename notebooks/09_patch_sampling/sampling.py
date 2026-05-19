"""Random patch sampling and classification."""

import numpy as np


def get_activity_threshold(activity: np.ndarray, percentile: float = 75) -> float:
    """Compute activity threshold by percentile."""
    threshold = np.percentile(activity, percentile)
    print(f"Threshold ({percentile}th percentile): {threshold:.2f}")
    print(f"Active pixels (> threshold): {(activity > threshold).sum()} / {activity.size}")
    print(f"Active pixel ratio: {(activity > threshold).mean() * 100:.2f}%")
    return threshold


def sample_random_patches(
    stack: np.ndarray,
    n_patches: int = 1000,
    patch_size: tuple = (64, 128, 128),
) -> tuple[list, np.ndarray, list]:
    """Sample n_patches random patches from stack."""
    T, H, W = stack.shape
    print(f"Patch size: {patch_size}")
    print(f"Sampling {n_patches} random patches...\n")

    patches = []
    patch_max_variances = []
    patch_centers = []

    for i in range(n_patches):
        # Random crop position
        t_start = np.random.randint(0, T - patch_size[0])
        h_start = np.random.randint(0, H - patch_size[1])
        w_start = np.random.randint(0, W - patch_size[2])

        # Extract patch
        patch = stack[
            t_start : t_start + patch_size[0],
            h_start : h_start + patch_size[1],
            w_start : w_start + patch_size[2],
        ]

        # Compute max temporal variance in this patch
        patch_activity = np.var(patch, axis=0)  # [H, W]
        max_var = float(np.max(patch_activity))

        patches.append(patch)
        patch_max_variances.append(max_var)
        patch_centers.append((t_start, h_start, w_start))

    patch_max_variances = np.array(patch_max_variances)

    print(f"Sampled {len(patches)} patches")
    print(f"Max variance per patch: [{patch_max_variances.min():.2f}, {patch_max_variances.max():.2f}]")

    return patches, patch_max_variances, patch_centers


def classify_patches(
    patch_max_variances: np.ndarray, threshold: float
) -> tuple[int, int, float]:
    """Classify patches as active or background."""
    active_mask = patch_max_variances > threshold
    n_active = active_mask.sum()
    n_background = (~active_mask).sum()
    active_ratio = (n_active / len(patch_max_variances)) * 100

    print(f"\n" + "=" * 80)
    print("PATCH CLASSIFICATION RESULTS")
    print("=" * 80)
    print(f"\nThreshold: {threshold:.2f}")
    print(f"Active patches (max_var > threshold): {n_active} / {len(patch_max_variances)}")
    print(f"Background patches (max_var ≤ threshold): {n_background} / {len(patch_max_variances)}")
    print(f"\nActive patch ratio: {active_ratio:.2f}%")

    return n_active, n_background, active_ratio
