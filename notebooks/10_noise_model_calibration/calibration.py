"""Calibrate noise model: variance = g × mean + σ²."""

import numpy as np


def calibrate_stack(stack: np.ndarray, stack_name: str) -> dict:
    """Calibrate noise model for a single stack.

    Returns dict with: gain, read_noise, bin_means, bin_variances, r_squared
    """
    print(f"\n{'='*80}")
    print(f"Calibrating {stack_name}")
    print(f"{'='*80}")

    st = stack
    T, H, W = st.shape

    # Identify background pixels: low temporal variance
    temporal_var = np.var(st, axis=0)  # [H, W]
    bg_threshold = np.percentile(temporal_var, 10)  # Bottom 10% are background
    bg_mask = temporal_var < bg_threshold

    print(f"Background pixels: {bg_mask.sum()} / {bg_mask.size}")

    # Extract background data
    bg_data = st[:, bg_mask]  # [T, n_bg_pixels]

    # Divide intensity range into 20 bins
    n_bins = 20
    intensity_min = bg_data.min()
    intensity_max = bg_data.max()
    bin_edges = np.linspace(intensity_min, intensity_max, n_bins + 1)

    # For each bin, compute mean and variance
    bin_means = []
    bin_variances = []

    for i in range(n_bins):
        mask = (bg_data >= bin_edges[i]) & (bg_data < bin_edges[i+1])
        if mask.sum() > 0:
            bin_data = bg_data[mask]
            bin_means.append(np.mean(bin_data))
            bin_variances.append(np.var(bin_data))

    bin_means = np.array(bin_means)
    bin_variances = np.array(bin_variances)

    # Fit line: variance = g * mean + σ²
    z = np.polyfit(bin_means, bin_variances, 1)
    gain = float(z[0])
    read_noise_sq = float(z[1])
    read_noise = float(np.sqrt(max(0, read_noise_sq)))  # sqrt of intercept

    # R² goodness of fit
    y_pred = gain * bin_means + read_noise_sq
    ss_res = np.sum((bin_variances - y_pred) ** 2)
    ss_tot = np.sum((bin_variances - np.mean(bin_variances)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    result = {
        "gain": gain,
        "read_noise": read_noise,
        "bin_means": bin_means,
        "bin_variances": bin_variances,
        "r_squared": r_squared,
    }

    print(f"\nFit results:")
    print(f"  Gain g = {gain:.6f}")
    print(f"  Read noise σ = {read_noise:.6f}")
    print(f"  Intercept σ² = {read_noise_sq:.6f}")
    print(f"  R² = {r_squared:.4f}")

    # Interpretation
    if r_squared > 0.95:
        print(f"  ✓ Excellent fit → Poisson-dominated")
    elif r_squared > 0.80:
        print(f"  ~ Good fit → Mostly Poisson")
    else:
        print(f"  ⚠ Poor fit → Mixed noise model, reconsider NLL")

    return result


def calibrate_all_stacks(stacks: dict[str, np.ndarray]) -> dict:
    """Calibrate noise model for all stacks."""
    results = {}
    for stack_name in ["F0", "F1", "F2", "F3"]:
        results[stack_name] = calibrate_stack(stacks[stack_name], stack_name)
    return results
