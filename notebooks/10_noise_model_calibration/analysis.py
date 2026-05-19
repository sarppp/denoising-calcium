"""Analyze calibration results and determine augmentation range."""

import numpy as np


def print_summary(calibration_results: dict) -> None:
    """Print calibration summary table."""
    print(f"\n{'='*100}")
    print("NOISE MODEL CALIBRATION SUMMARY")
    print(f"{'='*100}")
    print()
    print("| Stack | Gain g    | Read Noise σ | σ² (intercept) | R²    | Poisson? |")
    print("|-------|-----------|--------------|----------------|-------|----------|")

    for stack_name in ["F0", "F1", "F2", "F3"]:
        r = calibration_results[stack_name]
        g = r["gain"]
        sigma = r["read_noise"]
        sigma_sq = r["read_noise"] ** 2
        r_sq = r["r_squared"]
        poisson = "✓" if r_sq > 0.95 else ("~" if r_sq > 0.80 else "✗")
        print(f"| {stack_name}     | {g:9.6f} | {sigma:12.6f} | {sigma_sq:14.6f} | {r_sq:.3f} | {poisson}        |")

    print()


def get_augmentation_range(calibration_results: dict) -> dict:
    """Compute augmentation range with 3× safety margin."""
    gains = np.array([calibration_results[sn]["gain"] for sn in ["F0", "F1", "F2", "F3"]])
    g_min = gains.min()
    g_max = gains.max()
    g_ratio = g_max / g_min

    g_aug_min = g_min / 3
    g_aug_max = g_max * 3

    print(f"\n{'='*100}")
    print("AUGMENTATION RANGE")
    print(f"{'='*100}")

    print(f"\nGain range across stacks:")
    print(f"  Min gain (g_min): {g_min:.6f}")
    print(f"  Max gain (g_max): {g_max:.6f}")
    print(f"  Ratio: {g_ratio:.2f}×")

    print(f"\nAugmentation range (with 3× safety margin):")
    print(f"  g_aug_min = {g_min:.6f} / 3 = {g_aug_min:.6f}")
    print(f"  g_aug_max = {g_max:.6f} × 3 = {g_aug_max:.6f}")
    print(f"  Augmentation range: [{g_aug_min:.6f}, {g_aug_max:.6f}]")

    print(f"\nWhy 3× safety margin?")
    print(f"  - 1× covers observed data (F0-F3)")
    print(f"  - 3× covers potential Task 2 OOD gain")
    print(f"  - If Task 2 test gain outside [g_aug_min, g_aug_max]")
    print(f"    → you lose 5+ dB on Task 2")

    print(f"\n{'='*100}")

    return {
        "g_min": float(g_min),
        "g_max": float(g_max),
        "g_ratio": float(g_ratio),
        "g_aug_min": float(g_aug_min),
        "g_aug_max": float(g_aug_max),
    }


def print_nll_guidance(calibration_results: dict, aug_range: dict) -> None:
    """Print guidance for using results in NLL loss."""
    g_aug_min = aug_range["g_aug_min"]
    g_aug_max = aug_range["g_aug_max"]

    print(f"\n{'='*100}")
    print("USE IN NLL LOSS")
    print(f"{'='*100}")

    print(f"\nFor each stack, use fitted parameters in Poisson-Gaussian NLL:")
    print(f"\nNLL = -log p(y | x, g, σ)")
    print(f"    = -log N(y | x, σ²) - log Poisson(binom | g*x)")

    print(f"\nYour stack parameters:")
    for stack_name in ["F0", "F1", "F2", "F3"]:
        r = calibration_results[stack_name]
        print(f"  {stack_name}: g={r['gain']:.6f}, σ={r['read_noise']:.6f}")

    print(f"\nFor augmentation during training:")
    print(f"  Sample g ~ Uniform({g_aug_min:.6f}, {g_aug_max:.6f})")
    print(f"  Sample σ proportionally (or use range from calibration)")
    print(f"  This makes model robust to gain variation")

    print(f"\n{'='*100}")


def print_quality_checks(calibration_results: dict) -> None:
    """Print quality checks on fits."""
    print(f"\n{'='*100}")
    print("QUALITY CHECKS")
    print(f"{'='*100}")

    r_squared_values = np.array([calibration_results[sn]["r_squared"] for sn in ["F0", "F1", "F2", "F3"]])

    print(f"\nR² values (goodness of fit):")
    for stack_name in ["F0", "F1", "F2", "F3"]:
        r_sq = calibration_results[stack_name]["r_squared"]
        status = "✓" if r_sq > 0.95 else ("~" if r_sq > 0.80 else "✗")
        print(f"  {stack_name}: {r_sq:.4f} {status}")

    min_r2 = r_squared_values.min()
    if min_r2 > 0.95:
        print(f"\n✓ All R² > 0.95 → Poisson-Gaussian model valid")
        print(f"  Use standard Poisson-Gaussian NLL")
    elif min_r2 > 0.80:
        print(f"\n~ Some R² < 0.95 → Mixed noise model")
        print(f"  Consider more complex model (e.g., tuned Gaussian term)")
    else:
        print(f"\n✗ Poor fits → Model may not be accurate")
        print(f"  Investigate outliers or systematic issues")

    print(f"\n{'='*100}")
