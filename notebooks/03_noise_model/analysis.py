"""Analyze noise model fit results."""

import numpy as np


def print_fit_summary(fit_results: dict) -> None:
    """Print summary table of noise model fits."""
    print(f"\n{'='*100}")
    print("NOISE MODEL FIT RESULTS")
    print(f"{'='*100}\n")
    print("| Stack | Level | Fitted g | Fitted σ_r² | R²      | Library g | Library σ_r² | Error g | Error σ |")
    print("|-------|-------|----------|-------------|---------|-----------|--------------|---------|---------|")

    for name in ["A1", "B1", "C2", "D2"]:
        r = fit_results[name]
        err_g = r["fitted_g"] - r["lib_g"]
        err_sr = r["fitted_sr2"] - r["lib_sr2"]
        print(f"| {name:5s} | {r['level']:5d} | {r['fitted_g']:8.1f} | {r['fitted_sr2']:11.0f} | {r['r2']:7.4f} | "
              f"{r['lib_g']:9.1f} | {r['lib_sr2']:12.0f} | {err_g:+7.1f} | {err_sr:+7.0f} |")

    print()


def validate_anscombe_transform(fit_results: dict) -> None:
    """Check Anscombe stabilization (variance of transformed data should be ~1)."""
    print(f"\n{'='*100}")
    print("ANSCOMBE TRANSFORM VALIDATION")
    print(f"{'='*100}\n")

    for name in ["A1", "B1", "C2", "D2"]:
        r = fit_results[name]

        # Anscombe: y' = 2 × sqrt(y + 3/8)
        mu_fit = r["mu_fit"]
        var_fit = r["var_fit"]

        # Transform
        mu_transformed = 2 * np.sqrt(np.maximum(mu_fit, 0) + 3/8)
        var_transformed = 2 * np.sqrt(np.maximum(var_fit, 0) + 3/8)

        # Fitted line in transformed space
        g_fit = r["fitted_g"]
        sr2_fit = r["fitted_sr2"]
        pred_var_transformed = 2 * np.sqrt(np.maximum(g_fit * mu_fit + sr2_fit, 0) + 3/8)

        # Residual variance
        residuals = var_transformed - pred_var_transformed
        residual_var = np.var(residuals)

        print(f"{name}:")
        print(f"  Residual variance after Anscombe: {residual_var:.4f}")
        if residual_var < 1.2:
            print(f"  ✓ Stabilized (< 1.2) → Poisson-Gaussian model valid")
        else:
            print(f"  ⚠ Not fully stabilized → may need alternative variance function")
        print()
