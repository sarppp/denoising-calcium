"""Plotting functions for noise model analysis."""

import numpy as np
import matplotlib.pyplot as plt


def plot_variance_vs_mean(fit_results: dict) -> None:
    """Plot variance vs mean for all training stacks with fitted lines."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = {"A1": "blue", "B1": "green", "C2": "orange", "D2": "red"}

    for idx, name in enumerate(["A1", "B1", "C2", "D2"]):
        ax = axes[idx]
        r = fit_results[name]

        # Scatter plot
        ax.scatter(r["mu"], r["var"], alpha=0.3, s=1, color=colors[name])

        # Fitted line
        mu_range = np.linspace(r["mu"].min(), r["mu"].max(), 100)
        var_fit = r["fitted_g"] * mu_range + r["fitted_sr2"]
        ax.plot(mu_range, var_fit, color=colors[name], linewidth=2,
                label=f"Fitted: g={r['fitted_g']:.1f}, σ_r²={r['fitted_sr2']:.0f}")

        # Library line
        var_lib = r["lib_g"] * mu_range + r["lib_sr2"]
        ax.plot(mu_range, var_lib, color=colors[name], linestyle="--", linewidth=1.5, alpha=0.7,
                label=f"Library: g={r['lib_g']:.1f}, σ_r²={r['lib_sr2']:.0f}")

        ax.set_xlabel("Mean intensity (ADU)")
        ax.set_ylabel("Variance")
        ax.set_title(f"{name} (Level {r['level']}, R²={r['r2']:.4f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Noise Model Fit: variance = g × mean + σ_r²", fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.show()


def plot_g_vs_level(fit_results: dict) -> None:
    """Plot fitted gain g vs noise level."""
    fig, ax = plt.subplots(figsize=(8, 5))

    levels = []
    g_fitted = []
    g_lib = []
    names = []

    for name in ["A1", "B1", "C2", "D2"]:
        r = fit_results[name]
        levels.append(r["level"])
        g_fitted.append(r["fitted_g"])
        g_lib.append(r["lib_g"])
        names.append(name)

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, g_fitted, width, label="Fitted", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, g_lib, width, label="Library", color="orange", alpha=0.8)

    ax.set_ylabel("Gain g (ADU per photon)")
    ax.set_title("Gain Estimation: Fitted vs Library Values")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    plt.show()
