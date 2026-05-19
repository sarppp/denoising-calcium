"""Compute noise model fitting for training stacks."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from scipy import stats
from cidc import load_stack, NOISE_LEVELS


def fit_noise_model(stack_path: Path, stack_name: str, level: int, n_frames: int = 500) -> dict:
    """Fit noise model (variance = g × mean + σ_r²) to a single stack."""
    stack = np.asarray(load_stack(stack_path)[:n_frames], dtype=np.float32)

    mu = stack.mean(axis=0)  # [H, W]
    var = stack.var(axis=0, ddof=1)  # [H, W]

    mu_flat = mu.ravel()
    var_flat = var.ravel()

    # Clip extremes: dark pixels (dominated by read noise), bright pixels (saturated)
    mask = (mu_flat > np.percentile(mu_flat, 5)) & (mu_flat < np.percentile(mu_flat, 99))
    mu_fit = mu_flat[mask]
    var_fit = var_flat[mask]

    # Fit: variance = g × mean + σ_r²
    slope, intercept, r, p, se = stats.linregress(mu_fit, var_fit)

    lib = NOISE_LEVELS[level]

    result = {
        "stack_name": stack_name,
        "level": level,
        "fitted_g": float(slope),
        "fitted_sr2": float(intercept),
        "r2": float(r ** 2),
        "lib_g": float(lib.gain),
        "lib_sr2": float(lib.read_var),
        "mu": mu_flat,
        "var": var_flat,
        "mu_fit": mu_fit,
        "var_fit": var_fit,
    }

    print(f"{stack_name}  fitted: g={slope:.1f}  σ_r²={intercept:.0f}  R²={r**2:.4f}  |  "
          f"library: g={lib.gain}  σ_r²={lib.read_var}")

    return result


def fit_all_training_stacks(data_dir: Path = None, n_frames: int = 500) -> dict:
    """Fit noise model to all training stacks."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"

    train_stacks = {
        "A1": (data_dir / "train" / "A1.tif", 1),
        "B1": (data_dir / "train" / "B1.tif", 1),
        "C2": (data_dir / "train" / "C2.tif", 2),
        "D2": (data_dir / "train" / "D2.tif", 2),
    }

    fit_results = {}
    for name, (path, level) in train_stacks.items():
        fit_results[name] = fit_noise_model(path, name, level, n_frames)

    return fit_results
