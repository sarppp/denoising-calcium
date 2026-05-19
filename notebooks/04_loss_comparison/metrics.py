"""Compute metrics for loss function comparison."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from scipy.optimize import minimize
from cidc import load_stack, stsnr


def compute_loss_on_stack(stack: np.ndarray, reference: np.ndarray, loss_type: str = "mse") -> float:
    """Compute loss between stack and reference."""
    residual = stack - reference
    
    if loss_type == "mse":
        return float(np.mean(residual ** 2))
    elif loss_type == "mae":
        return float(np.mean(np.abs(residual)))
    elif loss_type == "ssim":
        # Simplified SSIM
        mean_pred = np.mean(stack)
        mean_ref = np.mean(reference)
        var_pred = np.var(stack)
        var_ref = np.var(reference)
        cov = np.mean((stack - mean_pred) * (reference - mean_ref))
        ssim = (2 * mean_pred * mean_ref) * (2 * cov) / (
            (mean_pred**2 + mean_ref**2) * (var_pred + var_ref + 1e-6)
        )
        return float(1 - ssim)  # Convert to loss
    else:
        return float(np.mean(residual ** 2))


def evaluate_losses(stacks: dict, reference: np.ndarray, loss_types: list = None) -> dict:
    """Evaluate multiple loss functions on validation stacks."""
    if loss_types is None:
        loss_types = ["mse", "mae"]
    
    results = {}
    print(f"Evaluating loss functions:")
    for stack_name in ["F1", "F2", "F3"]:
        results[stack_name] = {}
        st = stacks[stack_name]
        res = stsnr(st, reference)
        for loss_type in loss_types:
            loss = compute_loss_on_stack(st, reference, loss_type)
            results[stack_name][loss_type] = loss
            print(f"  {stack_name} {loss_type:5s}: {loss:10.2f}  (stSNR={res.st_snr:.2f})")
    
    return results
