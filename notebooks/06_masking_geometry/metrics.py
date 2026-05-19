"""Compute metrics for masking geometry analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np


def compute_masked_statistics(stack: np.ndarray, mask_size: int = 1, center_only: bool = True) -> dict:
    """Compute statistics for masked patch-based prediction."""
    T, H, W = stack.shape
    
    center_h, center_w = H // 2, W // 2
    patch_half = mask_size // 2
    
    results = {
        "mask_size": mask_size,
        "center_intensity_mean": float(np.mean(stack[:, center_h-patch_half:center_h+patch_half+1, 
                                                         center_w-patch_half:center_w+patch_half+1])),
        "center_intensity_var": float(np.var(stack[:, center_h-patch_half:center_h+patch_half+1,
                                                       center_w-patch_half:center_w+patch_half+1])),
    }
    
    return results
