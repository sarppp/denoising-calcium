"""Quick Fix B: Re-calibrate validation stacks using mean-intensity background.

The original method (temporal variance) fails because:
- On clean F0: low temporal variance selects saturated/constant pixels, not background
- On noisy stacks: low variance selects underexposed pixels, not representative background

Fix: Use low MEAN INTENSITY (dark pixels = background).
Works on both clean and noisy data because darkness is physical, not noise-dependent.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from scipy import stats
from cidc import load_stack


DATA = Path(__file__).parent.parent.parent / "data"

print("="*80)
print("QUICK FIX B: VALIDATION CALIBRATION WITH MEAN-INTENSITY BACKGROUND")
print("="*80)

for stack_name in ["F0", "F1", "F2", "F3"]:
    print(f"\n{stack_name}:")

    # Load stack
    stack = np.asarray(load_stack(DATA / "val" / f"{stack_name}.tif"), dtype=np.float32)
    T, H, W = stack.shape

    # Per-pixel statistics
    mean_per_pixel = stack.mean(axis=0)  # [490, 490]
    var_per_pixel = stack.var(axis=0, ddof=1)  # [490, 490]

    # Flatten
    mu_flat = mean_per_pixel.ravel()
    var_flat = var_per_pixel.ravel()

    # FIXED: Background = low mean intensity (not low variance)
    bg_mask = mu_flat < np.percentile(mu_flat, 10)
    mu_fit = mu_flat[bg_mask]
    var_fit = var_flat[bg_mask]

    # Fit
    slope, intercept, r, p, se = stats.linregress(mu_fit, var_fit)

    print(f"  Fitted:  g={slope:.4f}  σ_r²={intercept:.1f}  R²={r**2:.4f}")
    print(f"  Background pixels selected: {bg_mask.sum()} / {bg_mask.size}")

    # Diagnostics
    if slope < 0:
        print(f"  ❌ NEGATIVE GAIN (unphysical) → background selection still wrong")
    elif slope > 0 and r**2 > 0.5:
        print(f"  ✅ Positive gain, R² > 0.5 → calibration now valid")
    elif slope > 0:
        print(f"  ⚠️ Positive gain but R² = {r**2:.3f} < 0.5 → marginal")

print(f"\n{'='*80}")
print("Expected for F0-F3:")
print("  F0: g > 0, R² > 0.5 (clean reference)")
print("  F1, F2, F3: g > 0, R² > 0.5 (all positive, all reasonable fits)")
print(f"{'='*80}")
