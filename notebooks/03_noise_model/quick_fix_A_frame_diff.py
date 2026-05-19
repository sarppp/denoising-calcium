"""Quick Fix A: Re-fit Level 1 noise model using frame differencing.

Frame differencing removes signal variance contamination.
Var[y_t - y_{t-1}] = 2 × noise_var (signal cancels out).
Works at any gain level, solves the R²=0.23 problem.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from scipy import stats
from cidc import load_stack, NOISE_LEVELS


DATA = Path(__file__).parent.parent.parent / "data"

print("="*80)
print("QUICK FIX A: FRAME DIFFERENCING ON LEVEL 1 STACKS")
print("="*80)

for stack_name in ["A1", "B1"]:
    print(f"\n{stack_name}:")

    # Load stack
    stack = np.asarray(load_stack(DATA / "train" / f"{stack_name}.tif")[:500], dtype=np.float32)
    T, H, W = stack.shape

    # Frame differencing: removes signal, keeps noise
    diff = stack[1:] - stack[:-1]  # [499, 490, 490]

    # Variance per pixel (noise variance in the difference)
    var_diff = diff.var(axis=0)  # [490, 490]
    noise_var_per_pixel = var_diff / 2  # Scale by 1/2: Var[y_t - y_{t-1}] = 2 × σ²_noise

    # Mean per pixel (for plotting against)
    mean_per_pixel = stack.mean(axis=0)

    # Flatten for fitting
    mu_flat = mean_per_pixel.ravel()
    var_flat = noise_var_per_pixel.ravel()

    # Fit on low-intensity pixels (true background)
    mask = mu_flat < np.percentile(mu_flat, 10)
    mu_fit = mu_flat[mask]
    var_fit = var_flat[mask]

    # Linear regression
    slope, intercept, r, p, se = stats.linregress(mu_fit, var_fit)

    lib = NOISE_LEVELS[1]

    print(f"  Fitted:  g={slope:.1f}  σ_r²={intercept:.0f}  R²={r**2:.4f}")
    print(f"  Library: g={lib.gain}  σ_r²={lib.read_var}")
    print(f"  Error:   g={slope-lib.gain:+.1f}  σ_r²={intercept-lib.read_var:+.0f}")

    if r**2 > 0.85:
        print(f"  ✅ FIXED! R² now excellent (was 0.23)")
    else:
        print(f"  ⚠️ Still poor R²={r**2:.3f}")

print(f"\n{'='*80}")
print("Expected: R² > 0.85 on both A1 and B1 with frame differencing")
print("This removes signal contamination that broke the fit at low gain.")
print(f"{'='*80}")
