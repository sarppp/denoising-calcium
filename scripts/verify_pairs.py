"""Verify structural correspondence between validation files.

Challenge spec says the validation set has "noisy inputs and their
corresponding clean signal". Check that F0 is genuinely the same scene
as F1/F2/F3 (and different from the training samples).

Method: per-frame Pearson correlation of the raw pixel values across a
subset of frames. If F_k has F0 as its clean underlying signal, then for
each frame t, corr(F_k[t], F0[t]) should be substantial (scales with
inverse noise level). For an unrelated pair (say A1 vs F0), corr ≈ 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cidc import load_stack

DATA = Path("/app/workspace/data")


def frame_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 2D arrays, flattened."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def compare(base_name: str, other_name: str, n_frames: int = 30) -> None:
    base = load_stack(DATA / "val" / base_name) if "F" in base_name else load_stack(DATA / "train" / base_name)
    other = load_stack(DATA / "val" / other_name) if "F" in other_name else load_stack(DATA / "train" / other_name)
    t_idx = np.linspace(0, min(base.shape[0], other.shape[0]) - 1, n_frames, dtype=int)
    corrs = [frame_corr(np.asarray(base[t]), np.asarray(other[t])) for t in t_idx]
    mean_c = float(np.mean(corrs))
    min_c = float(np.min(corrs))
    max_c = float(np.max(corrs))
    print(
        f"  corr({base_name} , {other_name}): "
        f"mean={mean_c:+.4f}  min={min_c:+.4f}  max={max_c:+.4f}  "
        f"({n_frames} frames)"
    )


def main():
    print("=" * 72)
    print("Is F0 the clean signal of F1/F2/F3? (expect high correlation)")
    print("=" * 72)
    for name in ("F1.tif", "F2.tif", "F3.tif"):
        compare("F0.tif", name)

    print()
    print("=" * 72)
    print("Are F1/F2/F3 all the same scene? (expect high correlation)")
    print("=" * 72)
    compare("F1.tif", "F2.tif")
    compare("F1.tif", "F3.tif")
    compare("F2.tif", "F3.tif")

    print()
    print("=" * 72)
    print("Is F0 related to any TRAINING file? (expect ~0 correlation)")
    print("=" * 72)
    for name in ("A1.tif", "B1.tif", "C2.tif", "D2.tif"):
        compare("F0.tif", name)

    print()
    print("=" * 72)
    print("Are training files the same scene as each other? (expect ~0)")
    print("=" * 72)
    compare("A1.tif", "B1.tif")
    compare("A1.tif", "C2.tif")
    compare("B1.tif", "D2.tif")


if __name__ == "__main__":
    main()
