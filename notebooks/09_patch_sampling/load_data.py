"""Load F1 stack and compute activity map."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import load_stack


def load_f1_stack(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load F1 stack and return it with activity map (temporal variance per pixel)."""
    F1 = np.asarray(load_stack(data_dir / "F1.tif"), dtype=np.float32)
    print(f"F1 shape: {F1.shape}")

    # Compute activity map: temporal variance per pixel
    activity = np.var(F1, axis=0)  # [H, W]
    print(f"Activity map shape: {activity.shape}")
    print(f"Activity range: [{activity.min():.2f}, {activity.max():.2f}]")

    return F1, activity
