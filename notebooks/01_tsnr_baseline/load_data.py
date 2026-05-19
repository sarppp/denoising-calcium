"""Load validation stacks: F0 (clean), F1, F2, F3 (noisy at 3 levels)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import load_stack


def load_validation_stacks(data_dir: Path = None) -> dict[str, np.ndarray]:
    """Load F0 (clean reference) and F1, F2, F3 (noisy at levels 1, 2, 3)."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"

    f0 = np.asarray(load_stack(data_dir / "val" / "F0.tif"), dtype=np.float32)
    f1 = np.asarray(load_stack(data_dir / "val" / "F1.tif"), dtype=np.float32)
    f2 = np.asarray(load_stack(data_dir / "val" / "F2.tif"), dtype=np.float32)
    f3 = np.asarray(load_stack(data_dir / "val" / "F3.tif"), dtype=np.float32)

    print(f"F0 shape={f0.shape}  F1 shape={f1.shape}")
    print(f"F2 shape={f2.shape}  F3 shape={f3.shape}")

    return {"F0": f0, "F1": f1, "F2": f2, "F3": f3}
