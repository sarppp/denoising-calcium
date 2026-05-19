"""Load data for masking geometry analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import load_stack


def load_validation_stacks(data_dir: Path = None) -> dict[str, np.ndarray]:
    """Load validation stacks."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"

    stacks = {}
    for name in ["F0", "F1", "F2", "F3"]:
        stacks[name] = np.asarray(load_stack(data_dir / "val" / f"{name}.tif"), dtype=np.float32)
    return stacks
