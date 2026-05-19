"""Load data for gain augmentation analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import load_stack


def load_all_data(data_dir: Path = None) -> dict:
    """Load training and validation stacks."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "data"

    stacks = {}
    # Training
    for name in ["A1", "B1", "C2", "D2"]:
        stacks[name] = np.asarray(load_stack(data_dir / "train" / f"{name}.tif"), dtype=np.float32)
    # Validation
    for name in ["F0", "F1", "F2", "F3"]:
        stacks[name] = np.asarray(load_stack(data_dir / "val" / f"{name}.tif"), dtype=np.float32)
    
    print(f"Loaded {len(stacks)} stacks")
    return stacks
