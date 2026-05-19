"""Load data for architecture validation."""

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
    stacks["A1"] = np.asarray(load_stack(data_dir / "train" / "A1.tif"), dtype=np.float32)
    # Validation
    for name in ["F0", "F1", "F2", "F3"]:
        stacks[name] = np.asarray(load_stack(data_dir / "val" / f"{name}.tif"), dtype=np.float32)
    
    print(f"Loaded training A1: {stacks['A1'].shape}")
    print(f"Loaded validation F0-F3")
    return stacks
