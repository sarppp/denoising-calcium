"""Load all 4 F-stacks from data directory."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import load_stack


def load_all_stacks(data_dir: Path) -> dict[str, np.ndarray]:
    """Load F0, F1, F2, F3 as float32 arrays."""
    stacks = {}
    for name in ["F0", "F1", "F2", "F3"]:
        stacks[name] = np.asarray(load_stack(data_dir / f"{name}.tif"), dtype=np.float32)
        print(f"Loaded {name}: {stacks[name].shape}")
    return stacks
