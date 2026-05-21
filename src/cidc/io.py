"""I/O helpers for CIDC25 TIFF stacks.

Stacks are int16 ``[T, H, W]`` volumes of ~720 MB each. We never load a
whole stack into RAM; we use ``tifffile.memmap`` so that indexing returns
views we can ``np.asarray`` lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

__all__ = ["StackInfo", "load_stack", "stack_info", "iter_frames"]


@dataclass(frozen=True)
class StackInfo:
    """Basic summary of a stack (cheap to compute on a strided subset)."""

    path: Path
    shape: tuple[int, ...]
    dtype: np.dtype
    min: float
    max: float
    mean: float


def load_stack(path: str | Path) -> np.memmap:
    """Return a ``tifffile.memmap`` of the stack at ``path``.

    The returned object supports numpy-style slicing (``s[750]``,
    ``s[::10]``, ``s[:, y, x]``, ``s[::20, :20, :20]``). Do not call
    ``np.asarray`` on the whole memmap: slice first, convert after, to
    avoid loading 720 MB into RAM.
    """
    p = Path(path)
    return tifffile.memmap(str(p), mode="r")


def stack_info(path: str | Path, stride: int = 50) -> StackInfo:
    """Compute ``StackInfo`` using a strided subset to stay fast.

    We sample every ``stride``-th frame along the time axis, always
    including the last frame to avoid boundary bias.  For [1500, 490,
    490] stacks with stride=50 that is 31 frames ≈ 14 MB read.
    """
    p = Path(path)
    s = load_stack(p)
    indices = list(range(0, s.shape[0], stride))
    if indices[-1] != s.shape[0] - 1:
        indices.append(s.shape[0] - 1)
    sub = np.asarray(s[indices], dtype=np.float64)
    return StackInfo(
        path=p,
        shape=tuple(int(x) for x in s.shape),
        dtype=np.dtype(s.dtype),
        min=float(sub.min()),
        max=float(sub.max()),
        mean=float(sub.mean()),
    )


def iter_frames(stack: np.ndarray, start: int = 0, stop: int | None = None) -> Iterator[np.ndarray]:
    """Yield frames ``stack[t]`` as numpy arrays, one at a time."""
    T = stack.shape[0] if stop is None else int(stop)
    for t in range(int(start), T):
        frame = stack[t]
        yield frame if isinstance(frame, np.ndarray) else np.asarray(frame)
