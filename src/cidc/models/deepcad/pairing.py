"""Temporal Noise2Noise pairing for DeepCAD.

Given a window of ``2T`` frames, split into the odd and even sub-volumes
of ``T`` frames each. Either may be used as input with the other as
self-supervised target.
"""

from __future__ import annotations

import torch
from torch import Tensor


def temporal_halves(window: Tensor) -> tuple[Tensor, Tensor]:
    """Split a ``(B, C, 2T, H, W)`` stack into (odd_frames, even_frames).

    Parameters
    ----------
    window
        5D tensor; its temporal dimension must be even.

    Returns
    -------
    odd  : ``(B, C, T, H, W)`` — frames at indices 1, 3, 5, ...
    even : ``(B, C, T, H, W)`` — frames at indices 0, 2, 4, ...
    """
    if window.ndim != 5:
        raise ValueError(f"expected (B,C,2T,H,W), got {tuple(window.shape)}")
    if window.shape[2] % 2 != 0:
        raise ValueError(
            f"temporal length must be even for N2N pairing, got T={window.shape[2]}"
        )
    even = window[:, :, 0::2]
    odd = window[:, :, 1::2]
    return odd, even
