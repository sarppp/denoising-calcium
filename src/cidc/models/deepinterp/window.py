"""DeepInterpolation input/target windowing.

A DeepInterpolation sample is a block of ``2K`` noisy frames symmetrically
around a center frame, with the center frame *removed* from the input and
used as the self-supervised target. Because the sensor noise is independent
across frames, the network cannot copy it and has to exploit the temporal
signal correlation, which acts as a strong implicit denoiser.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def make_di_window(
    stack: np.ndarray | Tensor,
    center: int,
    half_context: int,
) -> tuple[Tensor, Tensor]:
    """Return ``(context, target)`` for a single DeepInterpolation sample.

    Parameters
    ----------
    stack
        Noisy video, shape ``(T, H, W)``. Numpy or torch.
    center
        Index of the target frame. Must satisfy
        ``half_context <= center < T - half_context``.
    half_context
        ``K`` above. Temporal context is ``2K`` frames (K before, K after).

    Returns
    -------
    context : Tensor of shape (2K, H, W), float32.
    target  : Tensor of shape (1, H, W),  float32.
    """
    K = int(half_context)
    if isinstance(stack, np.ndarray):
        stack_t = torch.from_numpy(np.ascontiguousarray(stack))
    else:
        stack_t = stack
    if stack_t.ndim != 3:
        raise ValueError(f"stack must be (T,H,W), got {tuple(stack_t.shape)}")
    T = stack_t.shape[0]
    if not (K <= center < T - K):
        raise IndexError(
            f"center={center} out of valid range [{K}, {T - K}) for K={K}"
        )
    idx = list(range(center - K, center)) + list(range(center + 1, center + K + 1))
    context = stack_t[idx].to(torch.float32)
    target = stack_t[center : center + 1].to(torch.float32)
    return context, target
