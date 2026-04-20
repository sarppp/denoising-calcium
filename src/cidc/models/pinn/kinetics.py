"""Differentiable explicit-Euler integrator for the calcium-kinetics ODE.

The continuous ODE per pixel is::

    dC/dt = -C/τ + s(t),        F(t) = C(t) + b

Discretising with explicit Euler at timestep ``dt``::

    C[t+1] = C[t] * (1 - dt/τ) + dt * s[t]
    F[t]   = C[t] + b

For stability we require ``dt/τ <= 1`` (i.e. τ >= dt). With ``dt = 1``
frame and our prior ``τ ∈ [5, 200]``, this is comfortably satisfied.

We return the *observation* sequence ``F[0..T-1]`` matching the shape of
the denoised trace. The initial condition ``C[0] = 0`` so ``F[0] = b``.
The integrator is written as a Python loop over the time axis because
``T`` is small (≤ 64) and the loop is fully differentiable via autograd.
For larger ``T`` it would be worth rewriting with ``torch.jit.script`` or
exponentially-decaying convolution, but 32–64 frames is fine.
"""

from __future__ import annotations

import torch
from torch import Tensor


__all__ = ["euler_forward"]


def euler_forward(
    tau: Tensor,
    baseline: Tensor,
    source: Tensor,
    dt: float = 1.0,
) -> Tensor:
    """Integrate the calcium ODE forward with explicit Euler.

    Shapes
    ------
    All of ``tau``, ``baseline``, ``source`` have the same spatial prefix
    ``(B, 1, H, W)`` (per-pixel values). ``source`` additionally has a
    time axis at position 2: ``(B, 1, T, H, W)``. ``tau`` and ``baseline``
    are broadcast over time.

    Parameters
    ----------
    tau
        ``(B, 1, 1, H, W)`` — per-pixel decay constant in frames,
        strictly positive.
    baseline
        ``(B, 1, 1, H, W)`` — per-pixel baseline.
    source
        ``(B, 1, T, H, W)`` — per-pixel per-frame source term.
    dt
        Integration timestep in frames (default 1.0).

    Returns
    -------
    ``(B, 1, T, H, W)`` reconstructed trace ``F[t]``.
    """
    if tau.dim() != 5 or baseline.dim() != 5:
        raise ValueError(
            "tau and baseline must be (B, 1, 1, H, W); got "
            f"{tuple(tau.shape)} and {tuple(baseline.shape)}"
        )
    if source.dim() != 5:
        raise ValueError(f"source must be (B, 1, T, H, W); got {tuple(source.shape)}")

    T = source.shape[2]
    decay = (1.0 - dt / tau).clamp(min=0.0, max=1.0)    # (B, 1, 1, H, W)

    # C starts at zero; F[0] = baseline + C[0] = baseline.
    C = torch.zeros_like(source[:, :, 0])                # (B, 1, H, W)
    out = []
    for t in range(T):
        F_t = C + baseline.squeeze(2)                    # squeeze the singleton time axis
        out.append(F_t)
        C = C * decay.squeeze(2) + dt * source[:, :, t]
    return torch.stack(out, dim=2)                       # (B, 1, T, H, W)
