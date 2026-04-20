"""Blind-spot masking for Noise2Void 3D.

Strategy (Krull et al. 2019, extended to 3D)
--------------------------------------------
For each training volume:
1. Select a small fraction (~0.5 %) of voxels to mask.
2. Replace each masked voxel with a random neighbour from a small 3D
   "donut" (ring) around it, excluding the center itself. This preserves
   local statistics while removing the value the network is trying to
   reconstruct.
3. Train the network to predict the *original* voxel value at the masked
   locations only. This is the self-supervised objective.

The method relies on the assumption that noise is statistically
independent across voxels while the signal is correlated — exactly the
Poisson-Gaussian calcium-imaging setting.
"""

from __future__ import annotations

import torch
from torch import Tensor


def stratified_blindspot(
    volume: Tensor,
    mask_fraction: float = 0.005,
    donut_radius: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply blind-spot masking to a 3D volume.

    Parameters
    ----------
    volume
        ``(B, C, T, H, W)`` tensor of noisy input (any space; Anscombe or
        raw).  Must be ``float``.
    mask_fraction
        Fraction of spatiotemporal voxels to mask.  0.005 (0.5 %) matches
        N2V defaults and keeps per-batch compute reasonable.
    donut_radius
        Max Chebyshev distance for the replacement neighbour.  2 gives a
        5x5x5 ring minus the center voxel.
    generator
        Optional torch RNG for reproducibility.

    Returns
    -------
    masked : Tensor
        Same shape as ``volume``.  Mask locations replaced with a random
        donut neighbour.
    mask : Tensor (bool)
        ``(B, 1, T, H, W)`` indicating which voxels were masked (True) —
        use to select loss contributions.
    """
    if volume.ndim != 5:
        raise ValueError(f"expected (B,C,T,H,W), got {tuple(volume.shape)}")
    B, C, T, H, W = volume.shape
    device = volume.device

    n_vox = T * H * W
    n_mask = max(1, int(round(mask_fraction * n_vox)))

    # Flat random indices per batch item, then unravel into (t,h,w).
    flat = torch.empty((B, n_mask), dtype=torch.long, device=device)
    for b in range(B):
        flat[b] = torch.randperm(n_vox, generator=generator, device=device)[:n_mask]
    tt = flat // (H * W)
    hh = (flat % (H * W)) // W
    ww = flat % W

    # Random donut offsets (dt, dh, dw) in [-r, r]^3 excluding (0,0,0).
    r = int(donut_radius)
    side = 2 * r + 1
    off = torch.randint(0, side * side * side - 1, (B, n_mask), device=device, generator=generator)
    # Skip the center (index r*(side*side) + r*side + r) by shifting.
    center_flat = r * side * side + r * side + r
    off = off + (off >= center_flat).long()
    dt = off // (side * side) - r
    dh = (off % (side * side)) // side - r
    dw = off % side - r

    src_t = (tt + dt).clamp(0, T - 1)
    src_h = (hh + dh).clamp(0, H - 1)
    src_w = (ww + dw).clamp(0, W - 1)

    masked = volume.clone()
    mask = torch.zeros((B, 1, T, H, W), dtype=torch.bool, device=device)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, n_mask)
    for c in range(C):
        masked[b_idx, c, tt, hh, ww] = volume[b_idx, c, src_t, src_h, src_w]
    mask[b_idx, 0, tt, hh, ww] = True
    return masked, mask
