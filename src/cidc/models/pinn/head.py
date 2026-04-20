"""Per-pixel PINN head: predicts τ, baseline, and source from features.

The head takes a feature map at the output resolution of the backbone
``(B, C_feat, T, H, W)`` and produces three tensors:

- ``tau``      — ``(B, 1, 1, H, W)``  per-pixel decay constant, clamped
                 to ``tau_range`` via a sigmoid reparameterisation.
- ``baseline`` — ``(B, 1, 1, H, W)``  per-pixel DC offset. If
                 ``baseline_from == 'median'`` the head ignores the
                 learned value and returns the median of the denoised
                 trace over T (set externally via ``set_baseline_prior``).
- ``source``   — ``(B, 1, T, H, W)``  per-pixel per-frame non-negative
                 source term, via softplus.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn


__all__ = ["PINNHead"]


class PINNHead(nn.Module):
    """Predicts (τ, baseline, source) per pixel from backbone features."""

    def __init__(
        self,
        in_ch: int,
        tau_range: tuple[float, float] = (5.0, 200.0),
        baseline_from: Literal["head", "median"] = "head",
    ) -> None:
        super().__init__()
        self.tau_min, self.tau_max = float(tau_range[0]), float(tau_range[1])
        if not self.tau_min > 0 or not self.tau_max > self.tau_min:
            raise ValueError(f"tau_range must be positive and ordered; got {tau_range}")
        self.baseline_from = baseline_from

        # Lightweight 3×3×3 conv + pointwise heads, one per output.
        self.shared = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, in_ch), num_channels=in_ch),
            nn.SiLU(inplace=True),
        )
        # τ: predicted via sigmoid → (tau_min, tau_max). Collapse time axis by mean.
        self.tau_head = nn.Conv3d(in_ch, 1, kernel_size=1)
        # baseline: real-valued scalar per pixel, time collapsed.
        self.baseline_head = nn.Conv3d(in_ch, 1, kernel_size=1)
        # source: time-resolved, non-negative (softplus).
        self.source_head = nn.Conv3d(in_ch, 1, kernel_size=1)

    def forward(
        self,
        features: Tensor,
        denoised: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``features: (B, C, T, H, W) -> (tau, baseline, source)``."""
        h = self.shared(features)

        tau_raw = self.tau_head(h).mean(dim=2, keepdim=True)        # (B, 1, 1, H, W)
        alpha = torch.sigmoid(tau_raw)
        tau = self.tau_min + alpha * (self.tau_max - self.tau_min)

        if self.baseline_from == "median":
            if denoised is None:
                raise ValueError("baseline_from='median' requires passing `denoised`")
            baseline = denoised.median(dim=2, keepdim=True).values   # (B, 1, 1, H, W)
        else:
            baseline = self.baseline_head(h).mean(dim=2, keepdim=True)

        source = torch.nn.functional.softplus(self.source_head(h))  # (B, 1, T, H, W)
        return tau, baseline, source
