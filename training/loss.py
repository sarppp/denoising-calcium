"""
Poisson-Gaussian Negative Log-Likelihood loss.

Model:   y = Poisson(x/g)·g  +  Normal(0, σ_r²)
Loss:    L = 0.5·(y−ŷ)²/σ_r²  +  ŷ/g  −  (y/g)·log(ŷ/g)

where y = observed (counts), ŷ = predicted (reconstruction),
      g = gain (photons/ADU), σ_r² = read-noise variance.

The loss supports:
  reduction='mean'  — scalar loss for the optimiser (training)
  reduction='none'  — per-element loss [B] for per-stack diagnostics
"""

import torch
import torch.nn as nn


class PGNLLLoss(nn.Module):

    def forward(
        self,
        y_pred:      torch.Tensor,   # [B, 1, T, H, W]
        y_true:      torch.Tensor,   # [B, 1, T, H, W]
        g:           torch.Tensor | float,
        sigma_r_sq:  torch.Tensor | float,
        mask:        torch.Tensor | None = None,  # [B, T, H, W], 0=predict 1=observe
        reduction:   str = 'mean',
    ) -> torch.Tensor:
        """
        Args:
            reduction: 'mean' → scalar; 'none' → [B] (mean over voxels per element).
        """
        eps = 1e-8

        y_pred = torch.clamp(y_pred, min=eps)
        y_true = torch.clamp(y_true, min=0.0)

        g          = self._as_bchannel(g,         y_pred)
        sigma_r_sq = self._as_bchannel(sigma_r_sq, y_pred)

        # Gaussian term
        gaussian = 0.5 * (y_true - y_pred) ** 2 / sigma_r_sq

        # Poisson term:  λ − y·log(λ),  λ = ŷ/g
        lam     = torch.clamp(y_pred / g, min=eps)
        log_lam = torch.clamp(torch.log(lam), min=-50.0, max=50.0)
        poisson = lam - (y_true / g) * log_lam

        loss = gaussian + poisson

        # Guard against NaN/Inf from pathological batches (e.g. A1/B1 poor fit).
        loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, 1e4))
        loss = torch.clamp(loss, min=0.0, max=1e4)

        if mask is not None:
            # Only count loss on *masked* (mask=0) voxels — the N2V3D constraint.
            # (mask=0 are the blind-spot voxels the model must predict from context.)
            masked = 1.0 - mask.unsqueeze(1)  # [B, 1, T, H, W], 1 where we predict
            loss = loss * masked
            if reduction == 'none':
                n_masked = masked.sum(dim=(1, 2, 3, 4)).clamp(min=1)
                return loss.sum(dim=(1, 2, 3, 4)) / n_masked   # [B]
            return loss.sum() / (masked.sum() + eps)
        else:
            if reduction == 'none':
                return loss.mean(dim=(1, 2, 3, 4))           # [B]
            return loss.mean()

    @staticmethod
    def _as_bchannel(v: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
        """Broadcast a scalar or [B] tensor to [B, 1, 1, 1, 1]."""
        t = torch.as_tensor(v, dtype=torch.float32, device=ref.device)
        t = torch.clamp(t, min=1e-8, max=1e6)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return t.view(-1, 1, 1, 1, 1)
