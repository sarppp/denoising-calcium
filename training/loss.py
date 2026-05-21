"""
Poisson-Gaussian Negative Log-Likelihood loss.

Sensor model:  y = g·Poisson(μ/g) + N(0, σ_r²)
Moments:       E[y|μ] = μ,  Var[y|μ] = g·μ + σ_r²  =: V(μ)

Gaussian approximation (CLT, valid for μ/g ≳ 3):

    NLL = ½·log V(μ) + ½·(y − μ)² / V(μ)

This is the heteroscedastic Gaussian NLL — the variance V depends on the
signal μ, so bright pixels contribute more loss at the same relative error.

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
        var_floor:   float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            reduction: 'mean' → scalar; 'none' → [B] (mean over voxels per element).
            var_floor: Minimum variance floor in ADU². Default 1.0 is well below
                       measured read_var (>= 2490). Prevents log(0) when mu is
                       transiently negative during early training.
        """
        y_pred = torch.clamp(y_pred, min=1e-8)
        y_true = torch.clamp(y_true, min=0.0)

        g          = self._as_bchannel(g,         y_pred)
        sigma_r_sq = self._as_bchannel(sigma_r_sq, y_pred)

        # Signal-dependent variance: V(mu) = g * mu + sigma_r^2
        var = g * y_pred + sigma_r_sq
        var = torch.clamp(var, min=var_floor)

        # Heteroscedastic Gaussian NLL: 0.5 * log(V) + 0.5 * (y - mu)^2 / V
        resid = y_true - y_pred
        loss = 0.5 * torch.log(var) + 0.5 * resid * resid / var

        if mask is not None:
            # Only count loss on *masked* (mask=0) voxels — the N2V3D constraint.
            # (mask=0 are the blind-spot voxels the model must predict from context.)
            masked = 1.0 - mask.unsqueeze(1)  # [B, 1, T, H, W], 1 where we predict
            loss = loss * masked
            if reduction == 'none':
                n_masked = masked.sum(dim=(1, 2, 3, 4)).clamp(min=1)
                return loss.sum(dim=(1, 2, 3, 4)) / n_masked   # [B]
            return loss.sum() / (masked.sum() + 1e-8)
        else:
            if reduction == 'none':
                return loss.mean(dim=(1, 2, 3, 4))           # [B]
            return loss.mean()

    @staticmethod
    def _as_bchannel(v: torch.Tensor | float, ref: torch.Tensor) -> torch.Tensor:
        """Broadcast a scalar or [B] tensor to [B, 1, 1, 1, 1]."""
        t = torch.as_tensor(v, dtype=torch.float32, device=ref.device)
        t = torch.clamp(t, min=1e-2, max=1e6)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return t.view(-1, 1, 1, 1, 1)
