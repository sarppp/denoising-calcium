"""Bi-directional Mamba block for a U-Net bottleneck.

We use the upstream ``mamba-ssm`` implementation of the selective SSM
(Gu & Dao 2024). The block processes a flattened ``(B, L, C)`` sequence
with two parallel Mamba branches — forward and reverse — summed with a
residual connection and LayerNormed.

The block is intentionally thin: all of the numerical heavy lifting lives
inside ``mamba_ssm.modules.mamba_simple.Mamba``. What we add is

1. Bi-directionality (vanilla Mamba is left-to-right causal).
2. A residual + pre-norm wrapper.
3. The (C, D, H, W) ↔ (L, C) rearrange so it slots into a U-Net.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from einops import rearrange


__all__ = ["BiMambaBlock"]


def _import_mamba():
    """Lazy import so this module can be loaded on CPU-only machines.

    ``mamba-ssm`` is a CUDA-only optional dependency (see pyproject extras).
    The failure message points the user at the install command.
    """
    try:
        from mamba_ssm import Mamba
    except ImportError as e:
        raise ImportError(
            "mamba-ssm is required for MambaUNet3D. Install on a CUDA box with:\n"
            "    uv pip install -e '.[mamba]'\n"
            "(requires nvcc in PATH)."
        ) from e
    return Mamba


class BiMambaBlock(nn.Module):
    """Pre-norm bi-directional Mamba block on a flattened volume.

    Expects input shape ``(B, C, D, H, W)``. Internally reshapes to
    ``(B, D*H*W, C)``, runs forward + reverse Mamba, sums them, adds the
    residual, and restores the 3-D shape.

    Parameters
    ----------
    d_model
        Channel dimension (``C``).
    d_state
        State-space hidden-state dim per channel (default 16, upstream default).
    d_conv
        Width of the local causal 1-D convolution inside Mamba (default 4).
    expand
        Inner-dim expansion factor (default 2 → hidden is ``2*C``).
    bidirectional
        If False, behaves like a plain causal Mamba (no reverse branch).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        Mamba = _import_mamba()
        self.bidirectional = bool(bidirectional)
        self.norm = nn.LayerNorm(d_model)
        self.fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if self.bidirectional:
            self.rev = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, D, H, W) -> (B, C, D, H, W)``."""
        b, c, d, h, w = x.shape
        seq = rearrange(x, "b c d h w -> b (d h w) c")          # (B, L, C)
        y = self.norm(seq)
        out_fwd = self.fwd(y)
        if self.bidirectional:
            out_rev = self.rev(y.flip(dims=[1])).flip(dims=[1])
            out = out_fwd + out_rev
        else:
            out = out_fwd
        seq = seq + out                                          # residual
        return rearrange(seq, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
