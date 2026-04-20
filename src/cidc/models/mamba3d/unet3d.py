"""3-D U-Net with a bi-directional Mamba bottleneck.

Encoder and decoder are structurally identical to ``n2v3d.UNet3D``
(same GroupNorm + SiLU convs, same Anscombe-inverse output head). The
only difference is the bottleneck: instead of two 3-D convolutions, we
stack ``n_layers`` :class:`BiMambaBlock` layers on the flattened
bottleneck volume.

Input / output contract matches every other CIDC25 model:

    forward(x_anscombe: (B, 1, T, H, W), params: NoiseParams) -> (B, 1, T, H, W)  # raw ADU
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...noise import NoiseParams
from ..n2v3d.unet3d import _conv3d_block
from .blocks import BiMambaBlock


__all__ = ["MambaUNet3D"]


class MambaUNet3D(nn.Module):
    """3-D U-Net whose bottleneck is a stack of bi-directional Mamba blocks."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        depth: int = 3,
        kernel: int = 3,
        pool: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("MambaUNet3D requires depth >= 2")
        self.depth = int(depth)
        chs = [base_ch * (2**i) for i in range(depth)]

        # Encoder: depth-1 downsampling stages, then an entry conv into the bottleneck.
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = in_ch
        for i in range(depth - 1):
            self.enc_blocks.append(_conv3d_block(prev, chs[i], kernel=kernel))
            self.downs.append(nn.MaxPool3d(pool))
            prev = chs[i]

        # Bottleneck: 1 conv block (dim-raise to chs[-1]) + stack of BiMambaBlocks.
        self.bottleneck_in = _conv3d_block(prev, chs[-1], kernel=kernel)
        self.bottleneck_mamba = nn.ModuleList(
            BiMambaBlock(
                d_model=chs[-1],
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bidirectional=bidirectional,
            )
            for _ in range(int(n_layers))
        )

        # Decoder: mirror the encoder.
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.ups.append(
                nn.ConvTranspose3d(chs[i + 1], chs[i], kernel_size=pool, stride=pool)
            )
            self.dec_blocks.append(_conv3d_block(2 * chs[i], chs[i], kernel=kernel))

        self.head = nn.Conv3d(chs[0], 1, kernel_size=1)

    def forward(self, x_anscombe: Tensor, params: NoiseParams) -> Tensor:
        skips: list[Tensor] = []
        h = x_anscombe
        for enc, down in zip(self.enc_blocks, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.bottleneck_in(h)
        for block in self.bottleneck_mamba:
            h = block(h)

        for up, dec, skip in zip(self.ups, self.dec_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-3:] != skip.shape[-3:]:
                dT = skip.shape[-3] - h.shape[-3]
                dH = skip.shape[-2] - h.shape[-2]
                dW = skip.shape[-1] - h.shape[-1]
                h = nn.functional.pad(h, (0, dW, 0, dH, 0, dT))
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        z_pred = self.head(h)
        g = float(params.gain)
        sr2 = float(max(params.read_var, 0.0))
        # Asymptotic inverse Anscombe in raw ADU (differentiable).
        return (z_pred / 2.0).pow(2) * g - 0.375 * g - sr2 / g
