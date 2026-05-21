"""3D U-Net for Noise2Void blind-spot denoising.

Architecture
------------
- Input:  ``(B, 1, T, H, W)`` — single-channel Anscombe-space sub-volume
  with blind-spot masking already applied (see ``mask.py``).
- Encoder: ``depth`` levels with ``base_ch * 2**i`` channels each, two
  3D convs per level (GroupNorm + SiLU), MaxPool3d between levels.
- Bottleneck: 2 convs at the deepest channel count.
- Decoder: ConvTranspose3d upsample + skip concat + 2 convs.
- Head: 1x1x1 conv to a single channel = predicted *Anscombe-space*
  clean signal. ``forward`` inverts the Anscombe VST (asymptotic form)
  to return raw ADU, directly consumable by ``cidc.losses.poisson_gaussian_nll``.

Defaults produce an RF of roughly 22x22x22 voxels at the input, which
covers a typical neuron (radius 2-3 px) and ~half of a calcium transient
(half-life ~45 frames in F0, but event onsets are much faster).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...noise import NoiseParams


def _conv3d_block(in_ch: int, out_ch: int, kernel: int = 3, convs: int = 2) -> nn.Sequential:
    """Two Conv3d + GroupNorm + SiLU blocks.

    GroupNorm uses min(8, out_ch) groups. When out_ch < 8 (e.g. the 1-ch
    head), groups == out_ch which is equivalent to InstanceNorm — acceptable
    for the final 1×1×1 conv but only reachable if base_ch < 8.
    """
    layers: list[nn.Module] = []
    pad = kernel // 2
    for i in range(convs):
        c_in = in_ch if i == 0 else out_ch
        layers += [
            nn.Conv3d(c_in, out_ch, kernel_size=kernel, padding=pad, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class UNet3D(nn.Module):
    """3D U-Net for N2V-style blind-spot denoising."""

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        depth: int = 3,
        kernel: int = 3,
        pool: int = 2,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        chs = [base_ch * (2**i) for i in range(depth)]

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = in_ch
        for i in range(depth - 1):
            self.enc_blocks.append(_conv3d_block(prev, chs[i], kernel=kernel))
            self.downs.append(nn.MaxPool3d(pool))
            prev = chs[i]
        self.bottleneck = _conv3d_block(prev, chs[-1], kernel=kernel)

        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.ups.append(
                nn.ConvTranspose3d(chs[i + 1], chs[i], kernel_size=pool, stride=pool)
            )
            self.dec_blocks.append(_conv3d_block(2 * chs[i], chs[i], kernel=kernel))

        self.head = nn.Conv3d(chs[0], 1, kernel_size=1)

    def forward(self, x_anscombe: Tensor, params: NoiseParams) -> Tensor:
        """Predict clean signal in raw ADU from an Anscombe-space volume.

        Parameters
        ----------
        x_anscombe
            ``(B, 1, T, H, W)`` tensor.  Typically already blind-spot
            masked (see ``mask.stratified_blindspot``).
        params
            NoiseParams to invert the Anscombe VST.

        Returns
        -------
        ``(B, 1, T, H, W)`` prediction in raw ADU.
        """
        skips: list[Tensor] = []
        h = x_anscombe
        for enc, down in zip(self.enc_blocks, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)
        h = self.bottleneck(h)

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
        return (z_pred / 2.0).pow(2) * g - 0.375 * g - sr2 / g
