"""Temporal U-Net for DeepInterpolation-style denoising.

Architecture summary
--------------------
- Input:  ``(B, 2K, H, W)`` — ``2K`` noisy frames in Anscombe space
  (center omitted). Default K=6 → temporal context 13 frames (9..17 is
  the target range; 13 is the sweet spot on T4 memory).
- Encoder: 3 resolution levels, [32, 64, 128] channels, 2 convs/level,
  GroupNorm + SiLU, MaxPool(2) between levels.
- Bottleneck: 2 convs at 128 channels.
- Decoder: mirror; bilinear upsample + 1x1 channel-halving + skip concat
  + 2 convs.
- Head: 1x1 conv to a single scalar channel = predicted *Anscombe-space*
  clean mean. The forward method applies the asymptotic inverse Anscombe
  at the tail so the returned tensor is in **raw ADU**, directly
  comparable to the noisy observation and usable with
  ``cidc.losses.poisson_gaussian_nll``.
- Receptive field: 41 px (see ``rf.receptive_field(3, 2, 3, 2)``). Target
  was ~32 px; 41 px is comfortably within budget and covers a neuron
  plus a ring of nearby context.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...noise import NoiseParams


def _conv_block(in_ch: int, out_ch: int, convs: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(convs):
        c_in = in_ch if i == 0 else out_ch
        layers += [
            nn.Conv2d(c_in, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.SiLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class TemporalUNet(nn.Module):
    """DeepInterpolation-style temporal U-Net.

    Parameters
    ----------
    half_context
        Half the temporal window, ``K``. Input channel count is ``2K``.
        Default 6 → 13-frame total context.
    base_ch
        Channels at the top level. Default 32.
    depth
        Number of resolution levels, encoder+bottleneck. Default 3.
    predict_space
        ``"anscombe"`` — model's final 1x1 conv output is in Anscombe
        space. The forward method inverts to raw ADU for you using the
        `NoiseParams` you pass at forward time.
    """

    def __init__(
        self,
        half_context: int = 6,
        base_ch: int = 32,
        depth: int = 3,
        predict_space: str = "anscombe",
    ) -> None:
        super().__init__()
        if predict_space != "anscombe":
            raise NotImplementedError("only anscombe-space prediction is supported")
        self.half_context = int(half_context)
        self.depth = int(depth)
        self.predict_space = predict_space

        in_ch = 2 * self.half_context
        chs = [base_ch * (2**i) for i in range(depth)]

        # Encoder: depth-1 downsampling stages + bottleneck.
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = in_ch
        for i in range(depth - 1):
            self.enc_blocks.append(_conv_block(prev, chs[i]))
            self.downs.append(nn.MaxPool2d(2))
            prev = chs[i]
        self.bottleneck = _conv_block(prev, chs[-1])

        # Decoder: mirror.
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            # ConvTranspose halves channels while upsampling spatially.
            self.ups.append(
                nn.ConvTranspose2d(chs[i + 1], chs[i], kernel_size=2, stride=2)
            )
            # After concat with skip, channels = 2 * chs[i].
            self.dec_blocks.append(_conv_block(2 * chs[i], chs[i]))

        self.head = nn.Conv2d(chs[0], 1, kernel_size=1)

    # ------------------------------------------------------------------ #
    def forward(self, x_anscombe: Tensor, params: NoiseParams) -> Tensor:
        """Predict the clean center frame in raw ADU.

        Parameters
        ----------
        x_anscombe
            ``(B, 2K, H, W)`` tensor of Anscombe-transformed context frames.
        params
            Noise parameters of *this* batch (for inverting the Anscombe
            transform at the output head). With continuous-gain augmentation
            these come from the sampled gain, not the raw-file gain.

        Returns
        -------
        ``(B, 1, H, W)`` predicted clean mean in raw ADU. Apply
        ``cidc.losses.poisson_gaussian_nll`` against the observed noisy
        center frame.
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
            # Pad in case input H/W is not a perfect multiple of 2**depth.
            if h.shape[-2:] != skip.shape[-2:]:
                dh = skip.shape[-2] - h.shape[-2]
                dw = skip.shape[-1] - h.shape[-1]
                h = nn.functional.pad(h, (0, dw, 0, dh))
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        z_pred = self.head(h)  # Anscombe-space prediction
        return _inverse_anscombe_torch(z_pred, params)


# ---------------------------------------------------------------------- #
# Differentiable Anscombe inverse (asymptotic form).                     #
# ---------------------------------------------------------------------- #


def _inverse_anscombe_torch(z: Tensor, params: NoiseParams) -> Tensor:
    """Asymptotic inverse Anscombe: y = (z/2)^2 * g - 3g/8 - sigma_r^2 / g.

    Matches ``cidc.noise.inverse_anscombe(method="asymptotic")`` but stays
    on device and differentiable. The network is trained with the
    Poisson-Gaussian NLL in raw ADU, so any residual bias of the
    asymptotic (vs Mäkitalo-Foi exact) inverse is absorbed by the head.
    """
    g = float(params.gain)
    sr2 = float(max(params.read_var, 0.0))
    return (z / 2.0).pow(2) * g - 0.375 * g - sr2 / g
