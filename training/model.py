"""3D U-Net model for denoising."""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """3D convolutional block with batchnorm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net with skip connections."""

    def __init__(self, in_channels=2, out_channels=1, channels=[32, 64, 128]):
        super().__init__()
        self.channels = channels

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, channels[0])
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(channels[0], channels[1])
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(channels[1], channels[2])

        # Bottleneck
        self.bottle = ConvBlock3D(channels[2], channels[2])

        # Decoder
        self.upconv2 = nn.ConvTranspose3d(channels[2], channels[1], 2, stride=2)
        self.dec2 = ConvBlock3D(channels[1] * 2, channels[1])
        self.upconv1 = nn.ConvTranspose3d(channels[1], channels[0], 2, stride=2)
        self.dec1 = ConvBlock3D(channels[0] * 2, channels[0])

        # Output
        self.out = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)

        # Bottleneck
        b = self.bottle(e3)

        # Decoder with skip connections
        up2 = self.upconv2(b)
        if up2.shape[-3:] != e2.shape[-3:]:
            dT = e2.shape[-3] - up2.shape[-3]
            dH = e2.shape[-2] - up2.shape[-2]
            dW = e2.shape[-1] - up2.shape[-1]
            up2 = nn.functional.pad(up2, (0, dW, 0, dH, 0, dT))
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = self.upconv1(d2)
        if up1.shape[-3:] != e1.shape[-3:]:
            dT = e1.shape[-3] - up1.shape[-3]
            dH = e1.shape[-2] - up1.shape[-2]
            dW = e1.shape[-1] - up1.shape[-1]
            up1 = nn.functional.pad(up1, (0, dW, 0, dH, 0, dT))
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        out = self.out(d1)
        return out
