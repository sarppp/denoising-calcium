"""Analytic receptive-field calculation for the temporal U-Net."""

from __future__ import annotations


def receptive_field(
    depth: int = 3,
    convs_per_level: int = 2,
    kernel: int = 3,
    pool: int = 2,
) -> int:
    """Input-space receptive field of a symmetric U-Net.

    A U-Net of given ``depth`` (including bottleneck) where each level has
    ``convs_per_level`` ``kernel x kernel`` convolutions and downsamples by
    ``pool`` between levels. Decoder mirrors the encoder. The formula below
    walks the network from deepest to shallowest, tracking RF in input
    pixels:

        rf = 1
        stride = pool ** (depth - 1)
        # bottleneck
        rf += convs_per_level * (kernel - 1) * stride
        for level from (depth-2) down to 0:
            stride //= pool
            # decoder convs at this level (stride matches this level)
            rf += convs_per_level * (kernel - 1) * stride
            # encoder convs at this level (same stride)
            rf += convs_per_level * (kernel - 1) * stride
        return rf
    """
    rf = 1
    stride = pool ** (depth - 1)
    # bottleneck block
    rf += convs_per_level * (kernel - 1) * stride
    for _ in range(depth - 1):
        stride //= pool
        # decoder level
        rf += convs_per_level * (kernel - 1) * stride
        # encoder level at the same resolution
        rf += convs_per_level * (kernel - 1) * stride
    return rf
