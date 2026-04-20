"""Analytic receptive-field calculation for a 3D U-Net.

Anisotropic-friendly: accept (kT, kH, kW) and (pT, pH, pW) tuples.
"""

from __future__ import annotations


def _rf_axis(depth: int, convs_per_level: int, k: int, p: int) -> int:
    rf = 1
    stride = p ** (depth - 1)
    rf += convs_per_level * (k - 1) * stride
    for _ in range(depth - 1):
        stride //= p
        rf += convs_per_level * (k - 1) * stride  # decoder
        rf += convs_per_level * (k - 1) * stride  # encoder
    return rf


def receptive_field_3d(
    depth: int = 3,
    convs_per_level: int = 2,
    kernel: tuple[int, int, int] = (3, 3, 3),
    pool: tuple[int, int, int] = (2, 2, 2),
) -> tuple[int, int, int]:
    """Per-axis input receptive field of a symmetric 3D U-Net."""
    return (
        _rf_axis(depth, convs_per_level, kernel[0], pool[0]),
        _rf_axis(depth, convs_per_level, kernel[1], pool[1]),
        _rf_axis(depth, convs_per_level, kernel[2], pool[2]),
    )
