"""Noise2Void 3D: blind-spot self-supervised denoiser for calcium imaging.

Components
----------
- ``unet3d.UNet3D``            — 3D U-Net (isotropic kernels, anisotropy
  optional via ``pool`` / ``kernel`` args).
- ``mask.stratified_blindspot``— random blind-spot mask + donut replacement
  for self-supervised target construction.
- ``augment.continuous_gain_augment_3d`` — shared gain resampler (wraps
  the one from ``deepinterp.augment``; exposed here for API symmetry).
- ``rf.receptive_field_3d``    — analytic 3D RF calculation.
"""
from ..deepinterp.augment import (
    continuous_gain_augment as continuous_gain_augment_3d,
    sample_log_uniform_gain,
)
from .mask import stratified_blindspot
from .rf import receptive_field_3d
from .unet3d import UNet3D

__all__ = [
    "UNet3D",
    "stratified_blindspot",
    "continuous_gain_augment_3d",
    "sample_log_uniform_gain",
    "receptive_field_3d",
]
