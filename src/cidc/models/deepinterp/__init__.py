"""DeepInterpolation-style temporal U-Net for CIDC25.

Components
----------
- ``unet.TemporalUNet``  — the network.
- ``window.make_di_window`` — DeepInterpolation input/target slicing.
- ``augment.continuous_gain_augment`` — resample noise at a fresh gain
  for Task-2 OOD generalisation.
- ``augment.sample_log_uniform_gain`` — log-uniform gain draw.
- ``rf.receptive_field`` — analytic receptive-field calculation.
"""
from .augment import continuous_gain_augment, sample_log_uniform_gain
from .rf import receptive_field
from .unet import TemporalUNet
from .window import make_di_window

__all__ = [
    "TemporalUNet",
    "make_di_window",
    "continuous_gain_augment",
    "sample_log_uniform_gain",
    "receptive_field",
]
