"""DeepCAD: 3D U-Net + temporal Noise2Noise for calcium imaging.

Core idea (Li et al. 2021, Nature Methods)
------------------------------------------
Consecutive frames of a calcium recording are two *independent* noisy
observations of the same underlying slowly-varying signal (the calcium
transient half-life is ~45 frames in F0).  Split a window into odd- and
even-indexed sub-volumes; use one as input and the other as target.
Training with plain MSE is then valid Noise2Noise (Lehtinen 2018): with
zero-mean independent noise the optimum of E[(x - f(y))^2] over `f` is
the denoised signal, not the noisy target.

For CIDC25 we additionally evaluate the Poisson-Gaussian NLL against the
target sub-volume (more statistically efficient in the low-SNR regime).

Components
----------
- ``unet3d.DeepCADNet`` — 3D U-Net (uses the n2v3d backbone).
- ``pairing.temporal_halves`` — split a ``(B, 1, 2T, H, W)`` window into
  odd/even sub-volumes for N2N training.
- ``augment.continuous_gain_augment_3d`` — shared gain resampler.
- ``rf.receptive_field_3d`` — re-exported from ``n2v3d``.
"""
from ..deepinterp.augment import (
    continuous_gain_augment as continuous_gain_augment_3d,
    sample_log_uniform_gain,
)
from ..n2v3d.rf import receptive_field_3d
from .pairing import temporal_halves
from .unet3d import DeepCADNet

__all__ = [
    "DeepCADNet",
    "temporal_halves",
    "continuous_gain_augment_3d",
    "sample_log_uniform_gain",
    "receptive_field_3d",
]
