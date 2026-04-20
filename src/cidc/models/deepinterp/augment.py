"""Continuous-gain augmentation for Task-2 (OOD noise level) generalisation.

Motivation
----------
The CIDC25 training stacks only cover gains ≈ 28 (level 1) and ≈ 249
(level 2), while Task 2 is tested at gain ≈ 990 (level 3), entirely out
of distribution. To force the network to learn a gain-invariant denoiser
we augment each training mini-batch on the fly:

1. Produce a near-clean estimate ``mu_hat`` from the current noisy batch
   (e.g. temporal mean of the DI context window).
2. Draw a fresh gain ``g`` log-uniformly from ``[g_lo, g_hi]``.
3. Resample the entire window + target with Poisson-Gaussian noise at
   that gain via ``cidc.noise.sample_poisson_gaussian``.
4. Feed the freshly noisy window to the network with the matched ``g``
   passed into the loss.

Sampling from ``[20, 2000]`` (log-uniform) brackets all three CIDC
levels and covers the OOD gap continuously.
"""

from __future__ import annotations

import numpy as np

from ...noise import NoiseParams, sample_poisson_gaussian


def sample_log_uniform_gain(
    rng: np.random.Generator,
    lo: float = 20.0,
    hi: float = 2000.0,
) -> float:
    """Draw one gain ``g`` from a log-uniform prior on ``[lo, hi]``."""
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def continuous_gain_augment(
    clean_estimate: np.ndarray,
    rng: np.random.Generator,
    gain_range: tuple[float, float] = (20.0, 2000.0),
    read_var: float = 2700.0,
) -> tuple[np.ndarray, NoiseParams]:
    """Generate a noisy realisation at a freshly-sampled gain.

    Parameters
    ----------
    clean_estimate
        Near-clean signal (e.g. temporal mean of a DI context window).
        Any shape; values in raw ADU.
    rng
        NumPy Generator for reproducibility.
    gain_range
        ``(lo, hi)`` bounds for the log-uniform gain draw.
    read_var
        Read-noise variance to use. Default ~ level-2 measurement; in
        practice we found read_var varies little (2490..3730) and the
        network is insensitive, so a fixed mid-range value is fine.

    Returns
    -------
    noisy : ndarray, same shape as ``clean_estimate``.
    params : NoiseParams used to generate it (pass to the loss).
    """
    g = sample_log_uniform_gain(rng, *gain_range)
    params = NoiseParams(gain=g, read_var=read_var)
    noisy = sample_poisson_gaussian(clean_estimate, params, rng=rng)
    return noisy, params
