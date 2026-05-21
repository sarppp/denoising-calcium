"""Losses for Poisson-Gaussian denoising.

Derivation of the primary loss (Poisson-Gaussian negative log-likelihood)
============================================================================

Sensor model (matches the CIDC25 measurements; see ``cidc.noise``)::

    N   ~ Poisson(mu / g)           # integer photon count
    eps ~ Normal(0, sigma_r^2)      # Gaussian read noise (ADU)
    y   = g * N + eps               # observation in ADU

The two first moments, with mu > 0 the underlying clean signal in ADU::

    E[y  | mu] = mu
    Var[y| mu] = g * mu + sigma_r^2       (=: V(mu))

The *exact* likelihood is a mixture::

    p(y | mu) = sum_{n=0..inf} Poisson(n; mu/g) * N(y - g*n; 0, sigma_r^2)

which is intractable for training.

Standard approximation (central limit, valid once mu/g >~ 3, which holds
for almost all foreground pixels in this dataset): replace the discrete
Poisson by a Gaussian with matching mean and variance and convolve with
the already-Gaussian read noise, yielding::

    y | mu ~ N( mu, V(mu) ),   V(mu) = g * mu + sigma_r^2

Dropping the `log(2 pi)/2` constant, the heteroscedastic Gaussian NLL is

    NLL(mu ; y, g, sigma_r^2) = 1/2 * log V(mu)  +  (y - mu)^2 / (2 V(mu))    (*)

with gradient

    d NLL / d mu = g / (2 V)             # variance-term pull
                 - (y - mu) / V          # residual-term pull
                 - g (y - mu)^2 / (2 V^2)# interaction

Key properties verified numerically in ``scripts/verify_nll.py``:

1. For fixed mu, NLL is convex-ish in y and *increases* on bright noisy
   regions: the log V term grows as ~ 1/2 log(g mu), i.e. bright pixels
   contribute more loss at the same relative error.
2. The gradient at mu = y is *small and negative* (pushes mu down very
   slightly because higher mu inflates the log-variance penalty); the
   exact zero sits at mu* = y - g/2 + O(1/y).  This is the intrinsic
   heteroscedastic-Gaussian bias and is negligible compared to per-pixel
   noise, but we document it so nobody mistakes it for a bug.
3. On a synthetic flat patch (mu_true = const, y = noisy observation),
   the empirical NLL minimiser converges to mu_true within Monte-Carlo
   error as the number of independent samples grows.

Numerical stability
-------------------
`V(mu)` is clamped at ``var_floor`` (default 1.0 ADU²) so negative
predictions (which the network can produce early in training) do not
cause log(0) / divide-by-zero.  We do *not* apply a softplus to mu; the
clamp on V is sufficient and keeps the residual term well-defined.
Note: ``var_floor`` is independent of ``read_var`` — for sufficiently
negative mu the clamped variance can be much smaller than ``read_var``,
sharpening the loss and gradients in that regime.

References
----------
- Foi, Trimeche, Katkovnik, Egiazarian. "Practical Poissonian-Gaussian
  noise modeling and fitting for single-image raw-data." IEEE TIP 2008.
- Laine et al. "High-Quality Self-Supervised Deep Image Denoising."
  NeurIPS 2019 (appendix derives the same NLL in Bayesian form).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

__all__ = [
    "poisson_gaussian_nll",
    "anscombe_mse",
    "calcium_kinetics_loss",
]


def poisson_gaussian_nll(
    mu: Tensor,
    y: Tensor,
    gain: float | Tensor,
    read_var: float | Tensor,
    reduce: Literal["mean", "sum", "none"] = "mean",
    var_floor: float = 1.0,
) -> Tensor:
    """Heteroscedastic-Gaussian approximation to the Poisson-Gaussian NLL.

    Computes, element-wise::

        V   = clamp(gain * mu + read_var, min=var_floor)
        nll = 0.5 * log(V) + 0.5 * (y - mu)^2 / V

    Parameters
    ----------
    mu
        Predicted clean signal in raw ADU.  Any shape.
    y
        Observed noisy signal in raw ADU.  Broadcastable to ``mu``.
    gain, read_var
        Scalars or tensors broadcastable to ``mu``.  Allowing tensors lets
        us pass per-sample gains during continuous-gain augmentation.
    reduce
        ``"mean"`` / ``"sum"`` / ``"none"``.
    var_floor
        Minimum variance used inside log() and 1/V to keep the loss finite
        when `mu` is transiently negative during training.  Default 1.0
        ADU^2 is well below the measured `read_var` (>= 2490).

    Returns
    -------
    Scalar if reduced, else tensor of shape of ``mu``.
    """
    if mu.shape != y.shape:
        y = y.expand_as(mu)
    if isinstance(gain, Tensor):
        gain = gain.to(mu)
    if isinstance(read_var, Tensor):
        read_var = read_var.to(mu)

    var = gain * mu + read_var
    var = torch.clamp(var, min=var_floor)
    resid = y - mu
    nll = 0.5 * torch.log(var) + 0.5 * resid * resid / var

    if reduce == "mean":
        return nll.mean()
    if reduce == "sum":
        return nll.sum()
    return nll


def anscombe_mse(
    z_pred: Tensor,
    z_target: Tensor,
    reduce: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """MSE in Anscombe (unit-variance) space.

    Provided as a cheap sanity-check alternative loss; under the VST the
    noise variance is ~1 everywhere so plain MSE is already a valid
    (approximate) NLL.  Useful for ablating against ``poisson_gaussian_nll``.
    """
    d = z_pred - z_target
    if reduce == "mean":
        return (d * d).mean()
    if reduce == "sum":
        return (d * d).sum()
    return d * d


def calcium_kinetics_loss(
    denoised: Tensor,
    reconstruction: Tensor,
    source: Tensor,
    sparsity_l1: float = 0.005,
    detach_input: bool = False,
    reduce: Literal["mean", "sum", "none"] = "mean",
) -> Tensor:
    """PINN auxiliary loss for calcium-kinetics regularisation.

    Combines (a) a reconstruction term enforcing that the denoised trace
    is explainable by the ODE ``dC/dt = -C/τ + s(t); F = C + b`` and
    (b) an L1 sparsity prior on the source term ``s(t)``.

    Parameters
    ----------
    denoised
        ``(B, 1, T, H, W)`` — denoiser output in raw ADU. Treated as the
        *target* for the kinetics reconstruction.
    reconstruction
        ``(B, 1, T, H, W)`` — ``euler_forward(τ, b, s)`` output.
    source
        ``(B, 1, T, H, W)`` — predicted per-frame source term ``s(t) >= 0``.
    sparsity_l1
        Weight on ``E|s|``. 0.001–0.01 keeps real transients intact; higher
        values eat fast-rise calcium events.
    detach_input
        If True, detach ``denoised`` before computing the MSE so the
        PINN loss *regularises* the kinetics head without backprop-ing
        through the denoiser. Usually False — joint training is fine.
    reduce
        ``"mean"`` / ``"sum"`` / ``"none"``.

    Returns
    -------
    Scalar if reduced, else tensor of shape of ``denoised``.
    """
    target = denoised.detach() if detach_input else denoised
    d = reconstruction - target
    mse = d * d
    if sparsity_l1 > 0:
        l1 = sparsity_l1 * source.abs()
        total = mse + l1
    else:
        total = mse
    if reduce == "mean":
        return total.mean()
    if reduce == "sum":
        return total.sum()
    return total
