"""PINN wrapper: any backbone + calcium-kinetics head.

The calcium-kinetics ODE for a single pixel trace ``C(t)`` is::

    dC/dt = -C/τ + s(t),             F(t) = C(t) + b

- ``τ`` — decay time constant (frames); pixel-specific because different
  neurons have different GCaMP kinetics.
- ``s(t)`` — non-negative source term (spiking/calcium influx); sparse.
- ``b`` — resting baseline fluorescence; pixel-specific.

At training time the PINN head predicts ``(τ, b, s)`` from features, we
integrate the ODE forward with explicit Euler, and compare the
reconstruction ``F_pinn(t)`` to the primary denoiser's output ``F_nn(t)``.
The resulting regulariser biases the denoiser toward biologically
plausible traces without replacing its output.

Components
----------
- ``kinetics.euler_forward`` — differentiable explicit-Euler integrator.
- ``head.PINNHead``          — per-pixel τ, baseline, source prediction.
- ``model.PINNWrapper``      — backbone + PINN head, returns both outputs.
"""

from .head import PINNHead
from .kinetics import euler_forward
from .model import PINNWrapper

__all__ = ["PINNWrapper", "PINNHead", "euler_forward"]
