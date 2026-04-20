"""Poisson-Gaussian noise model for CIDC25.

Contains:
- Empirically measured per-stack noise constants (`NOISE_LEVELS`, `FILE_NOISE`).
- Generalised Anscombe transform + inverse (Foi et al. 2008; Mäkitalo & Foi 2011).
- Poisson-Gaussian sampler for Task 2 augmentation.

References
----------
- Foi, Trimeche, Katkovnik, Egiazarian. *Practical Poissonian-Gaussian noise
  modeling and fitting for single-image raw-data.* IEEE TIP 2008.
- Mäkitalo, Foi. *Optimal inversion of the generalized Anscombe transformation
  for Poisson-Gaussian noise.* IEEE TIP 2011.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "NoiseParams",
    "NOISE_LEVELS",
    "FILE_NOISE",
    "anscombe",
    "inverse_anscombe",
    "sample_poisson_gaussian",
    "identify_noise_level",
]


# --------------------------------------------------------------------------- #
# Measured constants (see docs/concepts.md).                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NoiseParams:
    """Parameters of Var[y] = gain * Mean[y] + read_var."""

    gain: float
    read_var: float

    @property
    def read_std(self) -> float:
        return float(np.sqrt(max(self.read_var, 0.0)))


# Canonical per-level parameters — median of matching stacks (A1,B1 / C2,D2 / F3).
NOISE_LEVELS: dict[int, NoiseParams] = {
    1: NoiseParams(gain=28.4, read_var=2490.0),
    2: NoiseParams(gain=248.7, read_var=2700.0),
    3: NoiseParams(gain=990.5, read_var=3730.0),  # OOD (Task 2)
}

# Per-file assignments measured by `scripts/eda_numbers.py`.
FILE_NOISE: dict[str, NoiseParams] = {
    "A1.tif": NOISE_LEVELS[1],
    "B1.tif": NOISE_LEVELS[1],
    "C2.tif": NOISE_LEVELS[2],
    "D2.tif": NOISE_LEVELS[2],
    "F1.tif": NOISE_LEVELS[1],
    "F2.tif": NOISE_LEVELS[2],
    "F3.tif": NOISE_LEVELS[3],
    # F0.tif is clean; no noise params.
}


# --------------------------------------------------------------------------- #
# Generalised Anscombe transform.                                             #
# --------------------------------------------------------------------------- #


def anscombe(
    x: np.ndarray, params: NoiseParams, offset: float = 0.0
) -> np.ndarray:
    """Generalised Anscombe VST for Poisson-Gaussian noise.

    Given observations `y` with `Var[y] = gain * (E[y] - offset) + read_var`
    (plus optional additive `offset` representing a subtracted pedestal),
    the transform

        z = (2 / gain) * sqrt( gain * (y - offset) + 3/8 * gain^2 + read_var )

    yields `Var[z] ≈ 1` independent of signal intensity. The argument of the
    square root is clamped at 0 to handle negative dither values.
    """
    g = float(params.gain)
    sr2 = float(max(params.read_var, 0.0))
    y = np.asarray(x, dtype=np.float64) - offset
    inside = g * y + 0.375 * g * g + sr2
    inside = np.maximum(inside, 0.0)
    return (2.0 / g) * np.sqrt(inside)


def inverse_anscombe(
    z: np.ndarray,
    params: NoiseParams,
    method: Literal["exact", "asymptotic"] = "exact",
    offset: float = 0.0,
) -> np.ndarray:
    """Invert the generalised Anscombe transform.

    Two options:
    - "asymptotic" : trivial algebraic inverse (good for high counts).
    - "exact"      : Mäkitalo-Foi closed-form unbiased inverse (default;
                     noticeably better at low counts).

    Mäkitalo-Foi (2011) closed form::

        E[y | z] ≈ (z/2)^2 + sqrt(3/2)/(4 z) - (11/8)/z^2
                   + 5 sqrt(3/2)/(8 z^3) - 1/8 - read_var / gain^2

    scaled by `gain` and shifted by `offset`.
    """
    g = float(params.gain)
    sr2 = float(max(params.read_var, 0.0))
    z = np.asarray(z, dtype=np.float64)

    if method == "asymptotic":
        y_minus_off = ((z / 2.0) ** 2 * g) - 0.375 * g - sr2 / g
        return y_minus_off + offset

    with np.errstate(divide="ignore", invalid="ignore"):
        z_safe = np.where(z > 1e-6, z, 1e-6)
        z2 = z_safe * z_safe
        term = (
            (z_safe / 2.0) ** 2
            + np.sqrt(1.5) / (4.0 * z_safe)
            - 11.0 / (8.0 * z2)
            + 5.0 * np.sqrt(1.5) / (8.0 * z_safe * z2)
            - 0.125
            - sr2 / (g * g)
        )
    y_minus_off = g * term
    asym = ((z / 2.0) ** 2 * g) - 0.375 * g - sr2 / g
    out = np.where(z > 0.5, y_minus_off, asym)
    return out + offset


# --------------------------------------------------------------------------- #
# Poisson-Gaussian sampler (for augmentation / noise-level generalisation).   #
# --------------------------------------------------------------------------- #


def sample_poisson_gaussian(
    clean: np.ndarray,
    params: NoiseParams,
    rng: np.random.Generator | None = None,
    offset: float = 0.0,
) -> np.ndarray:
    """Draw a realisation of `y = Poisson(clean/gain) * gain + N(0, read_var) + offset`.

    The forward model::

        E[y] = clean + offset
        Var[y] = gain * clean + read_var
    """
    if rng is None:
        rng = np.random.default_rng()
    g = float(params.gain)
    sr = float(params.read_std)
    clean = np.asarray(clean, dtype=np.float64)
    lam = np.maximum(clean / g, 0.0)
    shots = rng.poisson(lam).astype(np.float64) * g
    read = rng.normal(0.0, sr, size=clean.shape)
    return shots + read + offset


# --------------------------------------------------------------------------- #
# Convenience: identify noise level from a file name.                         #
# --------------------------------------------------------------------------- #


def identify_noise_level(filename: str) -> NoiseParams | None:
    """Return measured `NoiseParams` for a known CIDC25 file, else None."""
    from pathlib import Path

    key = Path(filename).name
    return FILE_NOISE.get(key)
