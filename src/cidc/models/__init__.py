"""Neural denoisers for CIDC25.

Each architecture lives in its own subpackage (one folder per model):

- ``deepinterp`` — DeepInterpolation-style temporal U-Net (2-D).
- ``n2v3d``      — Noise2Void 3-D blind-spot denoiser.
- ``deepcad``    — DeepCAD: 3-D U-Net + temporal Noise2Noise.
- ``mamba3d``    — 3-D U-Net with bi-directional Mamba bottleneck.
- ``pinn``       — Backbone + per-pixel calcium-kinetics head.

Use :func:`build_model` to instantiate any of them from a
:class:`cidc.config.ModelConfig` (typically loaded from YAML).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from . import deepcad, deepinterp, mamba3d, n2v3d, pinn

if TYPE_CHECKING:
    from ..config import ModelConfig


__all__ = ["deepcad", "deepinterp", "mamba3d", "n2v3d", "pinn", "build_model"]


_REGISTRY = {
    "deepinterp": deepinterp.TemporalUNet,
    "n2v3d":      n2v3d.UNet3D,
    "deepcad":    deepcad.DeepCADNet,
    "mamba3d":    mamba3d.MambaUNet3D,
    "pinn":       pinn.PINNWrapper,
}


def build_model(config: "ModelConfig") -> nn.Module:
    """Instantiate a model from a :class:`cidc.config.ModelConfig`.

    Dispatches on ``config.name``. ``config.kwargs`` is passed through to
    the constructor verbatim.

    Raises
    ------
    ValueError
        If ``config.name`` is not registered.
    """
    name = config.name
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model name {name!r}. Registered: {sorted(_REGISTRY)}."
        )
    cls = _REGISTRY[name]
    return cls(**config.kwargs)
