"""Composable PINN wrapper for any CIDC25 backbone.

The wrapper instantiates a backbone from the registry and exposes a
forward method that returns *both*

- the denoiser output in raw ADU (same as the bare backbone), and
- the PINN head outputs ``(tau, baseline, source)``.

We tap the features *before* the backbone's final ``head`` conv. The
n2v3d / mamba3d / deepcad models all keep that convention; the 2-D
DeepInterp backbone tap is trickier (the features are 2-D, not 3-D) so
the wrapper rejects it with a clear error. The typical use is

    backbone = mamba3d                (3-D, T × H × W)
    PINN head on the 3-D feature map

which is the most useful setup: the head needs a temporal axis to
predict ``s(t)``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ...noise import NoiseParams
from ..deepcad.unet3d import DeepCADNet
from ..mamba3d.unet3d import MambaUNet3D
from ..n2v3d.unet3d import UNet3D
from .head import PINNHead


__all__ = ["PINNWrapper", "PINNOutput"]


class PINNOutput(dict):
    """Container with attribute access. Keys: denoised, tau, baseline, source, reconstruction."""

    def __getattr__(self, k):  # type: ignore[override]
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


# Model name -> class (for the backbone field inside PINN configs).
_BACKBONES_3D = {
    "n2v3d": UNet3D,
    "deepcad": DeepCADNet,
    "mamba3d": MambaUNet3D,
}


class PINNWrapper(nn.Module):
    """Backbone + per-pixel calcium-kinetics head.

    Parameters
    ----------
    backbone : dict
        ``{"name": <str>, "kwargs": {...}}``; ``name`` must be one of
        ``n2v3d``, ``deepcad``, ``mamba3d``. The 2-D ``deepinterp`` model
        is not supported (no T-axis on the tapped features).
    tau_range : (float, float)
        Clamp range for per-pixel τ, in frames.
    baseline_from : {"head", "median"}
        ``head``   — let the PINN head learn a per-pixel baseline.
        ``median`` — use the temporal median of the denoiser output.
    """

    def __init__(
        self,
        backbone: dict[str, Any],
        tau_range: tuple[float, float] = (5.0, 200.0),
        baseline_from: str = "head",
    ) -> None:
        super().__init__()
        if not isinstance(backbone, dict) or "name" not in backbone:
            raise ValueError("backbone must be a dict with a 'name' key")
        name = backbone["name"]
        kwargs = dict(backbone.get("kwargs", {}))
        if name not in _BACKBONES_3D:
            raise ValueError(
                f"PINN wrapper only supports 3-D backbones "
                f"({sorted(_BACKBONES_3D)}); got {name!r}"
            )
        self.backbone: nn.Module = _BACKBONES_3D[name](**kwargs)

        # Steal the final 1x1 conv so we can recover pre-head features.
        # Every 3-D backbone in this repo exposes `.head` as the last layer.
        if not hasattr(self.backbone, "head") or not isinstance(
            self.backbone.head, nn.Conv3d
        ):
            raise TypeError(
                f"Backbone {type(self.backbone).__name__} must expose "
                "an ``nn.Conv3d`` attribute named ``head`` for PINN tapping."
            )
        self._head_conv: nn.Conv3d = self.backbone.head

        # Feature channel count equals head's input channels.
        feat_ch = self._head_conv.in_channels
        self.pinn_head = PINNHead(
            in_ch=feat_ch,
            tau_range=tau_range,
            baseline_from=baseline_from,
        )

        # Cache for the feature tap.
        self._features: Tensor | None = None
        self._hook = self._head_conv.register_forward_pre_hook(self._capture_features)

    # ----- internal helpers ------------------------------------------------ #

    def _capture_features(self, _module, inputs):
        # inputs is a 1-tuple: (features,)
        self._features = inputs[0]

    # ----- forward --------------------------------------------------------- #

    def forward(
        self,
        x_anscombe: Tensor,
        params: NoiseParams,
        gain_tensor: Tensor | None = None,
    ) -> PINNOutput:
        """Run backbone + PINN head.

        Parameters
        ----------
        x_anscombe
            ``(B, 1, T, H, W)`` Anscombe-space input (blind-spot masked).
        params
            Batch-median noise params (scalar fallback).
        gain_tensor
            Optional ``(B, 1, 1, 1, 1)`` per-sample gain.  Forwarded to the
            backbone so each sample's Anscombe inverse uses its correct gain.

        Returns a PINNOutput with::

            denoised       : (B, 1, T, H, W)  — raw ADU prediction from the backbone
            tau            : (B, 1, 1, H, W)  — per-pixel decay constant
            baseline       : (B, 1, 1, H, W)  — per-pixel DC baseline
            source         : (B, 1, T, H, W)  — per-pixel per-frame source (≥0)
            reconstruction : (B, 1, T, H, W)  — ODE reconstruction F_pinn(t)

        The caller can then compute the PINN aux loss via
        :func:`cidc.losses.calcium_kinetics_loss`.
        """
        self._features = None
        denoised = self.backbone(x_anscombe, params, gain_tensor=gain_tensor)  # (B, 1, T, H, W)

        if self._features is None:
            raise RuntimeError("backbone forward did not trigger the feature-tap hook")
        features = self._features
        self._features = None

        tau, baseline, source = self.pinn_head(features, denoised=denoised)

        from .kinetics import euler_forward
        reconstruction = euler_forward(tau, baseline, source, dt=1.0)

        return PINNOutput(
            denoised=denoised,
            tau=tau,
            baseline=baseline,
            source=source,
            reconstruction=reconstruction,
        )

    def extra_repr(self) -> str:
        return f"backbone={type(self.backbone).__name__}"
