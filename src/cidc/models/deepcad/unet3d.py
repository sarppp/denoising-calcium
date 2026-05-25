"""DeepCAD network: 3D U-Net trained with temporal Noise2Noise.

Reuses the ``n2v3d.UNet3D`` backbone unchanged; the N2N vs blind-spot
distinction lives in the *training loop* (pairing + loss), not in the
architecture. We wrap it here only to give DeepCAD its own importable
class name and a place to attach DeepCAD-specific defaults later (e.g.
temporal stride, output cropping) without polluting the N2V file.
"""

from __future__ import annotations

from torch import Tensor

from ..n2v3d.unet3d import UNet3D
from ...noise import NoiseParams


class DeepCADNet(UNet3D):
    """Temporal-Noise2Noise 3D U-Net for CIDC25.

    Same signature as ``UNet3D``. Default ``base_ch=16`` and ``depth=3``
    give a reasonable starting point on T4 memory; scale up once the
    training loop is stable.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 16,
        depth: int = 3,
        kernel: int = 3,
        pool: int = 2,
    ) -> None:
        super().__init__(in_ch=in_ch, base_ch=base_ch, depth=depth, kernel=kernel, pool=pool)

    def forward(  # noqa: D401
        self,
        x_anscombe: Tensor,
        params: NoiseParams,
        gain_tensor: Tensor | None = None,
    ) -> Tensor:
        """Same as ``UNet3D.forward``; passes ``gain_tensor`` through."""
        return super().forward(x_anscombe, params, gain_tensor=gain_tensor)
