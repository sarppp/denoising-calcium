"""3-D U-Net with a bi-directional Mamba (selective SSM) bottleneck.

Components
----------
- ``blocks.BiMambaBlock``  — forward + reverse Mamba with a residual add
  and LayerNorm, applied to a flattened ``(B, L, C)`` sequence.
- ``unet3d.MambaUNet3D``   — 3-D U-Net whose bottleneck's second conv block
  is replaced by a stack of ``BiMambaBlock`` layers operating on the
  ``(B, D·H·W, C)`` flattened bottleneck volume. Encoder / decoder reuse
  the ``n2v3d`` design (GroupNorm + SiLU, isotropic convs).

Rationale
---------
Our measured temporal autocorrelation (``notebooks/06_proofs.py §6``)
gives ``τ(0.5) ≈ 45 frames``. The *effective* temporal context for
denoising is ~60–100 frames, which is much longer than the receptive
field of a 3-level 3-D U-Net at typical patch sizes (~22 frames). A
selective state-space layer at the bottleneck lets us extend the
effective temporal support to the full patch length (default 32) without
adding 3-D convolutions at earlier, memory-heavy resolutions.

Requires ``mamba-ssm`` and ``causal-conv1d`` (CUDA). Import fails loudly
if those are missing — we never silently fall back to a slow pure-PyTorch
SSM (that would invalidate all benchmarking).
"""

from .blocks import BiMambaBlock
from .unet3d import MambaUNet3D

__all__ = ["MambaUNet3D", "BiMambaBlock"]
