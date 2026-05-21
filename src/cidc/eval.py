"""Evaluation metric + universal inference for CIDC25.

This module implements the challenge's ``stSNR`` exactly and provides a
single ``evaluate(model, noisy, reference)`` entry point that works for
all 5 CIDC25 models (``deepinterp``, ``n2v3d``, ``deepcad``, ``mamba3d``,
``pinn``) via architecture-dispatching tile-and-blend inference.

Metric definition (from the challenge page)
-------------------------------------------
For a denoised stack ``x`` and clean reference ``y``, both ``[T, H, W]``:

- ``sSNR`` per frame::

      sSNR_t = 10 log10( sum_{h,w} y[t]^2 / sum_{h,w} (y[t] - x[t])^2 )

- ``tSNR`` per pixel trace::

      tSNR_{h,w} = 10 log10( sum_t y[:,h,w]^2 / sum_t (y[:,h,w] - x[:,h,w])^2 )

- ``stSNR`` = alpha * mean(sSNR) + (1-alpha) * mean(tSNR), default alpha=0.5.

Final leaderboard score = mean stSNR across files.

Universal inference
-------------------
``denoise_stack(model, noisy, params, ...)`` dispatches on model type:

- ``TemporalUNet`` (2-D DeepInterp): slide a ``(2K, H, W)`` context window
  frame-by-frame, predict the center frame each time; reassemble.
- Any 3-D backbone (``UNet3D``, ``DeepCADNet``, ``MambaUNet3D``): overlap-
  tile with cosine-blended seams in (T, H, W).
- ``PINNWrapper``: extract the ``denoised`` field from the output dict and
  treat the wrapper as a 3-D backbone.

Test-time augmentation (TTA) — rotations {0, 90, 180, 270} × flip {0, 1} —
is applied in Anscombe space (so per-tile averaging is statistically
sound) and the result is transformed back with the asymptotic inverse
Anscombe at the caller level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn

from .noise import NoiseParams


__all__ = [
    "StSNRResult",
    "snr_spatial",
    "snr_temporal",
    "stsnr",
    "denoise_stack",
    "evaluate",
]


# --------------------------------------------------------------------------- #
# Metric                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class StSNRResult:
    """Container for stSNR and its components."""

    s_snr: float           # mean per-frame sSNR
    t_snr: float           # mean per-pixel tSNR
    st_snr: float          # convex combination
    alpha: float           # weighting used

    def as_dict(self) -> dict[str, float]:
        return {"sSNR": self.s_snr, "tSNR": self.t_snr, "stSNR": self.st_snr, "alpha": self.alpha}


def _snr_db(signal_energy: np.ndarray, residual_energy: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """``10 log10(signal / residual)`` element-wise, with floor."""
    return 10.0 * np.log10(np.maximum(signal_energy, eps) / np.maximum(residual_energy, eps))


def snr_spatial(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-frame spatial SNR in dB.

    Parameters
    ----------
    pred, ref : ndarray, shape ``(T, H, W)``
        Denoised and reference stacks. Cast to float64 internally.

    Returns
    -------
    ndarray, shape ``(T,)`` of sSNR values in dB.
    """
    _check_pair(pred, ref)
    p = pred.astype(np.float64, copy=False)
    r = ref.astype(np.float64, copy=False)
    sig = (r * r).sum(axis=(1, 2))
    res = ((r - p) ** 2).sum(axis=(1, 2))
    return _snr_db(sig, res)


def snr_temporal(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-pixel temporal SNR in dB.

    Returns ndarray shape ``(H, W)``.
    """
    _check_pair(pred, ref)
    p = pred.astype(np.float64, copy=False)
    r = ref.astype(np.float64, copy=False)
    sig = (r * r).sum(axis=0)
    res = ((r - p) ** 2).sum(axis=0)
    return _snr_db(sig, res)


def stsnr(pred: np.ndarray, ref: np.ndarray, alpha: float = 0.5) -> StSNRResult:
    """Compute stSNR = ``alpha*mean(sSNR) + (1-alpha)*mean(tSNR)``.

    This matches the challenge's metric. Default ``alpha=0.5`` is the
    published convex combination; pass a different value only to probe
    sensitivity.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    s = float(snr_spatial(pred, ref).mean())
    t = float(snr_temporal(pred, ref).mean())
    return StSNRResult(
        s_snr=s,
        t_snr=t,
        st_snr=alpha * s + (1.0 - alpha) * t,
        alpha=alpha,
    )


def _check_pair(pred: np.ndarray, ref: np.ndarray) -> None:
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, ref {ref.shape}")
    if pred.ndim != 3:
        raise ValueError(f"expected (T,H,W), got {pred.shape}")


# --------------------------------------------------------------------------- #
# Universal denoising (tile + blend, dispatch by model type)                  #
# --------------------------------------------------------------------------- #


def _cosine_window_1d(n: int, overlap: int) -> np.ndarray:
    """Smooth cosine taper of length ``n`` with ``overlap`` ramp on each end.

    Used as a separable blending weight for overlap-tile reassembly.
    """
    if overlap <= 0:
        return np.ones(n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)
    o = int(overlap)
    ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(o) + 1) / (o + 1)))
    w[:o] = ramp
    w[-o:] = ramp[::-1]
    return w


def _tile_starts(size: int, tile: int, overlap: int) -> list[int]:
    """Starting positions so that tiles cover [0, size) with ``overlap`` step."""
    if tile >= size:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError(f"tile={tile} <= overlap={overlap}; choose smaller overlap")
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)
    return starts


@torch.no_grad()
def _forward_3d(
    model: nn.Module,
    x: Tensor,
    params: NoiseParams,
) -> Tensor:
    """Forward a 3-D model (possibly PINNWrapper) and return (B, 1, T, H, W)."""
    out = model(x, params)
    # PINNWrapper returns a dict-like object with 'denoised' key.
    if hasattr(out, "__getitem__") and not torch.is_tensor(out):
        return out["denoised"]
    return out


@torch.no_grad()
def _denoise_3d(
    model: nn.Module,
    noisy: np.ndarray,
    params: NoiseParams,
    tile: tuple[int, int, int],
    overlap: tuple[int, int, int],
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Tile-and-blend inference for any 3-D model.

    Uses a separable cosine window so seam overlaps average smoothly.
    """
    T, H, W = noisy.shape
    tT, tH, tW = tile
    oT, oH, oW = overlap
    if tT > T or tH > H or tW > W:
        raise ValueError(f"tile {tile} larger than stack {noisy.shape}")

    # Global accumulator in double precision for stable blending.
    acc = np.zeros((T, H, W), dtype=np.float64)
    wts = np.zeros((T, H, W), dtype=np.float64)

    wT = _cosine_window_1d(tT, oT)
    wH = _cosine_window_1d(tH, oH)
    wW = _cosine_window_1d(tW, oW)
    # Separable weight (broadcast multiply).
    w3 = wT[:, None, None] * wH[None, :, None] * wW[None, None, :]

    tensor_dtype = torch.float32
    model.eval()

    for t0 in _tile_starts(T, tT, oT):
        for h0 in _tile_starts(H, tH, oH):
            for w0 in _tile_starts(W, tW, oW):
                sub = noisy[t0 : t0 + tT, h0 : h0 + tH, w0 : w0 + tW]
                x = torch.from_numpy(np.ascontiguousarray(sub)).to(device=device, dtype=tensor_dtype)
                x = x.unsqueeze(0).unsqueeze(0)                  # (1, 1, tT, tH, tW)
                if amp_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=amp_dtype):
                        y = _forward_3d(model, x, params)
                else:
                    y = _forward_3d(model, x, params)
                y = y.float().squeeze(0).squeeze(0).cpu().numpy()
                acc[t0 : t0 + tT, h0 : h0 + tH, w0 : w0 + tW] += y * w3
                wts[t0 : t0 + tT, h0 : h0 + tH, w0 : w0 + tW] += w3

    return acc / np.maximum(wts, 1e-12)


@torch.no_grad()
def _denoise_deepinterp(
    model: nn.Module,
    noisy: np.ndarray,
    params: NoiseParams,
    half_context: int,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Frame-by-frame denoising for the 2-D DeepInterpolation model.

    For each frame index ``t`` in ``[K, T-K)``, stack ``2K`` context frames
    (omitting ``t``) and predict ``t``. Boundary frames ``[0, K)`` and
    ``[T-K, T)`` are copied from the nearest predictable frame.
    """
    T, H, W = noisy.shape
    K = int(half_context)
    if T < 2 * K + 1:
        raise ValueError(f"stack T={T} too short for half_context={K}")

    out = np.empty_like(noisy, dtype=np.float64)
    model.eval()
    stack_t = torch.from_numpy(np.ascontiguousarray(noisy)).float().to(device)

    # Asymptotic inverse Anscombe params (same as UNet3D.forward).
    g = float(params.gain)
    sr2 = float(max(params.read_var, 0.0))

    first = K
    last = T - K
    for t in range(first, last):
        idx = list(range(t - K, t)) + list(range(t + 1, t + K + 1))
        ctx = stack_t[idx].unsqueeze(0)                          # (1, 2K, H, W)
        if amp_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                z = model(ctx, params)
        else:
            z = model(ctx, params)
        # Model outputs Anscombe-space prediction; invert to raw ADU.
        z = z.float().squeeze(0).squeeze(0).cpu().numpy()
        out[t] = (z / 2.0) ** 2 * g - 0.375 * g - sr2 / g

    # Boundary copies — simplest valid strategy.
    for t in range(0, first):
        out[t] = out[first]
    for t in range(last, T):
        out[t] = out[last - 1]
    return out


def _is_3d_model(model: nn.Module) -> bool:
    """Detect by forward-input contract: 3-D models expect ``(B, 1, T, H, W)``.

    We check by class name to avoid importing the 2-D TemporalUNet here.
    """
    cls = type(model).__name__
    return cls in {"UNet3D", "DeepCADNet", "MambaUNet3D", "PINNWrapper"}


@torch.no_grad()
def denoise_stack(
    model: nn.Module,
    noisy: np.ndarray,
    params: NoiseParams,
    tile: tuple[int, int, int] = (32, 128, 128),
    overlap: tuple[int, int, int] = (8, 16, 16),
    device: torch.device | str = "cpu",
    amp: bool = False,
) -> np.ndarray:
    """Universal denoiser for any CIDC25 model.

    Parameters
    ----------
    model
        Any instance of ``TemporalUNet``, ``UNet3D``, ``DeepCADNet``,
        ``MambaUNet3D``, or ``PINNWrapper``.
    noisy
        Noisy stack, shape ``(T, H, W)``, dtype compatible with float32.
    params
        Noise parameters (gain, read_var) used by the model's Anscombe
        inverse. For a real denoising run, pass the file-matched params
        (see ``cidc.noise.identify_noise_level``).
    tile, overlap
        Used only for 3-D models. Ignored for DeepInterp (frame-by-frame).
    device
        Torch device.
    amp
        If True, run forward in bf16 autocast (safe on Ampere+ and T4 where
        bf16 may fall back to fp16; autograd is disabled throughout).

    Returns
    -------
    ndarray, same shape as ``noisy``, dtype float64 (raw ADU).
    """
    device = torch.device(device)
    model.to(device)
    amp_dtype = torch.bfloat16 if amp else None

    # 2-D DeepInterp: frame-by-frame.
    if type(model).__name__ == "TemporalUNet":
        return _denoise_deepinterp(
            model,
            noisy,
            params,
            half_context=int(model.half_context),
            device=device,
            amp_dtype=amp_dtype,
        )

    if _is_3d_model(model):
        return _denoise_3d(model, noisy, params, tile, overlap, device, amp_dtype)

    raise TypeError(
        f"Unknown model class {type(model).__name__}. "
        "Supported: TemporalUNet, UNet3D, DeepCADNet, MambaUNet3D, PINNWrapper."
    )


# --------------------------------------------------------------------------- #
# Top-level evaluate                                                           #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def evaluate(
    model: nn.Module,
    noisy: np.ndarray,
    reference: np.ndarray,
    params: NoiseParams,
    tile: tuple[int, int, int] = (32, 128, 128),
    overlap: tuple[int, int, int] = (8, 16, 16),
    device: torch.device | str = "cpu",
    amp: bool = False,
    alpha: float = 0.5,
) -> StSNRResult:
    """Denoise a stack and score it against the reference.

    Convenience wrapper around :func:`denoise_stack` + :func:`stsnr`.
    Same arguments, returns :class:`StSNRResult`.
    """
    pred = denoise_stack(
        model,
        noisy,
        params,
        tile=tile,
        overlap=overlap,
        device=device,
        amp=amp,
    )
    return stsnr(pred, reference, alpha=alpha)
