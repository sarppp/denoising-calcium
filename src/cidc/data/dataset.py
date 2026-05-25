"""Dataset classes that yield self-supervised training samples.

Design rationale
----------------
One raw-ADU window sampler (``CIDCStackDataset``) handles:

- memmap-backed TIFF access across multiple stacks,
- uniform sampling of ``(stack_idx, t0, h0, w0)`` top-left corners,
- per-sample log-uniform gain resampling (Task-2 OOD augmentation),
- spatial/temporal flips and 90° rotations,
- Anscombe variance-stabilisation on the way out.

Three thin model-specific subclasses convert a window into the
``(input, target, meta)`` structure each architecture expects.

Gain augmentation logic
-----------------------
We use the *noisy* window itself as the "clean" prior, because
CIDC25 training stacks (A1, B1, C2, D2) are noisy, not clean. To resample
noise at a fresh gain we:

1. Estimate a near-clean signal ``mu_hat`` = temporal mean of the window
   (√T reduction in noise; T=32 → ~5.6× SNR improvement).
2. Sample a fresh gain ``g ∈ [g_lo, g_hi]`` log-uniformly.
3. Draw a noisy realisation ``y = g * Poisson(mu_hat / g) + N(0, σ_r²)``
   via ``cidc.noise.sample_poisson_gaussian``.
4. Anscombe-transform and return.

This is an imperfect clean estimate (the mean still contains signal
correlated across frames) but it's the correct operational setup for a
self-supervised Task-2-robust model: the network is exposed to noise at
*any* gain covered by the prior, including the F3 operating point of
~991.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tifffile
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..noise import (
    FILE_NOISE,
    NoiseParams,
    anscombe,
    identify_noise_level,
    sample_poisson_gaussian,
)


__all__ = [
    "CIDCStackDataset",
    "DeepInterpDataset",
    "N2V3DDataset",
    "DeepCADDataset",
    "build_dataset",
]


# --------------------------------------------------------------------------- #
# Base: raw-ADU window sampler                                                #
# --------------------------------------------------------------------------- #


@dataclass
class _StackHandle:
    path: Path
    name: str
    params: NoiseParams
    shape: tuple[int, int, int]
    _memmap: np.memmap | None = None

    def array(self) -> np.memmap:
        if self._memmap is None:
            # tifffile.memmap returns a numpy memmap of shape (T,H,W), int16.
            self._memmap = tifffile.memmap(self.path)
        return self._memmap


class CIDCStackDataset(Dataset):
    """Raw-ADU window sampler. Subclass or wrap to produce (input, target)."""

    def __init__(
        self,
        stack_paths: Iterable[str | Path],
        patch: tuple[int, int, int] = (32, 128, 128),
        samples_per_epoch: int = 10_000,
        gain_aug_enabled: bool = True,
        gain_range: tuple[float, float] = (20.0, 2000.0),
        gain_aug_prob: float = 0.5,
        flip: bool = True,
        rot90: bool = True,
        temporal_reverse: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.patch = tuple(int(x) for x in patch)
        if len(self.patch) != 3:
            raise ValueError(f"patch must be (T, H, W); got {patch}")
        self.samples_per_epoch = int(samples_per_epoch)
        self.gain_aug_enabled = bool(gain_aug_enabled)
        self.gain_range = (float(gain_range[0]), float(gain_range[1]))
        self.gain_aug_prob = float(gain_aug_prob)
        self.flip = bool(flip)
        self.rot90 = bool(rot90)
        self.temporal_reverse = bool(temporal_reverse)
        self.seed = int(seed)

        self.stacks: list[_StackHandle] = []
        for p in stack_paths:
            path = Path(p)
            if not path.exists():
                raise FileNotFoundError(path)
            # tifffile.memmap is cheap; call once for shape.
            with tifffile.TiffFile(path) as tif:
                shp = (len(tif.pages), *tif.pages[0].shape)
            self.stacks.append(
                _StackHandle(
                    path=path,
                    name=path.stem,
                    params=identify_noise_level(path.name),
                    shape=shp,
                )
            )
        if not self.stacks:
            raise ValueError("no stacks provided")

    # ---- stdlib interface ------------------------------------------------- #

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a single raw-ADU window sample.

        Keys
        ----
        window : float32 ndarray, shape (T, H, W) in Anscombe space
        gain   : float  (the gain used; equals the stack gain unless aug fired)
        read_var : float
        stack  : str (stack name, for logging)
        """
        # Per-index deterministic RNG.
        rng = np.random.default_rng(self.seed + 1_000_003 * int(index))
        s = self.stacks[rng.integers(0, len(self.stacks))]
        arr = s.array()
        T, H, W = s.shape
        tT, tH, tW = self.patch
        if tT > T or tH > H or tW > W:
            raise ValueError(f"patch {self.patch} larger than stack {s.shape}")
        t0 = int(rng.integers(0, T - tT + 1))
        h0 = int(rng.integers(0, H - tH + 1))
        w0 = int(rng.integers(0, W - tW + 1))

        # Copy out as float32; memmap is int16.
        window = np.asarray(arr[t0 : t0 + tT, h0 : h0 + tH, w0 : w0 + tW], dtype=np.float32)

        # Gain augmentation: replace noise realisation at a fresh gain.
        params = s.params
        if self.gain_aug_enabled and rng.random() < self.gain_aug_prob:
            mu_hat = window.mean(axis=0, keepdims=True)          # (1, H, W)
            mu_hat = np.broadcast_to(mu_hat, window.shape).copy()
            g = float(np.exp(rng.uniform(np.log(self.gain_range[0]),
                                         np.log(self.gain_range[1]))))
            params = NoiseParams(gain=g, read_var=float(s.params.read_var))
            window = sample_poisson_gaussian(mu_hat, params, rng=rng)

        # Spatial flips + 90° rotations.
        if self.flip and rng.random() < 0.5:
            window = window[:, ::-1, :].copy()
        if self.flip and rng.random() < 0.5:
            window = window[:, :, ::-1].copy()
        if self.rot90:
            k = int(rng.integers(0, 4))
            if k:
                window = np.rot90(window, k=k, axes=(1, 2)).copy()
        if self.temporal_reverse and rng.random() < 0.5:
            window = window[::-1, :, :].copy()

        # Anscombe transform (unit-variance space).
        z = anscombe(window, params).astype(np.float32)

        return {
            "window_raw": window,
            "window": z,
            "gain": float(params.gain),
            "read_var": float(params.read_var),
            "stack": s.name,
        }


# --------------------------------------------------------------------------- #
# Model-specific wrappers                                                      #
# --------------------------------------------------------------------------- #


class _WrapperDataset(Dataset):
    """Shared boilerplate: hold a ``CIDCStackDataset`` and post-process items."""

    def __init__(self, base: CIDCStackDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)


class DeepInterpDataset(_WrapperDataset):
    """Yields ``(context: (2K, H, W), target: (1, H, W))`` in Anscombe space.

    The base patch length must be ``2K + 1``. The center frame is removed
    from the input and returned as the target.
    """

    def __init__(self, base: CIDCStackDataset, half_context: int) -> None:
        super().__init__(base)
        self.K = int(half_context)
        if base.patch[0] != 2 * self.K + 1:
            raise ValueError(
                f"DeepInterp requires patch[0] == 2*K+1={2 * self.K + 1}; "
                f"got {base.patch[0]}"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        z = item["window"]                                       # (2K+1, H, W)
        K = self.K
        idx = list(range(0, K)) + list(range(K + 1, 2 * K + 1))
        ctx = torch.from_numpy(np.ascontiguousarray(z[idx]))     # (2K, H, W)
        tgt = torch.from_numpy(np.ascontiguousarray(z[K : K + 1]))  # (1, H, W)
        item["input"] = ctx
        item["target"] = tgt
        return item


class N2V3DDataset(_WrapperDataset):
    """Yields ``(volume: (1, T, H, W), target: (1, T, H, W))`` — masking is
    applied on-GPU during the training step via ``stratified_blindspot``,
    so the dataset only needs to emit the raw volume twice (input/target
    are the same; the mask selects which voxels participate in the loss).
    """

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        z = item["window"]                                       # (T, H, W)
        vol = torch.from_numpy(np.ascontiguousarray(z)).unsqueeze(0)  # (1, T, H, W)
        item["input"] = vol
        item["target"] = vol.clone()
        return item


class DeepCADDataset(_WrapperDataset):
    """Yields ``(odd: (1, T, H, W), even: (1, T, H, W))`` — odd vs even
    frames of the base window, for temporal Noise2Noise.
    """

    def __init__(self, base: CIDCStackDataset) -> None:
        super().__init__(base)
        if base.patch[0] % 2 != 0:
            raise ValueError(
                f"DeepCAD requires even patch[0]; got {base.patch[0]}"
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.base[index]
        z = item["window"]                                       # (2T, H, W)
        zt = torch.from_numpy(np.ascontiguousarray(z))
        even = zt[0::2].unsqueeze(0)                             # (1, T, H, W)
        odd = zt[1::2].unsqueeze(0)
        item["input"] = odd
        item["target"] = even
        return item


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #


def build_dataset(
    cfg,
    stack_paths: Iterable[str | Path],
    samples_per_epoch: int = 10_000,
) -> _WrapperDataset:
    """Build the per-model dataset matching ``cfg.model.name``.

    ``cfg`` is a :class:`cidc.Config`. The dataset chosen depends on
    ``cfg.model.name``:

    - ``deepinterp``          → :class:`DeepInterpDataset`
    - ``n2v3d`` / ``mamba3d`` / ``pinn`` → :class:`N2V3DDataset`
    - ``deepcad``             → :class:`DeepCADDataset`
    """
    base = CIDCStackDataset(
        stack_paths=stack_paths,
        patch=cfg.data.patch,
        samples_per_epoch=samples_per_epoch,
        gain_aug_enabled=cfg.data.gain_aug.enabled,
        gain_range=cfg.data.gain_aug.log_uniform_range,
        gain_aug_prob=cfg.data.gain_aug.prob,
        flip=cfg.data.flip,
        rot90=cfg.data.rot90,
        temporal_reverse=cfg.data.temporal_reverse,
        seed=cfg.training.seed,
    )
    name = cfg.model.name
    if name == "deepinterp":
        K = int(cfg.model.kwargs.get("half_context", 6))
        return DeepInterpDataset(base, half_context=K)
    if name == "deepcad":
        return DeepCADDataset(base)
    if name in ("n2v3d", "mamba3d", "pinn"):
        return N2V3DDataset(base)
    raise ValueError(f"Unknown model name for dataset dispatch: {name!r}")
