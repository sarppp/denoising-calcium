"""
Dataset and N2V3D masking for training.

Two bugs fixed vs. the original:
  1. Gain augmentation is now LogUniform, not linear uniform.
     Linear uniform over [15, 1500] puts 99% of mass above g=150;
     LogUniform gives equal probability per decade.
  2. Noise map uses the gain-rescaled signal, not the original patch.
     The noise map must reflect variance at the *augmented* gain level.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import Dataset
from cidc import load_stack

from config import G_AUG_LOG_MIN, G_AUG_LOG_MAX, NOISE_PARAMS


class PatchDataset(Dataset):
    """3D patch dataset with LogUniform gain augmentation and N2V3D masking.

    Stacks are kept as tifffile memmaps on disk; only the extracted patch is
    materialised into RAM (as float32) in __getitem__.  This avoids holding
    ~5.6 GB of float32 arrays for the four training stacks.
    """

    def __init__(
        self,
        stack_names: list[str],
        noise_params: dict,
        patch_size: tuple[int, int, int],
        g_aug_log_min: float,
        g_aug_log_max: float,
        data_dir: Path,
        n_patches_per_stack: int = 250,
    ):
        self.stack_names     = stack_names
        self.noise_params    = noise_params
        self.patch_size      = patch_size
        self.g_aug_log_min   = g_aug_log_min
        self.g_aug_log_max   = g_aug_log_max
        self.n_patches       = n_patches_per_stack * len(stack_names)

        # Store paths and shapes only — stacks stay as memmaps on disk.
        self.stack_paths:  dict[str, Path]                = {}
        self.stack_shapes: dict[str, tuple[int, int, int]] = {}
        for name in stack_names:
            path  = data_dir / "train" / f"{name}.tif"
            mmap  = load_stack(path)                      # tifffile.memmap, not loaded
            shape = tuple(int(s) for s in mmap.shape)
            T, H, W = shape
            Tp, Hp, Wp = patch_size
            if T < Tp or H < Hp or W < Wp:
                raise ValueError(
                    f"Stack {name} shape {(T, H, W)} is smaller than patch size {patch_size}. "
                    f"Reduce --patch-size or use larger stacks."
                )
            self.stack_paths[name]  = path
            self.stack_shapes[name] = shape
            print(f"  Registered {name}: {shape}  "
                  f"g={noise_params[name]['g']}  σ_r²={noise_params[name]['sigma_r_sq']}")

        # Per-worker RNG (forked workers each get their own state).
        self.rng = np.random.default_rng()

        # Pre-compute index → stack name mapping (cycles evenly across stacks).
        reps = (self.n_patches + len(stack_names) - 1) // len(stack_names)
        self.stack_cycle = (stack_names * reps)[: self.n_patches]

    def __len__(self) -> int:
        return self.n_patches

    def __getitem__(self, idx: int) -> dict:
        stack_name = self.stack_cycle[idx]
        g_original = self.noise_params[stack_name]['g']
        sigma_r_sq = self.noise_params[stack_name]['sigma_r_sq']  # per-stack read noise

        # ── Lazy load: memmap slice → float32 (only this patch enters RAM) ───
        stack = load_stack(self.stack_paths[stack_name])
        T, H, W    = self.stack_shapes[stack_name]
        Tp, Hp, Wp = self.patch_size

        # ── Random crop (self.rng for worker-safe reproducibility) ───────────
        t = int(self.rng.integers(0, T - Tp + 1))
        h = int(self.rng.integers(0, H - Hp + 1))
        w = int(self.rng.integers(0, W - Wp + 1))
        patch = np.asarray(stack[t:t+Tp, h:h+Hp, w:w+Wp], dtype=np.float32)
        patch = np.maximum(patch, 0.0)  # clip any background-subtraction artefacts

        # ── LogUniform gain augmentation ─────────────────────────────────────
        # Equal probability per decade: sample log(g) uniformly then exponentiate.
        log_g = self.rng.uniform(self.g_aug_log_min, self.g_aug_log_max)
        g_aug = float(np.exp(log_g))

        # Rescale patch to augmented gain level.
        rescaled = patch * (g_aug / g_original)

        # Add fresh Poisson-Gaussian noise at the augmented gain.
        signal_for_poisson = np.maximum(rescaled, 0.0)
        poisson_lambda = np.clip(signal_for_poisson / g_aug, 0, 6500)
        poisson_noise = (
            self.rng.poisson(poisson_lambda).astype(np.float32) * g_aug
            - signal_for_poisson
        )
        gaussian_noise = self.rng.normal(
            0.0, np.sqrt(sigma_r_sq), rescaled.shape
        ).astype(np.float32)

        noisy = np.maximum(rescaled + poisson_noise + gaussian_noise, 0.0)

        # ── Noise map ─────────────────────────────────────────────────────────
        # σ_total² = σ_r² + g × signal  — evaluated at the *rescaled* signal,
        # not the original patch. Using the original was a bug: the noise map
        # would not match the actual noise level of the augmented observation.
        noise_std = np.sqrt(sigma_r_sq + g_aug * signal_for_poisson)
        # Use machine epsilon to avoid division-by-zero on all-zero patches;
        # 1e-8 was too large (produces ~1e8 noise map values on zero patches).
        noise_map = noise_std / (noise_std.max() + np.finfo(np.float32).eps)  # normalise to [0,1]

        # ── Assemble tensors ──────────────────────────────────────────────────
        x = np.stack([noisy, noise_map], axis=0).astype(np.float32)  # [2, T, H, W]
        # Target y = original patch (before gain augmentation), NOT rescaled.
        # This is intentional: the model learns to map noisy-augmented input
        # back to the original signal level. The PG NLL loss handles this
        # correctly since it compares prediction against y at the original gain.
        y = patch[np.newaxis].astype(np.float32)                      # [1, T, H, W]

        return {
            'x':          torch.from_numpy(x),
            'y':          torch.from_numpy(y),
            'g':          g_aug,
            'sigma_r_sq': sigma_r_sq,
            'stack_name': stack_name,
        }


class N2V3DMask:
    """N2V3D blind-spot masking — randomly zero-out a fraction of voxels.

    The model predicts masked voxels from their unmasked neighbours.
    Loss is computed only on masked (predict) voxels, forcing the network
    to learn the signal distribution without access to the target at that location.
    """

    def __init__(self, mask_ratio: float = 0.005):
        self.mask_ratio = mask_ratio
        self.rng = np.random.default_rng()

    def __call__(self, patch_shape: tuple[int, int, int]) -> torch.Tensor:
        """Return a float mask [T, H, W] where 0 = predict, 1 = observe."""
        n_total  = int(np.prod(patch_shape))
        n_masked = max(1, int(n_total * self.mask_ratio))

        # For very small patches, 1 voxel can far exceed mask_ratio.
        # Skip masking entirely in that case (model trains as a plain denoiser).
        if n_masked / n_total > self.mask_ratio * 2:
            return torch.ones(patch_shape, dtype=torch.float32)

        mask = np.ones(patch_shape, dtype=np.float32)
        indices = self.rng.choice(n_total, n_masked, replace=False)
        mask.flat[indices] = 0.0

        return torch.from_numpy(mask)
