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

from config import G_AUG_LOG_MIN, G_AUG_LOG_MAX, SIGMA_R_SQ_AUG, NOISE_PARAMS


class PatchDataset(Dataset):
    """3D patch dataset with LogUniform gain augmentation and N2V3D masking."""

    def __init__(
        self,
        stack_names: list[str],
        noise_params: dict,
        patch_size: tuple[int, int, int],
        g_aug_log_min: float,
        g_aug_log_max: float,
        sigma_r_sq_aug: float,
        data_dir: Path,
        n_patches_per_stack: int = 250,
    ):
        self.stack_names     = stack_names
        self.noise_params    = noise_params
        self.patch_size      = patch_size
        self.g_aug_log_min   = g_aug_log_min
        self.g_aug_log_max   = g_aug_log_max
        self.sigma_r_sq_aug  = sigma_r_sq_aug
        self.n_patches       = n_patches_per_stack * len(stack_names)

        self.stacks: dict[str, np.ndarray] = {}
        for name in stack_names:
            path  = data_dir / "train" / f"{name}.tif"
            stack = np.asarray(load_stack(path), dtype=np.float32)
            # Validate stack dimensions vs patch size
            T, H, W = stack.shape
            Tp, Hp, Wp = patch_size
            if T < Tp or H < Hp or W < Wp:
                raise ValueError(
                    f"Stack {name} shape {(T, H, W)} is smaller than patch size {patch_size}. "
                    f"Reduce --patch-size or use larger stacks."
                )
            self.stacks[name] = stack
            print(f"  Loaded {name}: {stack.shape}  "
                  f"g={noise_params[name]['g']}  σ_r²={noise_params[name]['sigma_r_sq']}")

        # Pre-compute index → stack name mapping (cycles evenly across stacks).
        reps = (self.n_patches + len(stack_names) - 1) // len(stack_names)
        self.stack_cycle = (stack_names * reps)[: self.n_patches]

    def __len__(self) -> int:
        return self.n_patches

    def __getitem__(self, idx: int) -> dict:
        stack_name = self.stack_cycle[idx]
        stack      = self.stacks[stack_name]
        T, H, W    = stack.shape
        Tp, Hp, Wp = self.patch_size
        g_original = self.noise_params[stack_name]['g']

        # ── Random crop ──────────────────────────────────────────────────────
        t = np.random.randint(0, T - Tp + 1)
        h = np.random.randint(0, H - Hp + 1)
        w = np.random.randint(0, W - Wp + 1)
        patch = stack[t:t+Tp, h:h+Hp, w:w+Wp].copy()
        patch = np.maximum(patch, 0.0)  # clip any background-subtraction artefacts

        # ── LogUniform gain augmentation ─────────────────────────────────────
        # Equal probability per decade: sample log(g) uniformly then exponentiate.
        # Linear uniform would heavily bias toward high-gain samples.
        g_aug = float(np.exp(np.random.uniform(self.g_aug_log_min, self.g_aug_log_max)))

        # Rescale patch to augmented gain level.
        rescaled = patch * (g_aug / g_original)

        # Add fresh Poisson-Gaussian noise at the augmented gain.
        signal_for_poisson = np.maximum(rescaled, 0.0)
        poisson_lambda = np.clip(signal_for_poisson / g_aug, 0, 6500)
        poisson_noise = (
            np.random.poisson(poisson_lambda).astype(np.float32) * g_aug
            - signal_for_poisson
        )
        gaussian_noise = np.random.normal(
            0.0, np.sqrt(self.sigma_r_sq_aug), rescaled.shape
        ).astype(np.float32)

        noisy = np.maximum(rescaled + poisson_noise + gaussian_noise, 0.0)

        # ── Noise map ─────────────────────────────────────────────────────────
        # σ_total² = σ_r² + g × signal  — evaluated at the *rescaled* signal,
        # not the original patch. Using the original was a bug: the noise map
        # would not match the actual noise level of the augmented observation.
        noise_std = np.sqrt(self.sigma_r_sq_aug + g_aug * signal_for_poisson)
        noise_map = noise_std / (noise_std.max() + 1e-8)  # normalise to [0,1]

        # ── Assemble tensors ──────────────────────────────────────────────────
        x = np.stack([noisy, noise_map], axis=0).astype(np.float32)  # [2, T, H, W]
        y = patch[np.newaxis].astype(np.float32)                      # [1, T, H, W]

        return {
            'x':          torch.from_numpy(x),
            'y':          torch.from_numpy(y),
            'g':          g_aug,
            'sigma_r_sq': self.sigma_r_sq_aug,
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

    def __call__(self, patch_shape: tuple[int, int, int]) -> torch.Tensor:
        """Return a float mask [T, H, W] where 0 = predict, 1 = observe."""
        n_total  = int(np.prod(patch_shape))
        n_masked = max(1, int(n_total * self.mask_ratio))

        mask = np.ones(patch_shape, dtype=np.float32)
        indices = np.random.choice(n_total, n_masked, replace=False)
        mask.flat[indices] = 0.0

        return torch.from_numpy(mask)
