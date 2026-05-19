"""Training pipeline for calcium imaging denoiser.

Architecture: U-Net 3D with N2V3D blind-spot masking
Loss: Poisson-Gaussian NLL with gain augmentation
Data: A1, B1, C2, D2 with gain augmentation over range [15, 1500]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging

from cidc import load_stack

# ============================================================================
# LOCKED PARAMETERS
# ============================================================================

NOISE_PARAMS = {
    'A1': {'g': 27.6,  'sigma_r_sq': 2490},
    'B1': {'g': 27.7,  'sigma_r_sq': 2490},
    'C2': {'g': 254.9, 'sigma_r_sq': 2700},
    'D2': {'g': 258.9, 'sigma_r_sq': 2700},
}

G_AUG_MIN = 15
G_AUG_MAX = 1500
SIGMA_R_SQ_AUG = 2700

PATCH_SIZE = (128, 128, 128)  # T, H, W
IN_CHANNELS = 2               # noisy + noise map
CHANNELS = [32, 64, 128]
MASK_RATIO = 0.005            # 0.5% voxels

BATCH_SIZE = 4
GRAD_CLIP = 1.0
LR = 1e-4
EPOCHS = 100

DATA_DIR = Path(__file__).parent / "data"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("TRAINING DENOISER WITH LOCKED PARAMETERS")
logger.info("="*80)
logger.info(f"Patch size: {PATCH_SIZE}")
logger.info(f"Batch size: {BATCH_SIZE}")
logger.info(f"Gain aug range: [{G_AUG_MIN}, {G_AUG_MAX}]")
logger.info(f"Mask ratio: {MASK_RATIO*100:.1f}%")

# ============================================================================
# DATASET
# ============================================================================

class PatchDataset(Dataset):
    """3D patch dataset with N2V3D blind-spot masking."""

    def __init__(self, stack_names, patch_size, n_patches_per_stack=1000):
        self.stack_names = stack_names
        self.patch_size = patch_size
        self.n_patches = n_patches_per_stack * len(stack_names)

        # Load all stacks
        self.stacks = {}
        for name in stack_names:
            level = 1 if name in ['A1', 'B1'] else 2
            path = DATA_DIR / "train" / f"{name}.tif"
            stack = np.asarray(load_stack(path), dtype=np.float32)
            self.stacks[name] = stack
            logger.info(f"Loaded {name}: {stack.shape}")

        self.stack_names_cycle = (stack_names * ((n_patches_per_stack // len(stack_names)) + 1))[:self.n_patches]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        stack_name = self.stack_names_cycle[idx]
        stack = self.stacks[stack_name]
        T, H, W = stack.shape
        Tp, Hp, Wp = self.patch_size

        # Random crop
        t = np.random.randint(0, T - Tp + 1)
        h = np.random.randint(0, H - Hp + 1)
        w = np.random.randint(0, W - Wp + 1)
        patch = stack[t:t+Tp, h:h+Hp, w:w+Wp].copy()

        # Random gain augmentation
        g = np.random.uniform(G_AUG_MIN, G_AUG_MAX)
        noisy = patch * (g / NOISE_PARAMS[stack_name]['g'])  # rescale to augmented gain

        # Add Poisson-Gaussian noise at augmented gain
        signal_variance = noisy.copy()
        poisson_noise = np.random.poisson(signal_variance / g) * g - signal_variance
        gaussian_noise = np.random.normal(0, np.sqrt(SIGMA_R_SQ_AUG), noisy.shape)
        noisy = noisy + poisson_noise + gaussian_noise

        # Noise map (estimated std per voxel neighborhood)
        noise_std = np.sqrt(SIGMA_R_SQ_AUG + g * np.maximum(patch, 0))
        noise_map = np.clip(noise_std / noise_std.max(), 0, 1)  # Normalize to [0,1]

        # Stack input: [noisy, noise_map]
        x = np.stack([noisy, noise_map], axis=0).astype(np.float32)
        y = patch[np.newaxis, :, :, :].astype(np.float32)

        return x, y, g, SIGMA_R_SQ_AUG


class N2V3DMask:
    """N2V3D blind-spot masking."""

    def __init__(self, mask_ratio=0.005):
        self.mask_ratio = mask_ratio

    def __call__(self, patch_shape):
        """Returns mask [T, H, W] where 1=observe, 0=predict."""
        n_total = np.prod(patch_shape)
        n_masked = max(1, int(n_total * self.mask_ratio))

        mask = np.ones(patch_shape, dtype=bool)
        indices = np.random.choice(n_total, n_masked, replace=False)
        mask.flat[indices] = False

        return torch.from_numpy(mask).float()


# ============================================================================
# MODEL (Simple 3D U-Net)
# ============================================================================

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, channels=[32, 64, 128]):
        super().__init__()
        self.channels = channels

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, channels[0])
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(channels[0], channels[1])
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(channels[1], channels[2])

        # Bottleneck
        self.bottle = ConvBlock3D(channels[2], channels[2])

        # Decoder
        self.upconv2 = nn.ConvTranspose3d(channels[2], channels[1], 2, stride=2)
        self.dec2 = ConvBlock3D(channels[1] * 2, channels[1])
        self.upconv1 = nn.ConvTranspose3d(channels[1], channels[0], 2, stride=2)
        self.dec1 = ConvBlock3D(channels[0] * 2, channels[0])

        # Output
        self.out = nn.Conv3d(channels[0], out_channels, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)

        # Bottleneck
        b = self.bottle(e3)

        # Decoder
        up2 = self.upconv2(b)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        out = self.out(d1)
        return out


# ============================================================================
# LOSS (Poisson-Gaussian NLL)
# ============================================================================

class PGNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, g, sigma_r_sq, mask=None):
        """
        Poisson-Gaussian NLL loss.

        L = -log p(y | ŷ, g, σ_r²)
          = 0.5 * log(σ_r²) + 0.5 * (y - ŷ)² / σ_r²
            - y/g * log(ŷ/g + 1e-8) + ŷ/g

        Args:
            y_pred: Model output [B, 1, T, H, W]
            y_true: Ground truth [B, 1, T, H, W]
            g: Gain per batch [B]
            sigma_r_sq: Read noise variance [B]
            mask: N2V3D mask [B, T, H, W] (1=observe, 0=predict)
        """
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Ensure g, sigma_r_sq are tensors on same device
        g = torch.as_tensor(g, dtype=torch.float32, device=y_pred.device)
        sigma_r_sq = torch.as_tensor(sigma_r_sq, dtype=torch.float32, device=y_pred.device)

        # Reshape for broadcasting
        g = g.view(-1, 1, 1, 1, 1)
        sigma_r_sq = sigma_r_sq.view(-1, 1, 1, 1, 1)

        # Gaussian term: (y - ŷ)² / (2σ_r²)
        gaussian = 0.5 * (y_true - y_pred) ** 2 / (sigma_r_sq + 1e-8)

        # Poisson term: -y/g * log(ŷ/g) + ŷ/g
        # Simplified as: ŷ/g - (y/g) * log(ŷ/g + 1e-8)
        poisson = y_pred / (g + 1e-8) - (y_true / (g + 1e-8)) * torch.log(y_pred / (g + 1e-8) + 1e-8)

        loss = gaussian + poisson

        # Apply mask (only compute loss on observed voxels)
        if mask is not None:
            loss = loss * mask.unsqueeze(1)
            loss = loss.sum() / (mask.sum() + 1e-8)
        else:
            loss = loss.mean()

        return loss


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, loader, optimizer, scaler, loss_fn, device, mask_fn, grad_clip=1.0):
    model.train()
    total_loss = 0

    for batch_idx, (x, y, g, sigma_r_sq) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        # N2V3D masking
        mask = mask_fn(y.shape[2:])
        mask = mask.to(device)

        optimizer.zero_grad()

        # Mixed precision forward
        with autocast():
            y_pred = model(x)
            loss = loss_fn(y_pred, y, g, sigma_r_sq, mask)

        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch {batch_idx+1}: loss={loss.item():.4f}")

    return total_loss / len(loader)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    # Dataset
    dataset = PatchDataset(['A1', 'B1', 'C2', 'D2'], PATCH_SIZE, n_patches_per_stack=250)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    logger.info(f"Dataset: {len(dataset)} patches")

    # Model
    model = UNet3D(in_channels=IN_CHANNELS, out_channels=1, channels=CHANNELS)
    model = model.to(device)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Optimizer & loss
    optimizer = Adam(model.parameters(), lr=LR)
    scaler = GradScaler()
    loss_fn = PGNLLLoss()
    mask_fn = N2V3DMask(mask_ratio=MASK_RATIO)

    # Training
    logger.info(f"\n{'='*80}")
    logger.info("STARTING TRAINING")
    logger.info(f"{'='*80}\n")

    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, loader, optimizer, scaler, loss_fn, device, mask_fn, GRAD_CLIP)
        logger.info(f"Epoch {epoch+1}/{EPOCHS}: avg_loss={avg_loss:.4f}")

        # Checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt_path = CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")


if __name__ == '__main__':
    main()
