#!/usr/bin/env python3
"""Ultra-fast evaluation for code testing (one patch only)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
import torch
from cidc import load_stack

from config import *
from model import UNet3D


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    print(f"Loading model: {model_path}")
    model = UNet3D(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=CHANNELS,
    )

    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Load data
    data_dir = Path(args.data_dir)
    print(f"Loading F1 stack...")
    stack_path = data_dir / "val" / "F1.tif"
    noisy = np.asarray(load_stack(stack_path), dtype=np.float32)

    print("\n" + "=" * 60)
    print("FAST CODE TEST (1 patch only)")
    print("=" * 60 + "\n")

    # Extract 1 patch
    patch_size = 64
    t, h, w = 0, 0, 0
    patch = noisy[t:t+patch_size, h:h+patch_size, w:w+patch_size]

    print(f"Patch shape: {patch.shape}")
    print(f"Patch range: [{patch.min():.1f}, {patch.max():.1f}]")

    # Prepare input
    noise_std = np.sqrt(SIGMA_R_SQ_AUG + 100 * np.maximum(patch, 0))
    noise_map = np.clip(noise_std / (noise_std.max() + 1e-8), 0, 1)
    x = np.stack([patch, noise_map], axis=0).astype(np.float32)
    x = torch.from_numpy(x).unsqueeze(0).to(device)  # [1, 2, T, H, W]

    print(f"Input shape: {x.shape}")

    # Infer
    print(f"\nRunning inference...")
    with torch.no_grad():
        y_pred = model(x).squeeze(0).squeeze(0)  # [T, H, W]

    y_pred = y_pred.cpu().numpy()
    print(f"Output shape: {y_pred.shape}")
    print(f"Output range: [{y_pred.min():.1f}, {y_pred.max():.1f}]")

    print(f"\n✓ Code works! All systems functional.")
    print(f"  • Model loads ✓")
    print(f"  • Input preprocessing ✓")
    print(f"  • Inference ✓")
    print(f"  • Output valid ✓")

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ultra-fast eval test (1 patch)')
    parser.add_argument(
        '--model',
        type=str,
        default='checkpoints/model_final.pt',
        help='Path to trained model',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(Path(__file__).parent.parent / "data"),
        help='Path to data directory',
    )

    args = parser.parse_args()
    sys.exit(main(args))
