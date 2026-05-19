#!/usr/bin/env python3
"""Plot training results from any checkpoint directory."""

import sys
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use('Agg')


def main():
    # Get checkpoint dir from argument or use default
    ckpt_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./checkpoints')

    # Load metadata
    metadata_path = ckpt_dir / 'metadata.json'
    if not metadata_path.exists():
        print(f"❌ No metadata found: {metadata_path}")
        return 1

    with open(metadata_path) as f:
        metadata = json.load(f)

    losses = metadata.get('all_losses', [])
    if not losses:
        print("❌ No losses in metadata")
        return 1

    # Create plots directory
    plot_dir = ckpt_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    print(f"Plotting {len(losses)} epochs from {ckpt_dir}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses)+1), losses, 'b-o', linewidth=2.5, markersize=6)
    plt.xlabel('Epoch', fontweight='bold', fontsize=12)
    plt.ylabel('Loss', fontweight='bold', fontsize=12)
    plt.title('Training Loss Curve', fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = plot_dir / 'loss.png'
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"✓ Saved: {loss_path}")

    # Save summary
    summary = f"""TRAINING SUMMARY
{'='*50}
Epochs completed: {len(losses)}
Final loss: {losses[-1]:.4f}
Initial loss: {losses[0]:.4f}
Loss reduction: {((losses[0]-losses[-1])/losses[0]*100):.1f}%

Training time: {metadata.get('elapsed_seconds', 0)/60:.1f} min
Batch size: {metadata.get('batch_size', 'N/A')}
Learning rate: {metadata.get('lr', 'N/A')}
Parameters: {metadata.get('n_parameters', 'N/A'):,}
"""
    summary_path = plot_dir / 'summary.txt'
    summary_path.write_text(summary)
    print(f"✓ Saved: {summary_path}")
    print(f"\n✓ All plots saved to: {plot_dir}\n")
    print(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main())
