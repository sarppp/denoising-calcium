"""Analyze architecture validation results."""

import numpy as np


def summarize_performance(val_results: dict) -> None:
    """Summarize validation performance across stacks."""
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")
    
    stsnrs = [val_results[s]['stsnr'] for s in ["F1", "F2", "F3"] if s in val_results]
    
    if stsnrs:
        print(f"Mean stSNR across F1, F2, F3: {np.mean(stsnrs):.2f} dB")
        print(f"Std dev: {np.std(stsnrs):.2f} dB")
        print(f"Generalization gap (F3-F1): {val_results.get('F3', {}).get('stsnr', 0) - val_results.get('F1', {}).get('stsnr', 0):.2f} dB")
