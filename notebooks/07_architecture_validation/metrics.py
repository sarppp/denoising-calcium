"""Compute metrics for architecture validation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
from cidc import stsnr


def evaluate_architecture_on_validation(val_stacks: dict, reference: np.ndarray, 
                                       predictions: dict = None) -> dict:
    """Evaluate architecture performance on validation stacks."""
    if predictions is None:
        # Use raw stacks if no predictions provided
        predictions = val_stacks
    
    results = {}
    print(f"{'='*80}")
    print("ARCHITECTURE VALIDATION RESULTS")
    print(f"{'='*80}\n")
    print(f"{'Stack':<8} {'sSNR (dB)':<12} {'tSNR (dB)':<12} {'stSNR (dB)':<12}")
    print("-" * 50)
    
    for stack_name in ["F1", "F2", "F3"]:
        if stack_name in predictions:
            pred = predictions[stack_name]
            res = stsnr(pred, reference)
            results[stack_name] = {
                "ssnr": float(res.s_snr),
                "tsnr": float(res.t_snr),
                "stsnr": float(res.st_snr),
            }
            print(f"{stack_name:<8} {res.s_snr:<12.2f} {res.t_snr:<12.2f} {res.st_snr:<12.2f}")
    
    print()
    return results
