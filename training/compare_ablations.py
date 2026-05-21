#!/usr/bin/env python3
"""Compare ablation runs (NLL vs MSE vs MAE) on val_F1 stSNR/sSNR/tSNR.

Usage:
    Run the three ablations (any order, any time):
        python train.py --stacks A1 B1 --loss nll --epochs 10 --run-name ablation_nll --no-resume
        python train.py --stacks A1 B1 --loss mse --epochs 10 --run-name ablation_mse --no-resume
        python train.py --stacks A1 B1 --loss mae --epochs 10 --run-name ablation_mae --no-resume

    Then compare:
        python compare_ablations.py

What to pay attention to:
    1. Winner = highest val_F1_stSNR at epoch 10.
    2. tSNR gap: if tSNR - sSNR > 1 → temporal bias (model over-smooths spatially).
       if tSNR - sSNR < -1 → spatial bias (model destroys temporal transients).
       A good model should be balanced (gap near 0).
    3. Loss curve: if still dropping linearly at epoch 10 → run 50-100 epochs for full training.
       if flattened by epoch 7-8 → 30-50 epochs is enough.
    4. NLL unstable (NaN, erratic)? → use MAE. A1/B1 noise model R² is poor (~0.27).
    5. MAE wins or ties NLL? → Poisson tails dominate, median-targeting loss is more robust.
"""

import json
from pathlib import Path

import pandas as pd


def main():
    runs_dir = Path(__file__).parent / "runs"
    ablations = ['ablation_nll', 'ablation_mse', 'ablation_mae']

    print("=" * 60)
    print("ABLATION COMPARISON")
    print("=" * 60)

    results = {}
    for run in ablations:
        paths = sorted(runs_dir.glob(f"*{run}*/metrics.jsonl"))
        if not paths:
            print(f"\n{run}: NOT FOUND (run not complete?)")
            continue

        df = pd.DataFrame([json.loads(l) for l in open(paths[-1])])
        last = df.iloc[-1]

        stsnr = last.get('val_F1_stSNR')
        ssnr = last.get('val_F1_sSNR')
        tsnr = last.get('val_F1_tSNR')
        loss_curve = [round(r['loss'], 3) for r in df.to_dict('records')]

        results[run] = {'stsnr': stsnr, 'ssnr': ssnr, 'tsnr': tsnr}

        print(f"\n{run}")
        if stsnr is not None:
            print(f"  stSNR={stsnr:.3f}  sSNR={ssnr:.3f}  tSNR={tsnr:.3f}")
        else:
            print(f"  No validation results at epoch 10")
        print(f"  loss curve: {loss_curve}")

    if len(results) >= 2:
        print("\n" + "=" * 60)
        print("WINNER")
        print("=" * 60)
        winner = max(results.items(), key=lambda x: x[1]['stsnr'])
        print(f"\n  {winner[0]}  stSNR={winner[1]['stsnr']:.3f}")

        # Check tSNR gap
        for name, r in results.items():
            gap = r['tsnr'] - r['ssnr']
            print(f"  {name}: tSNR-sSNR gap = {gap:+.3f}  "
                  f"({'temporal bias' if gap > 1 else 'spatial bias' if gap < -1 else 'balanced'})")


if __name__ == "__main__":
    main()
