#!/usr/bin/env python3
"""
Quick evaluation — fast validation on F1 only, for rapid iteration.

Uses the same vectorised SNR as evaluate.py but skips F2/F3 and subsamples
frames. Runs in < 5 min. Use evaluate.py for full submission evaluation.

Usage:
    python quick_eval.py --model runs/<run>/checkpoints/model_best.pt
    python quick_eval.py --model runs/<run>/checkpoints/model_best.pt --all-stacks
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time
import numpy as np
import torch
from cidc import load_stack

from config import IN_CHANNELS, OUT_CHANNELS, CHANNELS, PATCH_SIZE, BASELINE_STSNR
from model import UNet3D
from evaluate import evaluate, combined_score


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    model = UNet3D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS)
    ckpt  = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    model = model.to(device)

    ckpt_info = ""
    if isinstance(ckpt, dict):
        ep    = ckpt.get('epoch', '?')
        score = ckpt.get('score', None)
        ckpt_info = f"  epoch={ep}" + (f"  saved_score={score:+.2f}" if score else "")

    print(f"Loaded: {model_path}{ckpt_info}\n")

    data_dir = Path(args.data_dir)
    results  = evaluate(model, data_dir, device, fast=True, patch_size=tuple(args.patch_size))

    if not args.all_stacks:
        results = {'F1': results['F1']}

    print("=" * 60)
    for name, res in results.items():
        baseline = BASELINE_STSNR[name]
        delta    = res['stSNR'] - baseline
        beat     = "BEAT" if delta > 0 else "below"
        print(f"  {name}  stSNR={res['stSNR']:+.2f}  sSNR={res['sSNR']:+.2f}  "
              f"tSNR={res['tSNR']:+.2f}  Δbaseline={delta:+.2f}  [{beat}]")

    if args.all_stacks and len(results) == 3:
        score = combined_score(results)
        print(f"\n  Combined score: {score:+.2f} dB")

    print("\nFor full evaluation: python evaluate.py --model <path>")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Quick evaluation (fast, F1-only by default)')
    p.add_argument('--model',       default='checkpoints/model_final.pt')
    p.add_argument('--data-dir',    default=str(Path(__file__).parent.parent / "data"))
    p.add_argument('--patch-size',  type=int, nargs='+', default=list(PATCH_SIZE),
                        help='Patch size: single int or three ints (T H W)')
    p.add_argument('--all-stacks',  action='store_true', help='Also evaluate F2 and F3')
    args = p.parse_args()
    if len(args.patch_size) == 1:
        args.patch_size = (args.patch_size[0],) * 3
    elif len(args.patch_size) == 3:
        args.patch_size = tuple(args.patch_size)
    else:
        p.error("--patch-size accepts 1 or 3 integers")
    raise SystemExit(main(args))
