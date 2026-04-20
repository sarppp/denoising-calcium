"""Standalone stSNR scorer.

Usage
-----
    # Score a pre-denoised prediction against a clean reference.
    uv run python workspace/scripts/score.py \
        path/to/denoised.tif path/to/F0.tif

    # Or run a trained checkpoint on a noisy stack and score vs reference.
    uv run python workspace/scripts/score.py \
        --config configs/n2v3d.yaml \
        --ckpt   runs/n2v3d_v1/best.pt \
        --noisy  data/val/F1.tif \
        --ref    data/val/F0.tif

The first form is metric-only. The second form runs full tile-and-blend
inference using ``cidc.evaluate`` (same code the trainer uses for
validation), so online and offline scoring match exactly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
import torch

from cidc import (
    FILE_NOISE,
    NOISE_LEVELS,
    build_model,
    denoise_stack,
    load_config,
    stsnr,
)
from cidc.noise import identify_noise_level


def _load(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.memmap(path))
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3-D stack, got shape {arr.shape}")
    return arr.astype(np.float32)


def score_pair(pred_path: Path, ref_path: Path, alpha: float) -> None:
    pred = _load(pred_path)
    ref = _load(ref_path)
    r = stsnr(pred, ref, alpha=alpha)
    print(json.dumps(r.as_dict(), indent=2))


def score_ckpt(
    config_path: Path,
    ckpt_path: Path,
    noisy_path: Path,
    ref_path: Path,
    alpha: float,
) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer EMA weights at inference.
    if isinstance(state, dict) and "ema" in state and state["ema"]:
        full = model.state_dict()
        full.update(state["ema"])
        model.load_state_dict(full)
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)

    noisy = _load(noisy_path)
    ref = _load(ref_path)
    params = identify_noise_level(noisy_path.name)
    pred = denoise_stack(
        model,
        noisy,
        params,
        tile=cfg.inference.tile,
        overlap=cfg.inference.overlap,
        device=device,
        amp=cfg.training.amp,
    )
    r = stsnr(pred, ref, alpha=alpha)
    print(json.dumps(
        {"file": noisy_path.name, "ref": ref_path.name, **r.as_dict()},
        indent=2,
    ))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("pred", nargs="?", type=Path, help="denoised stack .tif (mode 1)")
    p.add_argument("ref", nargs="?", type=Path, help="clean reference stack .tif")
    p.add_argument("--config", type=Path, help="YAML config (mode 2)")
    p.add_argument("--ckpt",   type=Path, help="checkpoint .pt (mode 2)")
    p.add_argument("--noisy",  type=Path, help="noisy stack to denoise (mode 2)")
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    if args.config is not None:
        required = [args.ckpt, args.noisy, args.ref]
        if any(x is None for x in required):
            p.error("--config requires --ckpt, --noisy, --ref")
        score_ckpt(args.config, args.ckpt, args.noisy, args.ref, args.alpha)
    else:
        if args.pred is None or args.ref is None:
            p.error("provide `pred ref` (mode 1) OR --config/--ckpt/--noisy/--ref (mode 2)")
        score_pair(args.pred, args.ref, args.alpha)


if __name__ == "__main__":
    main()
