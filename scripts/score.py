"""Standalone stSNR scorer — run trained checkpoint on val stacks.

Usage
-----
    # Score all val stacks (F1, F2, F3) against F0 in one command:
    uv run python scripts/score.py \
        --config configs/n2v3d.yaml \
        --ckpt   runs/n2v3d_v1/best.pt \
        --data   data/train

    # Score a single noisy stack:
    uv run python scripts/score.py \
        --config configs/n2v3d.yaml \
        --ckpt   runs/n2v3d_v1/best.pt \
        --noisy  data/train/F1.tif \
        --ref    data/train/F0.tif

    # Score a pre-saved denoised .tif (metric only, no model needed):
    uv run python scripts/score.py path/to/denoised.tif path/to/F0.tif

TTA is read from cfg.inference.tta (rotations, flips) automatically.
Use --no-tta to disable it regardless of config.
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


def _load_model(config_path: Path, ckpt_path: Path, device: torch.device):
    cfg = load_config(config_path)
    model = build_model(cfg.model).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Prefer EMA weights at inference — trained with EMA, eval with EMA.
    if isinstance(state, dict) and "ema" in state and state["ema"]:
        full = model.state_dict()
        full.update(state["ema"])
        model.load_state_dict(full)
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model, cfg


def score_pair(pred_path: Path, ref_path: Path, alpha: float) -> None:
    """Mode 1: pre-saved denoised .tif vs reference."""
    pred = _load(pred_path)
    ref = _load(ref_path)
    r = stsnr(pred, ref, alpha=alpha)
    print(json.dumps(r.as_dict(), indent=2))


def score_one(
    model,
    cfg,
    noisy_path: Path,
    ref_path: Path,
    device: torch.device,
    alpha: float,
    no_tta: bool,
) -> dict:
    """Denoise one stack and score it. Returns result dict."""
    noisy = _load(noisy_path)
    ref   = _load(ref_path)
    params = identify_noise_level(noisy_path.name)

    tta_rotations = 1 if no_tta else cfg.inference.tta.rotations
    tta_flips     = False if no_tta else cfg.inference.tta.flips

    pred = denoise_stack(
        model,
        noisy,
        params,
        tile=tuple(cfg.inference.tile),
        overlap=tuple(cfg.inference.overlap),
        device=device,
        amp=cfg.training.amp,
        tta_rotations=tta_rotations,
        tta_flips=tta_flips,
    )
    r = stsnr(pred, ref, alpha=alpha)
    return {"file": noisy_path.name, "ref": ref_path.name, **r.as_dict()}


def main() -> None:
    p = argparse.ArgumentParser(description="CIDC25 inference + stSNR scorer")
    # Mode 1 — metric only
    p.add_argument("pred", nargs="?", type=Path, help="denoised .tif (mode 1)")
    p.add_argument("ref_pos", nargs="?", type=Path, help="reference .tif (mode 1)")
    # Mode 2 — checkpoint inference
    p.add_argument("--config", type=Path, help="YAML config")
    p.add_argument("--ckpt",   type=Path, help="checkpoint .pt  (best.pt recommended)")
    p.add_argument("--noisy",  type=Path, help="single noisy stack to score")
    p.add_argument("--ref",    type=Path, help="reference stack (F0.tif)")
    p.add_argument("--data",   type=Path,
                   help="data/train dir — scores F1, F2, F3 vs F0 all at once")
    p.add_argument("--alpha",  type=float, default=0.5)
    p.add_argument("--no-tta", action="store_true",
                   help="disable TTA regardless of config (faster, for quick checks)")
    args = p.parse_args()

    # ── Mode 1: metric only ────────────────────────────────────────────────────
    if args.config is None:
        if args.pred is None or args.ref_pos is None:
            p.error("provide `pred ref` (mode 1) OR --config --ckpt (mode 2)")
        score_pair(args.pred, args.ref_pos, args.alpha)
        return

    # ── Mode 2: checkpoint inference ──────────────────────────────────────────
    if args.ckpt is None:
        p.error("--config requires --ckpt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, cfg = _load_model(args.config, args.ckpt, device)

    tta_info = "off (--no-tta)" if args.no_tta else \
               f"rotations={cfg.inference.tta.rotations}, flips={cfg.inference.tta.flips}"
    print(f"TTA: {tta_info}")
    print()

    if args.data is not None:
        # Score all val stacks at once
        data_dir = args.data
        ref_path = data_dir / "F0.tif"
        if not ref_path.exists():
            p.error(f"F0.tif not found in {data_dir}")

        val_stacks = ["F1", "F2", "F3"]
        results = []
        for name in val_stacks:
            noisy_path = data_dir / f"{name}.tif"
            if not noisy_path.exists():
                print(f"  {name}: not found, skipping")
                continue
            print(f"  Scoring {name} ...", end=" ", flush=True)
            r = score_one(model, cfg, noisy_path, ref_path, device, args.alpha, args.no_tta)
            results.append(r)
            print(f"stSNR={r['stSNR']:+.2f} dB  (sSNR={r['sSNR']:+.2f}, tSNR={r['tSNR']:+.2f})")

        if results:
            print()
            print(f"{'Stack':<8} {'sSNR':>8} {'tSNR':>8} {'stSNR':>8}")
            print("-" * 36)
            for r in results:
                print(f"{r['file']:<8} {r['sSNR']:>+8.2f} {r['tSNR']:>+8.2f} {r['stSNR']:>+8.2f}")
            mean_st = sum(r["stSNR"] for r in results) / len(results)
            print("-" * 36)
            print(f"{'mean':<8} {'':>8} {'':>8} {mean_st:>+8.2f}")

    elif args.noisy is not None:
        ref_path = args.ref
        if ref_path is None:
            p.error("--noisy requires --ref")
        r = score_one(model, cfg, args.noisy, ref_path, device, args.alpha, args.no_tta)
        print(json.dumps(r, indent=2))
    else:
        p.error("provide --data (all val stacks) OR --noisy --ref (single stack)")


if __name__ == "__main__":
    main()
