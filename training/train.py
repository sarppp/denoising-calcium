#!/usr/bin/env python3
"""
Train 3D denoiser — gold-standard pipeline.

Every run produces a self-contained directory under training/runs/:
    run_YYYYMMDD_HHMMSS_<name>/
        config.json       full parameter snapshot
        metrics.jsonl     one JSON record per epoch (loss, val scores, lr)
        train.log         full text log
        checkpoints/
            model_epoch_NNN.pt
            model_best.pt     best combined val score
            model_final.pt    last epoch

Usage:
    python train.py                            # full training (100 epochs)
    python train.py --probe-only               # sanity-check pipeline, then exit
    python train.py --epochs 5 --run-name test
    python train.py --patch-size 64 64 64      # reduced patch for VRAM
    python train.py --resume runs/run_*/       # resume from existing experiment dir
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import json
import logging
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math

from config import (
    NOISE_PARAMS, BASELINE_STSNR,
    PATCH_SIZE, G_AUG_MIN, G_AUG_MAX, G_AUG_LOG_MIN, G_AUG_LOG_MAX,
    SIGMA_R_SQ_AUG, MASK_RATIO,
    IN_CHANNELS, OUT_CHANNELS, CHANNELS,
    BATCH_SIZE, LR, LR_MIN, WARMUP_EPOCHS, WEIGHT_DECAY, EPOCHS, GRAD_CLIP,
    N_PATCHES_PER_STACK, VAL_FREQ, CKPT_FREQ, ES_PATIENCE, ES_MIN_DELTA,
    TRAIN_STACKS, VAL_STACKS, DATA_DIR, RUNS_DIR,
)
from model import UNet3D
from loss import PGNLLLoss
from dataset import PatchDataset, N2V3DMask
from evaluate import evaluate, combined_score
from training_utils import (
    ExperimentDir, MetricsLogger, EarlyStopping,
    make_lr_scheduler, probe_training,
)


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_path: Path) -> logging.Logger:
    fmt = '%(asctime)s  %(levelname)-7s  %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path),
        ],
    )
    return logging.getLogger('train')


# ── N2V masking helper ────────────────────────────────────────────────────────

def apply_n2v_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace masked voxels (mask=0) with neighborhood mean.

    For N2V, the model must not see the target voxel value. We replace
    masked voxels in the noisy channel with the mean of their 3×3×3
    neighborhood (excluding the center). This prevents the model from
    trivially copying the input.

    Args:
        x: [B, C, T, H, W] input (channel 0 = noisy, channel 1 = noise_map)
        mask: [B, T, H, W] mask where 0 = predict (blind-spot), 1 = observe
    Returns:
        x_masked: same shape, with masked voxels replaced in noisy channel
    """
    B, C, T, H, W = x.shape
    mask_expanded = mask.unsqueeze(1)  # [B, 1, T, H, W]

    # Only mask the noisy channel (channel 0), keep noise_map (channel 1)
    noisy = x[:, 0:1]  # [B, 1, T, H, W]

    # Compute neighborhood mean using 3D conv with kernel that EXCLUDES center.
    # This is the correct N2V blind-spot: predict from neighbors only.
    kernel = torch.ones(1, 1, 3, 3, 3, device=x.device, dtype=x.dtype)
    kernel[0, 0, 1, 1, 1] = 0.0  # exclude center voxel
    kernel = kernel / kernel.sum()  # normalize to sum=1 (26 neighbors)
    neighbor_mean = F.conv3d(noisy, kernel, padding=1)  # [B, 1, T, H, W]

    # Replace masked voxels with neighborhood mean
    noisy_masked = noisy * mask_expanded + neighbor_mean * (1 - mask_expanded)

    # Reconstruct: masked noisy channel + original noise_map
    x_masked = x.clone()
    x_masked[:, 0:1] = noisy_masked
    return x_masked


# ── One training epoch ────────────────────────────────────────────────────────

def train_epoch(
    epoch:      int,
    model:      torch.nn.Module,
    loader:     DataLoader,
    optimizer:  torch.optim.Optimizer,
    loss_fn:    PGNLLLoss,
    mask_fn:    N2V3DMask,
    device:     torch.device,
    grad_clip:  float,
    log_freq:   int,
    logger:     logging.Logger,
) -> dict[str, float]:
    """Train one epoch. Returns loss stats dict."""
    model.train()

    total_loss      = 0.0
    n_batches       = 0
    stack_losses    = defaultdict(list)   # per-stack running totals

    for batch_idx, batch in enumerate(loader):
        x          = batch['x'].to(device)
        y          = batch['y'].to(device)
        g          = batch['g']
        sigma_r_sq = batch['sigma_r_sq']
        names      = batch['stack_name']  # list[str], length B

        # Guard: skip batches with corrupt inputs.
        if torch.isnan(x).any() or torch.isnan(y).any():
            logger.warning(f"  epoch {epoch+1} batch {batch_idx+1}: NaN in input — skipped")
            continue

        # N2V3D mask — independent mask per sample in the batch.
        # Each sample gets its own random blind-spot pattern so the model
        # cannot overfit to specific spatial coordinates.
        masks = [mask_fn(y.shape[2:]) for _ in range(y.shape[0])]  # list of [T, H, W]
        mask = torch.stack(masks, dim=0).to(device)                 # [B, T, H, W]

        # Apply N2V mask: replace masked voxels with neighborhood mean
        # so the model cannot trivially copy the target.
        x_masked = apply_n2v_mask(x, mask)

        optimizer.zero_grad()
        y_pred = model(x_masked)

        # Scalar loss for backward.
        loss = loss_fn(y_pred, y, g, sigma_r_sq, mask, reduction='mean')

        if not torch.isfinite(loss):
            logger.warning(f"  epoch {epoch+1} batch {batch_idx+1}: non-finite loss — skipped")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        # Per-stack loss (no_grad, re-use computed y_pred).
        with torch.no_grad():
            per_elem = loss_fn(y_pred, y, g, sigma_r_sq, mask, reduction='none')  # [B]
            for i, name in enumerate(names):
                stack_losses[name].append(per_elem[i].item())

        if (batch_idx + 1) % log_freq == 0:
            logger.info(f"  epoch {epoch+1}  batch {batch_idx+1}/{len(loader)}  "
                        f"loss={loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)

    stats = {'loss': avg_loss}
    for name, vals in stack_losses.items():
        stats[f'loss_{name}'] = float(np.mean(vals)) if vals else float('nan')
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args) -> int:
    # ── Experiment directory ──────────────────────────────────────────────────
    if args.resume:
        exp = ExperimentDir.__new__(ExperimentDir)
        exp.root  = Path(args.resume)
        exp.ckpts = exp.root / "checkpoints"
        exp.config_path  = exp.root / "config.json"
        exp.metrics_path = exp.root / "metrics.jsonl"
        exp.log_path     = exp.root / "train.log"
        print(f"Resuming experiment: {exp.root}")
    else:
        exp = ExperimentDir(RUNS_DIR, name=args.run_name)
        print(f"Experiment directory: {exp.root}")

    logger  = setup_logging(exp.log_path)
    metrics = MetricsLogger(exp.metrics_path)

    # ── Resolve parameters (CLI overrides config) ─────────────────────────────
    patch_size           = tuple(args.patch_size)   if args.patch_size   else PATCH_SIZE
    batch_size           = args.batch_size           or BATCH_SIZE
    epochs               = args.epochs               or EPOCHS
    lr                   = args.lr                   or LR
    lr_min               = args.lr_min               or LR_MIN
    warmup_epochs        = args.warmup_epochs        or WARMUP_EPOCHS
    weight_decay         = args.weight_decay         or WEIGHT_DECAY
    grad_clip            = args.grad_clip            or GRAD_CLIP
    n_patches_per_stack  = args.n_patches_per_stack  or N_PATCHES_PER_STACK
    mask_ratio           = args.mask_ratio           or MASK_RATIO
    channels             = tuple(args.channels) if args.channels else CHANNELS
    val_freq             = args.val_freq             or VAL_FREQ
    ckpt_freq            = args.ckpt_freq            or CKPT_FREQ
    es_patience          = args.es_patience          or ES_PATIENCE
    es_min_delta         = args.es_min_delta         or ES_MIN_DELTA
    train_stacks         = args.stacks               or TRAIN_STACKS
    data_dir             = Path(args.data_dir)

    # Gain augmentation range (CLI overrides config).
    g_aug_min = args.g_aug_min or G_AUG_MIN
    g_aug_max = args.g_aug_max or G_AUG_MAX
    g_aug_log_min = math.log(g_aug_min)
    g_aug_log_max = math.log(g_aug_max)

    # Reproducibility.
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Device.
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── Config snapshot ───────────────────────────────────────────────────────
    cfg = dict(
        loss=args.loss or 'nll',
        patch_size=patch_size, batch_size=batch_size, epochs=epochs,
        lr=lr, lr_min=lr_min, warmup_epochs=warmup_epochs,
        weight_decay=weight_decay, grad_clip=grad_clip,
        n_patches_per_stack=n_patches_per_stack, mask_ratio=mask_ratio,
        channels=channels, val_freq=val_freq, ckpt_freq=ckpt_freq,
        es_patience=es_patience, es_min_delta=es_min_delta,
        train_stacks=train_stacks, data_dir=str(data_dir),
        device=str(device), seed=args.seed,
        noise_params=NOISE_PARAMS, baseline_stsnr=BASELINE_STSNR,
    )
    if not args.resume:
        exp.save_config(cfg)

    # ── Log header ────────────────────────────────────────────────────────────
    logger.info("=" * 72)
    logger.info(f"TRAINING  {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Run dir:  {exp.root}")
    logger.info(f"Device:   {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU:      {props.name}  {props.total_memory/1e9:.1f} GB")
    logger.info(f"Config:   patch={patch_size}  bs={batch_size}  epochs={epochs}  "
                f"lr={lr}  wd={weight_decay}")
    logger.info(f"Augment:  LogUniform gain [{g_aug_min:.1f}, {g_aug_max:.1f}]  mask={mask_ratio:.3f}")
    logger.info("=" * 72)

    # ── Dataset ───────────────────────────────────────────────────────────────
    logger.info("Loading training stacks ...")
    dataset = PatchDataset(
        stack_names        = train_stacks,
        noise_params       = NOISE_PARAMS,
        patch_size         = patch_size,
        g_aug_log_min      = g_aug_log_min,
        g_aug_log_max      = g_aug_log_max,
        sigma_r_sq_aug     = SIGMA_R_SQ_AUG,
        data_dir           = data_dir,
        n_patches_per_stack= n_patches_per_stack,
    )
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
    )
    logger.info(f"Dataset: {len(dataset)} patches  {len(loader)} batches/epoch")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels  = IN_CHANNELS,
        out_channels = OUT_CHANNELS,
        channels     = list(channels),
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: UNet3D  channels={channels}  params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = make_lr_scheduler(optimizer, epochs, warmup_epochs, lr, lr_min)

    # ── Loss function ─────────────────────────────────────────────────────────
    # Three-way ablation on A1/B1 (poor R²) before committing to full training:
    #
    #   nll  — Poisson-Gaussian NLL. Principled when noise model fits (C2/D2 R²≈0.93).
    #           Risky for A1/B1 (R²≈0.27) and val stacks (R²<0.24).
    #   mse  — Mean squared error. Always stable. Suboptimal for Poisson noise
    #           (over-weights large residuals, under-weights small ones).
    #   mae  — Mean absolute error. Targets the median — more robust to heavy
    #           Poisson tails than MSE. Often competitive for shot-noise-dominated
    #           fluorescence data. nb04 measured it alongside MSE.
    #
    # Winner from ablation decides the full-training loss.
    # Hybrid fallback: use nll for C2/D2 batches, mae for A1/B1 batches.
    loss_name = (args.loss or 'nll').lower()

    class _SimpleWrapper(torch.nn.Module):
        """Wrap a voxel-wise loss to match PGNLLLoss signature."""
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, y_pred, y_true, g, sigma_r_sq, mask=None, reduction='mean'):
            elem = self.fn(y_pred, y_true)                   # [B, 1, T, H, W]
            if mask is not None:
                # Loss on masked voxels (mask=0) — same as PGNLLLoss
                masked = 1.0 - mask.unsqueeze(1)  # [B, 1, T, H, W]
                elem = elem * masked
            if reduction == 'none':
                if mask is not None:
                    n_masked = masked.sum(dim=(1, 2, 3, 4)).clamp(min=1)
                    return elem.sum(dim=(1, 2, 3, 4)) / n_masked  # [B]
                return elem.mean(dim=(1, 2, 3, 4))           # [B]
            if mask is not None:
                return elem.sum() / (masked.sum() + 1e-8)
            return elem.mean()

    if loss_name == 'mse':
        loss_fn = _SimpleWrapper(
            lambda yp, yt: (yp - yt).pow(2)
        )
    elif loss_name == 'mae':
        loss_fn = _SimpleWrapper(
            lambda yp, yt: (yp - yt).abs()
        )
    else:  # nll (default)
        loss_fn = PGNLLLoss()

    logger.info(f"Loss: {loss_name.upper()}")
    mask_fn   = N2V3DMask(mask_ratio=mask_ratio)
    early_stopping = EarlyStopping(patience=es_patience, min_delta=es_min_delta)

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    if not args.no_resume:
        existing = sorted(exp.ckpts.glob("model_epoch_*.pt"))
        if existing:
            ckpt_path = existing[-1]
            ckpt      = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            start_epoch = ckpt.get('epoch', 0)
            logger.info(f"Resumed from {ckpt_path.name}  (epoch {start_epoch})")

    if start_epoch >= epochs:
        logger.info(f"Already at epoch {start_epoch}/{epochs}. Use --no-resume to restart.")
        return 0

    # ── Probe ─────────────────────────────────────────────────────────────────
    # Always run a fast sanity check before committing to long training.
    # Catches data path errors, shape mismatches, and NaN loss immediately.
    logger.info("\nRunning probe (4 batches) ...")
    probe_training(model, loader, loss_fn, mask_fn, optimizer, device, n_batches=4, logger=logger)

    if args.probe_only:
        logger.info("--probe-only: exiting after successful probe.")
        return 0

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info(f"\nStarting training (epochs {start_epoch+1}–{epochs}) ...\n")
    best_score:     float | None = None
    session_start = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        # Train.
        train_stats = train_epoch(
            epoch, model, loader, optimizer, loss_fn, mask_fn,
            device, grad_clip, args.log_freq, logger,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        epoch_s    = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1:3d}/{epochs}  "
            f"loss={train_stats['loss']:.4f}  "
            f"[" + "  ".join(
                f"{k}={v:.3f}" for k, v in train_stats.items() if k.startswith('loss_')
            ) + f"]  lr={current_lr:.2e}  {epoch_s:.0f}s"
        )

        # Validation (fast, vectorised).
        val_results: dict | None = None
        score: float | None = None

        should_val  = (epoch + 1) % val_freq  == 0 or (epoch + 1) == epochs
        should_ckpt = (epoch + 1) % ckpt_freq == 0 or (epoch + 1) == epochs

        if should_val:
            logger.info(f"  Validating (fast mode) ...")
            val_results = evaluate(
                model, data_dir, device,
                fast       = True,
                patch_size = patch_size,
            )
            score = combined_score(val_results)
            logger.info(f"  Combined score: {score:+.2f} dB")

            # Best model.
            if best_score is None or score > best_score:
                best_score = score
                best_path  = exp.ckpts / "model_best.pt"
                torch.save({'model': model.state_dict(), 'epoch': epoch + 1,
                            'score': score}, best_path)
                logger.info(f"  Best model saved ({score:+.2f} dB)")

            if args.early_stopping:
                stop = early_stopping(score, model, epoch + 1, logger)
                if stop:
                    logger.info("Early stopping triggered.")
                    break

        # Checkpoint.
        if should_ckpt:
            ckpt_path = exp.ckpts / f"model_epoch_{epoch+1:03d}.pt"
            torch.save({
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch':     epoch + 1,
                'loss':      train_stats['loss'],
            }, ckpt_path)
            logger.info(f"  Checkpoint: {ckpt_path.name}")

        # Write metrics record.
        record: dict = {
            'epoch':    epoch + 1,
            'lr':       current_lr,
            'epoch_s':  round(epoch_s, 1),
            **train_stats,
        }
        if val_results:
            for name, res in val_results.items():
                record[f'val_{name}_stSNR'] = round(res['stSNR'], 4)
                record[f'val_{name}_sSNR']  = round(res['sSNR'],  4)
                record[f'val_{name}_tSNR']  = round(res['tSNR'],  4)
            record['val_combined'] = round(score, 4)
        metrics.log(record)

    # ── Finalise ──────────────────────────────────────────────────────────────
    elapsed = time.time() - session_start
    logger.info("=" * 72)
    logger.info(f"Training complete  ({elapsed/3600:.2f} h)")
    if best_score is not None:
        logger.info(f"Best combined val score: {best_score:+.2f} dB")

    final_path = exp.ckpts / "model_final.pt"
    torch.save({
        'model':     model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch':     epoch + 1,
        'loss':      train_stats['loss'],
        'elapsed_s': elapsed,
    }, final_path)
    logger.info(f"Final model: {final_path}")

    # Full evaluation at the end.
    logger.info("\nRunning full evaluation (all frames) ...")
    val_results = evaluate(model, data_dir, device, fast=False, patch_size=patch_size)
    score = combined_score(val_results)
    logger.info(f"Final combined score: {score:+.2f} dB")

    for name, baseline in BASELINE_STSNR.items():
        delta = val_results[name]['stSNR'] - baseline
        sign  = '+' if delta >= 0 else ''
        beat  = "BEAT" if delta > 0 else "BELOW"
        logger.info(f"  {name}: {val_results[name]['stSNR']:+.2f} dB  "
                    f"baseline={baseline:+.2f}  Δ={sign}{delta:.2f}  [{beat}]")

    metrics.log({'epoch': 'final', 'full_eval_combined': round(score, 4),
                 **{f'full_{k}_stSNR': round(v['stSNR'], 4)
                    for k, v in val_results.items()}})

    logger.info(f"\nAll outputs in: {exp.root}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Train 3D denoiser',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Loss function (ablation)
    p.add_argument('--loss', choices=['nll', 'mse', 'mae'], default='nll',
                   help='nll=Poisson-Gaussian NLL (default), mse=MSE baseline, mae=MAE (robust to Poisson tails)')

    # Run identity
    p.add_argument('--run-name',   default='run',  help='Short label appended to run directory name')
    p.add_argument('--resume',     default=None,   help='Path to existing run directory to resume')
    p.add_argument('--no-resume',  action='store_true', help='Ignore existing checkpoints, start fresh')
    p.add_argument('--probe-only', action='store_true', help='Run 4-batch probe then exit')

    # Reproducibility
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--device', default=None, help='cuda / cpu (auto-detected if omitted)')

    # Data
    p.add_argument('--data-dir', default=str(Path(__file__).parent.parent / "data"))
    p.add_argument('--stacks', nargs='+', default=None, help='Training stacks')
    p.add_argument('--n-patches-per-stack', type=int, default=None)
    p.add_argument('--g-aug-min', type=float, default=None)
    p.add_argument('--g-aug-max', type=float, default=None)
    p.add_argument('--mask-ratio', type=float, default=None)

    # Architecture
    p.add_argument('--patch-size', type=int, nargs=3, default=None, metavar=('T','H','W'))
    p.add_argument('--channels', type=int, nargs='+', default=None)

    # Optimisation
    p.add_argument('--epochs',         type=int,   default=None)
    p.add_argument('--batch-size',     type=int,   default=None)
    p.add_argument('--lr',             type=float, default=None)
    p.add_argument('--lr-min',         type=float, default=None)
    p.add_argument('--warmup-epochs',  type=int,   default=None)
    p.add_argument('--weight-decay',   type=float, default=None)
    p.add_argument('--grad-clip',      type=float, default=None)
    p.add_argument('--num-workers',    type=int,   default=4)

    # Checkpointing / validation
    p.add_argument('--val-freq',     type=int, default=None)
    p.add_argument('--ckpt-freq',    type=int, default=None)
    p.add_argument('--log-freq',     type=int, default=10)
    p.add_argument('--early-stopping',  action='store_true', default=True)
    p.add_argument('--no-early-stopping', dest='early_stopping', action='store_false')
    p.add_argument('--es-patience',  type=int,   default=None)
    p.add_argument('--es-min-delta', type=float, default=None)

    # Convenience presets
    p.add_argument('--quick', action='store_true',
                   help='2 epochs, 32³ patches, 10 patches/stack — end-to-end test')

    args = p.parse_args()

    if args.quick:
        args.epochs              = args.epochs or 2
        args.patch_size          = args.patch_size or [32, 32, 32]
        args.n_patches_per_stack = args.n_patches_per_stack or 10
        args.ckpt_freq           = 1
        args.val_freq            = 999
        args.run_name            = args.run_name or 'quick'

    raise SystemExit(main(args))
