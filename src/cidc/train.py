"""Universal CIDC25 training loop driven by a YAML config.

Usage
-----
    uv run python workspace/scripts/train.py configs/n2v3d.yaml \
        --data /app/workspace/data/train \
        --out  /app/workspace/runs/n2v3d_v1

The same script handles all five architectures by dispatching the
per-model training step. The dispatch table is at the bottom of the
file; add new models by registering a ``step_fn(model, batch, cfg)``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from . import NOISE_LEVELS, FILE_NOISE
from .config import Config, load_config
from .data import build_dataset
from .eval import evaluate
from .logging import RunLogger, Timer, format_bytes, format_duration
from .losses import (
    anscombe_mse,
    calcium_kinetics_loss,
    poisson_gaussian_nll,
)
from .models import build_model
from .models.n2v3d.mask import stratified_blindspot
from .noise import NoiseParams


# --------------------------------------------------------------------------- #
# Per-model training steps                                                    #
# --------------------------------------------------------------------------- #


def _make_params(batch: dict[str, Any], index: int = 0) -> NoiseParams:
    """Build a NoiseParams from the (augmented) gain/read_var in the batch.

    We use the first element's gain as the batch-level value because the
    dataset applies a single gain aug per sample but all Anscombe inverses
    in a minibatch share one scalar. If you want strict per-sample gains,
    refactor the forward to accept a (B,) gain tensor.
    """
    return NoiseParams(
        gain=float(batch["gain"][index].item()),
        read_var=float(batch["read_var"][index].item()),
    )


def step_deepinterp(model: nn.Module, batch: dict[str, Any], cfg) -> Tensor:
    """Predict masked center frame from ±K Anscombe-context frames."""
    ctx = batch["input"].to(model_device(model))                # (B, 2K, H, W)
    tgt_anscombe = batch["target"].to(model_device(model))      # (B, 1, H, W)
    params = _make_params(batch, 0)
    pred_adu = model(ctx, params)                               # (B, 1, H, W)
    # Target is Anscombe-space; invert to raw ADU so both pred and target
    # share the same space regardless of which loss is selected.
    tgt_raw = (tgt_anscombe / 2.0).pow(2) * params.gain - 0.375 * params.gain - params.read_var / params.gain
    if cfg.loss.name == "anscombe_mse":
        return ((pred_adu - tgt_raw) ** 2).mean()
    return poisson_gaussian_nll(pred_adu, tgt_raw, params.gain, params.read_var, var_floor=cfg.loss.var_floor)


def step_n2v3d(model: nn.Module, batch: dict[str, Any], cfg) -> Tensor:
    """Blind-spot masking in Anscombe space; PG-NLL on masked positions."""
    vol = batch["input"].to(model_device(model))                # (B, 1, T, H, W)
    params = _make_params(batch, 0)
    masked, mask = stratified_blindspot(vol, mask_fraction=0.005)
    pred_adu = model(masked, params)                            # (B, 1, T, H, W)
    # Undo Anscombe on the original volume to get raw-ADU targets.
    tgt_raw = (vol / 2.0).pow(2) * params.gain - 0.375 * params.gain - params.read_var / params.gain
    # Select masked positions only.
    pred_m = pred_adu[mask]
    tgt_m = tgt_raw[mask]
    return poisson_gaussian_nll(pred_m, tgt_m, params.gain, params.read_var, var_floor=cfg.loss.var_floor)


def step_mamba3d(model: nn.Module, batch: dict[str, Any], cfg) -> Tensor:
    """Same recipe as N2V 3-D — the Mamba bottleneck changes the backbone, not the signal."""
    return step_n2v3d(model, batch, cfg)


def step_deepcad(model: nn.Module, batch: dict[str, Any], cfg) -> Tensor:
    """Temporal Noise2Noise: odd halves as input, even halves as target."""
    odd = batch["input"].to(model_device(model))                # (B, 1, T, H, W) Anscombe
    even = batch["target"].to(model_device(model))              # (B, 1, T, H, W) Anscombe
    params = _make_params(batch, 0)
    pred_adu = model(odd, params)                               # raw ADU
    tgt_raw = (even / 2.0).pow(2) * params.gain - 0.375 * params.gain - params.read_var / params.gain
    return poisson_gaussian_nll(pred_adu, tgt_raw, params.gain, params.read_var, var_floor=cfg.loss.var_floor)


def step_pinn(model: nn.Module, batch: dict[str, Any], cfg) -> Tensor:
    """N2V-style primary loss + calcium-kinetics aux loss on the output trace."""
    vol = batch["input"].to(model_device(model))
    params = _make_params(batch, 0)
    masked, mask = stratified_blindspot(vol, mask_fraction=0.005)
    out = model(masked, params)                                 # PINNOutput

    # Primary loss (N2V on denoised output, raw ADU).
    tgt_raw = (vol / 2.0).pow(2) * params.gain - 0.375 * params.gain - params.read_var / params.gain
    pred_m = out.denoised[mask]
    tgt_m = tgt_raw[mask]
    primary = poisson_gaussian_nll(pred_m, tgt_m, params.gain, params.read_var, var_floor=cfg.loss.var_floor)

    # Aux loss (PINN kinetics).
    aux_cfg = cfg.loss.aux.get("pinn")
    if aux_cfg is not None and aux_cfg.enabled:
        aux = calcium_kinetics_loss(
            denoised=out.denoised,
            reconstruction=out.reconstruction,
            source=out.source,
            sparsity_l1=float(aux_cfg.extras.get("sparsity_l1", 0.005)),
            detach_input=bool(aux_cfg.extras.get("detach_input", False)),
        )
        return primary + float(aux_cfg.weight) * aux
    return primary


STEP_REGISTRY: dict[str, Callable[[nn.Module, dict[str, Any], Any], Tensor]] = {
    "deepinterp": step_deepinterp,
    "n2v3d":      step_n2v3d,
    "mamba3d":    step_mamba3d,
    "deepcad":    step_deepcad,
    "pinn":       step_pinn,
}


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


class EMA:
    """Exponential moving average of model parameters (decoupled from optimiser)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow and v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module) -> dict[str, Tensor]:
        """Swap model state to EMA, return the original for restoration."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict({**model.state_dict(), **self.shadow})
        return backup


def build_optimizer(model: nn.Module, cfg) -> AdamW:
    if cfg.training.optimizer.lower() != "adamw":
        raise NotImplementedError(f"only adamw is implemented; got {cfg.training.optimizer}")
    return AdamW(model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)


def build_scheduler(opt: AdamW, cfg, steps_per_epoch: int):
    """Cosine with warm restarts, ``training.restarts`` cycles over the full run."""
    if cfg.training.scheduler.lower() != "cosine_restarts":
        raise NotImplementedError(f"only cosine_restarts is implemented; got {cfg.training.scheduler}")
    total_steps = cfg.training.epochs * steps_per_epoch
    T_0 = max(1, total_steps // max(1, cfg.training.restarts))
    return CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=1)


# --------------------------------------------------------------------------- #
# Train loop                                                                   #
# --------------------------------------------------------------------------- #


def train(cfg, data_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log = RunLogger(out_dir, name=f"train_{cfg.model.name}", cuda=(device.type == "cuda"))
    log.log(kind="train-start", cfg_name=cfg.name, model=cfg.model.name,
            device=str(device), out=str(out_dir), data=str(data_root))

    # Seeds
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    # Build model
    model = build_model(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.log(kind="model", name=cfg.model.name, params_m=n_params / 1e6,
            patch=list(cfg.data.patch), batch=cfg.data.batch_size,
            grad_accum=cfg.training.grad_accum, amp=cfg.training.amp)

    # Data
    train_paths = [data_root / f"{name}.tif" for name in cfg.data.train_stacks]
    train_ds = build_dataset(cfg, train_paths, samples_per_epoch=cfg.data.samples_per_epoch)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimiser + schedule
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if cfg.training.amp else None
    ema = EMA(model, decay=cfg.training.ema_decay)

    step_fn = STEP_REGISTRY[cfg.model.name]

    best_val = -math.inf
    bad_epochs = 0
    global_step = 0
    log_every = cfg.training.log_every

    for epoch in range(cfg.training.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        count = 0
        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(train_loader):
            if amp_dtype is not None and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = step_fn(model, batch, cfg)
            else:
                loss = step_fn(model, batch, cfg)

            loss = loss / cfg.training.grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it + 1) % cfg.training.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                ema.update(model)
                global_step += 1
                if global_step % log_every == 0:
                    log.log(kind="step", epoch=epoch, step=global_step,
                            loss=running / max(1, count),
                            lr=opt.param_groups[0]["lr"])

            running += float(loss.item()) * cfg.training.grad_accum
            count += 1

        dt = time.time() - t0
        avg_loss = running / max(1, count)
        log.log(kind="epoch", epoch=epoch, train_loss=avg_loss, dt_sec=dt)

        # Validation with EMA weights.
        val_paths = [data_root.parent / "val" / f"{name}.tif" for name in cfg.data.val_stacks]
        ref_path = next((p for p in val_paths if p.stem == "F0"), None)
        if ref_path is None or not ref_path.exists():
            log.log(kind="val-skip", reason="F0 reference not available")
            continue
        import tifffile
        ref = np.asarray(tifffile.memmap(ref_path), dtype=np.float32)
        backup = ema.copy_to(model)
        try:
            model.eval()
            # Evaluate on F1/F2 (cheap subset: first 128 frames).
            scores = []
            for vp in val_paths:
                if vp.stem == "F0" or not vp.exists():
                    continue
                noisy = np.asarray(tifffile.memmap(vp), dtype=np.float32)[: min(128, ref.shape[0])]
                with Timer() as t:
                    r = evaluate(
                        model, noisy, ref[: noisy.shape[0]],
                        params=NOISE_LEVELS[int(vp.stem[-1])] if vp.stem[-1].isdigit() else FILE_NOISE[vp.name],
                        tile=cfg.inference.tile,
                        overlap=cfg.inference.overlap,
                        device=device,
                        amp=cfg.training.amp,
                    )
                scores.append(r.st_snr)
                log.log(kind="val", epoch=epoch, file=vp.stem,
                        sSNR=r.s_snr, tSNR=r.t_snr, stSNR=r.st_snr, wall_sec=t.dt)
            val_metric = float(np.mean(scores)) if scores else float("-inf")
        finally:
            model.load_state_dict({**model.state_dict(), **backup})

        # Checkpoint + early stopping.
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.shadow,
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "cfg": cfg.to_dict(),
            "val_stsnr": val_metric,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_metric > best_val:
            best_val = val_metric
            bad_epochs = 0
            torch.save(ckpt, out_dir / "best.pt")
            log.log(kind="best", epoch=epoch, stSNR=val_metric)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.training.early_stop.patience:
                log.log(kind="early-stop", bad_epochs=bad_epochs, best_stSNR=best_val)
                break
    log.log(kind="train-done", best_stSNR=best_val)
    log.close()


# --------------------------------------------------------------------------- #
# Optional standalone CLI (preserved; the main CLI is ``cidc train``).        #
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path, help="YAML config path")
    p.add_argument("--data", type=Path, required=True, help="train/val root dir")
    p.add_argument("--out", type=Path, required=True, help="output / checkpoint dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    train(cfg, args.data, args.out)


if __name__ == "__main__":
    main()
