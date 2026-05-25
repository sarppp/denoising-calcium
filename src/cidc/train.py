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
import tifffile
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

    KNOWN LIMITATION: we use index 0's gain as the batch-level scalar because
    the model forward (UNet3D, DeepCAD, Mamba, PINN) applies a single
    Anscombe inverse using one NoiseParams.  With per-sample gain augmentation
    this introduces a systematic bias for samples 1..B-1.

    Mitigation: the dataset applies gain augmentation with prob=0.5 (see
    DataConfig.gain_aug.prob), so ~50 % of samples share the un-augmented
    gain that is closest to the batch median.  Full fix requires refactoring
    the model forward to accept a (B,) gain tensor.
    """
    gains = batch["gain"]
    if isinstance(gains, torch.Tensor) and gains.numel() > 1:
        # Use batch-median gain to minimise worst-case Anscombe-inverse error.
        gain_val = float(gains.median().item())
    else:
        gain_val = float(gains[index].item() if hasattr(gains, "__getitem__") else gains)
    return NoiseParams(
        gain=gain_val,
        read_var=float(batch["read_var"][index].item()),
    )


def _simple_loss(name: str, pred: Tensor, tgt: Tensor,
                 gain: float, read_var: float, var_floor: float,
                 huber_delta: float = 1.0) -> Tensor:
    """Dispatch among the five supported primary losses.

    All losses receive ``pred`` and ``tgt`` in **raw ADU space**.

    Supported names (cfg.loss.name):
      poisson_gaussian_nll  — heteroscedastic Gaussian NLL (default, optimal
                              when noise model is correct; R²≥0.9 required)
      anscombe_mse          — MSE after forward Anscombe transform (variance-
                              stabilised; principled even when R² is poor,
                              does NOT assume the noise model is exactly right)
      mse                   — plain MSE in raw ADU (simple baseline)
      mae                   — MAE in raw ADU (L1; robust to heavy Poisson
                              tails, targets the conditional median)
      huber                 — Huber loss with δ=huber_delta; MSE near zero,
                              MAE in the tails (adaptive, outlier-resistant)

    Notes
    -----
    ``anscombe_mse`` was previously broken — it computed plain MSE because
    pred/tgt are passed in raw ADU.  Fixed: we now apply the forward
    Anscombe transform z = (2/g)·√(g·y + 3/8·g² + σ²) inside this branch
    so the residuals are in the unit-variance stabilised domain.
    """
    if name == "mse":
        return ((pred - tgt) ** 2).mean()
    if name == "mae":
        return (pred - tgt).abs().mean()
    if name == "huber":
        return F.huber_loss(pred, tgt, delta=huber_delta, reduction="mean")
    if name == "anscombe_mse":
        # Forward Anscombe transform: z = (2/g) * sqrt(g*y + 3/8*g² + σ²).
        # Stabilises Poisson-Gaussian noise to unit variance, making the MSE
        # loss distribution-agnostic even when the fitted R² is low.
        g = gain
        sr2 = read_var
        inside_pred = (g * pred + 0.375 * g * g + sr2).clamp(min=1e-6)
        inside_tgt = (g * tgt + 0.375 * g * g + sr2).clamp(min=1e-6)
        pred_a = (2.0 / g) * inside_pred.sqrt()
        tgt_a = (2.0 / g) * inside_tgt.sqrt()
        return ((pred_a - tgt_a) ** 2).mean()
    # Default: poisson_gaussian_nll
    return poisson_gaussian_nll(pred, tgt, gain, read_var, var_floor=var_floor)


def step_deepinterp(model: nn.Module, batch: dict[str, Any], cfg) -> Tensor:
    """Predict masked center frame from ±K Anscombe-context frames."""
    ctx = batch["input"].to(model_device(model))                # (B, 2K, H, W)
    tgt_anscombe = batch["target"].to(model_device(model))      # (B, 1, H, W)
    params = _make_params(batch, 0)
    pred_adu = model(ctx, params)                               # (B, 1, H, W)
    # Target is Anscombe-space; invert to raw ADU so both pred and target
    # share the same space regardless of which loss is selected.
    tgt_raw = (tgt_anscombe / 2.0).pow(2) * params.gain - 0.375 * params.gain - params.read_var / params.gain
    return _simple_loss(cfg.loss.name, pred_adu, tgt_raw,
                        params.gain, params.read_var, cfg.loss.var_floor,
                        huber_delta=cfg.loss.huber_delta)


def step_n2v3d(model: nn.Module, batch: dict[str, Any], cfg) -> Tensor:
    """Blind-spot masking in Anscombe space; primary loss on masked positions."""
    vol = batch["input"].to(model_device(model))                # (B, 1, T, H, W)
    params = _make_params(batch, 0)
    masked, mask = stratified_blindspot(vol, mask_fraction=0.005)
    pred_adu = model(masked, params)                            # (B, 1, T, H, W) raw ADU
    # Undo Anscombe on the original volume to get raw-ADU targets.
    tgt_raw = (vol / 2.0).pow(2) * params.gain - 0.375 * params.gain - params.read_var / params.gain
    # Select masked positions only — N2V self-supervised objective.
    pred_m = pred_adu[mask]
    tgt_m = tgt_raw[mask]
    return _simple_loss(cfg.loss.name, pred_m, tgt_m,
                        params.gain, params.read_var, cfg.loss.var_floor,
                        huber_delta=cfg.loss.huber_delta)


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
    return _simple_loss(cfg.loss.name, pred_adu, tgt_raw,
                        params.gain, params.read_var, cfg.loss.var_floor,
                        huber_delta=cfg.loss.huber_delta)


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
    primary = _simple_loss(cfg.loss.name, pred_m, tgt_m,
                           params.gain, params.read_var, cfg.loss.var_floor,
                           huber_delta=cfg.loss.huber_delta)

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
# Constants                                                                   #
# --------------------------------------------------------------------------- #

# Maximum number of non-finite (NaN / Inf) loss steps before the run is
# aborted early.  Matching UNSTABLE_NAN_LIMIT in scripts/ablation_verdict.py.
NAN_ABORT_LIMIT: int = 5


def _amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    """Return the correct AMP dtype for the given device.

    - Ampere+ (compute capability ≥ 8.0): bfloat16 — native hardware support,
      numerically stable, no loss scaling needed.
    - Turing / Volta (cc 7.x, e.g. T4, V100): float16 — the only dtype with
      hardware tensor-core acceleration on these GPUs.  Requires GradScaler.
    - CPU / AMP disabled: None.

    Using bfloat16 on a T4 (cc 7.5) causes autocast to fall back to fp32 for
    most ops → zero speedup.  This function prevents that mistake.
    """
    if not enabled or device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


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


def _probe(model: nn.Module, loader, step_fn, cfg, device: torch.device,
           opt, scaler, n_batches: int = 4) -> None:
    """Run N batches, assert loss is finite, then return.

    Catches data-path errors, shape mismatches, NaN losses, and CUDA OOM
    before committing to a long training run.  Raises ``RuntimeError`` on
    any failure so the caller can surface it cleanly.
    """
    model.train()
    losses: list[float] = []
    amp_dtype = _amp_dtype(device, cfg.training.amp)
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        opt.zero_grad(set_to_none=True)
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = step_fn(model, batch, cfg)
        else:
            loss = step_fn(model, batch, cfg)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Probe failed: non-finite loss ({loss.item()}) at batch {i + 1}. "
                "Check data paths, noise params, and loss function."
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        losses.append(float(loss.item()))
        print(f"[probe] batch {i + 1}/{n_batches}  loss={loss.item():.4f}", flush=True)
    if len(losses) >= 2 and losses[-1] >= losses[0] * 1.5:
        print(f"[probe] WARNING: loss not decreasing ({losses[0]:.4f} → {losses[-1]:.4f}). "
              "Check LR and data.", flush=True)
    else:
        print(f"[probe] OK  {losses[0]:.4f} → {losses[-1]:.4f}", flush=True)


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    ema: EMA,
    opt: AdamW,
    sched,
    scaler,
    cfg,
    val_stsnr: float,
    best_val: float,
    bad_epochs: int,
    global_step: int,
) -> None:
    """Atomically write a checkpoint (write to .tmp then rename).

    Atomic rename prevents a half-written file from corrupting a resume.
    All fields needed to fully resume training are included.
    """
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "ema": ema.shadow,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "cfg": cfg.to_dict(),
        # Resume state
        "val_stsnr": val_stsnr,
        "best_val": best_val,
        "bad_epochs": bad_epochs,
        "global_step": global_step,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(ckpt, tmp)
    tmp.rename(path)          # atomic on POSIX; safe on Windows with same filesystem


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    ema: EMA,
    opt: AdamW,
    sched,
    scaler,
    device: torch.device,
) -> dict:
    """Load a checkpoint and restore all stateful objects in-place.

    Returns the raw checkpoint dict so the caller can extract scalars
    (epoch, best_val, bad_epochs, global_step, val_stsnr).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if "ema" in ckpt and ckpt["ema"]:
        ema.shadow.update(ckpt["ema"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def train(cfg, data_root: Path, out_dir: Path,
          probe_only: bool = False, no_resume: bool = False) -> None:
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
    # steps_per_epoch = optimizer steps (not raw batch count); sched.step() is
    # called every grad_accum batches so we must divide here to get the right T_0.
    opt = build_optimizer(model, cfg)
    opt_steps_per_epoch = max(1, len(train_loader) // cfg.training.grad_accum)
    sched = build_scheduler(opt, cfg, steps_per_epoch=opt_steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.amp and device.type == "cuda")
    amp_dtype = _amp_dtype(device, cfg.training.amp)
    ema = EMA(model, decay=cfg.training.ema_decay)

    step_fn = STEP_REGISTRY[cfg.model.name]

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = -math.inf
    bad_epochs  = 0
    global_step = 0
    resumed     = False

    last_ckpt = out_dir / "last.pt"
    if not no_resume and last_ckpt.exists():
        print(f"\n[resume] Found checkpoint: {last_ckpt}", flush=True)
        ckpt = _load_checkpoint(last_ckpt, model, ema, opt, sched, scaler, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val    = float(ckpt.get("best_val",   ckpt.get("val_stsnr", -math.inf)))
        bad_epochs  = int(ckpt.get("bad_epochs",   0))
        global_step = int(ckpt.get("global_step",  0))
        resumed     = True
        log.log(kind="resumed", from_epoch=start_epoch - 1, best_val=best_val,
                bad_epochs=bad_epochs, global_step=global_step)
        print(f"[resume] Resuming from epoch {start_epoch}  "
              f"(best_val={best_val:+.3f}, bad_epochs={bad_epochs})", flush=True)
        if start_epoch >= cfg.training.epochs:
            print(f"[resume] Already completed {cfg.training.epochs} epochs. "
                  "Use --no-resume to re-train from scratch.", flush=True)
            log.log(kind="already-done", epochs=cfg.training.epochs)
            log.close()
            return

    # ── Probe (skip when resuming — pipeline was already validated) ───────────
    if not resumed:
        print("\n[probe] Running 4-batch sanity check …", flush=True)
        _probe(model, train_loader, step_fn, cfg, device, opt, scaler, n_batches=4)
        log.log(kind="probe-ok", loss_name=cfg.loss.name)
        if probe_only:
            print("[probe] --probe-only: exiting after successful probe.", flush=True)
            log.log(kind="probe-only-exit")
            log.close()
            return
    else:
        print("[probe] Skipped (resuming from checkpoint).", flush=True)

    log_every = cfg.training.log_every
    nan_step_count = 0   # cumulative across all epochs; abort at NAN_ABORT_LIMIT

    for epoch in range(start_epoch, cfg.training.epochs):
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

            # ── NaN / Inf guard ───────────────────────────────────────────────
            # Check BEFORE dividing by grad_accum and BEFORE backward so we
            # never propagate non-finite gradients into the model weights.
            if not torch.isfinite(loss):
                nan_step_count += 1
                log.log(
                    kind="nan-step",
                    epoch=epoch,
                    step=global_step,
                    loss_name=cfg.loss.name,
                    nan_count=nan_step_count,
                    loss=float(loss.item()),
                )
                opt.zero_grad(set_to_none=True)   # discard any partial gradient
                if nan_step_count >= NAN_ABORT_LIMIT:
                    log.log(
                        kind="nan-abort",
                        epoch=epoch,
                        step=global_step,
                        loss_name=cfg.loss.name,
                        nan_count=nan_step_count,
                        msg=(
                            f"{nan_step_count} non-finite loss steps exceeded "
                            f"NAN_ABORT_LIMIT={NAN_ABORT_LIMIT}. "
                            "Run aborted — try a more stable loss (anscombe_mse or mae)."
                        ),
                    )
                    log.close()
                    return
                continue   # skip backward for this batch, move to next

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

        # ── Save last.pt unconditionally after every epoch ────────────────────
        # This ensures a crash at any point can be resumed from the last
        # completed epoch, even when validation is skipped (data not found).
        val_metric = float("-inf")   # updated below if validation runs
        _save_checkpoint(
            out_dir / "last.pt",
            epoch, model, ema, opt, sched, scaler, cfg,
            val_stsnr=val_metric,
            best_val=best_val,
            bad_epochs=bad_epochs,
            global_step=global_step,
        )
        if (epoch + 1) % cfg.training.ckpt_every == 0:
            _save_checkpoint(
                out_dir / f"epoch_{epoch + 1:04d}.pt",
                epoch, model, ema, opt, sched, scaler, cfg,
                val_stsnr=val_metric,
                best_val=best_val,
                bad_epochs=bad_epochs,
                global_step=global_step,
            )
            log.log(kind="checkpoint", epoch=epoch, path=f"epoch_{epoch + 1:04d}.pt")

        # ── Validation with EMA weights ───────────────────────────────────────
        val_dir = data_root.parent / "val"
        ref_path = val_dir / f"{cfg.data.ref_stack}.tif"
        if not ref_path.exists():
            log.log(kind="val-skip", reason=f"ref_stack {cfg.data.ref_stack!r} not found at {ref_path}")
            continue
        ref = np.asarray(tifffile.memmap(ref_path), dtype=np.float32)
        val_paths = [val_dir / f"{name}.tif" for name in cfg.data.val_stacks]
        backup = ema.copy_to(model)
        try:
            model.eval()
            # Evaluate each val stack (cheap subset: first 128 frames).
            scores = []
            for vp in val_paths:
                if not vp.exists():
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
                        tta_rotations=cfg.inference.tta.rotations,
                        tta_flips=cfg.inference.tta.flips,
                    )
                scores.append(r.st_snr)
                log.log(kind="val", epoch=epoch, file=vp.stem,
                        sSNR=r.s_snr, tSNR=r.t_snr, stSNR=r.st_snr, wall_sec=t.dt)
            val_metric = float(np.mean(scores)) if scores else float("-inf")
        finally:
            model.load_state_dict({**model.state_dict(), **backup})

        # ── Update last.pt with actual val_metric, then early stop ───────────
        if val_metric > best_val:
            best_val = val_metric
            bad_epochs = 0
            _save_checkpoint(
                out_dir / "best.pt",
                epoch, model, ema, opt, sched, scaler, cfg,
                val_stsnr=val_metric,
                best_val=best_val,
                bad_epochs=bad_epochs,
                global_step=global_step,
            )
            log.log(kind="best", epoch=epoch, stSNR=val_metric)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.training.early_stop.patience:
                log.log(kind="early-stop", bad_epochs=bad_epochs, best_stSNR=best_val)
                break

        # Overwrite last.pt with the updated bad_epochs / best_val so resume
        # restores the correct early-stopping state.
        _save_checkpoint(
            out_dir / "last.pt",
            epoch, model, ema, opt, sched, scaler, cfg,
            val_stsnr=val_metric,
            best_val=best_val,
            bad_epochs=bad_epochs,
            global_step=global_step,
        )

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
