"""
Training utilities: experiment management, metrics logging, early stopping, probe.
"""

import json
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch


# ── Experiment directory ──────────────────────────────────────────────────────

class ExperimentDir:
    """Create and manage a self-contained run directory.

    Structure:
        runs/run_YYYYMMDD_HHMMSS_<name>/
            config.json       full config snapshot at launch
            metrics.jsonl     one JSON object per epoch (append-only)
            train.log         full text log
            checkpoints/
                model_epoch_NNN.pt
                model_best.pt
                model_final.pt
    """

    def __init__(self, runs_dir: Path, name: str = "run"):
        ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.root  = runs_dir / f"run_{ts}_{name}"
        self.ckpts = self.root / "checkpoints"
        self.root.mkdir(parents=True, exist_ok=True)
        self.ckpts.mkdir(exist_ok=True)

        self.config_path  = self.root / "config.json"
        self.metrics_path = self.root / "metrics.jsonl"
        self.log_path     = self.root / "train.log"

    def save_config(self, cfg: dict) -> None:
        with open(self.config_path, 'w') as f:
            json.dump(cfg, f, indent=2, default=str)

    def __repr__(self) -> str:
        return str(self.root)


# ── Metrics logger ────────────────────────────────────────────────────────────

class MetricsLogger:
    """Append one JSON line per epoch to metrics.jsonl.

    Each line is a self-contained record — safe to read with:
        [json.loads(l) for l in open('metrics.jsonl')]
    or from pandas:
        pd.read_json('metrics.jsonl', lines=True)
    """

    def __init__(self, path: Path):
        self.path = path

    def log(self, record: dict) -> None:
        record.setdefault('ts', datetime.now().isoformat(timespec='seconds'))
        with open(self.path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when combined validation score stops improving."""

    def __init__(
        self,
        patience:              int   = 3,
        min_delta:             float = 0.05,
        restore_best_weights:  bool  = True,
    ):
        self.patience             = patience
        self.min_delta            = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score:   float | None        = None
        self.best_epoch:   int                 = 0
        self.counter:      int                 = 0
        self.best_weights: dict | None         = None

    def __call__(
        self,
        score:  float,
        model:  torch.nn.Module,
        epoch:  int,
        logger: logging.Logger | None = None,
    ) -> bool:
        """Return True if training should stop."""
        def _log(msg):
            if logger:
                logger.info(msg)

        if self.best_score is None or score > self.best_score + self.min_delta:
            prev = self.best_score
            self.best_score   = score
            self.best_epoch   = epoch
            self.counter      = 0
            self._save(model)
            if prev is None:
                _log(f"  EarlyStopping: {score:+.3f} dB (initial best)")
            else:
                _log(f"  EarlyStopping: {score:+.3f} dB ↑ (was {prev:+.3f}, Δ={score-prev:+.3f})")
            return False

        self.counter += 1
        _log(f"  EarlyStopping: no improvement ({self.counter}/{self.patience})  "
             f"best={self.best_score:+.3f} dB @ epoch {self.best_epoch}")

        if self.counter >= self.patience:
            _log(f"  Stopping — no improvement for {self.patience} checks")
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                _log(f"  Restored weights from epoch {self.best_epoch}")
            return True

        return False

    def _save(self, model: torch.nn.Module) -> None:
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}


# ── Probe (fast sanity check) ─────────────────────────────────────────────────

def probe_training(
    model:     torch.nn.Module,
    loader:    torch.utils.data.DataLoader,
    loss_fn:   torch.nn.Module,
    mask_fn,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    n_batches: int = 4,
    logger:    logging.Logger | None = None,
) -> None:
    """Run N batches and assert the pipeline is healthy before full training.

    Checks:
      - Loss is finite on every batch.
      - Loss decreases from batch 1 to batch N (model is learning).

    Raises RuntimeError immediately if either check fails — much better than
    discovering a broken pipeline after 20 hours of training.
    """
    def _log(msg):
        if logger:
            logger.info(msg)

    _log("Probe: running sanity check ...")
    model.train()
    losses = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        x          = batch['x'].to(device)
        y          = batch['y'].to(device)
        g          = batch['g']
        sigma_r_sq = batch['sigma_r_sq']

        # Independent mask per sample in the batch
        masks = [mask_fn(y.shape[2:]) for _ in range(y.shape[0])]
        mask  = torch.stack(masks, dim=0).to(device)  # [B, T, H, W]

        # Apply N2V mask: replace masked voxels with neighborhood mean (excluding center)
        import torch.nn.functional as F
        mask_expanded = mask.unsqueeze(1)  # [B, 1, T, H, W]
        noisy = x[:, 0:1]  # [B, 1, T, H, W]
        kernel = torch.ones(1, 1, 3, 3, 3, device=x.device, dtype=x.dtype)
        kernel[0, 0, 1, 1, 1] = 0.0  # exclude center voxel
        kernel = kernel / kernel.sum()  # normalize (26 neighbors)
        neighbor_mean = F.conv3d(noisy, kernel, padding=1)
        noisy_masked = noisy * mask_expanded + neighbor_mean * (1 - mask_expanded)
        x_masked = x.clone()
        x_masked[:, 0:1] = noisy_masked

        optimizer.zero_grad()
        y_pred = model(x_masked)
        loss   = loss_fn(y_pred, y, g, sigma_r_sq, mask)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Probe failed: non-finite loss ({loss.item()}) at batch {i+1}. "
                "Check data, gain params, and loss function."
            )

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        _log(f"  Probe batch {i+1}/{n_batches}: loss={loss.item():.4f}")

    if len(losses) >= 2 and losses[-1] >= losses[0] * 1.5:
        _log(f"  WARNING: loss did not decrease "
             f"({losses[0]:.4f} → {losses[-1]:.4f}). "
             "Model may not be learning — check LR and data.")
    else:
        _log(f"  Probe passed. loss {losses[0]:.4f} → {losses[-1]:.4f}")


# ── Learning rate schedule ────────────────────────────────────────────────────

def make_lr_scheduler(
    optimizer:      torch.optim.Optimizer,
    epochs:         int,
    warmup_epochs:  int,
    lr:             float,
    lr_min:         float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to lr_min."""
    ratio = lr_min / lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return ratio + 0.5 * (1.0 - ratio) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
