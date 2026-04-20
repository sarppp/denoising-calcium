"""Unified CLI for CIDC25.

Entry point::

    uv run cidc SUBCOMMAND [args]

Subcommands
-----------
- ``train``     — launch a training run from a YAML config.
- ``infer``     — denoise a single stack with a trained checkpoint.
- ``score``     — compute stSNR for a (pred, ref) pair.
- ``bench``     — time inference + peak VRAM on a stack (60-min budget check).
- ``eval-all``  — run inference & score for many (config, ckpt) pairs.
- ``show``      — pretty-print a config after loading + defaulting.

Every subcommand writes unified logs (console + ``.log`` + ``.jsonl``) to
``--out`` if provided, else the current directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from . import FILE_NOISE, NOISE_LEVELS, build_model, denoise_stack, load_config, stsnr
from .logging import RunLogger, Timer, format_bytes, format_duration
from .noise import identify_noise_level


__all__ = ["main"]


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _load_stack(path: Path) -> np.ndarray:
    arr = np.asarray(tifffile.memmap(path))
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3-D stack, got shape {arr.shape}")
    return arr.astype(np.float32)


def _pick_device(requested: str | None) -> torch.device:
    """``requested`` may be ``cpu`` / ``cuda`` / ``cuda:N`` / None (auto)."""
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _load_ckpt(model, ckpt_path: Path, device: torch.device, prefer_ema: bool = True) -> dict:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "ema" in state and state["ema"] and prefer_ema:
        sd = model.state_dict()
        sd.update(state["ema"])
        model.load_state_dict(sd)
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    return state if isinstance(state, dict) else {}


def _noise_for(path: Path):
    """Infer NoiseParams from filename (F1/F2/F3 or A1/B1/...).

    Falls back to level-1 when the filename isn't in the empirical table
    (e.g. ``F1_small.tif`` or arbitrary user stacks).
    """
    p = identify_noise_level(path.name)
    return p if p is not None else NOISE_LEVELS[1]


# --------------------------------------------------------------------------- #
# Subcommand: show                                                            #
# --------------------------------------------------------------------------- #


def _cmd_show(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    d = cfg.to_dict()
    print(json.dumps(d, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: train                                                           #
# --------------------------------------------------------------------------- #


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train as _train
    cfg = load_config(args.config)
    if args.override_lr is not None:
        cfg.training.lr = float(args.override_lr)
    if args.override_epochs is not None:
        cfg.training.epochs = int(args.override_epochs)
    if args.override_batch is not None:
        cfg.data.batch_size = int(args.override_batch)
    if args.override_grad_accum is not None:
        cfg.training.grad_accum = int(args.override_grad_accum)
    _train(cfg, Path(args.data), Path(args.out))
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: infer                                                           #
# --------------------------------------------------------------------------- #


def _cmd_infer(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    device = _pick_device(args.device)
    out_dir = Path(args.out).parent if args.out else Path(".")
    with RunLogger(out_dir, name=f"infer_{Path(args.config).stem}", cuda=(device.type == "cuda")) as log:
        log.log(kind="infer-start", config=str(args.config), ckpt=str(args.ckpt),
                noisy=str(args.noisy), device=str(device))

        model = build_model(cfg.model).to(device)
        _load_ckpt(model, Path(args.ckpt), device, prefer_ema=not args.no_ema)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        log.log(kind="infer-model", name=cfg.model.name, params_m=n_params / 1e6)

        noisy = _load_stack(Path(args.noisy))
        params = _noise_for(Path(args.noisy))
        log.log(kind="infer-input", shape=list(noisy.shape), dtype=str(noisy.dtype),
                gain=params.gain, read_var=params.read_var)

        with Timer("denoise") as t:
            pred = denoise_stack(
                model, noisy, params,
                tile=cfg.inference.tile,
                overlap=cfg.inference.overlap,
                device=device,
                amp=cfg.training.amp,
            )
        wall = t.dt

        # Optional score vs reference.
        if args.ref:
            ref = _load_stack(Path(args.ref))
            r = stsnr(pred, ref, alpha=args.alpha)
            log.log(kind="infer-score", file=Path(args.noisy).name,
                    ref=Path(args.ref).name, **r.as_dict())

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Keep as float32; users can cast later.
            tifffile.imwrite(out_path, pred.astype(np.float32))
            log.log(kind="infer-written", path=str(out_path), size_b=out_path.stat().st_size)

        log.log(kind="infer-done", wall_sec=wall, budget_60m_ok=(wall < 60 * 60))
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: score                                                           #
# --------------------------------------------------------------------------- #


def _cmd_score(args: argparse.Namespace) -> int:
    if args.config and args.ckpt:
        # Full-pipeline mode: inference + score.
        cfg = load_config(args.config)
        device = _pick_device(args.device)
        model = build_model(cfg.model).to(device)
        _load_ckpt(model, Path(args.ckpt), device, prefer_ema=not args.no_ema)
        model.eval()
        noisy = _load_stack(Path(args.noisy))
        params = _noise_for(Path(args.noisy))
        pred = denoise_stack(model, noisy, params,
                             tile=cfg.inference.tile,
                             overlap=cfg.inference.overlap,
                             device=device, amp=cfg.training.amp)
        ref = _load_stack(Path(args.ref))
        r = stsnr(pred, ref, alpha=args.alpha)
        print(json.dumps({"file": Path(args.noisy).name, "ref": Path(args.ref).name,
                          **r.as_dict()}, indent=2))
        return 0
    # Pair-only mode.
    if not (args.pred and args.ref):
        print("error: provide `pred ref` or --config/--ckpt/--noisy/--ref", file=sys.stderr)
        return 2
    pred = _load_stack(Path(args.pred))
    ref = _load_stack(Path(args.ref))
    r = stsnr(pred, ref, alpha=args.alpha)
    print(json.dumps(r.as_dict(), indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: bench                                                           #
# --------------------------------------------------------------------------- #


def _cmd_bench(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    device = _pick_device(args.device)
    out_dir = Path(args.out) if args.out else Path(".")
    with RunLogger(out_dir, name=f"bench_{Path(args.config).stem}", cuda=(device.type == "cuda")) as log:
        log.log(kind="bench-start", config=str(args.config), device=str(device),
                noisy=str(args.noisy))

        model = build_model(cfg.model).to(device)
        if args.ckpt:
            _load_ckpt(model, Path(args.ckpt), device, prefer_ema=not args.no_ema)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        log.log(kind="bench-model", name=cfg.model.name, params_m=n_params / 1e6)

        noisy = _load_stack(Path(args.noisy))
        params = _noise_for(Path(args.noisy))

        # Warmup tile (amortise CUDA kernel compile).
        if device.type == "cuda":
            with torch.no_grad():
                tT, tH, tW = cfg.inference.tile
                tT = min(tT, noisy.shape[0]); tH = min(tH, noisy.shape[1]); tW = min(tW, noisy.shape[2])
                x = torch.from_numpy(noisy[:tT, :tH, :tW]).float().to(device)[None, None]
                _ = model(x, params) if cfg.model.name != "pinn" else model(x, params)["denoised"]
            torch.cuda.synchronize()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with Timer("denoise") as t:
            pred = denoise_stack(
                model, noisy, params,
                tile=cfg.inference.tile, overlap=cfg.inference.overlap,
                device=device, amp=cfg.training.amp,
            )
        wall = t.dt
        vram_peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0

        # Optional score.
        score = None
        if args.ref:
            ref = _load_stack(Path(args.ref))
            r = stsnr(pred, ref, alpha=args.alpha)
            score = r.as_dict()

        log.log(
            kind="bench-result",
            shape=list(noisy.shape),
            wall_sec=wall,
            wall_pretty=format_duration(wall),
            per_frame_sec=wall / noisy.shape[0],
            vram_peak_b=vram_peak,
            vram_peak_pretty=format_bytes(vram_peak),
            budget_60m_ok=(wall < 60 * 60),
            **(score or {}),
        )
    return 0


# --------------------------------------------------------------------------- #
# Subcommand: eval-all                                                        #
# --------------------------------------------------------------------------- #


def _cmd_eval_all(args: argparse.Namespace) -> int:
    """Run inference + score for many (config, ckpt) pairs across many stacks.

    For each pair, scores every `--noisy` stack against the same `--ref`.
    Emits one combined JSONL at ``<out>/eval_all.jsonl`` plus a Markdown
    summary table printed to stdout.
    """
    configs = [Path(p) for p in args.configs]
    ckpts = [Path(p) for p in args.ckpts] if args.ckpts else [None] * len(configs)
    if len(ckpts) != len(configs):
        print("error: --configs and --ckpts must have the same length", file=sys.stderr)
        return 2
    noisy_paths = [Path(p) for p in args.noisy]
    ref = _load_stack(Path(args.ref))
    device = _pick_device(args.device)
    out_dir = Path(args.out) if args.out else Path(".")

    rows: list[dict[str, Any]] = []
    with RunLogger(out_dir, name="eval_all", cuda=(device.type == "cuda")) as log:
        log.log(kind="eval-all-start", n_configs=len(configs), n_stacks=len(noisy_paths),
                device=str(device))
        for cfg_path, ckpt_path in zip(configs, ckpts):
            cfg = load_config(cfg_path)
            model = build_model(cfg.model).to(device)
            if ckpt_path is not None:
                _load_ckpt(model, ckpt_path, device, prefer_ema=not args.no_ema)
            model.eval()
            for np_path in noisy_paths:
                noisy = _load_stack(np_path)
                params = _noise_for(np_path)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                with Timer() as t:
                    pred = denoise_stack(model, noisy, params,
                                         tile=cfg.inference.tile,
                                         overlap=cfg.inference.overlap,
                                         device=device, amp=cfg.training.amp)
                r = stsnr(pred, ref[: pred.shape[0]], alpha=args.alpha)
                vram = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
                row = {
                    "model": cfg.model.name,
                    "config": cfg_path.name,
                    "ckpt": ckpt_path.name if ckpt_path else "UNTRAINED",
                    "stack": np_path.name,
                    "sSNR": r.s_snr, "tSNR": r.t_snr, "stSNR": r.st_snr,
                    "wall_sec": t.dt, "vram_peak_b": vram,
                }
                log.log(kind="eval-row", **row)
                rows.append(row)

    # Markdown summary.
    print()
    print("| model | config | ckpt | stack | sSNR | tSNR | stSNR | wall | VRAM |")
    print("|---|---|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['model']} | {r['config']} | {r['ckpt']} | {r['stack']} | "
              f"{r['sSNR']:.2f} | {r['tSNR']:.2f} | {r['stSNR']:.2f} | "
              f"{format_duration(r['wall_sec'])} | {format_bytes(r['vram_peak_b'])} |")
    return 0


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cidc", description="CIDC25 unified CLI.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # show
    s = sub.add_parser("show", help="pretty-print a resolved config")
    s.add_argument("config", type=Path)
    s.set_defaults(func=_cmd_show)

    # train
    s = sub.add_parser("train", help="train a model from a YAML config")
    s.add_argument("config", type=Path)
    s.add_argument("--data", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--override-lr", type=float, default=None)
    s.add_argument("--override-epochs", type=int, default=None)
    s.add_argument("--override-batch", type=int, default=None)
    s.add_argument("--override-grad-accum", type=int, default=None)
    s.set_defaults(func=_cmd_train)

    # infer
    s = sub.add_parser("infer", help="denoise a stack with a trained checkpoint")
    s.add_argument("config", type=Path)
    s.add_argument("--ckpt", type=Path, required=True)
    s.add_argument("--noisy", type=Path, required=True)
    s.add_argument("--ref", type=Path, help="optional clean reference for scoring")
    s.add_argument("--out", type=Path, help="write denoised stack as .tif")
    s.add_argument("--device", default="auto")
    s.add_argument("--alpha", type=float, default=0.5)
    s.add_argument("--no-ema", action="store_true", help="use raw weights, not EMA")
    s.set_defaults(func=_cmd_infer)

    # score
    s = sub.add_parser("score", help="compute stSNR for a (pred, ref) pair or full pipeline")
    s.add_argument("pred", nargs="?", type=Path)
    s.add_argument("ref", nargs="?", type=Path)
    s.add_argument("--config", type=Path)
    s.add_argument("--ckpt", type=Path)
    s.add_argument("--noisy", type=Path)
    s.add_argument("--alpha", type=float, default=0.5)
    s.add_argument("--device", default="auto")
    s.add_argument("--no-ema", action="store_true")
    s.set_defaults(func=_cmd_score)

    # bench
    s = sub.add_parser("bench", help="measure inference wall-clock + peak VRAM")
    s.add_argument("config", type=Path)
    s.add_argument("--ckpt", type=Path, help="optional; otherwise random-init")
    s.add_argument("--noisy", type=Path, required=True)
    s.add_argument("--ref", type=Path)
    s.add_argument("--out", type=Path)
    s.add_argument("--device", default="auto")
    s.add_argument("--alpha", type=float, default=0.5)
    s.add_argument("--no-ema", action="store_true")
    s.set_defaults(func=_cmd_bench)

    # eval-all
    s = sub.add_parser("eval-all", help="score many models on many stacks")
    s.add_argument("--configs", nargs="+", required=True)
    s.add_argument("--ckpts", nargs="*", default=None)
    s.add_argument("--noisy", nargs="+", required=True)
    s.add_argument("--ref", type=Path, required=True)
    s.add_argument("--out", type=Path)
    s.add_argument("--device", default="auto")
    s.add_argument("--alpha", type=float, default=0.5)
    s.add_argument("--no-ema", action="store_true")
    s.set_defaults(func=_cmd_eval_all)

    args = p.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
