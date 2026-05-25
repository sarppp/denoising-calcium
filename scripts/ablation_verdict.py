#!/usr/bin/env python3
"""Read src/cidc training run JSONL logs and print an ablation verdict.

The script reads the RunLogger JSONL format (train_<model>.jsonl) produced by
``src/cidc/train.py`` and answers two questions:

  1. Which loss wins: NLL, MAE, or MSE?
  2. Are 10 epochs enough, or should you run 50/100?

Usage
-----
    python scripts/ablation_verdict.py \\
        runs/nll_dir runs/mse_dir runs/mae_dir [--stack F1] [--epoch -1]

    # --stack   which val stack to rank on   (default: F1)
    # --epoch   which epoch row to compare   (-1 = last, default)

Decision rules (printed at the end)
-------------------------------------
NLL wins clearly (>1 dB above others on stSNR)
    → Use NLL for full training. A1/B1 R²≈0.27 didn't destabilise it.

MAE wins or ties NLL (within 0.5 dB)
    → Use MAE. Poisson tails dominate; median-targeting loss is more robust.

MSE wins
    → Unlikely but possible. Suggests noise is closer to Gaussian than Poisson.

NLL is unstable (NaN count > 0, or train loss diverges)
    → Use MAE. NLL blew up because A1/B1 noise model is too wrong.
    → Also consider Hybrid: NLL for C2/D2 batches, MAE for A1/B1.

After ranking:
    If loss curve still dropping linearly at epoch N → run 100 epochs.
    If loss curve flattening (last 3 epochs < 5% drop)  → 50 epochs is enough.
    If loss curve flattening at epoch 7–8 of 10          → 30 epochs may suffice.

Log format expected (kind field)
---------------------------------
    {"kind": "epoch",  "epoch": 1,  "train_loss": 42.3, "dt_sec": 64}
    {"kind": "val",    "epoch": 1,  "file": "F1", "sSNR": 3.2,
                        "tSNR": 2.1, "stSNR": 2.65, "wall_sec": 180}
    {"kind": "best",   "epoch": 5,  "stSNR": 3.1}
    {"kind": "early-stop", "bad_epochs": 5, "best_stSNR": 3.1}
    {"kind": "train-done", "best_stSNR": 3.1}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Parsing                                                                     #
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _find_jsonl(run_dir: Path) -> Path:
    """Find the train_*.jsonl file in a run directory."""
    candidates = sorted(run_dir.glob("train_*.jsonl"))
    if candidates:
        return candidates[0]
    # Fallback: any jsonl
    candidates = sorted(run_dir.glob("*.jsonl"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No train_*.jsonl found in {run_dir}. "
        "Run must be from src/cidc/train.py (RunLogger format)."
    )


def _parse_run(run_dir: Path) -> dict:
    """Extract epoch table, val metrics, and meta from a run directory."""
    jsonl = _find_jsonl(run_dir)
    rows = _read_jsonl(jsonl)

    # Collect per-epoch training loss.
    epoch_loss: dict[int, float] = {}
    for r in rows:
        if r.get("kind") == "epoch":
            epoch_loss[int(r["epoch"])] = float(r.get("train_loss", float("nan")))

    # Collect val metrics per (epoch, stack).
    val: dict[int, dict[str, dict]] = {}
    for r in rows:
        if r.get("kind") == "val":
            ep = int(r["epoch"])
            stack = str(r.get("file", "?"))
            val.setdefault(ep, {})[stack] = {
                "sSNR": float(r.get("sSNR", float("nan"))),
                "tSNR": float(r.get("tSNR", float("nan"))),
                "stSNR": float(r.get("stSNR", float("nan"))),
            }

    # Meta.
    cfg_name = "?"
    loss_name = "?"
    nan_count = 0
    for r in rows:
        if r.get("kind") == "train-start":
            cfg_name = r.get("cfg_name", "?")
        if r.get("kind") == "probe-ok":
            loss_name = r.get("loss_name", "?")
        if r.get("kind") == "step" and not _isfinite(r.get("loss", 1.0)):
            nan_count += 1

    # Infer loss from config name if probe-ok row is absent.
    if loss_name == "?" and cfg_name != "?":
        for kw in ("nll", "mse", "mae"):
            if kw in cfg_name:
                loss_name = kw
                break

    return {
        "run_dir": run_dir,
        "jsonl": jsonl,
        "cfg_name": cfg_name,
        "loss_name": loss_name,
        "epoch_loss": epoch_loss,
        "val": val,
        "nan_count": nan_count,
        "epochs_logged": sorted(epoch_loss.keys()),
    }


def _isfinite(v) -> bool:
    try:
        return float(v) == float(v) and abs(float(v)) < 1e30
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Display                                                                     #
# --------------------------------------------------------------------------- #


def _epoch_table(run: dict, stack: str) -> None:
    """Print epoch | train_loss | stSNR | sSNR | tSNR for one run."""
    print(f"\n  {'ep':>4}  {'train_loss':>12}  {'stSNR':>7}  {'sSNR':>7}  {'tSNR':>7}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}")
    for ep in run["epochs_logged"]:
        loss = run["epoch_loss"].get(ep, float("nan"))
        v = run["val"].get(ep, {}).get(stack, {})
        st  = v.get("stSNR", float("nan"))
        s   = v.get("sSNR",  float("nan"))
        t   = v.get("tSNR",  float("nan"))

        def _f(x): return f"{x:+7.3f}" if _isfinite(x) else "      ?"
        loss_s = f"{loss:12.4f}" if _isfinite(loss) else "           ?"
        print(f"  {ep:4d}  {loss_s}  {_f(st)}  {_f(s)}  {_f(t)}")


def _last_val(run: dict, stack: str, epoch: int = -1) -> dict:
    """Get val metrics for the requested epoch (-1 = last with val data)."""
    val_epochs = sorted(ep for ep in run["val"] if stack in run["val"][ep])
    if not val_epochs:
        return {}
    ep = val_epochs[epoch] if -len(val_epochs) <= epoch < len(val_epochs) else val_epochs[-1]
    return run["val"][ep].get(stack, {})


def _loss_trend(epoch_loss: dict[int, float]) -> str:
    """Characterise the loss curve from the last few epochs."""
    eps = sorted(epoch_loss.keys())
    if len(eps) < 3:
        return "too-short"
    losses = [epoch_loss[e] for e in eps[-4:] if _isfinite(epoch_loss[e])]
    if len(losses) < 2:
        return "noisy"
    first, last = losses[0], losses[-1]
    if not _isfinite(first) or first == 0:
        return "noisy"
    pct_drop = (first - last) / abs(first) * 100
    if pct_drop > 10:
        return f"still-dropping ({pct_drop:.1f}% over last {len(losses)} epochs)"
    if pct_drop > 2:
        return f"slowing ({pct_drop:.1f}% over last {len(losses)} epochs)"
    return f"flat ({pct_drop:.1f}% over last {len(losses)} epochs)"


# --------------------------------------------------------------------------- #
# Decision tree                                                               #
# --------------------------------------------------------------------------- #


NLL_WIN_THRESHOLD  = 1.0   # dB above all others → clear NLL win
MAE_TIE_THRESHOLD  = 0.5   # dB — within this of NLL → MAE preferred (more robust)
UNSTABLE_NAN_LIMIT = 5     # nan steps → unstable


def _verdict(runs: list[dict], stack: str, compare_epoch: int = -1) -> None:
    print("\n" + "═" * 68)
    print(" ABLATION VERDICT")
    print("═" * 68)

    # Gather final stSNR per run.
    scores: dict[str, float] = {}
    nans:   dict[str, int]   = {}
    for r in runs:
        v = _last_val(r, stack, compare_epoch)
        scores[r["loss_name"]] = v.get("stSNR", float("nan"))
        nans[r["loss_name"]]   = r["nan_count"]

    # Check for instability.
    nll_nan = nans.get("poisson_gaussian_nll", 0) + nans.get("nll", 0)
    nll_stable = nll_nan < UNSTABLE_NAN_LIMIT and _isfinite(scores.get("poisson_gaussian_nll",
                                                                        scores.get("nll", float("nan"))))

    # Print score table.
    print(f"\n  {'Loss':<25}  {'val_' + stack + '_stSNR':>14}  {'NaN steps':>10}")
    print(f"  {'─'*25}  {'─'*14}  {'─'*10}")
    for r in runs:
        ln = r["loss_name"]
        sc = scores.get(ln, float("nan"))
        sc_s = f"{sc:+.3f} dB" if _isfinite(sc) else "      ?"
        print(f"  {ln:<25}  {sc_s:>14}  {nans.get(ln, 0):>10}")

    # Determine winner.
    valid = {k: v for k, v in scores.items() if _isfinite(v)}
    if not valid:
        print("\n  ⚠  No valid val metrics found. Did validation run? Check --data path.")
        return

    best_loss = max(valid, key=valid.__getitem__)
    best_score = valid[best_loss]

    nll_key = next((k for k in valid if "nll" in k.lower()), None)
    mae_key = next((k for k in valid if k == "mae"), None)
    mse_key = next((k for k in valid if k == "mse"), None)

    nll_score = valid.get(nll_key, float("-inf")) if nll_key else float("-inf")
    mae_score = valid.get(mae_key, float("-inf")) if mae_key else float("-inf")

    print("\n  Decision:")
    if not nll_stable:
        print("  🔴 NLL UNSTABLE — NaN losses or non-finite final score.")
        print("     → Use MAE for full training.")
        print("     → Consider Hybrid: NLL for C2/D2 batches, MAE for A1/B1.")
        recommendation = "MAE (NLL unstable)"
    elif _isfinite(nll_score) and nll_score - max(mae_score, valid.get(mse_key, float("-inf"))) > NLL_WIN_THRESHOLD:
        print(f"  ✅ NLL wins clearly ({nll_score:+.3f} dB, >{NLL_WIN_THRESHOLD:.0f} dB above others).")
        print("     → Use poisson_gaussian_nll for full training.")
        print("     → A1/B1 R²≈0.27 didn't destabilise the loss in practice.")
        recommendation = "NLL (clear winner)"
    elif _isfinite(mae_score) and mae_score >= nll_score - MAE_TIE_THRESHOLD:
        print(f"  🟡 MAE ties or beats NLL (MAE={mae_score:+.3f}, NLL={nll_score:+.3f}).")
        print("     → Use MAE for full training.")
        print("     → Poisson tails dominate — median-targeting is more robust.")
        recommendation = "MAE (ties NLL)"
    elif best_loss == mse_key:
        print(f"  🔵 MSE wins ({best_score:+.3f} dB). Unusual.")
        print("     → Use MSE. Noise is closer to Gaussian than expected.")
        recommendation = "MSE (unexpected winner)"
    else:
        print(f"  ✅ NLL wins ({nll_score:+.3f} dB). Not a blowout but ahead.")
        print("     → Use poisson_gaussian_nll for full training.")
        recommendation = "NLL (ahead)"

    # Epoch count recommendation.
    print("\n  Epoch count:")
    for r in runs:
        trend = _loss_trend(r["epoch_loss"])
        n = len(r["epochs_logged"])
        print(f"    [{r['loss_name']:25s}] trend={trend}  ({n} epochs logged)")

    winner_run = next((r for r in runs if r["loss_name"] in recommendation.lower()
                       or ("nll" in recommendation.lower() and "nll" in r["loss_name"].lower())), runs[0])
    winner_trend = _loss_trend(winner_run["epoch_loss"])
    if "still-dropping" in winner_trend:
        print(f"\n  → Loss still dropping at epoch {max(winner_run['epochs_logged'])}.")
        print("     Run 100 epochs for full training.")
    elif "slowing" in winner_trend:
        print(f"\n  → Loss is slowing. 50 epochs likely sufficient.")
    else:
        print(f"\n  → Loss is flat. 30–50 epochs may suffice.")
        print("     Check if val score stopped improving (check 'best' row in logs).")

    print(f"\n  RECOMMENDATION: {recommendation}")
    print("═" * 68 + "\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Ablation verdict — read cidc train JSONL logs, print winner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("run_dirs", nargs="+", type=Path,
                   help="one or more run directories (each has a train_*.jsonl inside)")
    p.add_argument("--stack", default="F1",
                   help="val stack to rank on (default: F1)")
    p.add_argument("--epoch", type=int, default=-1,
                   help="which epoch to compare; -1 = last (default)")
    args = p.parse_args(argv)

    runs = []
    for rd in args.run_dirs:
        rd = Path(rd)
        if not rd.exists():
            print(f"WARNING: {rd} does not exist — skipped.", file=sys.stderr)
            continue
        try:
            run = _parse_run(rd)
            runs.append(run)
        except FileNotFoundError as e:
            print(f"WARNING: {e}", file=sys.stderr)

    if not runs:
        print("ERROR: no valid runs found.", file=sys.stderr)
        return 1

    for run in runs:
        print(f"\n{'─'*68}")
        print(f"  Run: {run['run_dir'].name}")
        print(f"  Config: {run['cfg_name']}  |  Loss: {run['loss_name']}")
        print(f"  JSONL: {run['jsonl'].name}  |  Epochs: {run['epochs_logged']}")
        _epoch_table(run, args.stack)

    _verdict(runs, args.stack, args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
