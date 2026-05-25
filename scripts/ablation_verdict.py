#!/usr/bin/env python3
"""Read src/cidc training run JSONL logs and print an ablation verdict.

The script reads the RunLogger JSONL format (train_<model>.jsonl) produced by
``src/cidc/train.py`` and answers two questions:

  1. Which loss wins on val stSNR?
  2. Are 10 epochs enough, or should you run longer?

Usage
-----
    # Minimal 3-arm ablation:
    python scripts/ablation_verdict.py runs/nll runs/mse runs/mae

    # Full 5-arm ablation (recommended given nb10 noise-model findings):
    python scripts/ablation_verdict.py \\
        runs/nll runs/mse runs/mae runs/anscombe_mse runs/huber \\
        [--stack F1] [--epoch -1]

    # --stack   which val stack to rank on   (default: F1)
    # --epoch   which epoch row to compare   (-1 = last, default)

Supported losses (any subset, any order)
-----------------------------------------
    poisson_gaussian_nll  Theoretically optimal when R²≥0.9. Risky on A1/B1
                          (R²≈0.27). nb10 shows R²≈0.001 for val stacks too.
    anscombe_mse          MSE in variance-stabilised (Anscombe) space. Principled
                          for Poisson-Gaussian even when R² is poor. Recommended
                          baseline given nb10 findings.
    mse                   Plain MSE in raw ADU. Simple baseline.
    mae                   MAE in raw ADU. Robust to heavy Poisson tails. Targets
                          the conditional median.
    huber                 Huber(δ=1). MSE near zero, MAE in tails. Adaptive.

Decision rules
--------------
NLL wins clearly (>1 dB above ALL others on stSNR)
    → Use NLL for full training. R²≈0.27 didn't destabilise it.

anscombe_mse wins or ties NLL (within 0.5 dB)
    → Use anscombe_mse. Variance stabilisation was enough without full NLL.

MAE or Huber wins / ties best (within 0.5 dB)
    → Use that loss. Distributional mismatch too large for model-based losses.

NLL is unstable (NaN steps > threshold, or non-finite final score)
    → Discard NLL. Use the highest-scoring stable loss.

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

# Ordered preference when multiple losses tie — left = preferred
# (more robust / fewer assumptions wins ties).
LOSS_PREFERENCE = [
    "anscombe_mse",
    "mae",
    "huber",
    "mse",
    "poisson_gaussian_nll",
    "nll",
]

# Keywords to scan config name for when probe-ok row is absent.
_CFG_NAME_KEYWORDS = [
    ("anscombe_mse", "anscombe_mse"),
    ("anscombe",     "anscombe_mse"),
    ("huber",        "huber"),
    ("nll",          "poisson_gaussian_nll"),
    ("mae",          "mae"),
    ("mse",          "mse"),
]


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

    # Per-epoch training loss.
    epoch_loss: dict[int, float] = {}
    for r in rows:
        if r.get("kind") == "epoch":
            epoch_loss[int(r["epoch"])] = float(r.get("train_loss", float("nan")))

    # Val metrics per (epoch, stack).
    val: dict[int, dict[str, dict]] = {}
    for r in rows:
        if r.get("kind") == "val":
            ep = int(r["epoch"])
            stack = str(r.get("file", "?"))
            val.setdefault(ep, {})[stack] = {
                "sSNR":  float(r.get("sSNR",  float("nan"))),
                "tSNR":  float(r.get("tSNR",  float("nan"))),
                "stSNR": float(r.get("stSNR", float("nan"))),
            }

    # Meta — prefer probe-ok row for loss name, fall back to config name.
    cfg_name  = "?"
    loss_name = "?"
    nan_count = 0
    aborted   = False
    for r in rows:
        if r.get("kind") == "train-start":
            cfg_name = r.get("cfg_name", "?")
        if r.get("kind") == "probe-ok":
            loss_name = r.get("loss_name", "?")
        # "nan-step" rows are written once per non-finite loss step (reliable).
        # Fallback: also catch old-format runs that logged NaN via kind="step".
        if r.get("kind") == "nan-step":
            nan_count = max(nan_count, int(r.get("nan_count", nan_count + 1)))
        elif r.get("kind") == "step" and not _isfinite(r.get("loss", 1.0)):
            nan_count += 1
        if r.get("kind") == "nan-abort":
            aborted = True
            nan_count = max(nan_count, int(r.get("nan_count", nan_count)))

    # Infer loss from config name if probe-ok row is absent.
    if loss_name == "?" and cfg_name != "?":
        for kw, canonical in _CFG_NAME_KEYWORDS:
            if kw in cfg_name.lower():
                loss_name = canonical
                break

    # Last resort: infer from directory name.
    if loss_name == "?":
        for kw, canonical in _CFG_NAME_KEYWORDS:
            if kw in run_dir.name.lower():
                loss_name = canonical
                break

    return {
        "run_dir":       run_dir,
        "jsonl":         jsonl,
        "cfg_name":      cfg_name,
        "loss_name":     loss_name,
        "epoch_loss":    epoch_loss,
        "val":           val,
        "nan_count":     nan_count,
        "aborted":       aborted,
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
        v   = run["val"].get(ep, {}).get(stack, {})
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


NLL_WIN_THRESHOLD   = 1.0   # dB above all others → clear NLL win
TIE_THRESHOLD       = 0.5   # dB — within this of best → effectively tied
UNSTABLE_NAN_LIMIT  = 5     # nan steps → unstable run


def _pref_rank(loss_name: str) -> int:
    """Lower = preferred when scores tie."""
    try:
        return LOSS_PREFERENCE.index(loss_name)
    except ValueError:
        return len(LOSS_PREFERENCE)


def _verdict(runs: list[dict], stack: str, compare_epoch: int = -1) -> None:
    print("\n" + "═" * 72)
    print(" ABLATION VERDICT")
    print("═" * 72)

    # Gather final stSNR, NaN counts, and abort flags per run.
    scores:   dict[str, float] = {}
    nans:     dict[str, int]   = {}
    stable:   dict[str, bool]  = {}
    aborted:  dict[str, bool]  = {}
    for r in runs:
        v   = _last_val(r, stack, compare_epoch)
        ln  = r["loss_name"]
        sc  = v.get("stSNR", float("nan"))
        nc  = r["nan_count"]
        ab  = r.get("aborted", False)
        is_stable = nc < UNSTABLE_NAN_LIMIT and _isfinite(sc) and not ab
        scores[ln]  = sc
        nans[ln]    = nc
        stable[ln]  = is_stable
        aborted[ln] = ab

    # Print score table.
    col = 26
    print(f"\n  {'Loss':<{col}}  {'val_' + stack + '_stSNR':>14}  {'NaN steps':>10}  {'Status':>9}")
    print(f"  {'─'*col}  {'─'*14}  {'─'*10}  {'─'*9}")
    for r in sorted(runs, key=lambda r: scores.get(r["loss_name"], float("-inf")), reverse=True):
        ln  = r["loss_name"]
        sc  = scores.get(ln, float("nan"))
        sc_s = f"{sc:+.3f} dB" if _isfinite(sc) else "         ?"
        if aborted.get(ln):
            status = "🔴ABORTED"
        elif stable.get(ln):
            status = "✅ stable"
        else:
            status = "⚠ unstable"
        print(f"  {ln:<{col}}  {sc_s:>14}  {nans.get(ln, 0):>10}  {status:>9}")

    # Filter to stable (non-aborted) runs only.
    stable_scores = {ln: sc for ln, sc in scores.items()
                     if stable.get(ln, False) and _isfinite(sc) and not aborted.get(ln, False)}

    print("\n  Decision:")

    if not stable_scores:
        print("  🔴 ALL runs are unstable (NaN losses or no val data).")
        print("     Check --data path and val_stacks config.")
        print("═" * 72 + "\n")
        return

    best_score = max(stable_scores.values())

    # Build ranked list of losses within TIE_THRESHOLD of best,
    # ordered by LOSS_PREFERENCE (most robust first).
    top = sorted(
        [(ln, sc) for ln, sc in stable_scores.items() if best_score - sc <= TIE_THRESHOLD],
        key=lambda x: _pref_rank(x[0]),
    )
    recommendation_loss, recommendation_score = top[0]

    # Check if NLL specifically is unstable.
    nll_key   = next((ln for ln in scores if "nll" in ln.lower()), None)
    nll_score = stable_scores.get(nll_key, float("-inf")) if nll_key else float("-inf")
    nll_ok    = stable.get(nll_key, False) if nll_key else False

    if nll_key and not nll_ok:
        print(f"  🔴 NLL is UNSTABLE (NaN steps={nans.get(nll_key, '?')}).")
        print(f"     NLL excluded from ranking.")

    # Print winner logic.
    if recommendation_loss == nll_key and nll_ok and _isfinite(nll_score):
        others_max = max(
            (sc for ln, sc in stable_scores.items() if "nll" not in ln.lower()),
            default=float("-inf")
        )
        if nll_score - others_max > NLL_WIN_THRESHOLD:
            print(f"  ✅ NLL wins clearly ({nll_score:+.3f} dB, >{NLL_WIN_THRESHOLD:.0f} dB above others).")
            print("     → Use poisson_gaussian_nll for full training.")
            print("     → R²≈0.27 on A1/B1 didn't destabilise the loss in practice.")
        else:
            print(f"  🟡 NLL leads but not by >{NLL_WIN_THRESHOLD:.0f} dB ({nll_score:+.3f} dB).")
            print("     → Use poisson_gaussian_nll for full training.")
            print("     → Monitor for instability if you add C2/D2 stacks later.")
    elif recommendation_loss == "anscombe_mse":
        print(f"  🟢 anscombe_mse wins/ties best ({recommendation_score:+.3f} dB).")
        print("     → Use anscombe_mse for full training.")
        print("     → Variance stabilisation was enough; full NLL not needed.")
        print("     → This is the safest choice given nb10 R²≈0.001 on val stacks.")
    elif recommendation_loss == "huber":
        print(f"  🟡 Huber wins/ties best ({recommendation_score:+.3f} dB).")
        print("     → Use huber for full training.")
        print("     → Outlier-robust loss suits the mixed noise model on A1/B1.")
    elif recommendation_loss == "mae":
        print(f"  🟡 MAE wins/ties best ({recommendation_score:+.3f} dB).")
        print("     → Use mae for full training.")
        print("     → Poisson tails dominate; median-targeting is more robust.")
    elif recommendation_loss == "mse":
        print(f"  🔵 MSE wins ({recommendation_score:+.3f} dB). Somewhat unexpected.")
        print("     → Use mse. Noise is closer to Gaussian than expected from nb10.")
    else:
        print(f"  ✅ {recommendation_loss} wins ({recommendation_score:+.3f} dB).")
        print(f"     → Use {recommendation_loss} for full training.")

    # Show all tied losses.
    if len(top) > 1:
        tied = [f"{ln} ({sc:+.3f})" for ln, sc in top[1:]]
        print(f"     (Tied within {TIE_THRESHOLD:.1f} dB: {', '.join(tied)})")

    # Epoch count recommendation.
    print("\n  Epoch count:")
    for r in runs:
        trend = _loss_trend(r["epoch_loss"])
        n = len(r["epochs_logged"])
        print(f"    [{r['loss_name']:26s}] trend={trend}  ({n} epochs logged)")

    winner_run = next(
        (r for r in runs if r["loss_name"] == recommendation_loss),
        runs[0]
    )
    winner_trend = _loss_trend(winner_run["epoch_loss"])
    if "still-dropping" in winner_trend:
        last_ep = max(winner_run["epochs_logged"], default=0)
        print(f"\n  → Loss still dropping at epoch {last_ep}. Run 100 epochs for full training.")
    elif "slowing" in winner_trend:
        print(f"\n  → Loss is slowing. 50 epochs likely sufficient.")
    else:
        print(f"\n  → Loss is flat. 30–50 epochs may suffice.")
        print("     Check if val score stopped improving (see 'best' rows in logs).")

    print(f"\n  RECOMMENDATION: {recommendation_loss}")
    print("═" * 72 + "\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Ablation verdict — read cidc train JSONL logs, print winner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "run_dirs", nargs="+", type=Path,
        help="one or more run directories (each has a train_*.jsonl inside); "
             "pass 3 for a minimal ablation (nll mse mae) or 5 for the full "
             "ablation (nll mse mae anscombe_mse huber)"
    )
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
            runs.append(_parse_run(rd))
        except FileNotFoundError as e:
            print(f"WARNING: {e}", file=sys.stderr)

    if not runs:
        print("ERROR: no valid runs found.", file=sys.stderr)
        return 1

    for run in runs:
        print(f"\n{'─'*72}")
        print(f"  Run:    {run['run_dir'].name}")
        print(f"  Config: {run['cfg_name']}  |  Loss: {run['loss_name']}")
        print(f"  JSONL:  {run['jsonl'].name}  |  Epochs: {run['epochs_logged']}")
        _epoch_table(run, args.stack)

    _verdict(runs, args.stack, args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
