#!/usr/bin/env python3
"""Read src/cidc training run JSONL logs and print a model-size / model-type verdict.

Unlike ablation_verdict.py (which compares loss functions), this script compares
different model architectures that all use the same loss.

Usage
-----
    python scripts/model_verdict.py \\
        runs/n2v3d_base runs/n2v3d_large runs/mamba_base runs/mamba_large \\
        [--stack F1] [--also F3] [--epoch -1]

    # --stack    primary val stack to rank on  (default: F1)
    # --also     also print a second stack     (default: F3, OOD check)
    # --epoch    which epoch to compare; -1 = last (default)

Decision rules
--------------
Large wins base by >1 dB on primary stack
    → Use large model. The extra capacity is worth the cost.

Large within 1 dB of base
    → Stick with base. No significant gain; base is faster and less overfit risk.

Mamba wins best N2V3D by >1 dB
    → Use Mamba. SSM temporal modelling helped.

Mamba within 1 dB of best N2V3D
    → Stick with N2V3D. Mamba didn't improve enough to justify the complexity
      and trickier install (mamba-ssm build issues on some remotes).

Always check OOD stack (F3) too — competition score is average across conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ─── Decision thresholds ────────────────────────────────────────────────────

CLEAR_WIN_DB   = 1.0   # dB gap = clear winner; upgrade is worth it
TIE_DB         = 0.5   # dB — within this = effectively tied


# ─── Parsing ────────────────────────────────────────────────────────────────

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
    jsonl = _find_jsonl(run_dir)
    rows  = _read_jsonl(jsonl)

    epoch_loss: dict[int, float] = {}
    for r in rows:
        if r.get("kind") == "epoch":
            epoch_loss[int(r["epoch"])] = float(r.get("train_loss", float("nan")))

    val: dict[int, dict[str, dict]] = {}
    for r in rows:
        if r.get("kind") == "val":
            ep    = int(r["epoch"])
            stack = str(r.get("file", "?"))
            val.setdefault(ep, {})[stack] = {
                "sSNR":  float(r.get("sSNR",  float("nan"))),
                "tSNR":  float(r.get("tSNR",  float("nan"))),
                "stSNR": float(r.get("stSNR", float("nan"))),
            }

    cfg_name   = "?"
    model_name = "?"
    nan_count  = 0
    aborted    = False
    for r in rows:
        if r.get("kind") == "train-start":
            cfg_name = r.get("cfg_name", "?")
        if r.get("kind") == "probe-ok":
            model_name = r.get("model_name", "?")
        if r.get("kind") == "nan-step":
            nan_count = max(nan_count, int(r.get("nan_count", nan_count + 1)))
        elif r.get("kind") == "step" and not _isfinite(r.get("loss", 1.0)):
            nan_count += 1
        if r.get("kind") == "nan-abort":
            aborted   = True
            nan_count = max(nan_count, int(r.get("nan_count", nan_count)))

    # Fall back to config name or directory name for model identification.
    if model_name == "?" and cfg_name != "?":
        model_name = cfg_name
    if model_name == "?":
        model_name = run_dir.name

    return {
        "run_dir":       run_dir,
        "jsonl":         jsonl,
        "cfg_name":      cfg_name,
        "model_name":    model_name,
        "label":         run_dir.name,   # human-readable: directory name
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


# ─── Display ────────────────────────────────────────────────────────────────

def _epoch_table(run: dict, stack: str) -> None:
    print(f"\n  {'ep':>4}  {'train_loss':>12}  {'stSNR':>7}  {'sSNR':>7}  {'tSNR':>7}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*7}")
    for ep in run["epochs_logged"]:
        loss = run["epoch_loss"].get(ep, float("nan"))
        v    = run["val"].get(ep, {}).get(stack, {})
        st   = v.get("stSNR", float("nan"))
        s    = v.get("sSNR",  float("nan"))
        t    = v.get("tSNR",  float("nan"))

        def _f(x): return f"{x:+7.3f}" if _isfinite(x) else "      ?"
        loss_s = f"{loss:12.4f}" if _isfinite(loss) else "           ?"
        print(f"  {ep:4d}  {loss_s}  {_f(st)}  {_f(s)}  {_f(t)}")


def _last_val(run: dict, stack: str, epoch: int = -1) -> dict:
    val_epochs = sorted(ep for ep in run["val"] if stack in run["val"][ep])
    if not val_epochs:
        return {}
    ep = val_epochs[epoch] if -len(val_epochs) <= epoch < len(val_epochs) else val_epochs[-1]
    return run["val"][ep].get(stack, {})


def _best_val(run: dict, stack: str) -> float:
    """Return the best stSNR across all epochs for this stack."""
    scores = [
        run["val"][ep][stack]["stSNR"]
        for ep in run["val"]
        if stack in run["val"][ep] and _isfinite(run["val"][ep][stack]["stSNR"])
    ]
    return max(scores) if scores else float("nan")


def _loss_trend(epoch_loss: dict[int, float]) -> str:
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


# ─── Verdict ────────────────────────────────────────────────────────────────

def _score_table(runs: list[dict], stack: str, compare_epoch: int, title: str) -> dict[str, float]:
    """Print ranked score table, return label→score dict."""
    scores = {}
    for r in runs:
        v  = _last_val(r, stack, compare_epoch)
        sc = v.get("stSNR", float("nan"))
        scores[r["label"]] = sc

    col = 30
    print(f"\n  {title}")
    print(f"  {'Run (directory)':<{col}}  {'val_' + stack + '_stSNR':>14}  {'NaN steps':>10}  {'Status':>9}")
    print(f"  {'─'*col}  {'─'*14}  {'─'*10}  {'─'*9}")

    for r in sorted(runs, key=lambda r: scores.get(r["label"], float("-inf")), reverse=True):
        lb  = r["label"]
        sc  = scores.get(lb, float("nan"))
        sc_s = f"{sc:+.3f} dB" if _isfinite(sc) else "         ?"
        if r["aborted"]:
            status = "🔴ABORTED"
        elif r["nan_count"] < 5 and _isfinite(sc):
            status = "✅ stable"
        else:
            status = "⚠ unstable"
        print(f"  {lb:<{col}}  {sc_s:>14}  {r['nan_count']:>10}  {status:>9}")

    return scores


def _combined(label: str, scores: dict, also_scores: dict) -> float:
    """Average of primary and OOD score.  Falls back to primary if OOD is missing."""
    s1 = scores.get(label, float("nan"))
    s2 = also_scores.get(label, float("nan"))
    if _isfinite(s1) and _isfinite(s2):
        return (s1 + s2) / 2.0
    return s1   # no OOD data → primary only


def _combined_table(runs: list[dict], scores: dict, also_scores: dict, stack: str, also: str) -> None:
    """Print combined (primary + OOD) score table, sorted by combined score."""
    col = 30
    print(f"\n  Combined score: ({stack} + {also}) / 2  ← competition averages across stacks")
    print(f"  {'Run (directory)':<{col}}  {'combined_stSNR':>14}  {'val_'+stack+'_stSNR':>16}  {'val_'+also+'_stSNR':>16}")
    print(f"  {'─'*col}  {'─'*14}  {'─'*16}  {'─'*16}")
    ranked = sorted(runs, key=lambda r: _combined(r["label"], scores, also_scores), reverse=True)
    for r in ranked:
        lb   = r["label"]
        comb = _combined(lb, scores, also_scores)
        s1   = scores.get(lb, float("nan"))
        s2   = also_scores.get(lb, float("nan"))
        c_s  = f"{comb:+.3f} dB" if _isfinite(comb) else "         ?"
        s1_s = f"{s1:+.3f} dB"   if _isfinite(s1)   else "         ?"
        s2_s = f"{s2:+.3f} dB"   if _isfinite(s2)   else "         ?"
        print(f"  {lb:<{col}}  {c_s:>14}  {s1_s:>16}  {s2_s:>16}")


def _verdict(runs: list[dict], stack: str, also: str | None, compare_epoch: int) -> None:
    print("\n" + "═" * 72)
    print(" MODEL VERDICT")
    print("═" * 72)

    scores = _score_table(runs, stack, compare_epoch, f"Primary stack: {stack}")
    also_scores: dict[str, float] = {}
    if also and also != stack:
        also_scores = _score_table(runs, also, compare_epoch, f"OOD stack: {also}")

    # When OOD scores are available, rank by combined score — that is what the
    # competition actually measures.  Primary-only ranking caused wrong picks when
    # a model is good on F1 but catastrophically bad on F3 (e.g. mamba_large).
    def _rank_score(r):
        return _combined(r["label"], scores, also_scores)

    # Stable runs only (no NaN abort, finite primary score).
    stable_runs = [
        r for r in runs
        if r["nan_count"] < 5 and _isfinite(scores.get(r["label"], float("nan"))) and not r["aborted"]
    ]

    # Print combined table when OOD data is available.
    if also_scores:
        _combined_table(stable_runs, scores, also_scores, stack, also)

    print("\n  Decision:")

    if not stable_runs:
        print("  🔴 ALL runs are unstable. Check --data path and val_stacks config.")
        print("═" * 72 + "\n")
        return

    # Rank by combined score (primary only if no OOD available).
    ranked  = sorted(stable_runs, key=_rank_score, reverse=True)
    best    = ranked[0]
    best_sc = _rank_score(best)

    # ── Size comparison: is large better than base? ─────────────────────────
    large_runs = [r for r in stable_runs if "large" in r["label"].lower()]
    base_runs  = [r for r in stable_runs if "large" not in r["label"].lower()]

    for family, keyword in [("N2V3D", "n2v3d"), ("Mamba", "mamba")]:
        fam_large = [r for r in large_runs if keyword in r["label"].lower()]
        fam_base  = [r for r in base_runs  if keyword in r["label"].lower()]
        if not fam_large or not fam_base:
            continue
        best_large = max(fam_large, key=_rank_score)
        best_base  = max(fam_base,  key=_rank_score)
        gap = _rank_score(best_large) - _rank_score(best_base)
        metric_label = f"combined {stack}+{also}" if also_scores else stack
        if gap > CLEAR_WIN_DB:
            print(f"  ✅ {family} large wins base by {gap:+.3f} dB ({metric_label}) → use large.")
        elif gap > TIE_DB:
            print(f"  🟡 {family} large leads base by {gap:+.3f} dB ({metric_label}, borderline).")
            print(f"     → Use large if you have VRAM budget; base is fine otherwise.")
        else:
            print(f"  🔵 {family} large tied/lost vs base ({gap:+.3f} dB {metric_label}) → stick with base.")

    # ── Architecture comparison: Mamba vs N2V3D ─────────────────────────────
    best_mamba = max(
        (r for r in stable_runs if "mamba" in r["label"].lower()),
        key=_rank_score, default=None
    )
    best_n2v3d = max(
        (r for r in stable_runs if "n2v3d" in r["label"].lower()),
        key=_rank_score, default=None
    )
    if best_mamba and best_n2v3d:
        gap = _rank_score(best_mamba) - _rank_score(best_n2v3d)
        metric_label = f"combined {stack}+{also}" if also_scores else stack
        if gap > CLEAR_WIN_DB:
            print(f"\n  ✅ Mamba wins N2V3D by {gap:+.3f} dB ({metric_label}) → use Mamba.")
            print("     SSM temporal modelling helped for calcium transient denoising.")
        elif gap > TIE_DB:
            print(f"\n  🟡 Mamba leads N2V3D by {gap:+.3f} dB ({metric_label}, borderline).")
            print("     → Use Mamba if install is stable; N2V3D is the safe fallback.")
        else:
            print(f"\n  🔵 Mamba tied/lost vs N2V3D ({gap:+.3f} dB {metric_label}) → stick with N2V3D.")
            print("     No benefit from SSM; N2V3D is simpler, faster, easier to deploy.")

    # ── OOD danger check ────────────────────────────────────────────────────
    if also_scores:
        print(f"\n  OOD danger check ({also} vs raw noisy baseline):")
        for r in sorted(stable_runs, key=lambda r: also_scores.get(r["label"], float("-inf")), reverse=True):
            lb  = r["label"]
            sc  = also_scores.get(lb, float("nan"))
            if not _isfinite(sc):
                continue
            # Warn if model is worse than doing nothing on OOD stack.
            # F3 raw noisy baseline ≈ −6.64 dB; any score below that is a regression.
            # We use a generous threshold: flag if score < −5 dB (slightly above true baseline).
            flag = "  ❌ WORSE THAN NO DENOISING" if sc < -5.0 else ""
            print(f"    {lb:<30s} {sc:+.3f} dB on {also}{flag}")

    # ── Overall recommendation ───────────────────────────────────────────────
    print(f"\n  Epoch trend (loss curve from last 4 epochs):")
    for r in runs:
        trend = _loss_trend(r["epoch_loss"])
        n     = len(r["epochs_logged"])
        print(f"    [{r['label']:30s}] {trend}  ({n} epochs logged)")

    metric_label = f"combined {stack}+{also}" if also_scores else stack
    print(f"\n  RECOMMENDATION: {best['label']}  ({best_sc:+.3f} dB {metric_label})")
    print(f"    {stack}: {scores.get(best['label'], float('nan')):+.3f} dB", end="")
    if also_scores and _isfinite(also_scores.get(best["label"], float("nan"))):
        print(f"  |  {also}: {also_scores[best['label']]:+.3f} dB", end="")
    print()
    print("═" * 72 + "\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Model verdict — compare architecture runs on val stSNR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "run_dirs", nargs="+", type=Path,
        help="run directories (each has a train_*.jsonl inside)"
    )
    p.add_argument("--stack", default="F1",
                   help="primary val stack to rank on (default: F1)")
    p.add_argument("--also",  default="F3",
                   help="also show this stack as OOD check (default: F3)")
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
        print(f"  Run:    {run['label']}")
        print(f"  Config: {run['cfg_name']}  |  Model: {run['model_name']}")
        print(f"  JSONL:  {run['jsonl'].name}  |  Epochs: {run['epochs_logged']}")
        _epoch_table(run, args.stack)

    _verdict(runs, args.stack, args.also, args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
