# CIDC25 — Progress Log and Next Steps

This file tracks what has been built, what decisions are pending, and
the exact commands to run at each stage. Read `KNOWN_ISSUES.md` first
for the full bug history.

---

## ✅ Done

### Pipeline
- Full training loop in `src/cidc/` with 5 model architectures (n2v3d, mamba3d,
  deepinterp, deepcad, pinn)
- YAML-driven config with strict schema (unknown keys rejected)
- Anscombe VST fused into model forward — unit-variance space throughout
- Gain augmentation: LogUniform([20, 2000]), prob=0.5 per batch
- Auto-resume from `last.pt` — crash-safe, atomic checkpoint writes
- NaN guard: detects non-finite loss before backward, aborts at 5 NaN steps,
  logs `nan-step` / `nan-abort` to JSONL
- EMA (decay=0.999) for validation
- Early stopping on stSNR_val, patience configurable
- Mixed-precision (AMP) throughout
- `--probe-only` flag for fast pipeline validation

### Loss ablation (5 arms)
- Configs: `ablation_nll`, `ablation_mse`, `ablation_mae`,
  `ablation_anscombe_mse`, `ablation_huber`
- All use N2V3D base model, identical hyperparameters, only `loss.name` differs
- `scripts/ablation_verdict.py` reads JSONL logs, ranks arms, prints recommendation
- Decision rules documented in `KNOWN_ISSUES.md` and `README.md`

### TTA (test-time augmentation)
- Implemented in `src/cidc/eval.py`: `_tta_transforms()` + TTA loop in `denoise_stack()`
- D4 group: up to 4 rotations × 2 flips = 8 augmentations
- Wired through config: `inference.tta.rotations` and `inference.tta.flips`
- `tta_rotations=1, tta_flips=False` = identity (no overhead, default for ablation)
- Full training configs have TTA enabled (rotations=4, flips=true)
- Was configured but not implemented previously — configs were dead weight

### Configs fixed
- `patch: [32,128,128]` → `[64,128,128]` in all full training configs
  (T=32 < τ₀.₅=46; T=64 captures 1.4× the decay length)
- `ref_stack: F0` separated from `val_stacks` in all configs
- F3 (OOD Task-2) added to `val_stacks` in all configs

### Documentation
- `README.md` — fully rewritten, all inaccuracies fixed
- `KNOWN_ISSUES.md` — all 9 bugs documented (BUG-01 through BUG-09)

---

## 🔄 Currently running

**5-arm loss ablation** (10 epochs each, sequential on T4):

```bash
export DATA=.../data/train
export RUNS=.../runs

uv run cidc train configs/ablation_nll.yaml          --data $DATA --out $RUNS/nll
uv run cidc train configs/ablation_mse.yaml          --data $DATA --out $RUNS/mse
uv run cidc train configs/ablation_mae.yaml          --data $DATA --out $RUNS/mae
uv run cidc train configs/ablation_anscombe_mse.yaml --data $DATA --out $RUNS/anscombe_mse
uv run cidc train configs/ablation_huber.yaml        --data $DATA --out $RUNS/huber
```

Expected: ~50 min per arm, ~4 hours total on T4.  
If a run crashes: re-run same command — auto-resumes from `last.pt`.

---

## ⏳ Next: after ablation finishes

### Step 1 — Read the verdict

```bash
python scripts/ablation_verdict.py \
    $RUNS/nll $RUNS/mse $RUNS/mae $RUNS/anscombe_mse $RUNS/huber \
    --stack F1

# Also check OOD generalisation:
python scripts/ablation_verdict.py \
    $RUNS/nll $RUNS/mse $RUNS/mae $RUNS/anscombe_mse $RUNS/huber \
    --stack F3
```

Look for:
- `RECOMMENDATION:` line — this is the loss to use
- `tSNR` column — must not collapse while `sSNR` rises
- `🔴ABORTED` — NLL blew up, definitely use the recommended alternative
- Trend: "still-dropping" → run 100 epochs; "flat" at epoch 7-8 → 50 epochs enough

### Step 2 — Model size ablation (optional but recommended, ~2 hours)

Compare base (0.5M params) vs large (4M params) with the winning loss:

```bash
# Edit loss.name in n2v3d_large.yaml to match ablation winner, then:
uv run cidc train configs/n2v3d_large.yaml --data $DATA --out $RUNS/n2v3d_large
```

Decision rule:
- Large wins F1 stSNR by >1 dB → use large for full training
- Large ties base (within 1 dB) → use base (faster, less risk of overfitting)

### Step 3 — Mamba3D vs N2V3D (optional, only if time allows)

```bash
# Edit configs/mamba3d.yaml: set loss.name to winner
uv run cidc train configs/mamba3d.yaml --data $DATA --out $RUNS/mamba3d_ablation
```

First check Mamba is installed:
```bash
uv run python -c "from src.cidc.models.mamba3d import MambaUNet3D; print('OK')"
```

If import fails, Mamba needs a special build step (CUDA extensions). Skip it
and stick with N2V3D if short on time.

### Step 4 — Full training

After choosing model + loss, run 100 epochs:

```bash
# Edit configs/n2v3d.yaml (or n2v3d_large.yaml):
#   loss.name: <winner from ablation>
#   epochs: 100

uv run cidc train configs/n2v3d.yaml --data $DATA --out $RUNS/n2v3d_full
```

Monitor:
```bash
tail -f $RUNS/n2v3d_full/train_n2v3d.jsonl | python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    if r.get('kind') in ('epoch','val','best','nan-abort','early-stop'):
        print(r)
"
```

Expected time: ~10 hours on T4 (2000 samples × 100 epochs × ~6 min/epoch).

---

## 📋 Key numbers to remember

| Metric | Value | Source |
|--------|-------|--------|
| τ₀.₅ (signal decay) | 46 frames | nb01 |
| Baseline F1 stSNR | +7.27 dB | nb07 |
| Baseline F2 stSNR | −0.79 dB | nb07 |
| Baseline F3 stSNR | −6.64 dB | nb07 (OOD floor) |
| R² A1/B1 noise model | 0.23–0.30 | nb03 |
| R² C2/D2 noise model | 0.91–0.95 | nb03 |
| R² val stacks F0-F3 | 0.001–0.24 | nb10 (all poor) |
| 3× gain mismatch | −14.94 dB | nb05 |
| Inference time / stack | ~9 sec (no TTA) | measured on T4 |
| Inference time / stack | ~72 sec (8× TTA) | estimated on T4 |
| Competition time limit | 60 min total | competition rules |

---

## 🚨 Submission checklist (before uploading)

- [ ] Val stSNR on F1 > +7.27 dB (beats raw noisy baseline)
- [ ] Val tSNR not collapsed (check each epoch's tSNR column in logs)
- [ ] Val F3 stSNR improving (OOD Task-2 — gain aug is the only lever)
- [ ] Inference on T4 with TTA (rotations=4, flips=true) stays under 60 min
      (estimate: ~72s/stack × N_test_stacks; safe up to ~40 stacks)
- [ ] No `nan-abort` in final training JSONL
- [ ] `best.pt` checkpoint used (not `last.pt`) — `best.pt` = best val stSNR
