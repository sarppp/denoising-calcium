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
- `n2v3d.yaml` batch=16 → batch=8 + grad_accum=2 to prevent T4 OOM
  (patch=[64,128,128] × batch=16 = ~13.7 GiB; halving batch keeps effective
  batch=16 via grad_accum while cutting peak VRAM to ~6.8 GiB)
- `n2v3d.yaml` `samples_per_epoch: 2000` added (default was 10,000 → 52h on T4;
  2000 → ~8h overnight)

### BUG-10 — bf16 hardcoded, T4 got zero AMP speedup (fixed)
- T4 is Turing (cc 7.5); bf16 requires Ampere (cc ≥ 8.0)
- `torch.bfloat16` on T4 silently falls back to fp32 — no tensor-core benefit
- Fixed in `src/cidc/train.py` and `src/cidc/eval.py` via `_amp_dtype(device, enabled)`:
  - cc ≥ 8.0 (A100, RTX 30xx, A10G) → `bfloat16`
  - cc < 8.0 (T4, V100) → `float16` (2–4× step speedup on T4)
- Expected: ~50 min/arm → ~15–20 min/arm after `git pull` on remote

### Large model configs added
- `configs/n2v3d_large.yaml` — base_ch=32, depth=4, ~4M params (8× base)
  batch=8, grad_accum=2, grad_ckpt=true; fits T4 16 GB
- `configs/mamba3d_large.yaml` — same backbone + Mamba SSM bottleneck
  (n_layers=4, bidirectional); batch=4, grad_accum=4; requires mamba-ssm installed
- Both have TTA enabled (rotations=4, flips=true) and loss.name placeholder

### Documentation
- `README.md` — fully rewritten, all inaccuracies fixed
- `KNOWN_ISSUES.md` — all 10 bugs documented (BUG-01 through BUG-10)
- `NEXT_STEPS.md` — this file

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

Expected: ~15–20 min per arm after fp16 fix (was ~50 min), ~1.5 hours total on T4.  
**Important:** do `git pull` on the remote before running remaining arms to get the fp16 fix.  
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

### Step 2 — Model size ablation (recommended, ~1 hour each)

Compare base (0.5M params) vs large (4M params) with the winning loss.
Edit `loss.name` in each config to match the ablation winner first.

```bash
# Run base and large in parallel if you have 2 GPUs, or sequentially:
uv run cidc train configs/n2v3d.yaml       --data $DATA --out $RUNS/n2v3d_base
uv run cidc train configs/n2v3d_large.yaml --data $DATA --out $RUNS/n2v3d_large
```

Decision rule:
- Large wins F1 stSNR by >1 dB → use large for full training
- Large ties base (within 1 dB) → use base (faster, less overfitting risk on A1/B1)

Model specs:
| Config | base_ch | depth | Params | Batch | grad_accum | Effective batch |
|--------|---------|-------|--------|-------|------------|----------------|
| n2v3d.yaml | 16 | 3 | ~0.5M | 8 | 2 | 16 |
| n2v3d_large.yaml | 32 | 4 | ~4M | 8 | 2 | 16 |

### Step 3 — Mamba3D (optional, only if time allows)

Mamba SSM bottleneck handles long-range temporal dependencies in O(T) vs O(T²).
Potentially better tSNR on long calcium transients (τ₀.₅=46 frames).

First check Mamba is installed:
```bash
uv run python -c "from src.cidc.models.mamba3d import MambaUNet3D; print('OK')"
```

If that works, run both Mamba configs with the winning loss:
```bash
# Edit loss.name in both configs to match ablation winner first
uv run cidc train configs/mamba3d.yaml       --data $DATA --out $RUNS/mamba3d_base
uv run cidc train configs/mamba3d_large.yaml --data $DATA --out $RUNS/mamba3d_large
```

If import fails: Mamba needs CUDA extensions (`pip install mamba-ssm`). Skip and
stick with N2V3D if time is short — N2V3D is proven to work.

Model specs:
| Config | base_ch | depth | SSM layers | Batch | grad_accum |
|--------|---------|-------|------------|-------|------------|
| mamba3d.yaml | 16 | 3 | 2 | 8 | 2 |
| mamba3d_large.yaml | 32 | 4 | 4 | 4 | 4 |

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

## 🖥️ GPU batch-size guide

`grad_accum > 1` is a **memory saver, not a speed-up**. It does N mini-forward+backward
passes to simulate a larger batch, trading wall-clock time for VRAM. Always prefer
`grad_accum=1` with a larger batch when the GPU has the headroom.

### ⚠️ Common mistake: don't increase grad_accum to go faster

```
grad_accum=4, batch=4  →  4 forward+backward passes  →  SLOWER
grad_accum=1, batch=16 →  1 forward+backward pass   →  FASTER  (same effective batch)
```

Only increase grad_accum when you're OOM. If you upgrade GPU, **set grad_accum=1 first**
and fill the freed VRAM by raising batch_size.

---

### N2V3D base (~0.5M params, patch=[64,128,128], 100 epochs, 2000 samples/epoch)

| GPU | VRAM | batch_size | grad_accum | eff. batch | ~100-epoch time | Config change needed |
|-----|------|-----------|------------|------------|----------------|----------------------|
| **T4** (current) | 16 GB | 8 | 2 | 16 | **~8 h** | none — default |
| V100 | 16 GB | 8 | 2 | 16 | ~8 h | none |
| A10G / RTX 3090 | 24 GB | 16 | **1** | 16 | **~4 h** | `batch_size: 16`, `grad_accum: 1` |
| A100 40 GB | 40 GB | 32 | **1** | 32 | **~2 h** | `batch_size: 32`, `grad_accum: 1` |
| A100 80 GB | 80 GB | 64 | **1** | 64 | **~1 h** | `batch_size: 64`, `grad_accum: 1` |

### N2V3D large (~4M params, patch=[64,128,128], grad_ckpt=true)

| GPU | VRAM | batch_size | grad_accum | eff. batch | ~100-epoch time | Config change needed |
|-----|------|-----------|------------|------------|----------------|----------------------|
| **T4** (current) | 16 GB | 8 | 2 | 16 | **~12 h** | none — default |
| A10G / RTX 3090 | 24 GB | 16 | **1** | 16 | **~6 h** | `batch_size: 16`, `grad_accum: 1`, `grad_ckpt: false` |
| A100 40 GB | 40 GB | 32 | **1** | 32 | **~3 h** | `batch_size: 32`, `grad_accum: 1`, `grad_ckpt: false` |
| A100 80 GB | 80 GB | 64 | **1** | 64 | **~1.5 h** | `batch_size: 64`, `grad_accum: 1`, `grad_ckpt: false` |

### Mamba3D base (~1M params, patch=[64,128,128], heavier than N2V3D base)

| GPU | VRAM | batch_size | grad_accum | eff. batch | ~100-epoch time | Config change needed |
|-----|------|-----------|------------|------------|----------------|----------------------|
| **T4** (current) | 16 GB | 8 | 2 | 16 | **~10 h** | none — default |
| A10G / RTX 3090 | 24 GB | 16 | **1** | 16 | **~5 h** | `batch_size: 16`, `grad_accum: 1` |
| A100 40 GB | 40 GB | 32 | **1** | 32 | **~2.5 h** | `batch_size: 32`, `grad_accum: 1` |
| A100 80 GB | 80 GB | 64 | **1** | 64 | **~1.5 h** | `batch_size: 64`, `grad_accum: 1` |

> Times assume 2000 samples/epoch. bf16 AMP on A100 (vs fp16 on T4) gives an additional ~20% speedup on top.  
> On A100 80GB you can also grow `patch: [128,128,128]` for more temporal context — same T4 times still hold with smaller batch.

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
| Inference time / stack | ~9 sec (no TTA, fp16) | measured on T4 |
| Inference time / stack | ~72 sec (8× TTA, fp16) | estimated on T4 |
| Competition time limit | 60 min total | competition rules |
| Safe TTA stack count | ~40 stacks | 40 × 72s = 48 min < 60 min |
| AMP dtype on T4 | float16 (NOT bfloat16) | T4 = Turing cc7.5, no native bf16 |
| AMP dtype on A100 | bfloat16 | Ampere cc8.0+, stable without scaler |

---

## 🚨 Submission checklist (before uploading)

- [ ] Val stSNR on F1 > +7.27 dB (beats raw noisy baseline)
- [ ] Val tSNR not collapsed (check each epoch's tSNR column in logs)
- [ ] Val F3 stSNR improving (OOD Task-2 — gain aug is the only lever)
- [ ] Inference on T4 with TTA (rotations=4, flips=true) stays under 60 min
      (estimate: ~72s/stack × N_test_stacks; safe up to ~40 stacks)
- [ ] No `nan-abort` in final training JSONL
- [ ] `best.pt` checkpoint used (not `last.pt`) — `best.pt` = best val stSNR
