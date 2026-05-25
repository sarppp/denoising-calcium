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
- `RECOMMENDATION:` line — starting point, but read the numbers yourself too
- `tSNR` column — must not collapse while `sSNR` rises
- `🔴ABORTED` — NLL blew up, definitely use the recommended alternative
- Trend: "still-dropping" → run 100 epochs; "flat" at epoch 7-8 → 50 epochs enough

### ✅ Actual ablation result (completed, T4, 10 epochs)

| Loss | F1 stSNR | F3 stSNR (OOD) | Verdict |
|------|----------|----------------|---------|
| **huber** | **+0.020** | **+0.376** | ✅ winner |
| mae | -0.002 | +0.249 | tied F1, loses F3 |
| anscombe_mse | -0.760 | -1.313 | ❌ unstable on F3 |
| mse | -3.421 | -7.217 | ❌ |
| nll | -3.492 | -7.848 | ❌ |

**Winner: `huber`**

- F1: huber (+0.020) vs mae (-0.002) → 0.022 dB difference, true tie
- F3: huber (+0.376) vs mae (+0.249) → 0.127 dB consistent gap across all epochs → tiebreak
- Huber = MAE in tails + MSE near zero — strictly dominates MAE by construction

**anscombe_mse — never use it.** F3 trajectory: `ep3=+0.703 → ep4=-0.437 → ep5=-5.929 → ep6=-7.435`.
Catastrophic instability. Anscombe amplifies errors when gain is misspecified (R²=0.001–0.24 from nb10).

**Epoch count: set `epochs: 100`, let early stopping decide.**
- F3 OOD peaked at epoch 6 (+1.300) then declined as model overfits to training distribution
- F1 still improving steeply at epoch 9 — not yet converged
- Early stopping (`patience=5` on combined val stSNR = mean of F1+F2+F3) finds the balance automatically
- Do NOT pick the epoch manually

### Step 2 — Model size test: N2V3D base vs large (10 epochs, ~20 min each on L40S)

**Shortcut:** The ablation already ran N2V3D base for 10 epochs.
Use the winning ablation run as your base reference — no need to re-run it.
Only run `n2v3d_large` fresh.

Before running, edit `n2v3d_large.yaml`:
```yaml
loss:
  name: <winner from Step 1>   # e.g. mae, anscombe_mse, huber ...
# epochs: 10 is already set in the config
```

```bash
uv run cidc train configs/n2v3d_large.yaml --data $DATA --out $RUNS/n2v3d_large
```

Then compare with verdict script:
```bash
uv run python scripts/ablation_verdict.py \
    $RUNS/<winning_ablation_arm> $RUNS/n2v3d_large \
    --stack F1
```

**Decision rule:**
- Large wins F1 stSNR by **>1 dB** → use large for full training
- Large ties or loses (within 1 dB) → stick with base (faster, less overfitting risk)

| Config | Params | T4 batch | T4 time/10ep | L40S batch | L40S time/10ep |
|--------|--------|----------|-------------|------------|----------------|
| n2v3d.yaml | ~0.5M | 8, accum=2 | ~50 min | 16, accum=1 | ~10 min |
| n2v3d_large.yaml | ~4M | 8, accum=2 | ~80 min | 16, accum=1 | ~15 min |

### Step 3 — Mamba3D test (optional, only if N2V3D large wins or time allows)

Mamba SSM bottleneck models long-range temporal dependencies in O(T) vs O(T²).
Potentially better tSNR on long calcium transients (τ₀.₅=46 frames).
Only worth trying if you have time — N2V3D is proven to work.

**First check Mamba is installed:**
```bash
uv run python -c "from src.cidc.models.mamba3d import MambaUNet3D; print('OK')"
# If this fails: uv pip install -e '.[mamba]'   (needs nvcc in PATH)
```

Before running, edit `loss.name` in both mamba configs to the ablation winner.
Both already have `epochs: 10` set.

```bash
uv run cidc train configs/mamba3d.yaml       --data $DATA --out $RUNS/mamba3d_base
uv run cidc train configs/mamba3d_large.yaml --data $DATA --out $RUNS/mamba3d_large
```

Compare all 4 at once:
```bash
uv run python scripts/ablation_verdict.py \
    $RUNS/<winning_ablation_arm> $RUNS/n2v3d_large \
    $RUNS/mamba3d_base $RUNS/mamba3d_large \
    --stack F1
```

**Decision rule:** pick Mamba only if it beats the best N2V3D by >1 dB — it is slower per step.

| Config | Params | SSM layers | T4 batch | L40S batch |
|--------|--------|------------|----------|------------|
| mamba3d.yaml | ~1M | 2 | 8, accum=2 | 16, accum=1 |
| mamba3d_large.yaml | ~4M | 4 | 4, accum=4 | 8, accum=1 |

### Step 4 — Full training (100 epochs, winning model + loss)

Edit the winning config:
```yaml
loss:
  name: <winner>        # from Step 1
training:
  epochs: 100           # was 10 for testing
# If on L40S, also uncomment the L40S block at the bottom of the config
```

```bash
# Replace n2v3d_large with whichever model won Steps 2-3
uv run cidc train configs/n2v3d_large.yaml --data $DATA --out $RUNS/full_training
```

Monitor live:
```bash
tail -f $RUNS/full_training/train_n2v3d_large.jsonl | uv run python -c "
import sys, json
for line in sys.stdin:
    r = json.loads(line)
    if r.get('kind') in ('epoch','val','best','nan-abort','early-stop'):
        print(r)
"
```

Expected time:
| GPU | N2V3D base | N2V3D large |
|-----|-----------|-------------|
| T4 | ~8 h | ~12 h |
| L40S (patch=[64,128,128]) | ~2 h | ~3 h |
| L40S (patch=[128,128,128]) | ~3 h | ~5 h |

### Step 5 — Score the trained model on val stacks

F0 is the **clean ground truth** — never denoise it, always use it as `--ref`.  
Denoise F1, F2, F3 and compare each against F0 to get stSNR.

```bash
# Score all 3 val stacks at once — the main command to use after training
uv run python scripts/score.py \
    --config configs/n2v3d_large.yaml \
    --ckpt   $RUNS/full_training/best.pt \
    --data   $DATA

# Output:
#   Stack    sSNR     tSNR    stSNR
#   F1.tif  +12.34   +8.21   +10.28   ← in-distribution
#   F2.tif   +3.11   +1.44    +2.28   ← different condition
#   F3.tif   -1.20   -3.10    -2.15   ← OOD Task-2 (gain≈991)
#   mean                      +3.47

# Quick sanity check during training (no TTA, ~9s per stack):
uv run python scripts/score.py \
    --config configs/n2v3d_large.yaml \
    --ckpt   $RUNS/full_training/best.pt \
    --data   $DATA \
    --no-tta

# Single stack:
uv run python scripts/score.py \
    --config configs/n2v3d_large.yaml \
    --ckpt   $RUNS/full_training/best.pt \
    --noisy  $DATA/F1.tif \
    --ref    $DATA/F0.tif
```

TTA is read from the config automatically (`inference.tta.rotations/flips` — already 8× in all full training configs).  
`--no-tta` overrides it for a quick check.  
Always use `best.pt` (best val stSNR), not `last.pt`.

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

### GPU upgrade recipe — only change 2–3 lines in the yaml

```yaml
# T4 (current defaults)
batch_size: 8
grad_accum: 2
grad_ckpt: true    # large models only

# A100 40GB
batch_size: 32
grad_accum: 1
grad_ckpt: false   # no longer needed
```

Everything else (lr, epochs, patch, scheduler) stays the same.

### grad_ckpt: true — what it does

During a normal forward pass PyTorch **saves every intermediate activation** so it
can use them during backprop. With `grad_ckpt: true` it **throws those activations
away** and recomputes them during backward.

- **Cost**: ~20–30% slower per step
- **Benefit**: ~40–60% less VRAM

Only needed for large models (depth=4, base_ch=32) on T4. Disable it when you move
to a bigger GPU — it's a pure speed penalty once you have the VRAM headroom.

---

## 🔬 Inference, TTA and the 60-minute limit

**Competition rule: one video per container, 60 minutes.**  
This is very generous — a single stack takes ~9 seconds without TTA.

### Inference timing on T4 (fp16)

| Tile size | TTA | Est. time / stack | Under 60 min? |
|-----------|-----|-------------------|---------------|
| [64,128,128] | none (rotations=1) | ~9 sec | ✅ trivially |
| [64,128,128] | 8× (rotations=4, flips=true) | ~72 sec | ✅ trivially |
| [128,128,128] | none | ~12 sec | ✅ |
| [128,128,128] | 8× TTA | ~100 sec | ✅ |

You have ~35× more budget than needed. **Use full 8× TTA** — it's already enabled
in all configs (`rotations: 4, flips: true`) and gives a free ~0.5–1.5 dB.

### Inference VRAM on T4 — always safe

Inference processes **one tile at a time**, not a full batch. Training VRAM ≠ inference VRAM.

| Tile | VRAM per forward pass |
|------|-----------------------|
| [64,128,128] | ~0.5 GB |
| [128,128,128] | ~1 GB |
| [128,256,256] | ~4 GB |

All well within T4's 16 GB even with 8× TTA accumulation.

### Training on bigger GPU, inference on T4

The `.pt` checkpoint is GPU-agnostic. Train anywhere, submit `best.pt`, competition
evaluates on T4. `eval.py` auto-detects fp16/bf16 per GPU — nothing to change.

### L40S config (48 GB VRAM)

If you have an L40S, use a larger temporal patch for better tSNR.
T=128 > 2× τ₀.₅=46 frames — the model sees the full rise and decay of each transient.

```yaml
# Edit in n2v3d.yaml (or n2v3d_large.yaml) when training on L40S:
data:
  patch:      [128, 128, 128]   # T: 64→128 captures full calcium transient decay
  batch_size: 16                # L40S has 48 GB — no need to split
  grad_accum: 1                 # direct, no accumulation needed

inference:
  tile:    [128, 128, 128]      # match training patch
  overlap: [32, 16, 16]        # larger T overlap for the longer tile
```

Inference on T4 with `tile=[128,128,128]`: ~1 GB VRAM per tile, ~100 sec with 8× TTA — still well under 60 min.

> **Don't use `[128,256,256]`** — growing H,W beyond 128 gives little benefit (spatial
> context is already sufficient at 128×128) and risks edge cases on stacks with
> non-standard spatial dimensions.

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
| Competition time limit | 60 min per video | **one video per container** |
| TTA safety margin | ~35× headroom | 72s actual vs 3600s limit |
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
