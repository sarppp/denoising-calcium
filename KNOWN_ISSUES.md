# Known Issues & Limitations — src/cidc

This file documents bugs that were **fixed**, bugs that were **confirmed but left as-is**
(with reasoning), and **architectural limitations** that require larger refactors.
Any agent or human continuing this work should read this first.

---

## Fixed (committed 2026-05-25)

### BUG-06 · F0 mixed into val_stacks — confusing and error-prone
**File:** `src/cidc/config.py`, `src/cidc/train.py`, all 8 YAML configs  
**What was wrong:** `val_stacks` contained F0 alongside F1/F2/F3. F0 is the clean
ground-truth reference, not a noisy stack to be scored. The training loop worked
around this with `if vp.stem == "F0": continue` and a `next(... if p.stem == "F0")`
search — hardcoded strings, not config-driven.  
**Fix:** Added `ref_stack: str = "F0"` as a dedicated field in `DataConfig`. The
`train.py` validation block now uses `cfg.data.ref_stack` directly, and F0 is
removed from every `val_stacks` list in every YAML. No hardcoded "F0" strings remain
in the training logic.  
**Impact:** Cleaner schema. Any stack can now be the reference by changing one field.

---

### BUG-01 · F3 missing from val_stacks (CRITICAL)
**File:** `src/cidc/config.py:80`, all ablation YAMLs, `configs/n2v3d.yaml`  
**What was wrong:** `val_stacks` defaulted to `["F0", "F1", "F2"]`. F3 is the OOD
Task-2 evaluation stack with gain ≈991 (vs 28/249 for F1/F2). Skipping it meant we
never saw whether the model generalises to the out-of-distribution noise level during
training.  
**Fix:** Added F3 to defaults and all YAML files.  
**Impact:** Any checkpoint trained before this fix was never evaluated on F3.

### BUG-02 · MAE loss silently falling through to NLL
**File:** `src/cidc/train.py` — all `step_*` functions  
**What was wrong:** `if cfg.loss.name == "mse": ... else: poisson_gaussian_nll(...)`.
The `mae` case was absent; specifying `loss.name: mae` silently ran NLL.  
**Fix:** Extracted `_simple_loss()` dispatcher that handles `mse`, `mae`,
`anscombe_mse`, and `poisson_gaussian_nll` explicitly. Any unknown name now
falls through to NLL with a clear code path.

### BUG-03 · import tifffile inside epoch loop
**File:** `src/cidc/train.py:299` (old)  
**What was wrong:** `import tifffile` was inside the validation block, re-executed
every epoch.  Python caches imports so there was no performance hit, but it's a
red flag that the block was added in a hurry.  
**Fix:** Moved to top-level imports.

### BUG-04 · Scheduler T₀ inflated when grad_accum > 1
**File:** `src/cidc/train.py` — `build_scheduler` call  
**What was wrong:** `steps_per_epoch = len(train_loader)` counted raw batch steps,
but `sched.step()` is only called every `grad_accum` batches (at each optimizer
step). With `grad_accum=2` and `restarts=3`, you got only 1–2 cosine restarts
instead of 3.  
**Fix:** `opt_steps_per_epoch = max(1, len(train_loader) // cfg.training.grad_accum)`.  
**Impact:** Harmless for all current configs (`grad_accum=1`); matters if you ever
increase it.

### BUG-05 · _make_params: gain from batch index 0 only
**File:** `src/cidc/train.py:_make_params`  
**What was wrong:** When gain augmentation generates different per-sample gains,
all samples in the batch had the Anscombe inverse applied with sample 0's gain.
This biases predictions for samples 1..B-1.  
**Fix (partial):** Now uses batch-median gain, halving the worst-case error.  
**Status:** See LIMITATION-01 below for why this isn't fully fixed.

---

## Confirmed but not fixed

### INFO-01 · `training/dataset.py` — target/gain mismatch (training/ is dead code)
`y = patch` (original gain) but loss was computed at `g_aug`. This is a real
training bug, but `training/` is retired. Use `src/cidc/` exclusively.

---

## Architectural Limitations (known, not going to fix now)

### LIMITATION-01 · Batch gain: one NoiseParams for the whole batch
**File:** `src/cidc/train.py:_make_params` + all `step_*` functions  
**What it is:** All 5 model forward passes call `model(x, params)` where `params`
is one `NoiseParams` shared across the whole batch. When gain augmentation is on
(`prob=0.5`), different samples can have very different gains. The Anscombe inverse
inside the model (`UNet3D.forward`, etc.) then applies the wrong gain to
non-representative samples.  
**Why not fixed now:** Fixing properly requires changing all 5 model `forward`
signatures to accept a `(B,)` gain tensor and broadcasting correctly through the
Anscombe inverse. That's a cross-cutting change across 5 model files and the
eval path. Risk/reward is low: with `prob=0.5` about half the batch uses the
un-augmented gain (close to the median), so the bias is limited.  
**How to fix later:** Add `gain: Tensor | float = None` to model forward.
In `UNet3D.forward`:
```python
g = gain if gain is not None else torch.as_tensor(params.gain, ...)
# g shape (B, 1, 1, 1, 1) when tensor
return (z_pred / 2.0).pow(2) * g - 0.375 * g - sr2 / g
```

### LIMITATION-02 · `eval.py:_is_3d_model` uses class-name string matching
**File:** `src/cidc/eval.py:_is_3d_model`  
**What it is:**
```python
def _is_3d_model(model):
    cls = type(model).__name__
    return cls in {"UNet3D", "DeepCADNet", "MambaUNet3D", "PINNWrapper"}
```
If a 3D model is subclassed, renamed, or a new one is added, it falls through
to `raise TypeError` at inference time.  
**Why not fixed now:** All current models are covered. Adding a new model requires
updating this set anyway, and the error message is clear.  
**How to fix later:** Add a `Is3DModel` mixin/protocol:
```python
class Is3DModel(Protocol):
    def is_3d(self) -> bool: return True
```
Or simpler: check for a `_IS_3D: bool = True` class attribute.

### ~~LIMITATION-03~~ · ✅ FIXED — Resume support (committed 2026-05-25)
**Was:** `src/cidc/train.py` started from scratch every time.  
**Fix:** Auto-resume is now the default. On any run, if `out_dir/last.pt` exists,
training picks up from `epoch + 1` with fully restored model, EMA, optimizer,
scheduler, scaler, `best_val`, `bad_epochs`, and `global_step`. Use `--no-resume`
to force a fresh start.

Additional improvements in the same commit:
- `last.pt` is saved **every epoch** (not just after validation), so a crash
  at any point leaves a valid checkpoint.
- `epoch_NNNN.pt` snapshots written every `cfg.training.ckpt_every` epochs
  for rollback to any specific epoch.
- Checkpoint saves are **atomic** (write to `.tmp` then `rename`) so a crash
  mid-save never corrupts the checkpoint file.
- Probe is **skipped on resume** (pipeline was already validated on the first run).

---

## Usage reference

### Running the ablation (all 3 losses, probe only first)
```bash
DATA=/app/workspace/data
RUNS=/app/workspace/runs

# Sanity check before committing
uv run cidc train configs/ablation_nll.yaml --data $DATA --out $RUNS/nll --probe-only
uv run cidc train configs/ablation_mse.yaml --data $DATA --out $RUNS/mse --probe-only
uv run cidc train configs/ablation_mae.yaml --data $DATA --out $RUNS/mae --probe-only

# If a run crashes, just re-run the same command — it auto-resumes from last.pt
uv run cidc train configs/ablation_nll.yaml --data $DATA --out $RUNS/nll

# Force restart from scratch (ignore existing checkpoint)
uv run cidc train configs/ablation_nll.yaml --data $DATA --out $RUNS/nll --no-resume

# Full 10-epoch ablation
uv run cidc train configs/ablation_nll.yaml --data $DATA --out $RUNS/nll
uv run cidc train configs/ablation_mse.yaml --data $DATA --out $RUNS/mse
uv run cidc train configs/ablation_mae.yaml --data $DATA --out $RUNS/mae

# Verdict
python scripts/ablation_verdict.py $RUNS/nll $RUNS/mse $RUNS/mae --stack F1
```

### Decision tree summary
| val_F1_stSNR at epoch 10 | Action |
|---|---|
| NLL > others by >1 dB | NLL for full training |
| MAE ties NLL (within 0.5 dB) | MAE for full training |
| MSE wins | MSE (unusual — double-check) |
| NLL has NaN steps | MAE (or Hybrid: NLL for C2/D2, MAE for A1/B1) |

Also check `tSNR` separately — a model that gains `sSNR` while losing `tSNR`
scores zero net improvement.
