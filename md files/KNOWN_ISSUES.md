# Known Issues & Limitations — src/cidc

This file documents bugs that were **fixed**, bugs that were **confirmed but left as-is**
(with reasoning), and **architectural limitations** that require larger refactors.
Any agent or human continuing this work should read this first.

---

## Fixed (committed 2026-05-25, fifth batch)

### LIMITATION-01 → BUG-11 · Per-sample gain tensor — LIMITATION-01 fully resolved
**Files:** `src/cidc/train.py`, `models/n2v3d/unet3d.py`, `models/mamba3d/unet3d.py`,
`models/deepcad/unet3d.py`, `models/deepinterp/unet.py`, `models/pinn/model.py`  
**What was wrong:** All model `forward()` calls took a single scalar `NoiseParams` for
the whole batch. Augmented samples (up to 50% of batch) had their Anscombe inverse applied
with the wrong gain, scaling their loss by `k = g_true/g_aug` — as small as 1/71 for
high-gain augmentation. The gain augmentation was effectively muted for the OOD samples
it was designed to help.  
**Fix:** `_make_params()` now returns `(NoiseParams, gains_tensor)`. All `step_*`
functions pass a `(B,1,1,1,1)` per-sample gain tensor to `model.forward()` and use it
for `tgt_raw`. All 5 model `forward()` signatures accept `gain_tensor: Tensor | None = None`
— `None` in the inference path (single tile, no augmentation) falls back to the scalar.

---

## Fixed (committed 2026-05-25, fourth batch)

### BUG-09 · No NaN guard in training loop — backward called on non-finite loss
**File:** `src/cidc/train.py` — main batch loop  
**What was wrong:** The training loop had no `torch.isfinite(loss)` check. When NLL
produced a NaN loss (likely on A1/B1 due to R²≈0.27 misspecification), `backward()`
was called on the NaN tensor — silently propagating NaN gradients into all model
weights and permanently corrupting the checkpoint. The run would continue for all
remaining epochs producing meaningless output. The verdict script tried to count NaN
steps via `kind="step"` rows but those only log every `log_every` steps, so most NaN
events were invisible.  
**Fix:** Check `torch.isfinite(loss)` **before** dividing by `grad_accum` and before
`backward()`. On a NaN loss: log `kind="nan-step"` to JSONL (with running total),
call `zero_grad()` to discard partial accumulation, then `continue` to the next
batch. If the cumulative NaN count reaches `NAN_ABORT_LIMIT=5`, log `kind="nan-abort"`
and `return` — the run exits cleanly without saving a corrupted checkpoint.  
**New JSONL rows:**
```json
{"kind": "nan-step",  "epoch": 2, "step": 412, "loss_name": "poisson_gaussian_nll", "nan_count": 1, "loss": "nan"}
{"kind": "nan-abort", "epoch": 2, "step": 601, "loss_name": "poisson_gaussian_nll", "nan_count": 5, "msg": "... try anscombe_mse or mae"}
```
**Verdict script updated:** `_parse_run` now reads `nan-step` rows (reliable, one
per event). Score table shows `🔴ABORTED` for runs that hit `nan-abort`. Aborted
runs are excluded from the winner comparison.

---

## Fixed (committed 2026-05-25, third batch)

### BUG-07 · `anscombe_mse` in `_simple_loss` was identical to plain `mse` (SILENT BUG)
**File:** `src/cidc/train.py:_simple_loss`  
**What was wrong:** Every `step_*` function converts targets from Anscombe space back to
raw ADU before calling `_simple_loss`. The `anscombe_mse` branch then computed
`((pred - tgt) ** 2).mean()` on raw-ADU tensors — identical to `mse`. The docstring
said "pred/tgt must already be in Anscombe space" but none of the call sites did that.
So `loss.name: anscombe_mse` silently ran plain MSE throughout all previous sessions.  
**Fix:** The forward Anscombe transform
`z = (2/g) * sqrt(g*y + 3/8*g² + σ²)` is now applied inside the `anscombe_mse`
branch of `_simple_loss` before computing MSE. Both pred and tgt are mapped to the
unit-variance stabilised domain, then MSE is computed there.  
**Why this matters:** nb10 showed R²≈0.001–0.24 for all val stacks — the Poisson-Gaussian
model is badly misspecified. `anscombe_mse` is the principled loss for exactly this
regime: it encodes the right inductive bias (noise scales with signal) without betting
on an exact NLL model. After this fix, `anscombe_mse` is a genuinely distinct and
arguably superior baseline compared to plain `mse`.

### BUG-08 · `huber` loss not implemented; `huber_delta` not in schema
**File:** `src/cidc/train.py:_simple_loss`, `src/cidc/config.py:LossConfig`  
**What was wrong:** Huber loss was documented as a candidate in comments/KNOWN_ISSUES
but not implemented. Specifying `loss.name: huber` would have fallen through to NLL.  
**Fix:** Added `huber` branch in `_simple_loss` using `torch.nn.functional.huber_loss`.
Added `huber_delta: float = 1.0` to `LossConfig`. All four `_simple_loss` call sites
updated to pass `huber_delta=cfg.loss.huber_delta`.  
**Added configs:** `configs/ablation_huber.yaml`, `configs/ablation_anscombe_mse.yaml`  
**Verdict script:** `scripts/ablation_verdict.py` updated to handle N arms (not
hardcoded 3), infer loss names from directory names, and produce correct decisions
for all 5 losses including preference ranking on ties.

---

## Fixed (committed 2026-05-25, earlier batches)

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

### ~~LIMITATION-01~~ · ✅ FIXED — Per-sample gain tensor in all model forwards
**File:** `src/cidc/train.py:_make_params` + all `step_*` functions  
**What it is:** All 5 model forward passes call `model(x, params)` where `params`
is one `NoiseParams` shared across the whole batch. When gain augmentation is on
(`prob=0.5`), different samples can have very different gains. The Anscombe inverse
inside the model (`UNet3D.forward`, etc.) then applies the batch-median gain to
every sample instead of each sample's own gain.

**What the median mitigation actually does (BUG-05):**

The BUG-05 fix switched from `gain[0]` to `gains.median()`. This helps un-augmented
samples but does **not** mitigate the problem for augmented samples:

- **Un-augmented samples (50% of batch):** fully protected. Because exactly 50% of
  the batch sits at `g_true`, the batch median always lands at `g_true` — not an
  approximation, it's exact. These samples get the correct Anscombe inverse.

- **Augmented samples (50% of batch):** not mitigated at all. Their input is in
  Anscombe space built with `g_aug`, but both `pred` and `tgt` are inverse-Anscombed
  with `g_median = g_true` (see `step_n2v3d` lines 144–146):

```python
pred_adu = model(masked, params)      # inverse Anscombe with g_true (wrong)
tgt_raw  = (vol / 2.0).pow(2) * params.gain - 0.375 * params.gain - ...
#          vol was built with g_aug, but params.gain = g_true (wrong)
```

Both sides scale by the same wrong factor `k = g_true / g_aug`, so the loss
is also scaled by `k`:

| g_aug | g_true (A1≈28) | scale k | Effective loss weight |
|-------|---------------|---------|----------------------|
| 50    | 28            | 0.56    | 2× too small         |
| 200   | 28            | 0.14    | 7× too small         |
| 991   | 28            | 0.028   | **35× too small**    |
| 2000  | 28            | 0.014   | **71× too small**    |

**Why this matters more than originally assessed:**

The augmented samples are the ones specifically meant to teach OOD generalization
to F3 (gain≈991). A high-gain augmented sample with `g_aug=991` training on an A1
stack (`g_true≈28`) contributes only **1/35th** of the gradient weight of an
un-augmented sample. The model learns almost exclusively from un-augmented
(in-distribution) samples — the gain augmentation is mostly muted, which directly
explains why F3 OOD performance is harder to improve.

This is **not low priority** if F3 OOD performance matters for the competition score.
The current code silently undermines the gain augmentation's intended effect.

**Why not fixed now:** Fixing properly requires changing all 5 model `forward`
signatures to accept a `(B,)` gain tensor and broadcasting correctly through the
Anscombe inverse — a cross-cutting change across 5 model files and the eval path.

**How to fix later:** Add `gain: Tensor | float = None` to model forward.
In `UNet3D.forward`:
```python
g = gain if gain is not None else torch.as_tensor(params.gain, ...)
# g shape (B, 1, 1, 1, 1) when tensor
return (z_pred / 2.0).pow(2) * g - 0.375 * g - sr2 / g
```
Also update `_make_params` to return the full `(B,)` gain tensor and pass it
through all `step_*` functions and `eval.py`.

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

### Running the ablation (all 5 losses, probe only first)
```bash
DATA=/app/workspace/data
RUNS=/app/workspace/runs

# Sanity check: probe all 5 arms before committing GPU time
uv run cidc train configs/ablation_nll.yaml          --data $DATA --out $RUNS/nll          --probe-only
uv run cidc train configs/ablation_mse.yaml          --data $DATA --out $RUNS/mse          --probe-only
uv run cidc train configs/ablation_mae.yaml          --data $DATA --out $RUNS/mae          --probe-only
uv run cidc train configs/ablation_anscombe_mse.yaml --data $DATA --out $RUNS/anscombe_mse --probe-only
uv run cidc train configs/ablation_huber.yaml        --data $DATA --out $RUNS/huber        --probe-only

# Full 10-epoch ablation (can be run in parallel on separate GPUs)
uv run cidc train configs/ablation_nll.yaml          --data $DATA --out $RUNS/nll
uv run cidc train configs/ablation_mse.yaml          --data $DATA --out $RUNS/mse
uv run cidc train configs/ablation_mae.yaml          --data $DATA --out $RUNS/mae
uv run cidc train configs/ablation_anscombe_mse.yaml --data $DATA --out $RUNS/anscombe_mse
uv run cidc train configs/ablation_huber.yaml        --data $DATA --out $RUNS/huber

# If a run crashes, just re-run — it auto-resumes from last.pt
# To force a fresh start: add --no-resume

# Verdict (pass any subset of dirs; script handles N arms)
python scripts/ablation_verdict.py $RUNS/nll $RUNS/mse $RUNS/mae $RUNS/anscombe_mse $RUNS/huber --stack F1
```

### Decision tree summary
| Result at epoch 10 | Action |
|---|---|
| NLL > all others by >1 dB AND stable | NLL for full training |
| anscombe_mse ties NLL (within 0.5 dB) | anscombe_mse — safer, no noise model dependency |
| MAE or Huber ties best | Use that loss — distributional assumptions too strong |
| MSE wins | MSE (unusual — check tSNR didn't collapse) |
| NLL has NaN steps | Discard NLL, use best stable loss |

**Why 5 arms?** nb10 showed R²≈0.001 for F1 — the Poisson-Gaussian model doesn't fit
the val stacks at all. `anscombe_mse` (fixed in this commit) encodes the right
variance-vs-mean relationship without betting on exact NLL. Huber catches the case
where outlier residuals dominate. Running all 5 takes ~5× the wall time of one arm
but costs nothing extra — they can run on separate GPUs in parallel.

Also check `tSNR` separately — a model that gains `sSNR` while losing `tSNR`
scores zero net improvement (stSNR = 0.5×sSNR + 0.5×tSNR).



### How Epochs work 

  Steps per epoch: samples_per_epoch=2000 ÷ batch_size=8 =
  250 steps/epoch. With log_every=20 you get one log line
  every 20 steps. Epoch 2 runs steps 501→750, you're at 640
  so ~110 steps left in that epoch.

  Timing: ~26 seconds per 20 steps → ~1.3 sec/step → 250
  steps × 1.3s = ~5 min per epoch, ~50 min per arm, ~4 hours
   for all 5.

  The loss spike in epoch 2 (40k → 250k → 185k) — this is
  NLL being volatile on A1/B1 (exactly what nb10 warned
  about). It's not aborting because no individual batch is
  NaN, just large. Worth watching — if it keeps spiking the
  verdict will flag NLL as unstable.

  val stSNR=-3.5 at epoch 1 — normal, the model starts worse
   than raw noise. It typically goes negative first then
  climbs. The +7.27 dB floor should be crossed around epoch
  4-6 if NLL stabilises.

  Full training will look identical — same log format, same
  step/val/best rows, same JSONL structure. The only
  differences will be:
  - More epochs (100 instead of 10)
  - Larger patch if you switch to 128³ (more VRAM, slower
  steps)
  - All 4 training stacks instead of just A1/B1
