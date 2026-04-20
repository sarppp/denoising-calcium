# Training debug notebooks

Small, focused marimo probes. Each imports from the real `cidc.*`
package — none of them re-implements models, losses, data, or metrics.
Their only purpose is to **make subtle correctness issues visible**
before they silently hurt a long training run.

Run any one with:

```bash
uv run --with="marimo[mcp]" marimo edit \
    workspace/notebooks/training_notebooks/01_loss_geometry.py \
    --mcp --no-token --host 0.0.0.0 --port 2719 --watch
```

## Order

1. **01_loss_geometry** — What does the PG-NLL actually minimise? What
   does `anscombe_mse` expect as input? Why did the earlier bias sweep
   look off-centre?
2. **02_forward_shapes** — Build every backbone from its YAML, run one
   random volume through it, print shapes / param count / NaN check.
   Fastest way to catch kernel / shape bugs.
3. **03_overfit_one_patch** — Tiny training loop that tries to overfit
   one synthetic patch with each loss. If a model can't fit one patch,
   no hyperparameter will save a full run.
4. **04_pinn_head** — Feed a synthetic calcium trace (known τ, b, s(t))
   into a `PINNWrapper` and check the head actually recovers them.
   Sweep the kinetics-aux weight.
5. **05_tile_seam** — Compare `denoise_stack` outputs at different
   `tile` / `overlap` settings. Seams > a couple of ADU mean the
   blending in `cidc.eval.denoise_stack` is misconfigured.


---



All 5 parse. Created `@/app/workspace/notebooks/training_notebooks/`:

- `@/app/workspace/notebooks/training_notebooks/README.md` — index + how to launch
- `@/app/workspace/notebooks/training_notebooks/01_loss_geometry.py` — verifies [poisson_gaussian_nll](cci:1://file:///app/workspace/src/cidc/losses.py:82:0-133:14) argmin ≈ `y − g/2`, bias-sweep geometry, and the **silent trap** from your earlier script (calling [anscombe_mse](cci:1://file:///app/workspace/src/cidc/losses.py:136:0-152:16) with raw ADU instead of Anscombe-space tensors)
- `@/app/workspace/notebooks/training_notebooks/02_forward_shapes.py` — builds every backbone via [cidc.build_model(cidc.load_config(...))](cci:1://file:///app/workspace/src/cidc/models/__init__.py:37:0-54:31) and prints shapes / params / NaN / VRAM; isolates the [mamba3d](cci:9://file:///app/workspace/src/cidc/models/mamba3d:0:0-0:0) kernel-build risk in its own try/except cell
- `@/app/workspace/notebooks/training_notebooks/03_overfit_one_patch.py` — minimal training loop on one synthetic patch using the real `cidc.losses` + [cidc.build_model](cci:1://file:///app/workspace/src/cidc/models/__init__.py:37:0-54:31); plots PG-NLL alongside the true `MSE(pred, clean)` and the **expected PG-NLL floor** `0.5·log(g·μ + σ_r²)`
- `@/app/workspace/notebooks/training_notebooks/04_pinn_head.py` — feeds [PINNWrapper](cci:2://file:///app/workspace/src/cidc/models/pinn/model.py:55:0-156:57) a synthetic trace with known τ/baseline/source (via [cidc.models.pinn.kinetics.euler_forward](cci:1://file:///app/workspace/src/cidc/models/pinn/kinetics.py:31:0-80:74)), freezes the backbone, and checks the head recovers τ
- `@/app/workspace/notebooks/training_notebooks/05_tile_seam.py` — runs [cidc.denoise_stack](cci:1://file:///app/workspace/src/cidc/eval.py:295:0-351:5) with two different `tile/overlap` settings and inspects seam signatures

# Launch

```bash
uv run --with="marimo[mcp]" marimo edit \
    workspace/notebooks/training_notebooks/01_loss_geometry.py \
    --mcp --no-token --host 0.0.0.0 --port 2719 --watch
```

(Your earlier error `Parent directory does not exist` was because CWD was already [/app/workspace](cci:9://file:///app/workspace:0:0-0:0) — either run from [/app](cci:9://file:///app:0:0-0:0) or drop the `workspace/` prefix. The `cd ..` you ran didn't actually move you because every `run_command` gets a fresh shell.)

# Two specific "not-a-bug-but-wrong" pitfalls already encoded

- **[anscombe_mse](cci:1://file:///app/workspace/src/cidc/losses.py:136:0-152:16) silently accepts raw ADU** and returns a finite number with the wrong argmin. Cell [_wrong_vs_right_anscombe](cci:1://file:///app/workspace/notebooks/training_notebooks/01_loss_geometry.py:166:0-207:10) in notebook 01 makes this visible by plotting both objectives on the same axes.
- **PG-NLL does not converge to 0.** Notebook 03 prints the analytical floor `0.5·log(V)` next to the observed final loss, so you never again mistake a correctly-converged run for a stalled one.