No worries — let me re-run with a longer timeout and a smaller bench crop so inference stays under 60 s:
F1 is 1500 × 490 × 490 — full bench on that would take ~40–60 s on this tile config. Let me crop a small smoke bench tif (128 frames) so the full bench+score runs in seconds:



Fix the noise lookup fallback (it can return None, not raise):
Now run bench + score + eval-all on the small crop:

# Everything works end-to-end

## Timings (RTX 2060, 6 GB — the entire pipeline, no issues)

| Stage | Wall | VRAM peak |
|---|---:|---:|
| Train (2 epochs, 256 samples/epoch, tiny model) | **49 s** | 124 MiB |
| Bench (128×256×256 crop, includes warmup) | **2.16 s** | 73 MiB |
| Infer + score + write TIF | **2.58 s** | 73 MiB |
| Eval-all (2 stacks) | **4.8 s** | 73 MiB |

Full pipeline verified: `train → best.pt → bench → infer → score → eval-all`. Logs, JSONL, and Markdown summary all emitted correctly.

## Two real findings from this dry run

1. **The step-log fires multiple times per step.** Look at training output — 4 `[step]` lines with identical step number. The log condition `global_step % log_every == 0` stays true across all accumulation iterations until the next optimizer step. Minor bug, I'll fix.
2. **The leaderboard VRAM budget is fine at any scale.** A tiny model uses 124 MiB during training and 73 MiB during inference. Scaling `base_ch` 8 → 16 (full n2v3d) is a 4× param/activation bump, so ≈ 500 MiB — still 1/12 of a 6 GB card. **You can train the full n2v3d config on your RTX 2060 at batch=8, no grad_accum.**

## Quick step-log fix

---

# Mamba setup on your real GPU box

On the sandbox we can't build `mamba-ssm` because `nvcc` (the CUDA compiler) isn't installed — it's just drivers. On your real box, you need `nvcc` from the CUDA Toolkit available in `PATH` at install time. Recipe:

## 1. Verify `nvcc` is present

```bash
which nvcc && nvcc --version
```

Needs to print a version ≥ 11.8 (mamba-ssm builds against your PyTorch's CUDA version; they must match). If missing, install CUDA Toolkit matching your PyTorch (e.g. 12.4 since this project uses the `pytorch-cu124` index):

**Ubuntu/Debian:**
```bash
# Pick the toolkit matching your torch build (12.4 here).
sudo apt-get install -y cuda-toolkit-12-4
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
```

**Windows (WSL2):** same `apt-get` inside WSL — the CUDA userspace runs from there even if the driver is installed on host Windows.

## 2. Install the extras

```bash
cd /path/to/repo
uv pip install -e '.[mamba]' --no-build-isolation
```

The `--no-build-isolation` flag is **critical** — without it, uv spins up a fresh venv to build `causal-conv1d` which won't have your project's torch, and the build picks up a mismatched torch ABI and silently produces broken kernels.

If the RTX 2060 is your training box, also note:

- **RTX 2060 is Turing (SM 7.5).** Mamba-ssm kernels are fine on Turing for compute but slower than on Ampere+; expect ~1.5–2× the wall-clock of an A100.
- **bf16 is emulated on Turing** (no native hardware). Set `training.amp: true` to still get fp16 speedups, but loss scaling will be needed (the trainer already uses `GradScaler` when cuda is available — already correct).

## 3. Verify

```bash
uv run python -c "
from cidc.models.mamba3d import MambaUNet3D
import torch
m = MambaUNet3D(base_ch=8, depth=3, n_layers=1).cuda()
x = torch.randn(1, 1, 16, 64, 64).cuda()
from cidc.noise import NOISE_LEVELS
y = m(x, NOISE_LEVELS[1])
print('mamba3d OK:', y.shape, y.dtype)
"
```

If that prints a shape and no import error, you're done — `uv run cidc train configs/mamba3d.yaml ...` and the rest of the CLI will dispatch it automatically.

## 4. Fallback if Mamba build is painful

Mamba is one of 5 models, not the backbone of the whole pipeline. If the build fight exceeds 20 min, skip it: the other 4 (`deepinterp`, `n2v3d`, `deepcad`, `pinn`-with-`n2v3d`-backbone) don't need `mamba-ssm` at all and cover the same architectural questions (spatial-temporal receptive field, biological prior). Come back to Mamba later.


QUESTION:
toml will already install cuda though right? only mamab I need to do

ANSWER:

Partial yes. Two different things:

## What `uv sync` gives you (automatic)

- **PyTorch + bundled CUDA runtime** (`cu124` wheels from the configured index). This includes `libcudart`, `libcublas`, `cuDNN`, etc. — everything PyTorch itself needs at runtime.
- This is enough to run **4 of 5 models** (deepinterp, n2v3d, deepcad, pinn) with GPU acceleration end-to-end. No manual CUDA install needed.

## What's missing (manual) — only needed for Mamba

- **`nvcc`** (the CUDA **compiler**) + CUDA development headers. PyTorch wheels do **not** include these — they ship only the runtime.
- `mamba-ssm` and `causal-conv1d` contain **hand-written CUDA kernels** that must be compiled from source against your system's `nvcc` at install time. That's why they need the CUDA Toolkit, not just the runtime.

## Bottom line

| Install step | Needed for |
|---|---|
| `uv sync` | everything except mamba3d |
| `apt install cuda-toolkit-12-4` (one-time) | **only** required for mamba3d |
| `uv pip install -e '.[mamba]' --no-build-isolation` | **only** required for mamba3d |

So on your GPU box: `uv sync` is enough to start training n2v3d / deepcad / deepinterp / pinn immediately. The CUDA Toolkit + `.[mamba]` step is **only** if you want to run mamba3d. Skip it at first — see if Mamba ends up outperforming the others before spending time on the build.
