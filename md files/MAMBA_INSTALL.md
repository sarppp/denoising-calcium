# Installing mamba-ssm on a Remote GPU

Guide for any agent or human picking up this project on a fresh remote GPU instance.

---

## TL;DR — the one command that works

```bash
CUDA_HOME=/usr/local/cuda-12.1 MAX_JOBS=4 uv sync --extra mamba
```

Then verify:
```bash
uv run python -c "from src.cidc.models.mamba3d import MambaUNet3D; print('OK')"
```

`--no-binary`, `--no-build-isolation`, and `--exclude-newer` are now baked into
`pyproject.toml` (`no-binary-package`, `no-build-isolation-package`, and the
`<2.3.0` pin), so you no longer need to pass them manually.

---

## Why `CUDA_HOME` is still required

| Flag | What goes wrong without it |
|---|---|
| `CUDA_HOME=/usr/local/cuda-12.1` | Build pulls CUDA 13 libraries that require GLIBC 2.32; system only has GLIBC 2.31 → `ImportError: version 'GLIBC_2.32' not found` |

The other flags that used to be needed are now handled automatically by uv via `pyproject.toml`:

| What it does | How it's configured |
|---|---|
| Build from source (no pre-built wheels) | `no-binary-package = ["mamba-ssm", "causal-conv1d"]` |
| Build sees torch from `.venv` | `no-build-isolation-package = ["mamba-ssm", "causal-conv1d"]` |
| Pins to mamba-ssm < 2.3.0 (avoids GLIBC 2.32 via nvidia-cuda-runtime 13.x) | `"mamba-ssm>=2.2.2,<2.3.0"` in optional deps |

---

## What the errors look like and what they mean

### Error 1 — GLIBC mismatch (most common)
```
ImportError: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found
(required by selective_scan_cuda.cpython-312-x86_64-linux-gnu.so)
```
**Cause:** Pre-built wheel or a source build that linked against CUDA 13.  
**Fix:** Use the full command above with `CUDA_HOME` + `--exclude-newer`.

### Error 2 — torch not found during build
```
ModuleNotFoundError: No module named 'torch'
  hint: add torch to [tool.uv.extra-build-dependencies]
  or uv pip install torch and re-run with --no-build-isolation
```
**Cause:** `uv` builds packages in an isolated env by default; torch isn't in it.  
**Fix:** Add `--no-build-isolation` so the build sees torch from `.venv`.

### Error 3 — module not found after a seemingly successful install
```
ModuleNotFoundError: No module named 'mamba_ssm'
```
**Cause:** Installed into the wrong Python (conda env instead of `.venv`).  
**Fix:** Add `--python .venv/bin/python` to the install command.

### Error 4 — GLIBC error comes back after `uv run` was fine before
**Cause:** `uv run` re-syncs the env from the lockfile on every invocation and may
revert manually installed packages to the last locked (wheel) version.  
**Fix:** Re-run the full install command. If this keeps happening, add a
`[tool.uv.no-binary]` entry to `pyproject.toml`:
```toml
[tool.uv]
no-binary = ["mamba-ssm", "causal-conv1d"]
```

---

## Prerequisites

Check these before running the install:

```bash
# CUDA compiler must be available
nvcc --version          # need release 12.x

# CUDA 12.1 must be at this exact path
ls /usr/local/cuda-12.1/bin/nvcc

# Check GLIBC version (must be < 2.32 for this guide to apply)
ldd --version | head -1  # e.g. "ldd (Ubuntu GLIBC 2.31-...)"

# Check torch is installed in .venv (required for --no-build-isolation)
uv run python -c "import torch; print(torch.__version__)"
```

If `nvcc` is missing or CUDA is at a different path (e.g. `/usr/local/cuda-12.4`),
adjust `CUDA_HOME` accordingly.

---

## On a machine with GLIBC ≥ 2.32 (e.g. newer cloud instance)

The full command still works, but you can simplify:

```bash
uv pip install "mamba-ssm>=2.2.2" "causal-conv1d>=1.4.0" \
  --no-build-isolation \
  --python .venv/bin/python
```

You can drop `--no-binary` and `--exclude-newer` — pre-built wheels will work fine
and are much faster to install. Keep `--no-build-isolation` and `--python` regardless.

---

## On a plain remote GPU (no Lightning AI studio / no conda conflict)

If `uv pip install` and `uv run` target the same Python (verify with
`uv run python -c "import sys; print(sys.executable)"`), you can drop
`--python .venv/bin/python`. Everything else stays the same.
