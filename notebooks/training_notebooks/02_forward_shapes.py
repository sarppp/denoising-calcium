"""02 — Forward shapes.

What this notebook is for
-------------------------
Build every registered model *from its YAML* using the same code path
the CLI uses (``cidc.load_config`` + ``cidc.build_model``), push one
random Anscombe-space volume through it, and print:

- parameter count,
- output shape and dtype,
- whether the output contains NaN / Inf,
- rough VRAM use for a single forward at the configured patch size.

This is the cheapest way to catch:

- wrong tensor rank expected by a model (e.g. DeepInterp is 2-D-ish),
- silent `pad` mismatches in U-Net decoders,
- custom CUDA kernels that won't compile (mamba3d),
- a PINN backbone forgetting to register its ``.head`` conv.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _setup():
    from pathlib import Path

    import torch

    from cidc import NOISE_LEVELS, build_model, load_config

    CONFIGS = Path("/app/workspace/configs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    return CONFIGS, NOISE_LEVELS, build_model, device, load_config, torch


@app.function
def report(name, model, x, params, device, torch):
    """Run one forward pass and print the key diagnostics."""
    n_params = sum(p.numel() for p in model.parameters())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out = model(x, params)
    out_tensor = out["denoised"] if isinstance(out, dict) else out
    has_nan = bool(torch.isnan(out_tensor).any())
    has_inf = bool(torch.isinf(out_tensor).any())
    vram = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
    print(f"{name:12s}  params={n_params/1e6:6.2f}M  "
          f"in={tuple(x.shape)}  out={tuple(out_tensor.shape)}  "
          f"dtype={out_tensor.dtype}  nan={has_nan}  inf={has_inf}  "
          f"VRAM={vram:.0f}MB")
    return out_tensor


@app.cell
def _n2v3d(
    CONFIGS, NOISE_LEVELS, build_model, device, load_config, torch,
):
    _cfg = load_config(CONFIGS / "n2v3d.yaml")
    _m = build_model(_cfg.model).to(device).eval()
    _T, _H, _W = _cfg.data.patch
    _x = torch.randn(1, 1, _T, _H, _W, device=device)
    report("n2v3d", _m, _x, NOISE_LEVELS[1], device, torch)
    del _m; torch.cuda.empty_cache() if device.type == "cuda" else None
    return


@app.cell
def _deepcad(
    CONFIGS, NOISE_LEVELS, build_model, device, load_config, torch,
):
    _cfg = load_config(CONFIGS / "deepcad.yaml")
    _m = build_model(_cfg.model).to(device).eval()
    _T, _H, _W = _cfg.data.patch
    _x = torch.randn(1, 1, _T, _H, _W, device=device)
    report("deepcad", _m, _x, NOISE_LEVELS[1], device, torch)
    del _m; torch.cuda.empty_cache() if device.type == "cuda" else None
    return


@app.cell
def _deepinterp(
    CONFIGS, NOISE_LEVELS, build_model, device, load_config, torch,
):
    _cfg = load_config(CONFIGS / "deepinterp.yaml")
    _m = build_model(_cfg.model).to(device).eval()
    _T, _H, _W = _cfg.data.patch
    _x = torch.randn(1, 1, _T, _H, _W, device=device)
    report("deepinterp", _m, _x, NOISE_LEVELS[1], device, torch)
    del _m; torch.cuda.empty_cache() if device.type == "cuda" else None
    return


@app.cell
def _mamba3d(
    CONFIGS, NOISE_LEVELS, build_model, device, load_config, torch,
):
    """Will fail if the Mamba CUDA kernels aren't built. That is the
    point of running this cell in isolation — it tells you *which*
    model is broken without disturbing the others."""
    try:
        _cfg = load_config(CONFIGS / "mamba3d.yaml")
        _m = build_model(_cfg.model).to(device).eval()
        _T, _H, _W = _cfg.data.patch
        _x = torch.randn(1, 1, _T, _H, _W, device=device)
        report("mamba3d", _m, _x, NOISE_LEVELS[1], device, torch)
        del _m
    except Exception as _e:
        print(f"mamba3d FAILED: {type(_e).__name__}: {_e}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return


@app.cell
def _pinn(
    CONFIGS, NOISE_LEVELS, build_model, device, load_config, torch,
):
    """PINN wraps a 3-D backbone; output is a dict with 'denoised'."""
    try:
        _cfg = load_config(CONFIGS / "pinn.yaml")
        _m = build_model(_cfg.model).to(device).eval()
        _T, _H, _W = _cfg.data.patch
        _x = torch.randn(1, 1, _T, _H, _W, device=device)
        with torch.no_grad():
            _out = _m(_x, NOISE_LEVELS[1])
        print("pinn dict keys:", sorted(_out.keys()))
        for _k, _v in _out.items():
            print(f"  {_k:14s} shape={tuple(_v.shape)}")
        del _m
    except Exception as _e:
        print(f"pinn FAILED: {type(_e).__name__}: {_e}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return


if __name__ == "__main__":
    app.run()
