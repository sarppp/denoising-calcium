"""05 — Tile seams.

What this notebook is for
-------------------------
``cidc.eval.denoise_stack`` runs the model over overlapping 3-D tiles
and cosine-blends the seams. If ``tile`` / ``overlap`` are chosen
badly — or if the blending is ever subtly broken — we get horizontal /
vertical lines in the output that look like model artefacts but are
actually just the tiling.

This notebook:

1. Makes a synthetic noisy cube (no real data required).
2. Runs ``denoise_stack`` with two different tile / overlap settings
   using the same model + weights.
3. Shows the difference map and the per-row / per-column standard
   deviation of that difference — any peak = a seam.

If both configs produce the same output (up to ~1 ADU) the blending is
healthy. If they diverge, the run is being tile-aliased and you need
to either (a) increase overlap, or (b) reduce tile size so the
receptive field is fully contained.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _setup():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from cidc import (
        NOISE_LEVELS,
        build_model,
        denoise_stack,
        load_config,
        sample_poisson_gaussian,
    )

    CONFIGS = Path("/app/workspace/configs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        CONFIGS,
        NOISE_LEVELS,
        build_model,
        denoise_stack,
        device,
        load_config,
        np,
        plt,
        sample_poisson_gaussian,
        torch,
    )


@app.cell
def _make_cube(NOISE_LEVELS, np, sample_poisson_gaussian):
    """T=64, H=W=256 synthetic cube — big enough to need tiling."""
    _rng = np.random.default_rng(0)
    _T, _H, _W = 64, 256, 256
    _yy, _xx = np.mgrid[:_H, :_W].astype(np.float32)
    clean = (
        400.0 * np.exp(-((_xx - 80) ** 2 + (_yy - 60) ** 2) / (2 * 10**2))
        + 250.0 * np.exp(-((_xx - 200) ** 2 + (_yy - 180) ** 2) / (2 * 14**2))
        + 180.0
    )
    clean = np.broadcast_to(clean, (_T, _H, _W)).astype(np.float32).copy()
    clean += 100.0 * np.sin(2 * np.pi * np.arange(_T)[:, None, None] / 30.0)
    params = NOISE_LEVELS[2]
    noisy = sample_poisson_gaussian(clean, params, rng=_rng).astype(np.float32)
    return clean, noisy, params


@app.cell
def _run_two_configs(
    CONFIGS,
    build_model,
    denoise_stack,
    device,
    load_config,
    noisy,
    params,
    torch,
):
    """Same RANDOM-init model, two tiling configurations.

    Random weights are fine — we only care about *self-consistency* of
    the tiling, not about denoising quality.
    """
    _cfg = load_config(CONFIGS / "n2v3d.yaml")
    _model = build_model(_cfg.model).to(device).eval()

    pred_small = denoise_stack(
        _model, noisy, params,
        tile=(32, 64, 64), overlap=(8, 16, 16),
        device=device, amp=False,
    ).astype("float32")
    pred_large = denoise_stack(
        _model, noisy, params,
        tile=(64, 128, 128), overlap=(16, 32, 32),
        device=device, amp=False,
    ).astype("float32")
    return pred_large, pred_small


@app.cell
def _inspect_seams(np, plt, pred_large, pred_small):
    """Where do the two configs disagree? Peaks = tile boundaries."""
    _diff = pred_small - pred_large
    _t = _diff.shape[0] // 2

    _fig, _ax = plt.subplots(1, 3, figsize=(12, 4))
    _v = max(1.0, float(np.abs(_diff[_t]).max()))
    _ax[0].imshow(_diff[_t], cmap="seismic", vmin=-_v, vmax=_v)
    _ax[0].set_title(f"pred_small - pred_large  (t={_t})  |max|={_v:.1f}")
    _ax[0].axis("off")

    _ax[1].plot(_diff[_t].std(axis=1)); _ax[1].set_xlabel("row")
    _ax[1].set_ylabel("std(diff) along H"); _ax[1].set_title("row-wise seam signature")
    _ax[2].plot(_diff[_t].std(axis=0)); _ax[2].set_xlabel("col")
    _ax[2].set_ylabel("std(diff) along W"); _ax[2].set_title("col-wise seam signature")
    _fig.tight_layout()

    print(f"max |pred_small - pred_large| = {np.abs(_diff).max():.2f} ADU")
    print(f"mean |pred_small - pred_large| = {np.abs(_diff).mean():.2f} ADU")
    print("Acceptable: ~ a few ADU.  Problematic: sharp lines / tens of ADU.")
    _fig
    return


if __name__ == "__main__":
    app.run()
