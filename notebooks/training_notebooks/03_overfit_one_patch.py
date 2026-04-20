"""03 — Overfit one patch.

What this notebook is for
-------------------------
The oldest ML debugger: if a model + loss + optimizer cannot drive the
loss to near-zero on a **single** synthetic patch, nothing about batch
size / learning-rate schedule / augmentation will help on the real
dataset.

This notebook lets you pick a model config and a loss, then runs a
tight training loop on *one* patch with known ground truth and plots
the loss curve.

What to watch for (subtle traps, not bugs):

- **PG-NLL floor is not zero.** The loss converges to ``0.5 log V`` at
  the point estimator, not to 0. A curve plateauing near ~4 is normal
  at gain≈249.
- **Anscombe-MSE input domain.** We feed the model Anscombe-space
  volumes, so the *loss* must also be computed in Anscombe space. The
  wrapper below reuses ``cidc.losses.anscombe_mse`` correctly.
- **Blind-spot masking.** ``n2v3d`` relies on masked inputs in real
  training. Here we bypass the mask because the goal is to check the
  architecture's *capacity*, not its self-supervision.
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
        anscombe,
        anscombe_mse,
        build_model,
        load_config,
        poisson_gaussian_nll,
        sample_poisson_gaussian,
    )

    CONFIGS = Path("/app/workspace/configs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        CONFIGS,
        NOISE_LEVELS,
        anscombe,
        anscombe_mse,
        build_model,
        device,
        load_config,
        np,
        plt,
        poisson_gaussian_nll,
        sample_poisson_gaussian,
        torch,
    )


@app.cell
def _make_patch(NOISE_LEVELS, np, sample_poisson_gaussian):
    """One clean 3-D patch with a few calcium-like transients."""
    _rng = np.random.default_rng(0)
    _T, _H, _W = 32, 64, 64
    _yy, _xx = np.mgrid[:_H, :_W].astype(np.float32)
    _spatial = (
        400.0 * np.exp(-((_xx - 18) ** 2 + (_yy - 22) ** 2) / (2 * 5**2))
        + 250.0 * np.exp(-((_xx - 45) ** 2 + (_yy - 40) ** 2) / (2 * 7**2))
    )
    _t = np.arange(_T)[:, None, None].astype(np.float32)
    _envelope = np.exp(-((_t - 12.0) ** 2) / (2 * 6.0**2))
    clean = (100.0 + _spatial[None] * _envelope).astype(np.float32)
    params = NOISE_LEVELS[2]
    noisy = sample_poisson_gaussian(clean, params, rng=_rng).astype(np.float32)
    return clean, noisy, params


@app.cell
def _config_pick(CONFIGS, load_config):
    """Swap this to overfit a different backbone.

    n2v3d is the cheapest to overfit; deepcad / mamba3d cost more.
    """
    cfg = load_config(CONFIGS / "n2v3d.yaml")
    print(f"overfitting model = {cfg.model.name}")
    return (cfg,)


@app.cell
def _train_loop(
    anscombe,
    anscombe_mse,
    build_model,
    cfg,
    clean,
    device,
    noisy,
    np,
    params,
    plt,
    poisson_gaussian_nll,
    torch,
):
    """Run ~300 steps. Two losses side-by-side for comparison."""
    _model = build_model(cfg.model).to(device).train()

    _x = torch.from_numpy(anscombe(noisy, params)).float().to(device)[None, None]
    _y = torch.from_numpy(noisy).float().to(device)[None, None]
    _clean = torch.from_numpy(clean).float().to(device)[None, None]

    _opt = torch.optim.AdamW(_model.parameters(), lr=3e-4, weight_decay=1e-4)
    _pgnll_hist, _amse_hist, _mse_vs_clean_hist = [], [], []

    for _step in range(300):
        _opt.zero_grad()
        _mu = _model(_x, params)                      # raw ADU
        _loss = poisson_gaussian_nll(
            _mu, _y, gain=params.gain, read_var=params.read_var,
        )
        _loss.backward()
        _opt.step()

        with torch.no_grad():
            _pgnll_hist.append(float(_loss))
            _amse_hist.append(float(anscombe_mse(
                torch.from_numpy(
                    anscombe(_mu.squeeze().cpu().numpy(), params)
                ).float().to(device),
                torch.from_numpy(anscombe(noisy, params)).float().to(device),
            )))
            _mse_vs_clean_hist.append(float(((_mu - _clean) ** 2).mean()))

    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3))
    _ax[0].plot(_pgnll_hist, label="PG-NLL (train)")
    _ax[0].plot(_amse_hist, label="Anscombe-MSE", alpha=0.6)
    _ax[0].set_xlabel("step"); _ax[0].set_ylabel("loss"); _ax[0].legend()
    _ax[0].set_title("training losses")
    _ax[1].semilogy(_mse_vs_clean_hist)
    _ax[1].set_xlabel("step"); _ax[1].set_ylabel("MSE(pred, clean)")
    _ax[1].set_title("true supervision error (log-scale)")
    _fig.tight_layout()

    print(f"final PG-NLL         = {_pgnll_hist[-1]:.3f}")
    print(f"final MSE vs clean   = {_mse_vs_clean_hist[-1]:.1f}")
    print(f"expected PG-NLL floor ≈ 0.5*log(g*mean+read_var) "
          f"= {0.5*np.log(params.gain*clean.mean()+params.read_var):.3f}")
    _fig
    return


@app.cell
def _show_prediction(
    anscombe, build_model, cfg, clean, device, noisy, np, params, plt, torch,
):
    """Re-train briefly and show (clean, noisy, pred, residual) at t=12."""
    _model = build_model(cfg.model).to(device).train()
    _x = torch.from_numpy(anscombe(noisy, params)).float().to(device)[None, None]
    _y = torch.from_numpy(noisy).float().to(device)[None, None]
    _opt = torch.optim.AdamW(_model.parameters(), lr=3e-4)
    from cidc import poisson_gaussian_nll as _pgn
    for _ in range(300):
        _opt.zero_grad()
        _mu = _model(_x, params)
        _pgn(_mu, _y, gain=params.gain, read_var=params.read_var).backward()
        _opt.step()
    _model.eval()
    with torch.no_grad():
        _pred = _model(_x, params).squeeze().cpu().numpy()

    _t = 12
    _fig, _ax = plt.subplots(1, 4, figsize=(12, 3))
    _vmax = float(np.percentile(clean[_t], 99))
    for _i, (_img, _title) in enumerate([
        (clean[_t],            "clean"),
        (noisy[_t],            "noisy"),
        (_pred[_t],            "pred"),
        (_pred[_t] - clean[_t],"pred - clean"),
    ]):
        _cmap = "seismic" if _i == 3 else "gray"
        _vm = max(abs(_img).max(), 1.0) if _i == 3 else _vmax
        _kw = dict(vmin=-_vm, vmax=_vm) if _i == 3 else dict(vmin=0, vmax=_vmax)
        _ax[_i].imshow(_img, cmap=_cmap, **_kw); _ax[_i].set_title(_title)
        _ax[_i].axis("off")
    _fig
    return


if __name__ == "__main__":
    app.run()
