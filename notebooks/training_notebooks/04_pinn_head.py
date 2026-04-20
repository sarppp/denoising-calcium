"""04 — PINN head sanity.

What this notebook is for
-------------------------
The PINN head predicts (τ, baseline, s(t)) per pixel and reconstructs a
trace with ``euler_forward``. Instead of waiting to see if a full
training run recovers sensible kinetics, we feed in a **synthetic trace
with known τ, baseline, and events** and ask:

1. Does ``euler_forward`` (the ODE integrator) round-trip the ground
   truth when fed the true (τ, b, s)?  (It must — otherwise the loss
   it regularises is mis-specified.)
2. Does freezing the backbone and training *only* the PINN head on
   a clean trace drive the kinetics loss down?
3. Does the τ prediction land inside the ``tau_range`` prior and near
   the true τ? How sensitive is it to ``loss.aux.pinn.weight``?

The idea is: if any of these fails on a **clean** synthetic trace, no
amount of tuning on real noisy stacks will fix it.
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
        calcium_kinetics_loss,
        load_config,
    )
    from cidc.models.pinn.kinetics import euler_forward

    CONFIGS = Path("/app/workspace/configs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        CONFIGS,
        NOISE_LEVELS,
        build_model,
        calcium_kinetics_loss,
        device,
        euler_forward,
        load_config,
        np,
        plt,
        torch,
    )


@app.cell
def _synthetic_truth(device, euler_forward, np, torch):
    """Build a (tiny) ground-truth kinetics tensor set.

    H, W small so this is fast; T = 32 matches the default PINN config.
    Two pixels have different τ; others are silent (s = 0, baseline 100).
    """
    _B, _T, _H, _W = 1, 32, 8, 8
    tau_true = torch.full((_B, 1, 1, _H, _W), 40.0, device=device)
    tau_true[..., 2, 2] = 15.0
    tau_true[..., 5, 5] = 80.0
    baseline_true = torch.full((_B, 1, 1, _H, _W), 100.0, device=device)
    source_true = torch.zeros(_B, 1, _T, _H, _W, device=device)
    source_true[:, :, 10, 2, 2] = 300.0     # impulse at t=10 pixel (2,2)
    source_true[:, :, 15, 5, 5] = 200.0     # impulse at t=15 pixel (5,5)
    F_true = euler_forward(tau_true, baseline_true, source_true, dt=1.0)
    print(f"F_true shape = {tuple(F_true.shape)}   "
          f"range [{float(F_true.min()):.1f}, {float(F_true.max()):.1f}]")
    return F_true, baseline_true, source_true, tau_true


@app.cell
def _integrator_roundtrip(F_true, baseline_true, euler_forward, np, plt,
                          source_true, tau_true):
    """Claim 1: the integrator is exact on its own output."""
    _F2 = euler_forward(tau_true, baseline_true, source_true, dt=1.0)
    _err = (_F2 - F_true).abs().max().item()
    print(f"max|euler_forward(τ, b, s) - F_true| = {_err:.2e}   "
          "(should be 0 up to float precision)")

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(F_true[0, 0, :, 2, 2].cpu().numpy(), label="pixel (2,2) τ=15")
    _ax.plot(F_true[0, 0, :, 5, 5].cpu().numpy(), label="pixel (5,5) τ=80")
    _ax.plot(F_true[0, 0, :, 0, 0].cpu().numpy(), label="silent pixel (0,0)")
    _ax.set_xlabel("t"); _ax.set_ylabel("F(t)"); _ax.legend()
    _ax.set_title("Synthetic ground-truth kinetics traces")
    _fig
    return


@app.cell
def _fit_pinn_head(
    CONFIGS,
    F_true,
    NOISE_LEVELS,
    build_model,
    calcium_kinetics_loss,
    device,
    euler_forward,
    load_config,
    np,
    plt,
    source_true,
    tau_true,
    torch,
):
    """Claim 2 + 3: freeze backbone, train PINN head to reconstruct F_true.

    We use ``cidc.build_model`` + ``configs/pinn.yaml`` unmodified. The
    'noisy input' fed to the backbone is the clean F_true itself — this
    isolates the head's ability to fit (τ, b, s) from the denoiser.
    """
    _cfg = load_config(CONFIGS / "pinn.yaml")
    # Shrink spatial to match the toy ground truth.
    _cfg.model.kwargs["backbone"]["kwargs"]["base_ch"] = 8
    _cfg.model.kwargs["backbone"]["kwargs"]["depth"] = 2
    _model = build_model(_cfg.model).to(device)

    # Freeze the denoising backbone — we only want to grade the head.
    for _p in _model.backbone.parameters():
        _p.requires_grad_(False)

    _opt = torch.optim.Adam(
        [p for p in _model.parameters() if p.requires_grad], lr=1e-2,
    )

    # We need the head output to match F_true. Feed F_true as 'noisy input'
    # after Anscombe-ing it (the backbone expects that) — the backbone is
    # frozen so it doesn't matter, only the *features* tapped by the head.
    from cidc.noise import anscombe as _anscombe_np
    _x_an = torch.from_numpy(
        _anscombe_np(F_true.squeeze().cpu().numpy(), NOISE_LEVELS[1])
    ).float().to(device)[None, None]

    _losses = []
    for _step in range(400):
        _opt.zero_grad()
        _out = _model(_x_an, NOISE_LEVELS[1])
        # The kinetics loss drives reconstruction -> denoised; here we
        # want reconstruction -> F_true, so we call euler_forward again
        # using the head's predictions and compare to F_true directly.
        _rec = euler_forward(_out["tau"], _out["baseline"], _out["source"], dt=1.0)
        _loss = (
            ((_rec - F_true) ** 2).mean()
            + 0.005 * _out["source"].abs().mean()
        )
        _loss.backward()
        _opt.step()
        _losses.append(float(_loss))

    # Final predictions.
    with torch.no_grad():
        _out = _model(_x_an, NOISE_LEVELS[1])
    _tau_pred = _out["tau"].squeeze().cpu().numpy()
    _bas_pred = _out["baseline"].squeeze().cpu().numpy()

    print(f"final loss = {_losses[-1]:.4f}")
    print(f"τ true      (2,2)={float(tau_true[...,2,2]):.1f}  "
          f"pred={_tau_pred[2,2]:.1f}")
    print(f"τ true      (5,5)={float(tau_true[...,5,5]):.1f}  "
          f"pred={_tau_pred[5,5]:.1f}")
    print(f"τ prior range = {_model.pinn_head.tau_min}..{_model.pinn_head.tau_max}  "
          f"pred range = [{_tau_pred.min():.1f}, {_tau_pred.max():.1f}]")
    print(f"baseline bias = {_bas_pred.mean() - 100.0:+.2f} "
          "(true baseline = 100)")

    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3))
    _ax[0].semilogy(_losses); _ax[0].set_xlabel("step")
    _ax[0].set_ylabel("PINN-head fit loss"); _ax[0].set_title("convergence")
    _ax[1].imshow(_tau_pred, cmap="viridis"); _ax[1].set_title("predicted τ map")
    _fig.colorbar(_ax[1].images[0], ax=_ax[1], fraction=0.046)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
