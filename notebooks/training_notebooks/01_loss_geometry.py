"""01 — Loss geometry.

What this notebook is for
-------------------------
Demonstrate *exactly* what ``cidc.losses.poisson_gaussian_nll`` and
``cidc.losses.anscombe_mse`` minimise, so nobody ever again calls them
with the wrong argument convention (which is what broke the earlier
inline sanity-check — the script passed a raw-ADU ``pred`` and ``tgt``
to ``anscombe_mse``, but that function expects Anscombe-space tensors).

Claims being verified against `cidc` source of truth:

1. The **PG-NLL argmin in mu** is at ``mu* ≈ y - g/2`` (tiny
   heteroscedastic pull, documented in ``losses.py`` lines 46-49).
2. The **bias-sweep minimum** (adding a constant ``b`` to clean and
   scoring against a noisy observation) sits near ``b ≈ 0`` only when
   the sweep is *centred* on the truth; the earlier script's bounds of
   ±40 were just the *grid edge*, not a real bias.
3. ``anscombe_mse(z_pred, z_target)`` takes **Anscombe-space** tensors.
   Calling it with raw ADU happens to return a finite number but
   geometrically minimises the wrong objective.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _setup():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from cidc import (
        NOISE_LEVELS,
        anscombe,
        anscombe_mse,
        poisson_gaussian_nll,
        sample_poisson_gaussian,
    )

    rng = np.random.default_rng(0)
    return (
        NOISE_LEVELS,
        anscombe,
        anscombe_mse,
        np,
        plt,
        poisson_gaussian_nll,
        rng,
        sample_poisson_gaussian,
        torch,
    )


@app.cell
def _toy_signal(NOISE_LEVELS, np, rng, sample_poisson_gaussian):
    """Toy clean frame + one noisy realisation at noise level 2."""
    _yy, _xx = np.mgrid[:128, :128].astype(np.float32)
    clean = (
        300.0 * np.exp(-((_xx - 40) ** 2 + (_yy - 40) ** 2) / (2 * 8**2))
        + 600.0 * np.exp(-((_xx - 90) ** 2 + (_yy - 60) ** 2) / (2 * 12**2))
        + 150.0
    ).astype(np.float32)
    params = NOISE_LEVELS[2]
    noisy = sample_poisson_gaussian(clean, params, rng=rng).astype(np.float32)
    print(f"clean.mean()={clean.mean():.1f}  noisy.mean()={noisy.mean():.1f}  "
          f"expected |diff| ~ sqrt(V)/sqrt(N) = "
          f"{np.sqrt(params.gain * clean.mean() + params.read_var) / np.sqrt(clean.size):.2f}")
    return clean, noisy, params


@app.cell
def _pgnll_argmin_in_mu(np, plt, poisson_gaussian_nll, params, torch):
    """Claim 1 from the docstring: argmin_mu PG-NLL(mu; y) ≈ y - g/2.

    We sweep a *scalar* mu against a *scalar* y and locate the minimum.
    This is element-wise, so we can verify per-pixel behaviour cleanly.
    """
    _y_scalar = 500.0
    _mu_grid = np.linspace(_y_scalar - 200, _y_scalar + 200, 4001)
    _losses = np.array([
        float(poisson_gaussian_nll(
            torch.tensor(_m), torch.tensor(_y_scalar),
            gain=params.gain, read_var=params.read_var,
        ))
        for _m in _mu_grid
    ])
    _mu_star = float(_mu_grid[_losses.argmin()])
    print(f"y = {_y_scalar}")
    print(f"argmin_mu = {_mu_star:.2f}")
    print(f"predicted y - g/2 = {_y_scalar - params.gain/2:.2f}")
    print(f"difference from prediction: {_mu_star - (_y_scalar - params.gain/2):+.2f} ADU")

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_mu_grid, _losses)
    _ax.axvline(_mu_star, color="r", lw=0.8, label=f"argmin={_mu_star:.0f}")
    _ax.axvline(_y_scalar, color="k", lw=0.8, ls="--", label=f"y={_y_scalar:.0f}")
    _ax.set_xlabel("mu"); _ax.set_ylabel("PG-NLL")
    _ax.set_title(f"PG-NLL argmin is y - g/2, not y  (gain={params.gain:.1f})")
    _ax.legend()
    _fig
    return


@app.cell
def _bias_sweep_centred(
    anscombe,
    anscombe_mse,
    clean,
    noisy,
    np,
    params,
    plt,
    poisson_gaussian_nll,
    torch,
):
    """Claim 2: bias sweep with correct API usage.

    Earlier inline script symptoms:
      'Anscombe-MSE optimum bias: -40.00'  <-- this was the grid edge
      'PG-NLL optimum bias:       +40.00'  <-- same, other edge
    Widen the grid and the minimum moves inside.

    This cell also shows the **correct** call:
      - PG-NLL: pass ADU (mu, y, gain, read_var).
      - Anscombe-MSE: convert BOTH sides to Anscombe space first.
    """
    _grid = np.linspace(-200, 200, 401)

    _ans_y = anscombe(noisy, params)
    _amse = np.array([
        float(anscombe_mse(
            torch.from_numpy(anscombe(clean + _b, params)).float(),
            torch.from_numpy(_ans_y).float(),
        ))
        for _b in _grid
    ])

    _pg = np.array([
        float(poisson_gaussian_nll(
            torch.from_numpy(clean + _b).float(),
            torch.from_numpy(noisy).float(),
            gain=params.gain, read_var=params.read_var,
        ))
        for _b in _grid
    ])

    print(f"Anscombe-MSE argmin: {_grid[_amse.argmin()]:+.2f}")
    print(f"PG-NLL       argmin: {_grid[_pg.argmin()]:+.2f}")
    print("Both should sit near 0 ± a few ADU.")

    _fig, _ax = plt.subplots(1, 2, figsize=(9, 3))
    _ax[0].plot(_grid, _amse); _ax[0].axvline(0, color="k", lw=0.5)
    _ax[0].set_title(f"anscombe_mse  argmin={_grid[_amse.argmin()]:+.1f}")
    _ax[0].set_xlabel("bias b"); _ax[0].set_ylabel("loss")
    _ax[1].plot(_grid, _pg); _ax[1].axvline(0, color="k", lw=0.5)
    _ax[1].set_title(f"poisson_gaussian_nll  argmin={_grid[_pg.argmin()]:+.1f}")
    _ax[1].set_xlabel("bias b")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _wrong_vs_right_anscombe(
    anscombe, anscombe_mse, clean, noisy, np, params, plt, torch,
):
    """Claim 3: passing raw ADU to anscombe_mse is a *silent* error.

    The function happily runs and returns a number, but the argmin of
    that number is at a different location than the 'real' Anscombe-
    space objective. This is the exact trap the earlier script fell
    into.
    """
    _grid = np.linspace(-100, 100, 201)

    # WRONG: feed raw ADU to anscombe_mse.
    _wrong = np.array([
        float(anscombe_mse(
            torch.from_numpy((clean + _b).astype(np.float32)),
            torch.from_numpy(noisy.astype(np.float32)),
        ))
        for _b in _grid
    ])
    # RIGHT: feed Anscombe-space tensors.
    _right = np.array([
        float(anscombe_mse(
            torch.from_numpy(anscombe(clean + _b, params)).float(),
            torch.from_numpy(anscombe(noisy, params)).float(),
        ))
        for _b in _grid
    ])

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.plot(_grid, _wrong / _wrong.max(), label="WRONG (raw ADU in)")
    _ax.plot(_grid, _right / _right.max(), label="RIGHT (Anscombe in)")
    _ax.axvline(0, color="k", lw=0.5)
    _ax.set_xlabel("bias b"); _ax.set_ylabel("loss / max")
    _ax.set_title(
        f"wrong argmin={_grid[_wrong.argmin()]:+.0f}   "
        f"right argmin={_grid[_right.argmin()]:+.0f}"
    )
    _ax.legend()
    _fig
    return


@app.cell
def _constant_fit_mse(np, noisy):
    """Sanity: plain MSE of a constant predictor against y is minimised
    at mean(y). Used as the baseline 'dumb' loss we must beat."""
    _opt = float(noisy.mean())
    print(f"MSE constant-model optimum = {_opt:.2f}   (= mean(noisy))")
    return


if __name__ == "__main__":
    app.run()
