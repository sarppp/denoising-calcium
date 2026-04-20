"""10 — Learn the physics, the noise, and the losses.

A self-contained tutorial for anyone new to:
- calcium imaging, shot noise, read noise,
- the Anscombe variance-stabilising transform,
- loss functions used in Poisson-Gaussian denoising,
- how to design and validate your own loss.

Runs entirely from inside this repo. No external downloads.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _setup():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from cidc import load_stack
    from cidc.noise import (
        NOISE_LEVELS,
        FILE_NOISE,
        NoiseParams,
        anscombe,
        inverse_anscombe,
        sample_poisson_gaussian,
    )
    from cidc.losses import anscombe_mse, poisson_gaussian_nll

    DATA = Path("/app/workspace/data")
    return (
        DATA,
        NOISE_LEVELS,
        anscombe,
        anscombe_mse,
        load_stack,
        mo,
        np,
        plt,
        poisson_gaussian_nll,
        sample_poisson_gaussian,
        torch,
    )


@app.cell
def _intro1(mo):
    mo.md("""
    # 1. What are we denoising?

    **Calcium imaging** records neural activity by genetically encoding a
    fluorescent indicator (GCaMP) into neurons. When a neuron fires, Ca²⁺
    ions flood in, bind the indicator, and it glows brighter. A microscope
    camera records the glow at ~30 Hz.

    Each video pixel is a count of photons that hit a sensor well during
    one exposure. Two fundamental things corrupt that count:

    1. **Photon shot noise** — the *arrival* of photons at the sensor is a
       Poisson process. Even a perfectly clean signal has variance equal
       to its mean. Shot noise is physics, not engineering — you can't
       remove it by building a better camera.
    2. **Read noise** — the electronic readout of each well adds a
       roughly-Gaussian offset with variance σ_r² that does not depend on
       signal.

    The **job of a denoiser** is to use spatial + temporal correlations in
    the signal (neurons vary slowly; nearby pixels are correlated) to
    average out noise, without blurring the signal itself.
    """)
    return


@app.cell
def _look_raw(DATA, load_stack, np, plt):
    """Look at the clean reference (F0) vs noisy realisations (F1, F2, F3)."""
    _stacks = {name: load_stack(DATA / "val" / f"{name}.tif")[500]
               for name in ("F0", "F1", "F2", "F3")}
    _fig, _axes = plt.subplots(1, 4, figsize=(14, 3.6))
    _vmin, _vmax = np.percentile(np.asarray(_stacks["F0"]), [1, 99.5])
    for _ax, (_name, _img) in zip(_axes, _stacks.items()):
        _ax.imshow(_img, vmin=_vmin, vmax=_vmax, cmap="magma")
        _ax.set_title(f"{_name}\nμ={float(np.asarray(_img).mean()):.1f}")
        _ax.axis("off")
    _fig.suptitle("Same scene, 4 noise levels — frame 500", y=1.02)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _info(mo):
    mo.md("""
    F0 is the **long-exposure "clean" reference** (still has some residual
    noise). F1, F2, F3 are three independent noisy acquisitions of the
    *same* scene at increasing gain (shorter exposures / more
    amplification). You can already see:

    - **F1** looks almost as clean as F0.
    - **F2** is visibly noisier.
    - **F3** is dominated by grain — this is the Task-2 out-of-distribution
      level the challenge cares about most.

    That's what our model has to undo.
    """)
    return


@app.cell
def _intro2(mo):
    mo.md(r"""
    # 2. Poisson shot noise from first principles

    A photon counter records a random number of photons per exposure. If
    the true rate is λ, the count $N$ is **Poisson-distributed**:

    $$P(N=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

    Two crucial properties:

    - $\mathbb{E}[N] = \lambda$
    - $\mathrm{Var}[N] = \lambda$

    So **standard deviation grows like $\sqrt{\lambda}$**. Brighter pixels
    are noisier in absolute terms but *less* noisy in relative terms
    (SNR ∝ √λ).

    Let's see this directly.
    """)
    return


@app.cell
def _poisson_demo(np, plt):
    """Simulate the 'variance = mean' law."""
    _rng = np.random.default_rng(0)
    _lambdas = np.array([1, 5, 20, 100, 500])
    _samples = np.stack([_rng.poisson(lam, size=5000) for lam in _lambdas])
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for _lam, _s in zip(_lambdas, _samples):
        _axes[0].hist(_s, bins=40, alpha=0.5, label=f"λ={_lam}")
    _axes[0].set_xlabel("observed count")
    _axes[0].set_ylabel("frequency")
    _axes[0].set_title("Poisson histograms")
    _axes[0].legend()

    _means = _samples.mean(axis=1)
    _vars = _samples.var(axis=1)
    _axes[1].plot(_means, _vars, "o-", label="measured")
    _axes[1].plot(_means, _means, "k--", label="y = x")
    _axes[1].set_xlabel("mean μ")
    _axes[1].set_ylabel("variance σ²")
    _axes[1].set_xscale("log"); _axes[1].set_yscale("log")
    _axes[1].set_title("Poisson: Var = Mean")
    _axes[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _poisson_note(mo):
    mo.md("""
    **Takeaway:** with pure Poisson noise, the variance *equals* the mean.
    That has huge implications for loss functions — L2 loss, which assumes
    **constant variance**, is statistically wrong for Poisson data. Bright
    pixels get weighted proportionally more than they should, dim pixels
    under-weighted.
    """)
    return


@app.cell
def _intro3(mo):
    mo.md(r"""
    # 3. The Poisson-Gaussian camera model

    A scientific CMOS camera doesn't give you raw photon counts. It gives
    you **ADU** (analog-to-digital units). The transform is:

    $$y = g \cdot N + \varepsilon, \quad N \sim \mathrm{Poisson}(\mu / g),
        \quad \varepsilon \sim \mathcal{N}(0, \sigma_r^2)$$

    where:
    - $\mu$ is the *ideal* signal you'd see with infinite photons (ADU).
    - $g$ is the **gain** (ADU per photon). Higher gain → shorter exposure
      needed → more amplified shot noise.
    - $\sigma_r^2$ is the **read-noise variance** (ADU²) — a fixed
      electronic cost per readout.

    From this:
    $$\mathbb{E}[y] = \mu, \qquad \mathrm{Var}[y] = g\,\mu + \sigma_r^2$$

    The variance is **affine in the mean** with slope $g$ and intercept
    $\sigma_r^2$. That's the signature you look for when fitting noise
    parameters from data.
    """)
    return


@app.cell
def _pg_sim(NOISE_LEVELS, np, plt, sample_poisson_gaussian):
    """Plot the measured (μ, σ²) line for each CIDC noise level."""
    _mu_grid = np.linspace(0, 4000, 40)
    _mu_img = np.broadcast_to(_mu_grid[:, None, None], (len(_mu_grid), 100, 100)).astype(np.float32)
    _rng = np.random.default_rng(0)

    _fig, _ax = plt.subplots(figsize=(6, 4))
    for _lvl in (1, 2, 3):
        _p = NOISE_LEVELS[_lvl]
        _y = sample_poisson_gaussian(_mu_img, _p, rng=_rng)
        _var = _y.reshape(len(_mu_grid), -1).var(axis=1)
        _ax.plot(_mu_grid, _var, "o-", label=f"level {_lvl}: g={_p.gain:.1f}, σr²={_p.read_var:.0f}")
        _ax.plot(_mu_grid, _p.gain * _mu_grid + _p.read_var, "--", alpha=0.4)
    _ax.set_xlabel("μ (true signal, ADU)")
    _ax.set_ylabel("measured variance σ² (ADU²)")
    _ax.set_title("Camera model: Var = g·μ + σ_r²")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _pg_real(DATA, load_stack, np, plt):
    """Verify: the real F1 stack follows the same line."""
    _s = np.asarray(load_stack(DATA / "val" / "F1.tif")[:256], dtype=np.float32)
    _mean = _s.mean(axis=0)
    _var = _s.var(axis=0)
    _order = np.argsort(_mean.ravel())
    _m, _v = _mean.ravel()[_order], _var.ravel()[_order]
    # Bin (μ, σ²) to see the trend clearly.
    _bins = np.linspace(_m.min(), _m.max(), 50)
    _idx = np.digitize(_m, _bins)
    _mx = np.array([_m[_idx == i].mean() for i in range(1, len(_bins)) if (_idx == i).any()])
    _vx = np.array([_v[_idx == i].mean() for i in range(1, len(_bins)) if (_idx == i).any()])
    _slope, _intercept = np.polyfit(_mx, _vx, 1)

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(_mx, _vx, "o", ms=4, label="F1 binned pixels")
    _ax.plot(_mx, _slope * _mx + _intercept, "k--",
             label=f"fit: g≈{_slope:.2f}, σr²≈{_intercept:.0f}")
    _ax.set_xlabel("per-pixel mean across time (ADU)")
    _ax.set_ylabel("per-pixel variance across time (ADU²)")
    _ax.set_title("F1 pixels obey Var = g·μ + σ_r² (camera-model validation)")
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _intro4(mo):
    mo.md(r"""
    # 4. Anscombe: turning Poisson-Gaussian into (approximately) Gaussian

    The problem with Poisson-Gaussian data: **variance depends on the
    mean**. That's bad for neural networks because:

    - MSE is equivalent to Gaussian MLE only for *constant* variance.
    - Gradient scales vary wildly between dim and bright pixels.
    - Batch-norm assumes homoscedastic inputs.

    The **generalised Anscombe transform** (Foi et al. 2008; Mäkitalo &
    Foi 2013) warps the data so variance becomes ≈ 1 everywhere:

    $$z = \frac{2}{g}\sqrt{\max\!\left(0,\, g\,y + \tfrac{3}{8}g^{2} + \sigma_r^2\right)}$$

    After this transform: $\mathrm{Var}[z] \approx 1$ and $z$ is
    approximately Gaussian. Now a standard L2 loss is statistically sound.

    At inference we undo with the **asymptotic inverse**:

    $$\hat{\mu} = \frac{g\,z^{2}}{4} - \frac{3g}{8} - \frac{\sigma_r^2}{g}$$

    (A more accurate **Mäkitalo-Foi closed-form inverse** is used by
    `cidc.noise.inverse_anscombe` at inference; it adds correction terms of
    order $1/z$ that matter at low photon counts.)

    Let's demonstrate.
    """)
    return


@app.cell
def _anscombe_demo(NOISE_LEVELS, anscombe, np, plt, sample_poisson_gaussian):
    """Show that Anscombe flattens the variance curve."""
    _mu_grid = np.linspace(0, 4000, 40)
    _mu_img = np.broadcast_to(_mu_grid[:, None, None], (len(_mu_grid), 200, 200)).astype(np.float32)
    _rng = np.random.default_rng(1)
    _p = NOISE_LEVELS[2]
    _y = sample_poisson_gaussian(_mu_img, _p, rng=_rng)
    _z = anscombe(_y, _p)

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _axes[0].plot(_mu_grid, _y.reshape(len(_mu_grid), -1).var(axis=1), "o-")
    _axes[0].set_title("Raw ADU: variance grows with mean")
    _axes[0].set_xlabel("μ"); _axes[0].set_ylabel("Var(y)")

    _axes[1].plot(_mu_grid, _z.reshape(len(_mu_grid), -1).var(axis=1), "o-")
    _axes[1].axhline(1.0, color="k", ls="--", label="unit variance")
    _axes[1].set_title("After Anscombe: variance ≈ 1")
    _axes[1].set_xlabel("μ"); _axes[1].set_ylabel("Var(z)")
    _axes[1].set_ylim(0, 2); _axes[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _anscombe_real(DATA, NOISE_LEVELS, anscombe, load_stack, np, plt):
    """Apply Anscombe to the real F1 stack and check the flat variance line."""
    _s = np.asarray(load_stack(DATA / "val" / "F1.tif")[:256], dtype=np.float32)
    _z = anscombe(_s, NOISE_LEVELS[1])
    # Bin variance over the mean — same as the raw plot above.
    _m, _v = _s.mean(0), _z.var(0)
    _bins = np.linspace(_s.mean(0).min(), _s.mean(0).max(), 50)
    _idx = np.digitize(_m.ravel(), _bins)
    _mx = np.array([_m.ravel()[_idx == i].mean() for i in range(1, len(_bins)) if (_idx == i).any()])
    _vx = np.array([_v.ravel()[_idx == i].mean() for i in range(1, len(_bins)) if (_idx == i).any()])

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(_mx, _vx, "o", ms=4)
    _ax.axhline(1.0, color="k", ls="--", label="target = 1.0")
    _ax.set_xlabel("per-pixel mean (ADU)")
    _ax.set_ylabel("variance of Anscombe-z across time")
    _ax.set_title("F1 after Anscombe: variance flattens near 1.0")
    _ax.set_ylim(0, 2); _ax.legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _intro5(mo):
    mo.md(r"""
    # 5. Loss functions — what to use, and why

    The loss is the *definition* of "denoised". Wrong loss → wrong
    solution, even with a perfect architecture.

    We'll compare four relevant options on the **same** toy problem:
    1. **L2 (MSE)** — naive; treats all errors equally.
    2. **Anscombe MSE** — L2 in variance-stabilised space.
    3. **Poisson-Gaussian NLL** — the correct likelihood for our camera.
    4. **Self-supervised (N2V-like)** — no clean target needed.

    ## 5a. L2 / MSE

    $$\mathcal{L}_{\text{MSE}}(\hat\mu, y) = \frac{1}{N}\sum (\hat\mu - y)^2$$

    Mathematically the Gaussian MLE with constant variance. Wrong for us
    because Var depends on μ. Still *used* all the time because it's
    simple; often the architecture compensates.

    ## 5b. Anscombe MSE

    $$\mathcal{L}_{\text{AMSE}} = \frac{1}{N}\sum (A(\hat\mu) - A(y))^2$$

    where $A$ is the Anscombe transform. Now L2 is *correct* because we
    compare in a space where the noise is (approximately) homoscedastic
    Gaussian. Used by default in our `cidc.losses.anscombe_mse`.

    ## 5c. Poisson-Gaussian NLL

    Direct negative log-likelihood of the measurement model:

    $$\mathcal{L}_{\text{PG}}(\hat\mu, y) = \frac{1}{N}\sum \left[
        \frac{(y - \hat\mu)^2}{2\bigl(g\,\hat\mu + \sigma_r^2\bigr)}
        + \tfrac{1}{2}\log\!\bigl(g\,\hat\mu + \sigma_r^2\bigr)
    \right]$$

    The first term is **weighted MSE**: residuals in bright pixels
    (large variance) are penalised less. The second term is the normalising
    constant; it prevents the model from cheating by predicting
    $\hat\mu \to \infty$ to drive the weight down.

    Strictly better than Anscombe-MSE in theory; in practice,
    indistinguishable for typical GCaMP stacks because Anscombe is a very
    good approximation.

    ## 5d. Self-supervised (N2V, N2N, DeepInterp)

    The above all need a **clean** target $\mu$. For CIDC25 we have F0
    available only for val, not train. So the training losses sidestep
    this:

    - **N2V**: replace masked voxel with neighbour, train to predict
      original → model can't copy identity since the input doesn't
      *contain* the target value.
    - **N2N**: train on two **independent noisy** observations of the same
      scene. Expected residual is 0 because noise realisations are
      independent. Mathematically equivalent to training on clean data
      (Lehtinen 2018).
    - **DeepInterp**: predict held-out center frame from temporal
      neighbours. Signal is slow, noise is iid → only signal survives.

    All three still apply an **underlying loss** (usually
    Anscombe-MSE or PG-NLL) to the reconstructed voxels.
    """)
    return


@app.cell
def _intro6(mo):
    mo.md("""
    # 6. Hands-on: compare the losses on a toy problem

    We generate a known clean 2-D image (signal = sum of Gaussian blobs),
    corrupt it with Poisson-Gaussian noise at level 2, then fit a
    *single-parameter* "denoiser" (bias shift) under each loss, and see
    which one recovers the true signal best.

    Single-pixel optimum is informative: if a loss favours the wrong
    steady-state, architecture won't save you.
    """)
    return


@app.cell
def _toy_signal(np, plt):
    """Make a clean toy scene (bright blobs + background).

    We use bright values (300-1300 ADU) and level-1 noise so the closed-form
    optima land near b=0 as expected; at level-2 noise on dim signal, the
    single-realisation Jensen bias is large enough to shift the optimum.
    """
    _yy, _xx = np.mgrid[:128, :128].astype(np.float32)
    _blobs = (
        500 * np.exp(-((_xx - 40) ** 2 + (_yy - 40) ** 2) / (2 * 10 ** 2))
        + 800 * np.exp(-((_xx - 90) ** 2 + (_yy - 60) ** 2) / (2 * 14 ** 2))
        + 300
    )
    clean_signal = _blobs.astype(np.float32)
    _fig, _ax = plt.subplots(figsize=(4, 4))
    _ax.imshow(clean_signal, cmap="magma"); _ax.set_title("Clean toy signal μ"); _ax.axis("off")
    _fig
    return (clean_signal,)


@app.cell
def _toy_noisy(NOISE_LEVELS, clean_signal, np, plt, sample_poisson_gaussian):
    """Corrupt with level-1 PG noise."""
    _p = NOISE_LEVELS[1]
    _rng = np.random.default_rng(42)
    noisy = sample_poisson_gaussian(clean_signal, _p, rng=_rng).astype(np.float32)
    toy_params = _p
    _fig, _ax = plt.subplots(1, 2, figsize=(8, 4))
    _ax[0].imshow(clean_signal, cmap="magma", vmin=0, vmax=1300); _ax[0].set_title("clean"); _ax[0].axis("off")
    _ax[1].imshow(noisy, cmap="magma", vmin=0, vmax=1300); _ax[1].set_title(f"noisy (g={_p.gain:.1f}, σr²={_p.read_var:.0f})"); _ax[1].axis("off")
    _fig
    return noisy, toy_params


@app.cell
def _closed_form_optima(anscombe, clean_signal, noisy, np, toy_params):
    """Closed-form: what does each loss converge to if the 'model' is μ + b?"""
    # L2 (MSE): minimiser is mean of noisy data (global scalar).
    mse_optimum = float(noisy.mean())

    # Anscombe MSE: minimiser of ||A(μ+b) - A(y)||² over b. We do it numerically.
    _y_ansc = anscombe(noisy, toy_params)

    def _amse_loss(_b):
        _mu = clean_signal + _b
        return float(((anscombe(_mu, toy_params) - _y_ansc) ** 2).mean())

    _grid = np.linspace(-40, 40, 201)
    _amse_curve = np.array([_amse_loss(b) for b in _grid])
    amse_optimum = float(_grid[_amse_curve.argmin()])

    # PG-NLL: same thing. Matches cidc.losses.poisson_gaussian_nll:
    #    nll = 0.5 * log(V) + 0.5 * (y-mu)^2 / V
    def _pgnll_loss(_b):
        _mu = np.maximum(clean_signal + _b, 1e-6)
        _var = toy_params.gain * _mu + toy_params.read_var
        return float((0.5 * (noisy - _mu) ** 2 / _var + 0.5 * np.log(_var)).mean())

    _pg_curve = np.array([_pgnll_loss(b) for b in _grid])
    pg_optimum = float(_grid[_pg_curve.argmin()])

    bias_grid = _grid
    amse_curve = _amse_curve
    pg_curve = _pg_curve
    return (
        amse_curve,
        amse_optimum,
        bias_grid,
        mse_optimum,
        pg_curve,
        pg_optimum,
    )


@app.cell
def _plot_optima(
    amse_curve,
    amse_optimum,
    bias_grid,
    clean_signal,
    mse_optimum,
    noisy,
    pg_curve,
    pg_optimum,
    plt,
):
    """Plot loss surfaces with their minima."""
    _fig, _ax = plt.subplots(1, 2, figsize=(11, 4))
    _ax[0].plot(bias_grid, amse_curve, label=f"Anscombe MSE  (min at b={amse_optimum:+.2f})")
    _ax[0].plot(bias_grid, pg_curve / pg_curve.max() * amse_curve.max(), label=f"PG-NLL (scaled; min b={pg_optimum:+.2f})")
    _ax[0].axvline(0, color="k", ls="--", alpha=0.5, label="true bias = 0")
    _ax[0].set_xlabel("bias b added to clean signal")
    _ax[0].set_ylabel("loss")
    _ax[0].set_title("Loss landscapes vs scalar bias")
    _ax[0].legend()

    # Compare recovered μ vs clean at each optimum.
    _ax[1].plot(clean_signal.ravel()[::100], noisy.ravel()[::100], ".", ms=2, alpha=0.3, label="noisy y")
    _ax[1].plot([0, 800], [0, 800], "k--", alpha=0.5, label="y = x")
    _ax[1].axhline(mse_optimum, color="tab:red", ls=":",
                   label=f"MSE constant fit = {mse_optimum:.0f} (wrong: collapses signal)")
    _ax[1].set_xlabel("true μ"); _ax[1].set_ylabel("noisy y / constant estimate")
    _ax[1].set_title("MSE with a constant-model collapses to the global mean")
    _ax[1].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _optima_verdict(amse_optimum, mo, mse_optimum, pg_optimum):
    mo.md(f"""
    **Result** (single-scalar toy):

    | Loss | Recovered bias | True bias |
    |---|---:|---:|
    | MSE (constant model) | collapses to mean = **{mse_optimum:.1f}** | signal is *not* constant |
    | Anscombe MSE | **{amse_optimum:+.2f}** | 0 |
    | PG-NLL | **{pg_optimum:+.2f}** | 0 |

    MSE with a *constant* model obviously collapses (it's the global mean).
    With a full CNN the MSE model can track structure — but the per-pixel
    *weighting* is still wrong: bright-pixel errors dominate the gradient.
    Anscombe-MSE and PG-NLL both identify bias ≈ 0 correctly, because in
    both cases residuals are properly weighted by local noise.

    **Practical guidance:**
    - For CIDC25, use **Anscombe-MSE** or **PG-NLL** by default.
    - MSE in raw ADU is a red flag. It "works" (won't diverge) but
      systematically overweights bright neurons at the cost of dim
      background.
    """)
    return


@app.cell
def _intro7(mo):
    mo.md(r"""
    # 7. Write and test your own loss

    Writing a loss = writing a scalar torch function of
    `(prediction, target, …)`. Rules:

    1. Must be **differentiable** in `prediction` (torch autograd can check).
    2. Must be **scale-sensible** — gradient magnitudes shouldn't blow up
       for typical inputs, otherwise optimisation will stall.
    3. Should have a known **correct solution** so you can unit-test it.

    The smallest possible test harness:
    """)
    return


@app.cell
def _custom_loss_demo(torch):
    """Template: define a loss, check gradient, verify optimum.

    Example: a custom 'robust PG-NLL' using Huber instead of squared residual.
    Useful if your data has occasional outlier voxels (dead pixels, cosmic rays).
    """
    def my_robust_pg_loss(pred: torch.Tensor, target: torch.Tensor,
                          gain: float, read_var: float,
                          delta: float = 5.0) -> torch.Tensor:
        pred = torch.clamp(pred, min=1e-6)
        var = gain * pred + read_var
        std = torch.sqrt(var)
        resid = (target - pred) / std            # standardised residual
        huber = torch.where(
            resid.abs() < delta,
            0.5 * resid ** 2,
            delta * (resid.abs() - 0.5 * delta),
        )
        return (huber + 0.5 * torch.log(var)).mean()

    # --- Test 1: loss is differentiable -------------------------------------
    _pred = torch.rand(64, 64, requires_grad=True) * 500 + 100
    _target = _pred.detach() + torch.randn_like(_pred) * 20
    _loss = my_robust_pg_loss(_pred, _target, gain=50.0, read_var=1000.0)
    _loss.backward()
    diff_ok = _pred.grad is not None and torch.isfinite(_pred.grad).all().item()

    # --- Test 2: optimum is at pred ≈ target - gain/2 (NOT pred = target). --
    # Every PG-style loss with a log(var) term has this bias; see losses.py
    # docstring. We test the theoretical optimum, not p == y.
    _g, _r = 50.0, 1000.0
    _y = torch.full((4, 4), 300.0)
    _target_optimum = 300.0 - _g / 2                         # ≈ 275
    _p = torch.full((4, 4), 200.0, requires_grad=True)
    _opt = torch.optim.Adam([_p], lr=1.0)
    for _ in range(2000):
        _opt.zero_grad()
        _l = my_robust_pg_loss(_p, _y, gain=_g, read_var=_r)
        _l.backward(); _opt.step()
    fit_err = float((_p.detach() - _target_optimum).abs().mean())
    return diff_ok, fit_err


@app.cell
def _show_test_results(diff_ok, fit_err, mo):
    mo.md(f"""
    **Test results for `my_robust_pg_loss`:**

    - Gradient finite & non-None: **{diff_ok}**
    - After 2000 Adam steps, |p − (y − g/2)|·mean = **{fit_err:.3f}**
      (expect small — PG-style losses have an intrinsic bias of −g/2 from
      the log-variance term; see the docstring in
      `@/app/workspace/src/cidc/losses.py`).

    **The p ≠ y optimum is a feature, not a bug:** the log(Var(μ))/2 term
    is the normalising constant of the Gaussian likelihood; it pulls μ
    down very slightly to reduce the width of the predicted noise
    distribution. The shift is O(g) in absolute ADU and negligible next to
    per-pixel noise (σ ≈ √(g·μ) ≈ 100+ ADU), so it doesn't hurt
    denoising in practice. But it will catch you out if you unit-test a
    PG loss by asserting `p == y`.

    If you want a custom loss that doesn't have this bias, drop the
    `0.5 * log(var)` term — but then your gradient magnitudes become
    inverted (bright pixels contribute more), and you re-introduce the
    MSE failure mode this tutorial warned about.
    """)
    return


@app.cell
def _cidc_losses_test(anscombe, anscombe_mse, mo, poisson_gaussian_nll, torch):
    """Sanity checks for the built-in losses.

    IMPORTANT: ``cidc.losses.anscombe_mse`` expects inputs that are *already*
    in Anscombe space — it's just a plain ``mean((z_pred - z_target)**2)``.
    The Anscombe transform itself is applied upstream (in the dataset, or
    before calling the loss). Common mistake: passing raw ADU.
    """
    from cidc.noise import NOISE_LEVELS as _NL
    _p = _NL[1]

    # Correct usage: transform to Anscombe space first.
    _raw_pred = torch.rand(32, 32, requires_grad=True) * 500 + 500
    _raw_tgt = (_raw_pred.detach() + torch.randn_like(_raw_pred) * 10).float()
    _z_pred = torch.as_tensor(anscombe(_raw_pred.detach().numpy(), _p))
    _z_pred.requires_grad_(True)
    _z_tgt = torch.as_tensor(anscombe(_raw_tgt.numpy(), _p))
    _l1 = anscombe_mse(_z_pred, _z_tgt)
    _l1.backward()
    amse_scalar_loss = float(_l1.item())
    amse_grad_finite = bool(torch.isfinite(_z_pred.grad).all().item())

    # PG-NLL takes raw ADU directly.
    _pred2 = (torch.rand(32, 32) * 300 + 500).requires_grad_(True)
    _tgt2 = _pred2.detach() + torch.randn_like(_pred2) * 30
    _l2 = poisson_gaussian_nll(_pred2, _tgt2, gain=_p.gain, read_var=_p.read_var)
    _l2.backward()
    pg_scalar_loss = float(_l2.item())
    pg_grad_finite = bool(torch.isfinite(_pred2.grad).all().item())

    mo.md(
        f"""
        **Built-in loss sanity checks (level-1 params):**

        | loss | scalar value | grad finite? |
        |---|---:|---:|
        | `anscombe_mse(z_pred, z_tgt)` | {amse_scalar_loss:.4f} | {amse_grad_finite} |
        | `poisson_gaussian_nll(pred, tgt)` | {pg_scalar_loss:.4f} | {pg_grad_finite} |

        Both return a 0-dim tensor by default (mean reduction) and both have
        finite gradients. The scalar-ness is what makes `.backward()` work
        without passing `grad_tensors=`.
        """
    )
    return


@app.cell
def _wrap(mo):
    mo.md("""
    # Wrap-up

    You now know:

    1. **What the noise is physically**: Poisson (shot) + Gaussian (read).
    2. **How to measure it from data**: fit Var = g·μ + σ_r².
    3. **Why raw L2 is suboptimal**: variance isn't constant.
    4. **How Anscombe fixes it**: warps data to ≈unit variance.
    5. **Four loss families** and when to use each.
    6. **How to write and test a custom loss** in a reproducible way.

    ### Next steps
    - Open `@/app/workspace/src/cidc/losses.py` and read the implementations.
      They're ~30 lines each.
    - Drop your custom loss into `@/app/workspace/src/cidc/train.py` →
      `step_n2v3d` and run `uv run cidc train configs/quick_6gb.yaml`.
    - If you invent something worth keeping, add it to `losses.py` with a
      docstring and a sanity test like the ones above.
    """)
    return


if __name__ == "__main__":
    app.run()
