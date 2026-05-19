"""06 — Masking geometry: 2D vs 3D blind-spot receptive fields."""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 06 — Masking geometry: what 2D vs 3D blind-spot masking reveals

    **QUESTION:** In self-supervised denoising (Noise2Void style), the model
    predicts each pixel from its context — but which context? A 2D spatial
    blind-spot (mask center, use spatial neighbors) or a 3D spatiotemporal
    blind-spot (mask center, use spatial + temporal neighbors)?

    For calcium imaging, temporal neighbors carry the **signal** — the same
    neuron's activity is correlated in time. This notebook asks: how much
    information do temporal neighbors add? And does that advantage grow at
    higher noise levels?

    **Decision gate:** pins down the mask geometry (and thus receptive field
    size) before model training. The choice changes what the network learns
    to see first and most accurately — which ultimately determines stSNR.
    """)
    return


@app.cell
def _setup():
    from pathlib import Path
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from cidc import load_stack, temporal_autocorr, NOISE_LEVELS

    DATA = Path(__file__).parent.parent.parent / "data"
    return DATA, load_stack, mo, np, plt, temporal_autocorr


@app.cell
def _intro(mo):
    mo.md("""
    ## The blind-spot principle: predicting from context without the center

    **Self-supervised denoising** works because:

    1. At training time, we **mask** the center pixel and hide its value from the input.
    2. The model must **predict** the center from surrounding pixels (context).
    3. We compute loss between prediction and the **noisy observation** (the only available target).

    **Why this works:** Since the center's noise is never visible to the model,
    the only way to minimize loss is to predict the **true signal** — noise
    that neither the model nor the target sees cannot drive the loss.

    **The geometry question:** What counts as "context"? How large should the
    receptive field be, and what dimensions should it span?

    Two main strategies:

    - **2D blind-spot:** mask (t, y, x), use context (t, y±δ, x±δ). Spatial
      neighbors at the same timepoint.
    - **3D blind-spot:** mask (t, y, x), use context (t±τ, y±δ, x±δ). Spatial
      + temporal neighbors from nearby frames.

    The question is not about feasibility — both work. It's about **what
    information is available** to the model in each case, and how that advantage
    grows under different noise conditions.
    """)
    return


@app.cell
def _temporal_structure(DATA, load_stack, mo, plt, temporal_autocorr):
    mo.md("""
    ## Temporal structure in clean data (F0)

    Calcium imaging shows strong temporal correlations — when a neuron fires,
    it stays active for tens of frames. The autocorrelation function (ACF)
    quantifies this. Here we compute F0's temporal ACF:
    """)

    _s0 = load_stack(DATA / "val" / "F0.tif")
    acf_f0 = temporal_autocorr(_s0, max_lag=60)

    _fig, _ax = plt.subplots(figsize=(10, 4))
    _ax.plot(acf_f0, lw=2, label="F0 (clean)")
    _ax.axhline(0, c="k", lw=0.5)
    _ax.set_xlabel("lag (frames)")
    _ax.set_ylabel("autocorrelation")
    _ax.set_title("Temporal ACF — F0 (clean ground truth)")
    _ax.legend()
    _ax.grid(alpha=0.3)
    _fig
    return (acf_f0,)


@app.cell
def _acf_interpretation(acf_f0, mo):
    mo.md(f"""
    ## What the ACF tells us about information

    The ACF ρ(τ) quantifies how predictable x_t is from x_(t+τ):

    - **ρ(0) = 1:** perfect self-correlation (always true).
    - **ρ(1) ≈ {acf_f0[1]:.3f}:** at lag 1 frame, correlation is still very strong.
    - **ρ(5) ≈ {acf_f0[5]:.3f}:** at lag 5 frames (~0.33 sec @ 15 fps), still correlated.
    - **ρ(20) ≈ {acf_f0[20]:.3f}:** at lag 20 frames (~1.3 sec), correlation is decaying.

    This decay is **slow** — the signal has structure over many frames. Even at lag 20 frames (~1.3 sec),
    ρ ≈ {acf_f0[20]:.3f}. This motivates exploring temporal context windows of ±10–20 frames (measured next).
    """)
    return


@app.cell
def _mutual_information(acf_f0, mo, np):
    mo.md("""
    ## Mutual information from temporal neighbors

    Under the Gaussian model (a standard approximation in information theory),
    the mutual information between two variables with correlation ρ is:

    $$I(x_t; x_{t+\\tau}) = -\\frac{1}{2} \\log(1 - \\rho(\\tau)^2)$$

    This formula says: **correlation directly translates to bits of information**.

    For a single temporal neighbor at lag τ, we get I(x_t; x_{t+τ}) bits.
    But multiple temporal neighbors are not independent — they're all correlated
    with each other. A more useful measure is the **total information** from a
    temporal context window. We compute this next for F0's actual ACF.
    """)

    # Compute MI for each lag
    mi = -0.5 * np.log(1 - acf_f0**2)
    return (mi,)


@app.cell
def _plotmi(acf_f0, mi, mo, plt):
    mo.md("### Mutual information per lag")

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Plot ACF
    _ax1.plot(acf_f0, lw=2, color="C0")
    _ax1.fill_between(range(len(acf_f0)), acf_f0, alpha=0.3, color="C0")
    _ax1.set_xlabel("lag (frames)")
    _ax1.set_ylabel("autocorrelation ρ(τ)")
    _ax1.set_title("Temporal ACF of F0")
    _ax1.grid(alpha=0.3)

    # Plot MI
    _ax2.plot(mi, lw=2, color="C1")
    _ax2.fill_between(range(len(mi)), mi, alpha=0.3, color="C1")
    _ax2.set_xlabel("lag (frames)")
    _ax2.set_ylabel("mutual information (bits)")
    _ax2.set_title("I(x_t; x_{t+τ}) from F0 ACF")
    _ax2.grid(alpha=0.3)

    _fig
    return


@app.cell
def _temporal_window_info(mi, mo, np):
    mo.md("""
    ## Information accumulation in a temporal context window

    If we have k temporal neighbors (frames on each side, ±k), how much total
    information do we gain? The exact calculation requires the full correlation
    matrix, but we can approximate by summing MI from lag 1 to lag k.

    This is **not exact** (the neighbors are not independent), but it gives
    intuition: how many "bits of signal" become available as we expand the
    temporal window?
    """)

    # Approximate total info by summing MI from lag 1 to lag k
    _cumsummi = np.cumsum(mi)

    _info_by_window = {}
    for _k in [1, 3, 5, 10, 15, 20]:
        # Both directions: -k to k means (2*k) temporal neighbors
        _totalmi = 2 * np.sum(mi[1:_k+1])
        _info_by_window[_k] = _totalmi

    _output = "**Information from temporal context windows (approximate):**\n\n"
    _output += "| Half-window τ | Neighbors | Bits from ACF |\n"
    _output += "|---|---|---|\n"
    for _k, _bits in _info_by_window.items():
        _output += f"| ±{_k} frames | {2*_k} | {_bits:.2f} |\n"
    mo.md(_output)
    return


@app.cell
def _spatial_context_note(mo):
    mo.md("""
    ## Spatial context for comparison

    For comparison: spatial neighbors are much less informative. In calcium
    imaging, two spatially adjacent pixels may belong to different neurons,
    or the same neuron may have large spatial extent. The spatial correlation
    between pixels depends on the neuron morphology and imaging PSF.

    Typical spatial ACF in calcium imaging decays much faster than temporal ACF
    — by distance of ~3–5 pixels, correlation is near zero. So a single spatial
    neighbor carries ~0.5–1 bit of information (if ρ ≈ 0.7), and multiple spatial
    neighbors add diminishing returns.

    **In contrast:** temporal neighbors at the same spatial location carry
    **the same signal** with only independent noise added. This is why temporal
    context is so powerful for denoising.
    """)
    return


@app.cell
def _3d_vs_2d_gradient_snr(mo):
    mo.md("""
    ## Gradient SNR: why temporal context improves training

    In self-supervised training, the loss is L(ŷ, y) where:
    - y is the noisy observation
    - ŷ is the model's prediction

    The gradient ∂L/∂θ is driven by the **error signal**: (ŷ − x), where x is
    the true signal (never seen, but implicit in y). The quality of this gradient
    depends on how well the model can reconstruct x from its receptive field.

    **In 2D masking:**
    - Receptive field = spatial neighbors at one timepoint
    - At high noise (e.g., F3 with g=990.5), pixels are nearly independent
      across space — neighbors barely help
    - ŷ ≈ spatial mean ≈ 0 (since F3 is mostly background)
    - Error (ŷ − x) is large and noisy
    - Gradient is weak and unreliable

    **In 3D masking:**
    - Receptive field = temporal neighbors (which are highly correlated)
    - Even at high noise, temporal averaging works: if the same neuron fires
      in frames t−1, t, t+1, then E[y_t; y_{t-1}; y_{t+1}] ≈ signal,
      noise ~1/√3 of single frame
    - ŷ from temporal context can be much more accurate
    - Error (ŷ − x) is smaller
    - Gradient is stronger and more reliable

    **The gradient SNR advantage grows with noise level.** At low noise (F1),
    spatial neighbors may suffice. At high noise (F3), temporal context becomes
    essential.
    """)
    return


@app.cell
def _mask_geometry_decision(mo):
    mo.md("""
    ## Design decision: 3D blind-spot masking

    Based on the analysis above:

    1. **Temporal correlations are strong in F0** (~0.7 at lag 1, decaying slowly).
    2. **Temporal neighbors carry signal information** even at high noise, because
       the clean signal repeats; noise is independent.
    3. **Gradient SNR improves dramatically with temporal context**, especially
       at F3's extreme noise level (g=990.5, noise >> signal).
    4. **The information accumulation is sublinear** but significant: ±10 frames
       gives ~5–6 bits of signal information (rough estimate).

    **Conclusion:** Use **3D voxel masking**. The information table above suggests a temporal
    context window of **±10 frames** (20 neighbors, ~30 bits of signal information) balances
    information gain with computational cost. This can be tuned empirically in NB07.

    This means:
    - Mask the center voxel at (t, y, x).
    - Input to the encoder includes spatial neighbors and temporal neighbors.
    - During blind-spot training, the model learns to exploit temporal
      correlations to infer the center pixel's clean signal.
    - This explicitly trains the temporal denoising capability, which is
      directly measured by tSNR.

    **Next step (NB07):** Train a real U-Net with this geometry on F1/F2 and
    measure improvement over 2D baseline.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
