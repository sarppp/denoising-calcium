"""01 — Basic look. Load a stack, see a frame, see the temporal mean."""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _setup():
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from cidc import load_stack, stack_info

    DATA = Path("/app/workspace/data")
    return DATA, load_stack, np, plt, stack_info


@app.cell
def _info_table(DATA, stack_info):
    """Quick info on every stack."""
    for _p in sorted(DATA.glob("*/*.tif")):
        _i = stack_info(_p)
        print(f"{_p.parent.name}/{_p.name:8s}  shape={_i.shape}  "
              f"dtype={_i.dtype}  min={_i.min:.0f}  mean={_i.mean:.0f}  "
              f"max={_i.max:.0f}")
    return


@app.cell
def _one_frame(DATA, load_stack, np, plt):
    """One frame from F0 (clean). Change the path to look at other stacks."""
    _stack = load_stack(DATA / "val" / "F0.tif")
    _frame = np.asarray(_stack[750])  # middle-ish frame

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.imshow(_frame, cmap="gray", vmin=0, vmax=np.percentile(_frame, 99))
    _ax.set_title("F0.tif  frame 750")
    _ax.axis("off")
    _fig
    return


@app.cell
def _temporal_mean(DATA, load_stack, np, plt):
    """Mean of every 10th frame. Cheap way to see overall structure."""
    _stack = load_stack(DATA / "val" / "F0.tif")
    _tmean = np.asarray(_stack[::10]).mean(axis=0)

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.imshow(_tmean, cmap="gray", vmin=0, vmax=np.percentile(_tmean, 99))
    _ax.set_title("F0.tif  temporal mean (every 10th frame)")
    _ax.axis("off")
    _fig
    return


@app.cell
def _train_frame(DATA, load_stack, np, plt):
    """One frame from a training stack (A1, noisy level 1). Change the
    filename to look at B1 / C2 / D2."""
    _stack = load_stack(DATA / "train" / "A1.tif")
    _frame = np.asarray(_stack[750])

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.imshow(_frame, cmap="gray", vmin=0, vmax=np.percentile(_frame, 99))
    _ax.set_title("A1.tif  frame 750  (training, noisy)")
    _ax.axis("off")
    _fig
    return


@app.cell
def _train_temporal_mean(DATA, load_stack, np, plt):
    """Training temporal mean — averaging frames denoises naturally, so
    even the noisy training stacks show clear neurons here."""
    _stack = load_stack(DATA / "train" / "A1.tif")
    _tmean = np.asarray(_stack[::10]).mean(axis=0)

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.imshow(_tmean, cmap="gray", vmin=0, vmax=np.percentile(_tmean, 99))
    _ax.set_title("A1.tif  temporal mean  (training)")
    _ax.axis("off")
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
