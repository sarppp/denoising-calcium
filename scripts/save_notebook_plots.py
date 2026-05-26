"""
Run every numbered notebook (01–10) and save all plots to workspace/notebook_plots/.

Each notebook's cells are executed in order inside an isolated namespace with:
  - that notebook's directory on sys.path (so `from plots import ...` works)
  - matplotlib set to non-interactive Agg backend
  - plt.show() patched to save figures to  notebook_plots/<nb_name>/plot_NNN.png

Usage:
    python save_notebook_plots.py              # all notebooks
    python save_notebook_plots.py 01 03 07     # specific ones
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = Path(__file__).parent
NOTEBOOKS_DIR = WORKSPACE / "notebooks"
OUT_ROOT = WORKSPACE / "notebook_plots"


def make_show_patcher(out_dir: Path, counter: list):
    """Return a plt.show replacement that saves all open figures."""
    def patched_show(*args, **kwargs):
        figs = [plt.figure(i) for i in plt.get_fignums()]
        if not figs:
            return
        for fig in figs:
            counter[0] += 1
            path = out_dir / f"plot_{counter[0]:03d}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"    saved {path.relative_to(WORKSPACE)}")
            plt.close(fig)
    return patched_show


def run_notebook(ipynb_path: Path, nb_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ipynb_path) as f:
        nb = json.load(f)

    counter = [0]

    # Run from the notebook's own directory so relative paths (e.g. Path("../../data/val"))
    # resolve correctly, matching how the notebook was designed to be executed.
    orig_cwd = os.getcwd()
    os.chdir(nb_dir)

    # Isolate sys.path to this notebook's dir
    orig_path = sys.path.copy()
    if str(nb_dir) not in sys.path:
        sys.path.insert(0, str(nb_dir))

    # Evict previously cached notebook-local modules so notebooks don't bleed
    local_names = {"plots", "metrics", "analysis", "load_data", "sampling",
                   "calibration", "smart_sampler"}
    for m in list(sys.modules):
        if m in local_names:
            del sys.modules[m]

    # Patch show globally; imported plots.py calls plt.show() directly
    plt.show = make_show_patcher(out_dir, counter)

    ns: dict = {"__name__": "__main__", "__file__": str(ipynb_path)}
    errors: list[str] = []

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"]).strip()
        if not src:
            continue
        # Drop IPython magic / shell lines
        lines = [l for l in src.splitlines()
                 if not l.startswith("%") and not l.startswith("!")]
        src = "\n".join(lines).strip()
        if not src:
            continue
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)  # noqa: S102
        except Exception as e:
            msg = f"cell {i}: {type(e).__name__}: {e}"
            errors.append(msg)
            print(f"    WARN {msg}")

    sys.path = orig_path
    os.chdir(orig_cwd)
    return counter[0], errors


def find_notebooks(filter_ids: list[str] | None) -> list[tuple[str, Path, Path]]:
    results = []
    for nb_dir in sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*")):
        nb_id = nb_dir.name[:2]
        if filter_ids and nb_id not in filter_ids:
            continue
        ipynb_files = list(nb_dir.glob("*.ipynb"))
        if not ipynb_files:
            print(f"  skip {nb_dir.name} — no .ipynb found")
            continue
        results.append((nb_dir.name, ipynb_files[0], nb_dir))
    return results


def main():
    filter_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    notebooks = find_notebooks(filter_ids)

    if not notebooks:
        print("No notebooks found.")
        return

    total_plots = 0
    for name, ipynb, nb_dir in notebooks:
        out_dir = OUT_ROOT / name
        print(f"\n[{name}]  {ipynb.name}")
        n, errors = run_notebook(ipynb, nb_dir, out_dir)
        total_plots += n
        status = f"{n} plot(s) saved"
        if errors:
            status += f", {len(errors)} cell error(s)"
        print(f"  -> {status}")

    print(f"\nDone. {total_plots} total plots in notebook_plots/")


if __name__ == "__main__":
    main()
