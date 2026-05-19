"""
Run every numbered notebook (01–10) and save all stdout to
workspace/notebook_plots/<nb_name>/output.txt.

Each cell's output is written with a header so you can find which cell produced
what. Errors are included inline so the file is complete even if a cell fails.

Usage:
    python save_notebook_text.py              # all notebooks
    python save_notebook_text.py 01 03 07     # specific ones
"""

import io
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

LOCAL_MODS = {"plots", "metrics", "analysis", "load_data", "sampling",
              "calibration", "smart_sampler"}


def run_notebook(ipynb_path: Path, nb_dir: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ipynb_path) as f:
        nb = json.load(f)

    orig_cwd = os.getcwd()
    os.chdir(nb_dir)

    orig_path = sys.path.copy()
    if str(nb_dir) not in sys.path:
        sys.path.insert(0, str(nb_dir))

    for m in list(sys.modules):
        if m in LOCAL_MODS:
            del sys.modules[m]

    # Silence plt.show() — we only care about text here
    plt.show = lambda *a, **k: plt.close("all")

    ns: dict = {"__name__": "__main__", "__file__": str(ipynb_path)}
    lines: list[str] = []
    cells_with_output = 0

    for i, cell in enumerate(nb["cells"]):
        cell_type = cell["cell_type"]
        src = "".join(cell["source"]).strip()
        if not src:
            continue

        if cell_type == "markdown":
            lines.append(f"{'='*72}")
            lines.append(f"Cell {i} [markdown]")
            lines.append(f"{'='*72}")
            lines.append(src)
            lines.append("")
            cells_with_output += 1

        elif cell_type == "code":
            clean = "\n".join(
                l for l in src.splitlines()
                if not l.startswith("%") and not l.startswith("!")
            ).strip()
            if not clean:
                continue

            buf = io.StringIO()
            orig_stdout = sys.stdout
            sys.stdout = buf
            try:
                exec(compile(clean, f"<cell {i}>", "exec"), ns)  # noqa: S102
                sys.stdout = orig_stdout
                captured = buf.getvalue()
            except Exception as e:
                sys.stdout = orig_stdout
                captured = buf.getvalue()
                captured += f"[ERROR] {type(e).__name__}: {e}\n"

            if captured.strip():
                lines.append(f"{'='*72}")
                lines.append(f"Cell {i} [code output]")
                lines.append(f"{'='*72}")
                lines.append(captured.rstrip())
                lines.append("")
                cells_with_output += 1

    out_file = out_dir / "output.txt"
    out_file.write_text("\n".join(lines) + "\n")

    sys.path = orig_path
    os.chdir(orig_cwd)
    return cells_with_output


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

    for name, ipynb, nb_dir in notebooks:
        out_dir = OUT_ROOT / name
        print(f"[{name}]  {ipynb.name}", end=" ... ", flush=True)
        n = run_notebook(ipynb, nb_dir, out_dir)
        print(f"{n} cell(s) with output -> {(out_dir / 'output.txt').relative_to(WORKSPACE)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
