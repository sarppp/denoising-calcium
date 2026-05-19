"""
Collect all notebook plots and text outputs into flat directories:

  workspace/all_plots/  — all PNGs named  <nb_name>_plot_NNN.png
  workspace/all_texts/  — all output.txt  named  <nb_name>.txt

Run after save_notebook_plots.py and save_notebook_text.py have been executed.

Usage:
    python collect_outputs.py
"""

import shutil
from pathlib import Path

WORKSPACE = Path(__file__).parent
SRC = WORKSPACE / "notebook_plots"
ALL_PLOTS = WORKSPACE / "all_plots"
ALL_TEXTS = WORKSPACE / "all_texts"


def collect():
    ALL_PLOTS.mkdir(exist_ok=True)
    ALL_TEXTS.mkdir(exist_ok=True)

    plots_copied = 0
    texts_copied = 0

    for nb_dir in sorted(SRC.iterdir()):
        if not nb_dir.is_dir():
            continue
        nb_name = nb_dir.name

        for png in sorted(nb_dir.glob("plot_*.png")):
            dest = ALL_PLOTS / f"{nb_name}_{png.name}"
            shutil.copy2(png, dest)
            plots_copied += 1

        txt = nb_dir / "output.txt"
        if txt.exists():
            dest = ALL_TEXTS / f"{nb_name}.txt"
            shutil.copy2(txt, dest)
            texts_copied += 1

    print(f"all_plots/  — {plots_copied} file(s)")
    print(f"all_texts/  — {texts_copied} file(s)")


if __name__ == "__main__":
    collect()
