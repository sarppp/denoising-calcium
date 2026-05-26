"""
Export every numbered notebook (01–10) to PDF via nbconvert → HTML → Playwright.

Output: workspace/all_pdfs/<nb_name>.pdf

Usage:
    python save_notebook_pdfs.py              # all notebooks
    python save_notebook_pdfs.py 01 03 07     # specific ones
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

WORKSPACE = Path(__file__).parent
NOTEBOOKS_DIR = WORKSPACE / "notebooks"
OUT_DIR = WORKSPACE / "all_pdfs"


def notebook_to_html(ipynb_path: Path, tmp_dir: Path) -> Path:
    html_path = tmp_dir / (ipynb_path.stem + ".html")
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "html",
            "--no-input",           # hide code cells — output + markdown only
            "--output", str(html_path),
            str(ipynb_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return html_path


def html_to_pdf(html_path: Path, pdf_path: Path, playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto(html_path.as_uri(), wait_until="networkidle")
    page.pdf(
        path=str(pdf_path),
        format="A4",
        margin={"top": "20mm", "bottom": "20mm", "left": "15mm", "right": "15mm"},
        print_background=True,
    )
    browser.close()


def find_notebooks(filter_ids: list[str] | None) -> list[tuple[str, Path]]:
    results = []
    for nb_dir in sorted(NOTEBOOKS_DIR.glob("[0-9][0-9]_*")):
        nb_id = nb_dir.name[:2]
        if filter_ids and nb_id not in filter_ids:
            continue
        ipynb_files = list(nb_dir.glob("*.ipynb"))
        if not ipynb_files:
            print(f"  skip {nb_dir.name} — no .ipynb found")
            continue
        results.append((nb_dir.name, ipynb_files[0]))
    return results


def main():
    filter_ids = sys.argv[1:] if len(sys.argv) > 1 else None
    notebooks = find_notebooks(filter_ids)

    if not notebooks:
        print("No notebooks found.")
        return

    OUT_DIR.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for name, ipynb in notebooks:
                pdf_path = OUT_DIR / f"{name}.pdf"
                print(f"[{name}]  ", end="", flush=True)
                try:
                    html_path = notebook_to_html(ipynb, tmp_dir)
                    html_to_pdf(html_path, pdf_path, pw)
                    size_kb = pdf_path.stat().st_size // 1024
                    print(f"-> {pdf_path.relative_to(WORKSPACE)}  ({size_kb} KB)")
                except Exception as e:
                    print(f"ERROR: {e}")

    print(f"\nDone. PDFs in all_pdfs/")


if __name__ == "__main__":
    main()
