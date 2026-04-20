"""Download AI4Life-CIDC25 training and validation data from Zenodo.

Usage:
    python scripts/download_data.py              # download all
    python scripts/download_data.py --split val  # only validation
    python scripts/download_data.py --split train
    python scripts/download_data.py --dry-run    # just print plan

Files are verified against the MD5 checksums published on Zenodo and
skipped if already present with a matching hash.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class ZFile:
    split: str           # "train" | "val"
    name: str            # file name on disk
    url: str             # direct content URL
    md5: str
    size: int            # bytes, for progress


# Source: https://zenodo.org/api/records/15799507 and /15807610
TRAIN_BASE = "https://zenodo.org/api/records/15799507/files"
VAL_BASE = "https://zenodo.org/api/records/15807610/files"

FILES: list[ZFile] = [
    # Training: 4 samples x 2 noise levels (suffix 1 or 2)
    ZFile("train", "A1.tif", f"{TRAIN_BASE}/A1.tif/content",
          "b32fdae18dac23533aa146137a6410d9", 720567094),
    ZFile("train", "B1.tif", f"{TRAIN_BASE}/B1.tif/content",
          "4779dfbd48b01572d6abfcd7c0cd0f8a", 720567094),
    ZFile("train", "C2.tif", f"{TRAIN_BASE}/C2.tif/content",
          "0fd704b52a9c22f4bc93087ee728cc58", 720567094),
    ZFile("train", "D2.tif", f"{TRAIN_BASE}/D2.tif/content",
          "b2371ab0d6e4f7026af70be8d41edb72", 720567094),
    # Validation: F0 clean + F1..F3 noisy levels
    ZFile("val", "F0.tif", f"{VAL_BASE}/F0.tif/content",
          "729db2266f02a35b55d0450906a01900", 720567094),
    ZFile("val", "F1.tif", f"{VAL_BASE}/F1.tif/content",
          "5a3ad29621abac46962953516b8c659d", 720567094),
    ZFile("val", "F2.tif", f"{VAL_BASE}/F2.tif/content",
          "db1b19c43982b67c9dec4e566b0d704d", 720567094),
    ZFile("val", "F3.tif", f"{VAL_BASE}/F3.tif/content",
          "032ee3732f86a416026a0cd8630c75c3", 720567094),
]


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def download(f: ZFile, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  -> {f.url}")

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        done = block_num * block_size
        pct = min(100.0, 100.0 * done / max(total_size, 1))
        sys.stdout.write(
            f"\r     {human(min(done, total_size))} / {human(total_size)} "
            f"({pct:5.1f}%)"
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(f.url, tmp, reporthook=hook)
    sys.stdout.write("\n")
    tmp.rename(dest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Don't re-hash existing files (trust size only).",
    )
    args = parser.parse_args()

    selected = [f for f in FILES if args.split in ("all", f.split)]
    total = sum(f.size for f in selected)
    print(f"Planned download: {len(selected)} files, ~{human(total)}")
    for f in selected:
        print(f"  [{f.split}] {f.name}  ({human(f.size)})")
    if args.dry_run:
        return 0

    for f in selected:
        dest = args.data_dir / f.split / f.name
        if dest.exists():
            if args.skip_md5 and dest.stat().st_size == f.size:
                print(f"[skip] {dest} already present")
                continue
            print(f"[check] {dest} — verifying md5...")
            if md5_of(dest) == f.md5:
                print(f"[skip]  {dest} verified")
                continue
            print(f"[warn] md5 mismatch, redownloading {dest}")
            dest.unlink()
        print(f"[get ] {f.split}/{f.name}")
        download(f, dest)
        got = md5_of(dest)
        if got != f.md5:
            print(f"[ERR ] md5 mismatch for {dest}: got {got}, expected {f.md5}")
            return 1
        print(f"[ok  ] {dest}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
