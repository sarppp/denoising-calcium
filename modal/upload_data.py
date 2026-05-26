"""
Upload TIF data to the cidc-data Modal volume.

Run once before training:
    modal run modal/upload_data.py

Volume layout (mirrors local data/):
    cidc-data/
      train/  A1.tif  B1.tif  C2.tif  D2.tif
      val/    F0.tif  F1.tif  F2.tif  F3.tif

Re-running is safe — already-uploaded files are skipped unless --force is passed.
"""

from pathlib import Path
import modal

_HERE = Path(__file__).parent
_ROOT = _HERE.parent              # workspace/

VOLUME_NAME = "cidc-data"

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("cidc25-upload")


@app.local_entrypoint()
def main(force: bool = False):
    """
    Upload train + val TIF files to the cidc-data volume.

        modal run modal/upload_data.py          # skip already-uploaded files
        modal run modal/upload_data.py --force  # re-upload everything
    """
    train_dir = _ROOT / "data" / "train"
    val_dir   = _ROOT / "data" / "val"

    files = [
        *[(f, f"train/{f.name}") for f in sorted(train_dir.glob("*.tif"))],
        *[(f, f"val/{f.name}")   for f in sorted(val_dir.glob("*.tif"))],
    ]

    if not files:
        print("❌  No TIF files found under data/train/ or data/val/")
        return

    total_gb = sum(f.stat().st_size for f, _ in files) / 1024 ** 3
    print(f"Found {len(files)} TIF files  ({total_gb:.1f} GB total):")
    for local, remote in files:
        size_mb = local.stat().st_size / 1024 ** 2
        print(f"  {local.name:<12}  {size_mb:>6.0f} MB  →  {remote}")
    print()

    # Skip files already in the volume (unless --force)
    if not force:
        try:
            existing = {e.path for e in vol.listdir("/", recursive=True)}
            pending = [(f, r) for f, r in files if r not in existing]
            if not pending:
                print("✅  All files already in volume.  Use --force to re-upload.")
                return
            if len(pending) < len(files):
                skipping = len(files) - len(pending)
                print(f"⏭   Skipping {skipping} already-uploaded file(s).  Uploading {len(pending)} new file(s)...\n")
            files = pending
        except Exception:
            pass  # volume empty or other issue — just upload everything

    print("Uploading to cidc-data volume...")
    with vol.batch_upload(force=force) as batch:
        for local, remote in files:
            batch.put_file(str(local), remote)
            print(f"  ⬆   {remote}")

    print()
    print("✅  Upload complete!")
    print(f"   Volume : {VOLUME_NAME}")
    print(f"   Files  : {len(files)} TIF(s) uploaded")
    print()
    print("Now launch training:")
    print("  modal run modal/app.py")
