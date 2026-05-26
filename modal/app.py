"""
CIDC25 — Modal training app
============================

# ── Upload data (one-time) ────────────────────────────────────────────────────
uv run --env-file modal/.env modal run modal/upload_data.py

# ── Full training (fresh start) ───────────────────────────────────────────────
uv run --env-file modal/.env modal run modal/app.py --no-resume

# ── Resume from last checkpoint ───────────────────────────────────────────────
uv run --env-file modal/.env modal run modal/app.py

# ── Quick 4-batch sanity check (~2 min) ───────────────────────────────────────
uv run --env-file modal/.env modal run modal/app.py --probe

# ── Override epoch count (e.g. test with 2 epochs) ────────────────────────────
uv run --env-file modal/.env modal run modal/app.py --no-resume --epochs 2

# ── Different config or run name ──────────────────────────────────────────────
uv run --env-file modal/.env modal run modal/app.py --config n2v3d.yaml --run-name test_base

# ── Monitor a running job ─────────────────────────────────────────────────────
modal app logs cidc25-training

# ── Volumes ───────────────────────────────────────────────────────────────────
#   cidc-data  →  /data   train/A1.tif … D2.tif + val/F0.tif … F3.tif
#   cidc-runs  →  /runs   best.pt, last.pt, train_*.jsonl  (persists between runs)
#
# Outputs download automatically to runs/<run-name>/ after training completes.
"""

from pathlib import Path
import os
import subprocess
import modal

# ── repo paths (resolved relative to this file) ────────────────────────────────
_HERE = Path(__file__).parent    # workspace/modal/
_ROOT = _HERE.parent             # workspace/

# ── config ─────────────────────────────────────────────────────────────────────
APP_NAME    = "cidc25-training"
DATA_VOLUME = "cidc-data"
RUNS_VOLUME = "cidc-runs"

# GPU options (uncomment to switch):
#   "t4"            — $0.59/h  ~12h  $7.08  (needs batch=8, grad_accum=2 in config)
#   "l40s"          — $1.95/h  ~5h   $9.75
#   "a100-80gb"     — $2.50/h  ~3h   $7.50
#   "h100"          — $3.95/h  ~2h   $7.90
#   "h200"          — $4.54/h  ~1.5h $6.81
#   "b200"          — $6.25/h  ~1.5h
#   "rtx-pro-6000"  — 96 GB VRAM, fits batch=16 natively
GPU_TYPE = "h200"

CONFIG_FILE = "n2v3d_large.yaml"   # winner from model comparison
RUN_NAME    = "full_training"       # sub-folder created under /runs/

REMOTE_DATA  = "/data"
REMOTE_RUNS  = "/runs"
REMOTE_APP   = "/app"

# ── volumes ────────────────────────────────────────────────────────────────────
data_vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
runs_vol = modal.Volume.from_name(RUNS_VOLUME, create_if_missing=True)

# ── image ──────────────────────────────────────────────────────────────────────
# Two-layer cache strategy:
#   Layer 1: heavy pip installs (PyTorch ~2 GB) — cached, rarely rebuilt
#   Layer 2: source code — rebuilt on every code change (fast, ~1 MB)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    # ── layer 1: heavy deps (cached) ──────────────────────────────────────────
    .pip_install(
        "torch==2.6.0+cu124",
        "torchvision==0.21.0+cu124",
        "torchaudio==2.6.0+cu124",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "numpy<2.0",
        "tifffile",
        "scipy",
        "pyyaml>=6.0",
        "einops>=0.8.0",
        "tqdm",
        "librosa",
        "hatchling",          # needed to build the cidc package
    )
    # ── layer 2: source code (copy=True required to allow pip install after) ────
    .add_local_dir(str(_ROOT / "src"),     remote_path=f"{REMOTE_APP}/src",          copy=True)
    # configs/ removed from image — passed as string at call time so config
    # changes never trigger a rebuild. Uncomment to revert to old behaviour:
    # .add_local_dir(str(_ROOT / "configs"), remote_path=f"{REMOTE_APP}/configs",    copy=True)
    .add_local_file(str(_ROOT / "pyproject.toml"), remote_path=f"{REMOTE_APP}/pyproject.toml", copy=True)
    # Install the cidc package (deps already in layer 1, --no-deps is instant)
    .run_commands(f"pip install --no-deps {REMOTE_APP}")
)

app = modal.App(name=APP_NAME)


# ── remote training function ───────────────────────────────────────────────────

@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        REMOTE_DATA: data_vol,
        REMOTE_RUNS: runs_vol,
    },
    timeout=60 * 60 * 8,    # 8-hour ceiling (full training ~3–5h)
)
def train_remote(
    config_yaml: str,               # full YAML content — passed from local, no image rebuild on change
    config_name: str = CONFIG_FILE,
    run_name: str = RUN_NAME,
    probe_only: bool = False,
    no_resume: bool = False,
    override_epochs: int = None,
):
    import subprocess as sp
    import tempfile
    import torch

    # Write the config to a temp file inside the container
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(config_yaml)
        config_path = f.name

    data_path = f"{REMOTE_DATA}/train"      # val_dir resolved as data/../val automatically
    out_path  = f"{REMOTE_RUNS}/{run_name}"

    # Print GPU info
    try:
        gpu_info = sp.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
        print(f"GPU: {gpu_info}")
    except Exception:
        pass

    amp_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    print(f"AMP dtype: {amp_dtype}")
    print()
    print("=" * 60)
    print("  CIDC25 Training")
    print("=" * 60)
    print(f"  Config : {config_name}  (written to {config_path})")
    print(f"  Data   : {data_path}   (val at {REMOTE_DATA}/val)")
    print(f"  Output : {out_path}")
    print(f"  Probe  : {probe_only}")
    print(f"  Resume : {not no_resume}")
    print("=" * 60)
    print()

    cmd = [
        "cidc", "train", config_path,
        "--data", data_path,
        "--out",  out_path,
    ]
    if probe_only:
        cmd.append("--probe-only")
    if no_resume:
        cmd.append("--no-resume")
    if override_epochs is not None:
        cmd += ["--override-epochs", str(override_epochs)]

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    result = sp.run(cmd, env=env)

    # Persist all checkpoint and log writes to the volume
    runs_vol.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

    print()
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Outputs: cidc-runs volume  →  /{run_name}/")
    print("=" * 60)

    return out_path


# ── local entrypoint ───────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    config: str = CONFIG_FILE,
    run_name: str = RUN_NAME,
    probe: bool = False,
    no_resume: bool = False,
    local_out: str = None,
    epochs: int = None,
):
    """
    Launch CIDC25 training on a Modal GPU.  Outputs download automatically.

    Examples:

        # Full training (default: n2v3d_large, A100 80GB)
        modal run modal/app.py

        # 4-batch pipeline sanity check (~2 min)
        modal run modal/app.py --probe

        # Ignore existing last.pt and start fresh
        modal run modal/app.py --no-resume

        # Different config / run name
        modal run modal/app.py --config n2v3d.yaml --run-name test_base
    """
    config_path = _ROOT / "configs" / config
    config_yaml = config_path.read_text()

    print(f"GPU      : {GPU_TYPE}")
    print(f"Config   : {config_path}")
    print(f"Run name : {run_name}")
    print()

    train_remote.remote(
        config_yaml=config_yaml,
        config_name=config,
        run_name=run_name,
        probe_only=probe,
        no_resume=no_resume,
        override_epochs=epochs,
    )

    # ── auto-download after training ──────────────────────────────────────────
    # Saves to workspace/runs/<run_name>/ by default
    out_dir = Path(local_out) if local_out else _ROOT / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥  Downloading outputs → {out_dir}/")

    downloaded, skipped = [], []
    for entry in runs_vol.listdir(run_name, recursive=True):
        rel = Path(entry.path).relative_to(run_name)
        local_path = out_dir / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip files already present with the same size (resume-safe)
        if local_path.exists() and local_path.stat().st_size == entry.size:
            skipped.append(rel.name)
            continue

        content = b"".join(runs_vol.read_file(entry.path))
        local_path.write_bytes(content)
        size_mb = len(content) / 1024 ** 2
        print(f"  ✅  {rel}  ({size_mb:.1f} MB)")
        downloaded.append(local_path)

    if skipped:
        print(f"  ⏭   skipped (already up-to-date): {', '.join(skipped)}")

    print(f"\n✅  Done!  {len(downloaded)} file(s) saved to {out_dir.resolve()}")
