#!/usr/bin/env bash
# Cloud instance setup — run once after provisioning.
# Tested on: RunPod (CUDA 12.4), Lambda Labs, Vast.ai (Ubuntu 22.04 + CUDA 12.x)
#
# Usage:
#   bash cloud_setup.sh            # full setup (code already present via git clone)
#   bash cloud_setup.sh --data-only  # skip uv install, just download data

set -euo pipefail

DATA_ONLY=false
for arg in "$@"; do [[ "$arg" == "--data-only" ]] && DATA_ONLY=true; done

WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Workspace: $WORKSPACE"

# ── 1. Install uv ─────────────────────────────────────────────────────────────
if ! $DATA_ONLY; then
    if ! command -v uv &>/dev/null; then
        echo "[1/4] Installing uv ..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    else
        echo "[1/4] uv already installed ($(uv --version))"
    fi

    # ── 2. Install Python dependencies ────────────────────────────────────────
    echo "[2/4] Installing dependencies (uv sync) ..."
    cd "$WORKSPACE"
    uv sync
    echo "      Done."
fi

# ── 3. Download data from Zenodo ───────────────────────────────────────────────
echo "[3/4] Downloading data from Zenodo (MD5-verified, skips existing files) ..."
cd "$WORKSPACE"
uv run python scripts/download_data.py
echo "      Done."

# ── 4. Smoke-test the training pipeline ───────────────────────────────────────
if ! $DATA_ONLY; then
    echo "[4/4] Running training probe (4 batches, verifies full pipeline) ..."
    cd "$WORKSPACE/training"
    uv run python train.py --probe-only --data-dir "$WORKSPACE/data"
    echo "      Probe passed."
fi

echo ""
echo "Setup complete. To start training:"
echo ""
echo "  cd $WORKSPACE/training"
echo ""
echo "  # Ablation first (MSE vs NLL on A1/B1, ~10 epochs each):"
echo "  uv run python train.py --stacks A1 B1 --loss nll --epochs 10 --run-name ablation_nll"
echo "  uv run python train.py --stacks A1 B1 --loss mse --epochs 10 --run-name ablation_mse"
echo ""
echo "  # Full training (after picking the winning loss):"
echo "  uv run python train.py --loss nll --run-name full_v1"
echo ""
echo "  # Monitor progress:"
echo "  tail -f runs/\$(ls -t runs/ | head -1)/train.log"
