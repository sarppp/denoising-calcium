#!/usr/bin/env bash
# Cloud instance setup — run once after provisioning.
# Tested on: RunPod (CUDA 12.4), Lambda Labs, Lightning Studio, Vast.ai (Ubuntu 22.04 + CUDA 12.x)
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
    # Force Python 3.12 — PyTorch does not have wheels for 3.13+ yet.
    echo "[2/4] Installing dependencies (Python 3.12) ..."
    cd "$WORKSPACE"
    uv venv --python 3.12
    uv sync --python 3.12
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
    cd "$WORKSPACE"
    uv run cidc train configs/ablation_mse.yaml \
        --data "$WORKSPACE/data/train" \
        --out  /tmp/cidc_probe \
        --probe-only
    echo "      Probe passed."
fi

echo ""
echo "Setup complete."
echo ""
echo "  export DATA=$WORKSPACE/data/train"
echo "  export RUNS=$WORKSPACE/runs"
echo ""
echo "  # Step 1 — 5-arm loss ablation (10 epochs each, ~15-20 min per arm on T4):"
echo "  uv run cidc train configs/ablation_nll.yaml          --data \$DATA --out \$RUNS/nll"
echo "  uv run cidc train configs/ablation_mse.yaml          --data \$DATA --out \$RUNS/mse"
echo "  uv run cidc train configs/ablation_mae.yaml          --data \$DATA --out \$RUNS/mae"
echo "  uv run cidc train configs/ablation_anscombe_mse.yaml --data \$DATA --out \$RUNS/anscombe_mse"
echo "  uv run cidc train configs/ablation_huber.yaml        --data \$DATA --out \$RUNS/huber"
echo ""
echo "  # Step 2 — Read verdict and pick winning loss:"
echo "  uv run python scripts/ablation_verdict.py \$RUNS/nll \$RUNS/mse \$RUNS/mae \$RUNS/anscombe_mse \$RUNS/huber --stack F1"
echo ""
echo "  # Step 3 — Full training (edit loss.name in config first):"
echo "  uv run cidc train configs/n2v3d.yaml --data \$DATA --out \$RUNS/n2v3d_full"
echo ""
echo "  See NEXT_STEPS.md for the complete roadmap."
