"""
Training configuration — three sections, strictly separated by concern.

SECTION 1  DATA     Measured from notebooks 01-10. Locked. Do not modify.
SECTION 2  MODEL    Architecture choices. Ablatable.
SECTION 3  TRAIN    Optimisation schedule. Tunable.
"""

import math
from pathlib import Path

# ============================================================================
# SECTION 1 — DATA
# Measured values from notebooks 01-10. Every number has a source.
# These are facts about the data, not hyperparameters. Do not tune them.
# ============================================================================

# Per-stack Poisson-Gaussian noise parameters (nb03).
# A1/B1: frame-differencing method (R²≈0.70). C2/D2: mean-intensity method (R²≈0.93).
NOISE_PARAMS: dict[str, dict] = {
    'A1': {'g': 27.6,   'sigma_r_sq': 2490.0},
    'B1': {'g': 27.7,   'sigma_r_sq': 2490.0},
    'C2': {'g': 250.96, 'sigma_r_sq': 2700.0},
    'D2': {'g': 258.9,  'sigma_r_sq': 2700.0},
}

# Baseline stSNR (nb01) — the floor the model must beat on every stack.
BASELINE_STSNR: dict[str, float] = {
    'F1': 7.274,
    'F2': -0.794,
    'F3': -6.639,
}

# Patch depth: τ₀.₅ = 46 frames (nb01 ACF) → 2×τ = 92 → next power of 2.
# Height/width from nb09 patch sampling experiments.
PATCH_T: int = 128
PATCH_H: int = 128
PATCH_W: int = 128
PATCH_SIZE: tuple[int, int, int] = (PATCH_T, PATCH_H, PATCH_W)

# Gain augmentation range (nb10, 3× safety margin for OOD generalisation).
# Applied as LogUniform — equal probability per decade across [G_AUG_MIN, G_AUG_MAX].
# Linear uniform is wrong here: it over-samples high gain and under-samples low gain.
G_AUG_MIN: float = 15.0
G_AUG_MAX: float = 1500.0
G_AUG_LOG_MIN: float = math.log(G_AUG_MIN)
G_AUG_LOG_MAX: float = math.log(G_AUG_MAX)

# Geometric mean gain — used as the inference-time noise map gain.
# The model sees the full [G_AUG_MIN, G_AUG_MAX] range during training;
# the geometric mean is the least-biased single value to use at inference.
G_INFER: float = math.exp((G_AUG_LOG_MIN + G_AUG_LOG_MAX) / 2)  # ≈ 150

# Read noise for augmented batches — median across training stacks.
SIGMA_R_SQ_AUG: float = 2700.0

# N2V3D masking (nb06 — mask_size=1 voxel, single-point blind-spot).
MASK_RATIO: float = 0.005  # 0.5 % of voxels predicted per patch

# Training and validation stack names.
TRAIN_STACKS: list[str] = ['A1', 'B1', 'C2', 'D2']
VAL_STACKS:   list[str] = ['F1', 'F2', 'F3']

# ============================================================================
# SECTION 2 — MODEL
# Architecture choices. Safe to ablate without touching data constants.
# ============================================================================

IN_CHANNELS:  int       = 2           # [noisy, noise_map]
OUT_CHANNELS: int       = 1           # denoised frame
CHANNELS:     list[int] = [32, 64, 128]  # encoder feature dimensions (3-level UNet)

# ============================================================================
# SECTION 3 — TRAINING
# Optimisation schedule. Tune freely.
# ============================================================================

BATCH_SIZE:   int   = 4
LR:           float = 1e-4     # peak learning rate (after warmup)
LR_MIN:       float = 1e-6     # cosine decay floor
WARMUP_EPOCHS: int  = 5        # linear LR warmup before cosine
WEIGHT_DECAY: float = 1e-4     # AdamW regularisation
EPOCHS:       int   = 100
GRAD_CLIP:    float = 1.0

N_PATCHES_PER_STACK: int = 250  # patches sampled per stack per epoch

# Validation / checkpointing cadence
VAL_FREQ:      int   = 10    # fast eval every N epochs
CKPT_FREQ:     int   = 10    # checkpoint every N epochs
ES_PATIENCE:   int   = 3     # early-stopping checks without improvement
ES_MIN_DELTA:  float = 0.05  # minimum dB improvement to reset patience

# Fast eval (used during training): subsample every Nth frame to keep eval < 3 min.
FAST_EVAL_STRIDE: int = 5  # 1500 → 300 frames

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR:  Path = Path(__file__).parent.parent
DATA_DIR:  Path = BASE_DIR / "data"
RUNS_DIR:  Path = Path(__file__).parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)
