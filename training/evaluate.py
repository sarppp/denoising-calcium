"""
Evaluation: stSNR = 0.5·sSNR + 0.5·tSNR.

All SNR functions are fully vectorised — no Python loops over pixels or frames.
The original loop-based implementation took O(H×W) iterations (240,100 on
490×490 stacks), making training-time validation impractical.

Two evaluation modes:
  fast   Subsample temporal axis (every FAST_EVAL_STRIDE frames).
         Used during training. < 3 min per stack on GPU.
  full   All 1500 frames. Used at submission / final evaluation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import time
import numpy as np
import torch
from cidc import load_stack

from config import (
    PATCH_SIZE, IN_CHANNELS, OUT_CHANNELS, CHANNELS,
    SIGMA_R_SQ_AUG, G_INFER, BASELINE_STSNR, FAST_EVAL_STRIDE,
)
from model import UNet3D


# ── SNR metrics (vectorised) ──────────────────────────────────────────────────

def spatial_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """sSNR: per-frame SNR averaged over time.  O(T·H·W), fully vectorised."""
    noise         = clean - denoised                       # [T, H, W]
    signal_power  = np.mean(clean ** 2,  axis=(1, 2))     # [T]
    noise_power   = np.mean(noise ** 2,  axis=(1, 2))     # [T]
    valid         = noise_power > 0
    if not valid.any():
        return 0.0
    return float(np.mean(signal_power[valid] / noise_power[valid]))


def temporal_snr(clean: np.ndarray, denoised: np.ndarray) -> float:
    """tSNR: per-pixel SNR averaged over space.  O(T·H·W), fully vectorised."""
    noise       = clean - denoised                # [T, H, W]
    signal_var  = np.var(clean,  axis=0)          # [H, W]
    noise_var   = np.var(noise,  axis=0)          # [H, W]
    valid       = noise_var > 0
    if not valid.any():
        return 0.0
    return float(np.mean(signal_var[valid] / noise_var[valid]))


def spatio_temporal_snr(
    clean:    np.ndarray,
    denoised: np.ndarray,
    alpha:    float = 0.5,
) -> dict[str, float]:
    """stSNR = α·sSNR + (1−α)·tSNR.  Returns all three components."""
    s = spatial_snr(clean, denoised)
    t = temporal_snr(clean, denoised)
    return {'stSNR': alpha * s + (1 - alpha) * t, 'sSNR': s, 'tSNR': t}


# ── Stack denoising ───────────────────────────────────────────────────────────

def _build_noise_map(patch: np.ndarray, g: float, sigma_r_sq: float) -> np.ndarray:
    """Noise map: σ_total = sqrt(σ_r² + g·signal), normalised to [0,1]."""
    std = np.sqrt(sigma_r_sq + g * np.maximum(patch, 0.0))
    return (std / (std.max() + 1e-8)).astype(np.float32)


def denoise_stack(
    model:      torch.nn.Module,
    stack:      np.ndarray,                     # [T, H, W]
    device:     torch.device,
    patch_size: int = PATCH_SIZE[0],
    overlap:    int | None = None,
    g:          float = G_INFER,
    sigma_r_sq: float = SIGMA_R_SQ_AUG,
) -> np.ndarray:
    """Sliding-window 3D denoising with overlap-averaging.

    overlap defaults to patch_size // 2 (50 % overlap).
    """
    if overlap is None:
        overlap = patch_size // 2
    stride = patch_size - overlap

    T, H, W   = stack.shape
    denoised  = np.zeros_like(stack, dtype=np.float64)
    count     = np.zeros_like(stack, dtype=np.float64)

    t_starts = list(range(0, max(1, T - patch_size + 1), stride))
    h_starts = list(range(0, max(1, H - patch_size + 1), stride))
    w_starts = list(range(0, max(1, W - patch_size + 1), stride))
    n_total   = len(t_starts) * len(h_starts) * len(w_starts)
    processed = 0

    print(f"    stack {T}×{H}×{W}  patch {patch_size}³  stride {stride}  "
          f"→ {n_total} patches")

    with torch.no_grad():
        for t in t_starts:
            for h in h_starts:
                for w in w_starts:
                    patch     = stack[t:t+patch_size, h:h+patch_size, w:w+patch_size]
                    noise_map = _build_noise_map(patch, g, sigma_r_sq)

                    x = np.stack([patch, noise_map], axis=0).astype(np.float32)
                    x = torch.from_numpy(x).unsqueeze(0).to(device)   # [1,2,T,H,W]

                    y_hat = model(x).squeeze(0).squeeze(0).cpu().numpy()  # [T,H,W]

                    denoised[t:t+patch_size, h:h+patch_size, w:w+patch_size] += y_hat
                    count[t:t+patch_size,    h:h+patch_size, w:w+patch_size] += 1

                    processed += 1
                    print(f"    {processed}/{n_total} ({100*processed//n_total}%)",
                          end='\r', flush=True)

    print(f"    {n_total}/{n_total} (100%)          ")
    np.divide(denoised, count, where=count > 0, out=denoised)
    return denoised.astype(np.float32)


# ── Evaluation runner ─────────────────────────────────────────────────────────

def evaluate(
    model:      torch.nn.Module,
    data_dir:   Path,
    device:     torch.device,
    fast:       bool = False,
    patch_size: int  = PATCH_SIZE[0],
    overlap:    int | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate the model on F1/F2/F3 against clean F0.

    Args:
        fast: If True, subsample every FAST_EVAL_STRIDE frames (training-time eval).
              If False, use all frames (submission eval).

    Returns:
        dict mapping stack name → {'stSNR', 'sSNR', 'tSNR'}
    """
    model.eval()
    stride_label = f"every {FAST_EVAL_STRIDE}th frame" if fast else "all frames"
    print(f"  Evaluation mode: {'fast' if fast else 'full'} ({stride_label})")

    clean_full = np.asarray(
        load_stack(data_dir / "val" / "F0.tif"), dtype=np.float32
    )
    if fast:
        clean = clean_full[::FAST_EVAL_STRIDE]
    else:
        clean = clean_full

    results: dict[str, dict[str, float]] = {}

    for name in ['F1', 'F2', 'F3']:
        noisy_full = np.asarray(
            load_stack(data_dir / "val" / f"{name}.tif"), dtype=np.float32
        )
        if fast:
            noisy = noisy_full[::FAST_EVAL_STRIDE]
        else:
            noisy = noisy_full

        t0       = time.time()
        denoised = denoise_stack(model, noisy, device, patch_size=patch_size, overlap=overlap)
        metrics  = spatio_temporal_snr(clean, denoised)
        elapsed  = time.time() - t0

        baseline = BASELINE_STSNR[name]
        delta    = metrics['stSNR'] - baseline
        sign     = '+' if delta >= 0 else ''
        print(f"  {name}  stSNR={metrics['stSNR']:+.2f} dB  "
              f"sSNR={metrics['sSNR']:+.2f}  tSNR={metrics['tSNR']:+.2f}  "
              f"Δbaseline={sign}{delta:.2f}  ({elapsed:.0f}s)")

        results[name] = {**metrics, 'elapsed_s': elapsed}

    return results


def combined_score(results: dict[str, dict[str, float]]) -> float:
    """Competition scoring: 0.5×F1 + 0.5×mean(F2,F3)."""
    return 0.5 * results['F1']['stSNR'] + 0.5 * (
        (results['F2']['stSNR'] + results['F3']['stSNR']) / 2
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return 1

    model = UNet3D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, channels=CHANNELS)
    ckpt  = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt)
    model = model.to(device)

    print(f"\nLoaded: {model_path}")
    print("=" * 72)

    results = evaluate(
        model,
        data_dir   = Path(args.data_dir),
        device     = device,
        fast       = args.fast,
        patch_size = args.patch_size,
    )

    score = combined_score(results)
    total = sum(r['elapsed_s'] for r in results.values())

    print("=" * 72)
    print(f"Combined score: {score:+.2f} dB")
    print(f"Total time:     {total/60:.1f} min", end="")
    if total > 3600:
        print("  ← EXCEEDS 60-min submission limit")
    else:
        print(f"  ({(3600-total)/60:.0f} min remaining under limit)")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained denoiser')
    parser.add_argument('--model',      default='checkpoints/model_final.pt')
    parser.add_argument('--data-dir',   default=str(Path(__file__).parent.parent / "data"))
    parser.add_argument('--patch-size', type=int, default=PATCH_SIZE[0])
    parser.add_argument('--fast',       action='store_true',
                        help='Subsample temporal axis for quick check')
    args = parser.parse_args()
    raise SystemExit(main(args))
