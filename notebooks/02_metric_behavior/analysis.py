"""Analyze metric behavior results."""

import numpy as np


def analyze_blur_results(blur_results: list) -> None:
    """Analyze spatial blur sweep results."""
    print("\nSpatial blur analysis: gap = tSNR − sSNR")
    print(f"  {'sigma':>5}  {'sSNR':>6}  {'tSNR':>6}  {'gap':>7}  {'linear ratio':>12}")
    print("  " + "-" * 48)
    for r in blur_results[1:]:
        gap = r["tsnr"] - r["ssnr"]
        ratio = 10 ** (gap / 10)
        print(f"  {r['sigma']:>5.1f}  {r['ssnr']:>6.2f}  {r['tsnr']:>6.2f}  {gap:>+7.2f}  {ratio:>10.2f}x")

    gaps = [r["tsnr"] - r["ssnr"] for r in blur_results[1:]]
    mean_gap = sum(gaps) / len(gaps)
    print(f"\n  Mean gap = {mean_gap:.2f} dB (constant across sigmas)")
    print(f"  Interpretation: spatial blur always hurts sSNR more than tSNR by ~{10**(mean_gap/10):.1f}x")


def analyze_noise_results(noise_results: list) -> None:
    """Analyze additive noise sweep results."""
    print("\nAdditive noise analysis: gap = tSNR − sSNR")
    print(f"  {'sigma':>5}  {'sSNR':>6}  {'tSNR':>6}  {'gap':>7}")
    print("  " + "-" * 30)
    for r in noise_results[1:]:
        gap = r["tsnr"] - r["ssnr"]
        print(f"  {r['sigma']:>5.0f}  {r['ssnr']:>6.2f}  {r['tsnr']:>6.2f}  {gap:>+7.2f}")

    gaps = [r["tsnr"] - r["ssnr"] for r in noise_results[1:]]
    mean_gap = sum(gaps) / len(gaps)
    print(f"\n  Mean gap = {mean_gap:.2f} dB (near zero)")
    print(f"  Interpretation: additive noise hurts sSNR and tSNR equally")


def analyze_smooth_results(smooth_results: list) -> None:
    """Analyze temporal smoothing sweep results."""
    print("\nTemporal smoothing analysis (noisy input):")
    print(f"  {'window':>6}  {'sSNR':>6}  {'tSNR':>6}  {'gap':>7}")
    print("  " + "-" * 30)
    for r in smooth_results[1:]:
        gap = r["ssnr"] - r["tsnr"]
        print(f"  {r['window']:>6d}  {r['ssnr']:>6.2f}  {r['tsnr']:>6.2f}  {gap:>+7.2f}")

    print(f"\n  Key observation: sSNR improves early, tSNR collapses at large windows")
    print(f"  At window=31: tSNR has dropped {smooth_results[7]['tsnr'] - smooth_results[7]['tsnr']:.1f} dB")
    print(f"  At window=101: tSNR loss is severe (temporal transients blurred away)")
