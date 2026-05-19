"""Generate summary table and recommendations."""

import numpy as np


def print_summary_table(results: dict, gains: dict):
    """Print markdown summary table of all metrics."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print()
    print("| Stack | Mean Int | Mean Var | Active % | stSNR | sSNR | tSNR | τ(0.5) | Gain   |")
    print("|-------|----------|----------|----------|-------|------|------|--------|--------|")

    for sn in ["F0", "F1", "F2", "F3"]:
        r = results[sn]
        g = gains[sn]["gain"] if sn in gains else np.nan
        print(
            f"| {sn}     | {r['mean_int']:8.1f} | {r['mean_var']:8.1f} | {r['active_ratio']:8.2f} | {r['stsn']:5.3f} | {r['ssnr']:4.3f} | {r['tsnr']:4.3f} | {r['tau']:6.1f} | {g:6.4f} |"
        )


def analyze_and_recommend(results: dict, gains: dict):
    """Analyze metrics and print recommendations."""
    print("\n" + "=" * 100)
    print("ANALYSIS & RECOMMENDATIONS")
    print("=" * 100)

    # 1. Gain variation
    gain_vals = [gains[sn]["gain"] for sn in ["F0", "F1", "F2", "F3"]]
    gain_ratio = max(gain_vals) / min(gain_vals)
    print(f"\n1. GAIN VARIATION: {gain_ratio:.2f}× ratio")
    print(f"   Gains: {[f'{g:.4f}' for g in gain_vals]}")
    if gain_ratio < 1.5:
        print("   ✓ Stacks similar → use ONE augmentation range")
    else:
        print(f"   ⚠ Stacks diverge → need PER-STACK weighting or wider augmentation")

    # 2. Active pixels
    active_vals = [results[sn]["active_ratio"] for sn in ["F0", "F1", "F2", "F3"]]
    min_active = min(active_vals)
    print(f"\n2. ACTIVE PIXELS (neurons): min = {min_active:.2f}%")
    if min_active < 5:
        print("   ⚠ CRITICAL: random patch sampling BROKEN → use STRATIFIED sampling")
    else:
        print("   ✓ Random patch sampling OK")

    # 3. Raw difficulty
    stsn_vals = [results[sn]["stsn"] for sn in ["F1", "F2", "F3"]]
    worst_stsn = min(stsn_vals)
    print(f"\n3. RAW DIFFICULTY (stSNR for noisy stacks): {worst_stsn:.3f}")
    if worst_stsn < 0.5:
        print("   ⚠ Very little headroom for improvement")
    elif worst_stsn < 1.0:
        print("   ~ Moderate headroom")
    else:
        print("   ✓ Plenty of headroom for improvement")

    # 4. Temporal structure
    taus = [results[sn]["tau"] for sn in ["F0", "F1", "F2", "F3"]]
    tau_std = float(np.std(taus))
    print(f"\n4. TEMPORAL COHERENCE (τ): {[f'{t:.1f}' for t in taus]}, std = {tau_std:.1f}")
    if tau_std < 10:
        print("   ✓ τ≈46 is UNIVERSAL across all stacks")
    else:
        print("   ⚠ τ is STACK-SPECIFIC")

    # Recommendation
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    if gain_ratio < 1.5 and min_active >= 5:
        print("\n✓ JOINT TRAINING RECOMMENDED")
        print("  - All 4 stacks are similar enough")
        print("  - Use ONE model with standard augmentation")
    else:
        print("\n⚠ PER-STACK STRATEGY RECOMMENDED")
        if gain_ratio >= 1.5:
            print(f"  - Gain diverges by {gain_ratio:.2f}×")
        if min_active < 5:
            print(f"  - Active pixel ratio too low ({min_active:.1f}%)")
        print("  - Consider separate models or heavy per-stack weighting")

    print("\n" + "=" * 100)
