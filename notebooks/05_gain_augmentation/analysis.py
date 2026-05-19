"""Analyze gain augmentation effects."""


def analyze_gain_robustness(results: list) -> None:
    """Analyze how gain variation affects performance."""
    print(f"\n{'='*80}")
    print("GAIN ROBUSTNESS ANALYSIS")
    print(f"{'='*80}\n")
    
    if len(results) > 0:
        baseline = results[[r['gain_factor'] for r in results].index(1.0)]['stsnr']
        print(f"Baseline (g=1.0): {baseline:.2f} dB\n")
        
        print(f"{'Gain':<8} {'stSNR':<8} {'Drop (dB)':<12} {'% Loss':<10}")
        print("-" * 40)
        
        for r in results:
            if r['gain_factor'] != 1.0:
                drop = baseline - r['stsnr']
                pct_loss = (drop / baseline * 100) if baseline > 0 else 0
                print(f"{r['gain_factor']:<8.1f} {r['stsnr']:<8.2f} {drop:<12.2f} {pct_loss:<10.1f}%")
