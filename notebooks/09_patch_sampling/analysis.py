"""Analysis and decision making for patch sampling strategy."""


def print_decision(active_ratio: float):
    """Print sampling strategy decision based on active patch ratio."""
    print(f"\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)

    if active_ratio > 20:
        print(f"✓ {active_ratio:.1f}% active → RANDOM SAMPLING FINE")
        print("  Enough signal in random patches")
    elif active_ratio > 5:
        print(f"⚠ {active_ratio:.1f}% active → BORDERLINE")
        print("  Consider smart sampler to ensure more neurons in batches")
    else:
        print(f"🚨 {active_ratio:.1f}% active → CRITICAL")
        print("  MUST use stratified/weighted sampling")
        print(f"  Random sampling wastes {100-active_ratio:.1f}% of compute on background")

    print(f"\n" + "=" * 80)


def print_sampler_comparison(random_ratio: float, smart_ratio: float):
    """Compare random vs smart sampler results."""
    improvement = smart_ratio - random_ratio

    print(f"\nRandom sampling: {random_ratio:.2f}% active")
    print(f"Smart sampler (80% active bias): {smart_ratio:.2f}% active")
    print(f"Improvement: +{improvement:.2f}%")


def print_summary(active_ratio: float):
    """Print final summary and recommendations."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nRandom sampling: {active_ratio:.2f}% active patches")
    print(f"\nIf using random sampling:")
    print(f"  - {100-active_ratio:.1f}% of batches are background")
    print(f"  - Model wastes compute learning empty regions")
    print(f"\nIf using smart sampler (80% active):")
    print(f"  - 80% of batches guaranteed to have neurons")
    print(f"  - 20% still see background (keeps model honest)")
    print(f"  - Much more efficient training")
    print(f"\n" + "=" * 80)
