"""Analyze masking geometry."""


def print_masking_summary(results: dict) -> None:
    """Print masking geometry analysis."""
    print(f"\n{'='*80}")
    print("MASKING GEOMETRY ANALYSIS")
    print(f"{'='*80}\n")
    print(f"Mask size: {results['mask_size']}")
    print(f"Center patch mean intensity: {results['center_intensity_mean']:.1f}")
    print(f"Center patch variance: {results['center_intensity_var']:.1f}")
