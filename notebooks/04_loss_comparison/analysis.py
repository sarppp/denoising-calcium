"""Analyze loss function comparison results."""


def print_loss_summary(results: dict) -> None:
    """Print summary of loss function values."""
    print(f"\n{'='*80}")
    print("LOSS FUNCTION COMPARISON")
    print(f"{'='*80}\n")
    
    if not results:
        print("No results to display")
        return
    
    # Get all loss types
    loss_types = list(results[list(results.keys())[0]].keys())
    
    print(f"{'Stack':<10} ", end="")
    for loss_type in loss_types:
        print(f"{loss_type:>12}", end="")
    print()
    print("-" * (10 + 12 * len(loss_types)))
    
    for stack_name in ["F1", "F2", "F3"]:
        if stack_name in results:
            print(f"{stack_name:<10} ", end="")
            for loss_type in loss_types:
                print(f"{results[stack_name].get(loss_type, 0):>12.2f}", end="")
            print()
