"""Plotting for loss comparison."""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_comparison(results: dict) -> None:
    """Plot loss values across stacks and loss types."""
    if not results:
        print("No results to plot")
        return
    
    loss_types = list(results[list(results.keys())[0]].keys())
    stacks = ["F1", "F2", "F3"]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(stacks))
    width = 0.25
    
    for i, loss_type in enumerate(loss_types):
        values = [results[s].get(loss_type, 0) for s in stacks]
        ax.bar(x + i*width, values, width, label=loss_type)
    
    ax.set_xlabel("Validation Stack")
    ax.set_ylabel("Loss Value")
    ax.set_title("Loss Function Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(stacks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    plt.show()
