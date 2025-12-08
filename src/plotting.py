"""
Plotting utilities for visualizing experiment results.

This module provides standardized plotting functions for:
- Dataset samples visualization
- Training curves
- Interpolation paths
- Barrier comparisons
- Model comparison charts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .config import CONFIG, FIGURES_DIR, CIFAR10_CLASSES
from .data import denormalize

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Default figure settings
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 150


def save_figure(fig: plt.Figure, name: str, dpi: int = DEFAULT_DPI):
    """Save figure to the figures directory."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {path}")


def plot_sample_grid(
    images: np.ndarray,
    labels: np.ndarray,
    title: str = "Dataset Samples",
    nrow: int = 4,
    ncol: int = 4,
    figsize: Tuple[int, int] = (12, 12),
    class_names: List[str] = None,
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a grid of image samples.

    Args:
        images: Array of images (N, H, W, C) in [0, 255] uint8
        labels: Array of labels
        title: Figure title
        nrow, ncol: Grid dimensions
        figsize: Figure size
        class_names: Optional class name list
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    if class_names is None:
        class_names = CIFAR10_CLASSES

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    fig.suptitle(title, fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(class_names[labels[i]], fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (14, 5),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training curves (loss and accuracy).

    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'id_acc', 'ood_acc'
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, [x * 100 for x in history['train_acc']], 'b-',
             label='Train Acc', linewidth=2)
    ax2.plot(epochs, [x * 100 for x in history['id_acc']], 'g-',
             label='ID Test Acc', linewidth=2)
    ax2.plot(epochs, [x * 100 for x in history['ood_acc']], 'r-',
             label='OOD Test Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_multiple_training_curves(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "id_acc",
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a single metric across multiple models.

    Args:
        histories: Dictionary mapping model names to their histories
        metric: Which metric to plot
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history[metric]) + 1)
        values = history[metric]
        if 'acc' in metric:
            values = [x * 100 for x in values]
        ax.plot(epochs, values, label=name, linewidth=2, color=color)

    ax.set_xlabel('Epoch')
    ylabel = 'Accuracy (%)' if 'acc' in metric else metric.replace('_', ' ').title()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_interpolation_path(
    results: Dict[str, np.ndarray],
    title: str = "Interpolation Path",
    figsize: Tuple[int, int] = (14, 5),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Plot loss and accuracy along interpolation path.

    Args:
        results: Dictionary with 'alphas', 'losses', 'accuracies'
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    alphas = results['alphas']
    losses = results['losses']
    accs = results['accuracies'] * 100

    # Loss plot
    ax1.plot(alphas, losses, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Interpolation')
    ax1.grid(True, alpha=0.3)

    # Mark barrier
    max_loss_idx = np.argmax(losses)
    ax1.axhline(y=losses[max_loss_idx], color='r', linestyle='--', alpha=0.5)
    ax1.scatter([alphas[max_loss_idx]], [losses[max_loss_idx]], color='r', s=100, zorder=5)

    # Accuracy plot
    ax2.plot(alphas, accs, 'g-o', linewidth=2, markersize=4)
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy vs Interpolation')
    ax2.grid(True, alpha=0.3)

    # Mark minimum
    min_acc_idx = np.argmin(accs)
    ax2.axhline(y=accs[min_acc_idx], color='r', linestyle='--', alpha=0.5)
    ax2.scatter([alphas[min_acc_idx]], [accs[min_acc_idx]], color='r', s=100, zorder=5)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_interpolation_comparison(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    metric: str = "losses",
    title: str = "Interpolation Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Compare interpolation paths for multiple model pairs.

    Args:
        results_dict: Dictionary mapping pair names to interpolation results
        metric: 'losses' or 'accuracies'
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    for i, (name, results) in enumerate(results_dict.items()):
        alphas = results['alphas']
        values = results[metric]
        if metric == 'accuracies':
            values = values * 100

        ax.plot(alphas, values, label=name, linewidth=2,
                color=colors[i], marker=markers[i % len(markers)], markersize=5)

    ax.set_xlabel(r'$\alpha$')
    ylabel = 'Accuracy (%)' if metric == 'accuracies' else 'Loss'
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_pre_post_rebasin(
    pre_results: Dict[str, np.ndarray],
    post_results: Dict[str, np.ndarray],
    dataset_name: str = "ID",
    title: str = "Pre vs Post Re-Basin",
    figsize: Tuple[int, int] = (14, 5),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Compare interpolation before and after rebasing.

    Args:
        pre_results: Results before rebasing
        post_results: Results after rebasing
        dataset_name: Name of dataset being evaluated
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    alphas = pre_results['alphas']

    # Loss comparison
    ax1.plot(alphas, pre_results['losses'], 'r-o', label='Pre-Rebasin',
             linewidth=2, markersize=4)
    ax1.plot(alphas, post_results['losses'], 'b-s', label='Post-Rebasin',
             linewidth=2, markersize=4)
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{dataset_name} Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy comparison
    ax2.plot(alphas, pre_results['accuracies'] * 100, 'r-o', label='Pre-Rebasin',
             linewidth=2, markersize=4)
    ax2.plot(alphas, post_results['accuracies'] * 100, 'b-s', label='Post-Rebasin',
             linewidth=2, markersize=4)
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{dataset_name} Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_barrier_comparison(
    barriers: Dict[str, Dict[str, float]],
    title: str = "Loss Barrier Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Bar plot comparing barriers across model pairs.

    Args:
        barriers: Dictionary mapping pair names to barrier metrics
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    pairs = list(barriers.keys())
    x = np.arange(len(pairs))
    width = 0.35

    pre_barriers = [barriers[p].get('pre_loss_barrier', 0) for p in pairs]
    post_barriers = [barriers[p].get('post_loss_barrier', 0) for p in pairs]

    bars1 = ax.bar(x - width/2, pre_barriers, width, label='Pre-Rebasin', color='salmon')
    bars2 = ax.bar(x + width/2, post_barriers, width, label='Post-Rebasin', color='steelblue')

    ax.set_xlabel('Model Pair')
    ax.set_ylabel('Loss Barrier')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_spurious_reliance_comparison(
    metrics: Dict[str, Dict[str, float]],
    title: str = "Spurious Reliance Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Compare spurious reliance metrics across models.

    Args:
        metrics: Dictionary mapping model names to their metrics
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    models = list(metrics.keys())
    x = np.arange(len(models))

    # SRS comparison
    srs_values = [metrics[m]['spurious_reliance_score'] for m in models]
    axes[0].bar(x, srs_values, color='coral')
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('SRS')
    axes[0].set_title('Spurious Reliance Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')

    # OOD Drop comparison
    ood_drops = [metrics[m]['ood_drop'] * 100 for m in models]
    axes[1].bar(x, ood_drops, color='steelblue')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('OOD Drop (%)')
    axes[1].set_title('OOD Accuracy Drop')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')

    # CF Flip Rate comparison
    flip_rates = [metrics[m]['cf_flip_rate'] * 100 for m in models]
    axes[2].bar(x, flip_rates, color='seagreen')
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('Flip Rate (%)')
    axes[2].set_title('Counterfactual Flip Rate')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_srs_interpolation(
    alphas: np.ndarray,
    srs_values: List[float],
    title: str = "Spurious Reliance Along Interpolation",
    figsize: Tuple[int, int] = (10, 6),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Plot spurious reliance score along interpolation path.

    Args:
        alphas: Interpolation alpha values
        srs_values: SRS at each alpha
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(alphas, srs_values, 'purple', linewidth=2, marker='o', markersize=5)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel('Spurious Reliance Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Shade the region between endpoints
    endpoint_avg = (srs_values[0] + srs_values[-1]) / 2
    ax.axhline(y=endpoint_avg, color='gray', linestyle='--', alpha=0.5,
               label='Endpoint Average')
    ax.legend()

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig


def plot_summary_table(
    data: Dict[str, Dict[str, float]],
    title: str = "Summary",
    figsize: Tuple[int, int] = (12, 4),
    save_name: Optional[str] = None,
) -> plt.Figure:
    """
    Create a table visualization of summary metrics.

    Args:
        data: Nested dictionary of metrics
        title: Figure title
        figsize: Figure size
        save_name: If provided, save figure with this name

    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Convert to table format
    rows = list(data.keys())
    cols = list(data[rows[0]].keys()) if rows else []

    cell_text = []
    for row in rows:
        row_data = []
        for col in cols:
            val = data[row].get(col, 0)
            if isinstance(val, float):
                row_data.append(f"{val:.4f}")
            else:
                row_data.append(str(val))
        cell_text.append(row_data)

    table = ax.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=cols,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title(title, fontsize=14, pad=20)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name)

    return fig
