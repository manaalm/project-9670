"""
Global configuration for the Git Re-Basin Spurious Features experiment.

This module defines all hyperparameters, paths, and settings used across
the entire experiment pipeline.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
DATA_DIR = PROJECT_ROOT / "data"

# =============================================================================
# Global Configuration Dictionary
# =============================================================================
CONFIG = {
    # Random seeds for reproducibility
    "seeds": {
        "global": 42,
        "model_A1": 1,
        "model_A2": 2,
        "model_R1": 1,
        "model_R2": 2,
    },

    # Dataset configuration
    "data": {
        "dataset": "CIFAR10",
        "num_classes": 10,
        "image_size": 32,
        "num_channels": 3,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },

    # Spurious patch configuration
    "patch": {
        "size": 6,  # Patch is 6x6 pixels
        "position": "top_left",  # Corner position
        "p_align_env_a": 0.95,  # Probability patch aligns with label in Env A
        "p_align_env_b": 0.05,  # Probability patch aligns with label in Env B
        # Colors for each class (10 distinct colors for CIFAR-10)
        "class_colors": [
            (255, 0, 0),      # 0: Red (airplane)
            (0, 255, 0),      # 1: Green (automobile)
            (0, 0, 255),      # 2: Blue (bird)
            (255, 255, 0),    # 3: Yellow (cat)
            (255, 0, 255),    # 4: Magenta (deer)
            (0, 255, 255),    # 5: Cyan (dog)
            (255, 128, 0),    # 6: Orange (frog)
            (128, 0, 255),    # 7: Purple (horse)
            (255, 255, 255),  # 8: White (ship)
            (128, 128, 128),  # 9: Gray (truck)
        ],
    },

    # Training configuration
    "training": {
        "batch_size": 128,
        "num_epochs": 30,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "lr_schedule": "cosine",  # 'cosine' or 'step'
        "num_workers": 4,
    },

    # Model configuration
    "model": {
        "architecture": "ConvNet",  # Simple ConvNet for easy rebasin
        "num_filters": [32, 64, 128, 256],  # Filter counts per conv block
        "fc_hidden": 256,
    },

    # Git Re-Basin configuration
    "rebasin": {
        "method": "weight_matching",
        "num_iterations": 100,  # For iterative matching
        "match_batch_size": 256,  # Batch size for activation matching
    },

    # Interpolation configuration
    "interpolation": {
        "num_alphas": 21,  # Number of interpolation points (0, 0.05, ..., 1.0)
        "eval_batch_size": 256,
    },

    # Robust training (mixture) configuration
    "robust_training": {
        "env_a_fraction": 0.5,  # Fraction of Env A in mixture
        "env_b_fraction": 0.5,  # Fraction of Env B in mixture
    },

    # Evaluation configuration
    "evaluation": {
        "ood_type": "no_patch",  # 'env_b' or 'no_patch'
    },
}

# =============================================================================
# Seed Setting Functions
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


# =============================================================================
# Directory Setup
# =============================================================================

def setup_directories():
    """Create all necessary directories."""
    for dir_path in [RESULTS_DIR, CHECKPOINTS_DIR, FIGURES_DIR, METRICS_DIR, DATA_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    return {
        "results": RESULTS_DIR,
        "checkpoints": CHECKPOINTS_DIR,
        "figures": FIGURES_DIR,
        "metrics": METRICS_DIR,
        "data": DATA_DIR,
    }


# =============================================================================
# CIFAR-10 Class Names
# =============================================================================

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_config():
    """Return a copy of the configuration dictionary."""
    import copy
    return copy.deepcopy(CONFIG)
