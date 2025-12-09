"""
Model pair enumeration and loading utilities.

This module provides:
- Pair type definitions (spurious-spurious, robust-robust, spurious-robust)
- Helper functions to load model pairs from checkpoints
- Utilities to build a pairs dataframe for analysis
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import pandas as pd

from .config import CHECKPOINTS_DIR, get_device
from .models import create_model, ConvNet
from .train import load_model


# Pair type constants
PAIR_TYPE_SS = "spurious-spurious"
PAIR_TYPE_RR = "robust-robust"
PAIR_TYPE_SR = "spurious-robust"


@dataclass
class ModelPair:
    """Represents a pair of models for analysis."""
    name: str
    model_a_name: str
    model_b_name: str
    pair_type: str
    model_a: Optional[ConvNet] = None
    model_b: Optional[ConvNet] = None
    model_b_aligned: Optional[ConvNet] = None


def get_model_type(model_name: str) -> str:
    """
    Determine if a model is spurious or robust based on its name.

    Args:
        model_name: Model name (e.g., 'A1', 'R2')

    Returns:
        'spurious' or 'robust'
    """
    if model_name.startswith('A'):
        return 'spurious'
    elif model_name.startswith('R'):
        return 'robust'
    else:
        raise ValueError(f"Unknown model name format: {model_name}")


def get_pair_type(model_a_name: str, model_b_name: str) -> str:
    """
    Determine the pair type based on model names.

    Args:
        model_a_name: First model name
        model_b_name: Second model name

    Returns:
        Pair type string (e.g., 'spurious-spurious')
    """
    type_a = get_model_type(model_a_name)
    type_b = get_model_type(model_b_name)

    if type_a == 'spurious' and type_b == 'spurious':
        return PAIR_TYPE_SS
    elif type_a == 'robust' and type_b == 'robust':
        return PAIR_TYPE_RR
    else:
        return PAIR_TYPE_SR


def get_standard_pairs() -> List[Tuple[str, str, str]]:
    """
    Get the standard model pairs used in the experiment.

    Returns:
        List of (model_a_name, model_b_name, pair_type) tuples
    """
    return [
        ('A1', 'A2', PAIR_TYPE_SS),
        ('R1', 'R2', PAIR_TYPE_RR),
        ('A1', 'R1', PAIR_TYPE_SR),
    ]


def get_extended_pairs() -> List[Tuple[str, str, str]]:
    """
    Get an extended set of model pairs including cross combinations.

    Returns:
        List of (model_a_name, model_b_name, pair_type) tuples
    """
    standard = get_standard_pairs()
    extended = [
        ('S2', 'R2', PAIR_TYPE_SR),  # Alternative S-R pair
        ('A1', 'R2', PAIR_TYPE_SR),  # Cross pair
        ('A2', 'R1', PAIR_TYPE_SR),  # Cross pair
    ]
    return standard + extended


def load_model_by_name(
    model_name: str,
    device: torch.device,
    config: Optional[Dict] = None,
    checkpoints_dir: Optional[Path] = None,
) -> ConvNet:
    """
    Load a model by its name from the checkpoints directory.

    Args:
        model_name: Model name (e.g., 'A1', 'R2')
        device: Torch device
        config: Optional config dict
        checkpoints_dir: Optional custom checkpoints directory

    Returns:
        Loaded model in eval mode
    """
    ckpt_dir = checkpoints_dir or CHECKPOINTS_DIR
    checkpoint_path = ckpt_dir / f"model_{model_name}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = create_model(config)
    model = load_model(model, checkpoint_path, device)
    return model


def load_aligned_model(
    ref_name: str,
    aligned_name: str,
    device: torch.device,
    config: Optional[Dict] = None,
    checkpoints_dir: Optional[Path] = None,
) -> Optional[ConvNet]:
    """
    Load an aligned model (model_B aligned to model_A).

    Args:
        ref_name: Reference model name (e.g., 'A1')
        aligned_name: Aligned model name (e.g., 'A2')
        device: Torch device
        config: Optional config dict
        checkpoints_dir: Optional custom checkpoints directory

    Returns:
        Loaded aligned model, or None if not found
    """
    ckpt_dir = checkpoints_dir or CHECKPOINTS_DIR
    checkpoint_path = ckpt_dir / f"model_{aligned_name}_aligned_to_{ref_name}.pt"

    if not checkpoint_path.exists():
        return None

    model = create_model(config)
    model = load_model(model, checkpoint_path, device)
    return model


def load_model_pair(
    model_a_name: str,
    model_b_name: str,
    device: torch.device,
    config: Optional[Dict] = None,
    checkpoints_dir: Optional[Path] = None,
    load_aligned: bool = True,
) -> ModelPair:
    """
    Load a complete model pair for analysis.

    Args:
        model_a_name: First model name
        model_b_name: Second model name
        device: Torch device
        config: Optional config dict
        checkpoints_dir: Optional custom checkpoints directory
        load_aligned: Whether to also load the aligned version of model_b

    Returns:
        ModelPair object with loaded models
    """
    pair_name = f"{model_a_name}-{model_b_name}"
    pair_type = get_pair_type(model_a_name, model_b_name)

    model_a = load_model_by_name(model_a_name, device, config, checkpoints_dir)
    model_b = load_model_by_name(model_b_name, device, config, checkpoints_dir)

    model_b_aligned = None
    if load_aligned:
        model_b_aligned = load_aligned_model(
            model_a_name, model_b_name, device, config, checkpoints_dir
        )

    return ModelPair(
        name=pair_name,
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        pair_type=pair_type,
        model_a=model_a,
        model_b=model_b,
        model_b_aligned=model_b_aligned,
    )


def load_all_standard_pairs(
    device: torch.device,
    config: Optional[Dict] = None,
    checkpoints_dir: Optional[Path] = None,
    load_aligned: bool = True,
) -> Dict[str, ModelPair]:
    """
    Load all standard model pairs.

    Args:
        device: Torch device
        config: Optional config dict
        checkpoints_dir: Optional custom checkpoints directory
        load_aligned: Whether to load aligned models

    Returns:
        Dictionary mapping pair names to ModelPair objects
    """
    pairs = {}
    for model_a, model_b, _ in get_standard_pairs():
        pair = load_model_pair(
            model_a, model_b, device, config, checkpoints_dir, load_aligned
        )
        pairs[pair.name] = pair
    return pairs


def create_pairs_dataframe(
    pairs_data: List[Dict],
) -> pd.DataFrame:
    """
    Create a pandas DataFrame from pairs analysis data.

    Args:
        pairs_data: List of dictionaries with pair analysis results

    Returns:
        DataFrame with one row per pair
    """
    df = pd.DataFrame(pairs_data)

    # Ensure proper column ordering
    column_order = [
        'pair_id', 'pair_type',
        'model_a', 'model_b',
        'barrier_id_raw', 'barrier_ood_raw',
        'barrier_id_rebasin', 'barrier_ood_rebasin',
        'dist_srs', 'dist_cka',
    ]

    # Add any extra columns that might be present
    extra_cols = [c for c in df.columns if c not in column_order]
    column_order = [c for c in column_order if c in df.columns] + extra_cols

    return df[column_order]


def get_pair_short_name(pair_type: str) -> str:
    """
    Get a short name for a pair type (for plotting).

    Args:
        pair_type: Full pair type string

    Returns:
        Short name (e.g., 'S-S', 'R-R', 'S-R')
    """
    mapping = {
        PAIR_TYPE_SS: 'S-S',
        PAIR_TYPE_RR: 'R-R',
        PAIR_TYPE_SR: 'S-R',
    }
    return mapping.get(pair_type, pair_type)


def check_checkpoints_exist(
    checkpoints_dir: Optional[Path] = None,
) -> Dict[str, bool]:
    """
    Check which model checkpoints exist.

    Args:
        checkpoints_dir: Optional custom checkpoints directory

    Returns:
        Dictionary mapping model names to existence status
    """
    ckpt_dir = checkpoints_dir or CHECKPOINTS_DIR

    models = ['A1', 'A2', 'R1', 'R2']
    aligned_pairs = [
        ('A1', 'A2'),
        ('R1', 'R2'),
        ('A1', 'R1'),
    ]

    status = {}

    # Check original models
    for name in models:
        path = ckpt_dir / f"model_{name}.pt"
        status[name] = path.exists()

    # Check aligned models
    for ref, aligned in aligned_pairs:
        path = ckpt_dir / f"model_{aligned}_aligned_to_{ref}.pt"
        key = f"{aligned}_aligned_to_{ref}"
        status[key] = path.exists()

    return status


def print_checkpoint_status(checkpoints_dir: Optional[Path] = None):
    """Print status of all expected checkpoints."""
    status = check_checkpoints_exist(checkpoints_dir)

    print("Checkpoint Status:")
    print("-" * 40)
    for name, exists in status.items():
        symbol = "[x]" if exists else "[ ]"
        print(f"  {symbol} {name}")
