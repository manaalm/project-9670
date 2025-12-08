"""
Weight interpolation utilities for loss landscape analysis.

This module provides functions for:
- Linear interpolation between model weights
- Evaluating models along interpolation paths
- Computing loss barriers
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm

from .models import ConvNet, clone_model
from .train import evaluate_model


def interpolate_state_dicts(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """
    Linearly interpolate between two state dicts.

    θ(α) = α * θ_a + (1 - α) * θ_b

    Note: At α=0, we get θ_b; at α=1, we get θ_a

    Args:
        state_a: First model's state dict
        state_b: Second model's state dict
        alpha: Interpolation coefficient

    Returns:
        Interpolated state dict
    """
    interpolated = {}

    for key in state_a.keys():
        if state_a[key].dtype in [torch.float32, torch.float64, torch.float16]:
            interpolated[key] = alpha * state_a[key] + (1 - alpha) * state_b[key]
        else:
            # For non-float tensors (like num_batches_tracked), use first model's value
            interpolated[key] = state_a[key].clone()

    return interpolated


def create_interpolated_model(
    model_a: ConvNet,
    model_b: ConvNet,
    alpha: float,
    device: torch.device,
) -> ConvNet:
    """
    Create a model with interpolated weights.

    Args:
        model_a: First model (at α=1)
        model_b: Second model (at α=0)
        alpha: Interpolation coefficient
        device: Torch device

    Returns:
        New model with interpolated weights
    """
    # Clone architecture from model_a
    interp_model = clone_model(model_a)
    interp_model = interp_model.to(device)

    # Get state dicts
    state_a = {k: v.to(device) for k, v in model_a.state_dict().items()}
    state_b = {k: v.to(device) for k, v in model_b.state_dict().items()}

    # Interpolate
    interp_state = interpolate_state_dicts(state_a, state_b, alpha)

    # Load interpolated state
    interp_model.load_state_dict(interp_state)

    return interp_model


def evaluate_interpolation_path(
    model_a: ConvNet,
    model_b: ConvNet,
    dataloader: DataLoader,
    device: torch.device,
    num_alphas: int = 21,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, np.ndarray]:
    """
    Evaluate models along the linear interpolation path.

    Args:
        model_a: First model (endpoint at α=1)
        model_b: Second model (endpoint at α=0)
        dataloader: DataLoader for evaluation
        device: Torch device
        num_alphas: Number of interpolation points
        criterion: Loss function (default: CrossEntropyLoss)

    Returns:
        Dictionary with 'alphas', 'losses', 'accuracies'
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    alphas = np.linspace(0, 1, num_alphas)
    losses = []
    accuracies = []

    for alpha in tqdm(alphas, desc="Interpolation"):
        # Create interpolated model
        interp_model = create_interpolated_model(model_a, model_b, alpha, device)

        # Evaluate
        loss, acc = evaluate_model(interp_model, dataloader, device, criterion)
        losses.append(loss)
        accuracies.append(acc)

    return {
        "alphas": alphas,
        "losses": np.array(losses),
        "accuracies": np.array(accuracies),
    }


def evaluate_interpolation_multi_dataset(
    model_a: ConvNet,
    model_b: ConvNet,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    num_alphas: int = 21,
    extra_metrics_fn: Optional[Callable] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Evaluate interpolation path on multiple datasets.

    Args:
        model_a: First model
        model_b: Second model
        dataloaders: Dictionary of named dataloaders (e.g., {'id': ..., 'ood': ...})
        device: Torch device
        num_alphas: Number of interpolation points
        extra_metrics_fn: Optional function to compute additional metrics
                         Signature: fn(model, device) -> Dict[str, float]

    Returns:
        Dictionary mapping dataset names to evaluation results
    """
    criterion = nn.CrossEntropyLoss()
    alphas = np.linspace(0, 1, num_alphas)

    results = {name: {"alphas": alphas, "losses": [], "accuracies": []}
               for name in dataloaders.keys()}

    if extra_metrics_fn is not None:
        results["extra_metrics"] = {"alphas": alphas}

    for alpha in tqdm(alphas, desc="Interpolation"):
        # Create interpolated model
        interp_model = create_interpolated_model(model_a, model_b, alpha, device)

        # Evaluate on each dataset
        for name, loader in dataloaders.items():
            loss, acc = evaluate_model(interp_model, loader, device, criterion)
            results[name]["losses"].append(loss)
            results[name]["accuracies"].append(acc)

        # Compute extra metrics if provided
        if extra_metrics_fn is not None:
            extra = extra_metrics_fn(interp_model, device)
            for key, value in extra.items():
                if key not in results["extra_metrics"]:
                    results["extra_metrics"][key] = []
                results["extra_metrics"][key].append(value)

    # Convert lists to arrays
    for name in results:
        for key in results[name]:
            if isinstance(results[name][key], list):
                results[name][key] = np.array(results[name][key])

    return results


def compute_loss_barrier(
    losses: np.ndarray,
    alphas: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute the loss barrier along an interpolation path.

    Barrier = max_α L(θ_α) - max(L(θ_0), L(θ_1))

    Args:
        losses: Array of losses at each alpha
        alphas: Array of alpha values

    Returns:
        (barrier_height, alpha_at_max)
    """
    endpoint_max = max(losses[0], losses[-1])
    max_loss = np.max(losses)
    max_alpha = alphas[np.argmax(losses)]

    barrier = max_loss - endpoint_max

    return float(barrier), float(max_alpha)


def compute_accuracy_barrier(
    accuracies: np.ndarray,
    alphas: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute the accuracy barrier (drop) along an interpolation path.

    Barrier = min(Acc(θ_0), Acc(θ_1)) - min_α Acc(θ_α)

    A positive value indicates accuracy drops along the path.

    Args:
        accuracies: Array of accuracies at each alpha
        alphas: Array of alpha values

    Returns:
        (barrier_height, alpha_at_min)
    """
    endpoint_min = min(accuracies[0], accuracies[-1])
    min_acc = np.min(accuracies)
    min_alpha = alphas[np.argmin(accuracies)]

    barrier = endpoint_min - min_acc

    return float(barrier), float(min_alpha)


def summarize_interpolation_results(
    results: Dict[str, Dict[str, np.ndarray]],
) -> Dict:
    """
    Summarize interpolation results with barrier computations.

    Args:
        results: Output from evaluate_interpolation_multi_dataset

    Returns:
        Summary dictionary with barriers and key metrics
    """
    summary = {}

    for name, data in results.items():
        if "losses" not in data:
            continue

        alphas = data["alphas"]
        losses = data["losses"]
        accs = data["accuracies"]

        loss_barrier, loss_max_alpha = compute_loss_barrier(losses, alphas)
        acc_barrier, acc_min_alpha = compute_accuracy_barrier(accs, alphas)

        summary[name] = {
            "loss_barrier": loss_barrier,
            "loss_barrier_alpha": loss_max_alpha,
            "acc_barrier": acc_barrier,
            "acc_barrier_alpha": acc_min_alpha,
            "endpoint_0_loss": float(losses[0]),
            "endpoint_1_loss": float(losses[-1]),
            "endpoint_0_acc": float(accs[0]),
            "endpoint_1_acc": float(accs[-1]),
            "max_loss": float(np.max(losses)),
            "min_acc": float(np.min(accs)),
        }

    return summary


def compare_pre_post_rebasin(
    model_a: ConvNet,
    model_b: ConvNet,
    model_b_rebased: ConvNet,
    dataloaders: Dict[str, DataLoader],
    device: torch.device,
    num_alphas: int = 21,
) -> Dict:
    """
    Compare interpolation results before and after rebasing.

    Args:
        model_a: Reference model
        model_b: Original model B
        model_b_rebased: Model B after rebasing to A
        dataloaders: Dictionary of dataloaders
        device: Torch device
        num_alphas: Number of interpolation points

    Returns:
        Dictionary with 'pre_rebasin' and 'post_rebasin' results
    """
    print("Evaluating pre-rebasin interpolation...")
    pre_results = evaluate_interpolation_multi_dataset(
        model_a, model_b, dataloaders, device, num_alphas
    )

    print("Evaluating post-rebasin interpolation...")
    post_results = evaluate_interpolation_multi_dataset(
        model_a, model_b_rebased, dataloaders, device, num_alphas
    )

    return {
        "pre_rebasin": {
            "raw": pre_results,
            "summary": summarize_interpolation_results(pre_results),
        },
        "post_rebasin": {
            "raw": post_results,
            "summary": summarize_interpolation_results(post_results),
        },
    }
