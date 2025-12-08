"""
Metrics for quantifying spurious feature reliance.

This module provides:
- OOD accuracy drop computation
- Counterfactual patch sensitivity
- Spurious Reliance Score (SRS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm

from .config import CONFIG
from .data import SpuriousPatchDataset, CounterfactualPatchDataset


def compute_ood_drop(
    id_accuracy: float,
    ood_accuracy: float,
) -> float:
    """
    Compute OOD accuracy drop.

    OOD Drop = Acc(ID) - Acc(OOD)

    A high positive value indicates strong reliance on spurious features.

    Args:
        id_accuracy: In-distribution accuracy
        ood_accuracy: Out-of-distribution accuracy

    Returns:
        OOD drop value
    """
    return id_accuracy - ood_accuracy


def compute_patch_sensitivity(
    model: nn.Module,
    counterfactual_dataset: CounterfactualPatchDataset,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 4,
) -> Dict[str, float]:
    """
    Compute counterfactual patch sensitivity metrics.

    For each sample, compares predictions with original vs swapped patch:
    - Accuracy change: How much accuracy drops when patch is swapped
    - Logit change: Mean change in true-class logit when patch is swapped

    Args:
        model: Model to evaluate
        counterfactual_dataset: Dataset providing (original, label, counterfactual) tuples
        device: Torch device
        batch_size: Batch size for evaluation
        num_workers: Number of data loading workers

    Returns:
        Dictionary with sensitivity metrics
    """
    model.eval()

    loader = DataLoader(
        counterfactual_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Metrics accumulators
    orig_correct = 0
    cf_correct = 0
    logit_changes = []
    prediction_flips = 0
    total = 0

    with torch.no_grad():
        for orig_img, label, cf_img in tqdm(loader, desc="Patch sensitivity"):
            orig_img = orig_img.to(device)
            cf_img = cf_img.to(device)
            label = label.to(device)

            # Get predictions
            orig_logits = model(orig_img)
            cf_logits = model(cf_img)

            orig_preds = orig_logits.argmax(dim=1)
            cf_preds = cf_logits.argmax(dim=1)

            # Accuracy metrics
            orig_correct += (orig_preds == label).sum().item()
            cf_correct += (cf_preds == label).sum().item()

            # Prediction flips (originally correct, now wrong)
            orig_is_correct = (orig_preds == label)
            cf_is_wrong = (cf_preds != label)
            prediction_flips += (orig_is_correct & cf_is_wrong).sum().item()

            # Logit change for true class
            batch_indices = torch.arange(len(label))
            orig_true_logits = orig_logits[batch_indices, label]
            cf_true_logits = cf_logits[batch_indices, label]
            logit_change = (orig_true_logits - cf_true_logits).cpu().numpy()
            logit_changes.extend(logit_change.tolist())

            total += len(label)

    orig_acc = orig_correct / total
    cf_acc = cf_correct / total
    flip_rate = prediction_flips / total
    mean_logit_change = np.mean(logit_changes)
    std_logit_change = np.std(logit_changes)

    return {
        "original_accuracy": orig_acc,
        "counterfactual_accuracy": cf_acc,
        "accuracy_drop": orig_acc - cf_acc,
        "prediction_flip_rate": flip_rate,
        "mean_logit_change": mean_logit_change,
        "std_logit_change": std_logit_change,
    }


def compute_spurious_reliance_score(
    model: nn.Module,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    counterfactual_dataset: CounterfactualPatchDataset,
    device: torch.device,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute the Spurious Reliance Score (SRS).

    SRS combines multiple metrics into a single score:
    SRS = w1 * OOD_drop + w2 * accuracy_drop_cf + w3 * flip_rate

    Where:
    - OOD_drop: Accuracy drop from ID to OOD (normalized)
    - accuracy_drop_cf: Accuracy drop when patch is swapped
    - flip_rate: Rate of prediction flips on counterfactuals

    Default weights: w1=0.4, w2=0.3, w3=0.3

    Higher SRS indicates stronger reliance on spurious (patch) features.

    Args:
        model: Model to evaluate
        id_loader: In-distribution test loader
        ood_loader: Out-of-distribution test loader
        counterfactual_dataset: Dataset for counterfactual evaluation
        device: Torch device
        weights: Optional custom weights for the components

    Returns:
        Dictionary with SRS and component metrics
    """
    if weights is None:
        weights = {"ood_drop": 0.4, "acc_drop_cf": 0.3, "flip_rate": 0.3}

    model.eval()
    criterion = nn.CrossEntropyLoss()

    # Compute ID and OOD accuracy
    id_loss, id_acc = _evaluate_loader(model, id_loader, device, criterion)
    ood_loss, ood_acc = _evaluate_loader(model, ood_loader, device, criterion)

    # Compute counterfactual sensitivity
    cf_metrics = compute_patch_sensitivity(model, counterfactual_dataset, device)

    # Compute OOD drop
    ood_drop = compute_ood_drop(id_acc, ood_acc)

    # Compute SRS (all components are in [0, 1] range approximately)
    srs = (
        weights["ood_drop"] * ood_drop +
        weights["acc_drop_cf"] * cf_metrics["accuracy_drop"] +
        weights["flip_rate"] * cf_metrics["prediction_flip_rate"]
    )

    return {
        "spurious_reliance_score": srs,
        "id_accuracy": id_acc,
        "ood_accuracy": ood_acc,
        "ood_drop": ood_drop,
        "cf_accuracy_drop": cf_metrics["accuracy_drop"],
        "cf_flip_rate": cf_metrics["prediction_flip_rate"],
        "cf_mean_logit_change": cf_metrics["mean_logit_change"],
        "id_loss": id_loss,
        "ood_loss": ood_loss,
    }


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Helper to evaluate a model on a dataloader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def compute_class_wise_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[int, float]:
    """
    Compute per-class accuracy.

    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Torch device
        num_classes: Number of classes

    Returns:
        Dictionary mapping class index to accuracy
    """
    model.eval()

    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    return {
        cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        for cls in range(num_classes)
    }


def compute_confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Torch device
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array (true x predicted)
    """
    model.eval()
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                conf_matrix[t, p] += 1

    return conf_matrix


def compare_model_metrics(
    models: Dict[str, nn.Module],
    id_loader: DataLoader,
    ood_loader: DataLoader,
    counterfactual_datasets: Dict[str, CounterfactualPatchDataset],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Compare spurious reliance metrics across multiple models.

    Args:
        models: Dictionary mapping model names to models
        id_loader: In-distribution test loader
        ood_loader: Out-of-distribution test loader
        counterfactual_datasets: Dictionary mapping model names to CF datasets
        device: Torch device

    Returns:
        Dictionary mapping model names to their metrics
    """
    results = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        cf_dataset = counterfactual_datasets.get(name, list(counterfactual_datasets.values())[0])

        metrics = compute_spurious_reliance_score(
            model, id_loader, ood_loader, cf_dataset, device
        )
        results[name] = metrics

    return results


def semantic_barrier_metric(
    srs_values: List[float],
    alphas: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute semantic barrier from SRS along interpolation path.

    Semantic barrier = max variation in SRS along the path

    Args:
        srs_values: SRS values at each interpolation point
        alphas: Interpolation alpha values

    Returns:
        (max_variation, alpha_at_max_variation)
    """
    srs_array = np.array(srs_values)

    # Compute variation from endpoints
    endpoint_avg = (srs_array[0] + srs_array[-1]) / 2
    variations = np.abs(srs_array - endpoint_avg)

    max_var = np.max(variations)
    max_var_alpha = alphas[np.argmax(variations)]

    return float(max_var), float(max_var_alpha)
