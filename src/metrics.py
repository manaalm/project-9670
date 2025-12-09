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


# =============================================================================
# Mechanism Distance Metrics
# =============================================================================

def get_srs_scalar(srs_results: Dict[str, float]) -> float:
    """
    Extract the scalar Spurious Reliance Score from a results dictionary.

    The SRS is computed as:
        SRS = lambda_ood * OOD_drop + lambda_cf * CF_accuracy_drop + lambda_flip * flip_rate

    Default weights (lambda values):
        lambda_ood = 0.4
        lambda_cf = 0.3
        lambda_flip = 0.3

    Args:
        srs_results: Dictionary from compute_spurious_reliance_score()

    Returns:
        Scalar SRS value
    """
    return srs_results['spurious_reliance_score']


def compute_srs_distance(
    srs_a: Dict[str, float],
    srs_b: Dict[str, float],
) -> float:
    """
    Compute the cue-reliance distance between two models based on their SRS.

    dist_srs = |SRS(A) - SRS(B)|

    Args:
        srs_a: SRS results for model A
        srs_b: SRS results for model B

    Returns:
        Absolute difference in SRS
    """
    srs_scalar_a = get_srs_scalar(srs_a)
    srs_scalar_b = get_srs_scalar(srs_b)
    return abs(srs_scalar_a - srs_scalar_b)


def compute_srs_for_model(
    model: nn.Module,
    id_loader: DataLoader,
    ood_loader: DataLoader,
    cf_dataset: 'CounterfactualPatchDataset',
    device: torch.device,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Convenience wrapper for compute_spurious_reliance_score.

    Args:
        model: Model to evaluate
        id_loader: In-distribution test loader
        ood_loader: Out-of-distribution test loader
        cf_dataset: Counterfactual patch dataset
        device: Torch device
        weights: Optional custom weights for SRS components

    Returns:
        Dictionary with SRS and component metrics
    """
    return compute_spurious_reliance_score(
        model, id_loader, ood_loader, cf_dataset, device, weights
    )


# =============================================================================
# Barrier Computation Helpers
# =============================================================================

def compute_barriers_from_interpolation(
    interp_results: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute loss and accuracy barriers from interpolation results.

    Args:
        interp_results: Results from evaluate_interpolation_multi_dataset()
            Expected keys: 'id', 'ood' each with 'alphas', 'losses', 'accuracies'

    Returns:
        Dictionary with barrier metrics for each dataset type:
        {
            'id': {'loss_barrier': ..., 'acc_barrier': ...},
            'ood': {'loss_barrier': ..., 'acc_barrier': ...}
        }
    """
    barriers = {}

    for dataset_name in ['id', 'ood']:
        if dataset_name not in interp_results:
            continue

        data = interp_results[dataset_name]
        alphas = data['alphas']
        losses = data['losses']
        accuracies = data['accuracies']

        # Compute loss barrier
        endpoint_max_loss = max(losses[0], losses[-1])
        max_loss = np.max(losses)
        loss_barrier = max_loss - endpoint_max_loss

        # Compute accuracy barrier
        endpoint_min_acc = min(accuracies[0], accuracies[-1])
        min_acc = np.min(accuracies)
        acc_barrier = endpoint_min_acc - min_acc

        barriers[dataset_name] = {
            'loss_barrier': float(loss_barrier),
            'acc_barrier': float(acc_barrier),
            'max_loss': float(max_loss),
            'min_acc': float(min_acc),
            'endpoint_loss_0': float(losses[0]),
            'endpoint_loss_1': float(losses[-1]),
            'endpoint_acc_0': float(accuracies[0]),
            'endpoint_acc_1': float(accuracies[-1]),
        }

    return barriers


def compute_all_barriers(
    pre_results: Dict[str, Dict[str, np.ndarray]],
    post_results: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> Dict[str, float]:
    """
    Compute all barrier metrics for a model pair (pre and post rebasin).

    Args:
        pre_results: Pre-rebasin interpolation results
        post_results: Post-rebasin interpolation results (optional)

    Returns:
        Flat dictionary with all barrier metrics:
        {
            'barrier_id_raw': ...,
            'barrier_ood_raw': ...,
            'barrier_id_rebasin': ...,
            'barrier_ood_rebasin': ...,
        }
    """
    output = {}

    # Pre-rebasin barriers
    pre_barriers = compute_barriers_from_interpolation(pre_results)
    output['barrier_id_raw'] = pre_barriers.get('id', {}).get('loss_barrier', np.nan)
    output['barrier_ood_raw'] = pre_barriers.get('ood', {}).get('loss_barrier', np.nan)
    output['barrier_id_acc_raw'] = pre_barriers.get('id', {}).get('acc_barrier', np.nan)
    output['barrier_ood_acc_raw'] = pre_barriers.get('ood', {}).get('acc_barrier', np.nan)

    # Post-rebasin barriers
    if post_results is not None:
        post_barriers = compute_barriers_from_interpolation(post_results)
        output['barrier_id_rebasin'] = post_barriers.get('id', {}).get('loss_barrier', np.nan)
        output['barrier_ood_rebasin'] = post_barriers.get('ood', {}).get('loss_barrier', np.nan)
        output['barrier_id_acc_rebasin'] = post_barriers.get('id', {}).get('acc_barrier', np.nan)
        output['barrier_ood_acc_rebasin'] = post_barriers.get('ood', {}).get('acc_barrier', np.nan)
    else:
        output['barrier_id_rebasin'] = np.nan
        output['barrier_ood_rebasin'] = np.nan
        output['barrier_id_acc_rebasin'] = np.nan
        output['barrier_ood_acc_rebasin'] = np.nan

    return output


# =============================================================================
# Statistical Analysis Utilities
# =============================================================================

def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    method: str = 'pearson',
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute correlation with bootstrapped confidence intervals.

    Args:
        x: First variable
        y: Second variable
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        method: 'pearson' or 'spearman'
        seed: Random seed

    Returns:
        Dictionary with:
        - 'correlation': Point estimate
        - 'ci_lower': Lower bound of CI
        - 'ci_upper': Upper bound of CI
        - 'p_value': Approximate p-value
    """
    from scipy import stats

    np.random.seed(seed)
    n = len(x)

    # Point estimate
    if method == 'pearson':
        corr, pval = stats.pearsonr(x, y)
    else:  # spearman
        corr, pval = stats.spearmanr(x, y)

    # Bootstrap
    bootstrap_corrs = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n, size=n)
        x_boot = x[indices]
        y_boot = y[indices]

        if method == 'pearson':
            r, _ = stats.pearsonr(x_boot, y_boot)
        else:
            r, _ = stats.spearmanr(x_boot, y_boot)

        bootstrap_corrs.append(r)

    bootstrap_corrs = np.array(bootstrap_corrs)

    # Confidence interval
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_corrs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha / 2))

    return {
        'correlation': float(corr),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'p_value': float(pval),
        'method': method,
    }


def fit_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Fit a simple linear regression and return summary statistics.

    Args:
        X: Feature matrix (n_samples, n_features) or (n_samples,) for single feature
        y: Target variable (n_samples,)
        feature_names: Optional names for features

    Returns:
        Dictionary with regression results:
        - 'coefficients': Dict mapping feature names to coefficients
        - 'intercept': Intercept term
        - 'r_squared': R^2 score
        - 'predictions': Predicted values
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Default feature names
    if feature_names is None:
        feature_names = [f'x{i}' for i in range(X.shape[1])]

    # Fit model
    model = LinearRegression()
    model.fit(X, y)

    # Get predictions
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Build coefficients dict
    coefficients = {}
    for name, coef in zip(feature_names, model.coef_):
        coefficients[name] = float(coef)

    return {
        'coefficients': coefficients,
        'intercept': float(model.intercept_),
        'r_squared': float(r2),
        'predictions': y_pred,
    }
