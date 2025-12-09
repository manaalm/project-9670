"""
Centered Kernel Alignment (CKA) for measuring representation similarity.

This module provides:
- Linear CKA computation between two sets of representations
- Activation capture utilities for extracting layer outputs
- Layerwise CKA computation between two models

Reference:
    Kornblith et al., "Similarity of Neural Network Representations Revisited" (2019)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from .config import CONFIG, set_seed


def linear_cka(
    X: np.ndarray,
    Y: np.ndarray,
    debiased: bool = False,
) -> float:
    """
    Compute Linear Centered Kernel Alignment between two representation matrices.

    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    For linear kernels:
    HSIC(X, Y) = ||Y^T X||_F^2 / (n-1)^2  (centered version)

    Args:
        X: First representation matrix (n_samples, n_features_x)
        Y: Second representation matrix (n_samples, n_features_y)
        debiased: If True, use debiased HSIC estimator (recommended for small n)

    Returns:
        CKA similarity score in [0, 1]
    """
    n = X.shape[0]
    assert Y.shape[0] == n, "X and Y must have same number of samples"

    # Center the representations
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    if debiased:
        # Debiased HSIC estimator (Song et al., 2012)
        # More accurate for small sample sizes
        XX = X @ X.T
        YY = Y @ Y.T

        # Zero out diagonal
        np.fill_diagonal(XX, 0)
        np.fill_diagonal(YY, 0)

        # Compute HSIC
        hsic_xy = np.sum(XX * YY) / (n * (n - 3))
        hsic_xx = np.sum(XX * XX) / (n * (n - 3))
        hsic_yy = np.sum(YY * YY) / (n * (n - 3))

        # Add correction terms
        sum_XX = np.sum(XX)
        sum_YY = np.sum(YY)

        hsic_xy -= 2 * np.sum(XX, axis=0) @ np.sum(YY, axis=1) / (n * (n - 2) * (n - 3))
        hsic_xy += sum_XX * sum_YY / (n * (n - 1) * (n - 2) * (n - 3))

        hsic_xx -= 2 * np.sum(XX, axis=0) @ np.sum(XX, axis=1) / (n * (n - 2) * (n - 3))
        hsic_xx += sum_XX * sum_XX / (n * (n - 1) * (n - 2) * (n - 3))

        hsic_yy -= 2 * np.sum(YY, axis=0) @ np.sum(YY, axis=1) / (n * (n - 2) * (n - 3))
        hsic_yy += sum_YY * sum_YY / (n * (n - 1) * (n - 2) * (n - 3))
    else:
        # Standard HSIC estimator
        # HSIC = tr(K_X H K_Y H) / (n-1)^2
        # For linear kernel: K_X = X X^T, simplified to ||Y^T X||_F^2
        XtX = X.T @ X  # (d_x, d_x)
        YtY = Y.T @ Y  # (d_y, d_y)
        XtY = X.T @ Y  # (d_x, d_y)

        hsic_xy = np.sum(XtY ** 2)
        hsic_xx = np.sum(XtX ** 2)
        hsic_yy = np.sum(YtY ** 2)

    # Compute CKA
    denominator = np.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-12:
        return 0.0

    cka = hsic_xy / denominator
    return float(np.clip(cka, 0.0, 1.0))


class ActivationCapture:
    """
    Utility class for capturing intermediate activations from a model.

    Usage:
        capture = ActivationCapture(model, ['block2', 'block3', 'fc1'])
        activations = capture.get_activations(dataloader, device, n_samples=2000)
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
    ):
        """
        Args:
            model: The neural network model
            layer_names: List of layer names to capture (e.g., ['block2', 'block3', 'fc1'])
        """
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []

    def _get_hook(self, name: str):
        """Create a forward hook that stores activations."""
        def hook(module, input, output):
            # Flatten spatial dimensions if present
            if len(output.shape) == 4:  # (batch, channels, h, w)
                out = output.view(output.size(0), -1)
            else:
                out = output
            self.activations[name].append(out.detach().cpu())
        return hook

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        self.hooks = []

        for name in self.layer_names:
            if hasattr(self.model, name):
                layer = getattr(self.model, name)
                hook = layer.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)
            else:
                # Try to find layer in nested structure
                found = False
                for module_name, module in self.model.named_modules():
                    if module_name == name:
                        hook = module.register_forward_hook(self._get_hook(name))
                        self.hooks.append(hook)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Layer '{name}' not found in model")

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(
        self,
        dataloader: DataLoader,
        device: torch.device,
        n_samples: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract activations for the specified layers.

        Args:
            dataloader: DataLoader providing input data
            device: Torch device
            n_samples: Maximum number of samples to process (None = all)

        Returns:
            Dictionary mapping layer names to activation matrices (n_samples, n_features)
        """
        self.model.eval()
        self.activations = {name: [] for name in self.layer_names}

        self._register_hooks()

        try:
            samples_processed = 0
            with torch.no_grad():
                for images, _ in dataloader:
                    images = images.to(device)

                    # Forward pass (hooks will capture activations)
                    _ = self.model(images)

                    samples_processed += images.size(0)
                    if n_samples is not None and samples_processed >= n_samples:
                        break
        finally:
            self._remove_hooks()

        # Concatenate and convert to numpy
        result = {}
        for name in self.layer_names:
            if self.activations[name]:
                concatenated = torch.cat(self.activations[name], dim=0)
                if n_samples is not None:
                    concatenated = concatenated[:n_samples]
                result[name] = concatenated.numpy()

        return result


def compute_layerwise_cka(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    layer_names: Optional[List[str]] = None,
    n_samples: int = 2000,
    debiased: bool = False,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Compute CKA similarity between two models at multiple layers.

    Args:
        model_a: First model
        model_b: Second model
        dataloader: DataLoader providing input data
        device: Torch device
        layer_names: List of layer names to compare (default: ['block2', 'block3', 'fc1'])
        n_samples: Number of samples to use for CKA computation
        debiased: Use debiased HSIC estimator
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping layer names to CKA scores
    """
    set_seed(seed)

    if layer_names is None:
        layer_names = ['block2', 'block3', 'fc1']

    # Create activation captures
    capture_a = ActivationCapture(model_a, layer_names)
    capture_b = ActivationCapture(model_b, layer_names)

    # Get activations
    acts_a = capture_a.get_activations(dataloader, device, n_samples)
    acts_b = capture_b.get_activations(dataloader, device, n_samples)

    # Compute CKA for each layer
    cka_scores = {}
    for name in layer_names:
        if name in acts_a and name in acts_b:
            cka = linear_cka(acts_a[name], acts_b[name], debiased=debiased)
            cka_scores[name] = cka

    return cka_scores


def compute_cka_distance(
    model_a: nn.Module,
    model_b: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    layer_names: Optional[List[str]] = None,
    n_samples: int = 2000,
    debiased: bool = False,
    seed: int = 42,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute CKA-based representation distance between two models.

    Distance is defined as: 1 - mean(CKA) across specified layers.

    Args:
        model_a: First model
        model_b: Second model
        dataloader: DataLoader providing input data
        device: Torch device
        layer_names: List of layer names to compare
        n_samples: Number of samples for CKA computation
        debiased: Use debiased HSIC estimator
        seed: Random seed

    Returns:
        Tuple of (distance, per_layer_cka_scores)
    """
    cka_scores = compute_layerwise_cka(
        model_a, model_b, dataloader, device,
        layer_names=layer_names,
        n_samples=n_samples,
        debiased=debiased,
        seed=seed,
    )

    if not cka_scores:
        return 1.0, {}

    mean_cka = np.mean(list(cka_scores.values()))
    distance = 1.0 - mean_cka

    return float(distance), cka_scores


def compute_singular_vector_alignment(
    model_a: nn.Module,
    model_b: nn.Module,
    layer_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute singular vector alignment distance between two models.

    For each specified layer, computes the alignment between top-k right singular
    vectors of the weight matrices. This is inspired by the observation that
    weight matching aligns dominant singular vector directions.

    Args:
        model_a: First model
        model_b: Second model
        layer_names: Layer names to compare (default: conv layers)
        top_k: Number of top singular vectors to compare

    Returns:
        Tuple of (distance, per_layer_alignment_scores)
    """
    if layer_names is None:
        layer_names = ['block0', 'block1', 'block2', 'block3']

    alignments = {}

    for name in layer_names:
        # Get weight matrices
        try:
            if hasattr(model_a, name):
                layer_a = getattr(model_a, name)
                layer_b = getattr(model_b, name)

                # For ConvBlock, get the conv weight
                if hasattr(layer_a, 'conv'):
                    W_a = layer_a.conv.weight.data.cpu().numpy()
                    W_b = layer_b.conv.weight.data.cpu().numpy()
                else:
                    W_a = layer_a.weight.data.cpu().numpy()
                    W_b = layer_b.weight.data.cpu().numpy()
            else:
                continue

            # Reshape to 2D if needed (for conv layers)
            if len(W_a.shape) > 2:
                W_a = W_a.reshape(W_a.shape[0], -1)
                W_b = W_b.reshape(W_b.shape[0], -1)

            # Compute SVD
            U_a, S_a, Vt_a = np.linalg.svd(W_a, full_matrices=False)
            U_b, S_b, Vt_b = np.linalg.svd(W_b, full_matrices=False)

            # Get top-k right singular vectors
            k = min(top_k, Vt_a.shape[0], Vt_b.shape[0])
            V_a = Vt_a[:k]  # (k, d)
            V_b = Vt_b[:k]  # (k, d)

            # Compute alignment as mean absolute cosine similarity
            # Between corresponding singular vectors
            alignment_scores = []
            for i in range(k):
                # Cosine similarity
                cos_sim = np.abs(np.dot(V_a[i], V_b[i]))
                alignment_scores.append(cos_sim)

            alignments[name] = float(np.mean(alignment_scores))

        except Exception as e:
            # Skip layers that can't be processed
            continue

    if not alignments:
        return 1.0, {}

    mean_alignment = np.mean(list(alignments.values()))
    distance = 1.0 - mean_alignment

    return float(distance), alignments


def create_cka_dataloader(
    base_dataset,
    n_samples: int = 2000,
    batch_size: int = 128,
    seed: int = 42,
) -> DataLoader:
    """
    Create a fixed subset dataloader for CKA computation.

    Args:
        base_dataset: Base dataset to sample from
        n_samples: Number of samples to include
        batch_size: Batch size for the loader
        seed: Random seed for reproducibility

    Returns:
        DataLoader with fixed subset
    """
    set_seed(seed)

    # Select fixed subset
    n_total = len(base_dataset)
    n_samples = min(n_samples, n_total)

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=n_samples, replace=False)

    subset = Subset(base_dataset, indices.tolist())

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True,
    )
