"""
Data loading and spurious feature injection for CIFAR-10.

This module provides:
- CIFAR-10 dataset wrapper with spurious patch injection
- Environment generators (Env A: aligned, Env B: flipped)
- OOD test set generators (no patch or flipped)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
from typing import Optional, Tuple, List, Dict
from PIL import Image

from .config import CONFIG, DATA_DIR


class SpuriousPatchDataset(Dataset):
    """
    CIFAR-10 dataset with spurious colored patch injection.

    The patch color can be:
    - Aligned with the label (spurious correlation)
    - Flipped (anti-correlated)
    - Random
    - No patch (clean)
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
        patch_mode: str = "aligned",  # 'aligned', 'flipped', 'random', 'none'
        p_align: float = 0.95,  # Probability of alignment when mode is 'aligned'
        config: Optional[Dict] = None,
    ):
        """
        Args:
            root: Root directory for CIFAR-10 data
            train: If True, use training set; else test set
            transform: Torchvision transforms to apply
            download: If True, download dataset if not present
            patch_mode: How to assign patch colors
            p_align: Probability that patch aligns with true label
            config: Configuration dictionary (uses global CONFIG if None)
        """
        self.config = config or CONFIG
        self.cifar = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=None  # We'll apply transforms after patching
        )
        self.transform = transform
        self.patch_mode = patch_mode
        self.p_align = p_align

        # Patch parameters
        patch_cfg = self.config["patch"]
        self.patch_size = patch_cfg["size"]
        self.class_colors = patch_cfg["class_colors"]
        self.num_classes = self.config["data"]["num_classes"]

        # Pre-compute patch assignments for reproducibility
        self._precompute_patches()

    def _precompute_patches(self):
        """Pre-compute which patch color each sample gets."""
        n_samples = len(self.cifar)
        self.patch_colors = []

        rng = np.random.RandomState(42)  # Fixed seed for reproducibility

        for idx in range(n_samples):
            _, label = self.cifar[idx]

            if self.patch_mode == "none":
                self.patch_colors.append(None)
            elif self.patch_mode == "aligned":
                # With probability p_align, use true label's color
                if rng.random() < self.p_align:
                    self.patch_colors.append(self.class_colors[label])
                else:
                    # Use a random different class's color
                    other_labels = [l for l in range(self.num_classes) if l != label]
                    wrong_label = rng.choice(other_labels)
                    self.patch_colors.append(self.class_colors[wrong_label])
            elif self.patch_mode == "flipped":
                # With probability p_align, use WRONG label's color
                if rng.random() < self.p_align:
                    other_labels = [l for l in range(self.num_classes) if l != label]
                    wrong_label = rng.choice(other_labels)
                    self.patch_colors.append(self.class_colors[wrong_label])
                else:
                    self.patch_colors.append(self.class_colors[label])
            elif self.patch_mode == "random":
                # Uniformly random color
                random_label = rng.randint(0, self.num_classes)
                self.patch_colors.append(self.class_colors[random_label])
            else:
                raise ValueError(f"Unknown patch_mode: {self.patch_mode}")

    def _add_patch(self, img: Image.Image, color: Tuple[int, int, int]) -> Image.Image:
        """Add a colored patch to the top-left corner of the image."""
        img_array = np.array(img)
        # Add patch to top-left corner
        img_array[:self.patch_size, :self.patch_size, :] = color
        return Image.fromarray(img_array)

    def __len__(self) -> int:
        return len(self.cifar)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.cifar[idx]

        # Add patch if applicable
        patch_color = self.patch_colors[idx]
        if patch_color is not None:
            img = self._add_patch(img, patch_color)

        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def get_patch_color(self, idx: int) -> Optional[Tuple[int, int, int]]:
        """Get the patch color for a specific sample."""
        return self.patch_colors[idx]

    def get_alignment_rate(self) -> float:
        """Compute actual alignment rate in the dataset."""
        if self.patch_mode == "none":
            return 0.0

        aligned_count = 0
        for idx in range(len(self)):
            _, label = self.cifar[idx]
            patch_color = self.patch_colors[idx]
            if patch_color == self.class_colors[label]:
                aligned_count += 1

        return aligned_count / len(self)


class CounterfactualPatchDataset(Dataset):
    """
    Dataset that provides counterfactual versions of samples with swapped patches.

    For each sample, generates a version where the patch color is changed
    to a different (incorrect) class color.
    """

    def __init__(
        self,
        base_dataset: SpuriousPatchDataset,
        swap_mode: str = "random_wrong",  # 'random_wrong' or 'specific'
        target_class: Optional[int] = None,  # For 'specific' mode
    ):
        """
        Args:
            base_dataset: The original SpuriousPatchDataset
            swap_mode: How to choose the counterfactual patch color
            target_class: If swap_mode='specific', use this class's color
        """
        self.base = base_dataset
        self.swap_mode = swap_mode
        self.target_class = target_class
        self.rng = np.random.RandomState(123)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        Returns:
            original_img: Original image with original patch
            label: True label
            counterfactual_img: Image with swapped patch color
        """
        # Get original
        original_img, label = self.base[idx]

        # Get base image without transform
        base_img, _ = self.base.cifar[idx]

        # Determine counterfactual patch color
        if self.swap_mode == "random_wrong":
            other_labels = [l for l in range(self.base.num_classes) if l != label]
            cf_label = self.rng.choice(other_labels)
            cf_color = self.base.class_colors[cf_label]
        elif self.swap_mode == "specific":
            cf_color = self.base.class_colors[self.target_class]
        else:
            raise ValueError(f"Unknown swap_mode: {self.swap_mode}")

        # Create counterfactual image
        cf_img = self.base._add_patch(base_img, cf_color)
        if self.base.transform is not None:
            cf_img = self.base.transform(cf_img)

        return original_img, label, cf_img


def get_transforms(train: bool = True, config: Optional[Dict] = None) -> transforms.Compose:
    """Get standard CIFAR-10 transforms."""
    cfg = config or CONFIG
    mean = cfg["data"]["mean"]
    std = cfg["data"]["std"]

    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def create_env_a_dataset(
    train: bool = True,
    config: Optional[Dict] = None,
) -> SpuriousPatchDataset:
    """
    Create Environment A dataset (spurious-aligned).

    Patch color matches true label with probability p_align (default 0.95).
    """
    cfg = config or CONFIG
    return SpuriousPatchDataset(
        root=str(DATA_DIR),
        train=train,
        transform=get_transforms(train, cfg),
        download=True,
        patch_mode="aligned",
        p_align=cfg["patch"]["p_align_env_a"],
        config=cfg,
    )


def create_env_b_dataset(
    train: bool = True,
    config: Optional[Dict] = None,
) -> SpuriousPatchDataset:
    """
    Create Environment B dataset (spurious-flipped).

    Patch color matches true label with probability 1 - p_align (default 0.05).
    """
    cfg = config or CONFIG
    return SpuriousPatchDataset(
        root=str(DATA_DIR),
        train=train,
        transform=get_transforms(train, cfg),
        download=True,
        patch_mode="aligned",
        p_align=cfg["patch"]["p_align_env_b"],  # Low alignment = flipped
        config=cfg,
    )


def create_no_patch_dataset(
    train: bool = True,
    config: Optional[Dict] = None,
) -> SpuriousPatchDataset:
    """Create dataset without any patches (clean CIFAR-10)."""
    cfg = config or CONFIG
    return SpuriousPatchDataset(
        root=str(DATA_DIR),
        train=train,
        transform=get_transforms(train, cfg),
        download=True,
        patch_mode="none",
        config=cfg,
    )


def create_mixed_env_dataset(
    env_a_fraction: float = 0.5,
    train: bool = True,
    config: Optional[Dict] = None,
) -> ConcatDataset:
    """
    Create a mixed dataset with samples from both Env A and Env B.

    This is used for training "robust" models that see both environments.
    """
    cfg = config or CONFIG

    # Create full datasets
    env_a = create_env_a_dataset(train=train, config=cfg)
    env_b = create_env_b_dataset(train=train, config=cfg)

    # Calculate subset sizes
    total_size = len(env_a)
    env_a_size = int(total_size * env_a_fraction)
    env_b_size = total_size - env_a_size

    # Create random subsets
    rng = np.random.RandomState(42)
    env_a_indices = rng.choice(len(env_a), size=env_a_size, replace=False)
    env_b_indices = rng.choice(len(env_b), size=env_b_size, replace=False)

    env_a_subset = Subset(env_a, env_a_indices.tolist())
    env_b_subset = Subset(env_b, env_b_indices.tolist())

    return ConcatDataset([env_a_subset, env_b_subset])


def get_dataloaders(
    train_dataset: Dataset,
    test_id_dataset: Dataset,
    test_ood_dataset: Dataset,
    config: Optional[Dict] = None,
) -> Dict[str, DataLoader]:
    """Create DataLoaders for training and evaluation."""
    cfg = config or CONFIG
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test_id": DataLoader(
            test_id_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test_ood": DataLoader(
            test_ood_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }


def create_standard_dataloaders(
    env_type: str = "spurious",  # 'spurious' or 'robust'
    ood_type: str = "no_patch",  # 'env_b' or 'no_patch'
    config: Optional[Dict] = None,
) -> Dict[str, DataLoader]:
    """
    Create standard train/test dataloaders for the experiment.

    Args:
        env_type: 'spurious' trains on Env A only, 'robust' trains on mixed
        ood_type: Type of OOD test set
        config: Configuration dictionary

    Returns:
        Dictionary with 'train', 'test_id', 'test_ood' DataLoaders
    """
    cfg = config or CONFIG

    # Training data
    if env_type == "spurious":
        train_dataset = create_env_a_dataset(train=True, config=cfg)
    else:  # robust
        robust_cfg = cfg["robust_training"]
        train_dataset = create_mixed_env_dataset(
            env_a_fraction=robust_cfg["env_a_fraction"],
            train=True,
            config=cfg,
        )

    # Test data (ID = Env A distribution)
    test_id_dataset = create_env_a_dataset(train=False, config=cfg)

    # Test data (OOD)
    if ood_type == "no_patch":
        test_ood_dataset = create_no_patch_dataset(train=False, config=cfg)
    else:  # env_b
        test_ood_dataset = create_env_b_dataset(train=False, config=cfg)

    return get_dataloaders(train_dataset, test_id_dataset, test_ood_dataset, cfg)


def denormalize(img_tensor: torch.Tensor, config: Optional[Dict] = None) -> np.ndarray:
    """
    Denormalize an image tensor back to [0, 255] range.

    Args:
        img_tensor: Normalized image tensor (C, H, W)
        config: Configuration dictionary

    Returns:
        Numpy array (H, W, C) in uint8 format
    """
    cfg = config or CONFIG
    mean = np.array(cfg["data"]["mean"])
    std = np.array(cfg["data"]["std"])

    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def get_sample_batch(
    dataset: Dataset,
    n_samples: int = 16,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of samples from a dataset."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)

    images = []
    labels = []
    for idx in indices:
        img, label = dataset[idx]
        images.append(img)
        labels.append(label)

    return torch.stack(images), torch.tensor(labels)
