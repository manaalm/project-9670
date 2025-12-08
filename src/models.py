"""
Neural network architectures for the Git Re-Basin experiment.

This module provides a simple ConvNet architecture that is designed to be
easily permutable for weight matching (Git Re-Basin).

Key design choices:
- Clear layer structure with consistent naming
- No skip connections (simplifies permutation matching)
- BatchNorm after each conv (requires careful permutation handling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

from .config import CONFIG


class ConvBlock(nn.Module):
    """
    A convolutional block: Conv -> BatchNorm -> ReLU.

    This is the basic building block of our ConvNet.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # No bias when using BatchNorm
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ConvNet(nn.Module):
    """
    Simple ConvNet for CIFAR-10 classification.

    Architecture:
    - 4 ConvBlocks with increasing filter counts
    - MaxPooling after blocks 1, 2, 3
    - Global average pooling
    - FC hidden layer -> output

    This architecture is designed for easy Git Re-Basin:
    - No skip connections
    - Clear sequential structure
    - Consistent conv->bn->relu pattern
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_filters: List[int] = None,
        fc_hidden: int = 256,
        input_channels: int = 3,
    ):
        super().__init__()

        if num_filters is None:
            num_filters = CONFIG["model"]["num_filters"]

        self.num_filters = num_filters
        self.fc_hidden = fc_hidden
        self.num_classes = num_classes

        # Convolutional blocks
        # Block 0: 3 -> 32, 32x32 -> 16x16
        self.block0 = ConvBlock(input_channels, num_filters[0])
        self.pool0 = nn.MaxPool2d(2, 2)

        # Block 1: 32 -> 64, 16x16 -> 8x8
        self.block1 = ConvBlock(num_filters[0], num_filters[1])
        self.pool1 = nn.MaxPool2d(2, 2)

        # Block 2: 64 -> 128, 8x8 -> 4x4
        self.block2 = ConvBlock(num_filters[1], num_filters[2])
        self.pool2 = nn.MaxPool2d(2, 2)

        # Block 3: 128 -> 256, 4x4 -> 4x4 (no pooling)
        self.block3 = ConvBlock(num_filters[2], num_filters[3])

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters[3], fc_hidden)
        self.fc1_bn = nn.BatchNorm1d(fc_hidden)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional blocks
        x = self.pool0(self.block0(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)

        # Global average pooling and flatten
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Fully connected
        x = self.fc1_relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2(x)

        return x

    def get_features(self, x: torch.Tensor, layer: str = "fc1") -> torch.Tensor:
        """
        Extract intermediate features.

        Args:
            x: Input tensor
            layer: Which layer to extract ('block0', 'block1', ..., 'fc1')

        Returns:
            Feature tensor
        """
        x = self.pool0(self.block0(x))
        if layer == "block0":
            return x

        x = self.pool1(self.block1(x))
        if layer == "block1":
            return x

        x = self.pool2(self.block2(x))
        if layer == "block2":
            return x

        x = self.block3(x)
        if layer == "block3":
            return x

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc1_relu(self.fc1_bn(self.fc1(x)))
        if layer == "fc1":
            return x

        return self.fc2(x)

    def get_layer_names(self) -> List[str]:
        """Get names of permutable layers in order."""
        return ["block0", "block1", "block2", "block3", "fc1"]


def create_model(config: Optional[Dict] = None) -> ConvNet:
    """Create a ConvNet model from configuration."""
    cfg = config or CONFIG
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    return ConvNet(
        num_classes=data_cfg["num_classes"],
        num_filters=model_cfg["num_filters"],
        fc_hidden=model_cfg["fc_hidden"],
        input_channels=data_cfg["num_channels"],
    )


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_state_flat(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_model_state_flat(model: nn.Module, flat_params: torch.Tensor):
    """Set model parameters from a flattened vector."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
        offset += numel


def clone_model(model: nn.Module) -> nn.Module:
    """Create a deep copy of a model."""
    import copy
    return copy.deepcopy(model)


def get_layer_info(model: ConvNet) -> Dict[str, Dict]:
    """
    Get information about each layer for permutation matching.

    Returns a dictionary mapping layer names to their shapes and types.
    """
    info = OrderedDict()

    # Conv blocks
    for i in range(4):
        block = getattr(model, f"block{i}")
        info[f"block{i}.conv"] = {
            "type": "conv",
            "weight_shape": block.conv.weight.shape,
            "in_channels": block.conv.in_channels,
            "out_channels": block.conv.out_channels,
        }
        info[f"block{i}.bn"] = {
            "type": "bn",
            "num_features": block.bn.num_features,
        }

    # FC layers
    info["fc1"] = {
        "type": "fc",
        "weight_shape": model.fc1.weight.shape,
        "in_features": model.fc1.in_features,
        "out_features": model.fc1.out_features,
    }
    info["fc1_bn"] = {
        "type": "bn1d",
        "num_features": model.fc1_bn.num_features,
    }
    info["fc2"] = {
        "type": "fc",
        "weight_shape": model.fc2.weight.shape,
        "in_features": model.fc2.in_features,
        "out_features": model.fc2.out_features,
    }

    return info


def model_agreement(
    model1: nn.Module,
    model2: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Compute prediction agreement between two models.

    Returns the fraction of samples where both models predict the same class.
    """
    model1.eval()
    model2.eval()

    agree_count = 0
    total_count = 0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)

            preds1 = model1(images).argmax(dim=1)
            preds2 = model2(images).argmax(dim=1)

            agree_count += (preds1 == preds2).sum().item()
            total_count += len(images)

    return agree_count / total_count
