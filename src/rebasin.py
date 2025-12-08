"""
Git Re-Basin implementation for weight matching.

This module implements permutation-based model alignment (Git Re-Basin)
using weight matching to find optimal neuron permutations that align
two independently trained models.

Reference: Ainsworth et al., "Git Re-Basin: Merging Models modulo Permutation Symmetries"

Key functions:
- weight_matching: Find optimal permutations to align model B to model A
- apply_permutation: Apply found permutations to a model
- rebasin: Full rebasing pipeline
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import copy

from .models import ConvNet, clone_model


def compute_weight_matching_cost(
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    prev_perm: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the cost matrix for weight matching between two layers.

    For conv layers: weight shape is (out_channels, in_channels, H, W)
    For fc layers: weight shape is (out_features, in_features)

    Args:
        weight_a: Weights from reference model
        weight_b: Weights from model to align
        prev_perm: Permutation applied to previous layer (for input channel ordering)

    Returns:
        Cost matrix of shape (n, n) where n is the output dimension
    """
    # Flatten to (out_dim, -1)
    wa = weight_a.view(weight_a.shape[0], -1)
    wb = weight_b.view(weight_b.shape[0], -1)

    # If previous permutation exists, apply it to input channels
    if prev_perm is not None:
        # Reshape to separate out_channels and in_channels
        if len(weight_b.shape) == 4:  # Conv layer
            out_c, in_c, h, w = weight_b.shape
            wb_reshaped = weight_b.clone()
            # Permute input channels
            wb_reshaped = wb_reshaped[:, prev_perm, :, :]
            wb = wb_reshaped.view(out_c, -1)
        elif len(weight_b.shape) == 2:  # FC layer
            out_f, in_f = weight_b.shape
            wb_reshaped = weight_b.clone()
            wb_reshaped = wb_reshaped[:, prev_perm]
            wb = wb_reshaped

    # Compute cost matrix: negative inner product (we want to maximize similarity)
    # Normalize for numerical stability
    wa_norm = wa / (wa.norm(dim=1, keepdim=True) + 1e-8)
    wb_norm = wb / (wb.norm(dim=1, keepdim=True) + 1e-8)

    # Cost = -similarity (we minimize cost to maximize similarity)
    cost = -torch.mm(wa_norm, wb_norm.t())

    return cost.cpu().numpy()


def solve_linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    """
    Solve the linear assignment problem to find optimal permutation.

    Args:
        cost_matrix: Cost matrix of shape (n, n)

    Returns:
        Permutation as array of indices
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind


def weight_matching(
    model_a: ConvNet,
    model_b: ConvNet,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal permutations to align model_b to model_a using weight matching.

    This implements the core Git Re-Basin algorithm:
    1. For each layer (in order), compute cost matrix based on weight similarity
    2. Solve linear assignment to find best permutation
    3. Propagate permutation info to next layer

    Args:
        model_a: Reference model (we align B to A)
        model_b: Model to align
        device: Torch device

    Returns:
        Dictionary mapping layer names to permutation tensors
    """
    model_a.eval()
    model_b.eval()

    permutations = OrderedDict()

    # Get state dicts
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    # Track previous permutation for propagation
    prev_perm = None

    # Layer order for our ConvNet architecture
    layers = [
        ("block0.conv.weight", "block0.bn"),
        ("block1.conv.weight", "block1.bn"),
        ("block2.conv.weight", "block2.bn"),
        ("block3.conv.weight", "block3.bn"),
        ("fc1.weight", "fc1_bn"),
    ]

    for layer_name, bn_name in layers:
        weight_a = state_a[layer_name].to(device)
        weight_b = state_b[layer_name].to(device)

        # Compute cost matrix
        cost = compute_weight_matching_cost(weight_a, weight_b, prev_perm)

        # Solve assignment
        perm = solve_linear_assignment(cost)
        perm_tensor = torch.tensor(perm, dtype=torch.long, device=device)

        # Store permutation
        permutations[layer_name] = perm_tensor

        # Update prev_perm for next layer
        prev_perm = perm_tensor

    return permutations


def apply_permutation_to_conv(
    conv_weight: torch.Tensor,
    conv_bn_weight: torch.Tensor,
    conv_bn_bias: torch.Tensor,
    conv_bn_mean: torch.Tensor,
    conv_bn_var: torch.Tensor,
    out_perm: torch.Tensor,
    in_perm: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Apply permutation to a conv layer and its batch norm.

    Args:
        conv_weight: Conv weight (out_c, in_c, H, W)
        conv_bn_*: BatchNorm parameters
        out_perm: Permutation for output channels
        in_perm: Permutation for input channels (from previous layer)

    Returns:
        Tuple of permuted tensors
    """
    # Permute output channels
    conv_weight = conv_weight[out_perm]
    conv_bn_weight = conv_bn_weight[out_perm]
    conv_bn_bias = conv_bn_bias[out_perm]
    conv_bn_mean = conv_bn_mean[out_perm]
    conv_bn_var = conv_bn_var[out_perm]

    # Permute input channels (if not first layer)
    if in_perm is not None:
        conv_weight = conv_weight[:, in_perm, :, :]

    return conv_weight, conv_bn_weight, conv_bn_bias, conv_bn_mean, conv_bn_var


def apply_permutation_to_fc(
    fc_weight: torch.Tensor,
    fc_bias: Optional[torch.Tensor],
    bn_weight: Optional[torch.Tensor],
    bn_bias: Optional[torch.Tensor],
    bn_mean: Optional[torch.Tensor],
    bn_var: Optional[torch.Tensor],
    out_perm: Optional[torch.Tensor],
    in_perm: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    """
    Apply permutation to a fully connected layer and its optional batch norm.

    Args:
        fc_weight: FC weight (out_f, in_f)
        fc_bias: FC bias (may be None)
        bn_*: BatchNorm parameters (may be None)
        out_perm: Permutation for output features
        in_perm: Permutation for input features

    Returns:
        Tuple of permuted tensors
    """
    # Permute output features
    if out_perm is not None:
        fc_weight = fc_weight[out_perm]
        if fc_bias is not None:
            fc_bias = fc_bias[out_perm]
        if bn_weight is not None:
            bn_weight = bn_weight[out_perm]
            bn_bias = bn_bias[out_perm]
            bn_mean = bn_mean[out_perm]
            bn_var = bn_var[out_perm]

    # Permute input features
    if in_perm is not None:
        fc_weight = fc_weight[:, in_perm]

    return fc_weight, fc_bias, bn_weight, bn_bias, bn_mean, bn_var


def apply_permutations(
    model: ConvNet,
    permutations: Dict[str, torch.Tensor],
    device: torch.device,
) -> ConvNet:
    """
    Apply permutations to a model to align it with a reference.

    Args:
        model: Model to permute
        permutations: Dictionary of permutations from weight_matching
        device: Torch device

    Returns:
        New model with permuted weights
    """
    # Clone the model
    aligned_model = clone_model(model)
    aligned_model = aligned_model.to(device)

    state = aligned_model.state_dict()

    # Get permutations in order
    perm_keys = list(permutations.keys())

    # Apply permutations layer by layer
    prev_perm = None

    for i, key in enumerate(perm_keys):
        perm = permutations[key]

        if "block" in key:
            # Conv block
            block_idx = int(key.split(".")[0].replace("block", ""))

            # Get current tensors
            conv_w = state[f"block{block_idx}.conv.weight"]
            bn_w = state[f"block{block_idx}.bn.weight"]
            bn_b = state[f"block{block_idx}.bn.bias"]
            bn_m = state[f"block{block_idx}.bn.running_mean"]
            bn_v = state[f"block{block_idx}.bn.running_var"]

            # Apply permutation
            conv_w, bn_w, bn_b, bn_m, bn_v = apply_permutation_to_conv(
                conv_w, bn_w, bn_b, bn_m, bn_v,
                out_perm=perm,
                in_perm=prev_perm if block_idx > 0 else None,
            )

            # Update state
            state[f"block{block_idx}.conv.weight"] = conv_w
            state[f"block{block_idx}.bn.weight"] = bn_w
            state[f"block{block_idx}.bn.bias"] = bn_b
            state[f"block{block_idx}.bn.running_mean"] = bn_m
            state[f"block{block_idx}.bn.running_var"] = bn_v

        elif "fc1" in key:
            # FC1 layer (has batch norm)
            fc_w = state["fc1.weight"]
            bn_w = state["fc1_bn.weight"]
            bn_b = state["fc1_bn.bias"]
            bn_m = state["fc1_bn.running_mean"]
            bn_v = state["fc1_bn.running_var"]

            # FC1 input comes from flattened conv output
            # prev_perm is for block3 output channels
            fc_w, _, bn_w, bn_b, bn_m, bn_v = apply_permutation_to_fc(
                fc_w, None, bn_w, bn_b, bn_m, bn_v,
                out_perm=perm,
                in_perm=prev_perm,
            )

            state["fc1.weight"] = fc_w
            state["fc1_bn.weight"] = bn_w
            state["fc1_bn.bias"] = bn_b
            state["fc1_bn.running_mean"] = bn_m
            state["fc1_bn.running_var"] = bn_v

            # Also need to permute fc2's input
            fc2_w = state["fc2.weight"]
            fc2_b = state["fc2.bias"]
            fc2_w = fc2_w[:, perm]
            state["fc2.weight"] = fc2_w
            # fc2 bias doesn't change (output dimension stays same)

        prev_perm = perm

    # Load permuted state
    aligned_model.load_state_dict(state)

    return aligned_model


def rebasin(
    model_ref: ConvNet,
    model_to_align: ConvNet,
    device: torch.device,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
) -> ConvNet:
    """
    Full Git Re-Basin pipeline: align model_to_align to model_ref.

    This is the main entry point for model alignment.

    Args:
        model_ref: Reference model (we align TO this)
        model_to_align: Model to be aligned
        device: Torch device
        dataloader: Optional dataloader (not used in weight matching, but kept for API)

    Returns:
        Aligned version of model_to_align
    """
    # Find optimal permutations
    permutations = weight_matching(model_ref, model_to_align, device)

    # Apply permutations
    aligned_model = apply_permutations(model_to_align, permutations, device)

    return aligned_model


def compute_weight_distance(
    model_a: nn.Module,
    model_b: nn.Module,
) -> float:
    """
    Compute L2 distance between model weights.

    Args:
        model_a: First model
        model_b: Second model

    Returns:
        L2 distance between flattened weight vectors
    """
    params_a = torch.cat([p.data.view(-1) for p in model_a.parameters()])
    params_b = torch.cat([p.data.view(-1) for p in model_b.parameters()])

    return (params_a - params_b).norm().item()


def compute_cosine_similarity(
    model_a: nn.Module,
    model_b: nn.Module,
) -> float:
    """
    Compute cosine similarity between model weights.

    Args:
        model_a: First model
        model_b: Second model

    Returns:
        Cosine similarity between flattened weight vectors
    """
    params_a = torch.cat([p.data.view(-1) for p in model_a.parameters()])
    params_b = torch.cat([p.data.view(-1) for p in model_b.parameters()])

    return torch.nn.functional.cosine_similarity(
        params_a.unsqueeze(0),
        params_b.unsqueeze(0)
    ).item()
