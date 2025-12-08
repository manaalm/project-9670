"""
Training and evaluation utilities for the Git Re-Basin experiment.

This module provides:
- Training loop with logging
- Evaluation functions
- Learning rate scheduling
- Checkpoint saving/loading
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Callable
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

from .config import CONFIG, CHECKPOINTS_DIR


class Trainer:
    """
    Trainer class for model training with logging and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Optional[Dict] = None,
    ):
        self.model = model
        self.device = device
        self.config = config or CONFIG

        # Training setup
        train_cfg = self.config["training"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            momentum=train_cfg["momentum"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg["num_epochs"],
        )

        # Logging
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "id_acc": [],
            "ood_acc": [],
            "lr": [],
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        verbose: bool = True,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (train_loss, train_accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not verbose)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss / total:.4f}",
                "acc": f"{100. * correct / total:.2f}%",
            })

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def evaluate(
        self,
        dataloader: DataLoader,
        desc: str = "Eval",
        verbose: bool = False,
    ) -> Tuple[float, float]:
        """
        Evaluate model on a dataset.

        Returns:
            (loss, accuracy)
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=desc, disable=not verbose)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        test_id_loader: DataLoader,
        test_ood_loader: DataLoader,
        num_epochs: Optional[int] = None,
        verbose: bool = True,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict:
        """
        Full training loop.

        Returns:
            Training history dictionary
        """
        if num_epochs is None:
            num_epochs = self.config["training"]["num_epochs"]

        best_id_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch, verbose)

            # Evaluate
            _, id_acc = self.evaluate(test_id_loader, "ID Test")
            _, ood_acc = self.evaluate(test_ood_loader, "OOD Test")

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Log
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["id_acc"].append(id_acc)
            self.history["ood_acc"].append(ood_acc)
            self.history["lr"].append(current_lr)

            if verbose:
                print(f"Epoch {epoch}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {100*train_acc:.2f}% | "
                      f"ID Acc: {100*id_acc:.2f}% | "
                      f"OOD Acc: {100*ood_acc:.2f}% | "
                      f"LR: {current_lr:.6f}")

            # Step scheduler
            self.scheduler.step()

            # Save best model
            if id_acc > best_id_acc and checkpoint_path is not None:
                best_id_acc = id_acc
                self.save_checkpoint(checkpoint_path)

        # Save final model if no best checkpoint was saved
        if checkpoint_path is not None and best_id_acc == 0.0:
            self.save_checkpoint(checkpoint_path)

        return self.history

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "config": self.config,
        }, path)

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """
    Evaluate a model on a dataset.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        device: Device to run on
        criterion: Loss function (default: CrossEntropyLoss)

    Returns:
        (loss, accuracy)
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get all predictions from a model.

    Returns:
        (predictions, true_labels, logits)
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_logits.append(outputs.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_logits),
    )


def load_model(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Load a model from checkpoint.

    Args:
        model: Model architecture (uninitialized or initialized)
        checkpoint_path: Path to checkpoint file
        device: Device to load to

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def save_model(model: nn.Module, path: Path):
    """Save just the model state dict."""
    torch.save({"model_state_dict": model.state_dict()}, path)


def save_training_history(history: Dict, path: Path):
    """Save training history to JSON."""
    # Convert numpy arrays to lists for JSON serialization
    history_json = {}
    for key, value in history.items():
        if isinstance(value, np.ndarray):
            history_json[key] = value.tolist()
        elif isinstance(value, list):
            history_json[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
        else:
            history_json[key] = value

    with open(path, 'w') as f:
        json.dump(history_json, f, indent=2)


def load_training_history(path: Path) -> Dict:
    """Load training history from JSON."""
    with open(path, 'r') as f:
        return json.load(f)
