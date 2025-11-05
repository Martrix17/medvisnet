"""
Base trainer for model training and evaluation.

Example:
    >>> trainer = BaseTrainer(
    ...     device="cuda",
    ...     model=model,
    ...     criterion=nn.CrossEntropyLoss(),
    ...     optimizer=optim.Adam(model.parameters())
    ... )
    >>> train_loss = trainer.train(train_loader, epoch=0, total_epochs=10)
    >>> val_results = trainer.evaluate(val_loader, epoch=0, total_epochs=10)
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer:
    """
    Handles training and evaluation loops with automatic mixed precision (AMP).

    Features:
    - Automatic mixed precision
    - Progress bars for loss tracking
    - Evaluation with optional loss computation and prediction collection
    """

    def __init__(
        self,
        device: str,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> None:
        """
        Args:
            device: Device for training ('cuda' or 'cpu').
            model: Neural network to train.
            criterion: Loss function (e.g., CrossEntropyLoss).
            optimizer: Optimizer instance (e.g., Adam, SGD).
        """
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = GradScaler(device=device)

    def train(self, loader: DataLoader, epoch: int, total_epochs: int) -> float:
        """
        Run one training epoch with mixed precision.

        Args:
            loader: Training data loader.
            epoch: Current epoch index (0-indexed).
            total_epochs: Total number of epochs.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"[Train] Epoch {epoch+1}/{total_epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with autocast(device_type=self.device):
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        compute_loss: bool = True,
        return_preds: bool = False,
    ) -> Dict[str, float | torch.Tensor]:
        """
        Run one evaluation epoch with optional loss and prediction collection.

        Args:
            loader: Validation/test data loader.
            epoch: Current epoch index (0-indexed).
            total_epochs: Total number of epochs.
            compute_loss: Compute and return average loss if True.
            return_preds: Collect and return predictions and targets if True.

        Returns:
            Dict containing:
            - 'loss' (float): Average loss if compute_loss=True
            - 'predictions' (Tensor): Model outputs [N, num_classes]
                if return_preds=True
            - 'targets' (Tensor): Ground truth labels [N] if return_preds=True
        """
        self.model.eval()
        total_loss = 0.0
        targets, preds = [], []
        pbar = tqdm(loader, desc=f"[Eval] Epoch {epoch+1}/{total_epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with autocast(device_type=self.device):
                outputs = self.model(imgs)

                if compute_loss:
                    loss = self.criterion(outputs, labels)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    total_loss += loss.item()

            if return_preds:
                targets.append(labels)
                preds.append(outputs)

        output = {}
        if compute_loss:
            output["loss"] = total_loss / len(loader)
        if return_preds:
            output["predictions"] = torch.cat(preds)
            output["targets"] = torch.cat(targets)

        return output
