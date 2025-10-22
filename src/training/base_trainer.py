"""BaseTrainer class for model training, evaluation with mixed precision."""

from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class BaseTrainer:
    """Handles model training, validation, and testing with mixed precision."""

    def __init__(
        self,
        device: str,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> None:
        """
        Args:
            device: Device to perform training/evaluation on ('cuda', 'cpu').
            model: Neural network to train.
            criterion: Loss function.
            optimizer: Optimizer instance.
        """
        self.device = device
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = GradScaler(device=device)

    def train(self, loader: DataLoader, epoch: int, total_epochs: int) -> float:
        """Runs one epoch of training."""
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
    ) -> Dict:
        """Runs one epoch of evaluation."""
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
