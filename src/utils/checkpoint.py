"""Checkpoint management for saving and loading model states."""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler


class CheckpointManager:
    """Handles saving and loading model and optimizer checkpoints."""

    def __init__(
        self,
        device: str,
        save_dir: str = "experiments/checkpoints",
        filename: str = "best_model.pt",
        patience: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            save_dir (str): Directory to save checkpoints.
            filename (str): Filename for the checkpoint.
            device (str): Device to map the loaded checkpoint to ('cuda', 'cpu').
            patience (int): number of epochs to wait between saving checkpoints
                            if validation loss is improving.
            verbose (bool): Whether to print status messages.
        """
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.patience = patience
        self.verbose = verbose

        self.best_loss = float("inf")
        self.counter = 0
        self.start_epoch = 0

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        scaler: GradScaler = None,
        scheduler: optim.lr_scheduler.LRScheduler = None,
        val_loss: float = float("inf"),
    ) -> None:
        """Save current model state."""
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scaler_state": scaler.state_dict() if scaler else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
        }

        path = os.path.join(self.save_dir, self.filename)
        torch.save(checkpoint, path)

    def save_if_improved(self, epoch, model, optimizer, scaler, scheduler, val_loss):
        """Save model if validation loss improves and patience interval passed."""
        if val_loss < self.best_loss:
            if epoch - self.counter >= self.patience:
                if self.verbose:
                    print(
                        f"💾 Validation improved on epoch {epoch} "
                        f"({self.best_loss:.4f} → {val_loss:.4f}). Saving checkpoint."
                    )
                self.best_loss = val_loss
                self.save(epoch, model, optimizer, scaler, scheduler, val_loss)
                self.counter = epoch

    def load(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        scaler: GradScaler = None,
        scheduler: optim.lr_scheduler.LRScheduler = None,
        resume_training: bool = False,
    ) -> int:
        """Load model and optional optimizer/scheduler states."""
        path = os.path.join(self.save_dir, self.filename)
        if not os.path.exists(path):
            print("⚠️ No checkpoint found. Starting fresh.")
            return 0

        checkpoint = torch.load(path, map_location=self.device)

        if "model_state" not in checkpoint:
            raise ValueError("Invalid checkpoint: missing model_state.")

        model.load_state_dict(checkpoint["model_state"])

        if optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scaler and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if scheduler and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

        print(f"✅ Loaded checkpoint from {path}.")

        if resume_training:
            self.start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"✅ Resuming from epoch {self.start_epoch}.")
            return self.start_epoch

        return 0
