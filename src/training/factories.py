"""
Factory functions for creating training/evaluation components from Hydra configurations.
"""

from typing import Tuple, cast

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import CovidRadiographyDataModule
from src.models.vit import VisionTransformer
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import MLflowLogger

from .callbacks import EarlyStopping
from .metrics import MetricsManager


def create_data_module(cfg) -> CovidRadiographyDataModule:
    """Create and setup data module from config."""
    data_module = CovidRadiographyDataModule(
        data_dir=cfg.data_dir,
        augment=cfg.augment,
        image_size=cast(Tuple[int, int], tuple(cfg.image_size)),
        mean=cast(Tuple[float, float, float], tuple(cfg.mean)),
        std=cast(Tuple[float, float, float], tuple(cfg.std)),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        val_split=cfg.val_split,
        test_split=cfg.test_split,
        seed=cfg.seed,
    )
    data_module.setup()
    return data_module


def create_vit_model(cfg, num_classes: int) -> VisionTransformer:
    """Create Vision Transformer model from config."""
    return VisionTransformer(
        model_name=cfg.model_name,
        num_classes=num_classes,
        num_hidden_layers=cfg.num_hidden_layers,
        dropout=cfg.dropout,
        weights=cfg.get("weights", None),
        freeze_backbone=cfg.freeze_backbone,
    )


def create_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    """Create optimizer from config (e.g., Adam, SGD) via reflection."""
    return getattr(optim, cfg.name)(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def create_criterion(cfg, weights: torch.Tensor) -> nn.Module:
    """Create loss function from config with class weights."""
    return getattr(nn, cfg.name)(weight=weights)


def create_scheduler(cfg, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler from config via reflection."""
    params = {k: v for k, v in cfg.items() if k != "name"}
    return getattr(optim.lr_scheduler, cfg.name)(optimizer=optimizer, **params)


def create_early_stopping(cfg) -> EarlyStopping:
    """Create early stopping callback from config."""
    return EarlyStopping(**cfg)


def create_metrics_manager(num_classes: int, device: str) -> MetricsManager:
    """Create checkpoint manager from config."""
    return MetricsManager(device=device, num_classes=num_classes, test_mode=False)


def create_checkpoint_manager(cfg, device: str) -> CheckpointManager:
    """Create MLFlow logger from config."""
    return CheckpointManager(
        device=device,
        save_dir=cfg.save_dir,
        filename=cfg.filename,
        patience=cfg.patience,
    )


def create_mlflow_logger(cfg) -> MLflowLogger:
    return MLflowLogger(
        run_name=cfg.run_name,
        uri=cfg.uri,
        experiment_name=cfg.experiment_name,
        log_system=cfg.log_system,
    )
