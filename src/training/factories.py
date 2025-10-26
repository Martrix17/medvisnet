"""Factory functions for training/evaluation setup with hydra configs."""

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataloader import CovidRadiographyDataModule
from src.models.vit import VisionTransformer
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import MLFlowLogger

from .callbacks import EarlyStopping
from .metrics import MetricsManager


def create_data_module(cfg) -> CovidRadiographyDataModule:
    data_module = CovidRadiographyDataModule(**cfg)
    data_module.setup()
    return data_module


def create_vit_model(cfg, num_classes: int) -> VisionTransformer:
    return VisionTransformer(
        model_name=cfg.model_name,
        num_classes=num_classes,
        dropout=cfg.dropout,
        weights=cfg.weights,
        freeze_backbone=cfg.freeze_backbone,
    )


def create_optimizer(cfg, model: nn.Module) -> optim.Optimizer:
    return getattr(optim, cfg.name)(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def create_criterion(cfg, weights: torch.Tensor) -> nn.Module:
    return getattr(nn, cfg.name)(weight=weights)


def create_scheduler(cfg, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
    params = {k: v for k, v in cfg.items() if k != "name"}
    return getattr(optim.lr_scheduler, cfg.name)(optimizer=optimizer, **params)


def create_early_stopping(cfg) -> EarlyStopping:
    return EarlyStopping(**cfg)


def create_metrics_manager(num_classes: int, device: str) -> MetricsManager:
    return MetricsManager(device=device, num_classes=num_classes, test_mode=False)


def create_checkpoint_manager(cfg, device: str) -> CheckpointManager:
    return CheckpointManager(
        device=device,
        save_dir=cfg.save_dir,
        filename=cfg.filename,
        patience=cfg.patience,
    )


def create_mlflow_logger(cfg) -> MLFlowLogger:
    return MLFlowLogger(
        run_name=cfg.run_name,
        uri=cfg.uri,
        experiment_name=cfg.experiment_name,
        log_system=cfg.log_system,
    )
