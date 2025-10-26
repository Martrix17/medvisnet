"""Trainer class for model fitting and testing with BaseTrainer."""

from typing import Dict, List

import torch
from matplotlib.figure import Figure
from torch import optim

from src.data.dataloader import CovidRadiographyDataModule
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import MLFlowLogger

from .base_trainer import BaseTrainer
from .callbacks import EarlyStopping
from .metrics import MetricsManager


class Trainer:
    """Handles model training, validation, and testing with mixed precision."""

    def __init__(
        self,
        base_trainer: BaseTrainer,
        epochs: int,
        data_module: CovidRadiographyDataModule,
        scheduler: optim.lr_scheduler._LRScheduler | None = None,
        metrics_manager: MetricsManager | None = None,
        logger: MLFlowLogger | None = None,
        early_stopping: EarlyStopping | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        val_every_n_epochs: int = 1,
        compute_metrics_n_val_epoch: int = 1,
    ) -> None:
        """
        Args:
            base_trainer: BaseTrainer instance for training/validation/testing.
            epochs: Number of total training epochs.
            data_module: Provides train/val/test dataloaders.
            scheduler: Learning rate scheduler (optional).
            metrics: MetricsManager instance (optional).
            logger: MLFlowLogger instance (optional).
            early_stopping: EarlyStopping callback (optional).
            checkpoint: CheckpointManager for saving/loading models (optional).
            resume_training: Resume training from a saved checkpoint.
            val_every_n_epochs: Run validation every N epochs.
            compute_metrics_n_val_epoch: Compute validation metrics every
                N validation epochs.
        """
        self.base_trainer = base_trainer
        self.epochs = epochs
        self.data_module = data_module
        self.scheduler = scheduler
        self.metrics_manager = metrics_manager
        self.logger = logger
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager

        self.val_every_n_epochs = val_every_n_epochs if val_every_n_epochs > 0 else None
        self.compute_metrics_n_val_epoch = (
            compute_metrics_n_val_epoch
            if self.val_every_n_epochs and compute_metrics_n_val_epoch > 0
            else None
        )

    def fit(self, load_checkpoint: bool = False, resume_training: bool = False) -> None:
        """Main training/validation loop."""
        start_epoch = 0
        if self.checkpoint_manager and load_checkpoint:
            start_epoch = self.checkpoint_manager.load(
                model=self.base_trainer.model,
                optimizer=self.base_trainer.optimizer,
                scaler=self.base_trainer.scaler,
                scheduler=self.scheduler,
                resume_training=resume_training,
            )

        for epoch in range(start_epoch, self.epochs):
            train_loss = self.base_trainer.train(
                loader=self.data_module.train_dataloader(),
                epoch=epoch,
                total_epochs=self.epochs,
            )

            val_loss = float("inf")
            run_validation = (
                self.val_every_n_epochs and (epoch + 1) % self.val_every_n_epochs == 0
            )
            if run_validation:
                compute_metrics = (
                    self.compute_metrics_n_val_epoch
                    and (epoch + 1) % self.compute_metrics_n_val_epoch == 0
                    and self.logger
                )

                val_result = self.base_trainer.evaluate(
                    loader=self.data_module.val_dataloader(),
                    epoch=epoch,
                    total_epochs=self.epochs,
                    compute_loss=True,
                    return_preds=compute_metrics,
                )
                val_loss = val_result["loss"]

                if compute_metrics and self.metrics_manager:
                    self.metrics_manager.reset()
                    val_metrics = self.metrics_manager.compute(
                        preds=val_result["predictions"], targets=val_result["targets"]
                    )
                    if self.logger:
                        self.logger.log_metrics(metrics=val_metrics, step=epoch + 1)

                if self.checkpoint_manager:
                    self.checkpoint_manager.save_if_improved(
                        model=self.base_trainer.model,
                        optimizer=self.base_trainer.optimizer,
                        scaler=self.base_trainer.scaler,
                        scheduler=self.scheduler,
                        val_loss=val_loss,
                        epoch=epoch,
                    )

            if self.logger:
                losses = {
                    "train_loss": train_loss,
                    "lr": self.base_trainer.optimizer.param_groups[0]["lr"],
                }
                if val_loss is not None:
                    losses["val_loss"] = val_loss
                self.logger.log_metrics(metrics=losses, step=epoch + 1)

            if self.scheduler:
                self.scheduler.step(val_loss)

            if self.early_stopping and self.early_stopping(val_loss):
                break

    def test(
        self, load_checkpoint: bool = False, plot_metrics: bool = True
    ) -> Dict[str, torch.Tensor | Figure]:
        """Evaluate model on test set."""
        if self.checkpoint_manager and load_checkpoint:
            self.checkpoint_manager.load(
                model=self.base_trainer.model,
                optimizer=self.base_trainer.optimizer,
                scaler=self.base_trainer.scaler,
                scheduler=self.scheduler,
                resume_training=False,
            )

        output = self.base_trainer.evaluate(
            loader=self.data_module.test_dataloader(),
            epoch=0,
            total_epochs=1,
            compute_loss=False,
            return_preds=True,
        )

        if not self.metrics_manager:
            print("No metrics manager provided; returning targets and predictions.")
            return output

        label_names = self.data_module.get_label_names()
        self.metrics_manager.set_mode(test_mode=True)
        metrics = self.metrics_manager.compute(
            preds=output["predictions"],
            targets=output["targets"],
            label_names=label_names,
        )
        output["metrics"] = metrics

        if self.logger:
            self.logger.log_text(
                text=metrics["report"], file_path="classification_report.txt"
            )

        if plot_metrics and self.metrics_manager:
            figures = self._visualize_metrics(
                preds=output["predictions"],
                targets=output["targets"],
                label_names=self.data_module.get_label_names(),
            )
            output["figures"] = figures

            if self.logger:
                self.logger.log_figure(fig=figures["auroc"], file_path="auroc.png")
                self.logger.log_figure(
                    fig=figures["roc_curve"], file_path="roc_curve.png"
                )
                self.logger.log_figure(
                    fig=figures["confmat"], file_path="confusion_matrix.png"
                )

        return output

    def _visualize_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor, label_names: List[str]
    ) -> Dict[str, Figure]:
        """Visualize test metrics"""
        assert self.metrics_manager is not None
        figures = {}

        auroc_metric = self.metrics_manager.metrics["auroc"]
        auroc_metric.reset()
        auroc_metric.update(preds, targets)
        fig_auroc, ax_auroc = auroc_metric.plot()
        ax_auroc.set_title("AUROC Curves")
        figures["auroc"] = fig_auroc

        roc_metric = self.metrics_manager.metrics["roc_curve"]
        roc_metric.reset()
        roc_metric.update(preds, targets)
        fig_roc, ax_roc = roc_metric.plot(score=True, labels=label_names)
        ax_roc.set_title("ROC Curves")
        figures["roc_curve"] = fig_roc

        confmat_metric = self.metrics_manager.metrics["confmat"]
        confmat_metric.reset()
        confmat_metric.update(torch.argmax(preds, dim=1), targets)
        fig_cm, ax_cm = confmat_metric.plot(labels=label_names)
        ax_cm.set_title("Confusion Matrix")
        figures["confmat"] = fig_cm

        return figures
