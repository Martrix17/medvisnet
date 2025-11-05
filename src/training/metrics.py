"""
Metrics management for multi-class classification metrics.

Example:
    >>> metrics = MetricsManager(num_classes=4, device="cuda")
    >>> results = metrics.compute(preds, targets)
    >>> print(results["Validation metrics/accuracy"])

    >>> metrics.set_mode(test_mode=True)
    >>> test_results = metrics.compute(
    ...     preds, targets, label_names=["COVID", "Normal", ...]
    ... )
"""

from typing import Dict, List

import torch
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassROC,
)


class MetricsManager:
    """
    Manages metric computation for multi-class classification.

    Supports two modes:
    - Training/validation: accuracy, precision, recall, F1, AUROC
    - Test: AUROC, ROC curves, confusion matrix, classification report
    """

    def __init__(self, num_classes: int, device: str, test_mode: bool = False) -> None:
        """
        Args:
            num_classes: Number of classes in the classification task.
            device: Device for metric computations ('cuda' or 'cpu').
            test_mode: Initialize in test mode if True (default: validation mode).
        """
        self.device = device
        self.num_classes = num_classes
        self.test_mode = test_mode
        self._init_metrics()

    def set_mode(self, test_mode: bool) -> None:
        """Switches between training/validation and test mode metrics sets."""
        if self.test_mode != test_mode:
            self.test_mode = test_mode
            self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize metrics based on current mode."""
        if self.test_mode:
            self.metrics = {
                "auroc": MulticlassAUROC(num_classes=self.num_classes).to(self.device),
                "roc_curve": MulticlassROC(num_classes=self.num_classes).to(
                    self.device
                ),
                "confmat": MulticlassConfusionMatrix(num_classes=self.num_classes).to(
                    self.device
                ),
            }
        else:
            self.metrics = {
                "accuracy": Accuracy(
                    task="multiclass", num_classes=self.num_classes
                ).to(self.device),
                "precision": Precision(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ).to(self.device),
                "recall": Recall(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ).to(self.device),
                "f1": F1Score(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ).to(self.device),
                "auroc": MulticlassAUROC(num_classes=self.num_classes).to(self.device),
            }

    def compute(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        label_names: List[str] | None = None,
    ) -> Dict[str, str | float | torch.Tensor]:
        """
        Compute metrics for current mode.

        Args:
            preds: Model predictions [N, num_classes].
            targets: Ground truth labels [N].
            label_names: Class names for classification report (test mode only).

        Returns:
            Dict with metric names as keys and computed values. In validation mode,
            keys are prefixed with "Validation metrics/". Test mode includes a
            "report" key with the classification report string.
        """
        preds_class = torch.argmax(preds, dim=1)

        results = {}
        for name, metric in self.metrics.items():
            metric.reset()
            if name in ["auroc", "roc_curve"]:
                metric.update(preds, targets)
            else:
                metric.update(preds_class, targets)

            value = metric.compute()

            name = f"Validation metrics/{name}" if not self.test_mode else name
            results[name] = (
                value.item()
                if isinstance(value, torch.Tensor) and value.numel() == 1
                else value
            )

        if self.test_mode:
            results["report"] = classification_report(
                y_true=targets.cpu().numpy(),
                y_pred=preds_class.cpu().numpy(),
                target_names=label_names,
                zero_division=0,
            )

        return results

    def reset(self) -> None:
        """Reset internal state for a new epoch or phase."""
        for metric in self.metrics.values():
            metric.reset()
