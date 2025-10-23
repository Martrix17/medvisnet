"""Metrics management for multi-class classification."""

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
    Metrics computation and aggregation for multi-class classification.
    """

    def __init__(self, num_classes: int, device: str, test_mode: bool = False) -> None:
        """
        Args:
            num_classes (int): Number of classes in the classification task.
            device (str): Device to perform metric computations on ('cuda', 'cpu').
        """
        self.device = device
        self.num_classes = num_classes
        self.test_mode = test_mode
        self._init_metrics()

    def set_mode(self, test_mode: bool) -> None:
        """Switch metrics between training/validation and test mode."""
        if self.test_mode != test_mode:
            self.test_mode = test_mode
            self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize metrics with the given number of classes."""
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
        """Compute all metrics."""
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
