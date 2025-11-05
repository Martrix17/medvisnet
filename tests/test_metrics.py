"""Unit tests for MetricsManager class."""

import torch

from src.training.metrics import MetricsManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_fake_batch(num_classes=3, batch_size=8):
    preds = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    return preds.to(DEVICE), targets.to(DEVICE)


def test_metrics_compute_train():
    preds, targets = make_fake_batch()
    manager = MetricsManager(num_classes=3, device=DEVICE, test_mode=False)
    result = manager.compute(preds, targets)

    for key in [
        "Validation metrics/accuracy",
        "Validation metrics/precision",
        "Validation metrics/recall",
        "Validation metrics/f1",
        "Validation metrics/auroc",
    ]:
        assert key in result

    for v in result.values():
        assert isinstance(v, (float, torch.Tensor))


def test_metrics_compute_test_mode_includes_report():
    preds, targets = make_fake_batch()
    manager = MetricsManager(num_classes=3, device=DEVICE, test_mode=True)
    result = manager.compute(preds, targets, label_names=["A", "B", "C"])

    assert "report" in result
    assert "roc_curve" in result
    assert "confmat" in result


def test_set_mode_switch_resets_metrics():
    manager = MetricsManager(num_classes=3, device=DEVICE, test_mode=False)
    acc_before = list(manager.metrics.keys())
    manager.set_mode(True)
    acc_after = list(manager.metrics.keys())

    assert acc_before != acc_after
    assert "roc_curve" in acc_after
    assert "accuracy" not in acc_after
