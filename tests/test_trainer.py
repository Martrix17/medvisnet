"""Smoke test for Trainer class."""

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_trainer import BaseTrainer
from src.training.metrics import MetricsManager
from src.training.trainer import Trainer

DEVICE = "cpu"


@pytest.fixture
def tiny_data_module():
    """Minimal data module with random tensors."""

    class TinyDataModule:
        def __init__(self, batch_size=2):
            self.batch_size = batch_size
            self.num_classes = 4

            self.X_train = torch.randn(8, 3, 32, 32)
            self.y_train = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])

            self.X_val = torch.randn(4, 3, 32, 32)
            self.y_val = torch.tensor([0, 1, 2, 3])

            self.X_test = torch.randn(4, 3, 32, 32)
            self.y_test = torch.tensor([0, 1, 2, 3])

        def train_dataloader(self):
            dataset = TensorDataset(self.X_train, self.y_train)
            return DataLoader(dataset, batch_size=self.batch_size)

        def val_dataloader(self):
            dataset = TensorDataset(self.X_val, self.y_val)
            return DataLoader(dataset, batch_size=self.batch_size)

        def test_dataloader(self):
            dataset = TensorDataset(self.X_test, self.y_test)
            return DataLoader(dataset, batch_size=self.batch_size)

        def get_label_names(self):
            return ["A", "B", "C", "D"]

        def get_num_classes(self):
            return 4

    return TinyDataModule()


def test_smoke_trainer_loop(tiny_data_module):
    """Minimal end-to-end training and testing smoke test."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 4))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    base_trainer = BaseTrainer(
        device=DEVICE, model=model, criterion=criterion, optimizer=optimizer
    )
    metrics_manager = MetricsManager(num_classes=4, device=DEVICE)

    trainer = Trainer(
        base_trainer=base_trainer,
        epochs=2,
        data_module=tiny_data_module,
        metrics_manager=metrics_manager,
        val_every_n_epochs=1,
        compute_metrics_n_val_epoch=1,
    )

    trainer.fit()

    output = trainer.test(plot_metrics=False)

    assert "predictions" in output
    assert "targets" in output
    assert output["predictions"].shape[0] == output["targets"].shape[0]
    assert "auroc" in output["metrics"]
