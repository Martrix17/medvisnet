"""Unit tests for BaseTrainer class."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_trainer import BaseTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)
        self.weight = torch.nn.init.uniform_(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


def make_fake_dataloader(batch_size=8, num_batches=5, num_classes=3):
    """Create small random dataset for quick training tests."""
    x = torch.randn(num_batches * batch_size, 10)
    y = torch.randint(0, num_classes, (num_batches * batch_size,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size)


def test_train_step_runs_and_updates_weights():
    model = DummyModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    trainer = BaseTrainer(
        device=DEVICE, model=model, criterion=criterion, optimizer=optimizer
    )
    loader = make_fake_dataloader()
    initial_weight = model.weight.detach().clone()
    loss = trainer.train(loader, epoch=0, total_epochs=1)

    assert isinstance(loss, float)
    assert loss > 0
    assert not torch.equal(model.weight.detach(), initial_weight)


@torch.no_grad()
def test_evaluate_returns_loss_and_predictions():
    model = DummyModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    trainer = BaseTrainer(DEVICE, model, criterion, optimizer)
    loader = make_fake_dataloader()
    result = trainer.evaluate(
        loader, epoch=0, total_epochs=1, compute_loss=True, return_preds=True
    )

    assert "loss" in result
    assert "predictions" in result
    assert "targets" in result
    assert result["predictions"].shape[0] == result["targets"].shape[0]


def test_evaluate_no_gradients():
    """Ensure evaluate does not update weights or gradients."""
    model = DummyModel().to(DEVICE)
    trainer = BaseTrainer(DEVICE, model, criterion=None, optimizer=None)
    loader = make_fake_dataloader()
    initial_weight = model.weight.detach().clone()
    trainer.evaluate(loader, epoch=0, total_epochs=1, compute_loss=False)

    assert torch.equal(model.weight.detach(), initial_weight)
    assert all(p.grad is None for p in model.parameters())
