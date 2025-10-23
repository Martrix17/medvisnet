"""Unit tests fro CheckpointManager."""

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.checkpoint import CheckpointManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)


def test_checkpoint_save_and_load(tmp_path):
    """Ensure checkpoint saves and loads model + optimizer correctly."""
    model = DummyModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler(device=DEVICE)
    ckpt = CheckpointManager(
        device=DEVICE, save_dir=tmp_path, patience=1, verbose=False
    )

    # Modify weights to confirm loading restores them
    original_weight = model.linear.weight.clone().detach()

    ckpt.save(epoch=3, model=model, optimizer=optimizer, scaler=scaler, val_loss=0.5)
    model.linear.weight.data += 10.0  # corrupt model weights

    ckpt.load(model, optimizer=optimizer, scaler=scaler)
    assert torch.allclose(model.linear.weight, original_weight, atol=1e-6)


def test_save_if_improved_patience_interval(tmp_path):
    """Checkpoint should only save when loss improves AND patience interval passes."""
    model = DummyModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler(device=DEVICE)
    ckpt = CheckpointManager(DEVICE, save_dir=tmp_path, patience=2, verbose=False)
    path = tmp_path / ckpt.filename

    # Epoch 0 - won't save because counter=0, epoch-counter=0, need >=2
    ckpt.save_if_improved(0, model, optimizer, scaler, None, val_loss=0.9)
    assert not path.exists(), "Should not save at epoch 0 with patience=2"

    # Epoch 1 - won't save because counter=0, epoch-counter=1, need >=2
    ckpt.save_if_improved(1, model, optimizer, scaler, None, val_loss=0.8)
    assert not path.exists(), "Should not save at epoch 1 with patience=2"

    # Epoch 2 - WILL save because counter=0, epoch-counter=2, which equals patience
    ckpt.save_if_improved(2, model, optimizer, scaler, None, val_loss=0.7)
    assert path.exists(), "Should save at epoch 2 (patience interval met)"

    # Epoch 3 - won't save because counter=2, epoch-counter=1, need >=2
    ckpt.save_if_improved(3, model, optimizer, scaler, None, val_loss=0.6)

    # Epoch 4 - WILL save because counter=2, epoch-counter=2, which equals patience
    ckpt.save_if_improved(4, model, optimizer, scaler, None, val_loss=0.5)
    assert path.exists(), "Should save at epoch 4 (patience interval met again)"


def test_load_without_checkpoint(tmp_path):
    """Handles missing checkpoint files."""
    model = DummyModel().to(DEVICE)
    ckpt = CheckpointManager(DEVICE, save_dir=tmp_path, verbose=False)
    result = ckpt.load(model)
    assert result == 0
