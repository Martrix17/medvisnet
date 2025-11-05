"""
Unit tests for CovidRadiographyDataModule class and dataset integration.
"""

import pytest
import torch

from src.data.dataloader import CovidRadiographyDataModule
from src.data.dataset import CovidRadiographyDataset


def test_data_module_setup(mock_covid_data):
    """Ensure that setup() correctly splits and initializes datasets."""
    dm = CovidRadiographyDataModule(
        data_dir=mock_covid_data, batch_size=2, val_split=0.2, test_split=0.2
    )
    dm.setup()

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None
    assert dm.class_to_idx is not None
    assert set(dm.class_to_idx.keys()) == {
        "COVID",
        "NORMAL",
        "LUNG_OPACITY",
        "VIRAL_PNEUMONIA",
    }

    total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    full_dataset = CovidRadiographyDataset(mock_covid_data)
    assert total == len(full_dataset)


def test_class_weights_sum_to_one(mock_covid_data):
    """Class weights should be normalized and positive."""
    dm = CovidRadiographyDataModule(mock_covid_data)
    dm.setup()
    weights = dm.get_class_weights()
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.all(weights > 0)


def test_train_dataloader_balanced_sampling(mock_covid_data):
    """Weighted sampler should balance sampled labels."""
    torch.manual_seed(42)
    dm = CovidRadiographyDataModule(mock_covid_data, batch_size=8)
    dm.setup()

    loader = dm.train_dataloader()
    batch_labels = []
    for i, (_, labels) in enumerate(loader):
        batch_labels.extend(labels.tolist())
        if i > 50:
            break

    counts = torch.bincount(torch.tensor(batch_labels))
    mean_count = counts.float().mean().item()
    imbalance_ratio = (counts.max().item() - counts.min().item()) / mean_count

    assert imbalance_ratio < 0.4, f"Sampler imbalance too high: {counts.tolist()}"


def test_dataloader_shapes(mock_covid_data):
    dm = CovidRadiographyDataModule(mock_covid_data, batch_size=4)
    dm.setup()

    loader = dm.train_dataloader()
    imgs, labels = next(iter(loader))

    assert imgs.ndim == 4  # B, C, H, W
    assert imgs.shape[1] == 3
    assert labels.ndim == 1
    assert labels.dtype == torch.long


def test_split_reproducibility(mock_covid_data):
    dm1 = CovidRadiographyDataModule(mock_covid_data, seed=123)
    dm2 = CovidRadiographyDataModule(mock_covid_data, seed=123)
    dm1.setup()
    dm2.setup()
    assert list(dm1.train_dataset.indices) == list(dm2.train_dataset.indices)


def test_label_names_and_num_classes(mock_covid_data):
    dm = CovidRadiographyDataModule(mock_covid_data)
    dm.setup()
    names = dm.get_label_names()
    assert isinstance(names, list) and all(isinstance(n, str) for n in names)
    assert dm.get_num_classes() == len(names)


def test_access_before_setup_raises(mock_covid_data):
    dm = CovidRadiographyDataModule(mock_covid_data)
    with pytest.raises(RuntimeError):
        dm.get_class_weights()


def test_setup_with_empty_dataset(tmp_path):
    (tmp_path / "EMPTY").mkdir()
    dm = CovidRadiographyDataModule(tmp_path)
    with pytest.raises(ValueError):
        dm.setup()
