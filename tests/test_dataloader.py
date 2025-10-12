"""
Unit tests for CovidRadiographyDataModule and dataset integration.
"""

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
    """Weighted sampler should roughly balance sampled labels."""
    dm = CovidRadiographyDataModule(mock_covid_data, batch_size=8)
    dm.setup()

    loader = dm.train_dataloader()
    batch_labels = []
    for i, (_, labels) in enumerate(loader):
        batch_labels.extend(labels.tolist())
        if i > 10:
            break

    counts = torch.bincount(torch.tensor(batch_labels))
    assert (counts.max() - counts.min()) < 5, "Sampler not balancing classes properly"


def test_dataloader_shapes(mock_covid_data):
    """Ensure dataloader returns correct image and label shapes."""
    dm = CovidRadiographyDataModule(mock_covid_data, batch_size=4)
    dm.setup()

    loader = dm.train_dataloader()
    imgs, labels = next(iter(loader))

    assert imgs.ndim == 4  # B, C, H, W
    assert imgs.shape[1] == 3
    assert labels.ndim == 1
    assert labels.dtype == torch.long
