"""
Unit tests for CovidRadiographyDataset.
"""

import torch

from src.data.dataset import CovidRadiographyDataset


def test_dataset_loading(mock_covid_data):
    dataset = CovidRadiographyDataset(mock_covid_data)
    assert len(dataset) > 0
    assert all(
        k in dataset.class_to_idx
        for k in ["COVID", "NORMAL", "LUNG_OPACITY", "VIRAL_PNEUMONIA"]
    )


def test_dataset_item(mock_covid_data):
    dataset = CovidRadiographyDataset(mock_covid_data)
    img, label = dataset[0]
    assert img.shape[0] == 3  # RGB
    assert label.dtype == torch.long
