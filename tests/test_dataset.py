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


def test_dataset_with_augmentation(mock_covid_data):
    dataset = CovidRadiographyDataset(
        mock_covid_data, augment=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )
    img, _ = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert -1.0 <= img.min() <= 1.0  # scaled to float
    assert -5 <= img.mean() <= 5  # sanity check that normalization works


def test_label_index_consistency(mock_covid_data):
    dataset = CovidRadiographyDataset(mock_covid_data)
    for label_name, idx in dataset.class_to_idx.items():
        assert dataset.idx_to_class[idx] == label_name


def test_skips_invalid_directories(tmp_path):
    (tmp_path / "INVALID").mkdir()
    dataset = CovidRadiographyDataset(tmp_path)
    assert len(dataset) == 0


def test_dataset_without_normalization(mock_covid_data):
    dataset = CovidRadiographyDataset(mock_covid_data, mean=None, std=None)
    img, _ = dataset[0]
    assert isinstance(img, torch.Tensor)


def test_class_order_is_sorted(mock_covid_data):
    dataset = CovidRadiographyDataset(mock_covid_data)
    classes = list(dataset.class_to_idx.keys())
    assert classes == sorted(classes)


def test_dataset_length_matches_files(mock_covid_data):
    expected_count = sum(
        len(list((mock_covid_data / c / "images").glob("*.png")))
        for c in ["COVID", "NORMAL", "LUNG_OPACITY", "VIRAL_PNEUMONIA"]
    )
    dataset = CovidRadiographyDataset(mock_covid_data)
    assert len(dataset) == expected_count
