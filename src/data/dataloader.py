"""
DataLoader utilities for COVID-19 Radiography dataset.

Features:
- Stratified train/val/test split
- WeightedRandomSampler for balanced training
- Class-weight computation for imbalanced loss handling
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .dataset import CovidRadiographyDataset


class CovidRadiographyDataModule:
    """
    DataModule-like wrapper for CovidRadiographyDataset.

    Handles:
        - Train/val/test split
        - Weighted sampling for training
        - Class weight computation for losses
        - DataLoader creation for each split
    """

    def __init__(
        self,
        data_dir: str,
        augment: bool = False,
        image_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ) -> None:
        """
        Args:
            data_dir: Path to dataset root.
            augment: If True, applies augmentations to train set.
            image_size: Size for resizing images.
            mean: Mean values for  normalization.
            std: Standard deviation values for  normalization.
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for DataLoaders.
            val_split: Fraction of data to use for validation.
            test_split: Fraction of data to use for testing.
            seed: Random seed for reproducibility.
        """
        self.data_dir = data_dir
        self.augment = augment
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None
        self.class_weights: Optional[torch.Tensor] = None
        self.class_to_idx: Optional[Dict[str, int]] = None
        self.label_names: Optional[List[str]] = None

    def setup(self) -> None:
        """Create stratified train/val/test splits."""
        full_dataset = CovidRadiographyDataset(
            data_dir=self.data_dir,
            augment=self.augment,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )

        labels = [lbl for _, lbl in full_dataset.samples]
        n_total = len(labels)

        train_idx, temp_idx = train_test_split(
            np.arange(n_total),
            test_size=self.val_split + self.test_split,
            stratify=labels,
            random_state=self.seed,
        )

        temp_labels = [labels[i] for i in temp_idx]
        val_rel_size = self.val_split / (self.val_split + self.test_split)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - val_rel_size,
            stratify=temp_labels,
            random_state=self.seed,
        )

        self.train_dataset = Subset(full_dataset, train_idx)
        self.val_dataset = Subset(
            CovidRadiographyDataset(
                self.data_dir,
                augment=False,
                image_size=self.image_size,
                mean=self.mean,
                std=self.std,
            ),
            val_idx,
        )
        self.test_dataset = Subset(
            CovidRadiographyDataset(
                self.data_dir,
                augment=False,
                image_size=self.image_size,
                mean=self.mean,
                std=self.std,
            ),
            test_idx,
        )

        self.class_to_idx = full_dataset.class_to_idx
        self.class_weights = self._compute_class_weights(full_dataset)
        self.label_names = self.get_label_names()

    def train_dataloader(self) -> DataLoader:
        """Return DataLoader with weighted sampling for training."""
        sampler = self._make_weighted_sampler(self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for validation (no balancing)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for test (no balancing)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_class_weights(self) -> torch.Tensor:
        """Expose computed class weights for loss function."""
        if self.class_weights is None:
            raise RuntimeError("Call setup() before accessing class weights.")
        return self.class_weights

    def get_label_names(self) -> List[str]:
        """Return list of class names ordered by class index."""
        if self.class_to_idx is None:
            raise RuntimeError("Call setup() before accessing label names.")
        sorted_classes = sorted(self.class_to_idx.items(), key=lambda x: x[1])
        return [name for name, _ in sorted_classes]

    def get_num_classes(self) -> int:
        """Return number of classes."""
        if not self.class_to_idx:
            raise RuntimeError("Call setup() before accessing class count.")
        return len(self.class_to_idx)

    def _compute_class_weights(self, dataset: CovidRadiographyDataset) -> torch.Tensor:
        """Compute class weights inverse to frequency."""
        labels = [lbl for _, lbl in dataset.samples]
        counts = Counter(labels)
        total = sum(counts.values())
        weights = [total / counts[c] for c in dataset.class_to_idx.keys()]
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        weights_tensor /= weights_tensor.sum()
        return weights_tensor

    def _make_weighted_sampler(self, dataset: Subset) -> WeightedRandomSampler:
        """Create WeightedRandomSampler for balanced training."""
        base_dataset = dataset.dataset
        indices = dataset.indices

        labels = [base_dataset.samples[i][1] for i in indices]
        label_indices = [base_dataset.class_to_idx[label] for label in labels]
        labels_tensor = torch.tensor(label_indices, dtype=torch.long)

        all_labels = [s[1] for s in base_dataset.samples]
        all_label_indices = [base_dataset.class_to_idx[label] for label in all_labels]
        class_counts = torch.bincount(torch.tensor(all_label_indices, dtype=torch.long))

        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels_tensor]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
