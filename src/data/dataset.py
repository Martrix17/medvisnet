"""
Custom PyTorch dataset for COVID-19 Radiography Database classification tasks.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class CovidRadiographyDataset(Dataset):
    """
    Dataset for COVID-19 Radiography Database.
    """

    def __init__(
        self,
        data_dir: str,
        augment: bool = False,
        image_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        """
        Args:
            data_dir: Path to dataset root.
            augment: If True, applies augmentations.
            image_size: Size for resizing images.
            mean: Mean values for  normalization.
            std: Standard deviation values for  normalization.
        """
        self.data_dir = Path(data_dir)

        self.samples = self._load_samples()
        self.class_to_idx = self._build_class_index()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.transform = v2.Compose(
            [
                v2.Resize(image_size),
                v2.RandomHorizontalFlip() if augment else v2.Lambda(lambda x: x),
                v2.RandomVerticalFlip() if augment else v2.Lambda(lambda x: x),
                v2.RandomRotation(5) if augment else v2.Lambda(lambda x: x),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std) if mean and std else v2.Lambda(lambda x: x),
            ]
        )

    def _load_samples(self) -> List[Tuple[Path, str]]:
        """Find all image files and infer labels from filenames."""
        samples = []
        for class_dir in self.data_dir.iterdir():
            if not class_dir.is_dir():
                continue
            img_dir = class_dir / "images"
            if not img_dir.exists():
                continue
            for img_path in img_dir.glob("*.png"):
                label = img_path.stem.split("-")[0].upper()
                samples.append((img_path, label))
        return samples

    def _build_class_index(self) -> Dict[str, int]:
        """Create label to index mapping."""
        classes = sorted({label for _, label in self.samples})
        return {cls_name: idx for idx, cls_name in enumerate(classes)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label_idx = self.class_to_idx[label]
        label = torch.tensor(label_idx, dtype=torch.long)
        return image, label
