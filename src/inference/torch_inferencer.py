"""
Inference module for PyTorch models.

Example:
    >>> # PyTorch inference
    >>> inferencer = TorchInferencer(
    ...     device="cuda",
    ...     model=model,
    ...     checkpoint_path="best_model.pt"
    ... )
    >>> result = inferencer.predict("image.png")
    >>> probs, preds = result["probs"], result["preds"]
"""

from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import v2

from .base_inferencer import BaseInferencer


class TorchInferencer(BaseInferencer):
    """Inference for PyTorch Vision Transformer models with checkpoint loading."""

    def __init__(
        self,
        device: str,
        model: nn.Module,
        checkpoint_path: str,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """
        Args:
            device: Device for inference ('cuda' or 'cpu').
            model: PyTorch model to load weights into.
            checkpoint_path: Path to .pt checkpoint file with 'model_state' key.
            image_size: Target size for resizing.
            mean: Channel means for normalization (ImageNet defaults).
            std: Channel stds for normalization (ImageNet defaults).
        """
        super().__init__(device=device, model=model)

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        if "model_state" not in checkpoint:
            raise KeyError("Invalid checkpoint: missing 'model_state' key.")

        model.load_state_dict(checkpoint["model_state"])
        self.model.to(device)

        self.transform = v2.Compose(
            [
                v2.Resize(image_size),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean, std),
            ]
        )

    def predict(self, image_path: str) -> Dict[str, torch.Tensor]:
        """
        Run inference on a single image.

        Args:
            image_path: Path to input image file.

        Returns:
            Dict with 'probs' [1, num_classes] and 'preds' [1] tensors.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image)
        return self._postprocess(logits)
