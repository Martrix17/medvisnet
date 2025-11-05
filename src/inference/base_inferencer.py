"""Base class for inference modules."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch


class BaseInferencer(ABC):
    """Abstract base class for model inference."""

    def __init__(self, device: str, model: torch.nn.Module | None) -> None:
        """
        Args:
            device: Device for inference ('cuda' or 'cpu').
            model: PyTorch model or None for non-PyTorch backends.
        """
        self.device = device

        if model:
            self.model = model.to(device)
            self.model.eval()

    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Run model inference (implementation required in subclasses)."""
        raise NotImplementedError

    def _postprocess(self, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert logits to probabilities and predictions.

        Returns:
            Dict with 'probs' [B, num_classes] and 'preds' [B] tensors.
        """
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"probs": probs, "preds": preds}
