"""
Inference module for ONNX models.

Example:
    >>> inferencer = ONNXInferencer(device="cuda", onnx_path="model.onnx")
    >>> result = inferencer.predict("image.png")
"""

from pathlib import Path
from typing import Dict, Tuple

import onnxruntime as ort
import torch
from PIL import Image
from torchvision.transforms import v2

from .base_inferencer import BaseInferencer


class ONNXInferencer(BaseInferencer):
    """Inference for ONNX models using onnxruntime with provider selection."""

    def __init__(
        self,
        device: str,
        onnx_path: str,
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """
        Args:
            device: Device for inference ('cuda' or 'cpu').
            onnx_path: Path to .onnx model file.
            image_size: Target size for resizing.
            mean: Channel means for normalization (ImageNet defaults).
            std: Channel stds for normalization (ImageNet defaults).
        """
        super().__init__(device=device, model=None)

        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        available_providers = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Suppress warnings
        self.session = ort.InferenceSession(
            onnx_path, providers=providers, sess_options=sess_options
        )

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
        Run inference on a single image using ONNX runtime.

        Args:
            image_path: Path to input image file.

        Returns:
            Dict with 'probs' [1, num_classes] and 'preds' [1] tensors.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)

        ort_inputs = {self.session.get_inputs()[0].name: image.numpy()}
        ort_outs = self.session.run(None, ort_inputs)

        logits = torch.tensor(ort_outs[0])
        return self._postprocess(logits)
