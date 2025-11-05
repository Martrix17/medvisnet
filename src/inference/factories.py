"""
Factory functions for creating inference components from Hydra configurations.
"""

from pathlib import Path

from src.models.vit import VisionTransformer

from .onnx_inferencer import ONNXInferencer
from .torch_inferencer import TorchInferencer


def create_inference_vit_model(cfg) -> VisionTransformer:
    """Create Vision Transformer model from config."""
    return VisionTransformer(
        model_name=cfg.model.model_name,
        num_classes=cfg.inference.num_classes,
        num_hidden_layers=cfg.model.num_hidden_layers,
        weights=cfg.model.get("weights", None),
        freeze_backbone=False,
    )


def create_inferencer(cfg):
    """Create PyTorch or ONNX inference modules from config."""
    backend = cfg.inference.backend.lower()
    device = cfg.inference.device

    if backend == "torch":
        model = create_inference_vit_model(cfg)
        checkpoint_path = str(
            Path(cfg.inference.checkpoint.save_dir) / cfg.inference.checkpoint.filename
        )
        return TorchInferencer(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
        )
    elif backend == "onnx":
        onnx_path = str(
            Path(cfg.inference.export.save_dir) / cfg.inference.export.filename
        )
        return ONNXInferencer(
            onnx_path=onnx_path,
            device=device,
            image_size=tuple(cfg.data.image_size),
            mean=tuple(cfg.data.mean),
            std=tuple(cfg.data.std),
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")
