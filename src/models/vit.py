"""
Vision transformer building class from torchvision implementations.

Example:
    >>> model = VisionTransformer("vit_b_16", num_classes=4, weights="IMAGENET1K_V1")
    >>> output = model(torch.randn(1, 3, 224, 224))
"""

import torch
import torch.nn as nn
from torchvision.models import vision_transformer


class VisionTransformer(nn.Module):
    """Vision Transformer wrapper with customizable classification head."""

    AVAILABLE_MODELS = {
        "vit_b_16": vision_transformer.vit_b_16,
        "vit_b_32": vision_transformer.vit_b_32,
        "vit_l_16": vision_transformer.vit_l_16,
        "vit_l_32": vision_transformer.vit_l_32,
    }

    def __init__(
        self,
        model_name: str,
        num_classes: int = 4,
        num_hidden_layers: int = 0,
        weights: str | None = None,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        """
        Args:
            model_name: Name of torchvision ViT model (see list_available_models()).
            num_classes: Number of output classes.
            num_hidden_layers: Number of hidden layers in classification head.
            weights: Pretrained weights identifier (e.g., "IMAGENET1K_V1") or None.
            dropout: Dropout rate in classification head.
            freeze_backbone: Freeze encoder parameters if True.
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.weights = weights
        self.dropout_rate = dropout
        self.freeze_backbone = freeze_backbone

        self.vit = self._build_model()
        self._replace_head(num_hidden_layers)
        self._freeze_backbone()

    def _build_model(self):
        """Build base ViT model from torchvision."""
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Input model_name ({self.model_name}) not found."
                f"See available models with 'list_available_models'."
            )

        builder = self.AVAILABLE_MODELS[self.model_name]
        return builder(weights=self.weights)

    def _replace_head(self, num_hidden_layers: int = 1) -> None:
        """
        Replace default head with custom classification head.

        Each hidden layer halves the dimension with BatchNorm, ReLU, and Dropout.
        """
        in_features = self.vit.heads.head.in_features
        layers = [nn.Dropout(self.dropout_rate)]

        current_dim = in_features
        for _ in range(num_hidden_layers):
            next_dim = max(1, current_dim // 2)
            layers.extend(
                [
                    nn.Linear(current_dim, next_dim),
                    nn.BatchNorm1d(next_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                ]
            )
            current_dim = next_dim

        layers.append(nn.Linear(current_dim, self.num_classes))
        self.vit.heads.head = nn.Sequential(*layers)

    def _freeze_backbone(self) -> None:
        """Freeze encoder parameters if freeze_backbone is True."""
        if self.freeze_backbone:
            for param in self.vit.encoder.parameters():
                param.requires_grad = False

    def get_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W].

        Returns:
            Logits of shape [B, num_classes].
        """
        return self.vit(x)

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Return list of available model names."""
        return list(cls.AVAILABLE_MODELS.keys())
