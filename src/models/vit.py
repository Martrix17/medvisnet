"""Vision transformer building from torchvision implementations."""

import torch
from torch import nn
from torchvision.models import vision_transformer


class VisionTransformer(nn.Module):
    """Vision Transformer model for image classification."""

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
            model_name (str): Name of the torchvision ViT model to use.
            num_classes (int): Number of output classes.
            num_hidden_layers (int): Number of hidden layers in classification head.
            weights (str): Pretrained weights to use. See torchvision docs.
            dropout (float): Dropout rate for the classification head.
            freeze_backbone (bool): Whether to freeze the backbone during training.
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
        """Build base model."""
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Input model_name ({self.model_name}) not found."
                f"See available models with 'list_available_models'."
            )

        builder = self.AVAILABLE_MODELS[self.model_name]
        return builder(weights=self.weights)

    def _replace_head(self, num_hidden_layers: int = 1) -> None:
        """Replace classification head with custom head."""
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
        """Freeze backbone encoder parameters."""
        if self.freeze_backbone:
            for param in self.vit.encoder.parameters():
                param.requires_grad = False

    def get_trainable_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x) -> torch.Tensor:
        """Forward pass."""
        return self.vit(x)

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available model names."""
        return list(cls.AVAILABLE_MODELS.keys())
