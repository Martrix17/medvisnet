"""
Unit tests for VisionTransformer class.
"""

import pytest
import torch

from src.models.vit import VisionTransformer


@pytest.mark.parametrize("model_name", VisionTransformer.list_available_models())
def test_model_initialization(model_name):
    model = VisionTransformer(model_name=model_name, num_classes=4)
    assert isinstance(model.vit, torch.nn.Module)
    assert model.vit.heads.head[-1].out_features == 4


def test_forward_pass_shapes():
    model = VisionTransformer(model_name="vit_b_16", num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 4)


def test_freeze_backbone_behavior():
    model = VisionTransformer(model_name="vit_b_16", freeze_backbone=True)
    encoder_params = [p.requires_grad for p in model.vit.encoder.parameters()]
    assert all(not req for req in encoder_params)


def test_trainable_param_count_changes_with_freeze():
    model_free = VisionTransformer(model_name="vit_b_16", freeze_backbone=True)
    model_unfree = VisionTransformer(model_name="vit_b_16", freeze_backbone=False)
    assert model_free.get_trainable_params() < model_unfree.get_trainable_params()
