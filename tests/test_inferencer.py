"""Unit tests for TorchInferencer and ONNXInferencer classes."""

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.inference.onnx_inferencer import ONNXInferencer
from src.inference.torch_inferencer import TorchInferencer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DummyModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.linear = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.linear(x)


@pytest.fixture
def dummy_checkpoint(tmp_path):
    """Creates a dummy PyTorch checkpoint file."""
    model = DummyModel()
    checkpoint_path = tmp_path / "dummy.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def dummy_onnx(tmp_path):
    """Creates a dummy ONNX file."""
    model = DummyModel().to(DEVICE)
    model.eval()
    onnx_path = tmp_path / "dummy.onnx"
    dummy_input = torch.randn((1, 3, 224, 224), device=DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        dynamo=True,
    )
    return onnx_path


@pytest.fixture
def dummy_image(tmp_path):
    img_path = tmp_path / "dummy.png"
    dummy = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
    Image.fromarray(dummy.numpy()).save(img_path)
    return img_path


def test_torch_inferencer_predict(dummy_checkpoint, dummy_image):
    model = DummyModel()
    inferencer = TorchInferencer(
        device=DEVICE,
        model=model,
        checkpoint_path=str(dummy_checkpoint),
        image_size=(224, 224),
    )
    outputs = inferencer.predict(dummy_image)

    assert isinstance(outputs, dict)
    assert isinstance(outputs["probs"], torch.Tensor)
    assert isinstance(outputs["preds"], torch.Tensor)
    assert outputs["probs"].shape == (1, 4)
    assert outputs["preds"].shape[0] == 1


def test_onnx_inferencer_predict(dummy_onnx, dummy_image):
    inferencer = ONNXInferencer(
        device=DEVICE,
        onnx_path=str(dummy_onnx),
        image_size=(224, 224),
    )
    outputs = inferencer.predict(dummy_image)

    assert isinstance(outputs, dict)
    assert isinstance(outputs["probs"], torch.Tensor)
    assert isinstance(outputs["preds"], torch.Tensor)
    assert outputs["probs"].shape == (1, 4)
    assert outputs["preds"].shape[0] == 1
