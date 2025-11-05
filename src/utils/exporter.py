"""
Export PyTorch models to ONNX format.

Example:
    >>> export_model(
    ...     device="cuda",
    ...     model=model,
    ...     checkpoint_path="best_model.pt",
    ...     export_path="model.onnx",
    ...     image_size=(224, 224),
    ...     validate_export=True
    ... )
"""

import warnings
from pathlib import Path
from typing import Tuple

import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.jit import TracerWarning


def export_model(
    device: str,
    model: nn.Module,
    checkpoint_path: str,
    export_path: str | Path,
    image_size: Tuple[int, int],
    validate_export: bool = True,
) -> None:
    """
    Export trained PyTorch model to ONNX format.

    features:
    - Exports PyTorch model from loaded checkpoint
    - Output validation and consistency check between PyTorch and ONNX models (optional)

    Args:
        device: Device for model loading ('cuda' or 'cpu').
        model: Model to export.
        checkpoint_path: Path to .pt checkpoint.
        export_path: Output path for .onnx file.
        image_size: Input image dimensions (H, W).
        validate_export: Run ONNX validation and compare outputs if True.
    """
    model.eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state" not in checkpoint:
        raise KeyError("⚠️ Invalid checkpoint: missing 'model_state' key.")

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    dummy_input = torch.randn((1, 3, *image_size), device=device)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning, message="Named tensors")
        torch.onnx.export(
            model,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

    print(f"Exported model to: {export_path.resolve()}")

    if validate_export:
        print("🔍 Validating exported ONNX model.")
        onnx_model = onnx.load(str(export_path))
        onnx.checker.check_model(onnx_model)

        available_providers = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
            print("Provider not available: Defaulting to CPU.")
        ort_session = ort.InferenceSession(str(export_path), providers=providers)

        with torch.no_grad():
            torch_out = model(dummy_input).cpu().numpy()

        ort_inputs = {"input": dummy_input.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        diff = abs(torch_out - ort_outs[0]).mean()
        print("ONNX validation passed.")
        print(f"Mean output difference: {diff:.8f}")

        if diff > 1e-5:
            print("Output difference is higher than >1e-5!")
