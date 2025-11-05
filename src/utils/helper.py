"""Utility functions for saving predictions, metrics, and visualizations."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf


def save_predictions(
    img_paths: List[str], preds: torch.Tensor, targets: torch.Tensor, output_dir: str
) -> None:
    """
    Save predictions and targets to CSV.

    Args:
        img_paths: List of image file paths.
        preds: Model predictions [N, num_classes].
        targets: Ground truth labels [N].
        output_dir: Directory to save predictions.csv.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    img_names = [Path(p).stem for p in img_paths]
    preds = torch.argmax(preds, dim=1)
    df = pd.DataFrame(
        {
            "image": img_names,
            "target": targets.cpu().tolist(),
            "prediction": preds.cpu().tolist(),
        }
    )
    df.to_csv(f"{output_dir}/predictions.csv", index=False)


def save_metrics(metrics: str, output_dir: str) -> None:
    """
    Save metrics report to text file.

    Args:
        metrics: Metrics text (e.g., classification report).
        output_dir: Directory to save metrics.txt.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/metrics.txt", "w", encoding="utf-8") as f:
        f.write(metrics)


def save_inference(outputs: Dict[str, str | torch.Tensor], output_dir: str) -> None:
    """
    Save single inference output to CSV.

    Args:
        outputs: Dict with 'class' (predicted class) and 'probs' (probabilities tensor).
        output_dir: Directory to save predictions.csv.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    probs = outputs["probs"]
    if isinstance(probs, torch.Tensor):
        prob_list = probs.cpu().tolist()
    else:
        raise TypeError("Expected 'probs' to be a torch.Tensor")
    df = pd.DataFrame(
        {
            "prediction": [outputs["class"]],
            "probabilities": [prob_list],
        }
    )
    df.to_csv(f"{output_dir}/predictions.csv", index=False)


def save_figures(
    figures: Dict[str, plt.Figure], output_dir: str, dpi: int = 300
) -> None:
    """
    Save matplotlib figures as PNG images.

    Args:
        figures: Dict mapping filenames to Figure objects.
        output_dir: Directory to save images.
        dpi: Image resolution (dots per inch).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(f"{output_dir}/{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def flatten_hparams_dict(d, parent_key="", sep=".") -> Dict[str, Any]:
    """
    Recursively flatten nested dict with dot-separated keys.

    Args:
        d: Nested dictionary to flatten.
        parent_key: Parent key prefix for recursion.
        sep: Separator for nested keys.

    Returns:
        Flattened dict (e.g., {"model.lr": 0.001, "data.batch_size": 32}).
    """
    items: List[Any] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_hparams_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_hparams_from_cfg(cfg) -> Dict[str, Any]:
    """
    Convert Hydra config to flattened hyperparameter dict.

    Args:
        cfg: Hydra DictConfig or OmegaConf config.

    Returns:
        Flattened dict (e.g., {"model.lr": 0.001}).
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return flatten_hparams_dict(cfg_dict)
