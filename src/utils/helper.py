"""Helper functions for saving outputs."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf


def save_predictions(
    img_paths: List[str], preds: torch.Tensor, targets: torch.Tensor, output_dir: str
) -> None:
    """Saves predictions and target labels in csv file."""
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
    df.to_csv(f"{output_dir}/test_predictions.csv", index=False)


def save_metrics(metrics: str, output_dir: str) -> None:
    """Saves output metrics as text file."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/metrics.txt", "w", encoding="utf-8") as f:
        f.write(metrics)


def save_figures(
    figures: Dict[str, plt.Figure], output_dir: str, dpi: int = 300
) -> None:
    """Saves matplotlib figures as png."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(f"{output_dir}/{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def flatten_dict(d, parent_key="", sep=".") -> Dict[str, Any]:
    """Recursively flatten nested dicts."""
    items: List[Any] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_hparams_from_cfg(cfg) -> Dict[str, Any]:
    """Convert Hydra config to flat dict."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return flatten_dict(cfg_dict)
