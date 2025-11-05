"""Main inference function for image classification."""

import hydra
from omegaconf import DictConfig

from src.inference.factories import create_inferencer
from src.utils.helper import save_inference


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(cfg: DictConfig):
    inferencer = create_inferencer(cfg)
    outputs = inferencer.predict(cfg.inference.image_path)

    local_save_dir = cfg.inference.local_save_dir
    labels = tuple(cfg.inference.labels)
    pred_class = labels[outputs["preds"].cpu()]
    outputs["class"] = pred_class

    print(f"Image classified as: {pred_class}")
    print(f"💾 Saving outputs locally to: {local_save_dir}")
    save_inference(outputs=outputs, output_dir=local_save_dir)


if __name__ == "__main__":
    main()
