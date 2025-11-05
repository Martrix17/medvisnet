"""Main test function for image classification."""

import hydra
from omegaconf import DictConfig  # , OmegaConf

from src.training.base_trainer import BaseTrainer
from src.training.factories import (
    create_checkpoint_manager,
    create_data_module,
    create_metrics_manager,
    create_mlflow_logger,
    create_vit_model,
)
from src.training.trainer import Trainer
from src.utils.helper import (
    extract_hparams_from_cfg,
    save_figures,
    save_metrics,
    save_predictions,
)


@hydra.main(version_base=None, config_path="config", config_name="test")
def main(cfg: DictConfig):
    """Main training function."""
    # print(OmegaConf.to_yaml(cfg))

    device = cfg.trainer.device
    data_module = create_data_module(cfg.data)
    num_classes = data_module.get_num_classes()

    model = create_vit_model(cfg.model, num_classes=num_classes)

    metrics_manager = create_metrics_manager(num_classes=num_classes, device=device)
    checkpoint_manager = create_checkpoint_manager(cfg.trainer.checkpoint, device)

    logger = create_mlflow_logger(cfg.logging)
    logger.set_run_name(cfg.logging.run_name + "_test")
    logger.end_run()
    hparams = extract_hparams_from_cfg(cfg)
    logger.start_run()
    logger.log_params(hparams)

    base_trainer = BaseTrainer(
        model=model, device=device, criterion=None, optimizer=None
    )
    trainer = Trainer(
        base_trainer=base_trainer,
        epochs=0,
        data_module=data_module,
        metrics_manager=metrics_manager,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
    )

    print(f"Starting testing on device: {device}")
    output = trainer.test(
        load_checkpoint=cfg.trainer.load_checkpoint,
        plot_metrics=cfg.trainer.plot_metrics,
    )
    print("✅ Testing complete.")

    logger.end_run()

    local_save_dir = cfg.trainer.local_save_dir
    print(f"💾 Saving outputs locally to: {local_save_dir}")

    assert data_module.test_dataset is not None, "test_dataset should be loaded"
    base_dataset = data_module.test_dataset.dataset
    test_indices = data_module.test_dataset.indices
    test_paths = [str(base_dataset.samples[idx][0]) for idx in test_indices]
    save_predictions(
        img_paths=test_paths,
        preds=output["predictions"],
        targets=output["targets"],
        output_dir=local_save_dir,
    )
    save_metrics(metrics=output["metrics"]["report"], output_dir=local_save_dir)
    save_figures(figures=output["figures"], output_dir=local_save_dir)


if __name__ == "__main__":
    main()
