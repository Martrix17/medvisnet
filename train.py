"""Main train function for image classification."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.base_trainer import BaseTrainer
from src.training.factories import (
    create_checkpoint_manager,
    create_criterion,
    create_data_module,
    create_early_stopping,
    create_metrics_manager,
    create_mlflow_logger,
    create_optimizer,
    create_scheduler,
    create_vit_model,
)
from src.training.trainer import Trainer
from src.utils.exporter import export_model
from src.utils.helper import extract_hparams_from_cfg


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    """Main training function."""
    print(OmegaConf.to_yaml(cfg))

    device = cfg.trainer.device
    data_module = create_data_module(cfg.data)
    weights = data_module.get_class_weights().to(device)
    num_classes = data_module.get_num_classes()

    model = create_vit_model(cfg.model, num_classes=num_classes)

    optimizer = create_optimizer(cfg.trainer.optimizer, model=model)
    criterion = create_criterion(cfg.trainer.criterion, weights=weights)
    scheduler = create_scheduler(cfg.trainer.scheduler, optimizer=optimizer)
    early_stopping = create_early_stopping(cfg.trainer.early_stopping)

    metrics_manager = create_metrics_manager(num_classes=num_classes, device=device)
    checkpoint_manager = create_checkpoint_manager(
        cfg.trainer.checkpoint, device=device
    )

    logger = create_mlflow_logger(cfg.logging)
    logger.set_run_name(cfg.logging.run_name + "_train")
    logger.end_run()
    hparams = extract_hparams_from_cfg(cfg)
    logger.start_run()
    logger.log_params(hparams)

    base_trainer = BaseTrainer(
        model=model, device=device, criterion=criterion, optimizer=optimizer
    )
    trainer = Trainer(
        base_trainer=base_trainer,
        epochs=cfg.trainer.epochs,
        data_module=data_module,
        scheduler=scheduler,
        metrics_manager=metrics_manager,
        logger=logger,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
        val_every_n_epochs=cfg.trainer.val_every_n_epochs,
        compute_metrics_n_val_epoch=cfg.trainer.compute_metrics_n_val_epoch,
    )

    print(f"Starting training for {cfg.trainer.epochs} epochs on device: {device}")
    trainer.fit(load_checkpoint=cfg.trainer.load_checkpoint)
    print(
        "✅ Training complete. Best model saved at:",
        checkpoint_manager.save_dir / checkpoint_manager.filename,
    )

    if cfg.trainer.export.export_after_training:
        export_model(
            model=model,
            checkpoint_path=str(
                checkpoint_manager.save_dir / checkpoint_manager.filename
            ),
            export_path=str(
                Path(cfg.trainer.export.save_dir) / cfg.trainer.export.filename
            ),
            image_size=cfg.data.image_size,
            device=device,
            validate_export=cfg.trainer.export.validate_export,
        )

    logger.end_run()


if __name__ == "__main__":
    main()
