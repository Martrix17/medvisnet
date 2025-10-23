"""MLFlow logger module for experiment tracking."""

from typing import Any, Dict

import mlflow
from matplotlib.figure import Figure


class MLFlowLogger:
    """MLFlow logger for experiment tracking."""

    def __init__(
        self,
        run_name: str,
        uri: str = "mlruns",
        experiment_name: str = "ViT_COVID19_Classification",
        log_system: bool = True,
    ) -> None:
        """
        Args:
            run_name (str): Name of the MLFlow run.
            uri (str): Tracking URI.
            experiment_name (str): Name of the experiment.
            log_system (bool): Whether to log system metrics.
        """
        self.uri = uri
        self.experiment_name = experiment_name
        self.log_system = log_system
        self.run_name = run_name

    def start_run(self) -> None:
        """Start MLFlow run-"""
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name, log_system_metrics=self.log_system)

    def set_experiment_name(self, experiment_name: str) -> None:
        """Set experiment name."""
        self.experiment_name = experiment_name

    def set_run_name(self, run_name: str) -> None:
        """Set run name."""
        self.run_name = run_name

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics to MLFlow."""
        mlflow.log_metrics(metrics=metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to MLFlow."""
        mlflow.log_params(params=params)

    def log_artifact(self, file_path: str) -> None:
        """Log file or drectory as an artifact."""
        mlflow.log_artifact(artifact_path=file_path)

    def log_text(self, text: str, file_path: str) -> None:
        """Log text content as an artifact."""
        mlflow.log_text(text=text, artifact_file=file_path)

    def log_figure(self, fig: Figure, file_path: str) -> None:
        """Log matplotlib figure as an artifact."""
        mlflow.log_figure(figure=fig, artifact_file=file_path)

    def log_tags(self, tags: Dict[str, Any]) -> None:
        """Log tags to MLFlow."""
        mlflow.set_tag(tags)

    def end_run(self) -> None:
        """End MLFlow run."""
        mlflow.end_run()
