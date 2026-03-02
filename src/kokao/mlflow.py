"""MLflow logging для Kokao Engine."""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow не установлен. Логирование отключено.")


class MLflowLogger:
    """MLflow логгер для Kokao Engine."""

    def __init__(self, experiment_name: str = "Kokao Engine", tracking_uri: Optional[str] = None):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._run_active = False

        if MLFLOW_AVAILABLE:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: Optional[str] = None) -> None:
        """Начать MLflow run."""
        if MLFLOW_AVAILABLE and not self._run_active:
            mlflow.start_run(run_name=run_name)
            self._run_active = True
            logger.info(f"MLflow run started: {run_name or 'default'}")

    def end_run(self) -> None:
        """Завершить MLflow run."""
        if MLFLOW_AVAILABLE and self._run_active:
            mlflow.end_run()
            self._run_active = False
            logger.info("MLflow run ended")

    def log_param(self, key: str, value: Any) -> None:
        """Логировать параметр."""
        if MLFLOW_AVAILABLE and self._run_active:
            mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """Логировать метрику."""
        if MLFLOW_AVAILABLE and self._run_active:
            mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Логировать несколько метрик."""
        if MLFLOW_AVAILABLE and self._run_active:
            mlflow.log_metrics(metrics, step=step)

    def log_model(self, model: Any, artifact_path: str = "model") -> None:
        """Логировать модель."""
        if MLFLOW_AVAILABLE and self._run_active:
            try:
                mlflow.pytorch.log_model(model, artifact_path)
            except Exception as e:
                logger.warning(f"Не удалось залогировать модель: {e}")

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()


# Global logger instance
_global_logger: Optional[MLflowLogger] = None


def get_mlflow_logger(experiment_name: str = "Kokao Engine") -> MLflowLogger:
    """Получить глобальный MLflow логгер."""
    global _global_logger
    if _global_logger is None:
        _global_logger = MLflowLogger(experiment_name)
    return _global_logger


def log_training(core: Any, loss: float, step: int) -> None:
    """Логировать шаг обучения."""
    logger = get_mlflow_logger()
    if logger._run_active:
        logger.log_metric("loss", loss, step=step)
        if hasattr(core, "w"):
            logger.log_metric("weights_sum", float(core.w.sum()), step=step)


def log_signal(core: Any, signal: float, target: float, step: int) -> None:
    """Логировать сигнал."""
    logger = get_mlflow_logger()
    if logger._run_active:
        logger.log_metric("signal", signal, step=step)
        logger.log_metric("target", target, step=step)
        logger.log_metric("error", abs(signal - target), step=step)
