"""MLflow Logging — 10 тестов."""

import os
import sys

import pytest

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.mlflow import MLFLOW_AVAILABLE, MLflowLogger, get_mlflow_logger


@pytest.mark.skipif(not MLFLOW_AVAILABLE, reason="mlflow not installed")
class TestMLflowLog:
    """MLflow логирование."""

    def test_mlflow_init_default(self):
        logger = MLflowLogger(experiment_name="Test")
        assert logger.experiment_name == "Test"

    def test_mlflow_init_custom_uri(self):
        logger = MLflowLogger(experiment_name="Test", tracking_uri="sqlite:///test.db")
        assert logger.tracking_uri == "sqlite:///test.db"

    def test_mlflow_start_run(self):
        logger = MLflowLogger(experiment_name="Test")
        logger.start_run(run_name="test")
        assert logger._run_active is True
        logger.end_run()

    def test_mlflow_end_run(self):
        logger = MLflowLogger(experiment_name="Test")
        logger.start_run()
        logger.end_run()
        assert logger._run_active is False

    def test_mlflow_context_manager(self):
        with MLflowLogger(experiment_name="Test") as logger:
            assert logger._run_active is True
        assert logger._run_active is False

    def test_mlflow_log_param_int(self):
        logger = MLflowLogger(experiment_name="Test")
        logger.start_run()
        logger.log_param("n_features", 10)
        logger.end_run()

    def test_mlflow_log_param_str(self):
        logger = MLflowLogger(experiment_name="Test")
        logger.start_run()
        logger.log_param("device", "cpu")
        logger.end_run()

    def test_mlflow_log_metric(self):
        logger = MLflowLogger(experiment_name="Test")
        logger.start_run()
        logger.log_metric("loss", 0.5, step=1)
        logger.end_run()

    def test_mlflow_log_metrics_dict(self):
        logger = MLflowLogger(experiment_name="Test")
        logger.start_run()
        logger.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)
        logger.end_run()

    def test_mlflow_singleton(self):
        logger1 = get_mlflow_logger()
        logger2 = get_mlflow_logger()
        assert logger1 is logger2
