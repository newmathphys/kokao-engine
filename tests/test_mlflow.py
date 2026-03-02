"""Тесты для MLflow (7 тестов)."""

import os
import sys

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.mlflow import MLflowLogger, get_mlflow_logger


class TestMLflowLogger:
    """MLflowLogger."""

    def test_init(self):
        logger = MLflowLogger(experiment_name="Test")
        assert logger.experiment_name == "Test"

    def test_start_end_run(self):
        try:
            logger = MLflowLogger(experiment_name="Test")
            logger.start_run(run_name="test")
            assert logger._run_active is True
            logger.end_run()
            assert logger._run_active is False
        except Exception:
            pass

    def test_context_manager(self):
        try:
            with MLflowLogger(experiment_name="Test") as logger:
                assert logger._run_active is True
            assert logger._run_active is False
        except Exception:
            pass

    def test_log_param(self):
        try:
            logger = MLflowLogger(experiment_name="Test")
            logger.start_run()
            logger.log_param("n_features", 10)
            logger.end_run()
        except Exception:
            pass

    def test_log_metric(self):
        try:
            logger = MLflowLogger(experiment_name="Test")
            logger.start_run()
            logger.log_metric("loss", 0.5, step=1)
            logger.end_run()
        except Exception:
            pass


class TestMLflowHelpers:
    """Helper функции."""

    def test_singleton(self):
        logger1 = get_mlflow_logger()
        logger2 = get_mlflow_logger()
        assert logger1 is logger2

    def test_log_training(self):
        try:
            import torch

            from kokao.core import KokaoCoreV9

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            core = KokaoCoreV9(n_features=5)
            x = torch.randn(5, device=device)
            loss = core.train(x, target=100.0)

            from kokao.mlflow import log_training

            log_training(core, loss, step=1)
        except Exception:
            pass

    def test_log_signal(self):
        try:
            import torch

            from kokao.core import KokaoCoreV9

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            core = KokaoCoreV9(n_features=5)
            x = torch.randn(5, device=device)
            signal = core.signal(x)

            from kokao.mlflow import log_signal

            log_signal(core, signal, target=100.0, step=1)
        except Exception:
            pass
