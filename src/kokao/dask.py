"""Dask Backend — альтернатива Ray для распределённого обучения."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from dask.distributed import Client, LocalCluster

    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask не установлен. DaskBackend недоступен.")

import torch

from .core import KokaoCoreV9


class DaskBackend:
    """Dask Backend для распределённого обучения Kokao.

    Альтернатива Ray для сред, где Dask предпочтительнее.
    """

    def __init__(
        self,
        n_workers: int = 4,
        threads_per_worker: int = 2,
        memory_limit: str = "2GB",
        dashboard_address: Optional[str] = None,
    ):
        """Инициализировать Dask Backend.

        Args:
            n_workers: Количество worker процессов
            threads_per_worker: Потоков на worker
            memory_limit: Лимит памяти на worker
            dashboard_address: Адрес dashboard (None для отключения)

        """
        self.n_workers = n_workers
        self.client: Optional[Client] = None

        if DASK_AVAILABLE:
            self.cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                memory_limit=memory_limit,
                dashboard_address=dashboard_address or ":8787",
            )
            self.client = Client(self.cluster)
            logger.info(f"Dask Backend инициализирован: {n_workers} workers")

    def __del__(self):
        """Очистить ресурсы."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
        if hasattr(self, "cluster"):
            self.cluster.close()

    def train_batch(
        self,
        x_batch: List[List[float]],
        target_batch: List[float],
        lr: float = 0.01,
        n_features: Optional[int] = None,
    ) -> float:
        """Обучить на батче данных через Dask.

        Args:
            x_batch: Список входных векторов
            target_batch: Список целевых значений
            lr: Learning rate
            n_features: Размерность признаков (если None, определяется автоматически)

        Returns:
            Средняя потеря

        """
        if not DASK_AVAILABLE or self.client is None:
            # Fallback: локальное обучение
            logger.warning("Dask недоступен, используем локальное обучение")
            return self._train_local(x_batch, target_batch, lr, n_features)

        if n_features is None:
            n_features = len(x_batch[0]) if x_batch else 10

        # Создаём futures для параллельного обучения
        futures = []
        for x, target in zip(x_batch, target_batch):
            # Отправляем задачу на worker
            future = self.client.submit(self._train_single, x, target, lr, n_features, pure=False)
            futures.append(future)

        # Собираем результаты
        results = self.client.gather(futures)
        return sum(results) / len(results)

    @staticmethod
    def _train_single(x: List[float], target: float, lr: float, n_features: int) -> float:
        """Обучить на одном примере (вызывается на worker)."""
        core = KokaoCoreV9(n_features=n_features)
        x_tensor = torch.tensor(x, dtype=torch.float32)
        loss = core.train(x_tensor, target=target, lr=lr)
        return loss

    def _train_local(
        self,
        x_batch: List[List[float]],
        target_batch: List[float],
        lr: float,
        n_features: Optional[int],
    ) -> float:
        """Локальное обучение (fallback)."""
        if n_features is None:
            n_features = len(x_batch[0]) if x_batch else 10

        core = KokaoCoreV9(n_features=n_features)
        total_loss = 0.0

        for x, target in zip(x_batch, target_batch):
            x_tensor = torch.tensor(x, dtype=torch.float32)
            loss = core.train(x_tensor, target=target, lr=lr)
            total_loss += loss

        return total_loss / len(x_batch)

    def get_status(self) -> Dict[str, Any]:
        """Получить статус Dask кластера."""
        if not DASK_AVAILABLE or self.client is None:
            return {"status": "unavailable"}

        return {
            "status": "active",
            "n_workers": self.n_workers,
            "dashboard": (
                self.client.dashboard_link if hasattr(self.client, "dashboard_link") else None
            ),
        }
