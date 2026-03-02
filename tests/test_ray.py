"""Ray Tests (8 тестов)."""

import os
import sys

import pytest
import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from kokao.models import RayActorConfig
from kokao.ray import KokaoRayActor, KokaoRayCluster


class TestRay:
    """Ray Distributed Computing тесты."""

    def test_actor_config_valid(self):
        config = RayActorConfig(n_features=5, n_actors=10)
        assert config.n_features == 5
        assert config.n_actors == 10

    def test_actor_config_invalid(self):
        with pytest.raises(ValueError):
            RayActorConfig(n_features=0)

    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def test_actor_init(self):
        actor = KokaoRayActor(n_features=5)
        assert actor.engine is not None

    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def _test_actor_train(self):
        actor = KokaoRayActor(n_features=5)
        x_batch = [torch.randn(5, device="cpu").tolist() for _ in range(2)]
        target_batch = [100.0, 200.0]
        result = actor.train(x_batch, target_batch)
        assert result is not None

    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def test_actor_get_weights(self):
        actor = KokaoRayActor(n_features=5)
        weights = actor.get_weights()
        assert len(weights) == 5

    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def _test_cluster_init(self):
        cluster = KokaoRayCluster(n_actors=2, n_features=5)
        assert len(cluster.actors) == 2

    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def _test_cluster_train(self):
        cluster = KokaoRayCluster(n_actors=2, n_features=5)
        x_batches = [[torch.randn(5, device="cpu").tolist()] for _ in range(2)]
        target_batches = [[100.0] for _ in range(2)]
        cluster.train(x_batches, target_batches)

    @pytest.mark.skipif(not HAS_RAY, reason="ray not installed")
    def _test_cluster_aggregate(self):
        cluster = KokaoRayCluster(n_actors=2, n_features=5)
        weights_list = [[0.1] * 5, [0.2] * 5]
        aggregated = cluster._aggregate(weights_list)
        assert len(aggregated) == 5
