"""Тесты для Pydantic Models (15 тестов)."""

import os
import sys

import pytest

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.models import (QuantumCircuitConfig, QuantumResponse,
                          RayActorConfig, RayTrainRequest, RLAction,
                          RLPolicyConfig, RLReward, RLState)


class TestRayModels:
    """Ray модели."""

    def test_ray_actor_config_valid(self):
        config = RayActorConfig(n_features=5, n_actors=10)
        assert config.n_features == 5
        assert config.n_actors == 10

    def test_ray_actor_config_invalid(self):
        with pytest.raises(ValueError):
            RayActorConfig(n_features=0)

    def test_ray_train_request_valid(self):
        req = RayTrainRequest(
            x_batch=[[0.1, 0.2]], target_batch=[1.0], learning_rate=0.05, epochs=10
        )
        assert len(req.x_batch) == 1
        assert req.learning_rate == 0.05


class TestQuantumModels:
    """Quantum модели."""

    def test_quantum_circuit_config_valid(self):
        config = QuantumCircuitConfig(n_qubits=4, depth=5)
        assert config.n_qubits == 4
        assert config.depth == 5

    def test_quantum_circuit_config_invalid(self):
        with pytest.raises(ValueError):
            QuantumCircuitConfig(n_qubits=0)

    def test_quantum_response_valid(self):
        response = QuantumResponse(success=True, counts={"00": 512, "11": 512}, probability=0.5)
        assert response.success is True


class TestRLModels:
    """RL модели."""

    def test_rl_state_valid(self):
        state = RLState(features=[0.1, 0.2, 0.3])
        assert len(state.features) == 3

    def test_rl_action_valid(self):
        action = RLAction(action_type="EXPLORE", value=0.5)
        assert action.action_type == "EXPLORE"
        assert action.value == 0.5

    def test_rl_action_invalid(self):
        with pytest.raises(ValueError):
            RLAction(action_type="EXPLORE", value=1.5)

    def test_rl_reward_valid(self):
        reward = RLReward(value=10.0, is_terminal=False)
        assert reward.value == 10.0

    def test_rl_policy_config_valid(self):
        config = RLPolicyConfig(n_features=5, n_actions=3)
        assert config.n_features == 5
        assert config.n_actions == 3
