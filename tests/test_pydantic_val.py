"""Pydantic Validation — 20 тестов."""

import os
import sys

import pytest

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.models import (QuantumCircuitConfig, QuantumResponse,
                          RayActorConfig, RayTrainRequest, RayTrainResponse,
                          RLAction, RLPolicyConfig, RLReward, RLState)


class TestPydanticVal:
    """Pydantic валидация."""

    def test_ray_config_valid(self):
        config = RayActorConfig(n_features=5, n_actors=10)
        assert config.n_features == 5
        assert config.n_actors == 10

    def test_ray_config_min_features(self):
        config = RayActorConfig(n_features=1)
        assert config.n_features == 1

    def test_ray_config_max_actors(self):
        config = RayActorConfig(n_features=5, n_actors=1000)
        assert config.n_actors == 1000

    def test_ray_config_invalid_features(self):
        with pytest.raises(ValueError):
            RayActorConfig(n_features=0)

    def test_ray_config_invalid_actors(self):
        with pytest.raises(ValueError):
            RayActorConfig(n_features=5, n_actors=1001)

    def test_ray_request_valid(self):
        req = RayTrainRequest(
            x_batch=[[0.1, 0.2]], target_batch=[1.0], learning_rate=0.01, epochs=10
        )
        assert req.learning_rate == 0.01

    def test_ray_request_invalid_lr(self):
        with pytest.raises(ValueError):
            RayTrainRequest(x_batch=[[0.1]], target_batch=[1.0], learning_rate=1.5)

    def test_ray_request_invalid_epochs(self):
        with pytest.raises(ValueError):
            RayTrainRequest(x_batch=[[0.1]], target_batch=[1.0], epochs=101)

    def test_ray_response_valid(self):
        resp = RayTrainResponse(
            success=True, loss=0.01, new_weights=[0.1] * 5, metrics={"acc": 0.9}
        )
        assert resp.success is True

    def test_quantum_config_valid(self):
        config = QuantumCircuitConfig(n_qubits=4, depth=5)
        assert config.n_qubits == 4

    def test_quantum_config_min_qubits(self):
        config = QuantumCircuitConfig(n_qubits=1)
        assert config.n_qubits == 1

    def test_quantum_config_max_qubits(self):
        config = QuantumCircuitConfig(n_qubits=100)
        assert config.n_qubits == 100

    def test_quantum_config_invalid_qubits(self):
        with pytest.raises(ValueError):
            QuantumCircuitConfig(n_qubits=0)

    def test_quantum_response_valid(self):
        resp = QuantumResponse(success=True, counts={"00": 512}, probability=0.5)
        assert resp.success is True

    def test_rl_state_valid(self):
        state = RLState(features=[0.1, 0.2, 0.3])
        assert len(state.features) == 3

    def test_rl_action_valid(self):
        action = RLAction(action_type="EXPLORE", value=0.5)
        assert action.value == 0.5

    def test_rl_action_invalid_value(self):
        with pytest.raises(ValueError):
            RLAction(action_type="EXPLORE", value=1.5)

    def test_rl_reward_valid(self):
        reward = RLReward(value=10.0, is_terminal=False)
        assert reward.value == 10.0

    def test_rl_policy_config_valid(self):
        config = RLPolicyConfig(n_features=5, n_actions=3)
        assert config.n_features == 5

    def test_rl_policy_config_invalid_lr(self):
        with pytest.raises(ValueError):
            RLPolicyConfig(n_features=5, n_actions=3, learning_rate=1.5)
