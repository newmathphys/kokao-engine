"""RL Tests (8 тестов)."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.models import RLAction, RLPolicyConfig, RLReward, RLState
from kokao.rl import KokaoPolicy, KokaoRLAgent


class TestRL:
    """Reinforcement Learning тесты."""

    def test_policy_init(self):
        policy = KokaoPolicy(n_features=5, n_actions=3)
        assert policy.kokao is not None
        assert policy.action_head is not None

    def test_policy_forward(self):
        policy = KokaoPolicy(n_features=5, n_actions=3)
        x = torch.randn(5, device="cpu")
        out = policy(x)
        assert out.shape == (3,)

    def test_policy_get_action(self):
        policy = KokaoPolicy(n_features=5, n_actions=3)
        x = torch.randn(5, device="cpu")
        action = policy.get_action(x)
        assert action in [0, 1, 2]

    def test_agent_init(self):
        agent = KokaoRLAgent(n_features=5, n_actions=3)
        assert agent.policy is not None
        assert agent.optimizer is not None

    def test_agent_act(self):
        agent = KokaoRLAgent(n_features=5, n_actions=3)
        state = torch.randn(5, device="cpu")
        action = agent.act(state)
        assert action in [0, 1, 2]

    def test_agent_train_step(self):
        agent = KokaoRLAgent(n_features=5, n_actions=3)
        state = torch.randn(5, device="cpu")
        action = agent.act(state)
        reward = 1.0
        loss = agent.train_step(state, action, reward)
        assert isinstance(loss, float)

    def test_rl_policy_config(self):
        config = RLPolicyConfig(n_features=5, n_actions=3, learning_rate=0.001)
        assert config.n_features == 5
        assert config.n_actions == 3

    def test_rl_state_action_reward(self):
        state = RLState(features=[0.1, 0.2, 0.3])
        action = RLAction(action_type="EXPLORE", value=0.5)
        reward = RLReward(value=10.0, is_terminal=False)
        assert len(state.features) == 3
        assert action.value == 0.5
        assert reward.value == 10.0
