"""RL Policy — 8 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.models import RLAction, RLPolicyConfig, RLReward, RLState
from kokao.rl import KokaoPolicy, KokaoRLAgent


class TestRLPolicy:
    """RL Policy тесты."""

    def test_rl_policy_init(self):
        policy = KokaoPolicy(n_features=5, n_actions=3)
        assert policy.kokao is not None

    def test_rl_policy_forward(self):
        policy = KokaoPolicy(n_features=5, n_actions=3)
        x = torch.randn(5, device="cpu")
        out = policy(x)
        assert out.shape == (3,)

    def test_rl_policy_get_action(self):
        policy = KokaoPolicy(n_features=5, n_actions=3)
        x = torch.randn(5, device="cpu")
        action = policy.get_action(x)
        assert action in [0, 1, 2]

    def test_rl_agent_init(self):
        agent = KokaoRLAgent(n_features=5, n_actions=3)
        assert agent.policy is not None

    def test_rl_agent_act(self):
        agent = KokaoRLAgent(n_features=5, n_actions=3)
        state = torch.randn(5, device="cpu")
        action = agent.act(state)
        assert action in [0, 1, 2]

    def test_rl_agent_train_step(self):
        agent = KokaoRLAgent(n_features=5, n_actions=3)
        state = torch.randn(5, device="cpu")
        action = agent.act(state)
        reward = 1.0
        loss = agent.train_step(state, action, reward)
        assert isinstance(loss, float)

    def test_rl_policy_config(self):
        config = RLPolicyConfig(n_features=5, n_actions=3)
        assert config.n_features == 5

    def test_rl_state_action_reward(self):
        state = RLState(features=[0.1, 0.2, 0.3])
        action = RLAction(action_type="EXPLORE", value=0.5)
        reward = RLReward(value=10.0, is_terminal=False)
        assert len(state.features) == 3
