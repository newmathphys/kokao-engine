"""Template for new module tests."""
import os
import sys

import pytest
import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_path)

from kokao.new_module import NewModule


class TestNewModule:
    """NewModule tests."""

    def test_basic(self):
        """Тест базовой функциональности."""
        module = NewModule(n_features=5)
        x = torch.randn(5, device="cpu")
        result = module.forward(x)
        assert isinstance(result, float)

    def test_edge_cases(self):
        """Тест edge cases (NaN, Inf, extreme values)."""
        module = NewModule(n_features=5)

        # NaN input
        x_nan = torch.tensor([float('nan')] * 5)
        with pytest.raises(ValueError):
            module.forward(x_nan)

        # Inf input
        x_inf = torch.tensor([float('inf')] * 5)
        with pytest.raises(ValueError):
            module.forward(x_inf)

        # Zero input
        x_zero = torch.zeros(5)
        result = module.forward(x_zero)
        assert isinstance(result, float)

    def test_extreme_values(self):
        """Тест extreme values."""
        module = NewModule(n_features=5)

        # Large values
        x_large = torch.tensor([1e5] * 5)
        result = module.forward(x_large)
        assert isinstance(result, float)

        # Small values
        x_small = torch.tensor([1e-5] * 5)
        result = module.forward(x_small)
        assert isinstance(result, float)
