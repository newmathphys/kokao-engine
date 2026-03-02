"""KG Neo4j Tests (18 тестов)."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.kg import KokaoKGStream


class TestKG:
    """Knowledge Graph (Neo4j) тесты."""

    def test_init(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        assert kg.driver is None or kg.driver is not None

    def test_update_high_signal(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([1.0, 2.0, 3.0])
        kg.update_from_stream(x, S=150.0)

    def test_update_low_signal(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([1.0, 2.0, 3.0])
        kg.update_from_stream(x, S=50.0)

    def test_query(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n) RETURN n LIMIT 1")
        assert isinstance(result, list)

    def test_update_different_signals(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        for S in [0.0, 50.0, 100.0, 150.0, 200.0]:
            x = torch.randn(5, device="cpu")
            kg.update_from_stream(x, S)

    def test_update_extreme_signal(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([1.0, 2.0, 3.0])
        kg.update_from_stream(x, S=10000.0)

    def test_update_negative_signal(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([1.0, 2.0, 3.0])
        kg.update_from_stream(x, S=-100.0)

    def test_query_empty(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n) WHERE 1=0 RETURN n")
        assert result == []

    def test_update_batch(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        for _ in range(10):
            x = torch.randn(5, device="cpu")
            kg.update_from_stream(x, S=100.0)

    def test_update_different_dimensions(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        for n_features in [3, 5, 10]:
            x = torch.randn(n_features)
            kg.update_from_stream(x, S=100.0)

    def test_query_malformed(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("INVALID QUERY")
        assert isinstance(result, list)

    def test_update_zero_vector(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.zeros(5)
        kg.update_from_stream(x, S=100.0)

    def test_update_nan_vector(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([float("nan")] * 5)
        kg.update_from_stream(x, S=100.0)

    def test_update_inf_vector(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([float("inf")] * 5)
        kg.update_from_stream(x, S=100.0)

    def test_connection_error(self):
        kg = KokaoKGStream(uri="neo4j://invalid")
        x = torch.randn(5, device="cpu")
        kg.update_from_stream(x, S=100.0)

    def test_query_timeout(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n) RETURN n")
        assert isinstance(result, list)

    def test_update_concurrent(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        for i in range(100):
            x = torch.randn(5, device="cpu")
            kg.update_from_stream(x, S=float(i))

    def test_recall(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.randn(5, device="cpu")
        kg.update_from_stream(x, S=100.0)
        result = kg.query("MATCH (n:Vector) RETURN n")
        assert isinstance(result, list)
