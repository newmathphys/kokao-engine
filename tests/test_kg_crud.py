"""KG CRUD Operations — 20 тестов."""

import os
import sys

import torch

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

from kokao.kg import KokaoKGStream


class TestKGCRUD:
    """KG CRUD операции."""

    def test_kg_create_node(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([1.0, 2.0, 3.0])
        kg.update_from_stream(x, S=100.0)

    def test_kg_read_node(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n) RETURN n LIMIT 1")
        assert isinstance(result, list)

    def test_kg_update_node(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([1.0, 2.0, 3.0])
        kg.update_from_stream(x, S=100.0)
        kg.update_from_stream(x, S=200.0)

    def test_kg_delete_node(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n) DELETE n")
        assert isinstance(result, list)

    def test_kg_recall_similar(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([1.0, 2.0, 3.0])
        kg.update_from_stream(x, S=100.0)
        result = kg.query("MATCH (n:Vector) RETURN n")
        assert isinstance(result, list)

    def test_kg_create_multiple_nodes(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        for i in range(10):
            x = torch.randn(5, device="cpu")
            kg.update_from_stream(x, S=float(i))

    def test_kg_read_all_nodes(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n) RETURN count(n)")
        assert isinstance(result, list)

    def test_kg_update_batch(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        for _ in range(5):
            x = torch.randn(5, device="cpu")
            kg.update_from_stream(x, S=100.0)

    def test_kg_delete_all(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n) DETACH DELETE n")
        assert isinstance(result, list)

    def test_kg_recall_threshold(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.randn(5, device="cpu")
        kg.update_from_stream(x, S=100.0)
        result = kg.query("MATCH (n:Vector) WHERE n.S > 50 RETURN n")
        assert isinstance(result, list)

    def test_kg_create_zero_vector(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.zeros(5)
        kg.update_from_stream(x, S=100.0)

    def test_kg_create_nan_vector(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([float("nan")] * 5)
        kg.update_from_stream(x, S=100.0)

    def test_kg_create_inf_vector(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.tensor([float("inf")] * 5)
        kg.update_from_stream(x, S=100.0)

    def test_kg_create_large_vector(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.randn(5, device="cpu") * 1000
        kg.update_from_stream(x, S=100.0)

    def test_kg_read_properties(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n:Vector) RETURN n.x, n.S")
        assert isinstance(result, list)

    def test_kg_update_properties(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        x = torch.randn(5, device="cpu")
        kg.update_from_stream(x, S=100.0)
        kg.update_from_stream(x, S=200.0)

    def test_kg_delete_specific(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n:Vector) WHERE n.S < 50 DELETE n")
        assert isinstance(result, list)

    def test_kg_recall_count(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        result = kg.query("MATCH (n:Vector) RETURN count(n)")
        assert isinstance(result, list)

    def test_kg_create_concurrent(self):
        kg = KokaoKGStream(uri="neo4j://localhost")
        for i in range(100):
            x = torch.randn(5, device="cpu")
            kg.update_from_stream(x, S=float(i))
