"""Hub API — 10 тестов."""

import os
import sys

import pytest

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, src_path)

# Проверка наличия модуля kokao_hub и TestClient
try:
    from starlette.testclient import TestClient

    from kokao.kokao_hub.api import app

    HAS_HUB = True
except (ImportError, ModuleNotFoundError):
    HAS_HUB = False
    app = None
    TestClient = None


@pytest.mark.skipif(not HAS_HUB, reason="kokao_hub or starlette not installed")
class TestHubAPI:
    """Hub API endpoints."""

    def test_hub_health_endpoint(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_hub_engines_list(self):
        client = TestClient(app)
        response = client.get("/engines")
        assert response.status_code == 200

    def test_hub_search_empty(self):
        client = TestClient(app)
        response = client.get("/engines/search")
        assert response.status_code == 200

    def test_hub_search_query(self):
        client = TestClient(app)
        response = client.get("/engines/search?q=test")
        assert response.status_code == 200

    def test_hub_download_not_found(self):
        client = TestClient(app)
        response = client.get("/download/nonexistent")
        assert response.status_code == 404

    def test_hub_health_response_format(self):
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "status" in data

    def test_hub_engines_format(self):
        client = TestClient(app)
        response = client.get("/engines")
        data = response.json()
        assert "engines" in data
        assert "total" in data

    def test_hub_api_title(self):
        assert app.title == "Kokao Engine API"

    def test_hub_docs_url(self):
        assert app.docs_url is not None

    def test_hub_openapi(self):
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
