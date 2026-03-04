"""Tests for API authentication."""
import pytest
from fastapi.testclient import TestClient
from sharpedge.api.app import create_app
from sharpedge.config import settings


@pytest.fixture
def client():
    settings.api_key = "test-key-123"
    app = create_app()
    return TestClient(app)


def test_health_endpoint_is_public(client):
    response = client.get("/api/health")
    assert response.status_code == 200


def test_pipeline_endpoint_requires_api_key(client):
    response = client.post("/api/pipeline/run")
    assert response.status_code == 401


def test_pipeline_endpoint_rejects_wrong_key(client):
    response = client.post("/api/pipeline/run", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401
