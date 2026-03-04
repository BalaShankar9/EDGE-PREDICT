"""Tests for pipeline router endpoints."""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sharpedge.api.app import create_app
from sharpedge.api.deps import get_db
from sharpedge.db.models import Base
from sharpedge.config import settings


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    settings.api_key = "test-key-123"
    app = create_app()
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key-123"}


def test_run_pipeline_success(client, auth_headers):
    with patch("sharpedge.api.routers.pipeline.DailyPipeline") as MockPipeline:
        mock = MockPipeline.return_value
        mock.predict.return_value = [{"prob_home": 0.5}]
        mock.filter_picks.return_value = []
        response = client.post("/api/pipeline/run", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_pipeline_status(client, auth_headers):
    response = client.get("/api/pipeline/status", headers=auth_headers)
    assert response.status_code == 200


def test_resolve_endpoint(client, auth_headers):
    response = client.post("/api/pipeline/resolve", headers=auth_headers)
    assert response.status_code == 200
