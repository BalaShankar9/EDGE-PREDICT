"""Tests for the agents API endpoints."""
from fastapi.testclient import TestClient
from sharpedge.api.app import create_app
from sharpedge.api.routers.agents import set_tracker
from sharpedge.agents.tracker import AgentTracker


def _client():
    app = create_app()
    return TestClient(app)


def test_list_agents():
    client = _client()
    response = client.get("/api/agents")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["meta"]["count"] == 9
    assert len(data["data"]) == 9


def test_list_agents_has_required_fields():
    client = _client()
    response = client.get("/api/agents")
    data = response.json()
    for agent in data["data"]:
        assert "name" in agent
        assert "type" in agent
        assert "description" in agent


def test_agent_leaderboard_no_tracker():
    set_tracker(None)
    client = _client()
    response = client.get("/api/agents/leaderboard")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data"] == []


def test_agent_leaderboard_with_tracker():
    tracker = AgentTracker()
    set_tracker(tracker)
    try:
        client = _client()
        response = client.get("/api/agents/leaderboard")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert isinstance(data["data"], list)
    finally:
        set_tracker(None)


def test_get_agent_stats_no_tracker():
    set_tracker(None)
    client = _client()
    response = client.get("/api/agents/statistical_agent")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data"]["agent_name"] == "statistical_agent"
    assert data["data"]["total_bets"] == 0


def test_get_agent_stats_with_tracker():
    tracker = AgentTracker()
    set_tracker(tracker)
    try:
        client = _client()
        response = client.get("/api/agents/gradient_agent")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["agent_name"] == "gradient_agent"
        assert data["data"]["total_bets"] == 0
    finally:
        set_tracker(None)
