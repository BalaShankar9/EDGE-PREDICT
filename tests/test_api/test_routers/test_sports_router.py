"""Tests for the sports API endpoints."""
from fastapi.testclient import TestClient
from sharpedge.api.app import create_app


def _client():
    app = create_app()
    return TestClient(app)


def test_list_sports():
    client = _client()
    response = client.get("/api/sports")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["meta"]["count"] >= 1  # at least football
    slugs = [s["slug"] for s in data["data"]]
    assert "football" in slugs


def test_list_sports_has_markets():
    client = _client()
    response = client.get("/api/sports")
    data = response.json()
    for sport in data["data"]:
        assert "markets" in sport
        assert "n_markets" in sport
        assert sport["n_markets"] == len(sport["markets"])


def test_get_sport_football():
    client = _client()
    response = client.get("/api/sports/football")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["data"]["name"] == "Football"
    assert data["data"]["slug"] == "football"
    assert "markets" in data["data"]
    assert "min_train_seasons" in data["data"]
    assert "default_mc_sims" in data["data"]


def test_get_sport_unknown():
    client = _client()
    response = client.get("/api/sports/curling")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "curling" in data["message"]


def test_all_six_sports_registered():
    client = _client()
    response = client.get("/api/sports")
    data = response.json()
    slugs = {s["slug"] for s in data["data"]}
    expected = {"football", "tennis", "basketball", "ice_hockey", "american_football", "baseball"}
    assert expected.issubset(slugs), f"Missing sports: {expected - slugs}"
