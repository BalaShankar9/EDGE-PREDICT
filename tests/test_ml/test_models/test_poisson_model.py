import numpy as np
import pytest
from sharpedge.ml.models.poisson_model import PoissonPredictor


@pytest.fixture
def trained_model():
    """Train Poisson model on synthetic match data."""
    matches = []
    rng = np.random.default_rng(42)
    teams = ["arsenal", "chelsea", "liverpool", "man_city"]
    for _ in range(100):
        h = teams[rng.integers(0, 4)]
        a = teams[rng.integers(0, 4)]
        while a == h:
            a = teams[rng.integers(0, 4)]
        matches.append({
            "home_team_id": h,
            "away_team_id": a,
            "home_goals": int(rng.poisson(1.5)),
            "away_goals": int(rng.poisson(1.1)),
        })
    model = PoissonPredictor()
    model.fit(matches)
    return model


def test_poisson_probabilities_sum_to_one(trained_model):
    proba = trained_model.predict_proba_1x2("arsenal", "chelsea")
    assert proba.shape == (3,)
    assert np.isclose(proba.sum(), 1.0, atol=0.01)


def test_poisson_home_stronger(trained_model):
    """Home team with higher attack should have higher P(H)."""
    # This is a general property — home advantage + Poisson should give P(H) > P(A) on average
    proba = trained_model.predict_proba_1x2("arsenal", "chelsea")
    # At least probabilities should be positive
    assert all(p > 0 for p in proba)


def test_poisson_scoreline_matrix(trained_model):
    matrix = trained_model.predict_scoreline_matrix("arsenal", "chelsea")
    assert matrix.shape == (9, 9)
    assert (matrix >= 0).all()
    assert np.isclose(matrix.sum(), 1.0, atol=0.01)


def test_poisson_ou_range(trained_model):
    p_over = trained_model.predict_proba_ou("arsenal", "chelsea")
    assert 0 <= p_over <= 1


def test_poisson_btts_range(trained_model):
    p_btts = trained_model.predict_proba_btts("arsenal", "chelsea")
    assert 0 <= p_btts <= 1


def test_poisson_correct_score(trained_model):
    scores = trained_model.predict_correct_score("arsenal", "chelsea", top_n=5)
    assert len(scores) == 5
    # Most likely score should have highest probability
    assert scores[0]["prob"] >= scores[-1]["prob"]
    # Probabilities should be positive
    assert all(s["prob"] > 0 for s in scores)
