"""Tests for the daily prediction pipeline."""
import pytest
from unittest.mock import MagicMock, patch
from sharpedge.pipeline.daily import DailyPipeline


@pytest.fixture
def mock_pipeline():
    with patch("sharpedge.pipeline.daily.ModelTrainer") as MockTrainer:
        trainer = MockTrainer.load.return_value
        trainer.xgb_model = MagicMock()
        trainer.xgb_model.predict_proba_1x2.return_value = [[0.55, 0.25, 0.20]]
        trainer.xgb_model.predict_proba_ou.return_value = [0.60]
        trainer.xgb_model.predict_proba_btts.return_value = [0.55]
        trainer.poisson_model = MagicMock()
        trainer.poisson_model.predict_proba_1x2.return_value = [0.50, 0.28, 0.22]
        trainer.ensemble = MagicMock()
        trainer.ensemble.predict.return_value = [[0.52, 0.26, 0.22]]
        trainer.calibrator = MagicMock()
        trainer.calibrator.calibrate.return_value = [[0.53, 0.26, 0.21]]
        trainer.feature_pipeline = MagicMock()
        pipeline = DailyPipeline(model_path="models/test.pkl")
        pipeline.trainer = trainer
        yield pipeline


def test_pipeline_creates_predictions(mock_pipeline):
    fixtures = [
        {"home_team": "Arsenal", "away_team": "Chelsea",
         "league": "Premier League", "match_date": "2026-03-04",
         "B365H": 1.85, "B365D": 3.40, "B365A": 4.20},
    ]
    predictions = mock_pipeline.predict(fixtures)
    assert len(predictions) == 1
    assert "prob_home" in predictions[0]
    assert "prob_draw" in predictions[0]
    assert "prob_away" in predictions[0]


def test_pipeline_filters_picks(mock_pipeline):
    predictions = [
        {"home_team": "Arsenal", "away_team": "Chelsea",
         "league": "Premier League", "match_date": "2026-03-04",
         "market": "1x2_home", "model_prob": 0.83, "model_spread": 0.03,
         "best_odds": 1.85, "bookmaker": "Bet365",
         "meta_agreement": 3, "risk_flags": []},
    ]
    picks = mock_pipeline.filter_picks(predictions)
    assert isinstance(picks, list)
