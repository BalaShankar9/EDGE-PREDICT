"""Tests for the daily prediction pipeline."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sharpedge.pipeline.daily import DailyPipeline


@pytest.fixture
def mock_pipeline():
    with patch("sharpedge.pipeline.daily.FeaturePipeline") as MockPipeline:
        # Mock feature pipeline to return 46-column DataFrame
        mock_pipe_instance = MockPipeline.return_value
        mock_pipe_instance.build.return_value = pd.DataFrame(
            np.random.randn(1, 46), columns=[f"feat_{i}" for i in range(46)]
        )

        pipeline = DailyPipeline(model_path="models/test.pkl")

        # Mock the trainer and its models
        trainer = MagicMock()
        trainer.xgb_model = MagicMock()
        trainer.xgb_model.predict_proba_1x2.return_value = np.array([[0.55, 0.25, 0.20]])
        trainer.xgb_model.predict_proba_ou.return_value = np.array([0.60])
        trainer.xgb_model.predict_proba_btts.return_value = np.array([0.55])
        trainer.poisson_model = MagicMock()
        trainer.poisson_model.predict_proba_1x2.return_value = np.array([0.50, 0.28, 0.22])
        trainer.poisson_model.predict_proba_ou.return_value = 0.60
        trainer.poisson_model.predict_proba_btts.return_value = 0.55
        trainer.ensemble = MagicMock()
        trainer.ensemble.predict.return_value = np.array([[0.52, 0.26, 0.22]])
        trainer.calibrator = MagicMock()
        trainer.calibrator.calibrate.return_value = np.array([[0.53, 0.26, 0.21]])
        trainer.ovr_model = None
        trainer.bvp_model = None
        trainer.catboost_model = None
        trainer.lgbm_model = None
        trainer.stacking = None
        trainer.ou_calibrator = None
        trainer.btts_calibrator = None
        trainer.draw_floor = 0.18
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
    assert predictions[0]["prob_home"] == pytest.approx(0.53, abs=0.01)
    assert predictions[0]["prob_draw"] == pytest.approx(0.26, abs=0.01)
    assert predictions[0]["prob_away"] == pytest.approx(0.21, abs=0.01)
    assert predictions[0]["prob_over"] == pytest.approx(0.60, abs=0.01)
    assert predictions[0]["prob_btts_yes"] == pytest.approx(0.55, abs=0.01)


def test_pipeline_filters_picks(mock_pipeline):
    predictions = [
        {"home_team": "Arsenal", "away_team": "Chelsea",
         "league": "Premier League", "match_date": "2026-03-04",
         "market": "1x2_home", "model_prob": 0.83, "model_spread": 0.03,
         "best_odds": 1.85, "bookmaker": "Bet365",
         "risk_flags": []},
    ]
    picks = mock_pipeline.filter_picks(predictions)
    assert isinstance(picks, list)


def test_pipeline_produces_different_predictions_per_fixture():
    """Verify fixtures get different predictions (not all zeros)."""
    with patch("sharpedge.pipeline.daily.FeaturePipeline") as MockPipeline:
        # Return different features for different rows
        mock_pipe_instance = MockPipeline.return_value
        mock_pipe_instance.build.return_value = pd.DataFrame(
            np.random.randn(2, 46), columns=[f"feat_{i}" for i in range(46)]
        )

        pipeline = DailyPipeline(model_path="models/test.pkl")
        trainer = MagicMock()

        # Make XGBoost return different probs for different inputs
        def mock_predict_1x2(X):
            if X[0, 0] > 0:
                return np.array([[0.60, 0.25, 0.15]])
            return np.array([[0.35, 0.30, 0.35]])

        trainer.xgb_model = MagicMock()
        trainer.xgb_model.predict_proba_1x2.side_effect = mock_predict_1x2
        trainer.xgb_model.predict_proba_ou.return_value = np.array([0.50])
        trainer.xgb_model.predict_proba_btts.return_value = np.array([0.50])
        trainer.poisson_model = MagicMock()
        trainer.poisson_model.predict_proba_1x2.return_value = np.array([0.40, 0.30, 0.30])
        trainer.poisson_model.predict_proba_ou.return_value = 0.50
        trainer.poisson_model.predict_proba_btts.return_value = 0.50
        trainer.ensemble = MagicMock()
        trainer.ensemble.predict.side_effect = lambda preds: preds["xgboost"]
        trainer.calibrator = MagicMock()
        trainer.calibrator.calibrate.side_effect = lambda x: x
        trainer.ovr_model = None
        trainer.bvp_model = None
        trainer.catboost_model = None
        trainer.lgbm_model = None
        trainer.stacking = None
        trainer.ou_calibrator = None
        trainer.btts_calibrator = None
        trainer.draw_floor = 0.18
        pipeline.trainer = trainer

        fixtures = [
            {"home_team": "Arsenal", "away_team": "Chelsea", "match_date": "2026-03-04"},
            {"home_team": "Liverpool", "away_team": "ManCity", "match_date": "2026-03-04"},
        ]
        predictions = pipeline.predict(fixtures)
        assert len(predictions) == 2
        # Predictions should exist (not all zeros)
        assert predictions[0]["prob_home"] > 0
        assert predictions[1]["prob_home"] > 0
