import pandas as pd
import numpy as np
import pytest
from sharpedge.ml.training.trainer import ModelTrainer, TrainingResult


@pytest.fixture
def synthetic_match_data():
    """Create synthetic match data across 5 seasons."""
    rng = np.random.default_rng(42)
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
    teams = ["arsenal", "chelsea", "liverpool", "man_city"]
    rows = []
    for season in seasons:
        dates = pd.date_range("2025-01-01", periods=40, freq="2D")
        for i in range(40):
            h = teams[i % 4]
            a = teams[(i + 1) % 4]
            hg = int(rng.integers(0, 4))
            ag = int(rng.integers(0, 3))
            ftr = "H" if hg > ag else ("A" if ag > hg else "D")
            rows.append(
                {
                    "match_date": dates[i],
                    "home_team_id": h,
                    "away_team_id": a,
                    "FTHG": hg,
                    "FTAG": ag,
                    "FTR": ftr,
                    "season": season,
                    "league": "Premier League",
                    "B365H": round(rng.uniform(1.3, 3.5), 2),
                    "B365D": round(rng.uniform(3.0, 4.0), 2),
                    "B365A": round(rng.uniform(2.0, 6.0), 2),
                    "PSH": round(rng.uniform(1.3, 3.5), 2),
                    "PSD": round(rng.uniform(3.0, 4.0), 2),
                    "PSA": round(rng.uniform(2.0, 6.0), 2),
                    "WHH": round(rng.uniform(1.3, 3.5), 2),
                    "WHD": round(rng.uniform(3.0, 4.0), 2),
                    "WHA": round(rng.uniform(2.0, 6.0), 2),
                }
            )
    return pd.DataFrame(rows)


def test_trainer_runs_end_to_end(synthetic_match_data):
    trainer = ModelTrainer(
        xgb_params={"n_estimators": 20, "max_depth": 3},
        n_splits=2,
        min_train_seasons=3,
    )
    result = trainer.train(synthetic_match_data)

    assert isinstance(result, TrainingResult)
    assert len(result.fold_metrics) == 2
    assert "mean_rps" in result.aggregate_metrics
    assert 0 <= result.aggregate_metrics["mean_rps"] <= 1
    assert 0 <= result.aggregate_metrics["mean_accuracy"] <= 1


def test_trainer_models_are_set(synthetic_match_data):
    trainer = ModelTrainer(
        xgb_params={"n_estimators": 20, "max_depth": 3},
        n_splits=1,
        min_train_seasons=3,
    )
    trainer.train(synthetic_match_data)

    assert trainer.xgb_model is not None
    assert trainer.poisson_model is not None
    assert trainer.ensemble is not None


def test_trainer_save_load(synthetic_match_data, tmp_path):
    trainer = ModelTrainer(
        xgb_params={"n_estimators": 20, "max_depth": 3},
        n_splits=1,
        min_train_seasons=3,
    )
    trainer.train(synthetic_match_data)

    trainer.save(tmp_path / "model")
    loaded = ModelTrainer.load(tmp_path / "model")
    assert loaded.xgb_model is not None
    assert loaded.poisson_model is not None
