import pandas as pd
import numpy as np
import pytest
from sharpedge.ml.training.walk_forward import WalkForwardCV


@pytest.fixture
def five_season_data():
    """Create matches across 5 seasons."""
    rows = []
    seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
    for season in seasons:
        for i in range(20):
            rows.append({
                "match_date": f"2025-01-{i+1:02d}",
                "home_team_id": "arsenal",
                "away_team_id": "chelsea",
                "season": season,
            })
    return pd.DataFrame(rows)


def test_walk_forward_produces_correct_splits(five_season_data):
    cv = WalkForwardCV(n_splits=2, min_train_seasons=3)
    splits = list(cv.split(five_season_data))
    assert len(splits) == 2


def test_walk_forward_train_before_val(five_season_data):
    """Training data must come from earlier seasons than validation."""
    cv = WalkForwardCV(n_splits=2, min_train_seasons=3)
    for train_idx, val_idx in cv.split(five_season_data):
        train_seasons = set(five_season_data.loc[train_idx, "season"])
        val_seasons = set(five_season_data.loc[val_idx, "season"])
        # All train seasons should be earlier than all val seasons
        assert max(train_seasons) < min(val_seasons)


def test_walk_forward_expanding_window(five_season_data):
    """Each fold should have more training data than the previous."""
    cv = WalkForwardCV(n_splits=2, min_train_seasons=3)
    train_sizes = []
    for train_idx, _ in cv.split(five_season_data):
        train_sizes.append(len(train_idx))
    # Second fold should have more training data
    assert train_sizes[1] > train_sizes[0]


def test_walk_forward_no_leakage(five_season_data):
    """No overlap between train and validation indices."""
    cv = WalkForwardCV(n_splits=2, min_train_seasons=3)
    for train_idx, val_idx in cv.split(five_season_data):
        assert len(set(train_idx) & set(val_idx)) == 0


def test_walk_forward_insufficient_seasons():
    """Should raise error with too few seasons."""
    df = pd.DataFrame({
        "season": ["2020-21"] * 10 + ["2021-22"] * 10,
        "match_date": range(20),
    })
    cv = WalkForwardCV(n_splits=2, min_train_seasons=3)
    with pytest.raises(ValueError, match="Need >3 seasons"):
        list(cv.split(df))
