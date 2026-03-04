import pandas as pd
import pytest
from sharpedge.ml.features.base import FeatureGroup


class DummyFeatureGroup(FeatureGroup):
    name = "dummy"
    feature_count = 2

    def compute(self, matches, **context):
        return pd.DataFrame({
            "dummy_feat_1": [1.0] * len(matches),
            "dummy_feat_2": [2.0] * len(matches),
        }, index=matches.index)

    def get_feature_names(self):
        return ["dummy_feat_1", "dummy_feat_2"]


def test_feature_group_interface():
    fg = DummyFeatureGroup()
    matches = pd.DataFrame({
        "match_date": ["2025-01-01", "2025-01-02"],
        "home_team_id": ["arsenal", "chelsea"],
        "away_team_id": ["chelsea", "arsenal"],
    })
    result = fg.compute(matches)
    assert len(result) == 2
    assert "dummy_feat_1" in result.columns
    assert fg.feature_count == 2


def test_feature_group_names():
    fg = DummyFeatureGroup()
    names = fg.get_feature_names()
    assert len(names) == 2
