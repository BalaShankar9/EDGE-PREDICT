"""Integration test for expanded backfill — verifies all 10 collectors are called."""
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from sharpedge.orchestrator import run_backfill

@pytest.fixture
def mock_all_collectors():
    collectors = [
        "sharpedge.orchestrator.FootballDataUKCollector",
        "sharpedge.orchestrator.ClubELOCollector",
        "sharpedge.orchestrator.UnderstatCollector",
        "sharpedge.orchestrator.FBrefCollector",
        "sharpedge.orchestrator.ForebetCollector",
        "sharpedge.orchestrator.PredictZCollector",
        "sharpedge.orchestrator.WinDrawWinCollector",
        "sharpedge.orchestrator.FootyStatsCollector",
        "sharpedge.orchestrator.FootballDataOrgCollector",
        "sharpedge.orchestrator.OpenMeteoCollector",
    ]
    mocks = {}
    patches = []
    for collector_path in collectors:
        name = collector_path.split(".")[-1]
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.source_name = name.lower()
        mock_instance.collect.return_value = pd.DataFrame({"col": [1, 2, 3]})
        mock_cls.return_value = mock_instance
        p = patch(collector_path, mock_cls)
        p.start()
        patches.append(p)
        mocks[name] = mock_instance
    yield mocks
    for p in patches:
        p.stop()

def test_backfill_calls_all_10_collectors(mock_all_collectors):
    results = run_backfill(seasons=["2024-25"])
    assert len(results) == 10
    for mock in mock_all_collectors.values():
        mock.collect.assert_called()

def test_backfill_multiple_seasons(mock_all_collectors):
    results = run_backfill(seasons=["2023-24", "2024-25"])
    assert len(results) == 20
