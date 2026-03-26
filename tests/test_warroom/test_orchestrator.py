"""Tests for the War Room orchestrator."""
from datetime import datetime, timezone

from sharpedge.warroom.orchestrator import (
    DailyReport,
    PipelineResult,
    WarRoomOrchestrator,
)


class TestPipelineResult:
    def test_has_expected_fields(self):
        r = PipelineResult(
            sport="football",
            timestamp=datetime.now(timezone.utc),
            matches_processed=10,
            predictions_generated=8,
            picks_produced=3,
            agents_used=4,
        )
        assert r.sport == "football"
        assert r.matches_processed == 10
        assert r.predictions_generated == 8
        assert r.picks_produced == 3
        assert r.agents_used == 4
        assert r.errors == []

    def test_errors_default_empty(self):
        r = PipelineResult(
            sport="tennis",
            timestamp=datetime.now(timezone.utc),
            matches_processed=0,
            predictions_generated=0,
            picks_produced=0,
            agents_used=0,
        )
        assert r.errors == []

    def test_errors_can_be_set(self):
        r = PipelineResult(
            sport="tennis",
            timestamp=datetime.now(timezone.utc),
            matches_processed=0,
            predictions_generated=0,
            picks_produced=0,
            agents_used=0,
            errors=["something broke"],
        )
        assert r.errors == ["something broke"]


class TestDailyReport:
    def test_has_expected_fields(self):
        report = DailyReport(
            date="2026-03-26",
            sports_processed=["football", "tennis"],
            total_matches=20,
            total_predictions=15,
            total_picks=5,
            results=[],
            circuit_breaker_level=0,
            bankroll=1000.0,
        )
        assert report.date == "2026-03-26"
        assert report.sports_processed == ["football", "tennis"]
        assert report.total_matches == 20
        assert report.total_predictions == 15
        assert report.total_picks == 5
        assert report.circuit_breaker_level == 0
        assert report.bankroll == 1000.0


class TestWarRoomOrchestrator:
    def test_register_sport_pipeline(self):
        orch = WarRoomOrchestrator()
        orch.register_sport_pipeline("football", {"agents": [1, 2, 3]})
        assert "football" in orch.get_active_sports()

    def test_get_active_sports_empty(self):
        orch = WarRoomOrchestrator()
        assert orch.get_active_sports() == []

    def test_get_active_sports_returns_registered(self):
        orch = WarRoomOrchestrator()
        orch.register_sport_pipeline("football", {"agents": []})
        orch.register_sport_pipeline("tennis", {"agents": []})
        sports = orch.get_active_sports()
        assert "football" in sports
        assert "tennis" in sports
        assert len(sports) == 2

    def test_run_sport_pipeline_unknown_sport(self):
        orch = WarRoomOrchestrator()
        result = orch.run_sport_pipeline("curling")
        assert result.sport == "curling"
        assert result.matches_processed == 0
        assert len(result.errors) == 1
        assert "No pipeline registered" in result.errors[0]

    def test_run_sport_pipeline_registered(self):
        orch = WarRoomOrchestrator()
        orch.register_sport_pipeline("football", {"agents": ["a1", "a2"]})
        result = orch.run_sport_pipeline("football")
        assert result.sport == "football"
        assert result.agents_used == 2
        assert result.errors == []

    def test_run_daily_processes_all_sports(self):
        orch = WarRoomOrchestrator()
        orch.register_sport_pipeline("football", {"agents": ["a1"]})
        orch.register_sport_pipeline("tennis", {"agents": ["a2", "a3"]})
        report = orch.run_daily()
        assert len(report.sports_processed) == 2
        assert "football" in report.sports_processed
        assert "tennis" in report.sports_processed
        assert len(report.results) == 2

    def test_run_daily_empty(self):
        orch = WarRoomOrchestrator()
        report = orch.run_daily()
        assert report.sports_processed == []
        assert report.total_matches == 0
        assert report.total_picks == 0

    def test_run_daily_accumulates(self):
        orch = WarRoomOrchestrator()
        orch.register_sport_pipeline("football", {"agents": []})
        orch.run_daily()
        orch.run_daily()
        assert len(orch._daily_results) == 2
