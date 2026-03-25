"""Tests for the AgentTracker — prediction recording, resolution, and stats."""
import numpy as np
import pytest

from sharpedge.agents.base_agent import AgentPrediction
from sharpedge.agents.tracker import AgentTracker


def _make_prediction(
    agent_name: str = "agent_a",
    match_id: str = "match_1",
    predicted_outcome: str = "home",
    probabilities: tuple[float, ...] = (0.6, 0.2, 0.2),
    confidence: float = 0.6,
) -> AgentPrediction:
    """Helper to create an AgentPrediction for testing."""
    outcomes = ("home", "draw", "away")
    return AgentPrediction(
        agent_name=agent_name,
        sport="football",
        match_id=match_id,
        market="1x2",
        outcomes=outcomes,
        probabilities=np.array(probabilities),
        predicted_outcome=predicted_outcome,
        confidence=confidence,
        uncertainty=0.1,
        reasoning="test reasoning",
    )


class TestRecordPrediction:
    """Tests for record_prediction and record_batch."""

    def test_record_prediction_stores_in_buffer(self):
        tracker = AgentTracker()
        pred = _make_prediction()
        tracker.record_prediction(pred)
        assert tracker.pending_count == 1

    def test_record_batch_stores_multiple(self):
        tracker = AgentTracker()
        preds = [
            _make_prediction(agent_name="agent_a", match_id="m1"),
            _make_prediction(agent_name="agent_b", match_id="m1"),
            _make_prediction(agent_name="agent_a", match_id="m2"),
        ]
        tracker.record_batch(preds)
        assert tracker.pending_count == 3


class TestResolve:
    """Tests for resolve()."""

    def test_resolve_returns_correct_and_incorrect(self):
        tracker = AgentTracker()
        tracker.record_prediction(_make_prediction(agent_name="good", predicted_outcome="home"))
        tracker.record_prediction(_make_prediction(agent_name="bad", predicted_outcome="away"))
        results = tracker.resolve("match_1", "home")
        assert results == {"good": True, "bad": False}

    def test_resolve_removes_matched_from_pending(self):
        tracker = AgentTracker()
        tracker.record_prediction(_make_prediction(match_id="m1"))
        tracker.record_prediction(_make_prediction(match_id="m2"))
        assert tracker.pending_count == 2
        tracker.resolve("m1", "home")
        assert tracker.pending_count == 1

    def test_multiple_matches_resolved_independently(self):
        tracker = AgentTracker()
        tracker.record_prediction(_make_prediction(agent_name="a", match_id="m1", predicted_outcome="home"))
        tracker.record_prediction(_make_prediction(agent_name="a", match_id="m2", predicted_outcome="away"))

        r1 = tracker.resolve("m1", "home")
        assert r1 == {"a": True}

        r2 = tracker.resolve("m2", "draw")
        assert r2 == {"a": False}

        assert tracker.pending_count == 0
        assert tracker.resolved_count == 2


class TestGetAgentStats:
    """Tests for get_agent_stats()."""

    def test_no_history_returns_defaults(self):
        tracker = AgentTracker()
        stats = tracker.get_agent_stats("nonexistent")
        assert stats["total_bets"] == 0
        assert stats["wins"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["avg_confidence"] == 0.0
        assert stats["brier_score"] == 1.0

    def test_stats_with_history_computes_win_rate(self):
        tracker = AgentTracker()
        # 3 correct, 1 incorrect -> 75% win rate
        for i in range(3):
            tracker.record_prediction(
                _make_prediction(agent_name="agent", match_id=f"win_{i}", predicted_outcome="home")
            )
        tracker.record_prediction(
            _make_prediction(agent_name="agent", match_id="loss_0", predicted_outcome="away")
        )
        for i in range(3):
            tracker.resolve(f"win_{i}", "home")
        tracker.resolve("loss_0", "home")

        stats = tracker.get_agent_stats("agent")
        assert stats["total_bets"] == 4
        assert stats["wins"] == 3
        assert stats["win_rate"] == 0.75

    def test_stats_with_last_n(self):
        tracker = AgentTracker()
        # First 3 are losses, last 2 are wins
        for i in range(3):
            tracker.record_prediction(
                _make_prediction(agent_name="agent", match_id=f"loss_{i}", predicted_outcome="away")
            )
        for i in range(2):
            tracker.record_prediction(
                _make_prediction(agent_name="agent", match_id=f"win_{i}", predicted_outcome="home")
            )
        for i in range(3):
            tracker.resolve(f"loss_{i}", "home")
        for i in range(2):
            tracker.resolve(f"win_{i}", "home")

        # All 5: 2/5 = 40%
        all_stats = tracker.get_agent_stats("agent")
        assert all_stats["total_bets"] == 5
        assert all_stats["win_rate"] == 0.4

        # Last 2 only: 2/2 = 100%
        recent_stats = tracker.get_agent_stats("agent", last_n=2)
        assert recent_stats["total_bets"] == 2
        assert recent_stats["win_rate"] == 1.0

    def test_brier_score_computed_correctly(self):
        tracker = AgentTracker()
        # Prediction: home=0.8, draw=0.1, away=0.1; actual: home
        # Brier = (1.0 - 0.8)^2 = 0.04
        tracker.record_prediction(
            _make_prediction(
                agent_name="agent", match_id="m1",
                predicted_outcome="home",
                probabilities=(0.8, 0.1, 0.1),
                confidence=0.8,
            )
        )
        tracker.resolve("m1", "home")
        stats = tracker.get_agent_stats("agent")
        assert stats["brier_score"] == pytest.approx(0.04, abs=1e-4)

    def test_brier_score_wrong_prediction(self):
        tracker = AgentTracker()
        # Prediction: home=0.7, draw=0.2, away=0.1; actual: away
        # Brier = (1.0 - 0.1)^2 = 0.81
        tracker.record_prediction(
            _make_prediction(
                agent_name="agent", match_id="m1",
                predicted_outcome="home",
                probabilities=(0.7, 0.2, 0.1),
                confidence=0.7,
            )
        )
        tracker.resolve("m1", "away")
        stats = tracker.get_agent_stats("agent")
        assert stats["brier_score"] == pytest.approx(0.81, abs=1e-4)


class TestLeaderboard:
    """Tests for get_leaderboard()."""

    def _populate_tracker(self, tracker: AgentTracker, agent_name: str, n_wins: int, n_total: int):
        """Helper: create n_total predictions, n_wins correct."""
        for i in range(n_wins):
            tracker.record_prediction(
                _make_prediction(agent_name=agent_name, match_id=f"{agent_name}_w{i}", predicted_outcome="home")
            )
            tracker.resolve(f"{agent_name}_w{i}", "home")
        for i in range(n_total - n_wins):
            tracker.record_prediction(
                _make_prediction(agent_name=agent_name, match_id=f"{agent_name}_l{i}", predicted_outcome="away")
            )
            tracker.resolve(f"{agent_name}_l{i}", "home")

    def test_leaderboard_ranked_by_win_rate(self):
        tracker = AgentTracker()
        self._populate_tracker(tracker, "strong", n_wins=9, n_total=10)
        self._populate_tracker(tracker, "weak", n_wins=6, n_total=10)
        self._populate_tracker(tracker, "medium", n_wins=7, n_total=10)

        board = tracker.get_leaderboard(min_bets=10)
        names = [e["agent_name"] for e in board]
        assert names == ["strong", "medium", "weak"]

    def test_leaderboard_filters_by_min_bets(self):
        tracker = AgentTracker()
        self._populate_tracker(tracker, "enough", n_wins=8, n_total=10)
        self._populate_tracker(tracker, "not_enough", n_wins=4, n_total=5)

        board = tracker.get_leaderboard(min_bets=10)
        names = [e["agent_name"] for e in board]
        assert "enough" in names
        assert "not_enough" not in names


class TestCounts:
    """Tests for pending_count and resolved_count."""

    def test_counts_track_correctly(self):
        tracker = AgentTracker()
        assert tracker.pending_count == 0
        assert tracker.resolved_count == 0

        tracker.record_prediction(_make_prediction(match_id="m1"))
        tracker.record_prediction(_make_prediction(match_id="m2"))
        assert tracker.pending_count == 2
        assert tracker.resolved_count == 0

        tracker.resolve("m1", "home")
        assert tracker.pending_count == 1
        assert tracker.resolved_count == 1

        tracker.resolve("m2", "away")
        assert tracker.pending_count == 0
        assert tracker.resolved_count == 2
