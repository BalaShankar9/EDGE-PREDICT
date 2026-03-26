"""Tests for the Auto-Retrainer."""
from datetime import datetime, timedelta, timezone

from sharpedge.warroom.retrainer import AutoRetrainer


class TestAutoRetrainer:
    def test_register_sport_creates_schedule(self):
        rt = AutoRetrainer()
        rt.register_sport("football")
        assert "football" in rt.schedules
        assert rt.schedules["football"].frequency_days == 30

    def test_register_sport_custom_frequency(self):
        rt = AutoRetrainer()
        rt.register_sport("football", frequency_days=7)
        assert rt.schedules["football"].frequency_days == 7

    def test_register_sport_default_for_unknown(self):
        rt = AutoRetrainer()
        rt.register_sport("cricket")
        assert rt.schedules["cricket"].frequency_days == 30  # fallback default

    def test_check_due_never_trained(self):
        rt = AutoRetrainer()
        rt.register_sport("football")
        assert rt.check_due("football") is True

    def test_check_due_recently_trained(self):
        rt = AutoRetrainer()
        rt.register_sport("football")
        rt.mark_trained("football")
        assert rt.check_due("football") is False

    def test_check_due_after_frequency_elapsed(self):
        rt = AutoRetrainer()
        rt.register_sport("football", frequency_days=30)
        # Manually set last_trained to 31 days ago
        rt._schedules["football"].last_trained = datetime.now(timezone.utc) - timedelta(days=31)
        assert rt.check_due("football") is True

    def test_check_due_unknown_sport(self):
        rt = AutoRetrainer()
        assert rt.check_due("unknown") is False

    def test_check_emergency_poor_performance(self):
        rt = AutoRetrainer()
        assert rt.check_emergency("football", win_rate=0.40, n_bets=60) is True

    def test_check_emergency_above_threshold(self):
        rt = AutoRetrainer()
        assert rt.check_emergency("football", win_rate=0.55, n_bets=100) is False

    def test_check_emergency_small_sample(self):
        rt = AutoRetrainer()
        assert rt.check_emergency("football", win_rate=0.30, n_bets=10) is False

    def test_check_emergency_at_threshold(self):
        rt = AutoRetrainer()
        # Exactly at threshold — should NOT trigger (must be strictly below)
        assert rt.check_emergency("football", win_rate=0.47, n_bets=50) is False

    def test_mark_trained_updates_schedule(self):
        rt = AutoRetrainer()
        rt.register_sport("tennis")
        rt.mark_trained("tennis")
        schedule = rt.schedules["tennis"]
        assert schedule.last_trained is not None
        assert schedule.next_due is not None
        assert schedule.next_due > schedule.last_trained

    def test_mark_trained_sets_correct_next_due(self):
        rt = AutoRetrainer()
        rt.register_sport("tennis")  # 14-day frequency
        rt.mark_trained("tennis")
        schedule = rt.schedules["tennis"]
        delta = (schedule.next_due - schedule.last_trained).days
        assert delta == 14

    def test_get_due_sports(self):
        rt = AutoRetrainer()
        rt.register_sport("football")
        rt.register_sport("tennis")
        rt.mark_trained("tennis")
        # football never trained -> due; tennis just trained -> not due
        due = rt.get_due_sports()
        assert "football" in due
        assert "tennis" not in due

    def test_retrain_history_recorded(self):
        rt = AutoRetrainer()
        rt.register_sport("football")
        rt.mark_trained("football")
        assert len(rt._retrain_history) == 1
        assert rt._retrain_history[0]["sport"] == "football"

    def test_default_frequencies(self):
        rt = AutoRetrainer()
        assert rt.DEFAULT_FREQUENCIES["tennis"] == 14
        assert rt.DEFAULT_FREQUENCIES["american_football"] == 60
