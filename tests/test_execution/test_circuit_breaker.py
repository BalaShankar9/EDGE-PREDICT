"""Tests for CircuitBreaker."""
import pytest
from datetime import datetime, timedelta, timezone
from sharpedge.execution.circuit_breaker import CircuitBreaker, CircuitBreakerStatus


class TestLevels:
    def test_level_0_green(self):
        cb = CircuitBreaker()
        status = cb.update(1000.0)
        assert status.level == 0
        assert status.color == "green"
        assert status.stake_multiplier == 1.0

    def test_level_1_at_5pct(self):
        cb = CircuitBreaker()
        cb.update(1000.0)  # set peak
        status = cb.update(950.0)
        assert status.level == 1
        assert status.color == "yellow"
        assert status.stake_multiplier == 0.75

    def test_level_2_at_10pct(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        status = cb.update(900.0)
        assert status.level == 2
        assert status.color == "orange"
        assert status.stake_multiplier == 0.50

    def test_level_3_at_15pct(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        status = cb.update(850.0)
        assert status.level == 3
        assert status.color == "red"
        assert status.stake_multiplier == 0.25

    def test_level_4_at_20pct(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        status = cb.update(800.0)
        assert status.level == 4
        assert status.color == "stop"
        assert status.stake_multiplier == 0.0

    def test_drawdown_pct_correct(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        status = cb.update(880.0)
        assert pytest.approx(status.drawdown_pct, abs=1e-6) == 0.12


class TestPause:
    def test_paused_at_level_4(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        cb.update(800.0)
        assert cb.is_paused is True

    def test_not_paused_at_level_3(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        cb.update(850.0)
        assert cb.is_paused is False

    def test_paused_until_is_future(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        status = cb.update(800.0)
        assert status.paused_until is not None
        assert status.paused_until > datetime.now(timezone.utc)

    def test_pause_clears_on_recovery(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        cb.update(800.0)  # triggers pause
        cb._paused_until = datetime.now(timezone.utc) - timedelta(hours=1)  # simulate expiry
        assert cb.is_paused is False

    def test_recovery_to_green_clears_pause(self):
        cb = CircuitBreaker()
        cb.update(1000.0)
        cb.update(800.0)  # level 4
        # Recovery above peak clears
        status = cb.update(1050.0)
        assert status.level == 0
        assert status.paused_until is None


class TestEdgeCases:
    def test_zero_peak_returns_green(self):
        cb = CircuitBreaker()
        status = cb.update(0.0)
        assert status.level == 0

    def test_peak_tracks_maximum(self):
        cb = CircuitBreaker()
        cb.update(500.0)
        cb.update(1000.0)
        cb.update(800.0)
        assert cb._peak == 1000.0

    def test_status_is_dataclass(self):
        cb = CircuitBreaker()
        status = cb.update(1000.0)
        assert isinstance(status, CircuitBreakerStatus)
