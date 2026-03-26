"""Tests for the Health Monitor."""
from sharpedge.warroom.health_monitor import (
    HealthMonitor,
    HealthStatus,
)


class TestHealthMonitor:
    def test_register_check(self):
        monitor = HealthMonitor()
        monitor.register_check("db", lambda: (HealthStatus.HEALTHY, "OK"))
        assert len(monitor._checks) == 1

    def test_run_checks_healthy(self):
        monitor = HealthMonitor()
        monitor.register_check("db", lambda: (HealthStatus.HEALTHY, "Connected"))
        monitor.register_check("api", lambda: (HealthStatus.HEALTHY, "Responding"))
        result = monitor.run_checks()
        assert result.overall == HealthStatus.HEALTHY
        assert len(result.components) == 2
        assert all(c.status == HealthStatus.HEALTHY for c in result.components)

    def test_run_checks_down(self):
        monitor = HealthMonitor()
        monitor.register_check("db", lambda: (HealthStatus.DOWN, "Connection refused"))
        result = monitor.run_checks()
        assert result.overall == HealthStatus.DOWN
        assert result.components[0].status == HealthStatus.DOWN
        assert result.components[0].message == "Connection refused"

    def test_run_checks_degraded(self):
        monitor = HealthMonitor()
        monitor.register_check("db", lambda: (HealthStatus.HEALTHY, "OK"))
        monitor.register_check("cache", lambda: (HealthStatus.DEGRADED, "Slow"))
        result = monitor.run_checks()
        assert result.overall == HealthStatus.DEGRADED

    def test_mixed_health_down_trumps_degraded(self):
        monitor = HealthMonitor()
        monitor.register_check("db", lambda: (HealthStatus.DOWN, "Dead"))
        monitor.register_check("cache", lambda: (HealthStatus.DEGRADED, "Slow"))
        monitor.register_check("api", lambda: (HealthStatus.HEALTHY, "OK"))
        result = monitor.run_checks()
        assert result.overall == HealthStatus.DOWN

    def test_exception_in_check_reported_as_down(self):
        def bad_check():
            raise RuntimeError("check exploded")

        monitor = HealthMonitor()
        monitor.register_check("unstable", bad_check)
        result = monitor.run_checks()
        assert result.overall == HealthStatus.DOWN
        assert result.components[0].status == HealthStatus.DOWN
        assert "check exploded" in result.components[0].message

    def test_empty_checks_returns_healthy(self):
        monitor = HealthMonitor()
        result = monitor.run_checks()
        assert result.overall == HealthStatus.HEALTHY
        assert result.components == []
