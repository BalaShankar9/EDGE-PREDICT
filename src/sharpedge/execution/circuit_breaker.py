"""Drawdown Protection Circuit Breaker.

LEVEL 0 (Green):  Normal operation
LEVEL 1 (Yellow): -5% drawdown -> reduce stakes by 25%
LEVEL 2 (Orange): -10% drawdown -> reduce stakes by 50%
LEVEL 3 (Red):    -15% drawdown -> reduce stakes by 75%
LEVEL 4 (Stop):   -20% drawdown -> PAUSE ALL BETTING for 48 hours
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class CircuitBreakerStatus:
    level: int              # 0-4
    color: str              # "green", "yellow", "orange", "red", "stop"
    drawdown_pct: float
    stake_multiplier: float  # 1.0, 0.75, 0.50, 0.25, 0.0
    message: str
    paused_until: datetime | None = None


class CircuitBreaker:
    """Monitors bankroll and enforces drawdown protection."""

    LEVELS = {
        0: ("green", 1.0, "Normal operation"),
        1: ("yellow", 0.75, "Caution: -5% drawdown, stakes reduced 25%"),
        2: ("orange", 0.50, "Warning: -10% drawdown, stakes reduced 50%"),
        3: ("red", 0.25, "Danger: -15% drawdown, stakes reduced 75%"),
        4: ("stop", 0.0, "STOPPED: -20% drawdown, all betting paused 48h"),
    }

    def __init__(self):
        self._peak = 0.0
        self._current = 0.0
        self._paused_until: datetime | None = None

    def update(self, bankroll: float) -> CircuitBreakerStatus:
        self._current = bankroll
        self._peak = max(self._peak, bankroll)

        if self._peak <= 0:
            return CircuitBreakerStatus(0, "green", 0.0, 1.0, "No data")

        dd = (self._peak - self._current) / self._peak

        if dd >= 0.20:
            level = 4
            if self._paused_until is None:
                self._paused_until = datetime.now(timezone.utc) + timedelta(hours=48)
        elif dd >= 0.15:
            level = 3
        elif dd >= 0.10:
            level = 2
        elif dd >= 0.05:
            level = 1
        else:
            level = 0
            self._paused_until = None

        color, mult, msg = self.LEVELS[level]
        return CircuitBreakerStatus(level, color, dd, mult, msg, self._paused_until)

    @property
    def is_paused(self) -> bool:
        if self._paused_until is None:
            return False
        return datetime.now(timezone.utc) < self._paused_until
