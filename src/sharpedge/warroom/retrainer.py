"""Automatic Model Retraining Scheduler.

Rules:
- Football: retrain monthly
- Tennis: retrain bi-weekly
- Basketball/NHL: retrain monthly
- NFL: retrain at mid-season and end-of-season
- MLB: retrain monthly
- Emergency retrain: if win rate drops below 47% for 50+ bets
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@dataclass
class RetrainSchedule:
    sport: str
    frequency_days: int
    last_trained: datetime | None
    next_due: datetime | None
    emergency: bool = False


class AutoRetrainer:
    """Manages automatic model retraining schedules."""

    DEFAULT_FREQUENCIES = {
        "football": 30,
        "tennis": 14,
        "basketball": 30,
        "ice_hockey": 30,
        "american_football": 60,  # NFL has fewer games
        "baseball": 30,
    }

    EMERGENCY_WIN_RATE_THRESHOLD = 0.47
    EMERGENCY_MIN_BETS = 50

    def __init__(self):
        self._schedules: dict[str, RetrainSchedule] = {}
        self._retrain_history: list[dict] = []

    def register_sport(self, sport: str, frequency_days: int | None = None) -> None:
        freq = frequency_days or self.DEFAULT_FREQUENCIES.get(sport, 30)
        self._schedules[sport] = RetrainSchedule(
            sport=sport, frequency_days=freq,
            last_trained=None, next_due=None,
        )

    def check_due(self, sport: str) -> bool:
        """Check if a sport's models are due for retraining."""
        schedule = self._schedules.get(sport)
        if schedule is None:
            return False
        if schedule.last_trained is None:
            return True  # never trained

        now = datetime.now(timezone.utc)
        elapsed = (now - schedule.last_trained).days
        return elapsed >= schedule.frequency_days

    def check_emergency(self, sport: str, win_rate: float, n_bets: int) -> bool:
        """Check if emergency retrain is needed based on poor performance."""
        if n_bets < self.EMERGENCY_MIN_BETS:
            return False
        return win_rate < self.EMERGENCY_WIN_RATE_THRESHOLD

    def mark_trained(self, sport: str) -> None:
        """Mark a sport's models as freshly trained."""
        now = datetime.now(timezone.utc)
        if sport in self._schedules:
            schedule = self._schedules[sport]
            schedule.last_trained = now
            schedule.next_due = now + timedelta(days=schedule.frequency_days)

        self._retrain_history.append({
            "sport": sport,
            "timestamp": now.isoformat(),
        })
        logger.info(f"Marked {sport} models as retrained")

    def get_due_sports(self) -> list[str]:
        """Return list of sports that need retraining."""
        return [sport for sport in self._schedules if self.check_due(sport)]

    @property
    def schedules(self) -> dict[str, RetrainSchedule]:
        return self._schedules.copy()
