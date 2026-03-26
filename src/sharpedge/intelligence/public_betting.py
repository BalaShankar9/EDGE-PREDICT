"""Public Betting Tracker — fade the public.

When 75% of public bets go one way but the line moves the OTHER way,
sharp money is on the opposite side.
"""
from dataclasses import dataclass


@dataclass
class PublicBettingData:
    match_id: str
    public_pct: dict[str, float]  # {"home": 0.75, "draw": 0.10, "away": 0.15}
    ticket_count: int = 0


class PublicBettingTracker:
    def __init__(self):
        self._data: dict[str, PublicBettingData] = {}

    def record(self, data: PublicBettingData) -> None:
        self._data[data.match_id] = data

    def compute_features(self, match_id: str) -> dict[str, float]:
        data = self._data.get(match_id)
        if data is None:
            return {
                "pb_public_pct_home": 0.5, "pb_public_pct_away": 0.5,
                "pb_public_heavy_side": 0.0, "pb_contrarian_signal": 0.0,
            }

        home_pct = data.public_pct.get("home", 0.5)
        away_pct = data.public_pct.get("away", 0.5)
        heavy = max(data.public_pct.values())

        return {
            "pb_public_pct_home": home_pct,
            "pb_public_pct_away": away_pct,
            "pb_public_heavy_side": heavy,
            "pb_contrarian_signal": 1.0 if heavy > 0.70 else 0.0,
        }
