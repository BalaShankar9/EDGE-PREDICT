"""NHL data collector.

Wraps the NHL Stats API (free, public) to download game results and
team stats.  Falls back gracefully to an empty DataFrame when the
API is unavailable.
"""

import logging
from typing import Any

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "home_win",
    "season",
]


class NHLCollector(BaseCollector):
    """Collect NHL game results via the NHL Stats API."""

    source_name = "nhl"
    base_url = "https://statsapi.web.nhl.com/api/v1"
    request_delay: float = 1.0
    cache_ttl_hours: int = 24

    def _collect(self, season: str = "20242025", **kwargs: Any) -> pd.DataFrame:
        """Collect NHL game results for a given season.

        Tries the NHL Stats API first, falls back to empty DataFrame.

        Parameters
        ----------
        season : str
            NHL season identifier, e.g. ``"20242025"``.

        Returns
        -------
        pd.DataFrame with columns matching ``REQUIRED_COLUMNS``.
        """
        try:
            url = f"{self.base_url}/schedule?season={season}&gameType=R"
            data = self._fetch_json(url)

            rows = []
            for date_entry in data.get("dates", []):
                game_date = pd.to_datetime(date_entry["date"])
                for game in date_entry.get("games", []):
                    teams = game.get("teams", {})
                    home_info = teams.get("home", {})
                    away_info = teams.get("away", {})

                    home_team = home_info.get("team", {}).get("abbreviation", "")
                    away_team = away_info.get("team", {}).get("abbreviation", "")
                    home_score = home_info.get("score", 0)
                    away_score = away_info.get("score", 0)

                    if home_team and away_team:
                        rows.append({
                            "game_date": game_date,
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_score": int(home_score),
                            "away_score": int(away_score),
                            "home_win": int(home_score > away_score),
                            "season": season,
                        })

            df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
            logger.info("Collected %d NHL games for season %s", len(df), season)
            return df

        except Exception as exc:
            logger.warning("NHL API collection failed: %s", exc)
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
