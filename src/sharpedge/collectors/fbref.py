"""
FBref Collector — comprehensive match stats via soccerdata library.

Wraps the soccerdata Python library which handles FBref scraping
with built-in rate limiting, caching, and data normalisation.
"""

import logging
from typing import Any, Optional

import pandas as pd
import soccerdata as sd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

FBREF_LEAGUES: dict[str, str] = {
    "Premier League": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie A": "ITA-Serie A",
    "Ligue 1": "FRA-Ligue 1",
}


class FBrefCollector(BaseCollector):
    """Collector for FBref match and team statistics."""

    source_name = "fbref"
    base_url = "https://fbref.com"
    request_delay = 4.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect match/team stats from FBref via soccerdata.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        season : str, optional
            Season string (e.g. "2024"). If None, uses soccerdata default.
        stat_type : str, optional
            Type of stats to retrieve. Options: "schedule", "shooting",
            "passing", "defense", or any stat type supported by soccerdata.
            Defaults to "schedule".
        """
        league: Optional[str] = kwargs.get("league")
        season: Optional[str] = kwargs.get("season")
        stat_type: str = kwargs.get("stat_type", "schedule")

        leagues = [FBREF_LEAGUES[league]] if league else list(FBREF_LEAGUES.values())
        seasons = [season] if season else None

        try:
            fbref = sd.FBref(leagues=leagues, seasons=seasons)

            if stat_type == "schedule":
                df = fbref.read_schedule()
            elif stat_type in ("shooting", "passing", "defense"):
                df = fbref.read_team_season_stats(stat_type=stat_type)
            else:
                df = fbref.read_team_season_stats(stat_type=stat_type)

            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            return df

        except Exception as e:
            logger.error(f"FBref collection failed: {e}")
            raise
