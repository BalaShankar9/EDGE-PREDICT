"""
Football-Data.co.uk Collector — CSV match results and bookmaker odds.

Downloads CSV files containing historical match data for the Big 5 European
leagues, including full-time/half-time scores, match statistics, and
bookmaker odds from multiple providers.
"""

import io
import logging
from typing import Any, Optional

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# Big 5 European league codes used by football-data.co.uk
LEAGUE_CODES: dict[str, str] = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
}

# Season label -> URL code mapping (last 5 seasons)
SEASON_CODES: dict[str, str] = {
    "2024-25": "2425",
    "2023-24": "2324",
    "2022-23": "2223",
    "2021-22": "2122",
    "2020-21": "2021",
}

# Columns we want to keep from the CSV (if available)
_KEEP_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG",
    "Referee",
    "HS", "AS", "HST", "AST",
    "HF", "AF", "HC", "AC",
    "HY", "AY", "HR", "AR",
    # Bookmaker odds
    "B365H", "B365D", "B365A",
    "PSH", "PSD", "PSA",
    "WHH", "WHD", "WHA",
    "MaxH", "MaxD", "MaxA",
    "AvgH", "AvgD", "AvgA",
]


class FootballDataUKCollector(BaseCollector):
    """Collector for football-data.co.uk CSV match data."""

    source_name = "football_data_uk"
    base_url = "https://www.football-data.co.uk"
    request_delay = 1.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect match data from football-data.co.uk.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        season : str, optional
            Season label (e.g. "2024-25"). If None, collects all 5 seasons.
        """
        league: Optional[str] = kwargs.get("league")
        season: Optional[str] = kwargs.get("season")

        leagues = {league: LEAGUE_CODES[league]} if league else LEAGUE_CODES
        seasons = {season: SEASON_CODES[season]} if season else SEASON_CODES

        frames: list[pd.DataFrame] = []

        for league_name, league_code in leagues.items():
            for season_label, season_code in seasons.items():
                cache_key = f"fduk_{league_code}_{season_code}"
                cached = self._get_cached(cache_key)

                if cached is not None:
                    df = pd.DataFrame(cached)
                    logger.debug(
                        f"Cache hit: {league_name} {season_label} "
                        f"({len(df)} rows)"
                    )
                else:
                    url = (
                        f"{self.base_url}/mmz4281/"
                        f"{season_code}/{league_code}.csv"
                    )
                    response = self._fetch(url)
                    df = pd.read_csv(io.StringIO(response.text))

                    # Keep only available columns from our desired list
                    available = [c for c in _KEEP_COLUMNS if c in df.columns]
                    df = df[available]

                    # Add metadata columns
                    df["league"] = league_name
                    df["season"] = season_label

                    # Normalise team names
                    df["home_team_id"] = df["HomeTeam"].apply(
                        lambda x: self.normalise_team(str(x))
                    )
                    df["away_team_id"] = df["AwayTeam"].apply(
                        lambda x: self.normalise_team(str(x))
                    )

                    # Cache the parsed data
                    self._set_cache(cache_key, df.to_dict(orient="list"))

                    logger.info(
                        f"Fetched {league_name} {season_label}: "
                        f"{len(df)} matches"
                    )

                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)
