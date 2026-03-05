"""
Understat Collector — shot-level xG data from inline JSON.

Scrapes expected goals (xG) data from understat.com, which embeds match
data as JSON inside JavaScript variables on league pages.
"""

import io
import json
import logging
import re
from typing import Any, Optional

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# Big 5 league codes used by understat.com
LEAGUE_CODES: dict[str, str] = {
    "Premier League": "EPL",
    "La Liga": "La_liga",
    "Bundesliga": "Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}

# Seasons available (year = the starting year of the season)
DEFAULT_SEASONS = list(range(2020, 2026))  # 2020 through 2025

# Regex to extract the datesData JSON blob from the page
_DATES_DATA_RE = re.compile(r"var\s+datesData\s*=\s*JSON\.parse\('(.+?)'\)")


class UnderstatCollector(BaseCollector):
    """Collector for Understat xG match data."""

    source_name = "understat"
    base_url = "https://understat.com"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect xG match data from Understat.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        season : int, optional
            Season starting year (e.g. 2024). If None, collects 2020-2025.
        """
        league: Optional[str] = kwargs.get("league")
        season_raw = kwargs.get("season")

        leagues = {league: LEAGUE_CODES[league]} if league else LEAGUE_CODES

        # Convert season label "2023-24" → starting year 2023
        if season_raw is not None:
            if isinstance(season_raw, str) and "-" in season_raw:
                season_int = int(season_raw.split("-")[0])
            else:
                season_int = int(season_raw)
            seasons = [season_int]
        else:
            seasons = DEFAULT_SEASONS

        frames: list[pd.DataFrame] = []

        for league_name, league_code in leagues.items():
            for season_year in seasons:
                cache_key = f"understat_{league_code}_{season_year}"
                cached = self._get_cached(cache_key)

                if cached is not None:
                    df = pd.DataFrame(cached)
                    logger.debug(
                        f"Cache hit: {league_name} {season_year} "
                        f"({len(df)} rows)"
                    )
                else:
                    url = (
                        f"{self.base_url}/league/"
                        f"{league_code}/{season_year}"
                    )
                    response = self._fetch(url)
                    html = response.text

                    # Structural fingerprinting
                    self._check_structure(
                        html, ["datesData", "teamsData"]
                    )

                    # Extract JSON from JavaScript
                    match = _DATES_DATA_RE.search(html)
                    if not match:
                        logger.warning(
                            f"No datesData found for {league_name} "
                            f"{season_year}"
                        )
                        continue

                    # Decode hex escapes and parse JSON
                    raw = match.group(1)
                    decoded = raw.encode().decode("unicode_escape")
                    matches_data = json.loads(decoded)

                    # Build rows from match data
                    rows = []
                    for m in matches_data:
                        is_result = m.get("isResult", False)
                        rows.append(
                            {
                                "match_id_understat": m.get("id"),
                                "date": m.get("datetime"),
                                "home_team": m.get("h", {}).get("title"),
                                "away_team": m.get("a", {}).get("title"),
                                "home_goals": m.get("goals", {}).get("h"),
                                "away_goals": m.get("goals", {}).get("a"),
                                "home_xg": m.get("xG", {}).get("h"),
                                "away_xg": m.get("xG", {}).get("a"),
                                "league": league_name,
                                "season": season_year,
                                "is_result": is_result,
                            }
                        )

                    df = pd.DataFrame(rows)

                    # Normalise team names
                    if not df.empty:
                        df["home_team_id"] = df["home_team"].apply(
                            lambda x: self.normalise_team(str(x))
                        )
                        df["away_team_id"] = df["away_team"].apply(
                            lambda x: self.normalise_team(str(x))
                        )

                    # Cache the parsed data
                    self._set_cache(cache_key, df.to_dict(orient="list"))

                    logger.info(
                        f"Fetched {league_name} {season_year}: "
                        f"{len(df)} matches"
                    )

                frames.append(df)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)
