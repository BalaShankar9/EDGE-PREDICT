"""
football-data.org Collector — fixtures and standings via REST API.

Fetches upcoming fixtures and current standings from the free
football-data.org API (v4). No API key needed for the basic free tier
(10 requests/minute).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# Competition IDs for the Big 5 European leagues
FOOTBALL_DATA_ORG_LEAGUES: dict[str, int] = {
    "Premier League": 2021,
    "La Liga": 2014,
    "Bundesliga": 2002,
    "Serie A": 2019,
    "Ligue 1": 2015,
}


class FootballDataOrgCollector(BaseCollector):
    """Collector for football-data.org fixtures and standings API."""

    source_name = "football_data_org"
    base_url = "https://api.football-data.org/v4"
    request_delay = 6.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect fixtures and/or standings from football-data.org.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        data_type : str, optional
            "fixtures" (default), "standings", or "both".
        """
        league: Optional[str] = kwargs.get("league")
        data_type: str = kwargs.get("data_type", "fixtures")

        leagues = (
            {league: FOOTBALL_DATA_ORG_LEAGUES[league]}
            if league
            else FOOTBALL_DATA_ORG_LEAGUES
        )

        if data_type == "both":
            fixtures_df = self._collect_fixtures(leagues)
            standings_df = self._collect_standings(leagues)
            frames = [
                df for df in [fixtures_df, standings_df] if not df.empty
            ]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames, ignore_index=True)
        elif data_type == "standings":
            return self._collect_standings(leagues)
        else:
            return self._collect_fixtures(leagues)

    def _collect_fixtures(
        self, leagues: dict[str, int]
    ) -> pd.DataFrame:
        """Fetch scheduled fixtures for given leagues."""
        frames: list[pd.DataFrame] = []

        for league_name, comp_id in leagues.items():
            url = (
                f"{self.base_url}/competitions/{comp_id}/matches"
                f"?status=SCHEDULED"
            )

            try:
                data = self._fetch_json(url)
                matches = data.get("matches", [])

                rows: list[dict] = []
                for match in matches:
                    row = self._parse_fixture(match, league_name)
                    if row:
                        rows.append(row)

                if rows:
                    df = pd.DataFrame(rows)
                    df["home_team_id"] = df["home_team"].apply(
                        lambda x: self.normalise_team(str(x))
                    )
                    df["away_team_id"] = df["away_team"].apply(
                        lambda x: self.normalise_team(str(x))
                    )
                    frames.append(df)

                logger.info(
                    f"Fetched {league_name} fixtures: {len(rows)} matches"
                )

            except Exception as e:
                logger.error(
                    f"Failed to collect {league_name} fixtures: {e}"
                )
                raise

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _collect_standings(
        self, leagues: dict[str, int]
    ) -> pd.DataFrame:
        """Fetch current standings for given leagues."""
        frames: list[pd.DataFrame] = []

        for league_name, comp_id in leagues.items():
            url = f"{self.base_url}/competitions/{comp_id}/standings"

            try:
                data = self._fetch_json(url)
                standings = data.get("standings", [])

                rows: list[dict] = []
                for standing_group in standings:
                    table = standing_group.get("table", [])
                    for entry in table:
                        row = self._parse_standing(entry, league_name)
                        if row:
                            rows.append(row)

                if rows:
                    frames.append(pd.DataFrame(rows))

                logger.info(
                    f"Fetched {league_name} standings: {len(rows)} entries"
                )

            except Exception as e:
                logger.error(
                    f"Failed to collect {league_name} standings: {e}"
                )
                raise

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _parse_fixture(match: dict, league_name: str) -> Optional[dict]:
        """Parse a single fixture from the API response.

        Parameters
        ----------
        match : dict
            Match object from the API response.
        league_name : str
            Name of the league.

        Returns
        -------
        dict or None
            Parsed fixture data, or None if parsing fails.
        """
        try:
            home_team = match.get("homeTeam", {}).get("name")
            away_team = match.get("awayTeam", {}).get("name")

            if not home_team or not away_team:
                return None

            utc_date = match.get("utcDate", "")
            matchday = match.get("matchday")

            return {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": utc_date,
                "matchday": matchday,
                "league": league_name,
                "source": "football_data_org",
                "data_type": "fixture",
                "scraped_date": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d"
                ),
            }

        except Exception:
            return None

    @staticmethod
    def _parse_standing(entry: dict, league_name: str) -> Optional[dict]:
        """Parse a single standing entry from the API response.

        Parameters
        ----------
        entry : dict
            Standing entry from the API response.
        league_name : str
            Name of the league.

        Returns
        -------
        dict or None
            Parsed standing data, or None if parsing fails.
        """
        try:
            team = entry.get("team", {})
            team_name = team.get("name")

            if not team_name:
                return None

            return {
                "team": team_name,
                "position": entry.get("position"),
                "played": entry.get("playedGames"),
                "won": entry.get("won"),
                "draw": entry.get("draw"),
                "lost": entry.get("lost"),
                "goals_for": entry.get("goalsFor"),
                "goals_against": entry.get("goalsAgainst"),
                "goal_difference": entry.get("goalDifference"),
                "points": entry.get("points"),
                "league": league_name,
                "source": "football_data_org",
                "data_type": "standing",
                "scraped_date": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%d"
                ),
            }

        except Exception:
            return None
