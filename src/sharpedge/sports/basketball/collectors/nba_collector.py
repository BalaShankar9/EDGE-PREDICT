"""NBA data collector.

Wraps the ``nba_api`` package to download game logs, team stats, and
standings.  Falls back gracefully to an empty DataFrame when the
optional dependency is not installed.
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


class NBACollector(BaseCollector):
    """Collect NBA game results via the ``nba_api`` package."""

    source_name = "nba"
    base_url = "https://stats.nba.com"
    request_delay: float = 1.0
    cache_ttl_hours: int = 24

    def _collect(self, season: str = "2024-25", **kwargs: Any) -> pd.DataFrame:
        """Collect NBA game results for a given season.

        Tries nba_api package first, falls back to empty DataFrame.

        Parameters
        ----------
        season : str
            NBA season identifier, e.g. ``"2024-25"``.

        Returns
        -------
        pd.DataFrame with columns matching ``REQUIRED_COLUMNS``.
        """
        try:
            from nba_api.stats.endpoints import leaguegamelog

            log = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star="Regular Season",
            )
            raw = log.get_data_frames()[0]

            # Parse into our standard schema
            games: dict[str, dict] = {}
            for _, row in raw.iterrows():
                game_id = row["GAME_ID"]
                matchup = row["MATCHUP"]
                is_home = "vs." in matchup

                if game_id not in games:
                    games[game_id] = {
                        "game_date": pd.to_datetime(row["GAME_DATE"]),
                        "season": season,
                    }

                team = row["TEAM_ABBREVIATION"]
                pts = int(row["PTS"])
                if is_home:
                    games[game_id]["home_team"] = team
                    games[game_id]["home_score"] = pts
                else:
                    games[game_id]["away_team"] = team
                    games[game_id]["away_score"] = pts

            rows = []
            for g in games.values():
                if "home_team" in g and "away_team" in g:
                    g["home_win"] = int(g["home_score"] > g["away_score"])
                    rows.append(g)

            df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
            logger.info("Collected %d NBA games for season %s", len(df), season)
            return df

        except ImportError:
            logger.info("nba_api not installed, returning empty DataFrame")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        except Exception as exc:
            logger.warning("nba_api collection failed: %s", exc)
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
