"""MLB data collector.

Wraps the ``pybaseball`` package to download game results and pitcher stats.
Falls back gracefully to an empty DataFrame when the optional dependency
is not installed.
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


class MLBCollector(BaseCollector):
    """Collect MLB game results via the ``pybaseball`` package."""

    source_name = "mlb"
    base_url = "https://www.baseball-reference.com"
    request_delay: float = 1.0
    cache_ttl_hours: int = 24

    def _collect(self, season: int = 2024, **kwargs: Any) -> pd.DataFrame:
        """Collect MLB game results for a given season.

        Tries pybaseball package first, falls back to empty DataFrame.

        Parameters
        ----------
        season : int
            MLB season year, e.g. ``2024``.

        Returns
        -------
        pd.DataFrame with columns matching ``REQUIRED_COLUMNS``.
        """
        try:
            from pybaseball import schedule_and_record

            # Collect for a sample of teams to build the game database
            teams = [
                "NYY", "BOS", "LAD", "HOU", "ATL", "NYM", "CHC", "SF",
                "SD", "PHI", "SEA", "CLE", "BAL", "MIN", "TB",
            ]

            all_rows = []
            seen_games = set()

            for team in teams:
                try:
                    sched = schedule_and_record(season, team)
                    for _, game in sched.iterrows():
                        game_date = pd.to_datetime(game.get("Date"))
                        opp = game.get("Opp", "")
                        is_home = "@" not in str(game.get("Unnamed: 4", ""))

                        r = game.get("R", 0)
                        ra = game.get("RA", 0)

                        if is_home:
                            key = (game_date, team, opp)
                            if key not in seen_games:
                                seen_games.add(key)
                                all_rows.append({
                                    "game_date": game_date,
                                    "home_team": team,
                                    "away_team": opp,
                                    "home_score": int(r),
                                    "away_score": int(ra),
                                    "home_win": int(r > ra),
                                    "season": str(season),
                                })
                except Exception:
                    continue

            df = pd.DataFrame(all_rows, columns=REQUIRED_COLUMNS)
            logger.info("Collected %d MLB games for season %d", len(df), season)
            return df

        except ImportError:
            logger.info("pybaseball not installed, returning empty DataFrame")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        except Exception as exc:
            logger.warning("MLB data collection failed: %s", exc)
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
