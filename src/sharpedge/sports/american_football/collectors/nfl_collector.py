"""NFL data collector.

Wraps the ``nfl_data_py`` package to download game results and team stats.
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


class NFLCollector(BaseCollector):
    """Collect NFL game results via the ``nfl_data_py`` package."""

    source_name = "nfl"
    base_url = "https://www.pro-football-reference.com"
    request_delay: float = 1.0
    cache_ttl_hours: int = 24

    def _collect(self, season: int = 2024, **kwargs: Any) -> pd.DataFrame:
        """Collect NFL game results for a given season.

        Tries nfl_data_py package first, falls back to empty DataFrame.

        Parameters
        ----------
        season : int
            NFL season year, e.g. ``2024``.

        Returns
        -------
        pd.DataFrame with columns matching ``REQUIRED_COLUMNS``.
        """
        try:
            import nfl_data_py as nfl

            schedule = nfl.import_schedules([season])

            rows = []
            for _, game in schedule.iterrows():
                home_team = game.get("home_team", "")
                away_team = game.get("away_team", "")
                home_score = game.get("home_score")
                away_score = game.get("away_score")

                if (
                    home_team and away_team
                    and pd.notna(home_score) and pd.notna(away_score)
                ):
                    rows.append({
                        "game_date": pd.to_datetime(game.get("gameday")),
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": int(home_score),
                        "away_score": int(away_score),
                        "home_win": int(home_score > away_score),
                        "season": str(season),
                    })

            df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
            logger.info("Collected %d NFL games for season %d", len(df), season)
            return df

        except ImportError:
            logger.info("nfl_data_py not installed, returning empty DataFrame")
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        except Exception as exc:
            logger.warning("NFL data collection failed: %s", exc)
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
