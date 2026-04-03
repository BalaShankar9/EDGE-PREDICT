"""OddsAPICollector — stub for The Odds API data source."""

from typing import Any

import pandas as pd

from sharpedge.collectors.base import BaseCollector


class OddsAPICollector(BaseCollector):
    """Collector for The Odds API (https://api.the-odds-api.com)."""

    source_name = "odds_api"
    base_url = "https://api.the-odds-api.com"

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "match_id",
                "home_team",
                "away_team",
                "bookmaker",
                "market",
                "odds_home",
                "odds_draw",
                "odds_away",
                "captured_at",
            ]
        )
