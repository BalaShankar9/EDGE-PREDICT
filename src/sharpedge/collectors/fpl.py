"""FPLCollector — stub for Fantasy Premier League API data source."""

from typing import Any

import pandas as pd

from sharpedge.collectors.base import BaseCollector


class FPLCollector(BaseCollector):
    """Collector for Fantasy Premier League API (https://fantasy.premierleague.com/api)."""

    source_name = "fpl"
    base_url = "https://fantasy.premierleague.com/api"

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "player_name",
                "team",
                "status",
                "chance_of_playing",
                "news",
            ]
        )
