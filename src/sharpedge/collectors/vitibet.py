"""VitibetCollector — stub for Vitibet prediction data source."""

from typing import Any

import pandas as pd

from sharpedge.collectors.base import BaseCollector


class VitibetCollector(BaseCollector):
    """Collector for Vitibet (https://www.vitibet.com)."""

    source_name = "vitibet"
    base_url = "https://www.vitibet.com"

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "home_team",
                "away_team",
                "match_date",
                "predicted_result",
                "prob_home",
                "prob_draw",
                "prob_away",
            ]
        )
