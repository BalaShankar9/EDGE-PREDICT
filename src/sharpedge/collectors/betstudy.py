"""BetStudyCollector — stub for BetStudy prediction data source."""

from typing import Any

import pandas as pd

from sharpedge.collectors.base import BaseCollector


class BetStudyCollector(BaseCollector):
    """Collector for BetStudy (https://www.betstudy.com)."""

    source_name = "betstudy"
    base_url = "https://www.betstudy.com"

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
