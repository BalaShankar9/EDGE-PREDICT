"""Loads raw data from collectors into DataFrames for feature engineering.

This is the bridge between Phase 1 (collectors) and Phase 2 (ML).
It loads data from collectors and merges into a unified match DataFrame.
"""
import logging
import pandas as pd
from sharpedge.collectors.football_data_uk import FootballDataUKCollector
from sharpedge.collectors.club_elo import ClubELOCollector
from sharpedge.collectors.understat import UnderstatCollector
from sharpedge.collectors.forebet import ForebetCollector

logger = logging.getLogger(__name__)


def load_historical_matches(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
) -> pd.DataFrame:
    """Load historical match data from Football-Data.co.uk."""
    collector = FootballDataUKCollector()
    kwargs = {}
    if leagues and len(leagues) == 1:
        kwargs["league"] = leagues[0]
    if seasons and len(seasons) == 1:
        kwargs["season"] = seasons[0]
    df = collector.collect(**kwargs)

    if leagues and len(leagues) > 1:
        df = df[df["league"].isin(leagues)]
    if seasons and len(seasons) > 1:
        df = df[df["season"].isin(seasons)]

    df["match_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.sort_values("match_date").reset_index(drop=True)
    return df


def load_elo_ratings(
    date_range: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Load ELO ratings from ClubELO."""
    collector = ClubELOCollector()
    if date_range:
        return collector.collect(date_range=date_range)
    return collector.collect()


def load_xg_data(
    leagues: list[str] | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Load xG data from Understat."""
    collector = UnderstatCollector()
    kwargs = {}
    if leagues and len(leagues) == 1:
        kwargs["league"] = leagues[0]
    if seasons and len(seasons) == 1:
        kwargs["season"] = seasons[0]
    return collector.collect(**kwargs)


def load_competitor_predictions() -> pd.DataFrame:
    """Load predictions from Forebet (and others as available)."""
    collector = ForebetCollector()
    return collector.collect()
