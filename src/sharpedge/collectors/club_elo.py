"""
ClubELO Collector — daily ELO ratings via CSV API.

Fetches club ELO ratings from api.clubelo.com, which provides both
per-date snapshots (all clubs for a given date) and full history for
individual teams.
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class ClubELOCollector(BaseCollector):
    """Collector for ClubELO rating data."""

    source_name = "club_elo"
    base_url = "http://api.clubelo.com"
    request_delay = 1.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect ELO rating data from ClubELO API.

        Parameters
        ----------
        team : str, optional
            Team name to fetch full history for (e.g. "Arsenal").
        date_str : str, optional
            Date string (YYYY-MM-DD) to fetch all clubs for that date.
        date_range : tuple[str, str], optional
            Start and end date strings for weekly snapshots.
        """
        team: Optional[str] = kwargs.get("team")
        date_str: Optional[str] = kwargs.get("date_str")
        date_range: Optional[tuple[str, str]] = kwargs.get("date_range")

        if team:
            return self._fetch_team_history(team)
        elif date_range:
            return self._fetch_date_range(date_range[0], date_range[1])
        elif date_str:
            return self._fetch_date(date_str)
        else:
            # Default: today's date
            today = datetime.now().strftime("%Y-%m-%d")
            return self._fetch_date(today)

    def _fetch_date(self, date_str: str) -> pd.DataFrame:
        """Fetch ELO ratings for all clubs on a given date."""
        cache_key = f"elo_date_{date_str}"
        cached = self._get_cached(cache_key)

        if cached is not None:
            return pd.DataFrame(cached)

        url = f"{self.base_url}/{date_str}"
        response = self._fetch(url)
        df = pd.read_csv(io.StringIO(response.text))

        # Normalise team names
        if "Club" in df.columns:
            df["team_id"] = df["Club"].apply(
                lambda x: self.normalise_team(str(x))
            )
        elif "club" in df.columns:
            df["team_id"] = df["club"].apply(
                lambda x: self.normalise_team(str(x))
            )

        df["snapshot_date"] = date_str
        self._set_cache(cache_key, df.to_dict(orient="list"))

        logger.info(f"Fetched ELO snapshot for {date_str}: {len(df)} clubs")
        return df

    def _fetch_team_history(self, team: str) -> pd.DataFrame:
        """Fetch full ELO history for a single team."""
        cache_key = f"elo_team_{team}"
        cached = self._get_cached(cache_key)

        if cached is not None:
            return pd.DataFrame(cached)

        url = f"{self.base_url}/{team}"
        response = self._fetch(url)
        df = pd.read_csv(io.StringIO(response.text))

        # Normalise team name
        if "Club" in df.columns:
            df["team_id"] = df["Club"].apply(
                lambda x: self.normalise_team(str(x))
            )
        elif "club" in df.columns:
            df["team_id"] = df["club"].apply(
                lambda x: self.normalise_team(str(x))
            )

        self._set_cache(cache_key, df.to_dict(orient="list"))

        logger.info(f"Fetched ELO history for {team}: {len(df)} entries")
        return df

    def _fetch_date_range(
        self, start: str, end: str
    ) -> pd.DataFrame:
        """Fetch weekly ELO snapshots across a date range."""
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")

        frames: list[pd.DataFrame] = []
        current = start_date

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            df = self._fetch_date(date_str)
            frames.append(df)
            current += timedelta(days=7)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)
