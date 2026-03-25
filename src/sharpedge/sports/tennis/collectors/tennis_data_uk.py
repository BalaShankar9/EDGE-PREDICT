"""TennisDataUKCollector — ATP/WTA match results and bookmaker odds.

Source: tennis-data.co.uk
Format: Excel (.xlsx) files with one file per year, separate for ATP and WTA.
"""

import io
import logging
from typing import Any

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# Mapping from raw tennis-data.co.uk column names to our standard names
_COLUMN_MAP = {
    "Date": "date",
    "Tournament": "tournament",
    "Surface": "surface",
    "Round": "round",
    "Best of": "best_of",
    "Winner": "winner",
    "Loser": "loser",
    "WRank": "winner_rank",
    "LRank": "loser_rank",
    "WPts": "winner_points",
    "LPts": "loser_points",
    "W1": "w_set1",
    "L1": "l_set1",
    "W2": "w_set2",
    "L2": "l_set2",
    "W3": "w_set3",
    "L3": "l_set3",
    "W4": "w_set4",
    "L4": "l_set4",
    "W5": "w_set5",
    "L5": "l_set5",
    "Wsets": "winner_sets",
    "Lsets": "loser_sets",
    "B365W": "b365_winner",
    "B365L": "b365_loser",
    "PSW": "ps_winner",
    "PSL": "ps_loser",
    "MaxW": "max_winner",
    "MaxL": "max_loser",
    "AvgW": "avg_winner",
    "AvgL": "avg_loser",
}


class TennisDataUKCollector(BaseCollector):
    """Collector for tennis-data.co.uk match data.

    Downloads Excel files with ATP/WTA match results including bookmaker odds.
    URL patterns:
        ATP: http://www.tennis-data.co.uk/{year}/{year}.xlsx
        WTA: http://www.tennis-data.co.uk/{year}w/{year}.xlsx
    """

    source_name = "tennis_data_uk"
    base_url = "http://www.tennis-data.co.uk"
    request_delay = 3.0
    cache_ttl_hours = 24  # cache for 24 hours

    REQUIRED_COLUMNS = [
        "date",
        "tournament",
        "surface",
        "round",
        "winner",
        "loser",
        "winner_rank",
        "loser_rank",
        "winner_points",
        "loser_points",
        "score",
        "best_of",
        "winner_sets",
        "loser_sets",
        "b365_winner",
        "b365_loser",
        "ps_winner",
        "ps_loser",
    ]

    def _build_urls(self, tour: str, year: int) -> list[str]:
        """Build candidate URLs to try for a given tour and year."""
        suffix = "w" if tour.lower() == "wta" else ""
        return [
            f"{self.base_url}/{year}{suffix}/{year}.xlsx",
            f"{self.base_url}/{year}/{tour.lower()}/{year}.xlsx",
            f"{self.base_url}/{year}{suffix}/{year}.csv",
        ]

    def _parse_excel(self, content: bytes) -> pd.DataFrame:
        """Parse Excel content into a standardised DataFrame."""
        df = pd.read_excel(io.BytesIO(content))
        return self._standardise(df)

    def _parse_csv(self, content: bytes) -> pd.DataFrame:
        """Parse CSV content into a standardised DataFrame."""
        df = pd.read_csv(io.BytesIO(content))
        return self._standardise(df)

    def _standardise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and compute derived fields."""
        # Rename known columns
        rename = {k: v for k, v in _COLUMN_MAP.items() if k in df.columns}
        df = df.rename(columns=rename)

        # Build composite score string from set columns if available
        set_cols_w = [c for c in df.columns if c.startswith("w_set")]
        set_cols_l = [c for c in df.columns if c.startswith("l_set")]
        if set_cols_w and set_cols_l and "score" not in df.columns:
            scores = []
            for _, row in df.iterrows():
                parts = []
                for wc, lc in zip(sorted(set_cols_w), sorted(set_cols_l)):
                    ws, ls = row.get(wc), row.get(lc)
                    if pd.notna(ws) and pd.notna(ls):
                        parts.append(f"{int(ws)}-{int(ls)}")
                scores.append(" ".join(parts) if parts else "")
            df["score"] = scores

        # Ensure winner_sets / loser_sets exist
        if "winner_sets" not in df.columns and set_cols_w:
            df["winner_sets"] = df[set_cols_w].notna().sum(axis=1)
        if "loser_sets" not in df.columns and set_cols_l:
            df["loser_sets"] = df[set_cols_l].notna().sum(axis=1)

        # Ensure best_of column exists
        if "best_of" not in df.columns:
            df["best_of"] = 3  # default to best-of-3

        # Ensure all required columns are present
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Parse date column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Convert odds columns to numeric
        odds_cols = [
            "b365_winner", "b365_loser", "ps_winner", "ps_loser",
            "max_winner", "max_loser", "avg_winner", "avg_loser",
        ]
        for col in odds_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Add tour metadata (will be overwritten by caller)
        df["source"] = self.source_name

        return df

    def _collect(self, tour: str = "atp", year: int = 2024, **kwargs: Any) -> pd.DataFrame:
        """Download and parse tennis match data for a given tour and year.

        Args:
            tour: "atp" or "wta"
            year: Season year (e.g. 2024)

        Returns:
            DataFrame with standardised columns. Empty if download fails.
        """
        cache_key = f"{tour}_{year}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info(f"[{self.source_name}] Cache hit for {cache_key}")
            return pd.DataFrame(cached)

        urls = self._build_urls(tour, year)

        for url in urls:
            try:
                response = self._fetch(url)
                content = response.content

                if url.endswith(".csv"):
                    df = self._parse_csv(content)
                else:
                    df = self._parse_excel(content)

                if not df.empty:
                    df["tour"] = tour.upper()
                    df["year"] = year
                    logger.info(
                        f"[{self.source_name}] Parsed {len(df)} matches "
                        f"for {tour.upper()} {year} from {url}"
                    )
                    # Cache the result
                    self._set_cache(cache_key, df.to_dict(orient="records"))
                    return df

            except Exception as exc:
                logger.debug(
                    f"[{self.source_name}] Failed to fetch {url}: {exc}"
                )
                continue

        logger.warning(
            f"[{self.source_name}] All URLs failed for {tour.upper()} {year}, "
            f"returning empty DataFrame"
        )
        return pd.DataFrame(columns=self.REQUIRED_COLUMNS)
