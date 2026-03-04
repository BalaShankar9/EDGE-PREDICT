"""
PredictZ Collector — match prediction scraper.

Scrapes match predictions (predicted result and confidence data)
from predictz.com using BeautifulSoup.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

PREDICTZ_LEAGUES: dict[str, str] = {
    "Premier League": "premier-league",
    "La Liga": "la-liga",
    "Bundesliga": "bundesliga",
    "Serie A": "serie-a",
    "Ligue 1": "ligue-1",
}


class PredictZCollector(BaseCollector):
    """Collector for PredictZ match predictions."""

    source_name = "predictz"
    base_url = "https://www.predictz.com"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect match predictions from PredictZ.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        """
        league: Optional[str] = kwargs.get("league")

        leagues = {league: PREDICTZ_LEAGUES[league]} if league else PREDICTZ_LEAGUES
        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            url = f"{self.base_url}/predictions/{league_path}/"

            try:
                response = self._fetch(url)
                html = response.text

                # Structural fingerprinting
                self._check_structure(html, ["predictions", "table"])

                soup = BeautifulSoup(html, "html.parser")

                # Look for prediction table rows
                rows: list[dict] = []
                tables = soup.select("table")
                for table in tables:
                    trs = table.select("tr")
                    for tr in trs:
                        row = self._parse_match_row(tr, league_name)
                        if row:
                            rows.append(row)

                if rows:
                    df = pd.DataFrame(rows)
                    # Normalise team names
                    df["home_team_id"] = df["home_team"].apply(
                        lambda x: self.normalise_team(str(x))
                    )
                    df["away_team_id"] = df["away_team"].apply(
                        lambda x: self.normalise_team(str(x))
                    )
                    frames.append(df)

                logger.info(
                    f"Fetched {league_name}: {len(rows)} predictions"
                )

            except Exception as e:
                logger.error(
                    f"Failed to collect {league_name} from PredictZ: {e}"
                )
                raise

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _parse_match_row(
        self, tr: Any, league_name: str
    ) -> Optional[dict]:
        """Parse a single match prediction row from PredictZ HTML.

        Parameters
        ----------
        tr : bs4.element.Tag
            A table row element.
        league_name : str
            Name of the league for this match.

        Returns
        -------
        dict or None
            Parsed match data, or None if parsing fails.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            # Try to extract team names from cells
            # PredictZ typically has: date | home | away | prediction | ...
            home_team = None
            away_team = None
            predicted_result = None

            # Look for links or text in cells that contain team names
            for i, td in enumerate(tds):
                link = td.select_one("a")
                text = (link or td).get_text(strip=True)
                if not text:
                    continue

                # Skip date-like cells and numeric-only cells
                if text.replace("/", "").replace("-", "").isdigit():
                    continue

                if home_team is None and len(text) > 2:
                    home_team = text
                elif away_team is None and len(text) > 2:
                    away_team = text

            if not home_team or not away_team:
                return None

            # Look for predicted result (1, X, 2)
            for td in tds:
                text = td.get_text(strip=True).upper()
                if text in ("1", "X", "2"):
                    predicted_result = text
                    break

            # Extract any probability/confidence data
            confidence = None
            for td in tds:
                text = td.get_text(strip=True)
                # Look for percentage-like values
                if text.endswith("%"):
                    try:
                        confidence = float(text.rstrip("%"))
                        break
                    except ValueError:
                        pass

            return {
                "home_team": home_team,
                "away_team": away_team,
                "predicted_result": predicted_result,
                "confidence": confidence,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"Failed to parse match row: {e}")
            return None
