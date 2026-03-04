"""
FootyStats Collector — rich match statistics scraper.

Scrapes predictions, over/under stats, BTTS stats, corners, cards,
and referee data from footystats.org using BeautifulSoup.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

FOOTYSTATS_LEAGUES: dict[str, str] = {
    "Premier League": "england/premier-league",
    "La Liga": "spain/la-liga",
    "Bundesliga": "germany/bundesliga",
    "Serie A": "italy/serie-a",
    "Ligue 1": "france/ligue-1",
}


class FootyStatsCollector(BaseCollector):
    """Collector for FootyStats match statistics and predictions."""

    source_name = "footystats"
    base_url = "https://footystats.org"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect match stats and predictions from FootyStats.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        """
        league: Optional[str] = kwargs.get("league")

        leagues = (
            {league: FOOTYSTATS_LEAGUES[league]} if league else FOOTYSTATS_LEAGUES
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            url = f"{self.base_url}/{league_path}"

            try:
                response = self._fetch(url)
                html = response.text

                # Structural fingerprinting
                self._check_structure(html, ["table", "team"])

                soup = BeautifulSoup(html, "html.parser")

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
                    df["home_team_id"] = df["home_team"].apply(
                        lambda x: self.normalise_team(str(x))
                    )
                    df["away_team_id"] = df["away_team"].apply(
                        lambda x: self.normalise_team(str(x))
                    )
                    frames.append(df)

                logger.info(
                    f"Fetched {league_name}: {len(rows)} match stats"
                )

            except Exception as e:
                logger.error(
                    f"Failed to collect {league_name} from FootyStats: {e}"
                )
                raise

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _parse_match_row(
        self, tr: Any, league_name: str
    ) -> Optional[dict]:
        """Parse a single match row from FootyStats HTML.

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

            home_team = None
            away_team = None
            predicted_result = None

            # Extract team names from cells
            for td in tds:
                link = td.select_one("a")
                text = (link or td).get_text(strip=True)
                if not text:
                    continue

                # Skip date-like cells and numeric-only cells
                if text.replace("/", "").replace("-", "").isdigit():
                    continue
                if text.replace(".", "").isdigit():
                    continue

                if home_team is None and len(text) > 2:
                    home_team = text
                elif away_team is None and len(text) > 2:
                    away_team = text

            if not home_team or not away_team:
                return None

            # Extract over/under and BTTS data
            over_25 = None
            btts = None
            corners = None

            for td in tds:
                text = td.get_text(strip=True)
                # Look for percentage values for O/U and BTTS
                if text.endswith("%"):
                    try:
                        val = float(text.rstrip("%"))
                        if over_25 is None:
                            over_25 = val
                        elif btts is None:
                            btts = val
                    except ValueError:
                        pass

            # Look for predicted result
            for td in tds:
                text = td.get_text(strip=True).upper()
                if text in ("1", "X", "2"):
                    predicted_result = text
                    break

            return {
                "home_team": home_team,
                "away_team": away_team,
                "predicted_result": predicted_result,
                "over_25_pct": over_25,
                "btts_pct": btts,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"Failed to parse match row: {e}")
            return None
