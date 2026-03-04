"""
WinDrawWin Collector — match prediction scraper.

Scrapes match predictions and outcome probabilities from
windrawwin.com using BeautifulSoup.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

WINDRAWWIN_LEAGUES: dict[str, str] = {
    "Premier League": "premier-league",
    "La Liga": "la-liga",
    "Bundesliga": "bundesliga",
    "Serie A": "serie-a",
    "Ligue 1": "ligue-1",
}


class WinDrawWinCollector(BaseCollector):
    """Collector for WinDrawWin match predictions."""

    source_name = "windrawwin"
    base_url = "https://www.windrawwin.com"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect match predictions from WinDrawWin.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        """
        league: Optional[str] = kwargs.get("league")

        leagues = (
            {league: WINDRAWWIN_LEAGUES[league]} if league else WINDRAWWIN_LEAGUES
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            url = f"{self.base_url}/predictions/{league_path}/"

            try:
                response = self._fetch(url)
                html = response.text

                # Structural fingerprinting
                self._check_structure(html, ["predictions", "table"])

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
                    f"Fetched {league_name}: {len(rows)} predictions"
                )

            except Exception as e:
                logger.error(
                    f"Failed to collect {league_name} from WinDrawWin: {e}"
                )
                raise

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _parse_match_row(
        self, tr: Any, league_name: str
    ) -> Optional[dict]:
        """Parse a single match prediction row from WinDrawWin HTML.

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
            prob_home = None
            prob_draw = None
            prob_away = None

            # Extract team names from cells
            for td in tds:
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

            # Look for predicted result (1, X, 2) or (W, D, L)
            for td in tds:
                text = td.get_text(strip=True).upper()
                if text in ("1", "X", "2"):
                    predicted_result = text
                    break
                if text in ("W", "D", "L"):
                    predicted_result = {"W": "1", "D": "X", "L": "2"}.get(
                        text, text
                    )
                    break

            # Look for probability percentages
            probs: list[float] = []
            for td in tds:
                text = td.get_text(strip=True)
                if text.endswith("%"):
                    try:
                        probs.append(float(text.rstrip("%")))
                    except ValueError:
                        pass

            if len(probs) >= 3:
                prob_home = probs[0]
                prob_draw = probs[1]
                prob_away = probs[2]

            return {
                "home_team": home_team,
                "away_team": away_team,
                "predicted_result": predicted_result,
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"Failed to parse match row: {e}")
            return None
