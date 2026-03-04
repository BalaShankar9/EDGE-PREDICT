"""
Forebet Collector — competitor ML predictions scraper.

Scrapes match predictions (probabilities and predicted scores)
from forebet.com using BeautifulSoup.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

FOREBET_LEAGUES: dict[str, str] = {
    "Premier League": "england/premier-league",
    "La Liga": "spain/la-liga",
    "Bundesliga": "germany/bundesliga",
    "Serie A": "italy/serie-a",
    "Ligue 1": "france/ligue-1",
}


class ForebetCollector(BaseCollector):
    """Collector for Forebet match predictions."""

    source_name = "forebet"
    base_url = "https://www.forebet.com"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect match predictions from Forebet.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        """
        league: Optional[str] = kwargs.get("league")

        leagues = {league: FOREBET_LEAGUES[league]} if league else FOREBET_LEAGUES
        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            url = (
                f"{self.base_url}/en/football-tips-and-predictions-for-"
                f"{league_path}"
            )

            try:
                response = self._fetch(url)
                html = response.text

                # Structural fingerprinting
                self._check_structure(html, ["rcnt", "foremark"])

                soup = BeautifulSoup(html, "html.parser")
                containers = soup.select(".rcnt")

                rows: list[dict] = []
                for container in containers:
                    row = self._parse_match_row(container, league_name)
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
                    f"Failed to collect {league_name} from Forebet: {e}"
                )
                raise

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _parse_match_row(
        self, container: Any, league_name: str
    ) -> Optional[dict]:
        """Parse a single match prediction row from Forebet HTML.

        Parameters
        ----------
        container : bs4.element.Tag
            A `.rcnt` container element.
        league_name : str
            Name of the league for this match.

        Returns
        -------
        dict or None
            Parsed match data, or None if parsing fails.
        """
        try:
            # Extract team names
            home_team = None
            away_team = None

            # Try .tnms first (newer layout)
            tnms = container.select_one(".tnms")
            if tnms:
                spans = tnms.select("span")
                if len(spans) >= 2:
                    home_team = spans[0].get_text(strip=True)
                    away_team = spans[1].get_text(strip=True)

            # Fallback to .homemark / .awaymark
            if not home_team:
                home_el = container.select_one(".homemark")
                if home_el:
                    home_team = home_el.get_text(strip=True)
            if not away_team:
                away_el = container.select_one(".awaymark")
                if away_el:
                    away_team = away_el.get_text(strip=True)

            if not home_team or not away_team:
                return None

            # Extract probabilities (H / D / A)
            prob_home, prob_draw, prob_away = None, None, None
            prob_spans = container.select(".fprc span")
            if len(prob_spans) >= 3:
                try:
                    prob_home = int(prob_spans[0].get_text(strip=True))
                    prob_draw = int(prob_spans[1].get_text(strip=True))
                    prob_away = int(prob_spans[2].get_text(strip=True))
                except (ValueError, TypeError):
                    pass

            # Extract predicted score
            predicted_home, predicted_away = None, None
            score_el = container.select_one(".predict_score, .ex_sc")
            if score_el:
                score_text = score_el.get_text(strip=True)
                parts = score_text.split("-")
                if len(parts) == 2:
                    try:
                        predicted_home = int(parts[0].strip())
                        predicted_away = int(parts[1].strip())
                    except (ValueError, TypeError):
                        pass

            return {
                "home_team": home_team,
                "away_team": away_team,
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "predicted_score_home": predicted_home,
                "predicted_score_away": predicted_away,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"Failed to parse match row: {e}")
            return None
