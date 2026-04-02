"""
WorldFootball Collector — comprehensive historical match data.

Scrapes schedule pages from worldfootball.net to retrieve historical
and upcoming match results including attendance figures.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

WORLD_FOOTBALL_LEAGUES: dict[str, str] = {
    "Premier League": "eng-premier-league",
    "La Liga": "esp-primera-division",
    "Bundesliga": "bundesliga",
    "Serie A": "ita-serie-a",
    "Ligue 1": "fra-ligue-1",
}

# WorldFootball.net covers matchdays 1–38 for most top-flight leagues
_MAX_MATCHDAYS = 38


class WorldFootballCollector(BaseCollector):
    """Collector for WorldFootball.net historical match data."""

    source_name = "world_football"
    base_url = "https://www.worldfootball.net"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect historical match data from WorldFootball.net.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all leagues.
        matchday : int, optional
            Specific matchday to collect (1–38). If None, collects all
            available matchdays for the league.
        season : str, optional
            Season string (e.g. "2024-2025"). Defaults to "2024-2025".
        """
        league: Optional[str] = kwargs.get("league")
        matchday: Optional[int] = kwargs.get("matchday")
        season: str = kwargs.get("season", "2024-2025")

        leagues = (
            {league: WORLD_FOOTBALL_LEAGUES[league]}
            if league
            else WORLD_FOOTBALL_LEAGUES
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_code in leagues.items():
            try:
                df = self._collect_league(
                    league_name, league_code, season, matchday
                )
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.error(
                    f"Failed to collect {league_name} from WorldFootball: {e}"
                )
                # Per-league error — continue with remaining leagues
                continue

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    #  Per-league collection                                               #
    # ------------------------------------------------------------------ #

    def _collect_league(
        self,
        league_name: str,
        league_code: str,
        season: str,
        matchday: Optional[int],
    ) -> Optional[pd.DataFrame]:
        """Collect all (or a single) matchday's data for one league.

        Parameters
        ----------
        league_name : str
            Human-readable league name.
        league_code : str
            WorldFootball internal league code (e.g. "eng-premier-league").
        season : str
            Season string (e.g. "2024-2025").
        matchday : int or None
            If provided, fetch only this matchday; otherwise fetch 1 through
            ``_MAX_MATCHDAYS`` and stop early when no more matches are found.
        """
        matchdays_to_fetch = (
            range(matchday, matchday + 1)
            if matchday is not None
            else range(1, _MAX_MATCHDAYS + 1)
        )

        all_rows: list[dict] = []

        for md in matchdays_to_fetch:
            cache_key = f"worldfootball_{league_code}_{season}_md{md}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug(
                    f"[world_football] Cache hit: {league_name} MD{md}"
                )
                all_rows.extend(cached)
                continue

            url = (
                f"{self.base_url}/schedule/"
                f"{league_code}-{season}-spieltag/{md}/"
            )

            try:
                response = self._fetch(url)
                html = response.text

                self._check_structure(html, ["spieltag", "table"])

                rows = self._parse_schedule_page(html, league_name, md)

                if not rows:
                    # No matches found for this matchday — likely past the
                    # last available matchday; stop iterating.
                    if matchday is None:
                        logger.debug(
                            f"[world_football] No data for {league_name} "
                            f"MD{md} — stopping early"
                        )
                        break
                else:
                    self._set_cache(cache_key, rows)
                    all_rows.extend(rows)
                    logger.info(
                        f"[world_football] {league_name} MD{md}: {len(rows)} matches"
                    )

            except Exception as e:
                logger.warning(
                    f"[world_football] Failed to fetch {league_name} MD{md}: {e}"
                )
                # Per-matchday error — continue with next matchday
                continue

        if not all_rows:
            return None

        df = pd.DataFrame(all_rows)
        df["home_team_id"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        df["away_team_id"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        return df

    # ------------------------------------------------------------------ #
    #  HTML parsing                                                        #
    # ------------------------------------------------------------------ #

    def _parse_schedule_page(
        self, html: str, league_name: str, matchday: int
    ) -> list[dict]:
        """Parse all match rows from a WorldFootball schedule page.

        Parameters
        ----------
        html : str
            Raw HTML content of the schedule page.
        league_name : str
            Name of the league for these matches.
        matchday : int
            Matchday number being parsed.

        Returns
        -------
        list of dict
            One dict per match found on the page.
        """
        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        # WorldFootball schedule pages use a standard results table
        # typically with class "standard_tabelle" or "tabelle_with_na"
        table = soup.select_one(
            "table.standard_tabelle, table.tabelle_with_na"
        )
        if not table:
            # Fallback: any table that contains match data
            for t in soup.select("table"):
                if t.select_one("td.standard"):
                    table = t
                    break

        if not table:
            logger.debug(
                f"[world_football] No schedule table found for {league_name} MD{matchday}"
            )
            return rows

        for tr in table.select("tr"):
            row = self._parse_match_row(tr, league_name, matchday)
            if row:
                rows.append(row)

        return rows

    def _parse_match_row(
        self, tr: Any, league_name: str, matchday: int
    ) -> Optional[dict]:
        """Parse a single match row from a WorldFootball schedule table.

        WorldFootball table column order (typical):
            date | time | home_team | score | away_team | attendance

        Parameters
        ----------
        tr : bs4.element.Tag
            A ``<tr>`` element from the schedule table.
        league_name : str
            Name of the league for this match.
        matchday : int
            Matchday number.

        Returns
        -------
        dict or None
            Parsed match data, or None if the row cannot be parsed.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 4:
                return None

            # ---- Date ----
            match_date: Optional[str] = None
            date_el = tr.select_one("td.standard a[href*='spieltag'], td:first-child")
            if date_el:
                raw_date = date_el.get_text(strip=True)
                parsed_dt = self._parse_date(raw_date)
                if parsed_dt:
                    match_date = parsed_dt.strftime("%Y-%m-%d")
                else:
                    match_date = raw_date

            # Skip header or empty rows
            if not match_date:
                raw_first = tds[0].get_text(strip=True)
                if not raw_first or raw_first.lower() in ("date", "datum", "#"):
                    return None
                match_date = raw_first

            # ---- Team names — extracted from links ----
            home_team: Optional[str] = None
            away_team: Optional[str] = None

            team_links = tr.select("td a[href*='/teams/']")
            if len(team_links) >= 2:
                home_team = team_links[0].get_text(strip=True)
                away_team = team_links[1].get_text(strip=True)

            # Fallback: look for non-date, non-score text cells
            if not home_team or not away_team:
                text_cells = [
                    td.get_text(strip=True)
                    for td in tds
                    if td.get_text(strip=True)
                    and not self._looks_like_score(td.get_text(strip=True))
                    and not self._looks_like_date(td.get_text(strip=True))
                ]
                if len(text_cells) >= 2:
                    home_team = text_cells[0]
                    away_team = text_cells[1]

            if not home_team or not away_team:
                return None

            # ---- Score ----
            home_goals: Optional[int] = None
            away_goals: Optional[int] = None

            score_el = tr.select_one(
                "td.standard a[href*='result'], "
                "td a[href*='_ergebnis'], "
                "td.ergebnis"
            )
            if not score_el:
                # Try any cell that looks like a score
                for td in tds:
                    text = td.get_text(strip=True)
                    if self._looks_like_score(text):
                        score_el = td
                        break

            if score_el:
                score_text = score_el.get_text(strip=True)
                home_goals, away_goals = self._parse_score(score_text)

            # ---- Attendance ----
            attendance: Optional[int] = None
            for td in reversed(tds):
                text = td.get_text(strip=True).replace(".", "").replace(",", "")
                try:
                    val = int(text)
                    # Attendance is typically > 500 and not a goals figure
                    if val > 500:
                        attendance = val
                        break
                except ValueError:
                    continue

            return {
                "match_date": match_date or "",
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "attendance": attendance,
                "matchday": matchday,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[world_football] Failed to parse match row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_date(raw: str) -> Optional[datetime]:
        """Attempt to parse a date string into a timezone-aware datetime.

        Tries several date formats commonly used on WorldFootball.net.

        Parameters
        ----------
        raw : str
            Raw date text from the HTML.

        Returns
        -------
        datetime or None
        """
        if not raw:
            return None
        formats = [
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%d.%m.%Y",
            "%d-%m-%Y",
            "%B %d, %Y",
            "%d %B %Y",
        ]
        cleaned = raw.strip()
        for fmt in formats:
            try:
                return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    @staticmethod
    def _looks_like_score(text: str) -> bool:
        """Return True if ``text`` resembles a football scoreline (e.g. "2:1").

        Parameters
        ----------
        text : str
            Cell text to inspect.
        """
        if not text:
            return False
        sep = None
        if ":" in text:
            sep = ":"
        elif " - " in text:
            sep = " - "
        elif "-" in text and len(text) <= 5:
            sep = "-"
        if sep is None:
            return False
        parts = text.split(sep)
        if len(parts) != 2:
            return False
        return all(p.strip().isdigit() for p in parts)

    @staticmethod
    def _looks_like_date(text: str) -> bool:
        """Return True if ``text`` resembles a date string.

        Parameters
        ----------
        text : str
            Cell text to inspect.
        """
        if not text:
            return False
        return (
            text.count(".") >= 2
            or text.count("/") >= 2
            or (len(text) == 10 and text[4] == "-" and text[7] == "-")
        )

    @staticmethod
    def _parse_score(text: str) -> tuple[Optional[int], Optional[int]]:
        """Parse a score string into (home_goals, away_goals).

        Handles formats like "2:1", "2 - 1", and "2-1".

        Parameters
        ----------
        text : str
            Score text (e.g. "2:1").

        Returns
        -------
        tuple of (int or None, int or None)
        """
        if not text:
            return None, None
        for sep in (":", " - ", "-"):
            if sep in text:
                parts = text.split(sep, 1)
                if len(parts) == 2:
                    try:
                        return int(parts[0].strip()), int(parts[1].strip())
                    except ValueError:
                        continue
        return None, None
