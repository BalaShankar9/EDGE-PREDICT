"""
FlashScore Collector — live scores, results, fixtures, and standings.

Scrapes FlashScore.com using BeautifulSoup for match results, upcoming
fixtures, and league standings across major European football leagues.

FlashScore uses a heavily div-based layout. Key class names:
  .event__match       — a single match row
  .event__participant — home or away team name
  .event__score       — score cell (home or away goals)
  .event__time        — kick-off time
  .table__row         — standings table row
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

FLASHSCORE_LEAGUES: dict[str, str] = {
    "Premier League": "football/england/premier-league",
    "La Liga": "football/spain/laliga",
    "Bundesliga": "football/germany/bundesliga",
    "Serie A": "football/italy/serie-a",
    "Ligue 1": "football/france/ligue-1",
    "Eredivisie": "football/netherlands/eredivisie",
    "Championship": "football/england/championship",
    "Liga Portugal": "football/portugal/liga-portugal",
    "Super Lig": "football/turkey/super-lig",
    "Belgian Pro League": "football/belgium/jupiler-pro-league",
}

# Additional headers FlashScore expects to serve rendered content
_FLASHSCORE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.flashscore.com/",
}

# Valid modes accepted by _collect()
_VALID_MODES = ("results", "fixtures", "standings")


class FlashScoreCollector(BaseCollector):
    """Collector for FlashScore live scores, results, fixtures, and standings."""

    source_name = "flashscore"
    base_url = "https://www.flashscore.com"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect match data from FlashScore.

        Parameters
        ----------
        mode : str
            One of "results", "fixtures", or "standings".
            Defaults to "results".
        league : str, optional
            League name key from FLASHSCORE_LEAGUES (e.g. "Premier League").
            If None, iterates all configured leagues.

        Returns
        -------
        pd.DataFrame
            Collected rows; schema depends on mode (see module docstring).
        """
        mode: str = kwargs.get("mode", "results")
        league: Optional[str] = kwargs.get("league")

        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {_VALID_MODES}"
            )

        leagues = (
            {league: FLASHSCORE_LEAGUES[league]}
            if league
            else FLASHSCORE_LEAGUES
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            try:
                df = self._collect_league(mode, league_name, league_path)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.error(
                    f"[{self.source_name}] Failed to collect {mode} for "
                    f"{league_name}: {e}"
                )
                # Continue collecting other leagues rather than aborting all

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    #  Per-league dispatch                                                 #
    # ------------------------------------------------------------------ #

    def _collect_league(
        self, mode: str, league_name: str, league_path: str
    ) -> pd.DataFrame:
        """Fetch and parse one league's page for the given mode."""
        url = f"{self.base_url}/{league_path}/{mode}/"
        response = self._fetch(url, headers=_FLASHSCORE_HEADERS)
        html = response.text

        if mode == "results":
            return self._parse_results(html, league_name)
        elif mode == "fixtures":
            return self._parse_fixtures(html, league_name)
        else:  # standings
            return self._parse_standings(html, league_name)

    # ------------------------------------------------------------------ #
    #  Results parsing                                                     #
    # ------------------------------------------------------------------ #

    def _parse_results(self, html: str, league_name: str) -> pd.DataFrame:
        """Parse completed match results from the HTML page.

        Expected output columns:
          home_team, away_team, home_goals, away_goals,
          match_date, league, home_team_id, away_team_id
        """
        self._check_structure(html, ["event__match", "event__participant"])

        soup = BeautifulSoup(html, "html.parser")
        match_containers = soup.select(
            "div.event__match, div[class*='event__match']"
        )

        rows: list[dict] = []
        for container in match_containers:
            row = self._parse_result_row(container, league_name)
            if row:
                rows.append(row)

        if not rows:
            logger.warning(
                f"[{self.source_name}] No result rows parsed for {league_name}"
            )
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["home_team_id"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        df["away_team_id"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        logger.info(
            f"[{self.source_name}] {league_name} results: {len(df)} matches"
        )
        return df

    def _parse_result_row(
        self, container: Any, league_name: str
    ) -> Optional[dict]:
        """Parse a single result match container.

        Parameters
        ----------
        container : bs4.element.Tag
            A div.event__match element.
        league_name : str
            Name of the league for this match.

        Returns
        -------
        dict or None
            Parsed result data, or None if parsing fails.
        """
        try:
            # Team names — FlashScore puts home participant first, away second
            participants = container.select(
                ".event__participant, [class*='event__participant']"
            )
            if len(participants) < 2:
                return None

            home_team = participants[0].get_text(strip=True)
            away_team = participants[1].get_text(strip=True)

            if not home_team or not away_team:
                return None

            # Score cells — first is home goals, second is away goals
            score_cells = container.select(
                ".event__score, [class*='event__score']"
            )
            home_goals: Optional[int] = None
            away_goals: Optional[int] = None
            if len(score_cells) >= 2:
                try:
                    home_goals = int(score_cells[0].get_text(strip=True))
                    away_goals = int(score_cells[1].get_text(strip=True))
                except (ValueError, TypeError):
                    pass

            # Match date — look for time element or data-start attribute
            match_date: Optional[str] = None
            time_el = container.select_one(
                ".event__time, [class*='event__time']"
            )
            if time_el:
                date_text = time_el.get_text(strip=True)
                match_date = self._normalise_date_string(date_text)

            # Fallback: data-start-time or data-time attributes on container
            if not match_date:
                for attr in ("data-start-time", "data-time", "data-stamp"):
                    stamp = container.get(attr)
                    if stamp:
                        try:
                            match_date = datetime.fromtimestamp(
                                int(stamp), tz=timezone.utc
                            ).strftime("%Y-%m-%d")
                        except (ValueError, TypeError):
                            pass
                        break

            return {
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "match_date": match_date,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[{self.source_name}] Failed to parse result row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Fixtures parsing                                                    #
    # ------------------------------------------------------------------ #

    def _parse_fixtures(self, html: str, league_name: str) -> pd.DataFrame:
        """Parse upcoming fixture data from the HTML page.

        Expected output columns:
          home_team, away_team, match_date, kick_off_time,
          league, home_team_id, away_team_id
        """
        self._check_structure(html, ["event__match", "event__participant"])

        soup = BeautifulSoup(html, "html.parser")
        match_containers = soup.select(
            "div.event__match, div[class*='event__match']"
        )

        rows: list[dict] = []
        for container in match_containers:
            row = self._parse_fixture_row(container, league_name)
            if row:
                rows.append(row)

        if not rows:
            logger.warning(
                f"[{self.source_name}] No fixture rows parsed for {league_name}"
            )
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["home_team_id"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        df["away_team_id"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        logger.info(
            f"[{self.source_name}] {league_name} fixtures: {len(df)} matches"
        )
        return df

    def _parse_fixture_row(
        self, container: Any, league_name: str
    ) -> Optional[dict]:
        """Parse a single upcoming fixture container.

        Parameters
        ----------
        container : bs4.element.Tag
            A div.event__match element.
        league_name : str
            Name of the league for this match.

        Returns
        -------
        dict or None
            Parsed fixture data, or None if parsing fails.
        """
        try:
            participants = container.select(
                ".event__participant, [class*='event__participant']"
            )
            if len(participants) < 2:
                return None

            home_team = participants[0].get_text(strip=True)
            away_team = participants[1].get_text(strip=True)

            if not home_team or not away_team:
                return None

            # Time element carries both date and kick-off time for fixtures
            match_date: Optional[str] = None
            kick_off_time: Optional[str] = None

            time_el = container.select_one(
                ".event__time, [class*='event__time']"
            )
            if time_el:
                time_text = time_el.get_text(strip=True)
                # FlashScore renders e.g. "12.04. 15:00" or "15:00"
                kick_off_time = self._extract_time(time_text)
                match_date = self._normalise_date_string(time_text)

            # Fallback: data-start-time on the container
            if not match_date:
                for attr in ("data-start-time", "data-time", "data-stamp"):
                    stamp = container.get(attr)
                    if stamp:
                        try:
                            dt = datetime.fromtimestamp(
                                int(stamp), tz=timezone.utc
                            )
                            match_date = dt.strftime("%Y-%m-%d")
                            kick_off_time = dt.strftime("%H:%M")
                        except (ValueError, TypeError):
                            pass
                        break

            return {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "kick_off_time": kick_off_time,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[{self.source_name}] Failed to parse fixture row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Standings parsing                                                   #
    # ------------------------------------------------------------------ #

    def _parse_standings(self, html: str, league_name: str) -> pd.DataFrame:
        """Parse league table standings from the HTML page.

        Expected output columns:
          position, team, played, won, drawn, lost,
          goals_for, goals_against, goal_diff, points,
          form_last5, league, team_id
        """
        self._check_structure(html, ["table__row", "table__cell"])

        soup = BeautifulSoup(html, "html.parser")
        rows_els = soup.select(
            "div.table__row, [class*='table__row']"
        )

        rows: list[dict] = []
        for row_el in rows_els:
            row = self._parse_standing_row(row_el, league_name)
            if row:
                rows.append(row)

        if not rows:
            logger.warning(
                f"[{self.source_name}] No standing rows parsed for {league_name}"
            )
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["team_id"] = df["team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        logger.info(
            f"[{self.source_name}] {league_name} standings: {len(df)} teams"
        )
        return df

    def _parse_standing_row(
        self, row_el: Any, league_name: str
    ) -> Optional[dict]:
        """Parse a single league table row.

        Parameters
        ----------
        row_el : bs4.element.Tag
            A div.table__row element.
        league_name : str
            Name of the league.

        Returns
        -------
        dict or None
            Parsed standings data, or None if parsing fails.
        """
        try:
            cells = row_el.select(
                ".table__cell, [class*='table__cell']"
            )
            if len(cells) < 8:
                return None

            # Position is usually the first numeric cell
            position: Optional[int] = None
            pos_el = row_el.select_one(
                ".table__cell--rank, [class*='table__cell--rank'], "
                ".tableCellRank, .rank"
            )
            if pos_el:
                try:
                    position = int(pos_el.get_text(strip=True))
                except (ValueError, TypeError):
                    pass

            # Team name cell
            team_el = row_el.select_one(
                ".table__cell--participant, [class*='participant'], "
                ".tableCellParticipant a, .team"
            )
            team: Optional[str] = None
            if team_el:
                team = team_el.get_text(strip=True)

            if not team:
                return None

            # Numeric stat cells — FlashScore order: P W D L GF GA GD Pts
            numeric_vals: list[Optional[int]] = []
            for cell in cells:
                text = cell.get_text(strip=True)
                # Skip cells that clearly hold non-numeric content
                if not text or any(c.isalpha() for c in text):
                    continue
                # Accept integers (including negative for goal difference)
                cleaned = text.lstrip("+-")
                if cleaned.isdigit():
                    try:
                        numeric_vals.append(int(text))
                    except ValueError:
                        numeric_vals.append(None)

            # Map positionally — guard against short rows
            def _safe(lst: list, idx: int) -> Optional[int]:
                return lst[idx] if idx < len(lst) else None

            played = _safe(numeric_vals, 0)
            won = _safe(numeric_vals, 1)
            drawn = _safe(numeric_vals, 2)
            lost = _safe(numeric_vals, 3)
            goals_for = _safe(numeric_vals, 4)
            goals_against = _safe(numeric_vals, 5)
            goal_diff = _safe(numeric_vals, 6)
            points = _safe(numeric_vals, 7)

            # Form — last 5 results (W/D/L badges)
            form_last5: Optional[str] = None
            form_el = row_el.select_one(
                ".form, [class*='form'], .table__cell--form"
            )
            if form_el:
                # Collect individual result indicators
                badges = form_el.select("span, div")
                form_chars = []
                for badge in badges[:5]:
                    text = badge.get_text(strip=True).upper()
                    if text in ("W", "D", "L"):
                        form_chars.append(text)
                if form_chars:
                    form_last5 = "".join(form_chars)

            return {
                "position": position,
                "team": team,
                "played": played,
                "won": won,
                "drawn": drawn,
                "lost": lost,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goal_diff": goal_diff,
                "points": points,
                "form_last5": form_last5,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(
                f"[{self.source_name}] Failed to parse standing row: {e}"
            )
            return None

    # ------------------------------------------------------------------ #
    #  Date / time helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_date_string(text: str) -> Optional[str]:
        """Try to extract a YYYY-MM-DD date from a FlashScore time string.

        FlashScore renders dates in formats such as:
          "12.04."     (day.month — current year implied)
          "12.04.2025"
          "15:00"      (today's match, no explicit date)
        """
        # Try DD.MM.YYYY
        m = re.search(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
        if m:
            try:
                return datetime(
                    int(m.group(3)), int(m.group(2)), int(m.group(1))
                ).strftime("%Y-%m-%d")
            except ValueError:
                pass

        # Try DD.MM. (current year)
        m = re.search(r"(\d{1,2})\.(\d{1,2})\.", text)
        if m:
            try:
                year = datetime.now(timezone.utc).year
                return datetime(
                    year, int(m.group(2)), int(m.group(1))
                ).strftime("%Y-%m-%d")
            except ValueError:
                pass

        return None

    @staticmethod
    def _extract_time(text: str) -> Optional[str]:
        """Extract HH:MM kick-off time from a FlashScore time string."""
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
        if m:
            return f"{int(m.group(1)):02d}:{m.group(2)}"
        return None
