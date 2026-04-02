"""
Soccerway Collector — results, standings, and upcoming fixtures.

Scrapes upcoming matches, recent results, and current league standings
from int.soccerway.com using BeautifulSoup.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

SOCCERWAY_LEAGUES: dict[str, str] = {
    "Premier League": "england/premier-league/2024-2025",
    "La Liga": "spain/primera-division/2024-2025",
    "Bundesliga": "germany/bundesliga/2024-2025",
    "Serie A": "italy/serie-a/2024-2025",
    "Ligue 1": "france/ligue-1/2024-2025",
}


class SoccerwayCollector(BaseCollector):
    """Collector for Soccerway match results, fixtures, and standings."""

    source_name = "soccerway"
    base_url = "https://int.soccerway.com"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect match and standings data from Soccerway.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all Big 5.
        mode : str, optional
            Collection mode: "fixtures" (default), "results", or "standings".
        """
        league: Optional[str] = kwargs.get("league")
        mode: str = kwargs.get("mode", "fixtures")

        leagues = (
            {league: SOCCERWAY_LEAGUES[league]} if league else SOCCERWAY_LEAGUES
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            try:
                if mode == "standings":
                    df = self._collect_standings(league_name, league_path)
                else:
                    df = self._collect_matches(league_name, league_path, mode=mode)

                if df is not None and not df.empty:
                    frames.append(df)

            except Exception as e:
                logger.error(
                    f"Failed to collect {league_name} ({mode}) from Soccerway: {e}"
                )
                # Per-league error — continue with remaining leagues
                continue

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    #  Matches — fixtures and results                                      #
    # ------------------------------------------------------------------ #

    def _collect_matches(
        self, league_name: str, league_path: str, mode: str = "fixtures"
    ) -> Optional[pd.DataFrame]:
        """Scrape upcoming fixtures or recent results for a single league.

        Parameters
        ----------
        league_name : str
            Human-readable league name.
        league_path : str
            Soccerway URL path segment for this league.
        mode : str
            "fixtures" returns upcoming matches; "results" returns the
            last 14 days of completed matches.
        """
        url = f"{self.base_url}/national/{league_path}/matches/"
        cache_key = f"soccerway_{mode}_{league_path}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[soccerway] Cache hit for {mode} {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url)
        html = response.text

        self._check_structure(html, ["matches", "table"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        cutoff_past = datetime.now(timezone.utc) - timedelta(days=14)
        now = datetime.now(timezone.utc)

        # Soccerway match rows are in a table with class "matches"
        for table in soup.select("table.matches, table#page_competition_1_block_competition_matches_summary_6_block_competition_matches_summary"):
            current_date: Optional[datetime] = None

            for tr in table.select("tr"):
                # Date header rows carry the date for subsequent match rows
                date_header = tr.select_one("th.date, td.date")
                if date_header:
                    raw_date = date_header.get_text(strip=True)
                    current_date = self._parse_date(raw_date)
                    continue

                row = self._parse_match_row(tr, league_name, current_date)
                if not row:
                    continue

                # Filter by mode
                match_dt = self._parse_date(row.get("match_date", ""))
                if mode == "fixtures":
                    if match_dt and match_dt >= now:
                        rows.append(row)
                    elif not match_dt and not row.get("home_goals"):
                        # No date parsed and no score — treat as upcoming
                        rows.append(row)
                elif mode == "results":
                    if match_dt and cutoff_past <= match_dt < now:
                        rows.append(row)
                    elif match_dt is None and row.get("home_goals") is not None:
                        rows.append(row)

        if rows:
            for r in rows:
                r["home_team_id"] = self.normalise_team(r["home_team"])
                r["away_team_id"] = self.normalise_team(r["away_team"])
            self._set_cache(cache_key, rows)

        logger.info(f"[soccerway] Fetched {mode} {league_name}: {len(rows)} rows")
        return pd.DataFrame(rows) if rows else None

    def _parse_match_row(
        self,
        tr: Any,
        league_name: str,
        current_date: Optional[datetime],
    ) -> Optional[dict]:
        """Parse a single match row from a Soccerway matches table.

        Parameters
        ----------
        tr : bs4.element.Tag
            A ``<tr>`` element from the matches table.
        league_name : str
            Name of the league for this match.
        current_date : datetime or None
            Date from the most recent date-header row.

        Returns
        -------
        dict or None
            Parsed match data, or None if the row cannot be parsed.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            # Team names — Soccerway uses .team-a / .team-b or plain links
            home_team = None
            away_team = None

            home_el = tr.select_one(".team-a a, td.team-a")
            away_el = tr.select_one(".team-b a, td.team-b")

            if home_el:
                home_team = home_el.get_text(strip=True)
            if away_el:
                away_team = away_el.get_text(strip=True)

            # Fallback: extract team names from links in data cells
            if not home_team or not away_team:
                links = tr.select("td a")
                team_links = [
                    a for a in links if "/teams/" in (a.get("href") or "")
                ]
                if len(team_links) >= 2:
                    home_team = team_links[0].get_text(strip=True)
                    away_team = team_links[1].get_text(strip=True)

            if not home_team or not away_team:
                return None

            # Score — look for .score-time or a cell with "X - Y" pattern
            home_goals: Optional[int] = None
            away_goals: Optional[int] = None
            kick_off: Optional[str] = None

            score_el = tr.select_one(".score-time, .score, span.score")
            if score_el:
                score_text = score_el.get_text(strip=True)
                # Completed score: "2 - 1" or "2:1"
                sep = "-" if "-" in score_text else (":" if ":" in score_text else None)
                if sep:
                    parts = score_text.split(sep)
                    if len(parts) == 2:
                        try:
                            home_goals = int(parts[0].strip())
                            away_goals = int(parts[1].strip())
                        except ValueError:
                            # Could be a time like "20:45" — treat as kick-off
                            if sep == ":" and ":" in score_text:
                                kick_off = score_text.strip()

            # Kick-off time from a dedicated cell if not found in score
            if not kick_off and home_goals is None:
                for td in tds:
                    text = td.get_text(strip=True)
                    if ":" in text and len(text) <= 8 and text[0].isdigit():
                        kick_off = text
                        break

            match_date_str = (
                current_date.strftime("%Y-%m-%d") if current_date else ""
            )

            return {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date_str,
                "kick_off": kick_off or "",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[soccerway] Failed to parse match row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Standings                                                           #
    # ------------------------------------------------------------------ #

    def _collect_standings(
        self, league_name: str, league_path: str
    ) -> Optional[pd.DataFrame]:
        """Scrape the current standings table for a single league.

        Parameters
        ----------
        league_name : str
            Human-readable league name.
        league_path : str
            Soccerway URL path segment for this league.
        """
        url = f"{self.base_url}/national/{league_path}/matches/"
        cache_key = f"soccerway_standings_{league_path}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[soccerway] Cache hit for standings {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url)
        html = response.text

        self._check_structure(html, ["table", "standings"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        # Soccerway standings table typically has class "standing" or "league-table"
        standings_table = soup.select_one(
            "table.standing, table.league-table, "
            "table#page_competition_1_block_competition_league_table_1_table"
        )
        if not standings_table:
            # Try any table that has a rank/position column
            for table in soup.select("table"):
                if table.select_one("th.rank, th.position"):
                    standings_table = table
                    break

        if not standings_table:
            logger.warning(f"[soccerway] No standings table found for {league_name}")
            return None

        for tr in standings_table.select("tbody tr"):
            row = self._parse_standing_row(tr, league_name)
            if row:
                rows.append(row)

        if rows:
            for r in rows:
                r["team_id"] = self.normalise_team(r["team"])
            self._set_cache(cache_key, rows)

        logger.info(f"[soccerway] Fetched standings {league_name}: {len(rows)} teams")
        return pd.DataFrame(rows) if rows else None

    def _parse_standing_row(self, tr: Any, league_name: str) -> Optional[dict]:
        """Parse a single row from a Soccerway standings table.

        Parameters
        ----------
        tr : bs4.element.Tag
            A ``<tr>`` element from the standings table.
        league_name : str
            Name of the league for this standing entry.

        Returns
        -------
        dict or None
            Parsed standing data, or None if the row cannot be parsed.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 5:
                return None

            # Position — first numeric cell
            position: Optional[int] = None
            pos_el = tr.select_one("td.rank, td.position")
            if pos_el:
                try:
                    position = int(pos_el.get_text(strip=True))
                except ValueError:
                    pass
            if position is None:
                try:
                    position = int(tds[0].get_text(strip=True))
                except ValueError:
                    return None

            # Team name — link in the row
            team = None
            team_link = tr.select_one("td.team a, td a[href*='/teams/']")
            if team_link:
                team = team_link.get_text(strip=True)
            if not team:
                # Fallback: second non-numeric cell
                for td in tds[1:]:
                    text = td.get_text(strip=True)
                    if text and not text.isdigit():
                        team = text
                        break

            if not team:
                return None

            # Numeric columns — played, won, drawn, lost, GF, GA, GD, points
            # Soccerway ordering: pos | team | P | W | D | L | GF | GA | GD | Pts
            numeric_vals: list[Optional[int]] = []
            for td in tds:
                text = td.get_text(strip=True)
                try:
                    numeric_vals.append(int(text))
                except ValueError:
                    numeric_vals.append(None)

            # Filter out the position (first element) and None team cells
            int_vals = [v for v in numeric_vals if v is not None]
            # Skip the position itself
            if int_vals and int_vals[0] == position:
                int_vals = int_vals[1:]

            played = int_vals[0] if len(int_vals) > 0 else None
            won = int_vals[1] if len(int_vals) > 1 else None
            drawn = int_vals[2] if len(int_vals) > 2 else None
            lost = int_vals[3] if len(int_vals) > 3 else None
            goals_for = int_vals[4] if len(int_vals) > 4 else None
            goals_against = int_vals[5] if len(int_vals) > 5 else None
            goal_diff = int_vals[6] if len(int_vals) > 6 else None
            points = int_vals[7] if len(int_vals) > 7 else None

            # Derive GD from GF/GA if the explicit GD cell is missing
            if goal_diff is None and goals_for is not None and goals_against is not None:
                goal_diff = goals_for - goals_against

            # Form — Soccerway sometimes renders last-5 form as colour-coded spans
            form: Optional[str] = None
            form_el = tr.select_one(".form, td.form")
            if form_el:
                form = form_el.get_text(strip=True)

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
                "form": form or "",
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[soccerway] Failed to parse standing row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_date(raw: str) -> Optional[datetime]:
        """Attempt to parse a date string into a timezone-aware datetime.

        Tries several common formats used on Soccerway before giving up.

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
            "%A, %d %B %Y",
            "%d %B %Y",
            "%B %d, %Y",
        ]
        cleaned = raw.strip()
        for fmt in formats:
            try:
                return datetime.strptime(cleaned, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None
