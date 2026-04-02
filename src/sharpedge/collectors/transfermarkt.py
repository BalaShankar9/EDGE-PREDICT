"""
Transfermarkt Collector — injuries, market values, and fixtures.

Scrapes injury reports and upcoming fixtures from transfermarkt.com
using BeautifulSoup. Transfermarkt uses table-based layouts with
class selectors like .items, .hauptlink, and .zentriert.

Note: Transfermarkt is aggressive with rate limiting — request_delay
is set to 4.0s and XMLHttpRequest headers are required.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

TRANSFERMARKT_LEAGUES: dict[str, dict[str, str]] = {
    "Premier League": {"path": "premier-league", "id": "GB1"},
    "La Liga": {"path": "laliga", "id": "ES1"},
    "Bundesliga": {"path": "1-bundesliga", "id": "L1"},
    "Serie A": {"path": "serie-a", "id": "IT1"},
    "Ligue 1": {"path": "ligue-1", "id": "FR1"},
    "Eredivisie": {"path": "eredivisie", "id": "NL1"},
    "Primeira Liga": {"path": "liga-nos", "id": "PO1"},
    "Championship": {"path": "championship", "id": "GB2"},
}

# Transfermarkt requires XMLHttpRequest header to avoid bot detection
_TM_EXTRA_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.transfermarkt.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class TransfermarktCollector(BaseCollector):
    """Collector for Transfermarkt injury reports and fixtures."""

    source_name = "transfermarkt"
    base_url = "https://www.transfermarkt.com"
    request_delay = 4.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect injury or fixture data from Transfermarkt.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all leagues.
        mode : str, optional
            Collection mode: "injuries" (default) or "fixtures".
        """
        league: Optional[str] = kwargs.get("league")
        mode: str = kwargs.get("mode", "injuries")

        leagues = (
            {league: TRANSFERMARKT_LEAGUES[league]}
            if league
            else TRANSFERMARKT_LEAGUES
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_meta in leagues.items():
            try:
                if mode == "injuries":
                    df = self._collect_injuries(league_name, league_meta)
                elif mode == "fixtures":
                    df = self._collect_fixtures(league_name, league_meta)
                else:
                    raise ValueError(f"Unknown mode: {mode!r}. Use 'injuries' or 'fixtures'.")

                if df is not None and not df.empty:
                    frames.append(df)

            except Exception as e:
                logger.error(
                    f"Failed to collect {league_name} ({mode}) from Transfermarkt: {e}"
                )
                # Per-league error — continue with other leagues
                continue

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    #  Injuries                                                            #
    # ------------------------------------------------------------------ #

    def _collect_injuries(
        self, league_name: str, league_meta: dict[str, str]
    ) -> Optional[pd.DataFrame]:
        """Scrape the injury page for a single league."""
        path = league_meta["path"]
        league_id = league_meta["id"]
        url = f"{self.base_url}/{path}/verletzungen/wettbewerb/{league_id}"

        cache_key = f"transfermarkt_injuries_{league_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[transfermarkt] Cache hit for injuries {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url, headers=_TM_EXTRA_HEADERS)
        html = response.text

        self._check_structure(html, ["items", "verletzungen"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        # Transfermarkt injury table has class="items"
        table = soup.select_one("table.items")
        if not table:
            logger.warning(f"[transfermarkt] No injury table found for {league_name}")
            return None

        for tr in table.select("tbody tr"):
            row = self._parse_injury_row(tr, league_name)
            if row:
                rows.append(row)

        if rows:
            # Apply team name normalisation
            for r in rows:
                r["team_id"] = self.normalise_team(r["team"])
            self._set_cache(cache_key, rows)

        logger.info(f"[transfermarkt] Fetched injuries {league_name}: {len(rows)} players")
        return pd.DataFrame(rows) if rows else None

    def _parse_injury_row(self, tr: Any, league_name: str) -> Optional[dict]:
        """Parse a single row from the Transfermarkt injury table.

        Parameters
        ----------
        tr : bs4.element.Tag
            A ``<tr>`` element from the ``table.items`` injury table.
        league_name : str
            Name of the league for this entry.

        Returns
        -------
        dict or None
            Parsed injury data, or None if the row cannot be parsed.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 5:
                return None

            # Player name — usually in .hauptlink anchor
            player = None
            player_el = tr.select_one(".hauptlink a")
            if player_el:
                player = player_el.get_text(strip=True)
            if not player:
                return None

            # Team name — second .hauptlink (team link) or .zentriert with team img
            team = None
            hauptlinks = tr.select(".hauptlink a")
            if len(hauptlinks) >= 2:
                team = hauptlinks[1].get_text(strip=True)
            # Fallback: look for inline team name cell
            if not team:
                for td in tds:
                    img = td.select_one("img.tiny_wappen, img[alt]")
                    if img and img.get("alt"):
                        team = img["alt"]
                        break

            # Injury type — typically in a td after the player/team block
            injury_type = None
            # Transfermarkt uses a dedicated column for injury description
            for td in tds:
                cls = " ".join(td.get("class", []))
                text = td.get_text(strip=True)
                if "zentriert" not in cls and text and len(text) > 3 and td != tds[0]:
                    # Skip cells we already captured
                    if team and text == team:
                        continue
                    if player and text == player:
                        continue
                    # Injury descriptions are usually a few words
                    if not any(c.isdigit() for c in text[:3]):
                        injury_type = text
                        break

            # Dates — look for cells with date patterns (dd.mm.yyyy or similar)
            since_date = None
            expected_return = None
            date_cells: list[str] = []
            for td in tds:
                text = td.get_text(strip=True)
                # Simple heuristic: contains a dot-separated or slash-separated date
                if (
                    text.count(".") >= 2
                    or text.count("/") >= 2
                    or (len(text) == 10 and "-" in text)
                ):
                    date_cells.append(text)

            if len(date_cells) >= 1:
                since_date = date_cells[0]
            if len(date_cells) >= 2:
                expected_return = date_cells[1]

            # Market value — look for cell containing € symbol
            market_value = None
            for td in tds:
                text = td.get_text(strip=True)
                if "€" in text or "m" in text.lower():
                    market_value = text
                    break

            return {
                "player": player,
                "team": team or "",
                "injury_type": injury_type or "",
                "since_date": since_date or "",
                "expected_return": expected_return or "",
                "market_value": market_value or "",
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[transfermarkt] Failed to parse injury row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Fixtures                                                            #
    # ------------------------------------------------------------------ #

    def _collect_fixtures(
        self, league_name: str, league_meta: dict[str, str]
    ) -> Optional[pd.DataFrame]:
        """Scrape the upcoming fixtures page for a single league."""
        path = league_meta["path"]
        league_id = league_meta["id"]
        url = f"{self.base_url}/{path}/spieltag/wettbewerb/{league_id}"

        cache_key = f"transfermarkt_fixtures_{league_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[transfermarkt] Cache hit for fixtures {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url, headers=_TM_EXTRA_HEADERS)
        html = response.text

        self._check_structure(html, ["spieltag", "items"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        # Fixtures are in .items table; rows alternate between header and match rows
        table = soup.select_one("table.items")
        if not table:
            logger.warning(f"[transfermarkt] No fixture table found for {league_name}")
            return None

        current_matchday = None
        current_date = None

        for tr in table.select("tr"):
            # Matchday header rows carry the round/spieltag info
            header_el = tr.select_one(".zentriert.spieltagsheader, th.zentriert")
            if header_el:
                current_matchday = header_el.get_text(strip=True)
                continue

            # Date rows — Transfermarkt often embeds date in its own row
            date_el = tr.select_one("td.zentriert[colspan]")
            if date_el:
                current_date = date_el.get_text(strip=True)
                continue

            row = self._parse_fixture_row(
                tr, league_name, current_matchday, current_date
            )
            if row:
                rows.append(row)

        if rows:
            for r in rows:
                r["home_team_id"] = self.normalise_team(r["home_team"])
                r["away_team_id"] = self.normalise_team(r["away_team"])
            self._set_cache(cache_key, rows)

        logger.info(f"[transfermarkt] Fetched fixtures {league_name}: {len(rows)} matches")
        return pd.DataFrame(rows) if rows else None

    def _parse_fixture_row(
        self,
        tr: Any,
        league_name: str,
        matchday: Optional[str],
        current_date: Optional[str],
    ) -> Optional[dict]:
        """Parse a single fixture row from the Transfermarkt spieltag table.

        Parameters
        ----------
        tr : bs4.element.Tag
            A ``<tr>`` element from the fixture table.
        league_name : str
            Name of the league for this fixture.
        matchday : str or None
            Current matchday header (e.g. "Matchday 32").
        current_date : str or None
            Date inherited from the most recent date row.

        Returns
        -------
        dict or None
            Parsed fixture data, or None if the row cannot be parsed.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            # Team names from .hauptlink anchors
            hauptlinks = tr.select(".hauptlink a")
            if len(hauptlinks) < 2:
                return None

            home_team = hauptlinks[0].get_text(strip=True)
            away_team = hauptlinks[1].get_text(strip=True)

            if not home_team or not away_team:
                return None

            # Kick-off time — look for a cell that resembles HH:MM
            kick_off = None
            match_date = current_date or ""
            for td in tds:
                text = td.get_text(strip=True)
                # Matches patterns like "20:45" or "3:00 PM"
                if ":" in text and len(text) <= 8:
                    kick_off = text
                    break
                # Date cell — dd.mm.yyyy or similar
                if (
                    (text.count(".") >= 2 or text.count("/") >= 2)
                    and not match_date
                ):
                    match_date = text

            # Venue — occasionally present in a dedicated cell
            venue = None
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "ort" in cls or "stadium" in cls.lower():
                    venue = td.get_text(strip=True)
                    break

            return {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "kick_off": kick_off or "",
                "venue": venue or "",
                "matchday": matchday or "",
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[transfermarkt] Failed to parse fixture row: {e}")
            return None
