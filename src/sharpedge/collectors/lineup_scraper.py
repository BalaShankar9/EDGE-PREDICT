"""
LineupScraper — confirmed and predicted team lineups.

The 30-minute window between lineup release and kickoff is the sharpest
edge in football betting.  Knowing that Haaland or Salah is benched
before the bookmakers reprice is where value is extracted.

Sources (all free / public):
  1. Rotowire  — https://www.rotowire.com/soccer/lineups.php
     Predicted + confirmed lineups with formation and injury notes.
  2. FotMob    — https://www.fotmob.com/api/matchDetails?matchId={id}
     JSON API with confirmed starters / subs ~1 hour before kickoff.
  3. FlashScore match-detail pages
     Lineup section embedded in match HTML.

_collect() row schema
---------------------
  match_id, home_team, away_team, match_date
  home_lineup        list[str]   starting 11
  away_lineup        list[str]   starting 11
  home_formation     str         e.g. "4-3-3"
  away_formation     str
  home_subs          list[str]   bench players
  away_subs          list[str]
  lineup_status      str         "predicted" | "confirmed"
  key_absences_home  list[str]   expected starters not in lineup
  key_absences_away  list[str]
  source             str
  scraped_at         str         ISO-8601 UTC timestamp
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# FotMob match IDs to poll (populated via kwargs or a seed list)
_DEFAULT_FOTMOB_IDS: list[int] = []

# FlashScore match paths to try
_FLASHSCORE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Referer": "https://www.flashscore.com/",
}

# Rotowire lineup page
_ROTOWIRE_URL = "https://www.rotowire.com/soccer/lineups.php"

# FotMob match detail endpoint
_FOTMOB_MATCH_URL = "https://www.fotmob.com/api/matchDetails?matchId={match_id}"

# Rotowire status markers
_CONFIRMED_MARKERS = {"confirmed", "official", "starting xi"}

# Formation regex: digits separated by hyphens, e.g. 4-3-3 or 4-2-3-1
_FORMATION_RE = re.compile(r"\b(\d[-\d]{2,})\b")


class LineupScraper(BaseCollector):
    """Collector for confirmed and predicted football lineups.

    The sharpest 30-minute window opens the moment lineups are released.
    This collector aggregates data from Rotowire (predicted/confirmed),
    FotMob (confirmed JSON), and FlashScore (confirmed HTML).
    """

    source_name = "lineup_scraper"
    base_url = ""
    request_delay = 3.0

    # ------------------------------------------------------------------ #
    #  Public helper — key absence detection                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_key_absences(
        actual_lineup: list[str],
        expected_starters: list[str],
    ) -> list[str]:
        """Identify players expected to start who are absent from the lineup.

        This is the primary edge signal: discovering that a high-impact
        player (e.g. Haaland, Salah, Mbappe) is not in the starting XI
        before bookmakers have adjusted their prices.

        Parameters
        ----------
        actual_lineup : list[str]
            Player names actually starting (normalised to lower-case).
        expected_starters : list[str]
            Players who normally start for this team.

        Returns
        -------
        list[str]
            Players present in expected_starters but absent from actual_lineup.
        """
        actual_lower = {p.lower().strip() for p in actual_lineup}
        return [
            player
            for player in expected_starters
            if player.lower().strip() not in actual_lower
        ]

    # ------------------------------------------------------------------ #
    #  Main collection entry-point                                        #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect lineup data from all configured sources.

        Parameters
        ----------
        fotmob_match_ids : list[int], optional
            FotMob match IDs to query directly via API.
        expected_starters : dict[str, list[str]], optional
            Mapping of normalised team name -> list of typical starters.
            Used to populate key_absences_home / key_absences_away.
        source : str, optional
            Restrict collection to a single source: "rotowire", "fotmob",
            or "flashscore".  Default: all sources.

        Returns
        -------
        pd.DataFrame
            One row per match, with lineup, formation, bench, and absence
            columns as described in the module docstring.
        """
        fotmob_ids: list[int] = kwargs.get(
            "fotmob_match_ids", _DEFAULT_FOTMOB_IDS
        )
        expected_starters: dict[str, list[str]] = kwargs.get(
            "expected_starters", {}
        )
        restrict_source: Optional[str] = kwargs.get("source")

        frames: list[pd.DataFrame] = []

        # --- Rotowire ---------------------------------------------------
        if restrict_source in (None, "rotowire"):
            try:
                df_rw = self._collect_rotowire()
                if not df_rw.empty:
                    frames.append(df_rw)
            except Exception as exc:
                logger.error(f"[{self.source_name}] Rotowire collection failed: {exc}")

        # --- FotMob JSON API -------------------------------------------
        if restrict_source in (None, "fotmob") and fotmob_ids:
            for mid in fotmob_ids:
                try:
                    row = self._collect_fotmob_match(mid)
                    if row:
                        frames.append(pd.DataFrame([row]))
                except Exception as exc:
                    logger.error(
                        f"[{self.source_name}] FotMob match {mid} failed: {exc}"
                    )

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Enrich with key-absence data using provided expected starters
        if expected_starters:
            df["key_absences_home"] = df.apply(
                lambda r: self.get_key_absences(
                    r.get("home_lineup") or [],
                    expected_starters.get(
                        self.normalise_team(str(r.get("home_team", ""))) or "", []
                    ),
                ),
                axis=1,
            )
            df["key_absences_away"] = df.apply(
                lambda r: self.get_key_absences(
                    r.get("away_lineup") or [],
                    expected_starters.get(
                        self.normalise_team(str(r.get("away_team", ""))) or "", []
                    ),
                ),
                axis=1,
            )
        else:
            # Default to empty lists when no expected-starters provided
            if "key_absences_home" not in df.columns:
                df["key_absences_home"] = [[] for _ in range(len(df))]
            if "key_absences_away" not in df.columns:
                df["key_absences_away"] = [[] for _ in range(len(df))]

        # Normalise team name columns
        df["home_team_id"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        df["away_team_id"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x))
        )

        return df

    # ------------------------------------------------------------------ #
    #  Rotowire                                                           #
    # ------------------------------------------------------------------ #

    def _collect_rotowire(self) -> pd.DataFrame:
        """Scrape Rotowire's soccer lineups page.

        Returns
        -------
        pd.DataFrame
            One row per match found on the page.
        """
        cache_key = f"rotowire_lineups_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[{self.source_name}] Rotowire cache hit")
            return pd.DataFrame(cached)

        response = self._fetch(_ROTOWIRE_URL)
        html = response.text
        self._check_structure(html, ["lineup", "player"])

        rows = self._parse_rotowire_html(html)

        if not rows:
            logger.warning(f"[{self.source_name}] Rotowire returned 0 lineup rows")
            return pd.DataFrame()

        logger.info(f"[{self.source_name}] Rotowire: {len(rows)} matches parsed")
        self._set_cache(cache_key, rows)
        return pd.DataFrame(rows)

    def _parse_rotowire_html(self, html: str) -> list[dict]:
        """Parse all match lineup cards from Rotowire's lineup page.

        Rotowire renders each match as a ``div.lineup__main`` block
        containing home and away lineup columns.

        Parameters
        ----------
        html : str
            Raw HTML from https://www.rotowire.com/soccer/lineups.php.

        Returns
        -------
        list[dict]
            Parsed match dicts.
        """
        soup = BeautifulSoup(html, "html.parser")
        scraped_at = datetime.now(timezone.utc).isoformat()

        # Each match block — Rotowire wraps each game in a container
        # with class variations; try several selectors
        match_blocks = soup.select(
            "div.lineup__main, div[class*='lineup-card'], "
            "div[class*='lineups__matchup'], section[class*='lineup']"
        )

        if not match_blocks:
            # Fallback: find all elements containing formation patterns
            match_blocks = soup.select("div[class*='lineup']")

        rows: list[dict] = []
        for block in match_blocks:
            row = self._parse_rotowire_block(block, scraped_at)
            if row:
                rows.append(row)

        return rows

    def _parse_rotowire_block(
        self, block: Any, scraped_at: str
    ) -> Optional[dict]:
        """Parse a single Rotowire match lineup card.

        Parameters
        ----------
        block : bs4.element.Tag
            A div element representing one match.
        scraped_at : str
            ISO-8601 UTC timestamp for when the scrape ran.

        Returns
        -------
        dict or None
            Parsed lineup data, or None if not parseable.
        """
        try:
            # Team names — look for team header elements
            team_els = block.select(
                ".lineup__team, [class*='lineup__team'], "
                "[class*='team-name'], h2, h3"
            )
            team_names = [el.get_text(strip=True) for el in team_els if el.get_text(strip=True)]
            # Deduplicate while preserving order, take first two
            seen: set[str] = set()
            unique_teams: list[str] = []
            for name in team_names:
                if name not in seen and len(name) > 1:
                    seen.add(name)
                    unique_teams.append(name)
                if len(unique_teams) == 2:
                    break

            if len(unique_teams) < 2:
                return None

            home_team, away_team = unique_teams[0], unique_teams[1]

            # Match date / time
            match_date: Optional[str] = None
            date_el = block.select_one(
                "[class*='game-time'], [class*='matchup__date'], "
                "[class*='lineup__time'], time"
            )
            if date_el:
                match_date = date_el.get_text(strip=True)
                # Try to parse datetime attr if present
                dt_attr = date_el.get("datetime") or date_el.get("data-time")
                if dt_attr:
                    match_date = str(dt_attr)

            # Formation
            home_formation = self._extract_formation(block, side="home")
            away_formation = self._extract_formation(block, side="away")

            # Lineup status — confirmed vs predicted
            status_text = block.get_text().lower()
            lineup_status = "confirmed" if any(
                marker in status_text for marker in _CONFIRMED_MARKERS
            ) else "predicted"

            # Player lists — home and away columns
            home_lineup, home_subs = self._extract_players(block, side="home")
            away_lineup, away_subs = self._extract_players(block, side="away")

            # Injury / absence notes
            key_absences_home = self._extract_absences(block, side="home")
            key_absences_away = self._extract_absences(block, side="away")

            return {
                "match_id": None,
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "home_lineup": home_lineup,
                "away_lineup": away_lineup,
                "home_formation": home_formation,
                "away_formation": away_formation,
                "home_subs": home_subs,
                "away_subs": away_subs,
                "lineup_status": lineup_status,
                "key_absences_home": key_absences_home,
                "key_absences_away": key_absences_away,
                "source": "rotowire",
                "scraped_at": scraped_at,
            }

        except Exception as exc:
            logger.debug(f"[{self.source_name}] Rotowire block parse failed: {exc}")
            return None

    def _extract_formation(self, block: Any, side: str) -> Optional[str]:
        """Extract formation string (e.g. '4-3-3') for the given side.

        Parameters
        ----------
        block : bs4.element.Tag
            The match block element.
        side : str
            'home' or 'away'.

        Returns
        -------
        str or None
            Formation string if found.
        """
        # Look for formation-specific elements
        side_class_hint = "home" if side == "home" else "away"
        formation_els = block.select(
            f"[class*='{side_class_hint}'][class*='formation'], "
            f"[class*='formation'], [data-formation]"
        )
        for el in formation_els:
            text = el.get("data-formation") or el.get_text(strip=True)
            m = _FORMATION_RE.search(str(text))
            if m:
                return m.group(1)

        # Fallback: scan all text in the block half
        # Split block text approximately in half
        full_text = block.get_text()
        half = len(full_text) // 2
        segment = full_text[:half] if side == "home" else full_text[half:]
        m = _FORMATION_RE.search(segment)
        if m:
            return m.group(1)

        return None

    def _extract_players(
        self, block: Any, side: str
    ) -> tuple[list[str], list[str]]:
        """Extract the starting XI and bench list for the given side.

        Parameters
        ----------
        block : bs4.element.Tag
            The match block element.
        side : str
            'home' or 'away'.

        Returns
        -------
        tuple[list[str], list[str]]
            (starters, subs) — each a list of player name strings.
        """
        starters: list[str] = []
        subs: list[str] = []

        # Try side-specific containers first
        side_hint = "home" if side == "home" else "away"
        containers = block.select(
            f"[class*='{side_hint}'], [class*='lineup__col']"
        )

        player_els: list[Any] = []
        sub_els: list[Any] = []

        for container in containers:
            # Starters
            player_els.extend(
                container.select(
                    "[class*='player-name'], [class*='lineup__player'], "
                    "li[class*='starter'], li[class*='player']"
                )
            )
            # Subs / bench
            sub_els.extend(
                container.select(
                    "[class*='bench'], [class*='sub'], li[class*='bench']"
                )
            )

        seen: set[str] = set()
        for el in player_els:
            name = el.get_text(strip=True)
            if name and name not in seen and len(name) > 2:
                seen.add(name)
                starters.append(name)
                if len(starters) >= 11:
                    break

        sub_seen: set[str] = set()
        for el in sub_els:
            name = el.get_text(strip=True)
            if name and name not in sub_seen and len(name) > 2:
                sub_seen.add(name)
                subs.append(name)

        return starters, subs

    def _extract_absences(self, block: Any, side: str) -> list[str]:
        """Extract players listed as injured / doubtful / ruled-out.

        Parameters
        ----------
        block : bs4.element.Tag
            Match block element.
        side : str
            'home' or 'away'.

        Returns
        -------
        list[str]
            Player names flagged as absent / injured for this side.
        """
        absences: list[str] = []
        side_hint = "home" if side == "home" else "away"

        # Look for injury-tagged player elements within the side container
        injury_keywords = frozenset(
            ["injured", "out", "doubtful", "doubt", "unavailable", "miss"]
        )
        containers = block.select(f"[class*='{side_hint}']")
        for container in containers:
            injured_els = container.select(
                "[class*='injured'], [class*='out'], [class*='doubt'], "
                "[class*='unavailable'], [class*='gtd']"
            )
            for el in injured_els:
                # Try to get just the player name from within the element
                name_el = el.select_one("[class*='name'], a, span")
                name = (name_el or el).get_text(strip=True)
                if name and len(name) > 2:
                    absences.append(name)
            # Also scan text for injury mentions
            for tag in container.select("li, p, span"):
                text = tag.get_text(strip=True).lower()
                if any(kw in text for kw in injury_keywords):
                    name_candidate = tag.get_text(strip=True)
                    # Heuristic: short entries (1-3 words) are likely player names
                    words = name_candidate.split()
                    if 1 <= len(words) <= 4 and name_candidate not in absences:
                        absences.append(name_candidate)

        return absences

    # ------------------------------------------------------------------ #
    #  FotMob JSON API                                                    #
    # ------------------------------------------------------------------ #

    def _collect_fotmob_match(self, match_id: int) -> Optional[dict]:
        """Fetch and parse a single FotMob match's lineup data.

        FotMob provides a JSON endpoint that returns confirmed lineups
        roughly 1 hour before kickoff.

        Parameters
        ----------
        match_id : int
            FotMob internal match identifier.

        Returns
        -------
        dict or None
            Parsed lineup row, or None on failure.
        """
        cache_key = f"fotmob_lineup_{match_id}_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[{self.source_name}] FotMob cache hit: {match_id}")
            return cached

        url = _FOTMOB_MATCH_URL.format(match_id=match_id)
        try:
            data = self._fetch_json(
                url,
                headers={
                    "Accept": "application/json",
                    "Referer": "https://www.fotmob.com/",
                    "x-mas": "eyJib2R5Ijp7InVybCI6Ii9hcGkvbGVhZ3VlcyIsImNvZGUiOjB9fQ==",
                },
            )
        except Exception as exc:
            logger.warning(f"[{self.source_name}] FotMob fetch failed for {match_id}: {exc}")
            return None

        row = self._parse_fotmob_data(data, match_id)
        if row:
            self._set_cache(cache_key, row)
        return row

    def _parse_fotmob_data(
        self, data: dict, match_id: int
    ) -> Optional[dict]:
        """Parse FotMob match detail JSON into a lineup row.

        FotMob JSON structure (relevant paths):
          data["lineup"]["homeTeam"]["starters"]  -> list of player dicts
          data["lineup"]["homeTeam"]["subs"]       -> list of player dicts
          data["lineup"]["awayTeam"]["starters"]
          data["lineup"]["awayTeam"]["subs"]
          data["general"]["homeTeam"]["name"]
          data["general"]["awayTeam"]["name"]

        Parameters
        ----------
        data : dict
            Parsed JSON from the FotMob matchDetails endpoint.
        match_id : int
            FotMob match ID (for logging).

        Returns
        -------
        dict or None
            Normalised lineup row, or None if lineups not yet available.
        """
        try:
            general = data.get("general", {})
            home_team = general.get("homeTeam", {}).get("name", "")
            away_team = general.get("awayTeam", {}).get("name", "")
            match_date = general.get("matchTimeUTCDate") or general.get("matchDate")

            lineup_data = data.get("lineup", {})
            if not lineup_data:
                logger.debug(f"[{self.source_name}] FotMob {match_id}: no lineup data yet")
                return None

            home_section = lineup_data.get("homeTeam", {})
            away_section = lineup_data.get("awayTeam", {})

            def _extract_names(player_list: list) -> list[str]:
                names = []
                for p in player_list or []:
                    name = (
                        p.get("name")
                        or p.get("playerName")
                        or p.get("shortName")
                        or ""
                    )
                    if name:
                        names.append(str(name))
                return names

            home_lineup = _extract_names(home_section.get("starters", []))
            home_subs = _extract_names(home_section.get("subs", []))
            away_lineup = _extract_names(away_section.get("starters", []))
            away_subs = _extract_names(away_section.get("subs", []))

            home_formation = (
                home_section.get("formation")
                or lineup_data.get("homeTeamFormation")
                or None
            )
            away_formation = (
                away_section.get("formation")
                or lineup_data.get("awayTeamFormation")
                or None
            )

            # If we have starters it's confirmed; else predicted
            lineup_status = "confirmed" if home_lineup else "predicted"

            return {
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "home_lineup": home_lineup,
                "away_lineup": away_lineup,
                "home_formation": home_formation,
                "away_formation": away_formation,
                "home_subs": home_subs,
                "away_subs": away_subs,
                "lineup_status": lineup_status,
                "key_absences_home": [],
                "key_absences_away": [],
                "source": "fotmob",
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            logger.debug(f"[{self.source_name}] FotMob parse error for {match_id}: {exc}")
            return None
