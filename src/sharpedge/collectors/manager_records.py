"""
ManagerRecords Collector — head coach appointment dates and career records.

The "new manager bounce" is real and well-documented: teams perform ~12%
better in the first 8 matches after a managerial change (+0.5 goals/game).
This feature is a significant edge for in-season prediction models.

Primary source: Transfermarkt
  List:    /[league]/trainer/wettbewerb/[league_id]
  Profile: /[name]/profil/trainer/[id]
"""

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

LEAGUE_IDS: dict[str, str] = {
    "Premier League": "GB1",
    "La Liga": "ES1",
    "Bundesliga": "L1",
    "Serie A": "IT1",
    "Ligue 1": "FR1",
    "Championship": "GB2",
    "Eredivisie": "NL1",
}

LEAGUE_SLUGS: dict[str, str] = {
    "Premier League": "premier-league",
    "La Liga": "laliga",
    "Bundesliga": "1-bundesliga",
    "Serie A": "serie-a",
    "Ligue 1": "ligue-1",
    "Championship": "championship",
    "Eredivisie": "eredivisie",
}

# New manager bounce threshold — matches and calendar days
NEW_MANAGER_BOUNCE_MATCHES = 8
NEW_MANAGER_BOUNCE_DAYS = 60

_TM_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.transfermarkt.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Date formats used by Transfermarkt
_DATE_FORMATS = [
    "%b %d, %Y",   # Jan 15, 2024
    "%d.%m.%Y",    # 15.01.2024
    "%Y-%m-%d",    # 2024-01-15
    "%B %d, %Y",   # January 15, 2024
    "%d/%m/%Y",    # 15/01/2024
]


class ManagerRecordsCollector(BaseCollector):
    """Collector for manager/head coach appointment dates and career records."""

    source_name = "manager_records"
    base_url = "https://www.transfermarkt.com"
    request_delay = 4.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect manager records.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all leagues.
        per_manager : bool, optional
            If True (default), fetches individual manager profile pages for
            full career records. Set to False for list-only (faster).
        """
        league: Optional[str] = kwargs.get("league")
        per_manager: bool = kwargs.get("per_manager", True)

        leagues = (
            {league: LEAGUE_IDS[league]}
            if league
            else LEAGUE_IDS
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_id in leagues.items():
            try:
                df = self._collect_league(
                    league_name, league_id, per_manager=per_manager
                )
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.error(
                    f"[manager_records] Failed to collect {league_name}: {e}"
                )
                continue

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  League-level collection                                             #
    # ------------------------------------------------------------------ #

    def _collect_league(
        self,
        league_name: str,
        league_id: str,
        per_manager: bool,
    ) -> Optional[pd.DataFrame]:
        """Fetch current managers for a league, then optionally drill into profiles."""
        slug = LEAGUE_SLUGS.get(league_name, league_name.lower().replace(" ", "-"))
        url = f"{self.base_url}/{slug}/trainer/wettbewerb/{league_id}"

        cache_key = f"manager_list_{league_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[manager_records] Cache hit for list {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url, headers=_TM_HEADERS)
        html = response.text

        self._check_structure(html, ["trainer"])

        soup = BeautifulSoup(html, "html.parser")
        manager_stubs = self._parse_manager_list(soup, league_name)

        if not manager_stubs:
            logger.warning(f"[manager_records] No managers parsed for {league_name}")
            return None

        rows: list[dict] = []
        if per_manager:
            for stub in manager_stubs:
                try:
                    profile = self._fetch_manager_profile(stub, league_name)
                    rows.append(profile if profile else stub)
                except Exception as e:
                    logger.debug(
                        f"[manager_records] Profile failed for "
                        f"{stub.get('manager_name', '?')}: {e}"
                    )
                    rows.append(stub)
        else:
            rows = manager_stubs

        # Compute new manager bounce flags for all rows
        today = datetime.now(timezone.utc).date()
        for r in rows:
            r.update(self._compute_bounce_flags(r, today))

        if rows:
            self._set_cache(cache_key, rows)

        logger.info(
            f"[manager_records] Fetched managers {league_name}: {len(rows)} managers"
        )
        return pd.DataFrame(rows) if rows else None

    # ------------------------------------------------------------------ #
    #  Parse manager list page                                             #
    # ------------------------------------------------------------------ #

    def _parse_manager_list(
        self, soup: BeautifulSoup, league_name: str
    ) -> list[dict]:
        """Parse the TM trainer list table — one row per club."""
        table = soup.select_one("table.items")
        if not table:
            logger.warning(f"[manager_records] No items table for {league_name}")
            return []

        stubs: list[dict] = []
        for tr in table.select("tbody tr"):
            stub = self._parse_manager_list_row(tr, league_name)
            if stub:
                stubs.append(stub)
        return stubs

    def _parse_manager_list_row(self, tr: Any, league_name: str) -> Optional[dict]:
        """Parse one row from the TM trainer list table.

        TM trainer list columns (typical): Club | Manager | Nationality |
        Age | Appointed | Contract Until
        """
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            # Club name
            team = ""
            club_links = tr.select("td.hauptlink a, .no-border-links a")
            if club_links:
                team = club_links[0].get_text(strip=True)

            # Manager name and profile link
            manager_name = ""
            tm_id = ""
            tm_slug = ""
            trainer_link = tr.select_one("a[href*='/profil/trainer/'], a[href*='trainer']")
            if trainer_link:
                manager_name = trainer_link.get_text(strip=True)
                href = trainer_link.get("href", "")
                id_match = re.search(r"/trainer/(\d+)", href)
                tm_id = id_match.group(1) if id_match else ""
                parts = href.strip("/").split("/")
                tm_slug = parts[0] if parts else ""

            if not manager_name:
                # fallback: second hauptlink
                hauptlinks = tr.select(".hauptlink a")
                if len(hauptlinks) >= 2:
                    manager_name = hauptlinks[1].get_text(strip=True)
                elif hauptlinks:
                    manager_name = hauptlinks[0].get_text(strip=True)

            if not manager_name:
                return None

            # Nationality
            nationality = ""
            flag = tr.select_one("img.flaggenrahmen, img[class*='flag']")
            if flag:
                nationality = flag.get("title", flag.get("alt", ""))

            # Appointed date — look for a cell with a recognisable date string
            appointed_raw = ""
            contract_until = ""
            for td in tds:
                text = td.get_text(strip=True)
                cls = " ".join(td.get("class", []))
                if not text or len(text) < 6:
                    continue
                if "zentriert" in cls or "datum" in cls:
                    parsed = _parse_date(text)
                    if parsed:
                        if not appointed_raw:
                            appointed_raw = text
                        elif not contract_until:
                            contract_until = text

            appointed_date = _parse_date(appointed_raw)

            return {
                "manager_name": manager_name,
                "team": team,
                "team_id": self.normalise_team(team),
                "tm_id": tm_id,
                "tm_slug": tm_slug,
                "nationality": nationality,
                "appointed_date": appointed_date.isoformat() if appointed_date else None,
                "contract_until": contract_until,
                "league": league_name,
                # Profile stats filled in by _fetch_manager_profile
                "days_in_charge": None,
                "career_matches": None,
                "career_wins": None,
                "career_draws": None,
                "career_losses": None,
                "career_win_pct": None,
                "current_team_matches": None,
                "current_team_wins": None,
                "current_team_win_pct": None,
                "previous_team": None,
                "previous_team_sack_date": None,
                "is_new_manager": None,
                "new_manager_bounce_expected": None,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[manager_records] Failed to parse list row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Per-manager profile                                                 #
    # ------------------------------------------------------------------ #

    def _fetch_manager_profile(
        self, stub: dict, league_name: str
    ) -> Optional[dict]:
        """Fetch and parse a manager's Transfermarkt profile page."""
        tm_slug = stub.get("tm_slug", "")
        tm_id = stub.get("tm_id", "")
        if not tm_slug or not tm_id:
            return stub

        url = f"{self.base_url}/{tm_slug}/profil/trainer/{tm_id}"
        cache_key = f"manager_profile_{tm_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            response = self._fetch(url, headers=_TM_HEADERS)
            html = response.text
        except Exception as e:
            logger.debug(f"[manager_records] Profile fetch failed {url}: {e}")
            return stub

        soup = BeautifulSoup(html, "html.parser")
        enriched = dict(stub)

        # Career stats table — TM shows coaching history with W/D/L per stint
        career_rows = self._parse_coaching_history(soup, stub.get("team", ""))
        if career_rows:
            total_matches = sum(r.get("matches", 0) or 0 for r in career_rows)
            total_wins = sum(r.get("wins", 0) or 0 for r in career_rows)
            total_draws = sum(r.get("draws", 0) or 0 for r in career_rows)
            total_losses = sum(r.get("losses", 0) or 0 for r in career_rows)

            enriched["career_matches"] = total_matches
            enriched["career_wins"] = total_wins
            enriched["career_draws"] = total_draws
            enriched["career_losses"] = total_losses
            enriched["career_win_pct"] = (
                round(total_wins / total_matches * 100, 1)
                if total_matches > 0 else None
            )

            # Current team stint — last entry whose club matches stub["team"]
            current_team = stub.get("team", "")
            current_rows = [
                r for r in career_rows
                if _team_match(r.get("club", ""), current_team)
            ]
            if current_rows:
                # Most recent stint at current club
                cr = current_rows[-1]
                enriched["current_team_matches"] = cr.get("matches")
                enriched["current_team_wins"] = cr.get("wins")
                cm = cr.get("matches") or 0
                cw = cr.get("wins") or 0
                enriched["current_team_win_pct"] = (
                    round(cw / cm * 100, 1) if cm > 0 else None
                )

            # Previous team — last entry whose club is NOT the current team
            prev_rows = [
                r for r in career_rows
                if not _team_match(r.get("club", ""), current_team)
            ]
            if prev_rows:
                prev = prev_rows[-1]
                enriched["previous_team"] = prev.get("club")
                enriched["previous_team_sack_date"] = prev.get("end_date")

        # Appointed date — if not found on list page, check profile info box
        if not enriched.get("appointed_date"):
            appointed_raw = self._extract_info_box(soup, ["In office since", "Im Amt seit", "Appointed"])
            if appointed_raw:
                parsed = _parse_date(appointed_raw)
                enriched["appointed_date"] = parsed.isoformat() if parsed else None

        self._set_cache(cache_key, enriched)
        return enriched

    def _parse_coaching_history(
        self, soup: BeautifulSoup, current_team: str
    ) -> list[dict]:
        """Parse the coaching history table from a TM manager profile page."""
        rows: list[dict] = []
        # TM coaching history is in table.items or a div with id "trainerStationen"
        table = soup.select_one(
            "div#trainerStationen table, table#trainerStationen, table.items"
        )
        if not table:
            return rows

        for tr in table.select("tbody tr"):
            entry = self._parse_coaching_row(tr)
            if entry:
                rows.append(entry)

        return rows

    def _parse_coaching_row(self, tr: Any) -> Optional[dict]:
        """Parse one row from the TM coaching history table.

        Expected columns: Club | From | To | Matches | W | D | L | Points
        """
        try:
            tds = tr.select("td")
            if len(tds) < 4:
                return None

            # Club name
            club = ""
            club_el = tr.select_one(".hauptlink a")
            if club_el:
                club = club_el.get_text(strip=True)
            if not club:
                for td in tds:
                    a = td.select_one("a")
                    if a:
                        club = a.get_text(strip=True)
                        break

            # Extract all numeric-looking centred cells
            centered: list[str] = []
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "zentriert" in cls:
                    centered.append(td.get_text(strip=True))

            # Date cells — look for anything parseable as a date
            dates: list[str] = []
            for td in tds:
                text = td.get_text(strip=True)
                if _parse_date(text):
                    dates.append(text)

            start_date = dates[0] if len(dates) > 0 else ""
            end_date = dates[1] if len(dates) > 1 else ""

            # Numeric stats: try to parse matches, W, D, L from centered cells
            numeric = []
            for c in centered:
                n = _safe_int(c)
                if n is not None:
                    numeric.append(n)

            return {
                "club": club,
                "start_date": start_date,
                "end_date": end_date,
                "matches": numeric[0] if len(numeric) > 0 else None,
                "wins": numeric[1] if len(numeric) > 1 else None,
                "draws": numeric[2] if len(numeric) > 2 else None,
                "losses": numeric[3] if len(numeric) > 3 else None,
            }

        except Exception as e:
            logger.debug(f"[manager_records] Failed to parse coaching row: {e}")
            return None

    def _extract_info_box(
        self, soup: BeautifulSoup, labels: list[str]
    ) -> str:
        """Extract a value from the TM profile info box by label."""
        for label in labels:
            el = soup.find(string=re.compile(re.escape(label), re.IGNORECASE))
            if el:
                parent = el.find_parent()
                if parent:
                    nxt = parent.find_next_sibling()
                    if nxt:
                        return nxt.get_text(strip=True)
        return ""

    # ------------------------------------------------------------------ #
    #  New manager bounce computation                                      #
    # ------------------------------------------------------------------ #

    def _compute_bounce_flags(
        self, row: dict, today: Any
    ) -> dict:
        """Compute is_new_manager and new_manager_bounce_expected flags.

        Parameters
        ----------
        row : dict
            Manager record with appointed_date and current_team_matches.
        today : date
            Today's date for age-of-appointment calculation.

        Returns
        -------
        dict with keys: days_in_charge, is_new_manager,
                        new_manager_bounce_expected.
        """
        out: dict = {
            "days_in_charge": None,
            "is_new_manager": False,
            "new_manager_bounce_expected": False,
        }

        appointed_str = row.get("appointed_date")
        if not appointed_str:
            return out

        try:
            appointed = datetime.fromisoformat(appointed_str).date()
        except (ValueError, TypeError):
            return out

        days_in_charge = (today - appointed).days
        out["days_in_charge"] = days_in_charge

        # is_new_manager: appointed within the last NEW_MANAGER_BOUNCE_DAYS days
        out["is_new_manager"] = days_in_charge <= NEW_MANAGER_BOUNCE_DAYS

        # new_manager_bounce_expected: fewer than NEW_MANAGER_BOUNCE_MATCHES games
        # since appointment (the bounce window is ~8 matches)
        current_matches = row.get("current_team_matches")
        if current_matches is not None:
            out["new_manager_bounce_expected"] = (
                current_matches < NEW_MANAGER_BOUNCE_MATCHES
                and out["is_new_manager"]
            )
        else:
            # Without match count, use calendar days as proxy (< 60 days)
            out["new_manager_bounce_expected"] = out["is_new_manager"]

        return out


# ------------------------------------------------------------------ #
#  Parsing helpers                                                     #
# ------------------------------------------------------------------ #

def _parse_date(raw: str) -> Optional[Any]:
    """Try to parse a date string using known TM date formats."""
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


def _safe_int(val: Any) -> Optional[int]:
    """Safely convert to int, stripping non-numeric chars."""
    if val is None:
        return None
    try:
        cleaned = re.sub(r"[^\d-]", "", str(val))
        return int(cleaned) if cleaned and cleaned != "-" else None
    except (ValueError, TypeError):
        return None


def _team_match(a: str, b: str) -> bool:
    """Loose team name match — normalise whitespace and case."""
    return a.strip().lower() == b.strip().lower()
