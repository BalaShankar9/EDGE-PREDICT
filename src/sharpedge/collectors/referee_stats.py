"""
RefereeStats Collector — per-referee disciplinary and match statistics.

Some referees issue 3× more penalties and cards than others. This is a
measurable edge that most prediction models completely ignore.

Primary source: Transfermarkt referee section
  URL pattern: /[league]/schiedsrichter/wettbewerb/[league_id]  (list)
               /[name]/profil/schiedsrichter/[id]               (per-referee)

Fallback / cross-reference: football-lineups.com referee stats page
  URL pattern: https://www.football-lineups.com/tourn/[league]/Stats/Referees/
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# Transfermarkt league IDs and URL slugs
LEAGUE_IDS: dict[str, str] = {
    "Premier League": "GB1",
    "La Liga": "ES1",
    "Bundesliga": "L1",
    "Serie A": "IT1",
    "Ligue 1": "FR1",
}

LEAGUE_SLUGS: dict[str, str] = {
    "Premier League": "premier-league",
    "La Liga": "laliga",
    "Bundesliga": "1-bundesliga",
    "Serie A": "serie-a",
    "Ligue 1": "ligue-1",
}

# football-lineups.com uses different league name strings
FOOTBALL_LINEUPS_LEAGUES: dict[str, str] = {
    "Premier League": "Premier_League",
    "La Liga": "La_Liga",
    "Bundesliga": "1._Bundesliga",
    "Serie A": "Serie_A",
    "Ligue 1": "Ligue_1",
}

_TM_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www.transfermarkt.com/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_FL_BASE = "https://www.football-lineups.com"


class RefereeStatsCollector(BaseCollector):
    """Collector for referee statistics from Transfermarkt (+ fallback)."""

    source_name = "referee_stats"
    base_url = "https://www.transfermarkt.com"
    request_delay = 4.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect referee statistics.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all leagues.
        per_referee : bool, optional
            If True (default), also fetches individual referee profile pages
            for detailed per-season stats. Set to False for a fast list-only
            run using aggregated season totals from the list page.
        use_fallback : bool, optional
            If True (default), cross-reference with football-lineups.com
            when Transfermarkt data is sparse.
        """
        league: Optional[str] = kwargs.get("league")
        per_referee: bool = kwargs.get("per_referee", True)
        use_fallback: bool = kwargs.get("use_fallback", True)

        leagues = (
            {league: LEAGUE_IDS[league]}
            if league
            else LEAGUE_IDS
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_id in leagues.items():
            try:
                df = self._collect_league(
                    league_name, league_id, per_referee=per_referee
                )
                if df is not None and not df.empty:
                    frames.append(df)
                elif use_fallback:
                    logger.info(
                        f"[referee_stats] TM empty for {league_name}, trying fallback"
                    )
                    fb = self._collect_fallback(league_name)
                    if fb is not None and not fb.empty:
                        frames.append(fb)
            except Exception as e:
                logger.error(
                    f"[referee_stats] Failed to collect {league_name}: {e}"
                )
                if use_fallback:
                    try:
                        fb = self._collect_fallback(league_name)
                        if fb is not None and not fb.empty:
                            frames.append(fb)
                    except Exception as fb_err:
                        logger.error(
                            f"[referee_stats] Fallback also failed for "
                            f"{league_name}: {fb_err}"
                        )
                continue

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Transfermarkt — referee list                                        #
    # ------------------------------------------------------------------ #

    def _collect_league(
        self,
        league_name: str,
        league_id: str,
        per_referee: bool,
    ) -> Optional[pd.DataFrame]:
        """Fetch the referee list page for one league, then optionally
        drill into each referee's profile page."""
        slug = LEAGUE_SLUGS.get(league_name, league_name.lower().replace(" ", "-"))
        url = f"{self.base_url}/{slug}/schiedsrichter/wettbewerb/{league_id}"

        cache_key = f"referee_list_{league_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[referee_stats] Cache hit for list {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url, headers=_TM_HEADERS)
        html = response.text

        self._check_structure(html, ["schiedsrichter"])

        soup = BeautifulSoup(html, "html.parser")
        referee_links = self._parse_referee_list(soup, league_name)

        if not referee_links:
            logger.warning(f"[referee_stats] No referees parsed for {league_name}")
            return None

        rows: list[dict] = []
        if per_referee:
            for ref_meta in referee_links:
                try:
                    profile = self._fetch_referee_profile(ref_meta, league_name)
                    if profile:
                        rows.append(profile)
                except Exception as e:
                    logger.debug(
                        f"[referee_stats] Profile fetch failed for "
                        f"{ref_meta.get('name', '?')}: {e}"
                    )
                    # Fall back to list-level data
                    rows.append(ref_meta)
        else:
            rows = referee_links

        if rows:
            self._set_cache(cache_key, rows)

        logger.info(
            f"[referee_stats] Fetched referees {league_name}: {len(rows)} referees"
        )
        return pd.DataFrame(rows) if rows else None

    def _parse_referee_list(
        self, soup: BeautifulSoup, league_name: str
    ) -> list[dict]:
        """Parse the TM referee list table — extracts names, IDs, season stats."""
        table = soup.select_one("table.items")
        if not table:
            logger.warning(f"[referee_stats] No items table for {league_name}")
            return []

        refs: list[dict] = []
        for tr in table.select("tbody tr"):
            ref = self._parse_referee_list_row(tr, league_name)
            if ref:
                refs.append(ref)
        return refs

    def _parse_referee_list_row(self, tr: Any, league_name: str) -> Optional[dict]:
        """Parse one row from the TM referee list table."""
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            # Referee name and profile URL from .hauptlink anchor
            name_el = tr.select_one(".hauptlink a, a[href*='schiedsrichter']")
            if not name_el:
                return None

            referee_name = name_el.get_text(strip=True)
            profile_href = name_el.get("href", "")

            # Extract TM referee ID from href like /john-smith/profil/schiedsrichter/12345
            tm_id_match = re.search(r"/schiedsrichter/(\d+)", profile_href)
            tm_id = tm_id_match.group(1) if tm_id_match else ""
            tm_slug = profile_href.split("/")[1] if "/" in profile_href else ""

            # Nationality / flag
            nationality = ""
            flag_img = tr.select_one("img.flaggenrahmen, img[class*='flag']")
            if flag_img:
                nationality = flag_img.get("title", flag_img.get("alt", ""))

            # Numeric stats from remaining cells — TM shows: matches, yellows,
            # yellow-reds, reds, penalty cols depending on season
            numeric_vals: list[str] = []
            for td in tds:
                cls = " ".join(td.get("class", []))
                if "zentriert" in cls:
                    numeric_vals.append(td.get_text(strip=True))

            matches_this_season = _safe_int(numeric_vals[0]) if len(numeric_vals) > 0 else None
            yellows_season = _safe_int(numeric_vals[1]) if len(numeric_vals) > 1 else None
            reds_season = _safe_int(numeric_vals[2]) if len(numeric_vals) > 2 else None

            return {
                "referee_name": referee_name,
                "tm_id": tm_id,
                "tm_slug": tm_slug,
                "nationality": nationality,
                "league": league_name,
                "matches_this_season": matches_this_season,
                "yellows_this_season": yellows_season,
                "reds_this_season": reds_season,
                # Profile stats filled later by _fetch_referee_profile
                "matches_officiated_career": None,
                "yellow_cards_per_game": None,
                "red_cards_per_game": None,
                "penalties_per_game": None,
                "home_win_pct": None,
                "draw_pct": None,
                "away_win_pct": None,
                "avg_fouls_per_game": None,
                "avg_goals_per_game": None,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[referee_stats] Failed to parse list row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Transfermarkt — per-referee profile                                 #
    # ------------------------------------------------------------------ #

    def _fetch_referee_profile(
        self, ref_meta: dict, league_name: str
    ) -> Optional[dict]:
        """Fetch and parse a referee's profile page for career stats."""
        slug = ref_meta.get("tm_slug", "")
        tm_id = ref_meta.get("tm_id", "")
        if not slug or not tm_id:
            return ref_meta  # Return list-level data as-is

        url = f"{self.base_url}/{slug}/profil/schiedsrichter/{tm_id}"
        profile_cache = f"referee_profile_{tm_id}"
        cached = self._get_cached(profile_cache)
        if cached is not None:
            return cached

        try:
            response = self._fetch(url, headers=_TM_HEADERS)
            html = response.text
        except Exception as e:
            logger.debug(f"[referee_stats] Could not fetch profile {url}: {e}")
            return ref_meta

        soup = BeautifulSoup(html, "html.parser")
        enriched = dict(ref_meta)

        # Career match count — shown in the info box
        career_matches = self._extract_stat(soup, ["Appearances", "Einsätze", "Matches"])
        enriched["matches_officiated_career"] = _safe_int(career_matches)

        # Stats table — TM referee profiles have a season-by-season table
        # with columns: Season | Competition | Matches | Yellow | Yellow-Red | Red
        stats_table = soup.select_one("table.items")
        if stats_table:
            season_rows = self._parse_profile_stats_table(stats_table)
            # Aggregate across all rows for per-game rates
            total_matches = sum(r.get("matches", 0) or 0 for r in season_rows)
            total_yellows = sum(r.get("yellows", 0) or 0 for r in season_rows)
            total_reds = sum(r.get("reds", 0) or 0 for r in season_rows)

            if total_matches > 0:
                enriched["yellow_cards_per_game"] = round(total_yellows / total_matches, 3)
                enriched["red_cards_per_game"] = round(total_reds / total_matches, 3)
                if enriched.get("matches_officiated_career") is None:
                    enriched["matches_officiated_career"] = total_matches

        self._set_cache(profile_cache, enriched)
        return enriched

    def _parse_profile_stats_table(self, table: Any) -> list[dict]:
        """Parse season rows from a TM referee profile stats table."""
        rows = []
        for tr in table.select("tbody tr"):
            tds = tr.select("td")
            if len(tds) < 4:
                continue
            centered = [
                td.get_text(strip=True)
                for td in tds
                if "zentriert" in " ".join(td.get("class", []))
            ]
            if not centered:
                continue
            rows.append({
                "matches": _safe_int(centered[0]) if len(centered) > 0 else None,
                "yellows": _safe_int(centered[1]) if len(centered) > 1 else None,
                "yellow_reds": _safe_int(centered[2]) if len(centered) > 2 else None,
                "reds": _safe_int(centered[3]) if len(centered) > 3 else None,
            })
        return rows

    def _extract_stat(self, soup: BeautifulSoup, labels: list[str]) -> str:
        """Extract a single stat value from the TM info box by label text."""
        for label in labels:
            el = soup.find(string=re.compile(re.escape(label), re.IGNORECASE))
            if el:
                parent = el.find_parent()
                if parent:
                    sibling = parent.find_next_sibling()
                    if sibling:
                        return sibling.get_text(strip=True)
        return ""

    # ------------------------------------------------------------------ #
    #  Fallback: football-lineups.com                                      #
    # ------------------------------------------------------------------ #

    def _collect_fallback(self, league_name: str) -> Optional[pd.DataFrame]:
        """Scrape referee stats from football-lineups.com as fallback."""
        fl_league = FOOTBALL_LINEUPS_LEAGUES.get(league_name)
        if not fl_league:
            return None

        url = f"{_FL_BASE}/tourn/{fl_league}/Stats/Referees/"
        cache_key = f"referee_fallback_{fl_league}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[referee_stats] Fallback cache hit for {league_name}")
            return pd.DataFrame(cached)

        try:
            response = self._fetch(url)
            html = response.text
        except Exception as e:
            logger.warning(f"[referee_stats] Fallback fetch failed for {league_name}: {e}")
            return None

        soup = BeautifulSoup(html, "html.parser")
        rows = self._parse_fallback_table(soup, league_name)

        if rows:
            self._set_cache(cache_key, rows)

        logger.info(
            f"[referee_stats] Fallback fetched {league_name}: {len(rows)} referees"
        )
        return pd.DataFrame(rows) if rows else None

    def _parse_fallback_table(
        self, soup: BeautifulSoup, league_name: str
    ) -> list[dict]:
        """Parse the football-lineups.com referee stats table."""
        # football-lineups uses a standard sortable table
        table = soup.select_one("table.leag, table.stats, table")
        if not table:
            return []

        # Read column headers
        headers: list[str] = []
        for th in table.select("thead th, tr:first-child th"):
            headers.append(th.get_text(strip=True).lower())

        def col(candidates: list[str]) -> Optional[int]:
            for cand in candidates:
                for i, h in enumerate(headers):
                    if cand in h:
                        return i
            return None

        idx_name = col(["referee", "name"])
        idx_matches = col(["games", "matches", "played"])
        idx_yellow = col(["yellow"])
        idx_red = col(["red"])
        idx_penalty = col(["penalty", "pen"])
        idx_fouls = col(["foul"])
        idx_goals = col(["goal"])

        rows: list[dict] = []
        tbody = table.select_one("tbody") or table
        for tr in tbody.select("tr"):
            tds = tr.select("td")
            if len(tds) < 2:
                continue

            def td_text(idx: Optional[int]) -> str:
                if idx is None or idx >= len(tds):
                    return ""
                return tds[idx].get_text(strip=True)

            referee_name = td_text(idx_name)
            if not referee_name:
                continue

            matches = _safe_int(td_text(idx_matches))
            yellows = _safe_int(td_text(idx_yellow))
            reds = _safe_int(td_text(idx_red))
            penalties = _safe_float(td_text(idx_penalty))
            fouls = _safe_float(td_text(idx_fouls))
            goals = _safe_float(td_text(idx_goals))

            ypc = round(yellows / matches, 3) if matches and yellows is not None else None
            rpc = round(reds / matches, 3) if matches and reds is not None else None

            rows.append({
                "referee_name": referee_name,
                "league": league_name,
                "matches_this_season": matches,
                "matches_officiated_career": None,
                "yellow_cards_per_game": ypc,
                "red_cards_per_game": rpc,
                "penalties_per_game": penalties,
                "home_win_pct": None,
                "draw_pct": None,
                "away_win_pct": None,
                "avg_fouls_per_game": fouls,
                "avg_goals_per_game": goals,
                "data_source": "football-lineups",
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            })

        return rows


# ------------------------------------------------------------------ #
#  Parsing helpers                                                     #
# ------------------------------------------------------------------ #

def _safe_int(val: Any) -> Optional[int]:
    """Safely convert a value to int, stripping non-numeric chars."""
    if val is None:
        return None
    try:
        cleaned = re.sub(r"[^\d-]", "", str(val))
        return int(cleaned) if cleaned and cleaned != "-" else None
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any) -> Optional[float]:
    """Safely convert a value to float."""
    if val is None:
        return None
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(val))
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None
