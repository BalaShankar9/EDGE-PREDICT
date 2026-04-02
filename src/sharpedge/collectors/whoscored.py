"""
WhoScoredCollector — player ratings, formations, and tactical stats.

WhoScored is one of the richest free football data sources, embedding
match data as JSON inside JavaScript on the page.

Modes
-----
match_ratings   Per-player per-match: rating, minutes, goals, assists, etc.
team_stats      Team-level tactical averages (possession, shots, passes, …).
formations      Formation used by each team in every match.
"""

import json
import logging
import re
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  League catalogue                                                    #
# ------------------------------------------------------------------ #

WHOSCORED_LEAGUES: dict[str, dict[str, Any]] = {
    "Premier League": {"path": "Tournaments/2/England-Premier-League", "id": 2},
    "La Liga": {"path": "Tournaments/4/Spain-LaLiga", "id": 4},
    "Bundesliga": {"path": "Tournaments/3/Germany-Bundesliga", "id": 3},
    "Serie A": {"path": "Tournaments/5/Italy-Serie-A", "id": 5},
    "Ligue 1": {"path": "Tournaments/22/France-Ligue-1", "id": 22},
    "Championship": {"path": "Tournaments/7/England-Championship", "id": 7},
    "Eredivisie": {"path": "Tournaments/13/Netherlands-Eredivisie", "id": 13},
}

# ------------------------------------------------------------------ #
#  Regex patterns for extracting embedded JSON from JavaScript         #
# ------------------------------------------------------------------ #

# Match-centre page: var matchCentreData = {...};
_MATCH_CENTRE_RE = re.compile(
    r"var\s+matchCentreData\s*=\s*(\{.+?\})\s*;",
    re.DOTALL,
)

# Fixture / tournament pages: var matchHeader = {...};
_MATCH_HEADER_RE = re.compile(
    r"var\s+matchHeader\s*=\s*(\{.+?\})\s*;",
    re.DOTALL,
)

# require.config args pattern used on some league pages
_REQUIRE_CONFIG_RE = re.compile(
    r'require\.config\.params\["args"\]\s*=\s*(\{.+?\})\s*;',
    re.DOTALL,
)

# General fallback: any JSON data object assigned to a known var
_GENERIC_JSON_RE = re.compile(
    r"(?:matchCentreData|matchHeader)\s*=\s*(\{.+?\})\s*;",
    re.DOTALL,
)

# ------------------------------------------------------------------ #
#  Extra headers WhoScored expects                                     #
# ------------------------------------------------------------------ #

_WS_HEADERS: dict[str, str] = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.5",
    "Referer": "https://www.whoscored.com/",
    "DNT": "1",
}


# ------------------------------------------------------------------ #
#  Collector                                                           #
# ------------------------------------------------------------------ #


class WhoScoredCollector(BaseCollector):
    """Collector for WhoScored player ratings, formations, and team stats.

    Usage
    -----
    col = WhoScoredCollector()

    # Player ratings from recent matches in the Premier League
    ratings_df = col.collect(mode="match_ratings", league="Premier League")

    # Team-level tactical averages
    team_df = col.collect(mode="team_stats", league="La Liga")

    # Formation usage across recent matches
    form_df = col.collect(mode="formations", league="Bundesliga")
    """

    source_name = "whoscored"
    base_url = "https://www.whoscored.com"
    request_delay = 5.0  # WhoScored rate-limits aggressively

    # ------------------------------------------------------------------ #
    #  Main dispatch                                                       #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Dispatch to the appropriate collection mode.

        Parameters
        ----------
        mode : str
            One of ``"match_ratings"``, ``"team_stats"``, ``"formations"``.
            Defaults to ``"match_ratings"``.
        league : str, optional
            League name from WHOSCORED_LEAGUES.  Defaults to "Premier League".
        max_matches : int, optional
            Max number of recent match pages to scrape (match_ratings /
            formations modes).  Default 10.
        """
        mode: str = kwargs.get("mode", "match_ratings")

        if mode == "match_ratings":
            return self._collect_match_ratings(**kwargs)
        if mode == "team_stats":
            return self._collect_team_stats(**kwargs)
        if mode == "formations":
            return self._collect_formations(**kwargs)

        raise ValueError(
            f"[{self.source_name}] Unknown mode '{mode}'. "
            "Use 'match_ratings', 'team_stats', or 'formations'."
        )

    # ------------------------------------------------------------------ #
    #  Mode: match_ratings                                                 #
    # ------------------------------------------------------------------ #

    def _collect_match_ratings(self, **kwargs: Any) -> pd.DataFrame:
        """Scrape per-player per-match ratings from recent match pages.

        Returns columns:
          match_id, league, home_team, away_team, match_date,
          player_id, player_name, team, position, shirt_number,
          rating, minutes_played, goals, assists, shots, key_passes,
          dribbles, tackles, interceptions, fouls, was_motm
        """
        league: str = kwargs.get("league", "Premier League")
        max_matches: int = int(kwargs.get("max_matches", 10))

        league_cfg = _resolve_league(league)
        if league_cfg is None:
            logger.warning(f"[{self.source_name}] Unknown league '{league}'")
            return pd.DataFrame()

        match_links = self._fetch_fixture_links(league_cfg, max_matches)
        if not match_links:
            logger.warning(
                f"[{self.source_name}] No match links found for '{league}'"
            )
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for match_id, match_url in match_links:
            cache_key = f"ws_match_ratings_{match_id}"
            cached = self._get_cached(cache_key)

            if cached is not None:
                frames.append(pd.DataFrame(cached))
                logger.debug(f"[{self.source_name}] Cache hit: match {match_id}")
                continue

            data = self._fetch_match_centre(match_url)
            if data is None:
                continue

            rows = self._parse_player_ratings(data, match_id, league)
            if rows:
                df = pd.DataFrame(rows)
                self._set_cache(cache_key, df.to_dict(orient="list"))
                frames.append(df)
                logger.info(
                    f"[{self.source_name}] match {match_id}: "
                    f"{len(rows)} player rows"
                )

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["home_team_norm"] = result["home_team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        result["away_team_norm"] = result["away_team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        result["team_norm"] = result["team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        return result

    def _fetch_fixture_links(
        self,
        league_cfg: dict[str, Any],
        max_matches: int,
    ) -> list[tuple[str, str]]:
        """Fetch the Fixtures page and return (match_id, match_url) pairs."""
        url = f"{self.base_url}/{league_cfg['path']}/Fixtures"
        cache_key = f"ws_fixture_links_{league_cfg['id']}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return [tuple(item) for item in cached[:max_matches]]  # type: ignore[misc]

        try:
            response = self._fetch(url, headers=_WS_HEADERS)
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] Fixtures page fetch failed: {exc}"
            )
            return []

        html = response.text
        self._check_structure(html, ["Match", "whoscored"])

        soup = BeautifulSoup(html, "html.parser")
        links: list[tuple[str, str]] = []

        # WhoScored match links look like /Matches/1234567/Live/...
        for tag in soup.find_all("a", href=re.compile(r"/Matches/\d+/")):
            href: str = tag.get("href", "")
            match = re.search(r"/Matches/(\d+)/", href)
            if not match:
                continue
            match_id = match.group(1)
            full_url = (
                href
                if href.startswith("http")
                else f"{self.base_url}{href}"
            )
            # Deduplicate — keep first occurrence
            ids_seen = {mid for mid, _ in links}
            if match_id not in ids_seen:
                links.append((match_id, full_url))

        # Cache for a short period (fixture list changes)
        if links:
            self._set_cache(cache_key, links)

        return links[:max_matches]

    def _fetch_match_centre(self, url: str) -> Optional[dict]:
        """Fetch a match page and extract the matchCentreData JSON blob."""
        try:
            response = self._fetch(url, headers=_WS_HEADERS)
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] Match page fetch failed ({url}): {exc}"
            )
            return None

        html = response.text
        self._check_structure(html, ["matchCentreData"])

        return _extract_json_blob(html, [_MATCH_CENTRE_RE, _GENERIC_JSON_RE])

    def _parse_player_ratings(
        self,
        data: dict,
        match_id: str,
        league: str,
    ) -> list[dict]:
        """Extract per-player stats from matchCentreData JSON."""
        rows: list[dict] = []

        home_team = (data.get("home") or {}).get("name", "")
        away_team = (data.get("away") or {}).get("name", "")
        match_date = data.get("startTime") or data.get("startDate") or ""

        for side_key, side_team in [("home", home_team), ("away", away_team)]:
            side_data = data.get(side_key, {})
            players = side_data.get("players", [])

            for player in players:
                try:
                    stats = player.get("stats", {})
                    rows.append({
                        "match_id": match_id,
                        "league": league,
                        "home_team": home_team,
                        "away_team": away_team,
                        "match_date": match_date,
                        "player_id": player.get("playerId") or player.get("id"),
                        "player_name": player.get("name"),
                        "team": side_team,
                        "position": player.get("position"),
                        "shirt_number": player.get("shirtNo"),
                        "rating": _safe_float(
                            player.get("ratings", {}).get("total")
                            or stats.get("ratings", {}).get("total")
                            or player.get("rating")
                        ),
                        "minutes_played": _safe_int(
                            stats.get("minsPlayed") or player.get("minsPlayed")
                        ),
                        "goals": _safe_int(
                            stats.get("goals") or player.get("goals")
                        ),
                        "assists": _safe_int(
                            stats.get("assistTotal") or player.get("assistTotal")
                        ),
                        "shots": _safe_int(
                            stats.get("shotsTotal") or player.get("shotsTotal")
                        ),
                        "key_passes": _safe_int(
                            stats.get("keyPassTotal") or player.get("keyPassTotal")
                        ),
                        "dribbles": _safe_int(
                            stats.get("dribbleWon") or player.get("dribbleWon")
                        ),
                        "tackles": _safe_int(
                            stats.get("tackleTotal") or player.get("tackleTotal")
                        ),
                        "interceptions": _safe_int(
                            stats.get("interceptionTotal")
                            or player.get("interceptionTotal")
                        ),
                        "fouls": _safe_int(
                            stats.get("foulsCommitted") or player.get("foulsCommitted")
                        ),
                        "was_motm": bool(
                            player.get("isManOfTheMatch")
                            or player.get("motm")
                            or False
                        ),
                    })
                except Exception as exc:
                    logger.debug(
                        f"[{self.source_name}] player parse error "
                        f"match {match_id}: {exc}"
                    )

        return rows

    # ------------------------------------------------------------------ #
    #  Mode: team_stats                                                    #
    # ------------------------------------------------------------------ #

    def _collect_team_stats(self, **kwargs: Any) -> pd.DataFrame:
        """Scrape team-level tactical averages from the TeamStatistics page.

        Returns columns:
          league, team, possession_avg, shots_pg, pass_accuracy,
          tackles_pg, interceptions_pg, fouls_pg, offsides_pg, rating_avg
        """
        league: str = kwargs.get("league", "Premier League")
        league_cfg = _resolve_league(league)
        if league_cfg is None:
            logger.warning(f"[{self.source_name}] Unknown league '{league}'")
            return pd.DataFrame()

        url = f"{self.base_url}/{league_cfg['path']}/TeamStatistics"
        cache_key = f"ws_team_stats_{league_cfg['id']}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[{self.source_name}] Cache hit: team_stats {league}")
            return pd.DataFrame(cached)

        try:
            response = self._fetch(url, headers=_WS_HEADERS)
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] TeamStatistics fetch failed: {exc}"
            )
            return pd.DataFrame()

        html = response.text
        rows = self._parse_team_stats_html(html, league)

        if not rows:
            logger.warning(
                f"[{self.source_name}] No team stats parsed for '{league}'"
            )
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["team_norm"] = df["team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        self._set_cache(cache_key, df.to_dict(orient="list"))
        return df

    def _parse_team_stats_html(self, html: str, league: str) -> list[dict]:
        """Parse team statistics table from WhoScored TeamStatistics page."""
        rows: list[dict] = []
        soup = BeautifulSoup(html, "html.parser")

        # WhoScored renders stats in a table with id "top-team-stats-summary-grid"
        # or similar.  Try multiple selectors for resilience.
        table = (
            soup.find("table", id=re.compile(r"top-team-stats", re.I))
            or soup.find("table", class_=re.compile(r"standings", re.I))
            or soup.find("table", id=re.compile(r"team-stats", re.I))
        )

        if table is None:
            # Fallback: try to parse embedded JSON (some pages use JS rendering)
            json_data = _extract_json_blob(
                html, [_REQUIRE_CONFIG_RE, _MATCH_HEADER_RE]
            )
            if json_data:
                return self._parse_team_stats_json(json_data, league)
            logger.warning(
                f"[{self.source_name}] Could not locate team stats table"
            )
            return rows

        # Map header text → column index
        header_row = table.find("thead")
        headers: list[str] = []
        if header_row:
            headers = [
                th.get_text(strip=True).lower()
                for th in header_row.find_all("th")
            ]

        col_map = _build_team_stats_col_map(headers)

        tbody = table.find("tbody")
        if not tbody:
            return rows

        for tr in tbody.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue
            try:
                rows.append({
                    "league": league,
                    "team": _cell_text(cells, col_map.get("team", 0)),
                    "possession_avg": _cell_float(cells, col_map.get("possession", -1)),
                    "shots_pg": _cell_float(cells, col_map.get("shots", -1)),
                    "pass_accuracy": _cell_float(cells, col_map.get("pass_accuracy", -1)),
                    "tackles_pg": _cell_float(cells, col_map.get("tackles", -1)),
                    "interceptions_pg": _cell_float(cells, col_map.get("interceptions", -1)),
                    "fouls_pg": _cell_float(cells, col_map.get("fouls", -1)),
                    "offsides_pg": _cell_float(cells, col_map.get("offsides", -1)),
                    "rating_avg": _cell_float(cells, col_map.get("rating", -1)),
                })
            except Exception as exc:
                logger.debug(
                    f"[{self.source_name}] team stats row parse error: {exc}"
                )

        return rows

    def _parse_team_stats_json(
        self, data: dict, league: str
    ) -> list[dict]:
        """Extract team stats when page uses JSON embedding."""
        rows: list[dict] = []
        teams = data.get("teams") or data.get("teamStats") or []
        for team_data in teams:
            try:
                rows.append({
                    "league": league,
                    "team": team_data.get("name") or team_data.get("teamName", ""),
                    "possession_avg": _safe_float(team_data.get("possessionPercentage")),
                    "shots_pg": _safe_float(team_data.get("shotsPerGame")),
                    "pass_accuracy": _safe_float(team_data.get("passSuccess")),
                    "tackles_pg": _safe_float(team_data.get("tacklesPerGame")),
                    "interceptions_pg": _safe_float(team_data.get("interceptionPerGame")),
                    "fouls_pg": _safe_float(team_data.get("foulsPerGame")),
                    "offsides_pg": _safe_float(team_data.get("offsidesPerGame")),
                    "rating_avg": _safe_float(team_data.get("rating")),
                })
            except Exception as exc:
                logger.debug(
                    f"[{self.source_name}] team stats JSON parse error: {exc}"
                )
        return rows

    # ------------------------------------------------------------------ #
    #  Mode: formations                                                    #
    # ------------------------------------------------------------------ #

    def _collect_formations(self, **kwargs: Any) -> pd.DataFrame:
        """Extract formation usage from recent match pages.

        Returns columns:
          match_id, league, home_team, away_team, match_date,
          home_formation, away_formation
        """
        league: str = kwargs.get("league", "Premier League")
        max_matches: int = int(kwargs.get("max_matches", 10))

        league_cfg = _resolve_league(league)
        if league_cfg is None:
            logger.warning(f"[{self.source_name}] Unknown league '{league}'")
            return pd.DataFrame()

        match_links = self._fetch_fixture_links(league_cfg, max_matches)
        if not match_links:
            return pd.DataFrame()

        rows: list[dict] = []
        for match_id, match_url in match_links:
            cache_key = f"ws_formations_{match_id}"
            cached = self._get_cached(cache_key)

            if cached is not None:
                rows.append(cached)
                continue

            data = self._fetch_match_centre(match_url)
            if data is None:
                continue

            row = self._parse_formation(data, match_id, league)
            if row:
                self._set_cache(cache_key, row)
                rows.append(row)
                logger.info(
                    f"[{self.source_name}] match {match_id}: "
                    f"formations {row.get('home_formation')} / "
                    f"{row.get('away_formation')}"
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["home_team_norm"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        df["away_team_norm"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        return df

    def _parse_formation(
        self,
        data: dict,
        match_id: str,
        league: str,
    ) -> Optional[dict]:
        """Extract formation strings from matchCentreData."""
        try:
            home_data = data.get("home") or {}
            away_data = data.get("away") or {}

            home_team = home_data.get("name", "")
            away_team = away_data.get("name", "")
            match_date = data.get("startTime") or data.get("startDate") or ""

            home_formation = _normalise_formation(
                home_data.get("formation")
                or home_data.get("formationName")
                or home_data.get("formationId")
            )
            away_formation = _normalise_formation(
                away_data.get("formation")
                or away_data.get("formationName")
                or away_data.get("formationId")
            )

            return {
                "match_id": match_id,
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "home_formation": home_formation,
                "away_formation": away_formation,
            }
        except Exception as exc:
            logger.debug(
                f"[{self.source_name}] formation parse error match {match_id}: {exc}"
            )
            return None


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _resolve_league(name: str) -> Optional[dict[str, Any]]:
    """Return league config dict or None for unknown league names."""
    return WHOSCORED_LEAGUES.get(name)


def _extract_json_blob(
    html: str,
    patterns: list[re.Pattern],  # type: ignore[type-arg]
) -> Optional[dict]:
    """Try each regex pattern and return the first valid JSON dict found."""
    for pattern in patterns:
        match = pattern.search(html)
        if not match:
            continue
        raw = match.group(1)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Sometimes the JSON is truncated by DOTALL greediness — try
            # a balanced-brace extraction as fallback.
            balanced = _extract_balanced_json(html, match.start(1))
            if balanced:
                return balanced
    return None


def _extract_balanced_json(html: str, start: int) -> Optional[dict]:
    """Walk forward from `start` counting braces to find a complete JSON object."""
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(html[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(html[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalise_formation(raw: Any) -> Optional[str]:
    """Normalise various formation representations to '4-3-3' style."""
    if raw is None:
        return None
    s = str(raw).strip()

    # Already formatted (e.g. "4-3-3")
    if re.match(r"^\d[-\d]+\d$", s):
        return s

    # Some sources use "4231" style — insert dashes heuristically
    if re.match(r"^\d{3,5}$", s):
        if len(s) == 3:
            return "-".join(list(s))
        if len(s) == 4:
            # Common 4-digit formations: 4141, 4231, 4321, 3421, 3511
            return f"{s[0]}-{s[1]}-{s[2]}-{s[3]}"
    return s or None


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    try:
        return int(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _cell_text(cells: list, idx: int) -> str:
    if idx < 0 or idx >= len(cells):
        return ""
    return cells[idx].get_text(strip=True)


def _cell_float(cells: list, idx: int) -> Optional[float]:
    return _safe_float(_cell_text(cells, idx).replace("%", "") or None)


def _build_team_stats_col_map(headers: list[str]) -> dict[str, int]:
    """Map semantic column names to indices based on header text."""
    mapping: dict[str, int] = {}
    for i, h in enumerate(headers):
        hl = h.lower()
        if "team" in hl and "team" not in mapping:
            mapping["team"] = i
        elif "possession" in hl or "poss" in hl:
            mapping["possession"] = i
        elif "shots" in hl:
            mapping["shots"] = i
        elif "pass" in hl and "acc" in hl:
            mapping["pass_accuracy"] = i
        elif "tackle" in hl:
            mapping["tackles"] = i
        elif "intercept" in hl:
            mapping["interceptions"] = i
        elif "foul" in hl:
            mapping["fouls"] = i
        elif "offside" in hl:
            mapping["offsides"] = i
        elif "rating" in hl:
            mapping["rating"] = i
    return mapping
