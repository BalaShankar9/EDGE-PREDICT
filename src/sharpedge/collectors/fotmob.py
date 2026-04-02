"""
FotMobCollector — advanced player metrics via FotMob's public JSON API.

FotMob exposes a clean, public JSON API — no HTML scraping required.
Data includes xG, player ratings, formations, squad injuries, and standings.

Modes
-----
fixtures          Matches for a date range (start_date → end_date).
match_details     Full per-player and per-team stats for a list of match IDs.
team_form         Recent form, injury list, and next fixture for every team
                  in a given league.
league_standings  Current standings enriched with xG / xGA / xGD.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  League catalogue                                                    #
# ------------------------------------------------------------------ #

FOTMOB_LEAGUES: dict[str, int] = {
    "Premier League": 47,
    "La Liga": 87,
    "Bundesliga": 54,
    "Serie A": 55,
    "Ligue 1": 53,
    "Championship": 48,
    "Eredivisie": 57,
    "Primeira Liga": 61,
    "Super Lig": 71,
    "Belgian Pro League": 56,
}

# ------------------------------------------------------------------ #
#  API configuration                                                   #
# ------------------------------------------------------------------ #

_API_BASE = "https://www.fotmob.com/api"

# FotMob sometimes requires this header (base64-encoded empty body marker)
_FM_HEADERS: dict[str, str] = {
    "x-fm-req": "eyJib2R5Ijp7fX0=",
    "Accept": "application/json",
    "Referer": "https://www.fotmob.com/",
    "Origin": "https://www.fotmob.com",
}


# ------------------------------------------------------------------ #
#  Collector                                                           #
# ------------------------------------------------------------------ #


class FotMobCollector(BaseCollector):
    """Collector for FotMob advanced match and player metrics.

    Usage
    -----
    col = FotMobCollector()

    # Fixtures for the next 3 days
    fixtures_df = col.collect(mode="fixtures", days_ahead=3)

    # Full details for specific matches
    details_df = col.collect(
        mode="match_details",
        match_ids=[1234567, 1234568],
    )

    # Recent form for every team in the Premier League
    form_df = col.collect(mode="team_form", league="Premier League")

    # League standings with xG data
    standings_df = col.collect(mode="league_standings", league="La Liga")
    """

    source_name = "fotmob"
    base_url = "https://www.fotmob.com"
    request_delay = 3.0

    # ------------------------------------------------------------------ #
    #  Main dispatch                                                       #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Dispatch to the appropriate collection mode.

        Parameters
        ----------
        mode : str
            One of ``"fixtures"``, ``"match_details"``, ``"team_form"``,
            ``"league_standings"``.  Defaults to ``"fixtures"``.
        start_date : str, optional
            Start date YYYY-MM-DD (fixtures mode).  Defaults to today.
        end_date : str, optional
            End date YYYY-MM-DD (fixtures mode).  Defaults to start_date.
        days_ahead : int, optional
            Convenience alternative to end_date.  Fetches this many extra
            days starting from start_date.  Default 0.
        league : str, optional
            League name from FOTMOB_LEAGUES (team_form / league_standings).
        match_ids : list[int], optional
            List of FotMob match IDs (match_details mode).
        """
        mode: str = kwargs.get("mode", "fixtures")

        if mode == "fixtures":
            return self._collect_fixtures(**kwargs)
        if mode == "match_details":
            return self._collect_match_details(**kwargs)
        if mode == "team_form":
            return self._collect_team_form(**kwargs)
        if mode == "league_standings":
            return self._collect_league_standings(**kwargs)

        raise ValueError(
            f"[{self.source_name}] Unknown mode '{mode}'. "
            "Use 'fixtures', 'match_details', 'team_form', or 'league_standings'."
        )

    # ------------------------------------------------------------------ #
    #  Mode: fixtures                                                      #
    # ------------------------------------------------------------------ #

    def _collect_fixtures(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch matches for a date range.

        Returns columns:
          match_id, home_team, away_team, league, league_id,
          start_time, status, home_score, away_score
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        start_date_str: str = kwargs.get("start_date", today)
        days_ahead: int = int(kwargs.get("days_ahead", 0))

        if "end_date" in kwargs:
            end_date_str: str = kwargs["end_date"]
        else:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_dt = start_dt + timedelta(days=days_ahead)
            end_date_str = end_dt.strftime("%Y-%m-%d")

        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

        frames: list[pd.DataFrame] = []
        current = start_dt
        while current <= end_dt:
            date_key = current.strftime("%Y%m%d")
            cache_key = f"fotmob_fixtures_{date_key}"
            cached = self._get_cached(cache_key)

            if cached is not None:
                frames.append(pd.DataFrame(cached))
                logger.debug(
                    f"[{self.source_name}] Cache hit: fixtures {date_key}"
                )
                current += timedelta(days=1)
                continue

            url = f"{_API_BASE}/matches?date={date_key}"
            try:
                data = self._fetch_json(url, headers=_FM_HEADERS)
            except Exception as exc:
                logger.error(
                    f"[{self.source_name}] fixtures fetch failed "
                    f"for {date_key}: {exc}"
                )
                current += timedelta(days=1)
                continue

            rows = self._parse_fixtures(data)
            if rows:
                df = pd.DataFrame(rows)
                self._set_cache(cache_key, df.to_dict(orient="list"))
                frames.append(df)
                logger.info(
                    f"[{self.source_name}] {len(rows)} fixtures for {date_key}"
                )

            current += timedelta(days=1)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["home_team_norm"] = result["home_team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        result["away_team_norm"] = result["away_team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        return result

    def _parse_fixtures(self, data: Any) -> list[dict]:
        """Parse the /matches response into fixture rows."""
        rows: list[dict] = []
        if not isinstance(data, dict):
            return rows

        # FotMob groups matches by league under "leagues" key
        leagues_data = data.get("leagues") or []
        for league_block in leagues_data:
            league_name = league_block.get("name") or league_block.get("ccode", "")
            league_id = league_block.get("id")
            matches = league_block.get("matches") or []

            for match in matches:
                try:
                    home = match.get("home") or {}
                    away = match.get("away") or {}
                    status = match.get("status") or {}

                    home_name = home.get("name") or home.get("longName", "")
                    away_name = away.get("name") or away.get("longName", "")
                    if not home_name or not away_name:
                        continue

                    start_ts = match.get("status", {}).get("utcTime") or match.get("utcTime")
                    start_time: Optional[str] = None
                    if start_ts:
                        try:
                            start_time = datetime.fromisoformat(
                                str(start_ts).replace("Z", "+00:00")
                            ).isoformat()
                        except ValueError:
                            start_time = str(start_ts)

                    score_str = status.get("scoreStr") or ""
                    home_score: Optional[int] = None
                    away_score: Optional[int] = None
                    if score_str and " - " in score_str:
                        parts = score_str.split(" - ")
                        home_score = _safe_int(parts[0])
                        away_score = _safe_int(parts[1])
                    else:
                        home_score = _safe_int(
                            match.get("homeScore")
                            or home.get("score")
                        )
                        away_score = _safe_int(
                            match.get("awayScore")
                            or away.get("score")
                        )

                    rows.append({
                        "match_id": match.get("id"),
                        "home_team": home_name,
                        "away_team": away_name,
                        "home_team_id": home.get("id"),
                        "away_team_id": away.get("id"),
                        "league": league_name,
                        "league_id": league_id,
                        "start_time": start_time,
                        "status": status.get("reason", {}).get("short")
                               or status.get("ongoing")
                               or ("FT" if status.get("finished") else "NS"),
                        "home_score": home_score,
                        "away_score": away_score,
                    })
                except Exception as exc:
                    logger.debug(
                        f"[{self.source_name}] fixture parse error: {exc}"
                    )

        return rows

    # ------------------------------------------------------------------ #
    #  Mode: match_details                                                 #
    # ------------------------------------------------------------------ #

    def _collect_match_details(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch full per-player and per-team stats for a list of matches.

        Returns two merged DataFrames stacked:
          - Team-level rows (player_id=None): xG, shots, possession, corners,
            fouls, formation
          - Player-level rows: rating, minutes, goals, assists, xG, xA,
            key_passes, touches, accurate_passes, tackles_won,
            interceptions, clearances, blocked_shots
        """
        match_ids: list[int] = list(kwargs.get("match_ids", []))
        if not match_ids:
            logger.warning(
                f"[{self.source_name}] match_details mode: "
                "no match_ids provided"
            )
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for match_id in match_ids:
            cache_key = f"fotmob_match_details_{match_id}"
            cached = self._get_cached(cache_key)

            if cached is not None:
                frames.append(pd.DataFrame(cached))
                logger.debug(
                    f"[{self.source_name}] Cache hit: match {match_id}"
                )
                continue

            url = f"{_API_BASE}/matchDetails?matchId={match_id}"
            try:
                data = self._fetch_json(url, headers=_FM_HEADERS)
            except Exception as exc:
                logger.error(
                    f"[{self.source_name}] match details fetch failed "
                    f"for {match_id}: {exc}"
                )
                continue

            rows = self._parse_match_details(data, match_id)
            if rows:
                df = pd.DataFrame(rows)
                self._set_cache(cache_key, df.to_dict(orient="list"))
                frames.append(df)
                logger.info(
                    f"[{self.source_name}] match {match_id}: "
                    f"{len(rows)} rows"
                )

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        for col in ("home_team", "away_team", "team"):
            if col in result.columns:
                result[f"{col}_norm"] = result[col].apply(
                    lambda x: self.normalise_team(str(x)) or x
                    if pd.notna(x) else x
                )
        return result

    def _parse_match_details(self, data: Any, match_id: int) -> list[dict]:
        """Parse matchDetails JSON into team-level and player-level rows."""
        rows: list[dict] = []
        if not isinstance(data, dict):
            return rows

        general = data.get("general") or {}
        home_team = (general.get("homeTeam") or {}).get("name", "")
        away_team = (general.get("awayTeam") or {}).get("name", "")
        match_date = general.get("matchTimeUTC") or general.get("matchTime") or ""

        # ---- Team-level stats ----
        content = data.get("content") or {}
        stats_block = content.get("stats") or {}
        team_stats = stats_block.get("Periods", {}).get("All", {}) or {}

        def _extract_team_stat(key: str, side: str) -> Optional[float]:
            entry = team_stats.get(key)
            if not entry:
                return None
            # FotMob stat entries are often dicts with "home"/"away" keys
            if isinstance(entry, dict):
                return _safe_float(entry.get(side))
            return None

        for side_key, side_name in [("home", home_team), ("away", away_team)]:
            rows.append({
                "match_id": match_id,
                "row_type": "team",
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "team": side_name,
                "side": side_key,
                "xg": _extract_team_stat("Expected goals (xG)", side_key),
                "shots": _extract_team_stat("Shots", side_key),
                "shots_on_target": _extract_team_stat("Shots on target", side_key),
                "possession": _extract_team_stat("Ball possession", side_key),
                "corners": _extract_team_stat("Corner kicks", side_key),
                "fouls": _extract_team_stat("Fouls", side_key),
                "formation": self._get_formation(data, side_key),
                # Player-level columns left null for team rows
                "player_id": None,
                "player_name": None,
                "rating": None,
                "minutes_played": None,
                "goals": None,
                "assists": None,
                "player_xg": None,
                "player_xa": None,
                "key_passes": None,
                "touches": None,
                "accurate_passes": None,
                "tackles_won": None,
                "interceptions": None,
                "clearances": None,
                "blocked_shots": None,
            })

        # ---- Player-level stats ----
        lineup = content.get("lineup") or {}
        for side_key, side_name in [("home", home_team), ("away", away_team)]:
            players_block = lineup.get(side_key) or {}
            players = players_block.get("players") or []
            # Also include substitutes
            bench = players_block.get("bench") or []
            all_players = players + bench

            for player_entry in all_players:
                # Some response shapes nest the player under a "player" key
                if isinstance(player_entry, list):
                    # Lineup can be nested list per row
                    for p in player_entry:
                        self._append_player_row(
                            rows, p, match_id, home_team, away_team,
                            match_date, side_name
                        )
                else:
                    self._append_player_row(
                        rows, player_entry, match_id, home_team, away_team,
                        match_date, side_name
                    )

        return rows

    def _append_player_row(
        self,
        rows: list[dict],
        player_entry: Any,
        match_id: int,
        home_team: str,
        away_team: str,
        match_date: str,
        team: str,
    ) -> None:
        """Parse one player dict and append a row."""
        if not isinstance(player_entry, dict):
            return
        try:
            # FotMob sometimes nests stats under "stats" or "usualPosition"
            stats = player_entry.get("stats") or {}
            # stats can be a list of {key, stat} dicts or a plain dict
            stat_dict = _flatten_stat_list(stats) if isinstance(stats, list) else stats

            rows.append({
                "match_id": match_id,
                "row_type": "player",
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "team": team,
                "side": None,
                "xg": None,
                "shots": None,
                "shots_on_target": None,
                "possession": None,
                "corners": None,
                "fouls": None,
                "formation": None,
                "player_id": player_entry.get("id") or player_entry.get("playerId"),
                "player_name": (
                    player_entry.get("name")
                    or player_entry.get("shortName")
                    or player_entry.get("playerName")
                ),
                "rating": _safe_float(
                    player_entry.get("ratingProps", {}).get("num")
                    or player_entry.get("rating")
                    or stat_dict.get("FotMob rating")
                ),
                "minutes_played": _safe_int(
                    player_entry.get("timeSubbedIn")
                    or stat_dict.get("Minutes played")
                    or stat_dict.get("minutesPlayed")
                ),
                "goals": _safe_int(
                    stat_dict.get("Goals") or stat_dict.get("goals")
                ),
                "assists": _safe_int(
                    stat_dict.get("Assists") or stat_dict.get("assists")
                ),
                "player_xg": _safe_float(
                    stat_dict.get("Expected goals (xG)")
                    or stat_dict.get("xG")
                ),
                "player_xa": _safe_float(
                    stat_dict.get("Expected assists (xA)")
                    or stat_dict.get("xA")
                ),
                "key_passes": _safe_int(
                    stat_dict.get("Key passes") or stat_dict.get("keyPasses")
                ),
                "touches": _safe_int(
                    stat_dict.get("Touches") or stat_dict.get("touches")
                ),
                "accurate_passes": _safe_int(
                    stat_dict.get("Accurate passes")
                    or stat_dict.get("accuratePasses")
                ),
                "tackles_won": _safe_int(
                    stat_dict.get("Tackles won") or stat_dict.get("tacklesWon")
                ),
                "interceptions": _safe_int(
                    stat_dict.get("Interceptions") or stat_dict.get("interceptions")
                ),
                "clearances": _safe_int(
                    stat_dict.get("Clearances") or stat_dict.get("clearances")
                ),
                "blocked_shots": _safe_int(
                    stat_dict.get("Blocked shots") or stat_dict.get("blockedShots")
                ),
            })
        except Exception as exc:
            logger.debug(
                f"[{self.source_name}] player row parse error "
                f"match {match_id}: {exc}"
            )

    def _get_formation(self, data: dict, side: str) -> Optional[str]:
        """Extract formation string for 'home' or 'away' from match details."""
        try:
            content = data.get("content") or {}
            lineup = content.get("lineup") or {}
            side_data = lineup.get(side) or {}
            formation = (
                side_data.get("formation")
                or side_data.get("formationName")
            )
            return str(formation) if formation else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Mode: team_form                                                     #
    # ------------------------------------------------------------------ #

    def _collect_team_form(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch recent form, injury list, and next fixture for league teams.

        Returns columns:
          team_id, team_name, league, league_id,
          last5_results, last5_goals_for, last5_goals_against,
          injuries_count, next_match_id, next_opponent, next_match_date
        """
        league: str = kwargs.get("league", "Premier League")
        league_id = FOTMOB_LEAGUES.get(league)
        if league_id is None:
            logger.warning(f"[{self.source_name}] Unknown league '{league}'")
            return pd.DataFrame()

        # Get team list from league overview
        teams = self._fetch_league_teams(league_id, league)
        if not teams:
            return pd.DataFrame()

        rows: list[dict] = []
        for team_id, team_name in teams:
            cache_key = f"fotmob_team_form_{team_id}"
            cached = self._get_cached(cache_key)
            if cached is not None:
                rows.append(cached)
                continue

            row = self._fetch_team_form_data(team_id, team_name, league, league_id)
            if row:
                self._set_cache(cache_key, row)
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["team_norm"] = df["team_name"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        return df

    def _fetch_league_teams(
        self, league_id: int, league_name: str
    ) -> list[tuple[int, str]]:
        """Return list of (team_id, team_name) for a league."""
        cache_key = f"fotmob_league_teams_{league_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return [tuple(item) for item in cached]  # type: ignore[misc]

        url = f"{_API_BASE}/leagues?id={league_id}"
        try:
            data = self._fetch_json(url, headers=_FM_HEADERS)
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] league teams fetch failed "
                f"for {league_name}: {exc}"
            )
            return []

        teams: list[tuple[int, str]] = []
        if isinstance(data, dict):
            # teams listed under "table" → rows or "teams"
            for table in data.get("table", [{}]):
                for row in table.get("data", {}).get("table", {}).get("all", []):
                    tid = row.get("id")
                    tname = row.get("name") or row.get("shortName", "")
                    if tid and tname:
                        teams.append((int(tid), str(tname)))

            # Fallback: check "teamsData" key
            if not teams:
                for tid, tdata in (data.get("teamsData") or {}).items():
                    tname = (tdata.get("title") or tdata.get("name") or "")
                    if tname:
                        teams.append((int(tid), str(tname)))

        if teams:
            self._set_cache(cache_key, teams)
        return teams

    def _fetch_team_form_data(
        self,
        team_id: int,
        team_name: str,
        league: str,
        league_id: int,
    ) -> Optional[dict]:
        """Fetch form data for a single team."""
        url = f"{_API_BASE}/teams?id={team_id}&ccode3=GBR"
        try:
            data = self._fetch_json(url, headers=_FM_HEADERS)
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] team form fetch failed "
                f"for {team_name} ({team_id}): {exc}"
            )
            return None

        if not isinstance(data, dict):
            return None

        history = data.get("history") or {}
        recent_matches: list[Any] = (
            history.get("allMatches") or history.get("matches") or []
        )[-5:]  # last 5

        results_str = ""
        gf = ga = 0
        for match in recent_matches:
            try:
                home_id = (match.get("home") or {}).get("id")
                home_score = _safe_int(
                    (match.get("home") or {}).get("score")
                )
                away_score = _safe_int(
                    (match.get("away") or {}).get("score")
                )
                if home_score is None or away_score is None:
                    continue
                is_home = home_id == team_id
                team_goals = home_score if is_home else away_score
                opp_goals = away_score if is_home else home_score
                gf += team_goals
                ga += opp_goals
                if team_goals > opp_goals:
                    results_str += "W"
                elif team_goals == opp_goals:
                    results_str += "D"
                else:
                    results_str += "L"
            except Exception as exc:
                logger.debug(
                    f"[{self.source_name}] form match parse error: {exc}"
                )

        # Injury list
        squad = data.get("squad") or {}
        injuries: list[Any] = squad.get("injured") or []
        injuries_count = len(injuries)

        # Next fixture
        next_matches: list[Any] = history.get("nextMatches") or []
        next_match: Optional[dict[str, Any]] = (
            next_matches[0] if next_matches else None
        )
        next_match_id: Optional[int] = None
        next_opponent: Optional[str] = None
        next_match_date: Optional[str] = None

        if next_match and isinstance(next_match, dict):
            next_match_id = next_match.get("id")
            home = next_match.get("home") or {}
            away = next_match.get("away") or {}
            next_opponent = (
                away.get("name") if home.get("id") == team_id else home.get("name")
            )
            next_match_date = (
                next_match.get("status", {}).get("utcTime")
                or next_match.get("utcTime")
            )

        return {
            "team_id": team_id,
            "team_name": team_name,
            "league": league,
            "league_id": league_id,
            "last5_results": results_str,
            "last5_goals_for": gf,
            "last5_goals_against": ga,
            "injuries_count": injuries_count,
            "next_match_id": next_match_id,
            "next_opponent": next_opponent,
            "next_match_date": str(next_match_date) if next_match_date else None,
        }

    # ------------------------------------------------------------------ #
    #  Mode: league_standings                                              #
    # ------------------------------------------------------------------ #

    def _collect_league_standings(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch current standings with xG enrichment.

        Returns columns:
          league, league_id, position, team_id, team_name,
          played, wins, draws, losses,
          goals_for, goals_against, gd, points,
          xg, xga, xgd
        """
        league: str = kwargs.get("league", "Premier League")
        league_id = FOTMOB_LEAGUES.get(league)
        if league_id is None:
            logger.warning(f"[{self.source_name}] Unknown league '{league}'")
            return pd.DataFrame()

        cache_key = f"fotmob_standings_{league_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(
                f"[{self.source_name}] Cache hit: standings {league}"
            )
            return pd.DataFrame(cached)

        url = f"{_API_BASE}/leagues?id={league_id}"
        try:
            data = self._fetch_json(url, headers=_FM_HEADERS)
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] standings fetch failed "
                f"for {league}: {exc}"
            )
            return pd.DataFrame()

        rows = self._parse_standings(data, league, league_id)
        if not rows:
            logger.warning(
                f"[{self.source_name}] No standings rows parsed for '{league}'"
            )
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["team_norm"] = df["team_name"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        self._set_cache(cache_key, df.to_dict(orient="list"))
        return df.sort_values("position").reset_index(drop=True)

    def _parse_standings(
        self,
        data: Any,
        league: str,
        league_id: int,
    ) -> list[dict]:
        """Extract standings rows from the leagues endpoint response."""
        rows: list[dict] = []
        if not isinstance(data, dict):
            return rows

        # FotMob table structure: data["table"] is a list of table objects
        for table_block in data.get("table", []):
            all_rows = (
                (table_block.get("data") or {})
                .get("table", {})
                .get("all", [])
            )
            for entry in all_rows:
                try:
                    xg_data = entry.get("xg") or {}
                    rows.append({
                        "league": league,
                        "league_id": league_id,
                        "position": _safe_int(entry.get("idx") or entry.get("rank")),
                        "team_id": entry.get("id"),
                        "team_name": entry.get("name") or entry.get("shortName", ""),
                        "played": _safe_int(entry.get("played") or entry.get("mp")),
                        "wins": _safe_int(entry.get("wins") or entry.get("w")),
                        "draws": _safe_int(entry.get("draws") or entry.get("d")),
                        "losses": _safe_int(entry.get("losses") or entry.get("l")),
                        "goals_for": _safe_int(
                            entry.get("scoresFor") or entry.get("gf")
                        ),
                        "goals_against": _safe_int(
                            entry.get("scoresAgainst") or entry.get("ga")
                        ),
                        "gd": _safe_int(
                            entry.get("goalConDiff") or entry.get("gd")
                        ),
                        "points": _safe_int(entry.get("pts") or entry.get("points")),
                        "xg": _safe_float(
                            xg_data.get("for") or entry.get("xG")
                        ),
                        "xga": _safe_float(
                            xg_data.get("against") or entry.get("xGA")
                        ),
                        "xgd": _safe_float(
                            xg_data.get("diff") or entry.get("xGD")
                        ),
                    })
                except Exception as exc:
                    logger.debug(
                        f"[{self.source_name}] standings row parse error: {exc}"
                    )

        return rows


# ------------------------------------------------------------------ #
#  Module-level helpers                                                #
# ------------------------------------------------------------------ #


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    try:
        return int(float(val)) if val is not None else None
    except (TypeError, ValueError):
        return None


def _flatten_stat_list(stats: list) -> dict:
    """Convert FotMob's [{key: ..., stat: ...}] format to a plain dict."""
    result: dict = {}
    for entry in stats:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key") or entry.get("title") or entry.get("name")
        val = entry.get("stat") or entry.get("value")
        if key:
            result[key] = val
    return result
