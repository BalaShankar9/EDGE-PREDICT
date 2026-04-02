"""
SofascoreCollector — live match data, team form, and pre-match stats
via the Sofascore public JSON API.

No HTML scraping needed: the API returns JSON directly.

Modes
-----
fixtures  Scheduled events for a date range.
form      Last-5 results for each team in the fetched fixtures.
stats     Pre-match statistics for a specific event.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

_API_BASE = "https://api.sofascore.com/api/v1"

# Extra headers that Sofascore expects on API calls
_API_HEADERS: dict[str, str] = {
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}


class SofascoreCollector(BaseCollector):
    """Collector for Sofascore live / scheduled match data.

    Usage
    -----
    col = SofascoreCollector()

    # Fixture list for today + tomorrow
    fixtures_df = col.collect(mode="fixtures")

    # Form table for teams in those fixtures
    form_df     = col.collect(mode="form", fixtures_df=fixtures_df)

    # Pre-match stats for a specific event
    stats_df    = col.collect(mode="stats", event_id=12345678)
    """

    source_name = "sofascore"
    base_url = "https://www.sofascore.com"
    request_delay = 3.0

    # ------------------------------------------------------------------ #
    #  Main dispatch                                                       #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Dispatch to the appropriate collection mode.

        Parameters
        ----------
        mode : str
            One of ``"fixtures"``, ``"form"``, ``"stats"``.
            Defaults to ``"fixtures"``.
        date : str, optional
            Start date in YYYY-MM-DD format (fixtures mode).  Defaults to today.
        days_ahead : int, optional
            How many additional days to fetch (fixtures mode).  Default 1.
        fixtures_df : pd.DataFrame, optional
            DataFrame returned by a previous fixtures call (form mode).
        event_id : int, optional
            Sofascore event ID (stats mode).
        """
        mode: str = kwargs.get("mode", "fixtures")

        if mode == "fixtures":
            return self._collect_fixtures(**kwargs)
        if mode == "form":
            return self._collect_form(**kwargs)
        if mode == "stats":
            return self._collect_stats(**kwargs)

        raise ValueError(
            f"[{self.source_name}] Unknown mode '{mode}'. "
            "Use 'fixtures', 'form', or 'stats'."
        )

    # ------------------------------------------------------------------ #
    #  Mode: fixtures                                                      #
    # ------------------------------------------------------------------ #

    def _collect_fixtures(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch scheduled football events for a date range.

        Returns columns:
          event_id, home_team, away_team, home_team_id, away_team_id,
          tournament, tournament_id, season_id, start_time, status,
          match_date
        """
        today_dt = datetime.now(timezone.utc)
        date_str: str = kwargs.get("date", today_dt.strftime("%Y-%m-%d"))
        days_ahead: int = int(kwargs.get("days_ahead", 1))

        start_date = datetime.strptime(date_str, "%Y-%m-%d")
        frames: list[pd.DataFrame] = []

        for offset in range(days_ahead + 1):
            target = start_date + timedelta(days=offset)
            target_str = target.strftime("%Y-%m-%d")
            url = f"{_API_BASE}/sport/football/scheduled-events/{target_str}"

            try:
                data = self._fetch_json(url, headers=_API_HEADERS)
                rows = self._parse_scheduled_events(data, target_str)
                if rows:
                    frames.append(pd.DataFrame(rows))
                logger.info(
                    f"[{self.source_name}] {len(rows)} fixtures for {target_str}"
                )
            except Exception as exc:
                logger.error(
                    f"[{self.source_name}] fixtures fetch failed for "
                    f"{target_str}: {exc}"
                )

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["home_team_norm"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        df["away_team_norm"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x)) or x
        )
        return df

    def _parse_scheduled_events(
        self, data: Any, date_str: str
    ) -> list[dict]:
        rows: list[dict] = []
        events: list[Any] = []

        if isinstance(data, dict):
            events = data.get("events", [])
        elif isinstance(data, list):
            events = data

        for event in events:
            try:
                eid = event.get("id")
                home = event.get("homeTeam", {})
                away = event.get("awayTeam", {})
                tournament = event.get("tournament", {})
                season = event.get("season", {})
                status = event.get("status", {})

                home_name = home.get("name") or home.get("shortName")
                away_name = away.get("name") or away.get("shortName")
                if not home_name or not away_name:
                    continue

                start_ts = event.get("startTimestamp")
                start_time: Optional[str] = None
                if start_ts:
                    start_time = datetime.fromtimestamp(
                        start_ts, tz=timezone.utc
                    ).isoformat()

                rows.append({
                    "event_id": eid,
                    "home_team": home_name,
                    "away_team": away_name,
                    "home_team_id": home.get("id"),
                    "away_team_id": away.get("id"),
                    "tournament": tournament.get("name"),
                    "tournament_id": tournament.get("uniqueTournament", {}).get("id"),
                    "season_id": season.get("id"),
                    "start_time": start_time,
                    "status": status.get("type") or status.get("description"),
                    "match_date": date_str,
                })
            except Exception as exc:
                logger.debug(f"[{self.source_name}] event parse error: {exc}")

        return rows

    # ------------------------------------------------------------------ #
    #  Mode: form                                                          #
    # ------------------------------------------------------------------ #

    def _collect_form(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch last-5 results for each unique team in a fixtures DataFrame.

        Parameters
        ----------
        fixtures_df : pd.DataFrame
            Must have columns ``home_team_id`` and ``away_team_id``
            containing Sofascore team IDs.

        Returns columns:
          team_id, team_name,
          last5_wins, last5_draws, last5_losses,
          last5_goals_for, last5_goals_against, last5_gd,
          last5_form_string
        """
        fixtures_df: Optional[pd.DataFrame] = kwargs.get("fixtures_df")
        if fixtures_df is None or fixtures_df.empty:
            logger.warning(f"[{self.source_name}] form mode: no fixtures_df provided")
            return pd.DataFrame()

        if "home_team_id" not in fixtures_df.columns or "away_team_id" not in fixtures_df.columns:
            logger.warning(
                f"[{self.source_name}] form mode: fixtures_df missing "
                "home_team_id / away_team_id columns"
            )
            return pd.DataFrame()

        # Collect unique (team_id, team_name) pairs
        home_pairs = (
            fixtures_df[["home_team_id", "home_team"]]
            .rename(columns={"home_team_id": "team_id", "home_team": "team_name"})
        )
        away_pairs = (
            fixtures_df[["away_team_id", "away_team"]]
            .rename(columns={"away_team_id": "team_id", "away_team": "team_name"})
        )
        teams = (
            pd.concat([home_pairs, away_pairs], ignore_index=True)
            .dropna(subset=["team_id"])
            .drop_duplicates(subset=["team_id"])
        )

        rows: list[dict] = []
        for _, team_row in teams.iterrows():
            team_id = team_row["team_id"]
            team_name = team_row.get("team_name", "")
            result = self._fetch_team_form(int(team_id), str(team_name))
            if result:
                rows.append(result)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def _fetch_team_form(self, team_id: int, team_name: str) -> Optional[dict]:
        """Fetch and summarise last-5 events for a team."""
        url = f"{_API_BASE}/team/{team_id}/events/last/5"
        try:
            data = self._fetch_json(url, headers=_API_HEADERS)
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] form fetch failed for "
                f"team {team_id} ({team_name}): {exc}"
            )
            return None

        events: list[Any] = []
        if isinstance(data, dict):
            events = data.get("events", [])
        elif isinstance(data, list):
            events = data

        wins = draws = losses = gf = ga = 0
        form_chars: list[str] = []

        for event in events[-5:]:  # Safety: take at most last 5
            try:
                home_id = event.get("homeTeam", {}).get("id")
                home_score = (
                    event.get("homeScore", {}).get("current")
                    or event.get("homeScore", {}).get("normaltime")
                )
                away_score = (
                    event.get("awayScore", {}).get("current")
                    or event.get("awayScore", {}).get("normaltime")
                )
                if home_score is None or away_score is None:
                    continue

                home_score = int(home_score)
                away_score = int(away_score)
                is_home = home_id == team_id

                team_goals = home_score if is_home else away_score
                opp_goals = away_score if is_home else home_score
                gf += team_goals
                ga += opp_goals

                if team_goals > opp_goals:
                    wins += 1
                    form_chars.append("W")
                elif team_goals == opp_goals:
                    draws += 1
                    form_chars.append("D")
                else:
                    losses += 1
                    form_chars.append("L")
            except Exception as exc:
                logger.debug(f"[{self.source_name}] form event parse error: {exc}")

        return {
            "team_id": team_id,
            "team_name": team_name,
            "last5_wins": wins,
            "last5_draws": draws,
            "last5_losses": losses,
            "last5_goals_for": gf,
            "last5_goals_against": ga,
            "last5_gd": gf - ga,
            "last5_form_string": "".join(form_chars),
        }

    # ------------------------------------------------------------------ #
    #  Mode: stats                                                         #
    # ------------------------------------------------------------------ #

    def _collect_stats(self, **kwargs: Any) -> pd.DataFrame:
        """Fetch pre-match (or live) statistics for a specific event.

        Parameters
        ----------
        event_id : int
            Sofascore event ID.

        Returns columns:
          event_id, stat_name, home_value, away_value, stat_type
        """
        event_id: Optional[int] = kwargs.get("event_id")
        if event_id is None:
            raise ValueError(
                f"[{self.source_name}] stats mode requires 'event_id' kwarg"
            )

        # Primary: live/post-match statistics
        stats_url = f"{_API_BASE}/event/{event_id}/statistics"
        # Fallback: event details (includes some pre-match info)
        detail_url = f"{_API_BASE}/event/{event_id}"

        rows: list[dict] = []

        try:
            data = self._fetch_json(stats_url, headers=_API_HEADERS)
            rows.extend(self._parse_statistics(data, event_id))
        except Exception as exc:
            logger.warning(
                f"[{self.source_name}] stats fetch failed for "
                f"event {event_id}: {exc}. Falling back to event details."
            )
            try:
                data = self._fetch_json(detail_url, headers=_API_HEADERS)
                rows.extend(self._parse_event_detail_stats(data, event_id))
            except Exception as exc2:
                logger.error(
                    f"[{self.source_name}] event detail fetch failed for "
                    f"event {event_id}: {exc2}"
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Pivot to wide format: one row per event with stat columns
        wide = self._pivot_stats(df)
        return wide

    def _parse_statistics(self, data: Any, event_id: int) -> list[dict]:
        """Parse the /event/{id}/statistics response."""
        rows: list[dict] = []
        groups: list[Any] = []

        if isinstance(data, dict):
            groups = data.get("statistics", [])
            # Sometimes wrapped in a list of period groups
            if not groups:
                groups = data.get("statisticsGroups", [])
        elif isinstance(data, list):
            groups = data

        for group in groups:
            # Each group has a period (ALL, 1ST, 2ND) and items
            period = group.get("period", "ALL") if isinstance(group, dict) else "ALL"
            items = group.get("statisticsItems", []) if isinstance(group, dict) else []

            for item in items:
                try:
                    name = item.get("name") or item.get("key") or item.get("type", "")
                    home_val = item.get("home")
                    away_val = item.get("away")
                    stat_type = item.get("statisticsType") or item.get("type", "")

                    rows.append({
                        "event_id": event_id,
                        "period": period,
                        "stat_name": name,
                        "home_value": home_val,
                        "away_value": away_val,
                        "stat_type": stat_type,
                    })
                except Exception as exc:
                    logger.debug(
                        f"[{self.source_name}] stat item parse error: {exc}"
                    )

        return rows

    def _parse_event_detail_stats(self, data: Any, event_id: int) -> list[dict]:
        """Extract the limited pre-match stats available in the event detail."""
        rows: list[dict] = []
        if not isinstance(data, dict):
            return rows

        event = data.get("event", data)

        # Team rankings / form often exposed directly on the event object
        stat_fields = [
            ("homeTeamForm", "awayTeamForm", "form_string"),
            ("homeTeamSeasonGoals", "awayTeamSeasonGoals", "season_goals"),
            ("homeTeamSeasonGoalsConc", "awayTeamSeasonGoalsConc", "season_goals_conceded"),
        ]

        for home_key, away_key, name in stat_fields:
            hv = event.get(home_key)
            av = event.get(away_key)
            if hv is not None or av is not None:
                rows.append({
                    "event_id": event_id,
                    "period": "PRE",
                    "stat_name": name,
                    "home_value": hv,
                    "away_value": av,
                    "stat_type": "info",
                })

        return rows

    # ------------------------------------------------------------------ #
    #  Standings helper (bonus — not a mode, but useful for callers)      #
    # ------------------------------------------------------------------ #

    def fetch_standings(
        self,
        tournament_id: int,
        season_id: int,
        standing_type: str = "total",
    ) -> pd.DataFrame:
        """Fetch standings for a tournament season.

        Parameters
        ----------
        tournament_id : int
            Sofascore unique-tournament ID.
        season_id : int
            Season ID.
        standing_type : str
            One of ``"total"``, ``"home"``, ``"away"``.

        Returns
        -------
        pd.DataFrame
            Columns: position, team_id, team_name, played, wins, draws,
            losses, goals_for, goals_against, gd, points
        """
        url = (
            f"{_API_BASE}/unique-tournament/{tournament_id}"
            f"/season/{season_id}/standings/{standing_type}"
        )
        try:
            data = self._fetch_json(url, headers=_API_HEADERS)
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] standings fetch failed for "
                f"tournament {tournament_id} season {season_id}: {exc}"
            )
            return pd.DataFrame()

        rows: list[dict] = []
        standings_list: list[Any] = []

        if isinstance(data, dict):
            # Top-level key is usually "standings" which is a list of groups
            for group in data.get("standings", []):
                standings_list.extend(group.get("rows", []))
        elif isinstance(data, list):
            for group in data:
                if isinstance(group, dict):
                    standings_list.extend(group.get("rows", []))

        for row in standings_list:
            try:
                team = row.get("team", {})
                rows.append({
                    "position": row.get("position"),
                    "team_id": team.get("id"),
                    "team_name": team.get("name") or team.get("shortName"),
                    "played": row.get("matches"),
                    "wins": row.get("wins"),
                    "draws": row.get("draws"),
                    "losses": row.get("losses"),
                    "goals_for": row.get("scoresFor"),
                    "goals_against": row.get("scoresAgainst"),
                    "gd": row.get("scoreDiffFormatted") or (
                        (row.get("scoresFor", 0) or 0) - (row.get("scoresAgainst", 0) or 0)
                    ),
                    "points": row.get("points"),
                })
            except Exception as exc:
                logger.debug(f"[{self.source_name}] standings row parse error: {exc}")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["team_name_norm"] = df["team_name"].apply(
            lambda x: self.normalise_team(str(x)) or x if x else x
        )
        return df.sort_values("position").reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pivot_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Pivot long stats DataFrame to wide format.

        Returns one row per event with columns like:
          event_id, home_possession, away_possession,
          home_shots_on_target, away_shots_on_target, home_xg, away_xg, ...
        """
        if df.empty:
            return df

        # Keep only the "ALL" period rows to avoid duplication
        all_period = df[df["period"].isin(["ALL", "PRE"])] if "period" in df.columns else df

        records: list[dict] = []
        for event_id, grp in all_period.groupby("event_id"):
            wide: dict[str, Any] = {"event_id": event_id}
            for _, stat_row in grp.iterrows():
                name_raw = str(stat_row.get("stat_name", "")).lower()
                # Normalise common stat names
                name = (
                    name_raw
                    .replace(" ", "_")
                    .replace("/", "_per_")
                    .replace("%", "pct")
                    .replace("(", "")
                    .replace(")", "")
                    .strip("_")
                )
                wide[f"home_{name}"] = stat_row.get("home_value")
                wide[f"away_{name}"] = stat_row.get("away_value")
            records.append(wide)

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records)
