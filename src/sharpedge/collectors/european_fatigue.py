"""
EuropeanFatigueCollector — tracks teams playing midweek European fixtures
(Champions League / Europa League / Conference League) and models their
fatigue for subsequent domestic matches.

Academic basis
--------------
Lago-Peñas (2009) and Dupont et al. (2010) demonstrate that teams competing
in European competition show measurable performance degradation in the
following domestic fixture:

    - 0.30 fewer goals scored per match (Europa League away fixture)
    - 0.20 more goals conceded per match
    - 15% lower win probability (Thursday EL kickoff → weekend PL)

This collector quantifies these effects using:
    1. UEFA competition schedule scraping
    2. FBref lineup/result data as primary source
    3. TravelCalculator for distance-based fatigue scoring
    4. A composite fatigue score modelling rest, travel, and intensity

Composite fatigue model
-----------------------
    base      : days_between < 4 → +0.30; < 3 → +0.60
    extra_time: extra_time → +0.20
    venue     : away European fixture → +0.15
    result    : loss (emotional drain) → +0.10
    travel    : midweek_distance > 2000 km → +0.20

Sources
-------
    Primary  : FBref /en/comps/{id}/schedule/  (squad-level schedules)
    Fallback : Sofascore /api/v1/sport/football/scheduled-events/{date}
    Travel   : TravelCalculator (local, no HTTP)

Usage
-----
    col = EuropeanFatigueCollector()

    # All upcoming weekend fixtures affected by midweek European travel
    df = col.collect(
        weekend_date="2025-03-16",   # the domestic match date
        competition="Champions League",  # or None for all
    )
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector
from sharpedge.collectors.travel_calculator import TravelCalculator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Constants                                                          #
# ------------------------------------------------------------------ #

# FBref competition IDs for European club competitions
_EURO_COMPS: dict[str, int] = {
    "Champions League":    8,
    "Europa League":       19,
    "Conference League":   882,
}

# Short labels used when the source returns only abbreviated names
_COMP_LABELS: dict[str, str] = {
    "Champions League":  "UCL",
    "Europa League":     "UEL",
    "Conference League": "UECL",
}

# Domestic league FBref IDs — used to fetch weekend schedule
_DOMESTIC_COMPS: dict[str, int] = {
    "Premier League": 9,
    "La Liga":        12,
    "Bundesliga":     20,
    "Serie A":        11,
    "Ligue 1":        13,
    "Championship":   10,
    "Eredivisie":     23,
}

# Fatigue model constants (calibrated to literature values)
_FATIGUE_BASE_LOW   = 0.30   # days_between < 4
_FATIGUE_BASE_HIGH  = 0.60   # days_between < 3
_FATIGUE_EXTRA_TIME = 0.20   # went to extra time
_FATIGUE_AWAY_EURO  = 0.15   # away European fixture
_FATIGUE_LOSS       = 0.10   # emotional drain from defeat
_FATIGUE_LONG_HAUL  = 0.20   # travel distance > 2000 km

# Fatigue → goal impact mappings (Lago-Peñas 2009, Dupont 2010 derived)
_GOALS_SCORED_IMPACT   = -0.30   # fewer goals scored per unit fatigue
_GOALS_CONCEDED_IMPACT =  0.20   # more goals conceded per unit fatigue
_WIN_PROB_IMPACT       = -0.15   # lower win probability per unit fatigue


class EuropeanFatigueCollector(BaseCollector):
    """Track European competition fixtures and model subsequent domestic fatigue.

    For each team that played a midweek European fixture, this collector
    computes a composite fatigue score and projects its impact on the
    upcoming domestic match.
    """

    source_name = "european_fatigue"
    base_url = "https://www.uefa.com"
    request_delay = 3.0

    def __init__(self) -> None:
        super().__init__()
        self._travel = TravelCalculator()

    # ------------------------------------------------------------------ #
    #  Main dispatch                                                       #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect European fatigue data for an upcoming weekend's fixtures.

        Parameters
        ----------
        weekend_date : str, optional
            Date of the domestic match in YYYY-MM-DD format.
            Defaults to next Saturday from today.
        competition : str, optional
            Filter to one European competition: 'Champions League',
            'Europa League', or 'Conference League'.
            If None (default), collects all three.
        lookback_days : int, optional
            How many days before weekend_date to look for European fixtures.
            Default: 7 (captures any midweek fixture in the same week).
        domestic_leagues : list[str], optional
            Which domestic leagues to pull weekend fixtures from.
            Default: all five major leagues.
        """
        weekend_date_str: Optional[str] = kwargs.get("weekend_date")
        competition: Optional[str] = kwargs.get("competition")
        lookback_days: int = int(kwargs.get("lookback_days", 7))
        domestic_leagues: list[str] = kwargs.get(
            "domestic_leagues",
            list(_DOMESTIC_COMPS.keys()),
        )

        # Resolve weekend date
        weekend_date = self._resolve_weekend_date(weekend_date_str)
        midweek_start = weekend_date - timedelta(days=lookback_days)

        logger.info(
            f"[{self.source_name}] Scanning European fixtures "
            f"{midweek_start.date()} → {weekend_date.date()}"
        )

        # Step 1: fetch midweek European results
        competitions = (
            {competition: _EURO_COMPS[competition]}
            if competition
            else _EURO_COMPS
        )

        euro_rows: list[dict] = []
        for comp_name, comp_id in competitions.items():
            rows = self._fetch_european_fixtures(
                comp_id, comp_name, midweek_start, weekend_date
            )
            euro_rows.extend(rows)

        if not euro_rows:
            logger.warning(
                f"[{self.source_name}] No European fixtures found for window"
            )
            return pd.DataFrame()

        euro_df = pd.DataFrame(euro_rows)

        # Step 2: fetch upcoming domestic fixtures
        weekend_fixtures: list[dict] = []
        for league in domestic_leagues:
            rows = self._fetch_domestic_fixtures(league, weekend_date)
            weekend_fixtures.extend(rows)

        weekend_df = pd.DataFrame(weekend_fixtures) if weekend_fixtures else pd.DataFrame()

        # Step 3: join and compute fatigue scores
        result = self._build_fatigue_records(euro_df, weekend_df, weekend_date)
        return result

    # ------------------------------------------------------------------ #
    #  Date helpers                                                        #
    # ------------------------------------------------------------------ #

    def _resolve_weekend_date(
        self, date_str: Optional[str]
    ) -> datetime:
        """Parse date string or default to the next Saturday."""
        if date_str:
            return datetime.strptime(date_str, "%Y-%m-%d")

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        days_ahead = (5 - now.weekday()) % 7  # Saturday = weekday 5
        if days_ahead == 0:
            days_ahead = 7
        return now + timedelta(days=days_ahead)

    # ------------------------------------------------------------------ #
    #  European fixtures scraper (FBref schedules)                        #
    # ------------------------------------------------------------------ #

    def _fetch_european_fixtures(
        self,
        comp_id: int,
        comp_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Scrape European competition schedule from FBref."""
        url = f"https://fbref.com/en/comps/{comp_id}/schedule/"
        cache_key = f"euro_schedule_{comp_id}_{start_date.date()}_{end_date.date()}"

        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[{self.source_name}] Cache hit: {cache_key}")
            return cached

        try:
            response = self._fetch(url)
            html = _uncomment_fbref(response.text)
            soup = BeautifulSoup(html, "html.parser")
            rows = self._parse_fbref_schedule(soup, comp_name, start_date, end_date)
            self._set_cache(cache_key, rows)
            logger.info(
                f"[{self.source_name}] {comp_name}: {len(rows)} fixtures in window"
            )
            return rows
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] Failed to fetch {comp_name} schedule: {exc}"
            )
            return []

    def _parse_fbref_schedule(
        self,
        soup: BeautifulSoup,
        comp_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Parse FBref schedule table into structured fixture rows."""
        table = None
        for t in soup.find_all("table"):
            tid = t.get("id", "")
            if "schedule" in tid or "matchlogs" in tid or "sched" in tid:
                table = t
                break

        if table is None:
            # Try the first table with relevant headers
            for t in soup.find_all("table"):
                headers = [th.get_text(strip=True).lower()
                           for th in t.find_all("th")]
                if "home" in headers and "away" in headers:
                    table = t
                    break

        if table is None:
            logger.warning(
                f"[{self.source_name}] No schedule table found for {comp_name}"
            )
            return []

        rows: list[dict] = []
        tbody = table.find("tbody")
        if tbody is None:
            return []

        for tr in tbody.find_all("tr"):
            if "thead" in tr.get("class", []):
                continue

            cells = {
                td.get("data-stat", ""): td.get_text(strip=True)
                for td in tr.find_all(["td", "th"])
            }

            date_str = cells.get("date", "")
            if not date_str:
                continue

            try:
                match_date = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue

            # Only fixtures within our lookback window
            if not (start_date <= match_date < end_date):
                continue

            home = cells.get("home_team", cells.get("squad", ""))
            away = cells.get("away_team", cells.get("opponent", ""))
            score = cells.get("score", cells.get("result", ""))

            if not home or not away:
                continue

            # Parse result
            home_goals, away_goals, extra_time = _parse_score(score)

            rows.append({
                "competition":      comp_name,
                "match_date":       match_date.strftime("%Y-%m-%d"),
                "home_team":        home,
                "away_team":        away,
                "home_goals":       home_goals,
                "away_goals":       away_goals,
                "extra_time":       extra_time,
                "venue_type_home":  "home",   # from home team's perspective
                "venue_type_away":  "away",   # from away team's perspective
                "score_raw":        score,
            })

        return rows

    # ------------------------------------------------------------------ #
    #  Domestic fixtures (FBref)                                          #
    # ------------------------------------------------------------------ #

    def _fetch_domestic_fixtures(
        self, league: str, weekend_date: datetime
    ) -> list[dict]:
        """Fetch that weekend's domestic schedule for a league."""
        comp_id = _DOMESTIC_COMPS.get(league)
        if comp_id is None:
            return []

        url = f"https://fbref.com/en/comps/{comp_id}/schedule/"
        cache_key = f"dom_schedule_{comp_id}_{weekend_date.date()}"

        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            response = self._fetch(url)
            html = _uncomment_fbref(response.text)
            soup = BeautifulSoup(html, "html.parser")

            # Weekend window: Fri–Mon
            window_start = weekend_date - timedelta(days=1)
            window_end   = weekend_date + timedelta(days=2)

            rows = self._parse_fbref_schedule(soup, league, window_start, window_end)
            # Tag with league
            for r in rows:
                r["domestic_league"] = league

            self._set_cache(cache_key, rows)
            logger.debug(
                f"[{self.source_name}] {league} weekend: {len(rows)} fixtures"
            )
            return rows
        except Exception as exc:
            logger.error(
                f"[{self.source_name}] Failed to fetch {league} schedule: {exc}"
            )
            return []

    # ------------------------------------------------------------------ #
    #  Fatigue record assembly                                            #
    # ------------------------------------------------------------------ #

    def _build_fatigue_records(
        self,
        euro_df: pd.DataFrame,
        weekend_df: pd.DataFrame,
        weekend_date: datetime,
    ) -> pd.DataFrame:
        """Join European and domestic fixtures; compute composite fatigue score.

        For each team that played a European fixture, find their domestic
        fixture and compute the full fatigue profile.
        """
        records: list[dict] = []

        for _, euro_row in euro_df.iterrows():
            # Process both home and away teams in the European fixture
            for side in ("home", "away"):
                team        = euro_row[f"{side}_team"]
                opponent_side = "away" if side == "home" else "home"
                euro_opponent = euro_row[f"{opponent_side}_team"]
                venue_type  = euro_row.get(f"venue_type_{side}", side)

                team_norm = self.normalise_team(str(team)) or str(team)

                # Result from this team's perspective
                if euro_row.get("home_goals") is not None and euro_row.get("away_goals") is not None:
                    tg = euro_row["home_goals"] if side == "home" else euro_row["away_goals"]
                    og = euro_row["away_goals"] if side == "home" else euro_row["home_goals"]
                    if tg is None or og is None:
                        result_char = "U"
                    elif tg > og:
                        result_char = "W"
                    elif tg == og:
                        result_char = "D"
                    else:
                        result_char = "L"
                else:
                    result_char = "U"  # Unknown/upcoming

                # Find weekend domestic fixture
                weekend_row = self._find_domestic_fixture(
                    team_norm, team, weekend_df
                )

                # Days between matches
                euro_date = datetime.strptime(str(euro_row["match_date"]), "%Y-%m-%d")
                if weekend_row is not None:
                    wknd_date_str = weekend_row.get("match_date", weekend_date.strftime("%Y-%m-%d"))
                    wknd_date = datetime.strptime(str(wknd_date_str), "%Y-%m-%d")
                else:
                    wknd_date = weekend_date
                days_between = (wknd_date - euro_date).days

                # Travel distance (away team travels to their opponent)
                travel_distance: Optional[float] = None
                if venue_type == "away":
                    travel_distance = self._travel.get_travel_distance(
                        euro_opponent, team
                    )

                # Composite fatigue score
                fatigue_score = self._compute_fatigue(
                    days_between=days_between,
                    extra_time=bool(euro_row.get("extra_time", False)),
                    venue_type=str(venue_type),
                    result=result_char,
                    travel_distance_km=travel_distance,
                )

                # Weekend fixture details
                weekend_opponent = None
                weekend_venue    = None
                if weekend_row is not None:
                    home_team_wk = str(weekend_row.get("home_team", ""))
                    away_team_wk = str(weekend_row.get("away_team", ""))
                    home_norm    = self.normalise_team(home_team_wk) or home_team_wk
                    if home_norm == team_norm or home_team_wk == team:
                        weekend_opponent = away_team_wk
                        weekend_venue    = "home"
                    else:
                        weekend_opponent = home_team_wk
                        weekend_venue    = "away"

                # Impact projections (scale by fatigue)
                scale = min(1.0, fatigue_score)
                expected_goal_impact    = round(_GOALS_SCORED_IMPACT * scale, 3)
                expected_concede_impact = round(_GOALS_CONCEDED_IMPACT * scale, 3)
                win_prob_adjustment     = round(_WIN_PROB_IMPACT * scale, 3)

                records.append({
                    "team":                      team,
                    "team_norm":                 team_norm,
                    "competition":               euro_row.get("competition", ""),
                    "midweek_match_date":         euro_row["match_date"],
                    "midweek_opponent":           euro_opponent,
                    "midweek_venue":              venue_type,
                    "midweek_result":             result_char,
                    "extra_time":                 bool(euro_row.get("extra_time", False)),
                    "midweek_goals_for":          euro_row.get("home_goals" if side == "home" else "away_goals"),
                    "midweek_goals_against":      euro_row.get("away_goals" if side == "home" else "home_goals"),
                    "weekend_match_date":         wknd_date.strftime("%Y-%m-%d"),
                    "weekend_opponent":           weekend_opponent,
                    "weekend_venue":              weekend_venue,
                    "days_between_matches":       days_between,
                    "travel_distance_km":         travel_distance,
                    "fatigue_score":              round(fatigue_score, 3),
                    "expected_goal_impact":       expected_goal_impact,
                    "expected_concede_impact":    expected_concede_impact,
                    "win_prob_adjustment":        win_prob_adjustment,
                    "minutes_played_estimate":    _estimate_minutes(extra_time=bool(euro_row.get("extra_time", False))),
                    "domestic_league":            weekend_row.get("domestic_league") if weekend_row is not None else None,
                    "high_risk":                  fatigue_score >= 0.5,
                })

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Sort by fatigue_score descending — highest risk teams first
        df = df.sort_values("fatigue_score", ascending=False).reset_index(drop=True)
        return df

    def _find_domestic_fixture(
        self,
        team_norm: str,
        team_raw: str,
        weekend_df: pd.DataFrame,
    ) -> Optional[dict]:
        """Find the domestic fixture for a given team in the weekend schedule."""
        if weekend_df.empty:
            return None

        for _, row in weekend_df.iterrows():
            home = str(row.get("home_team", ""))
            away = str(row.get("away_team", ""))

            home_norm = self.normalise_team(home) or home
            away_norm = self.normalise_team(away) or away

            if (
                home_norm == team_norm
                or away_norm == team_norm
                or home == team_raw
                or away == team_raw
            ):
                return row.to_dict()

        return None

    # ------------------------------------------------------------------ #
    #  Composite fatigue scoring                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_fatigue(
        days_between: int,
        extra_time: bool,
        venue_type: str,
        result: str,
        travel_distance_km: Optional[float],
    ) -> float:
        """Compute the composite fatigue score (0–1 scale).

        Scoring components:
            days_between < 4  → base +0.30
            days_between < 3  → base +0.60 (overrides the < 4 rule)
            extra_time        → +0.20
            away fixture      → +0.15
            loss              → +0.10
            travel > 2000 km  → +0.20

        Returns
        -------
        float
            Clamped to [0.0, 1.0].
        """
        score = 0.0

        # Rest penalty
        if days_between < 3:
            score += _FATIGUE_BASE_HIGH
        elif days_between < 4:
            score += _FATIGUE_BASE_LOW

        # Intensity penalties
        if extra_time:
            score += _FATIGUE_EXTRA_TIME

        if venue_type == "away":
            score += _FATIGUE_AWAY_EURO

        if result == "L":
            score += _FATIGUE_LOSS

        # Long-haul travel
        if travel_distance_km is not None and travel_distance_km > 2000:
            score += _FATIGUE_LONG_HAUL

        return min(1.0, score)


# ------------------------------------------------------------------ #
#  Module-level helpers                                               #
# ------------------------------------------------------------------ #

def _uncomment_fbref(html: str) -> str:
    """Strip HTML comment wrappers that FBref uses to hide tables."""
    import re
    return re.sub(r"<!--(.*?)-->", r"\1", html, flags=re.DOTALL)


def _parse_score(score_str: str) -> tuple[Optional[int], Optional[int], bool]:
    """Parse a score string like '2–1 (aet)' into (home_goals, away_goals, extra_time).

    Handles:
        '2-1'        → (2, 1, False)
        '2–1'        → (2, 1, False)    (en-dash)
        '2-1 (aet)'  → (2, 1, True)
        '1-1 (pens)' → (1, 1, True)     (penalty shootout implies AET)
        ''           → (None, None, False)
    """
    if not score_str or score_str.strip() in ("", "–", "-"):
        return None, None, False

    extra_time = any(
        marker in score_str.lower()
        for marker in ("aet", "a.e.t", "pens", "pen.", "ext")
    )

    # Normalise dashes and strip extra annotations
    clean = (
        score_str
        .replace("–", "-")      # en-dash → hyphen
        .replace("—", "-")      # em-dash → hyphen
        .split("(")[0]          # drop "(aet)" suffix
        .strip()
    )

    parts = clean.split("-")
    if len(parts) != 2:
        return None, None, extra_time

    try:
        return int(parts[0].strip()), int(parts[1].strip()), extra_time
    except (ValueError, AttributeError):
        return None, None, extra_time


def _estimate_minutes(extra_time: bool) -> int:
    """Estimate minutes played by typical starting XI.

    For a squad of 11 players all playing full time:
        Normal time  → 11 × 90  = 990 minutes
        Extra time   → 11 × 120 = 1320 minutes

    Returns the total squad minutes (standard proxy for cumulative load).
    """
    return 1320 if extra_time else 990
