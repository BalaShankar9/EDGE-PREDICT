"""
Capology Collector — player and team wage data.

Wage bill is the single best predictor of league position (correlation ~0.9
in top leagues). Capology publishes weekly and annual wages per player and
aggregated totals per club.

Scrapes: https://www.capology.com/{league_path}/payrolls/2024-2025/

Two modes:
    "teams"   — one row per club: total wage bill, averages, squad stats
    "players" — one row per player: individual wage, contract, position, etc.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

CAPOLOGY_LEAGUES: dict[str, str] = {
    "Premier League": "uk/premier-league",
    "La Liga": "es/la-liga",
    "Bundesliga": "de/1-bundesliga",
    "Serie A": "it/serie-a",
    "Ligue 1": "fr/ligue-1",
    "Championship": "uk/championship",
}

_SEASON = "2024-2025"


class CapologyCollector(BaseCollector):
    """Collector for Capology wage data (team payrolls and per-player wages)."""

    source_name = "capology"
    base_url = "https://www.capology.com"
    request_delay = 4.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect wage data from Capology.

        Parameters
        ----------
        league : str, optional
            League name (e.g. "Premier League"). If None, collects all leagues.
        mode : str, optional
            "teams" (default) or "players".
        """
        league: Optional[str] = kwargs.get("league")
        mode: str = kwargs.get("mode", "teams")

        leagues = (
            {league: CAPOLOGY_LEAGUES[league]}
            if league
            else CAPOLOGY_LEAGUES
        )
        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            try:
                if mode == "players":
                    df = self._collect_players(league_name, league_path)
                else:
                    df = self._collect_teams(league_name, league_path)

                if df is not None and not df.empty:
                    frames.append(df)

            except Exception as e:
                logger.error(
                    f"[capology] Failed to collect {league_name} ({mode}): {e}"
                )
                continue

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Teams mode                                                          #
    # ------------------------------------------------------------------ #

    def _collect_teams(
        self, league_name: str, league_path: str
    ) -> Optional[pd.DataFrame]:
        """Scrape per-team wage aggregates for a league."""
        url = f"{self.base_url}/{league_path}/payrolls/{_SEASON}/"
        cache_key = f"capology_teams_{league_path.replace('/', '_')}_{_SEASON}"

        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[capology] Cache hit for teams {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url)
        html = response.text

        self._check_structure(html, ["payroll", "wage"])

        soup = BeautifulSoup(html, "html.parser")
        rows = self._parse_team_table(soup, league_name)

        if rows:
            # Assign wage_bill_rank based on total weekly wage bill descending
            rows_sorted = sorted(
                rows,
                key=lambda r: r.get("total_wage_bill_weekly") or 0,
                reverse=True,
            )
            for rank, r in enumerate(rows_sorted, start=1):
                r["wage_bill_rank"] = rank

            for r in rows:
                r["team_id"] = self.normalise_team(r["team"])
            self._set_cache(cache_key, rows)

        logger.info(
            f"[capology] Fetched team payrolls {league_name}: {len(rows)} teams"
        )
        return pd.DataFrame(rows) if rows else None

    def _parse_team_table(
        self, soup: BeautifulSoup, league_name: str
    ) -> list[dict]:
        """Parse the main payroll table and aggregate to team level."""
        # Capology renders a DataTable with id "table" or class "table"
        table = soup.select_one("table#table, table.table, table")
        if not table:
            logger.warning(f"[capology] No payroll table found for {league_name}")
            return []

        # Collect per-player rows first, then aggregate
        player_rows = self._parse_player_rows_from_table(table, league_name)
        if not player_rows:
            return []

        df = pd.DataFrame(player_rows)
        if "weekly_wage" not in df.columns or df["weekly_wage"].isna().all():
            return []

        team_rows: list[dict] = []
        for team, grp in df.groupby("team"):
            wages = grp["weekly_wage"].dropna()
            if wages.empty:
                continue

            highest_idx = grp["weekly_wage"].idxmax()
            highest_player = grp.loc[highest_idx, "player_name"] if pd.notna(highest_idx) else ""

            # Age: Capology shows ages as integers; average across squad
            ages = grp["age"].dropna()
            avg_age = round(ages.mean(), 1) if not ages.empty else None

            total_weekly = wages.sum()
            team_rows.append({
                "team": team,
                "total_wage_bill_weekly": int(total_weekly),
                "total_wage_bill_annual": int(total_weekly * 52),
                "avg_wage_weekly": int(wages.mean()),
                "median_wage_weekly": int(wages.median()),
                "highest_paid_player": highest_player,
                "highest_wage": int(wages.max()),
                "squad_size": len(grp),
                "avg_age": avg_age,
                "wage_bill_rank": None,  # filled after sort
                "league": league_name,
                "season": _SEASON,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            })

        return team_rows

    # ------------------------------------------------------------------ #
    #  Players mode                                                        #
    # ------------------------------------------------------------------ #

    def _collect_players(
        self, league_name: str, league_path: str
    ) -> Optional[pd.DataFrame]:
        """Scrape per-player wage data for a league."""
        url = f"{self.base_url}/{league_path}/payrolls/{_SEASON}/"
        cache_key = f"capology_players_{league_path.replace('/', '_')}_{_SEASON}"

        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[capology] Cache hit for players {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(url)
        html = response.text

        self._check_structure(html, ["payroll", "wage"])

        soup = BeautifulSoup(html, "html.parser")
        table = soup.select_one("table#table, table.table, table")
        if not table:
            logger.warning(f"[capology] No player table found for {league_name}")
            return None

        rows = self._parse_player_rows_from_table(table, league_name)

        if rows:
            for r in rows:
                r["team_id"] = self.normalise_team(r["team"])
            self._set_cache(cache_key, rows)

        logger.info(
            f"[capology] Fetched player wages {league_name}: {len(rows)} players"
        )
        return pd.DataFrame(rows) if rows else None

    def _parse_player_rows_from_table(
        self, table: Any, league_name: str
    ) -> list[dict]:
        """Parse individual player rows from a Capology payroll table.

        Capology column order (may vary): Player | Position | Age | Country |
        Weekly Wage | Annual Wage | Contract Expiry | Club
        """
        rows: list[dict] = []

        # Map header text → column index
        headers: list[str] = []
        thead = table.select_one("thead")
        if thead:
            for th in thead.select("th"):
                headers.append(th.get_text(strip=True).lower())

        def col_idx(candidates: list[str]) -> Optional[int]:
            for cand in candidates:
                for i, h in enumerate(headers):
                    if cand in h:
                        return i
            return None

        idx_player = col_idx(["player", "name"])
        idx_team = col_idx(["club", "team"])
        idx_position = col_idx(["position", "pos"])
        idx_age = col_idx(["age"])
        idx_country = col_idx(["country", "nationality", "nat"])
        idx_weekly = col_idx(["weekly", "weekly wage", "week"])
        idx_annual = col_idx(["annual", "yearly", "year"])
        idx_expiry = col_idx(["expiry", "contract", "expires"])

        tbody = table.select_one("tbody") or table
        for tr in tbody.select("tr"):
            row = self._parse_player_row(
                tr,
                league_name,
                idx_player=idx_player,
                idx_team=idx_team,
                idx_position=idx_position,
                idx_age=idx_age,
                idx_country=idx_country,
                idx_weekly=idx_weekly,
                idx_annual=idx_annual,
                idx_expiry=idx_expiry,
            )
            if row:
                rows.append(row)

        return rows

    def _parse_player_row(
        self,
        tr: Any,
        league_name: str,
        *,
        idx_player: Optional[int],
        idx_team: Optional[int],
        idx_position: Optional[int],
        idx_age: Optional[int],
        idx_country: Optional[int],
        idx_weekly: Optional[int],
        idx_annual: Optional[int],
        idx_expiry: Optional[int],
    ) -> Optional[dict]:
        """Parse a single player row from the Capology payroll table."""
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            def td_text(idx: Optional[int]) -> str:
                if idx is None or idx >= len(tds):
                    return ""
                return tds[idx].get_text(strip=True)

            # Player name — prefer anchor text
            player_name = ""
            if idx_player is not None and idx_player < len(tds):
                a = tds[idx_player].select_one("a")
                player_name = a.get_text(strip=True) if a else td_text(idx_player)
            if not player_name:
                # Fallback: first td with a link
                for td in tds:
                    a = td.select_one("a")
                    if a:
                        player_name = a.get_text(strip=True)
                        break
            if not player_name:
                return None

            # Team — anchor in club cell
            team = ""
            if idx_team is not None and idx_team < len(tds):
                a = tds[idx_team].select_one("a")
                team = a.get_text(strip=True) if a else td_text(idx_team)

            weekly_raw = td_text(idx_weekly)
            annual_raw = td_text(idx_annual)

            return {
                "player_name": player_name,
                "team": team,
                "position": td_text(idx_position),
                "weekly_wage": _parse_wage(weekly_raw),
                "annual_wage": _parse_wage(annual_raw),
                "age": _parse_int(td_text(idx_age)),
                "contract_expiry": td_text(idx_expiry),
                "country": td_text(idx_country),
                "league": league_name,
                "season": _SEASON,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[capology] Failed to parse player row: {e}")
            return None


# ------------------------------------------------------------------ #
#  Parsing helpers                                                     #
# ------------------------------------------------------------------ #

def _parse_wage(raw: str) -> Optional[int]:
    """Convert wage strings like '£45,000', '$120k', '€80,000' to int."""
    if not raw:
        return None
    # Strip currency symbols, commas, spaces
    cleaned = re.sub(r"[£$€,\s]", "", raw)
    # Handle shorthand: 45k → 45000
    k_match = re.match(r"^([\d.]+)[kK]$", cleaned)
    if k_match:
        return int(float(k_match.group(1)) * 1_000)
    m_match = re.match(r"^([\d.]+)[mM]$", cleaned)
    if m_match:
        return int(float(m_match.group(1)) * 1_000_000)
    try:
        return int(cleaned)
    except (ValueError, TypeError):
        return None


def _parse_int(raw: str) -> Optional[int]:
    """Parse a simple integer string, returning None on failure."""
    try:
        return int(raw.strip())
    except (ValueError, TypeError):
        return None
