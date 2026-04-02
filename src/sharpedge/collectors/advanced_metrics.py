"""
AdvancedMetricsCollector — FBref advanced stats scraper.

Scrapes expected threat (xT proxies), PPDA, pressing stats, and progressive
passes directly from FBref's HTML tables for supported competitions.

FBref is strict about rate limiting — request_delay is set to 5.0 seconds
with additional jitter from the base class.

Modes
-----
shooting    Per-team shooting quality: xG, npxG, shots, goals vs xG
passing     Progressive passes, key passes, final-third entries, crosses
defense     Tackles, blocks, interceptions, clearances, errors
possession  Touches in pen area, progressive carries, dribbles
pressing    PPDA proxy, high press successes, ball recoveries
all         Merge all modes into one comprehensive DataFrame

Usage
-----
    col = AdvancedMetricsCollector()

    # Single mode
    df = col.collect(mode="shooting", league="Premier League")

    # All modes merged
    df = col.collect(mode="all", league="Bundesliga", season="2024-2025")
"""

import logging
import re
import time
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

# FBref competition IDs — stat pages are at /en/comps/{id}/{stat}/
FBREF_COMPS: dict[str, int] = {
    "Premier League": 9,
    "La Liga": 12,
    "Bundesliga": 20,
    "Serie A": 11,
    "Ligue 1": 13,
    "Championship": 10,
    "Eredivisie": 23,
}

# Valid collection modes
_MODES = ("shooting", "passing", "defense", "possession", "pressing", "all")

# Table IDs in FBref HTML — squad-level "for" tables
_TABLE_IDS: dict[str, str] = {
    "shooting": "stats_shooting_squads",
    "passing": "stats_passing_squads",
    "passing_types": "stats_passing_types_squads",
    "defense": "stats_defense_squads",
    "possession": "stats_possession_squads",
    "misc": "stats_misc_squads",
}

# Per-mode column extraction config
# Each entry is (fbref_header_text_fragment, output_column_name, cast_type)
# Header fragments are lowercased for matching.
_SHOOTING_COLS = [
    # (header_key, col_name)  — matched against flattened header labels
    ("squad",            "team"),
    ("90s",              "_90s"),          # denominator for per-90 calcs
    ("sh",               "_shots_total"),
    ("sot",              "_shots_on_target"),
    ("xg",               "_xg_total"),
    ("npxg",             "_npxg_total"),
    ("g-xg",             "_goals_minus_xg"),
]

_PASSING_COLS = [
    ("squad",            "team"),
    ("90s",              "_90s"),
    ("cmp%",             "_pass_completion_pct"),
    ("prg",              "_progressive_passes"),
    ("kp",               "_key_passes"),
    ("1/3",              "_passes_into_final_third"),
    ("ppa",              "_passes_into_penalty_area"),
    ("crs",              "_crosses"),
]

_DEFENSE_COLS = [
    ("squad",            "team"),
    ("90s",              "_90s"),
    ("tkl",              "_tackles"),
    ("tkl%",             "_tackles_won_pct"),
    ("blocks",           "_blocks"),
    ("int",              "_interceptions"),
    ("clr",              "_clearances"),
    ("err",              "_errors_leading_to_shot"),
]

_POSSESSION_COLS = [
    ("squad",            "team"),
    ("90s",              "_90s"),
    ("poss",             "_possession_pct"),
    ("touch",            "_touches_att_pen"),
    ("prg",              "_progressive_carries"),
    ("prgr",             "_progressive_passes_received"),
    ("succ",             "_dribbles_completed"),
]

# Pressing / misc: PPDA requires calculating pressures / defensive actions.
# FBref misc stats give us pressures and recoveries directly.
_MISC_COLS = [
    ("squad",            "team"),
    ("90s",              "_90s"),
    ("press",            "_pressures"),
    ("succ",             "_press_successes"),
    ("recov",            "_recoveries"),
    ("crd",              "_cards_yellow"),  # secondary
]


class AdvancedMetricsCollector(BaseCollector):
    """Scrapes advanced per-team statistics from FBref HTML tables.

    All metrics are returned as raw totals/percentages; per-90 normalisation
    is applied in post-processing where the denominator is available.
    """

    source_name = "advanced_metrics"
    base_url = "https://fbref.com"
    request_delay = 5.0  # FBref enforces strict rate limits

    # ------------------------------------------------------------------ #
    #  Main dispatch                                                       #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect advanced metrics from FBref.

        Parameters
        ----------
        mode : str
            One of 'shooting', 'passing', 'defense', 'possession',
            'pressing', 'all'. Default: 'shooting'.
        league : str
            Competition name matching FBREF_COMPS keys.
            Default: 'Premier League'.
        season : str, optional
            Season string like '2024-2025'. If omitted, FBref returns
            the current season automatically.
        """
        mode: str = kwargs.get("mode", "shooting")
        league: str = kwargs.get("league", "Premier League")
        season: Optional[str] = kwargs.get("season")

        if mode not in _MODES:
            raise ValueError(
                f"[{self.source_name}] Unknown mode '{mode}'. "
                f"Use one of: {_MODES}"
            )

        if league not in FBREF_COMPS:
            raise ValueError(
                f"[{self.source_name}] Unknown league '{league}'. "
                f"Supported: {list(FBREF_COMPS.keys())}"
            )

        comp_id = FBREF_COMPS[league]

        if mode == "all":
            return self._collect_all(comp_id, league, season)

        return self._collect_mode(mode, comp_id, league, season)

    # ------------------------------------------------------------------ #
    #  Mode: all                                                           #
    # ------------------------------------------------------------------ #

    def _collect_all(
        self,
        comp_id: int,
        league: str,
        season: Optional[str],
    ) -> pd.DataFrame:
        """Fetch all modes and merge on normalised team name."""
        frames: dict[str, pd.DataFrame] = {}
        single_modes = ("shooting", "passing", "defense", "possession", "pressing")

        for mode in single_modes:
            try:
                df = self._collect_mode(mode, comp_id, league, season)
                if not df.empty:
                    frames[mode] = df
                    logger.info(
                        f"[{self.source_name}] {mode}: {len(df)} teams"
                    )
            except Exception as exc:
                logger.warning(
                    f"[{self.source_name}] {mode} collection failed: {exc}"
                )

        if not frames:
            return pd.DataFrame()

        # Merge all on 'team'
        merged: Optional[pd.DataFrame] = None
        for mode, df in frames.items():
            if merged is None:
                merged = df
            else:
                # Use outer join to keep all teams even if one mode is missing
                suffix = f"_{mode}"
                common = [c for c in df.columns if c != "team" and c in merged.columns]
                df = df.rename(columns={c: f"{c}{suffix}" for c in common})
                merged = merged.merge(df, on="team", how="outer")

        return merged.reset_index(drop=True) if merged is not None else pd.DataFrame()

    # ------------------------------------------------------------------ #
    #  Mode dispatchers                                                    #
    # ------------------------------------------------------------------ #

    def _collect_mode(
        self,
        mode: str,
        comp_id: int,
        league: str,
        season: Optional[str],
    ) -> pd.DataFrame:
        """Route to the correct stat-page parser."""
        dispatch = {
            "shooting":  self._parse_shooting,
            "passing":   self._parse_passing,
            "defense":   self._parse_defense,
            "possession": self._parse_possession,
            "pressing":  self._parse_pressing,
        }
        parser = dispatch[mode]
        stat_path = "misc" if mode == "pressing" else mode
        soup = self._fetch_stat_page(comp_id, stat_path, season)
        df = parser(soup, comp_id, league)
        df["league"] = league
        df["mode"] = mode
        return df

    # ------------------------------------------------------------------ #
    #  HTTP helpers                                                        #
    # ------------------------------------------------------------------ #

    def _build_url(
        self,
        comp_id: int,
        stat: str,
        season: Optional[str],
    ) -> str:
        """Construct the FBref stats URL."""
        if season:
            # e.g. /en/comps/9/2024-2025/shooting/2024-2025-Premier-League-Stats
            return (
                f"{self.base_url}/en/comps/{comp_id}/{season}/{stat}/"
            )
        return f"{self.base_url}/en/comps/{comp_id}/{stat}/"

    def _fetch_stat_page(
        self,
        comp_id: int,
        stat: str,
        season: Optional[str],
    ) -> BeautifulSoup:
        """Fetch and parse an FBref squad stats page."""
        url = self._build_url(comp_id, stat, season)
        cache_key = f"fbref_adv_{comp_id}_{stat}_{season or 'current'}"

        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[{self.source_name}] Cache hit: {cache_key}")
            return BeautifulSoup(cached["html"], "html.parser")

        response = self._fetch(url)
        html = response.text

        # FBref buries tables in HTML comments — uncomment them
        html = _uncomment_tables(html)

        self._set_cache(cache_key, {"html": html})
        return BeautifulSoup(html, "html.parser")

    # ------------------------------------------------------------------ #
    #  Table parsing helpers                                               #
    # ------------------------------------------------------------------ #

    def _find_squad_table(
        self, soup: BeautifulSoup, table_id: str
    ) -> Optional[BeautifulSoup]:
        """Locate the squad-level stats table by ID."""
        table = soup.find("table", {"id": table_id})
        if table is None:
            # Try partial match — FBref sometimes appends season suffix
            for t in soup.find_all("table"):
                tid = t.get("id", "")
                if table_id in tid and "opponent" not in tid:
                    return t
        return table

    def _table_to_df(
        self, table: BeautifulSoup, league: str
    ) -> pd.DataFrame:
        """Convert an FBref HTML table (with multi-level headers) to DataFrame."""
        if table is None:
            return pd.DataFrame()

        # FBref uses <thead> with two rows: group header + stat header
        thead = table.find("thead")
        header_rows = thead.find_all("tr") if thead else []

        # Build flat column labels from the last header row, prefixing group
        # where relevant to distinguish duplicate stat names (e.g. "Sh" in
        # shooting vs possession).
        cols: list[str] = []
        if header_rows:
            last_header = header_rows[-1]
            for th in last_header.find_all(["th", "td"]):
                text = th.get_text(strip=True).lower()
                data_stat = th.get("data-stat", "").lower()
                cols.append(data_stat or text or f"col_{len(cols)}")
        else:
            cols = []

        # Parse data rows
        tbody = table.find("tbody")
        if tbody is None:
            return pd.DataFrame()

        rows: list[dict] = []
        for tr in tbody.find_all("tr"):
            if "thead" in tr.get("class", []) or tr.get("class") == ["thead"]:
                continue  # Skip repeat headers embedded in tbody

            cells = tr.find_all(["td", "th"])
            if not cells:
                continue

            row: dict[str, Any] = {}
            for i, cell in enumerate(cells):
                key = (
                    cell.get("data-stat", "").lower()
                    or (cols[i] if i < len(cols) else f"col_{i}")
                )
                val = cell.get_text(strip=True)
                row[key] = val if val != "" else None

            # Skip summary / blank rows
            team_val = row.get("team") or row.get("squad", "")
            if not team_val or team_val.lower() in ("squad", ""):
                continue

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Normalise 'squad' -> 'team' if needed
        if "squad" in df.columns and "team" not in df.columns:
            df = df.rename(columns={"squad": "team"})
        elif "squad" in df.columns:
            df["team"] = df["squad"].combine_first(df["team"])
            df = df.drop(columns=["squad"])

        # Apply team name normalisation
        if "team" in df.columns:
            df["team_raw"] = df["team"].copy()
            df["team"] = df["team"].apply(
                lambda x: self.normalise_team(str(x)) or str(x)
            )

        return df

    def _to_float(self, series: pd.Series) -> pd.Series:
        """Coerce a Series to float, handling % and comma-formatted numbers."""
        return (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
            .replace({"": None, "None": None, "nan": None})
            .astype(float, errors="ignore")
        )

    def _per90(self, series: pd.Series, ninety_series: pd.Series) -> pd.Series:
        """Divide a raw-total Series by 90s played."""
        ninety = pd.to_numeric(ninety_series, errors="coerce")
        total = pd.to_numeric(series, errors="coerce")
        return (total / ninety.replace(0, pd.NA)).round(3)

    # ------------------------------------------------------------------ #
    #  Mode: shooting                                                      #
    # ------------------------------------------------------------------ #

    def _parse_shooting(
        self, soup: BeautifulSoup, comp_id: int, league: str
    ) -> pd.DataFrame:
        """Parse FBref shooting stats page.

        Returns: team, shots_90, shots_on_target_90, goals_per_shot,
                 xg_90, npxg_90, xg_per_shot, goals_minus_xg
        """
        table = self._find_squad_table(soup, _TABLE_IDS["shooting"])
        if table is None:
            logger.warning(f"[{self.source_name}] Shooting table not found for {league}")
            return pd.DataFrame()

        raw = self._table_to_df(table, league)
        if raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()
        out["team"] = raw.get("team", raw.get("squad", pd.Series(dtype=str)))

        nineties = pd.to_numeric(raw.get("90s", pd.Series(dtype=float)), errors="coerce")

        # Shots
        shots = pd.to_numeric(raw.get("sh", pd.Series(dtype=float)), errors="coerce")
        sot   = pd.to_numeric(raw.get("sot", pd.Series(dtype=float)), errors="coerce")
        gls   = pd.to_numeric(raw.get("gls", pd.Series(dtype=float)), errors="coerce")
        xg    = pd.to_numeric(raw.get("xg", pd.Series(dtype=float)), errors="coerce")
        npxg  = pd.to_numeric(raw.get("npxg", pd.Series(dtype=float)), errors="coerce")

        out["shots_90"]            = (shots / nineties.replace(0, pd.NA)).round(3)
        out["shots_on_target_90"]  = (sot / nineties.replace(0, pd.NA)).round(3)
        out["goals_per_shot"]      = (gls / shots.replace(0, pd.NA)).round(3)
        out["xg_90"]               = (xg / nineties.replace(0, pd.NA)).round(3)
        out["npxg_90"]             = (npxg / nineties.replace(0, pd.NA)).round(3)
        out["xg_per_shot"]         = (xg / shots.replace(0, pd.NA)).round(3)

        # goals_minus_xg already a direct stat on FBref
        g_xg = raw.get("g-xg", raw.get("g_minus_xg", pd.Series(dtype=float)))
        out["goals_minus_xg"] = pd.to_numeric(g_xg, errors="coerce")

        return out.dropna(subset=["team"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Mode: passing                                                       #
    # ------------------------------------------------------------------ #

    def _parse_passing(
        self, soup: BeautifulSoup, comp_id: int, league: str
    ) -> pd.DataFrame:
        """Parse FBref passing stats page.

        Returns: team, pass_completion_pct, progressive_passes_90,
                 key_passes_90, passes_into_final_third_90,
                 passes_into_penalty_area_90, crosses_90
        """
        table = self._find_squad_table(soup, _TABLE_IDS["passing"])
        if table is None:
            logger.warning(f"[{self.source_name}] Passing table not found for {league}")
            return pd.DataFrame()

        raw = self._table_to_df(table, league)
        if raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()
        out["team"] = raw.get("team", raw.get("squad", pd.Series(dtype=str)))

        nineties = pd.to_numeric(raw.get("90s", pd.Series(dtype=float)), errors="coerce")

        cmp_pct  = raw.get("cmp_pct", raw.get("pass_pct", pd.Series(dtype=float)))
        prg      = pd.to_numeric(raw.get("prgp", raw.get("prg", pd.Series(dtype=float))), errors="coerce")
        kp       = pd.to_numeric(raw.get("kp", pd.Series(dtype=float)), errors="coerce")
        final_3  = pd.to_numeric(raw.get("1/3", raw.get("final_third", pd.Series(dtype=float))), errors="coerce")
        ppa      = pd.to_numeric(raw.get("ppa", pd.Series(dtype=float)), errors="coerce")
        crs      = pd.to_numeric(raw.get("crs", pd.Series(dtype=float)), errors="coerce")

        out["pass_completion_pct"]         = pd.to_numeric(cmp_pct, errors="coerce").round(1)
        out["progressive_passes_90"]       = (prg / nineties.replace(0, pd.NA)).round(3)
        out["key_passes_90"]               = (kp / nineties.replace(0, pd.NA)).round(3)
        out["passes_into_final_third_90"]  = (final_3 / nineties.replace(0, pd.NA)).round(3)
        out["passes_into_penalty_area_90"] = (ppa / nineties.replace(0, pd.NA)).round(3)
        out["crosses_90"]                  = (crs / nineties.replace(0, pd.NA)).round(3)

        return out.dropna(subset=["team"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Mode: defense                                                       #
    # ------------------------------------------------------------------ #

    def _parse_defense(
        self, soup: BeautifulSoup, comp_id: int, league: str
    ) -> pd.DataFrame:
        """Parse FBref defensive stats page.

        Returns: team, tackles_90, tackles_won_pct, blocks_90,
                 interceptions_90, clearances_90, errors_leading_to_shot
        """
        table = self._find_squad_table(soup, _TABLE_IDS["defense"])
        if table is None:
            logger.warning(f"[{self.source_name}] Defense table not found for {league}")
            return pd.DataFrame()

        raw = self._table_to_df(table, league)
        if raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()
        out["team"] = raw.get("team", raw.get("squad", pd.Series(dtype=str)))

        nineties = pd.to_numeric(raw.get("90s", pd.Series(dtype=float)), errors="coerce")

        tkl    = pd.to_numeric(raw.get("tkl", pd.Series(dtype=float)), errors="coerce")
        tkl_w  = pd.to_numeric(raw.get("tklw", raw.get("tkl_won", pd.Series(dtype=float))), errors="coerce")
        blocks = pd.to_numeric(raw.get("blocks", pd.Series(dtype=float)), errors="coerce")
        interc = pd.to_numeric(raw.get("int", pd.Series(dtype=float)), errors="coerce")
        clr    = pd.to_numeric(raw.get("clr", pd.Series(dtype=float)), errors="coerce")
        err    = pd.to_numeric(raw.get("err", pd.Series(dtype=float)), errors="coerce")

        out["tackles_90"]              = (tkl / nineties.replace(0, pd.NA)).round(3)
        out["tackles_won_pct"]         = (tkl_w / tkl.replace(0, pd.NA) * 100).round(1)
        out["blocks_90"]               = (blocks / nineties.replace(0, pd.NA)).round(3)
        out["interceptions_90"]        = (interc / nineties.replace(0, pd.NA)).round(3)
        out["clearances_90"]           = (clr / nineties.replace(0, pd.NA)).round(3)
        out["errors_leading_to_shot"]  = err  # Season total — intentional

        return out.dropna(subset=["team"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Mode: possession                                                    #
    # ------------------------------------------------------------------ #

    def _parse_possession(
        self, soup: BeautifulSoup, comp_id: int, league: str
    ) -> pd.DataFrame:
        """Parse FBref possession stats page.

        Returns: team, possession_pct, touches_att_pen_90,
                 progressive_carries_90, progressive_passes_received_90,
                 dribbles_completed_90
        """
        table = self._find_squad_table(soup, _TABLE_IDS["possession"])
        if table is None:
            logger.warning(f"[{self.source_name}] Possession table not found for {league}")
            return pd.DataFrame()

        raw = self._table_to_df(table, league)
        if raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()
        out["team"] = raw.get("team", raw.get("squad", pd.Series(dtype=str)))

        nineties = pd.to_numeric(raw.get("90s", pd.Series(dtype=float)), errors="coerce")

        poss     = pd.to_numeric(raw.get("poss", pd.Series(dtype=float)), errors="coerce")
        touch_ap = pd.to_numeric(raw.get("touches_att_pen", raw.get("att_pen", pd.Series(dtype=float))), errors="coerce")
        prg_c    = pd.to_numeric(raw.get("prgc", raw.get("prg_carries", pd.Series(dtype=float))), errors="coerce")
        prg_r    = pd.to_numeric(raw.get("prgr", raw.get("prg_recv", pd.Series(dtype=float))), errors="coerce")
        drib     = pd.to_numeric(raw.get("succ", raw.get("drib", pd.Series(dtype=float))), errors="coerce")

        out["possession_pct"]                   = poss.round(1)
        out["touches_att_pen_90"]               = (touch_ap / nineties.replace(0, pd.NA)).round(3)
        out["progressive_carries_90"]           = (prg_c / nineties.replace(0, pd.NA)).round(3)
        out["progressive_passes_received_90"]   = (prg_r / nineties.replace(0, pd.NA)).round(3)
        out["dribbles_completed_90"]            = (drib / nineties.replace(0, pd.NA)).round(3)

        return out.dropna(subset=["team"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Mode: pressing                                                      #
    # ------------------------------------------------------------------ #

    def _parse_pressing(
        self, soup: BeautifulSoup, comp_id: int, league: str
    ) -> pd.DataFrame:
        """Parse FBref misc stats for pressing metrics.

        PPDA (passes allowed per defensive action) is approximated as:
            ppda ≈ opponent_passes_per_press_success
        Since FBref doesn't expose opponent pass counts at team level in the
        misc table, we report a pressing intensity index instead:
            pressing_intensity = press_successes / pressures  (0–1 scale)

        Returns: team, pressures_90, pressing_intensity,
                 high_press_successes_90, recoveries_90, ppda_proxy
        """
        table = self._find_squad_table(soup, _TABLE_IDS["misc"])
        if table is None:
            logger.warning(f"[{self.source_name}] Misc table not found for {league}")
            return pd.DataFrame()

        raw = self._table_to_df(table, league)
        if raw.empty:
            return pd.DataFrame()

        out = pd.DataFrame()
        out["team"] = raw.get("team", raw.get("squad", pd.Series(dtype=str)))

        nineties = pd.to_numeric(raw.get("90s", pd.Series(dtype=float)), errors="coerce")

        press   = pd.to_numeric(raw.get("press", raw.get("pressures", pd.Series(dtype=float))), errors="coerce")
        succ    = pd.to_numeric(raw.get("press_succ", raw.get("succ", pd.Series(dtype=float))), errors="coerce")
        recov   = pd.to_numeric(raw.get("recov", pd.Series(dtype=float)), errors="coerce")

        # PPDA proxy: lower pressing intensity → higher PPDA (less pressing)
        # We invert so that high value = high pressing (more intuitive)
        pressing_intensity = (succ / press.replace(0, pd.NA)).round(3)

        # ppda_proxy: pressures per recovery — lower = more efficient pressing
        ppda_proxy = (press / recov.replace(0, pd.NA)).round(3)

        out["pressures_90"]             = (press / nineties.replace(0, pd.NA)).round(3)
        out["high_press_successes_90"]  = (succ / nineties.replace(0, pd.NA)).round(3)
        out["recoveries_90"]            = (recov / nineties.replace(0, pd.NA)).round(3)
        out["pressing_intensity"]       = pressing_intensity   # succ/pressures: 0-1
        out["ppda_proxy"]               = ppda_proxy           # pressures/recovery

        return out.dropna(subset=["team"]).reset_index(drop=True)


# ------------------------------------------------------------------ #
#  Module-level helpers                                               #
# ------------------------------------------------------------------ #

def _uncomment_tables(html: str) -> str:
    """FBref hides some tables inside HTML comments. Uncomment them all."""
    # Remove comment wrappers: <!-- ... -->  around table/div content
    return re.sub(r"<!--(.*?)-->", r"\1", html, flags=re.DOTALL)
