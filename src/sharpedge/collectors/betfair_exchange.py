"""
BetfairExchangeCollector — Betfair exchange odds and trading volume.

Betfair's exchange is the sharpest market in the world.  When smart money
moves, it shows up here first — in tightening spreads, rising matched
volume, and steam moves on back/lay prices.

Primary scraping strategy
--------------------------
Betfair's exchange pages are heavily JavaScript-rendered, so we pursue
three escalating strategies:

  1. Betfair static HTML data attributes and embedded JSON blobs
     https://www.betfair.com/exchange/plus/football
  2. OddsChecker football markets page (shows Betfair exchange column
     alongside bookmaker odds — mostly server-rendered)
     https://www.oddschecker.com/football
  3. Fallback: Betfair's exchange betting API public price endpoint
     https://www.betfair.com/exchange/plus/football (alternate routes)

_collect() row schema
---------------------
  match_id              str or None
  home_team, away_team  str
  match_date            str
  back_home             float   best back price (home win)
  back_draw             float
  back_away             float
  lay_home              float   best lay price (home win)
  lay_draw              float
  lay_away              float
  spread_home           float   back_home - lay_home
  spread_draw           float
  spread_away           float
  volume_matched        float   total GBP matched on this market
  volume_home           float   % of volume on home
  volume_draw           float
  volume_away           float
  implied_prob_home     float   1 / back_home
  implied_prob_draw     float
  implied_prob_away     float
  last_traded_home      float
  last_traded_draw      float
  last_traded_away      float
  market_sentiment      str     signal from get_market_sentiment()
  source                str
  scraped_at            str
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

# ------------------------------------------------------------------ #
#  URL constants                                                      #
# ------------------------------------------------------------------ #

_BETFAIR_FOOTBALL_URL = "https://www.betfair.com/exchange/plus/football"
_ODDSCHECKER_URL = "https://www.oddschecker.com/football"

# OddsChecker league paths that surface Betfair exchange odds
_ODDSCHECKER_LEAGUES: dict[str, str] = {
    "Premier League": "https://www.oddschecker.com/football/english/premier-league",
    "Championship":   "https://www.oddschecker.com/football/english/championship",
    "La Liga":        "https://www.oddschecker.com/football/spanish/la-liga",
    "Bundesliga":     "https://www.oddschecker.com/football/german/bundesliga",
    "Serie A":        "https://www.oddschecker.com/football/italian/serie-a",
    "Ligue 1":        "https://www.oddschecker.com/football/french/ligue-1",
    "Champions League": "https://www.oddschecker.com/football/champions-league",
    "Europa League":  "https://www.oddschecker.com/football/europa-league",
}

# OddsChecker identifies bookmaker columns by data attribute
_BETFAIR_BOOKIE_ATTR_VALUES = frozenset(["BF", "BETFAIR", "BFX", "betfair"])

# Thresholds for get_market_sentiment()
_VOLUME_SKEW_THRESHOLD = 0.60   # >60 % on one side = smart money signal
_TIGHT_SPREAD_THRESHOLD = 0.02  # spread < 2 % of mid-price = high liquidity

# Pattern to find embedded JSON data blobs in script tags
_JSON_STATE_RE = re.compile(
    r"(?:window\.__PRELOADED_STATE__|window\.__data__|__NEXT_DATA__)\s*=\s*(\{.*?\});",
    re.DOTALL,
)


class BetfairExchangeCollector(BaseCollector):
    """Collector for Betfair exchange back/lay odds and matched volume.

    Scrapes Betfair's public exchange pages and OddsChecker's Betfair
    column to capture the world's sharpest odds market.
    """

    source_name = "betfair_exchange"
    base_url = "https://www.betfair.com"
    request_delay = 3.0

    # ------------------------------------------------------------------ #
    #  Public signal helper                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_market_sentiment(odds_data: dict) -> str:
        """Derive a smart-money signal from exchange market data.

        Rules (applied in order):
          1. If matched volume on a single outcome exceeds 60 % of total
             volume — 'steam_move' (directional smart money).
          2. If any back/lay spread is tighter than 2 % of the mid-price
             — 'high_confidence' (liquid, well-formed market).
          3. Otherwise — 'neutral'.

        Parameters
        ----------
        odds_data : dict
            A single collected row dict with back_*, lay_*, volume_* keys.

        Returns
        -------
        str
            One of: "steam_move_home", "steam_move_draw", "steam_move_away",
            "high_confidence", "neutral".
        """
        # Steam move detection
        for side in ("home", "draw", "away"):
            vol_pct = odds_data.get(f"volume_{side}")
            if vol_pct is not None and vol_pct > _VOLUME_SKEW_THRESHOLD:
                return f"steam_move_{side}"

        # Tight spread detection — any outcome with very tight spread
        for side in ("home", "draw", "away"):
            back = odds_data.get(f"back_{side}")
            lay = odds_data.get(f"lay_{side}")
            if back and lay and back > 0:
                mid = (back + lay) / 2
                spread_pct = (lay - back) / mid if mid > 0 else None
                if spread_pct is not None and spread_pct < _TIGHT_SPREAD_THRESHOLD:
                    return "high_confidence"

        return "neutral"

    # ------------------------------------------------------------------ #
    #  Main collection entry-point                                        #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect Betfair exchange odds and volume data.

        Parameters
        ----------
        league : str, optional
            League name key from _ODDSCHECKER_LEAGUES (e.g. "Premier League").
            If None, collects all configured leagues.
        source : str, optional
            "betfair" — try Betfair.com directly (may be JS-rendered).
            "oddschecker" — use OddsChecker as proxy (default).
            "all" — try both, merge results.

        Returns
        -------
        pd.DataFrame
            One row per match with back/lay/spread/volume columns.
        """
        league: Optional[str] = kwargs.get("league")
        strategy: str = kwargs.get("source", "oddschecker")

        leagues = (
            {league: _ODDSCHECKER_LEAGUES[league]}
            if league and league in _ODDSCHECKER_LEAGUES
            else _ODDSCHECKER_LEAGUES
        )

        frames: list[pd.DataFrame] = []

        if strategy in ("betfair", "all"):
            try:
                df_bf = self._collect_betfair_direct()
                if not df_bf.empty:
                    frames.append(df_bf)
            except Exception as exc:
                logger.error(f"[{self.source_name}] Betfair direct failed: {exc}")

        if strategy in ("oddschecker", "all"):
            for league_name, league_url in leagues.items():
                try:
                    df_oc = self._collect_oddschecker_league(league_name, league_url)
                    if not df_oc.empty:
                        frames.append(df_oc)
                except Exception as exc:
                    logger.error(
                        f"[{self.source_name}] OddsChecker {league_name} failed: {exc}"
                    )

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # Derive spread columns
        for side in ("home", "draw", "away"):
            back_col = f"back_{side}"
            lay_col = f"lay_{side}"
            spread_col = f"spread_{side}"
            if back_col in df.columns and lay_col in df.columns:
                df[spread_col] = df.apply(
                    lambda r, b=back_col, l=lay_col: (
                        round(r[l] - r[b], 4)
                        if pd.notna(r[b]) and pd.notna(r[l])
                        else None
                    ),
                    axis=1,
                )

        # Implied probability from back price
        for side in ("home", "draw", "away"):
            back_col = f"back_{side}"
            prob_col = f"implied_prob_{side}"
            if back_col in df.columns:
                df[prob_col] = df[back_col].apply(
                    lambda p: round(1.0 / p, 4) if p and p > 0 else None
                )

        # Market sentiment signal
        df["market_sentiment"] = df.apply(
            lambda r: self.get_market_sentiment(r.to_dict()), axis=1
        )

        # Normalise team names
        df["home_team_id"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        df["away_team_id"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x))
        )

        return df

    # ------------------------------------------------------------------ #
    #  Betfair.com direct scraping                                        #
    # ------------------------------------------------------------------ #

    def _collect_betfair_direct(self) -> pd.DataFrame:
        """Attempt to scrape Betfair's exchange football landing page.

        Betfair renders most odds via JavaScript, but the initial HTML
        sometimes contains:
          - Embedded JSON state blobs in <script> tags
          - data-* attributes on market containers

        Returns
        -------
        pd.DataFrame
            Parsed rows, or empty DataFrame if JS-wall encountered.
        """
        cache_key = f"betfair_direct_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[{self.source_name}] Betfair direct cache hit")
            return pd.DataFrame(cached)

        response = self._fetch(
            _BETFAIR_FOOTBALL_URL,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://www.betfair.com/",
            },
        )
        html = response.text

        rows = self._parse_betfair_json_blobs(html)

        if not rows:
            # Fallback: try data-* attribute parsing
            rows = self._parse_betfair_data_attrs(html)

        if not rows:
            logger.warning(
                f"[{self.source_name}] Betfair direct: no parseable data "
                f"(likely JS-rendered — use OddsChecker strategy)"
            )
            return pd.DataFrame()

        logger.info(f"[{self.source_name}] Betfair direct: {len(rows)} matches")
        self._set_cache(cache_key, rows)
        return pd.DataFrame(rows)

    def _parse_betfair_json_blobs(self, html: str) -> list[dict]:
        """Extract and parse any embedded JSON state from Betfair HTML.

        Betfair sometimes pre-renders market data as a window.__data__ or
        __NEXT_DATA__ blob that contains event/market/runner prices.

        Parameters
        ----------
        html : str
            Raw HTML from the Betfair exchange football page.

        Returns
        -------
        list[dict]
            Parsed match rows, empty list if no usable JSON found.
        """
        rows: list[dict] = []
        scraped_at = datetime.now(timezone.utc).isoformat()

        for script_content in re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL):
            # Look for JSON blobs containing market/runner data
            m = _JSON_STATE_RE.search(script_content)
            if not m:
                continue
            try:
                blob = json.loads(m.group(1))
                events = self._extract_events_from_blob(blob)
                for event in events:
                    row = self._build_row_from_event(event, scraped_at, source="betfair")
                    if row:
                        rows.append(row)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.debug(f"[{self.source_name}] JSON blob parse error: {exc}")
                continue

        return rows

    def _extract_events_from_blob(self, blob: Any) -> list[dict]:
        """Recursively search a JSON blob for football event/market objects.

        Parameters
        ----------
        blob : Any
            Arbitrary nested JSON value.

        Returns
        -------
        list[dict]
            Found event-like dicts containing runner price data.
        """
        results: list[dict] = []
        if isinstance(blob, dict):
            # Heuristic: a market object has "runners" or "selections" and
            # the event name usually contains " v " for football
            if ("runners" in blob or "selections" in blob) and (
                "v" in str(blob.get("event", ""))
                or "v" in str(blob.get("eventName", ""))
                or "v" in str(blob.get("name", ""))
            ):
                results.append(blob)
            else:
                for value in blob.values():
                    results.extend(self._extract_events_from_blob(value))
        elif isinstance(blob, list):
            for item in blob:
                results.extend(self._extract_events_from_blob(item))
        return results

    def _build_row_from_event(
        self, event: dict, scraped_at: str, source: str
    ) -> Optional[dict]:
        """Convert a Betfair market JSON object into a normalised row.

        Parameters
        ----------
        event : dict
            Event/market dict extracted from a JSON blob.
        scraped_at : str
            ISO-8601 UTC timestamp.
        source : str
            Source label for the row.

        Returns
        -------
        dict or None
            Normalised row or None if parsing fails.
        """
        try:
            event_name = (
                event.get("eventName")
                or event.get("name")
                or event.get("event", "")
            )
            # Expect "TeamA v TeamB" or "TeamA vs TeamB"
            sep = re.compile(r"\s+(?:v\.?s?\.?|versus)\s+", re.IGNORECASE)
            teams = sep.split(str(event_name), maxsplit=1)
            if len(teams) < 2:
                return None
            home_team = teams[0].strip()
            away_team = teams[1].strip()

            runners = event.get("runners") or event.get("selections") or []
            if len(runners) < 3:
                return None

            # Betfair 1X2 runner order: Home / Draw / Away
            def _runner_prices(r: dict) -> tuple[Optional[float], Optional[float], Optional[float]]:
                """Extract best back, lay, and last-traded from a runner."""
                back_price: Optional[float] = None
                lay_price: Optional[float] = None
                last_traded: Optional[float] = None

                # Available price formats vary by Betfair endpoint
                ex = r.get("ex", {})
                available_to_back = ex.get("availableToBack", [])
                available_to_lay = ex.get("availableToLay", [])

                if available_to_back:
                    back_price = float(available_to_back[0].get("price", 0) or 0) or None
                if available_to_lay:
                    lay_price = float(available_to_lay[0].get("price", 0) or 0) or None

                # Alternative flat fields
                if back_price is None:
                    for key in ("bestAvailableToBackPrice", "backPrice", "price"):
                        if key in r:
                            try:
                                back_price = float(r[key])
                            except (ValueError, TypeError):
                                pass
                            break

                if lay_price is None:
                    for key in ("bestAvailableToLayPrice", "layPrice"):
                        if key in r:
                            try:
                                lay_price = float(r[key])
                            except (ValueError, TypeError):
                                pass
                            break

                for key in ("lastPriceTraded", "lastTradedPrice"):
                    if key in r:
                        try:
                            last_traded = float(r[key])
                        except (ValueError, TypeError):
                            pass
                        break

                return back_price, lay_price, last_traded

            back_h, lay_h, last_h = _runner_prices(runners[0])
            back_d, lay_d, last_d = _runner_prices(runners[1])
            back_a, lay_a, last_a = _runner_prices(runners[2])

            # Total matched volume
            volume_matched: Optional[float] = None
            for key in ("totalMatched", "matchedAmount", "volumeMatched"):
                if key in event:
                    try:
                        volume_matched = float(event[key])
                    except (ValueError, TypeError):
                        pass
                    break

            return {
                "match_id": event.get("id") or event.get("marketId"),
                "home_team": home_team,
                "away_team": away_team,
                "match_date": event.get("openDate") or event.get("startTime"),
                "back_home": back_h,
                "back_draw": back_d,
                "back_away": back_a,
                "lay_home": lay_h,
                "lay_draw": lay_d,
                "lay_away": lay_a,
                "volume_matched": volume_matched,
                "volume_home": None,
                "volume_draw": None,
                "volume_away": None,
                "last_traded_home": last_h,
                "last_traded_draw": last_d,
                "last_traded_away": last_a,
                "source": source,
                "scraped_at": scraped_at,
            }

        except Exception as exc:
            logger.debug(f"[{self.source_name}] Event parse error: {exc}")
            return None

    def _parse_betfair_data_attrs(self, html: str) -> list[dict]:
        """Parse Betfair HTML for data-* attributes containing odds.

        Betfair sometimes embeds price data in element data attributes
        (e.g. data-selection-id, data-back-price) for progressive
        enhancement.

        Parameters
        ----------
        html : str
            Raw HTML from the Betfair exchange page.

        Returns
        -------
        list[dict]
            Parsed rows.
        """
        soup = BeautifulSoup(html, "html.parser")
        scraped_at = datetime.now(timezone.utc).isoformat()
        rows: list[dict] = []

        # Market containers — Betfair uses various class names
        market_els = soup.select(
            "[class*='market-catalogue'], [class*='event-information'], "
            "[data-market-id], [class*='coupon-line']"
        )

        for market_el in market_els:
            row = self._parse_market_element(market_el, scraped_at)
            if row:
                rows.append(row)

        return rows

    def _parse_market_element(
        self, market_el: Any, scraped_at: str
    ) -> Optional[dict]:
        """Parse a single Betfair market HTML element.

        Parameters
        ----------
        market_el : bs4.element.Tag
            A market container element.
        scraped_at : str
            ISO-8601 UTC timestamp.

        Returns
        -------
        dict or None
            Parsed row or None.
        """
        try:
            # Event name
            event_el = market_el.select_one(
                "[class*='event-name'], [class*='match-title'], "
                "[class*='market-title'], h3, h4"
            )
            if not event_el:
                return None
            event_name = event_el.get_text(strip=True)

            sep = re.compile(r"\s+(?:v\.?s?\.?|versus|@)\s+", re.IGNORECASE)
            teams = sep.split(event_name, maxsplit=1)
            if len(teams) < 2:
                return None

            home_team = teams[0].strip()
            away_team = teams[1].strip()
            if not home_team or not away_team:
                return None

            # Price buttons — look for back/lay price cells
            price_els = market_el.select(
                "[class*='bet-button'], [class*='price'], "
                "[data-back-price], [data-lay-price]"
            )

            prices: list[Optional[float]] = []
            for el in price_els[:6]:
                raw = (
                    el.get("data-back-price")
                    or el.get("data-lay-price")
                    or el.get_text(strip=True)
                )
                prices.append(_parse_price(str(raw)))

            # Expect at least 3 back prices (home/draw/away)
            def _safe(lst: list, idx: int) -> Optional[float]:
                return lst[idx] if idx < len(lst) else None

            return {
                "match_id": market_el.get("data-market-id"),
                "home_team": home_team,
                "away_team": away_team,
                "match_date": None,
                "back_home": _safe(prices, 0),
                "back_draw": _safe(prices, 1),
                "back_away": _safe(prices, 2),
                "lay_home": _safe(prices, 3),
                "lay_draw": _safe(prices, 4),
                "lay_away": _safe(prices, 5),
                "volume_matched": None,
                "volume_home": None,
                "volume_draw": None,
                "volume_away": None,
                "last_traded_home": None,
                "last_traded_draw": None,
                "last_traded_away": None,
                "source": "betfair",
                "scraped_at": scraped_at,
            }

        except Exception as exc:
            logger.debug(f"[{self.source_name}] Market element parse error: {exc}")
            return None

    # ------------------------------------------------------------------ #
    #  OddsChecker proxy                                                  #
    # ------------------------------------------------------------------ #

    def _collect_oddschecker_league(
        self, league_name: str, league_url: str
    ) -> pd.DataFrame:
        """Scrape Betfair exchange column from OddsChecker's league page.

        OddsChecker is server-side rendered and shows Betfair exchange
        prices in a dedicated column alongside traditional bookmakers.

        Parameters
        ----------
        league_name : str
            Human-readable league name.
        league_url : str
            Full URL to the OddsChecker league match-odds page.

        Returns
        -------
        pd.DataFrame
            Rows with Betfair back prices (lay not available from OddsChecker).
        """
        cache_key = (
            f"oddschecker_{league_name.replace(' ', '_')}_"
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H')}"
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[{self.source_name}] OddsChecker cache hit: {league_name}")
            return pd.DataFrame(cached)

        response = self._fetch(
            league_url,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://www.oddschecker.com/",
            },
        )
        html = response.text

        rows = self._parse_oddschecker_page(html, league_name)

        if not rows:
            logger.warning(f"[{self.source_name}] OddsChecker {league_name}: 0 rows")
            return pd.DataFrame()

        logger.info(f"[{self.source_name}] OddsChecker {league_name}: {len(rows)} matches")
        self._set_cache(cache_key, rows)
        return pd.DataFrame(rows)

    def _parse_oddschecker_page(
        self, html: str, league_name: str
    ) -> list[dict]:
        """Parse an OddsChecker match-odds league page.

        OddsChecker renders a table where:
          - Each row is one match (tr.diff-row or tr[data-bk-link])
          - Columns correspond to bookmakers (th/td with data-bk attributes)
          - We identify the Betfair exchange column by its data-bk value

        Parameters
        ----------
        html : str
            Raw HTML from OddsChecker.
        league_name : str
            League name for the rows.

        Returns
        -------
        list[dict]
            Parsed rows with Betfair back prices.
        """
        soup = BeautifulSoup(html, "html.parser")
        scraped_at = datetime.now(timezone.utc).isoformat()
        rows: list[dict] = []

        # Find the odds table
        table = soup.select_one(
            "table.eventTable, table[class*='odds-table'], "
            "table[class*='bet-table'], #oddsTableContainer table"
        )
        if not table:
            logger.debug(f"[{self.source_name}] OddsChecker: no odds table found")
            return rows

        # Locate the Betfair exchange column index from the header row
        betfair_col_idx = self._find_betfair_column(table)

        # Parse each match row
        match_rows = table.select("tr.diff-row, tr[data-bk-link], tr[class*='match']")
        if not match_rows:
            match_rows = table.select("tbody tr")

        for tr in match_rows:
            row = self._parse_oddschecker_row(
                tr, betfair_col_idx, league_name, scraped_at
            )
            if row:
                rows.append(row)

        return rows

    def _find_betfair_column(self, table: Any) -> Optional[int]:
        """Identify the Betfair exchange column index in an OddsChecker table.

        Parameters
        ----------
        table : bs4.element.Tag
            The odds table element.

        Returns
        -------
        int or None
            Zero-indexed column position of the Betfair column, or None.
        """
        header_row = table.select_one("thead tr, tr.eventTableHeaderRow")
        if not header_row:
            return None

        for idx, th in enumerate(header_row.select("th, td")):
            bk_val = (
                th.get("data-bk", "")
                or th.get("data-bookie", "")
                or th.get_text(strip=True)
            )
            if str(bk_val).upper() in {v.upper() for v in _BETFAIR_BOOKIE_ATTR_VALUES}:
                return idx
            # Also check for a child image alt text
            img = th.select_one("img[alt]")
            if img and "betfair" in str(img.get("alt", "")).lower():
                return idx

        logger.debug(f"[{self.source_name}] OddsChecker: Betfair column not found in header")
        return None

    def _parse_oddschecker_row(
        self,
        tr: Any,
        betfair_col_idx: Optional[int],
        league_name: str,
        scraped_at: str,
    ) -> Optional[dict]:
        """Parse a single OddsChecker match row.

        Parameters
        ----------
        tr : bs4.element.Tag
            A table row representing one match.
        betfair_col_idx : int or None
            Column index of the Betfair exchange column.
        league_name : str
            League name for this row.
        scraped_at : str
            ISO-8601 UTC timestamp.

        Returns
        -------
        dict or None
            Parsed row with Betfair back prices, or None.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            # Match name — typically in first or second cell
            match_name = ""
            for td in tds[:3]:
                text = td.get_text(strip=True)
                if " v " in text or " vs " in text.lower():
                    match_name = text
                    break

            # Fallback: data attribute or href anchor
            if not match_name:
                a_el = tr.select_one("a[href]")
                if a_el:
                    match_name = a_el.get_text(strip=True)
                    if not match_name:
                        # Try to parse from href slug "team-a-v-team-b"
                        href = a_el.get("href", "")
                        slug = href.split("/")[-1]
                        match_name = slug.replace("-", " ").title()

            if not match_name:
                return None

            sep = re.compile(r"\s+(?:v\.?s?\.?|versus)\s+", re.IGNORECASE)
            teams = sep.split(match_name, maxsplit=1)
            if len(teams) < 2:
                return None

            home_team = teams[0].strip()
            away_team = teams[1].strip()

            # Betfair prices — locate by column index or data-bk attribute
            back_home: Optional[float] = None
            back_draw: Optional[float] = None
            back_away: Optional[float] = None

            # OddsChecker structures prices differently per market:
            # Some pages have one row per match with 3 outcomes inline;
            # others have separate rows for home/draw/away.
            # Strategy: collect all prices with Betfair bk markers.

            bf_tds = tr.select(
                f"[data-bk='BF'], [data-bk='BETFAIR'], "
                f"[data-bk='BFX'], [data-bookie='BF']"
            )

            if bf_tds:
                prices = [_parse_price(td.get_text(strip=True)) for td in bf_tds[:3]]
                back_home = prices[0] if len(prices) > 0 else None
                back_draw = prices[1] if len(prices) > 1 else None
                back_away = prices[2] if len(prices) > 2 else None
            elif betfair_col_idx is not None and betfair_col_idx < len(tds):
                # Use the identified column offset; OddsChecker repeats 1X2
                # in groups of 3 columns per bookmaker on some layouts
                bf_td = tds[betfair_col_idx]
                back_home = _parse_price(bf_td.get_text(strip=True))
                # Attempt adjacent cells for draw/away if available
                if betfair_col_idx + 2 < len(tds):
                    back_draw = _parse_price(tds[betfair_col_idx + 1].get_text(strip=True))
                    back_away = _parse_price(tds[betfair_col_idx + 2].get_text(strip=True))

            # Match date/time
            match_date: Optional[str] = None
            date_el = tr.select_one(
                "[class*='date'], [class*='time'], time, [data-time]"
            )
            if date_el:
                match_date = (
                    date_el.get("datetime")
                    or date_el.get("data-time")
                    or date_el.get_text(strip=True)
                )

            return {
                "match_id": tr.get("data-event-id") or tr.get("id"),
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "back_home": back_home,
                "back_draw": back_draw,
                "back_away": back_away,
                "lay_home": None,
                "lay_draw": None,
                "lay_away": None,
                "volume_matched": None,
                "volume_home": None,
                "volume_draw": None,
                "volume_away": None,
                "last_traded_home": None,
                "last_traded_draw": None,
                "last_traded_away": None,
                "league": league_name,
                "source": "oddschecker_betfair",
                "scraped_at": scraped_at,
            }

        except Exception as exc:
            logger.debug(f"[{self.source_name}] OddsChecker row parse error: {exc}")
            return None


# ------------------------------------------------------------------ #
#  Module-level helpers                                               #
# ------------------------------------------------------------------ #

def _parse_price(text: str) -> Optional[float]:
    """Convert a price string to a decimal float.

    Handles decimal ("2.50"), fractional ("5/2"), and percentage formats.

    Parameters
    ----------
    text : str
        Raw text from a price cell.

    Returns
    -------
    float or None
        Decimal price, or None if not parseable.
    """
    text = text.strip().replace(",", "")
    if not text or text in ("-", "–", "—", "?", "N/A", "SP", "EVS", ""):
        return None

    # Decimal
    try:
        val = float(text)
        return val if val > 1.0 else None
    except ValueError:
        pass

    # Fractional odds e.g. "5/2"
    frac = re.match(r"^(\d+)/(\d+)$", text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den:
            return round(num / den + 1.0, 4)

    # Evens shorthand
    if text.upper() in ("EVS", "EVENS"):
        return 2.0

    return None
