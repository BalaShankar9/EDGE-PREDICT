"""
OddsPortal Collector — historical closing odds scraper.

Scrapes historical closing odds (essential for CLV tracking) from
oddsportal.com using BeautifulSoup static HTML parsing.  OddsPortal
heavily relies on JavaScript, but match rows and some odds data are
embedded in the page as static HTML table rows, data attributes, and
JSON-LD metadata; this collector parses all available static content.
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

ODDSPORTAL_LEAGUES: dict[str, str] = {
    "Premier League": "soccer/england/premier-league",
    "La Liga": "soccer/spain/laliga",
    "Bundesliga": "soccer/germany/bundesliga",
    "Serie A": "soccer/italy/serie-a",
    "Ligue 1": "soccer/france/ligue-1",
}

# OddsPortal paginates results in steps of ~20 matches per page
_MAX_PAGES = 5


class OddsPortalCollector(BaseCollector):
    """Collector for OddsPortal historical closing odds."""

    source_name = "oddsportal"
    base_url = "https://www.oddsportal.com"
    request_delay = 4.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect historical closing odds from OddsPortal.

        Parameters
        ----------
        league : str, optional
            League name key from ODDSPORTAL_LEAGUES (e.g. "Premier League").
            If None, collects all available leagues.
        max_pages : int, optional
            Maximum result pages to paginate through per league (default 5).
        """
        league: Optional[str] = kwargs.get("league")
        max_pages: int = int(kwargs.get("max_pages", _MAX_PAGES))

        leagues = (
            {league: ODDSPORTAL_LEAGUES[league]}
            if league
            else ODDSPORTAL_LEAGUES
        )

        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            try:
                df = self._collect_league(league_name, league_path, max_pages)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.error(
                    f"[oddsportal] Failed to collect {league_name}: {e}"
                )
                # Continue with remaining leagues

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    #  Per-league collection                                               #
    # ------------------------------------------------------------------ #

    def _collect_league(
        self, league_name: str, league_path: str, max_pages: int
    ) -> pd.DataFrame:
        """Scrape closing odds for a single league across multiple pages.

        Parameters
        ----------
        league_name : str
            Human-readable league name.
        league_path : str
            URL path segment for this league.
        max_pages : int
            Maximum pages to paginate through.

        Returns
        -------
        pd.DataFrame
            Combined rows across all pages for this league.
        """
        cache_key = (
            f"oddsportal_{league_path}_"
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[oddsportal] Cache hit: {league_name}")
            return pd.DataFrame(cached)

        rows: list[dict] = []
        page = 1

        while page <= max_pages:
            # OddsPortal results pagination: /results/#/page/N/
            if page == 1:
                url = f"{self.base_url}/{league_path}/results/"
            else:
                url = f"{self.base_url}/{league_path}/results/#/page/{page}/"

            try:
                response = self._fetch(url)
                html = response.text
            except Exception as e:
                logger.warning(
                    f"[oddsportal] Failed to fetch page {page} for {league_name}: {e}"
                )
                break

            # Structural sanity check
            if not self._check_structure(html, ["table-main"]):
                logger.warning(
                    f"[oddsportal] Unexpected structure on page {page} for {league_name}"
                )
                break

            soup = BeautifulSoup(html, "html.parser")

            # Try JSON-LD first — some pages embed structured data
            jsonld_rows = self._parse_jsonld(soup, league_name)
            if jsonld_rows:
                rows.extend(jsonld_rows)
                logger.debug(
                    f"[oddsportal] JSON-LD page {page}: {len(jsonld_rows)} matches for {league_name}"
                )
            else:
                # Fallback: HTML table parsing
                table = soup.select_one("table#tournamentTable, table.table-main")
                if not table:
                    break

                page_rows = self._parse_results_table(table, league_name)
                if not page_rows:
                    break

                rows.extend(page_rows)
                logger.debug(
                    f"[oddsportal] HTML page {page}: {len(page_rows)} matches for {league_name}"
                )

            if not self._has_next_page(soup, page):
                break

            page += 1

        logger.info(
            f"[oddsportal] {league_name}: {len(rows)} matches collected"
        )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["home_team_id"] = df["home_team"].apply(
            lambda x: self.normalise_team(str(x))
        )
        df["away_team_id"] = df["away_team"].apply(
            lambda x: self.normalise_team(str(x))
        )

        self._set_cache(cache_key, df.to_dict(orient="records"))
        return df

    # ------------------------------------------------------------------ #
    #  JSON-LD extraction                                                  #
    # ------------------------------------------------------------------ #

    def _parse_jsonld(self, soup: Any, league_name: str) -> list[dict]:
        """Extract match data from JSON-LD structured data blocks.

        OddsPortal sometimes embeds SportsEvent schema objects that carry
        team names and match times in a machine-readable format.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed page.
        league_name : str
            League name to attach to each row.

        Returns
        -------
        list[dict]
            Parsed match rows, potentially without odds (those come from HTML).
        """
        rows: list[dict] = []
        scraped_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        for script in soup.select("script[type='application/ld+json']"):
            try:
                data = json.loads(script.string or "")
            except (json.JSONDecodeError, TypeError):
                continue

            # Handle both single objects and arrays
            events = data if isinstance(data, list) else [data]

            for event in events:
                if not isinstance(event, dict):
                    continue
                if event.get("@type") not in ("SportsEvent", "Event"):
                    continue

                try:
                    competitors = event.get("competitor", [])
                    if len(competitors) < 2:
                        continue

                    home_team = competitors[0].get("name", "").strip()
                    away_team = competitors[1].get("name", "").strip()

                    if not home_team or not away_team:
                        continue

                    match_date = event.get("startDate", "")
                    score_raw = event.get("result", {})

                    home_goals: Optional[int] = None
                    away_goals: Optional[int] = None
                    if isinstance(score_raw, dict):
                        score_str = score_raw.get("description", "")
                        m = re.match(r"^(\d+)[:\-](\d+)", str(score_str))
                        if m:
                            home_goals = int(m.group(1))
                            away_goals = int(m.group(2))

                    rows.append({
                        "home_team": home_team,
                        "away_team": away_team,
                        "score": _format_score(home_goals, away_goals),
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "match_date": match_date,
                        "odds_home": None,
                        "odds_draw": None,
                        "odds_away": None,
                        "n_bookmakers": None,
                        "league": league_name,
                        "source": self.source_name,
                        "scraped_date": scraped_date,
                    })

                except Exception as e:
                    logger.debug(f"[oddsportal] JSON-LD event parse error: {e}")
                    continue

        return rows

    # ------------------------------------------------------------------ #
    #  HTML table parsing                                                  #
    # ------------------------------------------------------------------ #

    def _parse_results_table(
        self, table: Any, league_name: str
    ) -> list[dict]:
        """Parse match rows from OddsPortal's HTML results table.

        Parameters
        ----------
        table : bs4.element.Tag
            The results ``<table>`` element.
        league_name : str
            Name of the league.

        Returns
        -------
        list[dict]
            Parsed match rows.
        """
        rows: list[dict] = []
        trs = table.select("tr")
        current_date: Optional[str] = None

        for tr in trs:
            # Date separator rows
            date_el = tr.select_one(
                "th.first2, td.table-main__doubleparttitle, "
                "th.table-main__doubleparttitle, td.center.datet"
            )
            if date_el:
                current_date = date_el.get_text(strip=True)
                continue

            row = self._parse_result_row(tr, league_name, current_date)
            if row:
                rows.append(row)

        return rows

    def _parse_result_row(
        self, tr: Any, league_name: str, current_date: Optional[str]
    ) -> Optional[dict]:
        """Parse a single result row from OddsPortal's table.

        OddsPortal encodes much of the odds data in JavaScript, but the
        static HTML retains team names, scores, and sometimes closing odds
        in data attributes (``data-odd``, ``data-odds-*``) on table cells.

        Parameters
        ----------
        tr : bs4.element.Tag
            A ``<tr>`` element from the results table.
        league_name : str
            Name of the league.
        current_date : str or None
            Date string from the nearest preceding date-separator row.

        Returns
        -------
        dict or None
            Parsed result / odds data, or None if parsing fails.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            # Skip header rows
            if tr.select_one("th"):
                return None

            # ---- Team names ------------------------------------------------
            # OddsPortal wraps participant names in <a> tags inside a dedicated cell
            participants_el = tr.select_one(
                "td.table-main__participants, td[class*='participants'], "
                "td.name.table-participant"
            )
            if not participants_el:
                # Fallback: the cell with the most anchor text
                for td in tds:
                    if td.select("a") and len(td.get_text(strip=True)) > 5:
                        participants_el = td
                        break

            if not participants_el:
                return None

            # Anchors may be wrapped in a span; get all <a> children
            anchors = participants_el.select("a")
            home_team: Optional[str] = None
            away_team: Optional[str] = None

            if len(anchors) >= 2:
                home_team = anchors[0].get_text(strip=True)
                away_team = anchors[1].get_text(strip=True)
            else:
                raw = participants_el.get_text(separator=" - ", strip=True)
                parts = re.split(r"\s*[–—\-]{1,2}\s*", raw, maxsplit=1)
                if len(parts) == 2:
                    home_team = parts[0].strip()
                    away_team = parts[1].strip()

            if not home_team or not away_team:
                return None

            # ---- Score -----------------------------------------------------
            home_goals: Optional[int] = None
            away_goals: Optional[int] = None
            score_str: Optional[str] = None

            score_el = tr.select_one(
                "td.table-main__score, td[class*='score'], "
                "a.table-main__score, td.result"
            )
            if not score_el:
                for td in tds:
                    text = td.get_text(strip=True)
                    if re.match(r"^\d+[:\-]\d+$", text):
                        score_el = td
                        break

            if score_el:
                score_text = score_el.get_text(strip=True)
                m = re.match(r"^(\d+)[:\-](\d+)", score_text)
                if m:
                    home_goals = int(m.group(1))
                    away_goals = int(m.group(2))
                    score_str = f"{home_goals}:{away_goals}"

            # ---- Match date ------------------------------------------------
            match_time: Optional[str] = None
            time_el = tr.select_one(
                "td.table-main__time, td[class*='time'], span.table-main__time"
            )
            if time_el:
                match_time = time_el.get_text(strip=True)
            else:
                # Look for HH:MM pattern
                for td in tds:
                    t = td.get_text(strip=True)
                    if re.match(r"^\d{1,2}:\d{2}$", t):
                        match_time = t
                        break

            match_date: Optional[str] = None
            if current_date and match_time:
                match_date = f"{current_date} {match_time}"
            elif current_date:
                match_date = current_date
            elif match_time:
                match_date = match_time

            # ---- Closing odds from data attributes or text -----------------
            odds_home: Optional[float] = None
            odds_draw: Optional[float] = None
            odds_away: Optional[float] = None
            n_bookmakers: Optional[int] = None

            # Strategy 1: data-odd / data-odds attributes on cells
            for td in tds:
                attr_keys = td.attrs.keys() if hasattr(td, "attrs") else []
                for attr in attr_keys:
                    attr_lower = attr.lower()
                    val = td.get(attr, "")
                    parsed = _parse_odd(str(val))
                    if parsed is None:
                        continue
                    if "home" in attr_lower or attr_lower in ("data-odd-1",):
                        odds_home = parsed
                    elif "draw" in attr_lower or "tie" in attr_lower or attr_lower in ("data-odd-x",):
                        odds_draw = parsed
                    elif "away" in attr_lower or attr_lower in ("data-odd-2",):
                        odds_away = parsed

            # Strategy 2: odds text in dedicated odds cells
            if odds_home is None:
                odds_cells = tr.select(
                    "td.odds-nowrp, td[class*='odds'], td.right.odds"
                )
                if len(odds_cells) >= 3:
                    odds_home = _parse_odd(odds_cells[0].get_text(strip=True))
                    odds_draw = _parse_odd(odds_cells[1].get_text(strip=True))
                    odds_away = _parse_odd(odds_cells[2].get_text(strip=True))
                elif len(odds_cells) == 2:
                    odds_home = _parse_odd(odds_cells[0].get_text(strip=True))
                    odds_away = _parse_odd(odds_cells[1].get_text(strip=True))

            # Bookmaker count
            bk_el = tr.select_one(
                "td[class*='bookmaker'], td[data-bookie-count], "
                "td.center.info-value"
            )
            if bk_el:
                raw_bk = bk_el.get("data-bookie-count") or bk_el.get_text(strip=True)
                try:
                    n_bookmakers = int(str(raw_bk).strip())
                except (ValueError, TypeError):
                    pass

            return {
                "home_team": home_team,
                "away_team": away_team,
                "score": score_str,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "match_date": match_date,
                "odds_home": odds_home,
                "odds_draw": odds_draw,
                "odds_away": odds_away,
                "n_bookmakers": n_bookmakers,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[oddsportal] Failed to parse result row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Pagination helper                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _has_next_page(soup: Any, current_page: int) -> bool:
        """Detect whether a next page exists on OddsPortal.

        OddsPortal dynamically loads page data, but the page-navigation
        links are often embedded in the static HTML as ``<a>`` elements
        inside a pagination ``<div>``.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed page HTML.
        current_page : int
            The page number just fetched.

        Returns
        -------
        bool
            True if evidence of a further page is present.
        """
        # Explicit next-page links
        next_link = soup.select_one(
            "a[rel='next'], a.next-page, a[class*='next'], li.next a"
        )
        if next_link and next_link.get("href"):
            return True

        # Numbered pagination links containing a page number > current
        for a in soup.select("div.pagination a, ul.pagination li a, a[class*='pagination']"):
            try:
                n = int(a.get_text(strip=True))
                if n > current_page:
                    return True
            except (ValueError, TypeError):
                continue

        # OddsPortal sometimes stores next-page info as data-page-count
        pagination_el = soup.select_one("[data-page-count], [data-pages]")
        if pagination_el:
            attr = pagination_el.get("data-page-count") or pagination_el.get("data-pages")
            try:
                total = int(str(attr))
                if total > current_page:
                    return True
            except (ValueError, TypeError):
                pass

        return False


# ------------------------------------------------------------------ #
#  Module-level helpers                                               #
# ------------------------------------------------------------------ #

def _parse_odd(text: str) -> Optional[float]:
    """Convert an odds string to a decimal float.

    Handles decimal notation ("2.45"), fractional notation ("5/2"),
    and American moneyline notation ("+150", "-110").

    Parameters
    ----------
    text : str
        Raw odds text from an HTML cell or data attribute.

    Returns
    -------
    float or None
        Decimal odds value, or None if the text is not parseable.
    """
    text = str(text).strip()
    if not text or text in ("-", "–", "—", "?", "N/A", ""):
        return None

    # Decimal odds
    try:
        val = float(text)
        if 1.0 < val < 1000.0:
            return val
        return None
    except ValueError:
        pass

    # Fractional odds e.g. "5/2"
    frac = re.match(r"^(\d+)/(\d+)$", text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den:
            return round(num / den + 1.0, 4)

    # American moneyline e.g. "+150" or "-110"
    ml = re.match(r"^([+\-])(\d+)$", text)
    if ml:
        sign = ml.group(1)
        val = int(ml.group(2))
        if sign == "+":
            return round(val / 100 + 1.0, 4)
        else:
            return round(100 / val + 1.0, 4) if val else None

    return None


def _format_score(
    home_goals: Optional[int], away_goals: Optional[int]
) -> Optional[str]:
    """Format home and away goals as a 'N:M' score string.

    Parameters
    ----------
    home_goals : int or None
    away_goals : int or None

    Returns
    -------
    str or None
    """
    if home_goals is not None and away_goals is not None:
        return f"{home_goals}:{away_goals}"
    return None
