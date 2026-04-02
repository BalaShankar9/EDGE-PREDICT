"""
BetExplorer Collector — live odds and results scraper.

Scrapes upcoming match odds from multiple bookmakers and recent
results from betexplorer.com using BeautifulSoup.

Two collection modes:
  - "odds"    : upcoming matches with average H/D/A odds + bookmaker count
  - "results" : recent results with final scores
"""

import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)

BETEXPLORER_LEAGUES: dict[str, str] = {
    "Premier League": "soccer/england/premier-league",
    "La Liga": "soccer/spain/laliga",
    "Bundesliga": "soccer/germany/bundesliga",
    "Serie A": "soccer/italy/serie-a",
    "Ligue 1": "soccer/france/ligue-1",
    "Eredivisie": "soccer/netherlands/eredivisie",
    "Primeira Liga": "soccer/portugal/primeira-liga",
    "Championship": "soccer/england/championship",
    "Bundesliga 2": "soccer/germany/2-bundesliga",
    "Serie B": "soccer/italy/serie-b",
}

# Max pages to paginate through per league to avoid infinite loops
_MAX_PAGES = 5


class BetExplorerCollector(BaseCollector):
    """Collector for BetExplorer odds and match results."""

    source_name = "betexplorer"
    base_url = "https://www.betexplorer.com"
    request_delay = 3.0

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect odds or results from BetExplorer.

        Parameters
        ----------
        league : str, optional
            League name key from BETEXPLORER_LEAGUES (e.g. "Premier League").
            If None, collects all leagues.
        mode : str, optional
            "odds" (default) to scrape upcoming match odds, or
            "results" to scrape recent match results.
        """
        league: Optional[str] = kwargs.get("league")
        mode: str = kwargs.get("mode", "odds")

        if mode not in ("odds", "results"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'odds' or 'results'.")

        leagues = (
            {league: BETEXPLORER_LEAGUES[league]}
            if league
            else BETEXPLORER_LEAGUES
        )

        frames: list[pd.DataFrame] = []

        for league_name, league_path in leagues.items():
            try:
                if mode == "odds":
                    df = self._collect_odds(league_name, league_path)
                else:
                    df = self._collect_results(league_name, league_path)

                if not df.empty:
                    frames.append(df)

            except Exception as e:
                logger.error(
                    f"[betexplorer] Failed to collect {league_name} ({mode}): {e}"
                )
                # Continue with remaining leagues rather than aborting

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ #
    #  Odds collection                                                     #
    # ------------------------------------------------------------------ #

    def _collect_odds(self, league_name: str, league_path: str) -> pd.DataFrame:
        """Scrape upcoming match odds for a single league with pagination.

        Parameters
        ----------
        league_name : str
            Human-readable league name.
        league_path : str
            URL path segment for the league.

        Returns
        -------
        pd.DataFrame
            Rows with home_team, away_team, match_date, odds_home,
            odds_draw, odds_away, n_bookmakers, league, source, scraped_date.
        """
        cache_key = f"betexplorer_odds_{league_path}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[betexplorer] Cache hit for odds: {league_name}")
            return pd.DataFrame(cached)

        rows: list[dict] = []
        page = 1

        while page <= _MAX_PAGES:
            # BetExplorer uses ?page=N for pagination (page 1 has no param)
            if page == 1:
                url = f"{self.base_url}/{league_path}/"
            else:
                url = f"{self.base_url}/{league_path}/?page={page}"

            try:
                response = self._fetch(url)
                html = response.text
            except Exception as e:
                logger.warning(
                    f"[betexplorer] Failed to fetch odds page {page} for {league_name}: {e}"
                )
                break

            if not self._check_structure(html, ["table-main"]):
                logger.warning(
                    f"[betexplorer] Unexpected structure on odds page {page} for {league_name}"
                )
                break

            soup = BeautifulSoup(html, "html.parser")
            table = soup.select_one("table.table-main")
            if not table:
                break

            page_rows = self._parse_odds_table(table, league_name)
            if not page_rows:
                # Empty page — no more matches
                break

            rows.extend(page_rows)
            logger.debug(
                f"[betexplorer] Odds page {page}: {len(page_rows)} matches for {league_name}"
            )

            # Check for a next-page link; stop if none
            if not self._has_next_page(soup, page):
                break

            page += 1

        logger.info(f"[betexplorer] Odds {league_name}: {len(rows)} matches collected")

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["home_team_id"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)))
        df["away_team_id"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)))

        self._set_cache(cache_key, df.to_dict(orient="records"))
        return df

    def _parse_odds_table(
        self, table: Any, league_name: str
    ) -> list[dict]:
        """Parse all match rows from a BetExplorer odds table.

        Parameters
        ----------
        table : bs4.element.Tag
            The ``table.table-main`` element.
        league_name : str
            Name of the league.

        Returns
        -------
        list[dict]
            List of parsed match dicts (skips rows that fail to parse).
        """
        rows: list[dict] = []
        trs = table.select("tr")
        current_date: Optional[str] = None

        for tr in trs:
            # Date header rows carry a date that applies to following match rows
            date_el = tr.select_one("td.table-main__doubleparttitle, th.table-main__doubleparttitle")
            if date_el:
                current_date = date_el.get_text(strip=True)
                continue

            row = self._parse_odds_row(tr, league_name, current_date)
            if row:
                rows.append(row)

        return rows

    def _parse_odds_row(
        self, tr: Any, league_name: str, current_date: Optional[str]
    ) -> Optional[dict]:
        """Parse a single match odds row.

        Parameters
        ----------
        tr : bs4.element.Tag
            A ``<tr>`` element from the odds table.
        league_name : str
            Name of the league.
        current_date : str or None
            Date string from the nearest preceding date-header row.

        Returns
        -------
        dict or None
            Parsed odds data, or None if the row is not a valid match row.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 4:
                return None

            # The participant cell typically has class table-main__participants
            participants_el = tr.select_one(
                "td.table-main__participants, td[class*='participants']"
            )
            if not participants_el:
                # Fallback: first <td> containing an em-dash or " - "
                for td in tds:
                    text = td.get_text(strip=True)
                    if " - " in text or "\u2013" in text:
                        participants_el = td
                        break

            if not participants_el:
                return None

            teams_text = participants_el.get_text(separator=" - ", strip=True)
            # Split on em-dash, en-dash, or " - "
            parts = re.split(r"\s*[–—\-]{1,2}\s*", teams_text, maxsplit=1)
            if len(parts) < 2:
                return None

            home_team = parts[0].strip()
            away_team = parts[1].strip()

            if not home_team or not away_team:
                return None

            # Match time — span.table-main__time or td containing time pattern
            match_time: Optional[str] = None
            time_el = tr.select_one("span.table-main__time, td.table-main__time")
            if time_el:
                match_time = time_el.get_text(strip=True)
            else:
                # Look for HH:MM pattern in any td
                for td in tds:
                    t = td.get_text(strip=True)
                    if re.match(r"^\d{1,2}:\d{2}$", t):
                        match_time = t
                        break

            # Combine date + time if both available
            match_date: Optional[str] = None
            if current_date and match_time:
                match_date = f"{current_date} {match_time}"
            elif current_date:
                match_date = current_date
            elif match_time:
                match_date = match_time

            # Odds cells: td.table-main__odds (1 / X / 2 columns)
            odds_cells = tr.select("td.table-main__odds, td[class*='odds']")
            odds_home: Optional[float] = None
            odds_draw: Optional[float] = None
            odds_away: Optional[float] = None

            if len(odds_cells) >= 3:
                odds_home = _parse_odd(odds_cells[0].get_text(strip=True))
                odds_draw = _parse_odd(odds_cells[1].get_text(strip=True))
                odds_away = _parse_odd(odds_cells[2].get_text(strip=True))
            elif len(odds_cells) == 2:
                # Some lines only show home/away (no draw)
                odds_home = _parse_odd(odds_cells[0].get_text(strip=True))
                odds_away = _parse_odd(odds_cells[1].get_text(strip=True))

            # Bookmaker count — often in a data attribute or a dedicated cell
            n_bookmakers: Optional[int] = None
            bk_el = tr.select_one(
                "td[data-count], td.table-main__bookmakers, td[class*='bookmaker']"
            )
            if bk_el:
                raw = bk_el.get("data-count") or bk_el.get_text(strip=True)
                try:
                    n_bookmakers = int(str(raw))
                except (ValueError, TypeError):
                    pass

            return {
                "home_team": home_team,
                "away_team": away_team,
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
            logger.debug(f"[betexplorer] Failed to parse odds row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Results collection                                                  #
    # ------------------------------------------------------------------ #

    def _collect_results(self, league_name: str, league_path: str) -> pd.DataFrame:
        """Scrape recent results for a single league with pagination.

        Parameters
        ----------
        league_name : str
            Human-readable league name.
        league_path : str
            URL path segment for the league.

        Returns
        -------
        pd.DataFrame
            Rows with home_team, away_team, home_goals, away_goals,
            match_date, league, source, scraped_date.
        """
        cache_key = f"betexplorer_results_{league_path}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug(f"[betexplorer] Cache hit for results: {league_name}")
            return pd.DataFrame(cached)

        rows: list[dict] = []
        page = 1

        while page <= _MAX_PAGES:
            if page == 1:
                url = f"{self.base_url}/{league_path}/results/"
            else:
                url = f"{self.base_url}/{league_path}/results/?page={page}"

            try:
                response = self._fetch(url)
                html = response.text
            except Exception as e:
                logger.warning(
                    f"[betexplorer] Failed to fetch results page {page} for {league_name}: {e}"
                )
                break

            if not self._check_structure(html, ["table-main"]):
                logger.warning(
                    f"[betexplorer] Unexpected structure on results page {page} for {league_name}"
                )
                break

            soup = BeautifulSoup(html, "html.parser")
            table = soup.select_one("table.table-main")
            if not table:
                break

            page_rows = self._parse_results_table(table, league_name)
            if not page_rows:
                break

            rows.extend(page_rows)
            logger.debug(
                f"[betexplorer] Results page {page}: {len(page_rows)} matches for {league_name}"
            )

            if not self._has_next_page(soup, page):
                break

            page += 1

        logger.info(
            f"[betexplorer] Results {league_name}: {len(rows)} matches collected"
        )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["home_team_id"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)))
        df["away_team_id"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)))

        self._set_cache(cache_key, df.to_dict(orient="records"))
        return df

    def _parse_results_table(
        self, table: Any, league_name: str
    ) -> list[dict]:
        """Parse all result rows from a BetExplorer results table.

        Parameters
        ----------
        table : bs4.element.Tag
            The ``table.table-main`` element from the results page.
        league_name : str
            Name of the league.

        Returns
        -------
        list[dict]
            List of parsed result dicts.
        """
        rows: list[dict] = []
        trs = table.select("tr")
        current_date: Optional[str] = None

        for tr in trs:
            date_el = tr.select_one(
                "td.table-main__doubleparttitle, th.table-main__doubleparttitle"
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
        """Parse a single match result row.

        Parameters
        ----------
        tr : bs4.element.Tag
            A ``<tr>`` element from the results table.
        league_name : str
            Name of the league.
        current_date : str or None
            Date string from the nearest preceding date-header row.

        Returns
        -------
        dict or None
            Parsed result data, or None if the row is not a valid result row.
        """
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            # Teams
            participants_el = tr.select_one(
                "td.table-main__participants, td[class*='participants']"
            )
            if not participants_el:
                for td in tds:
                    text = td.get_text(strip=True)
                    if " - " in text or "\u2013" in text:
                        participants_el = td
                        break

            if not participants_el:
                return None

            teams_text = participants_el.get_text(separator=" - ", strip=True)
            parts = re.split(r"\s*[–—\-]{1,2}\s*", teams_text, maxsplit=1)
            if len(parts) < 2:
                return None

            home_team = parts[0].strip()
            away_team = parts[1].strip()

            if not home_team or not away_team:
                return None

            # Score cell — typically "2:1" or "2-1"
            home_goals: Optional[int] = None
            away_goals: Optional[int] = None

            score_el = tr.select_one(
                "td.table-main__score, td[class*='score'], a[class*='score']"
            )
            if not score_el:
                # Fallback: look for N:M pattern in any td
                for td in tds:
                    text = td.get_text(strip=True)
                    m = re.match(r"^(\d+)[:\-](\d+)$", text)
                    if m:
                        score_el = td
                        break

            if score_el:
                score_text = score_el.get_text(strip=True)
                m = re.match(r"^(\d+)[:\-](\d+)", score_text)
                if m:
                    home_goals = int(m.group(1))
                    away_goals = int(m.group(2))

            # Match time for the date string
            match_time: Optional[str] = None
            time_el = tr.select_one("span.table-main__time, td.table-main__time")
            if time_el:
                match_time = time_el.get_text(strip=True)

            match_date: Optional[str] = None
            if current_date and match_time:
                match_date = f"{current_date} {match_time}"
            elif current_date:
                match_date = current_date
            elif match_time:
                match_date = match_time

            return {
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "match_date": match_date,
                "league": league_name,
                "source": self.source_name,
                "scraped_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.debug(f"[betexplorer] Failed to parse result row: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Pagination helper                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _has_next_page(soup: Any, current_page: int) -> bool:
        """Detect whether a next page of results/odds exists.

        BetExplorer uses ``<a class="pagination__item--next">`` or a
        standard ``<a rel="next">`` link for pagination.

        Parameters
        ----------
        soup : BeautifulSoup
            Parsed page HTML.
        current_page : int
            The page number just fetched (1-indexed).

        Returns
        -------
        bool
            True if a next-page link is present.
        """
        # Look for an explicit "next" pagination anchor
        next_link = soup.select_one(
            "a[rel='next'], a.pagination__item--next, a.next, li.next a"
        )
        if next_link and next_link.get("href"):
            return True

        # Fallback: check pagination list for a page > current
        pagination = soup.select("a.pagination__item, ul.pagination li a")
        for a in pagination:
            try:
                page_num = int(a.get_text(strip=True))
                if page_num > current_page:
                    return True
            except (ValueError, TypeError):
                continue

        return False


# ------------------------------------------------------------------ #
#  Module-level helpers                                               #
# ------------------------------------------------------------------ #

def _parse_odd(text: str) -> Optional[float]:
    """Convert an odds string like '2.45' or '5/2' to a decimal float.

    Parameters
    ----------
    text : str
        Raw text from an odds cell.

    Returns
    -------
    float or None
        Decimal odds, or None if the text is not a recognisable odds value.
    """
    text = text.strip()
    if not text or text in ("-", "–", "—", "?", "N/A"):
        return None
    try:
        return float(text)
    except ValueError:
        pass
    # Fractional odds e.g. "5/2"
    frac = re.match(r"^(\d+)/(\d+)$", text)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den:
            return round(num / den + 1.0, 4)
    return None
