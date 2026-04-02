"""
PredictionAggregator — crowd-wisdom collector that scrapes predictions from
multiple free prediction sites and combines them into a consensus signal.

Sources:
  - forebet.com   — AI probabilities (already have ForebetCollector; this adds detail)
  - predictz.com  — Mathematical predictions
  - windrawwin.com — Expert predictions
  - betstudy.com  — Form-based predictions
  - soccervista.com — Algorithm predictions
  - aiscore.com   — AI predictions

The consensus view ("6 out of 8 sites pick Home") is a strong edge signal
when it aligns with our own model.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pandas as pd
from bs4 import BeautifulSoup

from sharpedge.collectors.base import BaseCollector

logger = logging.getLogger(__name__)


class PredictionAggregator(BaseCollector):
    """Aggregates match predictions from multiple free prediction sites.

    Each row in the returned DataFrame represents one source's prediction
    for one match.  Use ``get_consensus()`` to collapse to a per-match
    consensus view.
    """

    source_name = "prediction_aggregator"
    base_url = ""          # Multiple sources — no single base URL
    request_delay = 3.0

    # ------------------------------------------------------------------ #
    #  Public collection interface                                         #
    # ------------------------------------------------------------------ #

    def _collect(self, **kwargs: Any) -> pd.DataFrame:
        """Collect predictions from all available sources.

        Parameters
        ----------
        date : str, optional
            Target date in YYYY-MM-DD format.  Defaults to today (UTC).
        include_tomorrow : bool, optional
            Whether to also fetch tomorrow's predictions.  Default True.
        """
        today_dt = datetime.now(timezone.utc)
        date_str: str = kwargs.get("date", today_dt.strftime("%Y-%m-%d"))
        include_tomorrow: bool = kwargs.get("include_tomorrow", True)

        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        dates = [target_date]
        if include_tomorrow:
            dates.append(target_date + timedelta(days=1))

        all_frames: list[pd.DataFrame] = []

        # Each scraper is independent — failures don't abort the others
        for dt in dates:
            for scraper_fn in (
                self._scrape_forebet,
                self._scrape_predictz,
                self._scrape_windrawwin,
                self._scrape_betstudy,
                self._scrape_soccervista,
                self._scrape_aiscore,
            ):
                try:
                    df = scraper_fn(dt)
                    if df is not None and not df.empty:
                        all_frames.append(df)
                except Exception as exc:
                    logger.error(
                        f"[{self.source_name}] {scraper_fn.__name__} failed "
                        f"for {dt.strftime('%Y-%m-%d')}: {exc}"
                    )

        if not all_frames:
            return pd.DataFrame()

        result = pd.concat(all_frames, ignore_index=True)
        result = self._build_match_ids(result)
        return result

    # ------------------------------------------------------------------ #
    #  Per-source scrapers                                                 #
    # ------------------------------------------------------------------ #

    def _scrape_forebet(self, dt: datetime) -> Optional[pd.DataFrame]:
        """Scrape today's AI predictions from forebet.com."""
        date_str = dt.strftime("%Y-%m-%d")
        url = f"https://www.forebet.com/en/football-predictions/{date_str}"
        logger.debug(f"[forebet] Fetching {url}")

        response = self._fetch(url)
        html = response.text
        self._check_structure(html, ["rcnt"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        for container in soup.select(".rcnt"):
            row = self._parse_forebet_row(container, date_str)
            if row:
                rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["home_team"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        df["away_team"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        logger.info(f"[forebet] {len(rows)} predictions for {date_str}")
        return df

    def _parse_forebet_row(self, container: Any, date_str: str) -> Optional[dict]:
        try:
            home_team = away_team = None

            tnms = container.select_one(".tnms")
            if tnms:
                spans = tnms.select("span")
                if len(spans) >= 2:
                    home_team = spans[0].get_text(strip=True)
                    away_team = spans[1].get_text(strip=True)

            if not home_team:
                el = container.select_one(".homemark")
                home_team = el.get_text(strip=True) if el else None
            if not away_team:
                el = container.select_one(".awaymark")
                away_team = el.get_text(strip=True) if el else None

            if not home_team or not away_team:
                return None

            prob_home = prob_draw = prob_away = None
            prob_spans = container.select(".fprc span")
            if len(prob_spans) >= 3:
                try:
                    prob_home = float(prob_spans[0].get_text(strip=True))
                    prob_draw = float(prob_spans[1].get_text(strip=True))
                    prob_away = float(prob_spans[2].get_text(strip=True))
                except (ValueError, TypeError):
                    pass

            predicted_result = None
            if prob_home is not None and prob_draw is not None and prob_away is not None:
                max_prob = max(prob_home, prob_draw, prob_away)
                if max_prob == prob_home:
                    predicted_result = "H"
                elif max_prob == prob_draw:
                    predicted_result = "D"
                else:
                    predicted_result = "A"

            # Convert 0-100 percentages to 0-1 scale
            if prob_home is not None:
                prob_home /= 100.0
                prob_draw /= 100.0
                prob_away /= 100.0

            confidence = max(
                (p for p in [prob_home, prob_draw, prob_away] if p is not None),
                default=None,
            )

            league_el = container.select_one(".ld, .league, .lnm")
            league = league_el.get_text(strip=True) if league_el else None

            return {
                "home_team": home_team,
                "away_team": away_team,
                "league": league,
                "match_date": date_str,
                "source": "forebet",
                "predicted_result": predicted_result,
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "confidence": confidence,
            }
        except Exception as exc:
            logger.debug(f"[forebet] parse error: {exc}")
            return None

    # ------------------------------------------------------------------ #

    def _scrape_predictz(self, dt: datetime) -> Optional[pd.DataFrame]:
        """Scrape predictions from predictz.com (daily predictions page)."""
        date_str = dt.strftime("%Y-%m-%d")
        url = f"https://www.predictz.com/predictions/{dt.strftime('%Y/%m/%d')}/"
        logger.debug(f"[predictz] Fetching {url}")

        response = self._fetch(url)
        html = response.text
        self._check_structure(html, ["table"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        for table in soup.select("table"):
            for tr in table.select("tr"):
                row = self._parse_predictz_row(tr, date_str)
                if row:
                    rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["home_team"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        df["away_team"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        logger.info(f"[predictz] {len(rows)} predictions for {date_str}")
        return df

    def _parse_predictz_row(self, tr: Any, date_str: str) -> Optional[dict]:
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            home_team = away_team = predicted_result = None
            confidence = None

            non_numeric_texts = []
            for td in tds:
                link = td.select_one("a")
                text = (link or td).get_text(strip=True)
                if not text:
                    continue
                if text.replace("/", "").replace("-", "").replace(":", "").isdigit():
                    continue
                if text.upper() in ("1", "X", "2"):
                    predicted_result = {"1": "H", "X": "D", "2": "A"}[text.upper()]
                    continue
                if text.endswith("%"):
                    try:
                        confidence = float(text.rstrip("%")) / 100.0
                    except ValueError:
                        pass
                    continue
                if len(text) > 2:
                    non_numeric_texts.append(text)

            if len(non_numeric_texts) >= 2:
                home_team = non_numeric_texts[0]
                away_team = non_numeric_texts[1]

            if not home_team or not away_team:
                return None

            return {
                "home_team": home_team,
                "away_team": away_team,
                "league": None,
                "match_date": date_str,
                "source": "predictz",
                "predicted_result": predicted_result,
                "prob_home": None,
                "prob_draw": None,
                "prob_away": None,
                "confidence": confidence,
            }
        except Exception as exc:
            logger.debug(f"[predictz] parse error: {exc}")
            return None

    # ------------------------------------------------------------------ #

    def _scrape_windrawwin(self, dt: datetime) -> Optional[pd.DataFrame]:
        """Scrape predictions from windrawwin.com (daily predictions page)."""
        date_str = dt.strftime("%Y-%m-%d")
        url = (
            f"https://www.windrawwin.com/predictions/future/"
            f"{dt.strftime('%Y%m%d')}/"
        )
        logger.debug(f"[windrawwin] Fetching {url}")

        response = self._fetch(url)
        html = response.text
        self._check_structure(html, ["table"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        for table in soup.select("table"):
            for tr in table.select("tr"):
                row = self._parse_windrawwin_row(tr, date_str)
                if row:
                    rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["home_team"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        df["away_team"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        logger.info(f"[windrawwin] {len(rows)} predictions for {date_str}")
        return df

    def _parse_windrawwin_row(self, tr: Any, date_str: str) -> Optional[dict]:
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            home_team = away_team = predicted_result = None
            prob_home = prob_draw = prob_away = None

            non_numeric_texts = []
            probs: list[float] = []

            for td in tds:
                link = td.select_one("a")
                text = (link or td).get_text(strip=True)
                if not text:
                    continue
                if text.replace("/", "").replace("-", "").replace(":", "").isdigit():
                    continue

                text_upper = text.upper()
                if text_upper in ("1", "X", "2"):
                    predicted_result = {"1": "H", "X": "D", "2": "A"}[text_upper]
                    continue
                if text_upper in ("W", "D", "L"):
                    predicted_result = {"W": "H", "D": "D", "L": "A"}[text_upper]
                    continue

                if text.endswith("%"):
                    try:
                        probs.append(float(text.rstrip("%")) / 100.0)
                    except ValueError:
                        pass
                    continue

                if len(text) > 2:
                    non_numeric_texts.append(text)

            if len(non_numeric_texts) >= 2:
                home_team = non_numeric_texts[0]
                away_team = non_numeric_texts[1]

            if not home_team or not away_team:
                return None

            if len(probs) >= 3:
                prob_home, prob_draw, prob_away = probs[0], probs[1], probs[2]

            if predicted_result is None and prob_home is not None:
                max_prob = max(prob_home, prob_draw, prob_away)
                if max_prob == prob_home:
                    predicted_result = "H"
                elif max_prob == prob_draw:
                    predicted_result = "D"
                else:
                    predicted_result = "A"

            confidence = (
                max(prob_home, prob_draw, prob_away)
                if prob_home is not None
                else None
            )

            return {
                "home_team": home_team,
                "away_team": away_team,
                "league": None,
                "match_date": date_str,
                "source": "windrawwin",
                "predicted_result": predicted_result,
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "confidence": confidence,
            }
        except Exception as exc:
            logger.debug(f"[windrawwin] parse error: {exc}")
            return None

    # ------------------------------------------------------------------ #

    def _scrape_betstudy(self, dt: datetime) -> Optional[pd.DataFrame]:
        """Scrape form-based predictions from betstudy.com."""
        date_str = dt.strftime("%Y-%m-%d")
        # BetStudy uses YYYY-MM-DD in the path
        url = f"https://www.betstudy.com/predictions/{date_str}/"
        logger.debug(f"[betstudy] Fetching {url}")

        response = self._fetch(url)
        html = response.text
        self._check_structure(html, ["table", "prediction"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        # BetStudy uses a predictions table; rows contain team names and tips
        for table in soup.select("table.predictions, table"):
            for tr in table.select("tr"):
                row = self._parse_betstudy_row(tr, date_str)
                if row:
                    rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["home_team"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        df["away_team"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        logger.info(f"[betstudy] {len(rows)} predictions for {date_str}")
        return df

    def _parse_betstudy_row(self, tr: Any, date_str: str) -> Optional[dict]:
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            home_team = away_team = predicted_result = None
            confidence = None
            non_numeric_texts = []

            for td in tds:
                link = td.select_one("a")
                text = (link or td).get_text(strip=True)
                if not text:
                    continue
                if text.replace("/", "").replace("-", "").replace(":", "").isdigit():
                    continue

                text_upper = text.upper()
                # BetStudy prediction columns: 1 / X / 2 or Home / Draw / Away
                if text_upper in ("1", "X", "2"):
                    predicted_result = {"1": "H", "X": "D", "2": "A"}[text_upper]
                    continue
                if text_upper in ("HOME", "DRAW", "AWAY"):
                    predicted_result = {"HOME": "H", "DRAW": "D", "AWAY": "A"}[text_upper]
                    continue

                if text.endswith("%"):
                    try:
                        confidence = float(text.rstrip("%")) / 100.0
                    except ValueError:
                        pass
                    continue

                if len(text) > 2:
                    non_numeric_texts.append(text)

            if len(non_numeric_texts) >= 2:
                home_team = non_numeric_texts[0]
                away_team = non_numeric_texts[1]

            if not home_team or not away_team:
                return None

            return {
                "home_team": home_team,
                "away_team": away_team,
                "league": None,
                "match_date": date_str,
                "source": "betstudy",
                "predicted_result": predicted_result,
                "prob_home": None,
                "prob_draw": None,
                "prob_away": None,
                "confidence": confidence,
            }
        except Exception as exc:
            logger.debug(f"[betstudy] parse error: {exc}")
            return None

    # ------------------------------------------------------------------ #

    def _scrape_soccervista(self, dt: datetime) -> Optional[pd.DataFrame]:
        """Scrape algorithm predictions from soccervista.com."""
        date_str = dt.strftime("%Y-%m-%d")
        url = f"https://www.soccervista.com/soccer-predictions-for-{date_str}"
        logger.debug(f"[soccervista] Fetching {url}")

        response = self._fetch(url)
        html = response.text
        self._check_structure(html, ["table"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        for table in soup.select("table"):
            for tr in table.select("tr"):
                row = self._parse_soccervista_row(tr, date_str)
                if row:
                    rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["home_team"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        df["away_team"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        logger.info(f"[soccervista] {len(rows)} predictions for {date_str}")
        return df

    def _parse_soccervista_row(self, tr: Any, date_str: str) -> Optional[dict]:
        try:
            tds = tr.select("td")
            if len(tds) < 3:
                return None

            home_team = away_team = predicted_result = None
            odds_home = odds_draw = odds_away = None
            non_numeric_texts = []
            numeric_vals: list[float] = []

            for td in tds:
                link = td.select_one("a")
                text = (link or td).get_text(strip=True)
                if not text:
                    continue
                if text.replace("/", "").replace("-", "").replace(":", "").isdigit():
                    continue

                text_upper = text.upper()
                if text_upper in ("1", "X", "2"):
                    predicted_result = {"1": "H", "X": "D", "2": "A"}[text_upper]
                    continue
                if text_upper in ("HOME", "DRAW", "AWAY", "1X2"):
                    if text_upper != "1X2":
                        predicted_result = {"HOME": "H", "DRAW": "D", "AWAY": "A"}[text_upper]
                    continue

                # SoccerVista shows decimal odds like 2.10, 3.40, 3.80
                try:
                    val = float(text.replace(",", "."))
                    if 1.0 < val < 50.0:
                        numeric_vals.append(val)
                    continue
                except ValueError:
                    pass

                if len(text) > 2:
                    non_numeric_texts.append(text)

            if len(non_numeric_texts) >= 2:
                home_team = non_numeric_texts[0]
                away_team = non_numeric_texts[1]

            if not home_team or not away_team:
                return None

            # Decimal odds → implied probabilities (de-vigged naively)
            if len(numeric_vals) >= 3:
                odds_home, odds_draw, odds_away = numeric_vals[0], numeric_vals[1], numeric_vals[2]
                raw_h = 1.0 / odds_home
                raw_d = 1.0 / odds_draw
                raw_a = 1.0 / odds_away
                total = raw_h + raw_d + raw_a
                prob_home = round(raw_h / total, 4) if total > 0 else None
                prob_draw = round(raw_d / total, 4) if total > 0 else None
                prob_away = round(raw_a / total, 4) if total > 0 else None

                if predicted_result is None and prob_home is not None:
                    max_p = max(prob_home, prob_draw, prob_away)
                    predicted_result = (
                        "H" if max_p == prob_home
                        else "D" if max_p == prob_draw
                        else "A"
                    )
                confidence = max(prob_home, prob_draw, prob_away) if prob_home is not None else None
            else:
                prob_home = prob_draw = prob_away = confidence = None

            return {
                "home_team": home_team,
                "away_team": away_team,
                "league": None,
                "match_date": date_str,
                "source": "soccervista",
                "predicted_result": predicted_result,
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "confidence": confidence,
            }
        except Exception as exc:
            logger.debug(f"[soccervista] parse error: {exc}")
            return None

    # ------------------------------------------------------------------ #

    def _scrape_aiscore(self, dt: datetime) -> Optional[pd.DataFrame]:
        """Scrape AI predictions from aiscore.com (JSON-backed page)."""
        date_str = dt.strftime("%Y-%m-%d")
        # AiScore renders predictions in JSON endpoints embedded in the page;
        # fall back to parsing the HTML prediction list.
        url = f"https://www.aiscore.com/football/predictions/{date_str}"
        logger.debug(f"[aiscore] Fetching {url}")

        response = self._fetch(
            url,
            headers={"Accept": "text/html,application/xhtml+xml,*/*"},
        )
        html = response.text
        self._check_structure(html, ["prediction"])

        soup = BeautifulSoup(html, "html.parser")
        rows: list[dict] = []

        # AiScore prediction cards / list items
        for card in soup.select(
            ".prediction-item, .match-item, [class*='prediction'], [class*='match-row']"
        ):
            row = self._parse_aiscore_card(card, date_str)
            if row:
                rows.append(row)

        # Fallback: try a JSON API endpoint that AiScore sometimes exposes
        if not rows:
            rows = self._fetch_aiscore_json(dt)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["home_team"] = df["home_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        df["away_team"] = df["away_team"].apply(lambda x: self.normalise_team(str(x)) or x)
        logger.info(f"[aiscore] {len(rows)} predictions for {date_str}")
        return df

    def _parse_aiscore_card(self, card: Any, date_str: str) -> Optional[dict]:
        try:
            texts = [
                el.get_text(strip=True)
                for el in card.select("[class*='team'], [class*='name'], span, a")
                if el.get_text(strip=True)
            ]
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique_texts: list[str] = []
            for t in texts:
                if t not in seen:
                    seen.add(t)
                    unique_texts.append(t)

            team_texts = [t for t in unique_texts if len(t) > 2 and not t.replace(".", "").isdigit()]
            if len(team_texts) < 2:
                return None

            home_team, away_team = team_texts[0], team_texts[1]

            # Look for probability values (often rendered as percentage divs)
            prob_texts = [
                el.get_text(strip=True)
                for el in card.select("[class*='prob'], [class*='percent'], [class*='pct']")
            ]
            probs: list[float] = []
            for pt in prob_texts:
                try:
                    probs.append(float(pt.replace("%", "").strip()) / 100.0)
                except ValueError:
                    pass

            prob_home = prob_draw = prob_away = None
            if len(probs) >= 3:
                prob_home, prob_draw, prob_away = probs[0], probs[1], probs[2]

            # Look for explicit prediction badge (1/X/2 or W/D/L)
            predicted_result = None
            for el in card.select("[class*='pick'], [class*='tip'], [class*='result']"):
                t = el.get_text(strip=True).upper()
                if t in ("1", "X", "2"):
                    predicted_result = {"1": "H", "X": "D", "2": "A"}[t]
                    break
                if t in ("HOME", "DRAW", "AWAY"):
                    predicted_result = {"HOME": "H", "DRAW": "D", "AWAY": "A"}[t]
                    break

            if predicted_result is None and prob_home is not None:
                max_p = max(prob_home, prob_draw, prob_away)
                predicted_result = (
                    "H" if max_p == prob_home
                    else "D" if max_p == prob_draw
                    else "A"
                )

            confidence = (
                max(prob_home, prob_draw, prob_away)
                if prob_home is not None
                else None
            )

            return {
                "home_team": home_team,
                "away_team": away_team,
                "league": None,
                "match_date": date_str,
                "source": "aiscore",
                "predicted_result": predicted_result,
                "prob_home": prob_home,
                "prob_draw": prob_draw,
                "prob_away": prob_away,
                "confidence": confidence,
            }
        except Exception as exc:
            logger.debug(f"[aiscore] card parse error: {exc}")
            return None

    def _fetch_aiscore_json(self, dt: datetime) -> list[dict]:
        """Attempt AiScore's undocumented API endpoint as a fallback."""
        date_str = dt.strftime("%Y-%m-%d")
        api_url = (
            f"https://www.aiscore.com/api/match/list"
            f"?sport=football&date={date_str}&type=prediction"
        )
        try:
            data = self._fetch_json(
                api_url,
                headers={"Accept": "application/json", "Referer": "https://www.aiscore.com/"},
            )
        except Exception as exc:
            logger.debug(f"[aiscore] JSON API unavailable: {exc}")
            return []

        rows: list[dict] = []
        # Structure varies — try common shapes
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("data", "matches", "predictions", "results"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break

        for item in items:
            try:
                home = item.get("home_team") or item.get("homeTeam") or item.get("home", {}).get("name")
                away = item.get("away_team") or item.get("awayTeam") or item.get("away", {}).get("name")
                if not home or not away:
                    continue

                ph = item.get("prob_home") or item.get("probHome") or item.get("homeWin")
                pd_ = item.get("prob_draw") or item.get("probDraw") or item.get("draw")
                pa = item.get("prob_away") or item.get("probAway") or item.get("awayWin")

                # Normalise to 0-1 if provided as percentages
                if ph is not None and ph > 1:
                    ph, pd_, pa = ph / 100.0, pd_ / 100.0, pa / 100.0

                predicted_result = item.get("prediction") or item.get("tip")
                if predicted_result in ("1", 1):
                    predicted_result = "H"
                elif predicted_result in ("X", "x", "draw"):
                    predicted_result = "D"
                elif predicted_result in ("2", 2):
                    predicted_result = "A"
                elif predicted_result is None and ph is not None:
                    max_p = max(ph, pd_, pa)
                    predicted_result = "H" if max_p == ph else "D" if max_p == pd_ else "A"

                rows.append({
                    "home_team": str(home),
                    "away_team": str(away),
                    "league": item.get("league") or item.get("tournament"),
                    "match_date": date_str,
                    "source": "aiscore",
                    "predicted_result": predicted_result,
                    "prob_home": ph,
                    "prob_draw": pd_,
                    "prob_away": pa,
                    "confidence": max(ph, pd_, pa) if ph is not None else None,
                })
            except Exception as exc:
                logger.debug(f"[aiscore] JSON item parse error: {exc}")

        return rows

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_match_ids(df: pd.DataFrame) -> pd.DataFrame:
        """Add a canonical match_id column (home_v_away, lower-cased, spaces→_)."""
        def make_id(row: pd.Series) -> str:
            home = str(row.get("home_team", "")).lower().replace(" ", "_")
            away = str(row.get("away_team", "")).lower().replace(" ", "_")
            return f"{home}_v_{away}"

        df = df.copy()
        df["match_id"] = df.apply(make_id, axis=1)
        # Reorder columns for readability
        front_cols = ["match_id", "home_team", "away_team", "league", "match_date",
                      "source", "predicted_result", "prob_home", "prob_draw",
                      "prob_away", "confidence"]
        remaining = [c for c in df.columns if c not in front_cols]
        return df[front_cols + remaining]

    # ------------------------------------------------------------------ #
    #  Consensus aggregation                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_consensus(df: pd.DataFrame) -> pd.DataFrame:
        """Collapse per-source rows to a per-match consensus view.

        Parameters
        ----------
        df : pd.DataFrame
            Output of ``collect()`` — one row per source per match.

        Returns
        -------
        pd.DataFrame
            One row per match with columns:
              match_id, home_team, away_team, league, match_date,
              n_sources,
              pct_home, pct_draw, pct_away,
              consensus_pick, consensus_confidence,
              avg_prob_home, avg_prob_draw, avg_prob_away
        """
        if df.empty:
            return pd.DataFrame()

        required = {"match_id", "predicted_result"}
        if not required.issubset(df.columns):
            logger.warning("get_consensus: missing required columns")
            return pd.DataFrame()

        records: list[dict] = []

        group_keys = ["match_id"]
        for key in ["home_team", "away_team", "league", "match_date"]:
            if key in df.columns:
                group_keys.append(key)

        for group_vals, group in df.groupby(group_keys, dropna=False):
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            meta = dict(zip(group_keys, group_vals))

            valid = group.dropna(subset=["predicted_result"])
            n = len(valid)
            if n == 0:
                continue

            counts = valid["predicted_result"].value_counts()
            n_home = int(counts.get("H", 0))
            n_draw = int(counts.get("D", 0))
            n_away = int(counts.get("A", 0))

            pct_home = round(n_home / n, 4)
            pct_draw = round(n_draw / n, 4)
            pct_away = round(n_away / n, 4)

            # Consensus pick = whatever the majority says
            max_pct = max(pct_home, pct_draw, pct_away)
            if max_pct == pct_home:
                consensus_pick = "H"
            elif max_pct == pct_draw:
                consensus_pick = "D"
            else:
                consensus_pick = "A"

            # Confidence = % agreement on the consensus pick (0-1)
            consensus_confidence = round(max_pct, 4)

            # Average model probabilities where available
            avg_ph = avg_pd = avg_pa = None
            prob_rows = group.dropna(subset=["prob_home", "prob_draw", "prob_away"])
            if not prob_rows.empty:
                avg_ph = round(float(prob_rows["prob_home"].mean()), 4)
                avg_pd = round(float(prob_rows["prob_draw"].mean()), 4)
                avg_pa = round(float(prob_rows["prob_away"].mean()), 4)

            records.append({
                **meta,
                "n_sources": n,
                "n_home_picks": n_home,
                "n_draw_picks": n_draw,
                "n_away_picks": n_away,
                "pct_home": pct_home,
                "pct_draw": pct_draw,
                "pct_away": pct_away,
                "consensus_pick": consensus_pick,
                "consensus_confidence": consensus_confidence,
                "avg_prob_home": avg_ph,
                "avg_prob_draw": avg_pd,
                "avg_prob_away": avg_pa,
            })

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).sort_values(
            "consensus_confidence", ascending=False
        ).reset_index(drop=True)
