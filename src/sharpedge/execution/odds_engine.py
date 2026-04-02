"""Live Odds Engine — fetch, compare, steam-detect, CLV.

Wraps The Odds API (v4) and persists snapshots into MatchOdds.
Designed to run every 15 minutes during trading hours.

Free-tier budget: 500 requests/month (~16/day).  We fetch once per
15-minute cycle across all 5 top soccer leagues, so a single cycle
uses 5 requests — well within budget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from thefuzz import fuzz

from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.models import Match, MatchOdds

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Map our internal sport slug → one or more Odds API sport keys
_SPORT_KEY_MAP: dict[str, list[str]] = {
    "soccer": [
        "soccer_epl",
        "soccer_germany_bundesliga",
        "soccer_italy_serie_a",
        "soccer_spain_la_liga",
        "soccer_france_ligue_one",
    ],
}

# Bookmakers roughly ordered by sharpness (Pinnacle first)
_SHARP_BOOKS = {"pinnacle", "pinnaclesports"}

# Team-name fuzzy-match threshold
_TEAM_MATCH_THRESHOLD = 75

# In-process cache: keyed by (sport_key, regions, markets) → (payload, fetched_at)
_CACHE: dict[tuple[str, str, str], tuple[list[dict], datetime]] = {}
_CACHE_TTL_SECONDS = 600  # 10 minutes — don't re-fetch within a cycle


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BestOdds:
    """Best available price per outcome across all surveyed bookmakers."""

    home_odds: Optional[float] = None
    home_book: Optional[str] = None
    draw_odds: Optional[float] = None
    draw_book: Optional[str] = None
    away_odds: Optional[float] = None
    away_book: Optional[str] = None
    over_odds: Optional[float] = None
    over_book: Optional[str] = None
    under_odds: Optional[float] = None
    under_book: Optional[str] = None
    overround: Optional[float] = None  # computed from best odds (should be ~1.0 for efficient market)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class BookmakerComparison:
    """Per-bookmaker efficiency metrics."""

    bookmaker: str
    margin_pct: float  # overround expressed as % above 100  (lower = sharper)
    n_markets: int
    avg_deviation_from_best: float  # average % below best available price


@dataclass
class SteamMove:
    """Detected sharp-money line movement."""

    match_id: Optional[int]  # DB match id if resolved
    outcome: str  # e.g. "home", "draw", "away", "over", "under"
    direction: str  # "shortening" (price falling = probability rising) or "drifting"
    magnitude_pct: float  # average % move across bookmakers
    detected_at: datetime
    bookmakers_moving: list[str]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _implied_prob(odds: float) -> float:
    """Decimal odds → implied probability (safe)."""
    if odds <= 1.0:
        return 1.0
    return 1.0 / odds


def _overround_from_best(best: BestOdds) -> Optional[float]:
    """Sum of implied probs from best prices across h2h outcomes."""
    probs = []
    for odds in (best.home_odds, best.draw_odds, best.away_odds):
        if odds is not None and odds > 1.0:
            probs.append(_implied_prob(odds))
    if len(probs) < 2:
        return None
    return round(sum(probs), 4)


def _fuzzy_team_match(name: str, candidates: list[str]) -> Optional[str]:
    """Return the best-matching candidate above threshold, or None."""
    best_score = 0
    best_match = None
    for candidate in candidates:
        score = fuzz.ratio(name.lower(), candidate.lower())
        if score > best_score:
            best_score = score
            best_match = candidate
    if best_score >= _TEAM_MATCH_THRESHOLD:
        return best_match
    return None


def _snapshot_hash(home_odds: Optional[float], draw_odds: Optional[float],
                   away_odds: Optional[float], over_odds: Optional[float],
                   under_odds: Optional[float]) -> str:
    """MD5 of the odds values for dedup."""
    payload = json.dumps(
        [home_odds, draw_odds, away_odds, over_odds, under_odds],
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class LiveOddsEngine:
    """Fetch, compare, steam-detect and persist live odds.

    Usage
    -----
    engine = LiveOddsEngine()
    summary = engine.run_odds_cycle()
    """

    def __init__(self, request_timeout: int = 20) -> None:
        self._timeout = request_timeout
        # Remaining API request count reported by the Odds API headers
        self._requests_remaining: Optional[int] = None
        self._requests_used: Optional[int] = None

    # ------------------------------------------------------------------
    # 1. Fetch
    # ------------------------------------------------------------------

    def fetch_odds(
        self,
        sport: str = "soccer",
        regions: str = "uk,eu",
        markets: str = "h2h,totals",
    ) -> list[dict]:
        """Fetch live odds from The Odds API.

        Returns a list of standardised match dicts:
            {
                match_id: str,          # API event_id
                home_team: str,
                away_team: str,
                commence_time: str,     # ISO-8601 UTC
                sport_key: str,
                bookmakers: [
                    {
                        name: str,
                        markets: [
                            {
                                key: str,  # "h2h" or "totals"
                                outcomes: [{"name": str, "price": float, "point": float|None}]
                            }
                        ]
                    }
                ]
            }

        Results are cached for _CACHE_TTL_SECONDS to conserve API quota.
        """
        if not settings.odds_api_key:
            logger.warning("odds_api_key not configured — returning empty odds list")
            return []

        sport_keys = _SPORT_KEY_MAP.get(sport, [sport])
        all_matches: list[dict] = []

        for sport_key in sport_keys:
            cache_key = (sport_key, regions, markets)
            cached_payload, cached_at = _CACHE.get(cache_key, (None, None))

            if cached_payload is not None and cached_at is not None:
                age = (datetime.now(tz=timezone.utc) - cached_at).total_seconds()
                if age < _CACHE_TTL_SECONDS:
                    logger.debug(
                        "Cache hit for %s (age %.0fs)", sport_key, age
                    )
                    all_matches.extend(cached_payload)
                    continue

            url = f"{_ODDS_API_BASE}/sports/{sport_key}/odds/"
            params = {
                "apiKey": settings.odds_api_key,
                "regions": regions,
                "markets": markets,
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            }

            try:
                logger.info("Fetching odds: %s regions=%s markets=%s", sport_key, regions, markets)
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.get(url, params=params)

                # Surface rate-limit info
                self._requests_remaining = int(
                    resp.headers.get("x-requests-remaining", -1)
                )
                self._requests_used = int(
                    resp.headers.get("x-requests-used", -1)
                )
                if self._requests_remaining != -1:
                    logger.info(
                        "Odds API quota: %d used, %d remaining",
                        self._requests_used,
                        self._requests_remaining,
                    )
                    if self._requests_remaining < 20:
                        logger.warning(
                            "Odds API quota nearly exhausted: %d requests remaining",
                            self._requests_remaining,
                        )

                if resp.status_code == 401:
                    logger.error("Odds API: invalid API key")
                    return all_matches
                if resp.status_code == 422:
                    logger.warning("Odds API: sport key %s not found, skipping", sport_key)
                    continue
                resp.raise_for_status()

                raw_events: list[dict] = resp.json()
                parsed = self._parse_events(raw_events, sport_key)
                _CACHE[cache_key] = (parsed, datetime.now(tz=timezone.utc))
                all_matches.extend(parsed)

            except httpx.TimeoutException:
                logger.error("Timeout fetching odds for %s", sport_key)
            except httpx.HTTPStatusError as exc:
                logger.error("HTTP %s fetching odds for %s: %s",
                             exc.response.status_code, sport_key, exc)
            except Exception:
                logger.exception("Unexpected error fetching odds for %s", sport_key)

        logger.info("Fetched %d matches total", len(all_matches))
        return all_matches

    def _parse_events(self, raw_events: list[dict], sport_key: str) -> list[dict]:
        """Normalise the raw Odds API response into our internal format."""
        results: list[dict] = []
        for event in raw_events:
            bookmakers_parsed = []
            for bm in event.get("bookmakers", []):
                markets_parsed = []
                for mkt in bm.get("markets", []):
                    outcomes = []
                    for outcome in mkt.get("outcomes", []):
                        outcomes.append(
                            {
                                "name": outcome.get("name", ""),
                                "price": float(outcome.get("price", 0.0)),
                                "point": outcome.get("point"),  # for totals line
                            }
                        )
                    markets_parsed.append(
                        {"key": mkt.get("key", ""), "outcomes": outcomes}
                    )
                bookmakers_parsed.append(
                    {"name": bm.get("key", bm.get("title", "unknown")), "markets": markets_parsed}
                )

            results.append(
                {
                    "match_id": event.get("id", ""),
                    "home_team": event.get("home_team", ""),
                    "away_team": event.get("away_team", ""),
                    "commence_time": event.get("commence_time", ""),
                    "sport_key": sport_key,
                    "bookmakers": bookmakers_parsed,
                }
            )
        return results

    # ------------------------------------------------------------------
    # 2. Best odds
    # ------------------------------------------------------------------

    def find_best_odds(
        self,
        match_home: str,
        match_away: str,
        market: str = "h2h",
        odds_data: Optional[list[dict]] = None,
    ) -> BestOdds:
        """Find the best available price per outcome for a given match.

        Uses fuzzy team-name matching to handle variations between the API
        and internal names (e.g. "Arsenal" vs "Arsenal FC").

        Parameters
        ----------
        match_home : str
            Home team name as it appears in your DB / prediction pipeline.
        match_away : str
            Away team name.
        market : str
            "h2h" (1X2) or "totals" (over/under).
        odds_data : list[dict] | None
            Pre-fetched odds; if None the engine will call fetch_odds().
        """
        if odds_data is None:
            odds_data = self.fetch_odds()

        best = BestOdds()

        # Locate the matching event
        event = self._resolve_event(match_home, match_away, odds_data)
        if event is None:
            logger.warning(
                "No odds found for %s vs %s in %d events",
                match_home, match_away, len(odds_data)
            )
            return best

        for bm in event.get("bookmakers", []):
            bm_name = bm["name"]
            for mkt in bm.get("markets", []):
                if mkt["key"] != market:
                    continue
                for outcome in mkt["outcomes"]:
                    name = outcome["name"].lower()
                    price = outcome["price"]
                    if price <= 1.0:
                        continue

                    if market == "h2h":
                        if name in ("home", event["home_team"].lower()):
                            if best.home_odds is None or price > best.home_odds:
                                best.home_odds = price
                                best.home_book = bm_name
                        elif name == "draw":
                            if best.draw_odds is None or price > best.draw_odds:
                                best.draw_odds = price
                                best.draw_book = bm_name
                        elif name in ("away", event["away_team"].lower()):
                            if best.away_odds is None or price > best.away_odds:
                                best.away_odds = price
                                best.away_book = bm_name

                    elif market == "totals":
                        if name.startswith("over"):
                            if best.over_odds is None or price > best.over_odds:
                                best.over_odds = price
                                best.over_book = bm_name
                        elif name.startswith("under"):
                            if best.under_odds is None or price > best.under_odds:
                                best.under_odds = price
                                best.under_book = bm_name

        best.overround = _overround_from_best(best)
        return best

    def _resolve_event(
        self, home: str, away: str, odds_data: list[dict]
    ) -> Optional[dict]:
        """Fuzzy-match team names to find the right event."""
        home_names = [e["home_team"] for e in odds_data]
        away_names = [e["away_team"] for e in odds_data]

        best_score = 0
        best_event = None

        for event in odds_data:
            h_score = fuzz.ratio(home.lower(), event["home_team"].lower())
            a_score = fuzz.ratio(away.lower(), event["away_team"].lower())
            combined = (h_score + a_score) / 2
            if combined > best_score:
                best_score = combined
                best_event = event

        if best_score >= _TEAM_MATCH_THRESHOLD:
            return best_event
        return None

    # ------------------------------------------------------------------
    # 3. Compare bookmakers
    # ------------------------------------------------------------------

    def compare_bookmakers(self, odds_data: list[dict]) -> list[BookmakerComparison]:
        """Rank bookmakers by margin (lower = sharper).

        For each h2h market the margin is: sum(1/p_i) - 1 (overround).
        We average across all markets per bookmaker.

        Returns list sorted ascending by margin_pct (Pinnacle will typically
        appear first or near the top).
        """
        # accumulate per-book: [overround values], [deviations from best price]
        book_overrounds: dict[str, list[float]] = {}
        book_deviations: dict[str, list[float]] = {}
        book_market_count: dict[str, int] = {}

        for event in odds_data:
            # Build best price per outcome across all books for this event
            best_by_outcome: dict[str, float] = {}
            for bm in event.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "h2h":
                        continue
                    for oc in mkt["outcomes"]:
                        name = oc["name"]
                        price = oc["price"]
                        if price > best_by_outcome.get(name, 0.0):
                            best_by_outcome[name] = price

            # Now per-bookmaker stats
            for bm in event.get("bookmakers", []):
                bm_name = bm["name"]
                for mkt in bm.get("markets", []):
                    if mkt["key"] != "h2h":
                        continue
                    outcomes = mkt["outcomes"]
                    if len(outcomes) < 2:
                        continue

                    overround = sum(
                        _implied_prob(oc["price"])
                        for oc in outcomes
                        if oc["price"] > 1.0
                    )

                    book_overrounds.setdefault(bm_name, []).append(overround)
                    book_market_count[bm_name] = book_market_count.get(bm_name, 0) + 1

                    # Deviation from best price
                    for oc in outcomes:
                        best_p = best_by_outcome.get(oc["name"])
                        if best_p and oc["price"] > 1.0 and best_p > 1.0:
                            dev = (best_p - oc["price"]) / best_p
                            book_deviations.setdefault(bm_name, []).append(dev)

        comparisons: list[BookmakerComparison] = []
        for bm_name, overrounds in book_overrounds.items():
            avg_margin = (sum(overrounds) / len(overrounds)) - 1.0
            deviations = book_deviations.get(bm_name, [0.0])
            avg_dev = sum(deviations) / len(deviations)
            comparisons.append(
                BookmakerComparison(
                    bookmaker=bm_name,
                    margin_pct=round(avg_margin * 100, 3),
                    n_markets=book_market_count.get(bm_name, 0),
                    avg_deviation_from_best=round(avg_dev * 100, 3),
                )
            )

        comparisons.sort(key=lambda x: x.margin_pct)
        return comparisons

    # ------------------------------------------------------------------
    # 4. Steam move detection
    # ------------------------------------------------------------------

    def detect_steam_moves(
        self,
        match_home: str,
        match_away: str,
        threshold_pct: float = 3.0,
        odds_data: Optional[list[dict]] = None,
    ) -> list[SteamMove]:
        """Detect steam moves by comparing live prices to the last DB snapshot.

        A steam move fires when >= 2 bookmakers move the same direction by
        more than `threshold_pct` percent on the same outcome.
        """
        if odds_data is None:
            odds_data = self.fetch_odds()

        moves: list[SteamMove] = []
        session = get_session()
        try:
            # Resolve DB match
            db_match = self._resolve_db_match(session, match_home, match_away)
            match_id = db_match.id if db_match else None

            # Load last stored odds snapshot for this match
            if match_id is None:
                logger.debug(
                    "No DB match found for %s vs %s — steam detection skipped",
                    match_home, match_away,
                )
                return []

            stored_rows: list[MatchOdds] = (
                session.query(MatchOdds)
                .filter(MatchOdds.match_id == match_id)
                .order_by(MatchOdds.captured_at.desc())
                .all()
            )
            if not stored_rows:
                logger.debug("No stored odds for match %d — nothing to compare", match_id)
                return []

            # Build previous snapshot: {bookmaker: {outcome: price}}
            prev: dict[str, dict[str, float]] = {}
            for row in stored_rows:
                bm = row.bookmaker
                prev.setdefault(bm, {})
                if row.odds_home is not None:
                    prev[bm]["home"] = row.odds_home
                if row.odds_draw is not None:
                    prev[bm]["draw"] = row.odds_draw
                if row.odds_away is not None:
                    prev[bm]["away"] = row.odds_away
                if row.odds_over is not None:
                    prev[bm]["over"] = row.odds_over
                if row.odds_under is not None:
                    prev[bm]["under"] = row.odds_under

            # Build current snapshot from live fetch
            event = self._resolve_event(match_home, match_away, odds_data)
            if event is None:
                return []

            curr: dict[str, dict[str, float]] = {}
            for bm in event.get("bookmakers", []):
                bm_name = bm["name"]
                curr.setdefault(bm_name, {})
                for mkt in bm.get("markets", []):
                    for oc in mkt["outcomes"]:
                        name = oc["name"].lower()
                        price = oc["price"]
                        if mkt["key"] == "h2h":
                            if name in ("home", event["home_team"].lower()):
                                curr[bm_name]["home"] = price
                            elif name == "draw":
                                curr[bm_name]["draw"] = price
                            elif name in ("away", event["away_team"].lower()):
                                curr[bm_name]["away"] = price
                        elif mkt["key"] == "totals":
                            if name.startswith("over"):
                                curr[bm_name]["over"] = price
                            elif name.startswith("under"):
                                curr[bm_name]["under"] = price

            # Compare
            outcomes = ["home", "draw", "away", "over", "under"]
            for outcome in outcomes:
                shortening_books: list[str] = []
                drifting_books: list[str] = []
                magnitudes: list[float] = []

                for bm_name in set(curr) & set(prev):
                    old_price = prev[bm_name].get(outcome)
                    new_price = curr[bm_name].get(outcome)
                    if old_price is None or new_price is None:
                        continue
                    if old_price <= 1.0 or new_price <= 1.0:
                        continue

                    change_pct = ((new_price - old_price) / old_price) * 100.0
                    if abs(change_pct) >= threshold_pct:
                        magnitudes.append(abs(change_pct))
                        if change_pct < 0:  # price fell = shortening (steam on this side)
                            shortening_books.append(bm_name)
                        else:
                            drifting_books.append(bm_name)

                if len(shortening_books) >= 2:
                    moves.append(
                        SteamMove(
                            match_id=match_id,
                            outcome=outcome,
                            direction="shortening",
                            magnitude_pct=round(
                                sum(magnitudes[: len(shortening_books)]) / len(shortening_books), 2
                            ),
                            detected_at=datetime.now(tz=timezone.utc),
                            bookmakers_moving=shortening_books,
                        )
                    )
                if len(drifting_books) >= 2:
                    moves.append(
                        SteamMove(
                            match_id=match_id,
                            outcome=outcome,
                            direction="drifting",
                            magnitude_pct=round(
                                sum(magnitudes[-len(drifting_books) :]) / len(drifting_books), 2
                            ),
                            detected_at=datetime.now(tz=timezone.utc),
                            bookmakers_moving=drifting_books,
                        )
                    )

            if moves:
                logger.info(
                    "Steam detected for %s vs %s: %d move(s)",
                    match_home, match_away, len(moves)
                )
        finally:
            session.close()

        return moves

    # ------------------------------------------------------------------
    # 5. Store snapshot
    # ------------------------------------------------------------------

    def store_snapshot(self, odds_data: list[dict]) -> int:
        """Persist current odds to MatchOdds table.

        Skips records that are identical to the most recent stored snapshot
        (dedup by MD5 of odds values per bookmaker × market).

        Returns the number of new rows inserted.
        """
        session = get_session()
        stored = 0
        now = datetime.now(tz=timezone.utc)

        try:
            for event in odds_data:
                home = event["home_team"]
                away = event["away_team"]
                db_match = self._resolve_db_match(session, home, away)
                if db_match is None:
                    logger.debug(
                        "No DB match for %s vs %s — snapshot skipped", home, away
                    )
                    continue

                for bm in event.get("bookmakers", []):
                    bm_name = bm["name"]
                    # Aggregate all markets for this bm into a single row per market-type
                    market_buckets: dict[str, dict[str, Optional[float]]] = {}

                    for mkt in bm.get("markets", []):
                        key = mkt["key"]
                        bucket = market_buckets.setdefault(
                            key,
                            {
                                "odds_home": None,
                                "odds_draw": None,
                                "odds_away": None,
                                "odds_over": None,
                                "odds_under": None,
                            },
                        )
                        for oc in mkt["outcomes"]:
                            name = oc["name"].lower()
                            price = oc["price"]
                            if key == "h2h":
                                if name in ("home", event["home_team"].lower()):
                                    bucket["odds_home"] = price
                                elif name == "draw":
                                    bucket["odds_draw"] = price
                                elif name in ("away", event["away_team"].lower()):
                                    bucket["odds_away"] = price
                            elif key == "totals":
                                if name.startswith("over"):
                                    bucket["odds_over"] = price
                                elif name.startswith("under"):
                                    bucket["odds_under"] = price

                    for mkt_key, bucket in market_buckets.items():
                        new_hash = _snapshot_hash(
                            bucket["odds_home"],
                            bucket["odds_draw"],
                            bucket["odds_away"],
                            bucket["odds_over"],
                            bucket["odds_under"],
                        )

                        # Check last stored row for this match/bm/market
                        last_row: Optional[MatchOdds] = (
                            session.query(MatchOdds)
                            .filter(
                                MatchOdds.match_id == db_match.id,
                                MatchOdds.bookmaker == bm_name,
                                MatchOdds.market == mkt_key,
                                MatchOdds.odds_type == "live",
                            )
                            .order_by(MatchOdds.captured_at.desc())
                            .first()
                        )

                        if last_row is not None:
                            last_hash = _snapshot_hash(
                                last_row.odds_home,
                                last_row.odds_draw,
                                last_row.odds_away,
                                last_row.odds_over,
                                last_row.odds_under,
                            )
                            if last_hash == new_hash:
                                continue  # identical — skip

                        row = MatchOdds(
                            match_id=db_match.id,
                            bookmaker=bm_name,
                            market=mkt_key,
                            odds_type="live",
                            odds_home=bucket["odds_home"],
                            odds_draw=bucket["odds_draw"],
                            odds_away=bucket["odds_away"],
                            odds_over=bucket["odds_over"],
                            odds_under=bucket["odds_under"],
                            captured_at=now,
                        )
                        session.add(row)
                        stored += 1

            session.commit()
            logger.info("Stored %d odds snapshot row(s)", stored)
        except Exception:
            session.rollback()
            logger.exception("Error storing odds snapshot")
        finally:
            session.close()

        return stored

    # ------------------------------------------------------------------
    # 6. Closing odds
    # ------------------------------------------------------------------

    def get_closing_odds(self, match_home: str, match_away: str) -> dict:
        """Return the last odds captured before the match kicked off.

        This is the "closing line" — the reference price for CLV computation.

        Returns
        -------
        dict with keys: home, draw, away, over, under, bookmaker, captured_at
        or an empty dict if no data is available.
        """
        session = get_session()
        try:
            db_match = self._resolve_db_match(session, match_home, match_away)
            if db_match is None:
                return {}

            # Determine kick-off datetime
            kickoff: Optional[datetime] = None
            if db_match.match_date and db_match.kick_off_time:
                try:
                    ko_str = f"{db_match.match_date}T{db_match.kick_off_time}:00"
                    kickoff = datetime.fromisoformat(ko_str).replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            elif db_match.match_date:
                kickoff = datetime(
                    db_match.match_date.year,
                    db_match.match_date.month,
                    db_match.match_date.day,
                    23, 59, 59,
                    tzinfo=timezone.utc,
                )

            query = (
                session.query(MatchOdds)
                .filter(
                    MatchOdds.match_id == db_match.id,
                    MatchOdds.market == "h2h",
                )
            )
            if kickoff is not None:
                query = query.filter(MatchOdds.captured_at <= kickoff)

            # Prefer Pinnacle for the closing line
            pinnacle_row: Optional[MatchOdds] = (
                query.filter(MatchOdds.bookmaker.in_(["pinnacle", "pinnaclesports"]))
                .order_by(MatchOdds.captured_at.desc())
                .first()
            )
            row: Optional[MatchOdds] = pinnacle_row or (
                query.order_by(MatchOdds.captured_at.desc()).first()
            )

            if row is None:
                return {}

            # Also fetch totals closing line
            totals_row: Optional[MatchOdds] = (
                session.query(MatchOdds)
                .filter(
                    MatchOdds.match_id == db_match.id,
                    MatchOdds.market == "totals",
                    MatchOdds.bookmaker == row.bookmaker,
                )
                .order_by(MatchOdds.captured_at.desc())
                .first()
            )

            return {
                "home": row.odds_home,
                "draw": row.odds_draw,
                "away": row.odds_away,
                "over": totals_row.odds_over if totals_row else None,
                "under": totals_row.odds_under if totals_row else None,
                "bookmaker": row.bookmaker,
                "captured_at": row.captured_at,
            }
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 7. Market efficiency
    # ------------------------------------------------------------------

    def compute_market_efficiency(
        self, sport_slug: str = "soccer", lookback_days: int = 30
    ) -> dict:
        """Compute how efficient the market has been over the lookback window.

        Compares average implied probabilities from stored odds against actual
        outcomes, and surfaces any systematic bias.

        Returns
        -------
        {
            efficiency_score: float,   # Brier-like: 0=perfect, higher=worse
            overround_avg: float,      # average market margin
            home_bias: float,          # how much market over-prices home (positive = over-priced)
            draw_bias: float,
            away_bias: float,
            sample_size: int,
        }
        """
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=lookback_days)
        session = get_session()

        try:
            # Pull all resolved matches with stored odds in the window
            rows: list[MatchOdds] = (
                session.query(MatchOdds)
                .join(Match, MatchOdds.match_id == Match.id)
                .filter(
                    MatchOdds.market == "h2h",
                    MatchOdds.captured_at >= cutoff,
                    Match.result.isnot(None),
                )
                .all()
            )

            if not rows:
                return {
                    "efficiency_score": None,
                    "overround_avg": None,
                    "home_bias": None,
                    "draw_bias": None,
                    "away_bias": None,
                    "sample_size": 0,
                }

            # Aggregate per match: take sharpest (lowest margin) bookmaker row
            match_data: dict[int, dict] = {}
            for row in rows:
                match = session.get(Match, row.match_id)
                if match is None or not all(
                    [row.odds_home, row.odds_draw, row.odds_away]
                ):
                    continue

                overround = (
                    _implied_prob(row.odds_home)
                    + _implied_prob(row.odds_draw)
                    + _implied_prob(row.odds_away)
                )
                existing = match_data.get(row.match_id)
                if existing is None or overround < existing["overround"]:
                    match_data[row.match_id] = {
                        "result": match.result,
                        "odds_home": row.odds_home,
                        "odds_draw": row.odds_draw,
                        "odds_away": row.odds_away,
                        "overround": overround,
                    }

            n = len(match_data)
            if n == 0:
                return {
                    "efficiency_score": None,
                    "overround_avg": None,
                    "home_bias": None,
                    "draw_bias": None,
                    "away_bias": None,
                    "sample_size": 0,
                }

            brier_sum = 0.0
            overround_sum = 0.0
            home_bias_sum = 0.0
            draw_bias_sum = 0.0
            away_bias_sum = 0.0

            for data in match_data.values():
                result = data["result"]  # "H", "D", "A"
                p_h = _implied_prob(data["odds_home"]) / data["overround"]
                p_d = _implied_prob(data["odds_draw"]) / data["overround"]
                p_a = _implied_prob(data["odds_away"]) / data["overround"]

                actual_h = 1.0 if result == "H" else 0.0
                actual_d = 1.0 if result == "D" else 0.0
                actual_a = 1.0 if result == "A" else 0.0

                brier_sum += (p_h - actual_h) ** 2 + (p_d - actual_d) ** 2 + (p_a - actual_a) ** 2
                overround_sum += data["overround"]
                home_bias_sum += p_h - actual_h
                draw_bias_sum += p_d - actual_d
                away_bias_sum += p_a - actual_a

            return {
                "efficiency_score": round(brier_sum / n, 4),
                "overround_avg": round((overround_sum / n - 1.0) * 100, 3),
                "home_bias": round(home_bias_sum / n, 4),
                "draw_bias": round(draw_bias_sum / n, 4),
                "away_bias": round(away_bias_sum / n, 4),
                "sample_size": n,
            }
        finally:
            session.close()

    # ------------------------------------------------------------------
    # 8. Full cycle
    # ------------------------------------------------------------------

    def run_odds_cycle(self) -> dict:
        """Execute one full odds cycle: fetch → compare → steam → store.

        Returns a summary dict suitable for logging or alerting.
        Designed to run every 15 minutes during trading hours.
        """
        start = time.monotonic()
        logger.info("Starting odds cycle")

        result: dict = {
            "started_at": datetime.now(tz=timezone.utc).isoformat(),
            "matches_fetched": 0,
            "bookmakers": [],
            "steam_moves": [],
            "rows_stored": 0,
            "quota_remaining": None,
            "errors": [],
        }

        try:
            odds_data = self.fetch_odds()
            result["matches_fetched"] = len(odds_data)
            result["quota_remaining"] = self._requests_remaining

            if not odds_data:
                result["errors"].append("No odds data returned from API")
                return result

            # Bookmaker comparison
            comparisons = self.compare_bookmakers(odds_data)
            result["bookmakers"] = [
                {
                    "name": c.bookmaker,
                    "margin_pct": c.margin_pct,
                    "n_markets": c.n_markets,
                    "avg_deviation_from_best_pct": c.avg_deviation_from_best,
                }
                for c in comparisons[:10]  # top 10
            ]

            # Steam moves — scan upcoming matches (within 48 h)
            cutoff = datetime.now(tz=timezone.utc) + timedelta(hours=48)
            steam_all: list[dict] = []
            for event in odds_data:
                try:
                    commence = datetime.fromisoformat(
                        event["commence_time"].replace("Z", "+00:00")
                    )
                except ValueError:
                    continue
                if commence > cutoff:
                    continue
                moves = self.detect_steam_moves(
                    event["home_team"], event["away_team"], odds_data=odds_data
                )
                for move in moves:
                    steam_all.append(
                        {
                            "home": event["home_team"],
                            "away": event["away_team"],
                            "outcome": move.outcome,
                            "direction": move.direction,
                            "magnitude_pct": move.magnitude_pct,
                            "bookmakers": move.bookmakers_moving,
                        }
                    )
            result["steam_moves"] = steam_all
            if steam_all:
                logger.warning(
                    "Steam moves detected: %d event(s) — %s",
                    len(steam_all),
                    ", ".join(
                        f"{m['home']} vs {m['away']} ({m['outcome']} {m['direction']})"
                        for m in steam_all[:5]
                    ),
                )

            # Persist
            rows_stored = self.store_snapshot(odds_data)
            result["rows_stored"] = rows_stored

        except Exception as exc:
            logger.exception("Odds cycle failed")
            result["errors"].append(str(exc))

        elapsed = time.monotonic() - start
        result["elapsed_seconds"] = round(elapsed, 2)
        logger.info(
            "Odds cycle complete: %d matches, %d steam moves, %d rows stored in %.1fs",
            result["matches_fetched"],
            len(result["steam_moves"]),
            result["rows_stored"],
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_db_match(self, session, home: str, away: str) -> Optional[Match]:
        """Find the DB Match record via fuzzy name matching."""
        # Only look at recent/upcoming matches (within ±7 days) to avoid
        # false matches on common team names
        from datetime import date as date_type

        today = date_type.today()
        window_start = today - timedelta(days=7)
        window_end = today + timedelta(days=14)

        candidates: list[Match] = (
            session.query(Match)
            .filter(
                Match.match_date >= window_start,
                Match.match_date <= window_end,
            )
            .all()
        )

        best_score = 0
        best_match = None

        for match in candidates:
            h_name = match.home_team.canonical_name if match.home_team else ""
            a_name = match.away_team.canonical_name if match.away_team else ""
            h_score = fuzz.ratio(home.lower(), h_name.lower())
            a_score = fuzz.ratio(away.lower(), a_name.lower())
            combined = (h_score + a_score) / 2
            if combined > best_score:
                best_score = combined
                best_match = match

        if best_score >= _TEAM_MATCH_THRESHOLD:
            return best_match
        return None
