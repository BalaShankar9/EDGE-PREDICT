"""Live Result Resolver with CLV (Closing Line Value) tracking.

Fetches match results from football-data.org (primary) or football-data.co.uk
CSVs (fallback), resolves all pending DailyPicks, computes CLV%, updates the
BankrollLedger, and triggers agent performance resolution via AgentPerformance
snapshots.
"""
from __future__ import annotations

import io
import logging
from datetime import date, datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session
from thefuzz import fuzz

from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.models import (
    AgentPerformance,
    BankrollLedger,
    DailyPick,
    Match,
    PipelineRun,
)
from sharpedge.pipeline.resolver import resolve_pick

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FD_ORG_BASE = "https://api.football-data.org/v4"
_FD_UK_BASE = "https://www.football-data.co.uk/mmz4281"

# Maps football-data.co.uk league codes to season URL fragments
_FD_UK_LEAGUE_CODES: dict[str, str] = {
    "Premier League": "E0",
    "La Liga": "SP1",
    "Bundesliga": "D1",
    "Serie A": "I1",
    "Ligue 1": "F1",
    "Eredivisie": "N1",
    "Liga Portugal": "P1",
    "Belgian Pro League": "B1",
    "Turkish Super Lig": "T1",
    "Scottish Premiership": "SC0",
    "Championship": "EC",
}

# Fuzzy-match threshold (0–100). 75 handles "Man United" vs "Manchester United FC"
_FUZZY_THRESHOLD = 75

# Finished statuses returned by football-data.org v4
_FINISHED_STATUSES = {"FINISHED"}


# ---------------------------------------------------------------------------
# LiveResolver
# ---------------------------------------------------------------------------


class LiveResolver:
    """Resolves pending DailyPicks against actual match results.

    Workflow per date
    -----------------
    1. fetch_results(date)  — API primary, CSV fallback
    2. For each pending pick, fuzzy-match team names to a result row
    3. resolve_pick()       — determines win/loss/void + P&L
    4. compute_clv()        — measures beating the closing line
    5. update_bankroll()    — immutable ledger entry
    6. Persist pick.result / pick.profit_loss / pick.resolved_at
    7. Trigger agent performance snapshots
    """

    def __init__(self, http_timeout: int = 30) -> None:
        self._http_timeout = http_timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Resolve all pending dates.  Entry-point for schedulers/CLI.

        Returns
        -------
        dict
            Aggregated summary across all resolved dates.
        """
        pending = self.get_pending_dates()
        if not pending:
            logger.info("live_resolver: no pending dates found — nothing to do")
            return {"dates_processed": 0, "total_resolved": 0, "total_profit_loss": 0.0}

        logger.info("live_resolver: %d pending date(s): %s", len(pending), pending)

        agg: dict = {
            "dates_processed": 0,
            "total_resolved": 0,
            "total_wins": 0,
            "total_losses": 0,
            "total_voids": 0,
            "total_profit_loss": 0.0,
            "avg_clv_pct": 0.0,
            "by_date": {},
        }
        clv_values: list[float] = []

        for target_date in pending:
            try:
                summary = self.resolve_day(target_date)
                agg["dates_processed"] += 1
                agg["total_resolved"] += summary.get("resolved", 0)
                agg["total_wins"] += summary.get("wins", 0)
                agg["total_losses"] += summary.get("losses", 0)
                agg["total_voids"] += summary.get("voids", 0)
                agg["total_profit_loss"] += summary.get("profit_loss", 0.0)
                agg["by_date"][str(target_date)] = summary
                if summary.get("clv_values"):
                    clv_values.extend(summary["clv_values"])
            except Exception:
                logger.exception("live_resolver: error processing date %s", target_date)

        if clv_values:
            agg["avg_clv_pct"] = round(sum(clv_values) / len(clv_values), 4)

        logger.info(
            "live_resolver run complete — %d dates, %d picks resolved, P&L=%.4f",
            agg["dates_processed"],
            agg["total_resolved"],
            agg["total_profit_loss"],
        )
        return agg

    def get_pending_dates(self) -> list[date]:
        """Return distinct match dates that have at least one unresolved pick."""
        with get_session() as session:
            rows = (
                session.execute(
                    select(DailyPick.match_date)
                    .where(DailyPick.result.is_(None))
                    .distinct()
                    .order_by(DailyPick.match_date)
                )
                .scalars()
                .all()
            )
            return list(rows)

    def resolve_day(self, target_date: date) -> dict:
        """Resolve all unresolved picks for *target_date*.

        Returns
        -------
        dict
            Summary with keys: date, resolved, wins, losses, voids,
            profit_loss, not_matched, clv_values, errors.
        """
        logger.info("live_resolver: resolving %s", target_date)

        results = self.fetch_results(target_date)
        if not results:
            logger.warning("live_resolver: no results available for %s", target_date)
            return {
                "date": str(target_date),
                "resolved": 0,
                "wins": 0,
                "losses": 0,
                "voids": 0,
                "profit_loss": 0.0,
                "not_matched": 0,
                "clv_values": [],
                "errors": 0,
            }

        summary = {
            "date": str(target_date),
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "voids": 0,
            "profit_loss": 0.0,
            "not_matched": 0,
            "clv_values": [],
            "errors": 0,
        }

        with get_session() as session:
            pending_picks: list[DailyPick] = (
                session.execute(
                    select(DailyPick)
                    .where(
                        DailyPick.match_date == target_date,
                        DailyPick.result.is_(None),
                    )
                )
                .scalars()
                .all()
            )

            if not pending_picks:
                logger.info("live_resolver: no pending picks for %s", target_date)
                return summary

            logger.info(
                "live_resolver: %d pending picks for %s",
                len(pending_picks),
                target_date,
            )

            for pick in pending_picks:
                try:
                    matched_result = self._match_pick_to_result(pick, results)
                    if matched_result is None:
                        logger.warning(
                            "live_resolver: no result match for pick %d (%s vs %s)",
                            pick.id,
                            pick.home_team,
                            pick.away_team,
                        )
                        summary["not_matched"] += 1
                        continue

                    # Only resolve finished matches
                    if matched_result.get("status") not in _FINISHED_STATUSES:
                        logger.debug(
                            "live_resolver: pick %d match not finished (status=%s)",
                            pick.id,
                            matched_result.get("status"),
                        )
                        summary["not_matched"] += 1
                        continue

                    home_goals: Optional[int] = matched_result.get("home_goals")
                    away_goals: Optional[int] = matched_result.get("away_goals")
                    stake = pick.stake_flat or pick.stake_kelly or 1.0

                    resolution = resolve_pick(
                        pick_market=pick.pick_market,
                        home_goals=home_goals,
                        away_goals=away_goals,
                        best_odds=pick.best_odds,
                        stake=stake,
                    )
                    result_str: str = resolution["result"]
                    profit_loss: float = resolution["profit_loss"]

                    # CLV calculation — use closing odds from meta if stored,
                    # fall back to the implied_prob-derived closing odds
                    closing_odds = self._extract_closing_odds(pick)
                    clv_pct: Optional[float] = None
                    if closing_odds is not None and closing_odds > 1.0:
                        clv_pct = self.compute_clv(pick.best_odds, closing_odds)
                        summary["clv_values"].append(clv_pct)

                    # Persist resolution on the pick
                    pick.result = result_str
                    pick.profit_loss = profit_loss
                    pick.resolved_at = datetime.now(timezone.utc)

                    # Store CLV in risk_flags meta so it's queryable
                    if clv_pct is not None:
                        flags = dict(pick.risk_flags or {})
                        flags["clv_pct"] = round(clv_pct, 4)
                        pick.risk_flags = flags

                    session.add(pick)

                    # Bankroll ledger
                    try:
                        self.update_bankroll(
                            pick_id=pick.id,
                            result=result_str,
                            profit_loss=profit_loss,
                            stake=stake,
                            session=session,
                        )
                    except Exception:
                        logger.exception(
                            "live_resolver: bankroll update failed for pick %d", pick.id
                        )

                    # Tally
                    summary["resolved"] += 1
                    summary["profit_loss"] += profit_loss
                    if result_str == "win":
                        summary["wins"] += 1
                    elif result_str == "loss":
                        summary["losses"] += 1
                    else:
                        summary["voids"] += 1

                except Exception:
                    logger.exception(
                        "live_resolver: error resolving pick %d", pick.id
                    )
                    summary["errors"] += 1

            try:
                session.commit()
                logger.info(
                    "live_resolver: committed %d resolutions for %s",
                    summary["resolved"],
                    target_date,
                )
            except Exception:
                session.rollback()
                logger.exception(
                    "live_resolver: DB commit failed for %s — rolling back", target_date
                )
                raise

        # Trigger agent performance snapshots (outside the main transaction)
        try:
            self._trigger_agent_performance(target_date, summary)
        except Exception:
            logger.exception(
                "live_resolver: agent performance update failed for %s", target_date
            )

        summary["profit_loss"] = round(summary["profit_loss"], 4)
        return summary

    def fetch_results(self, match_date: date) -> list[dict]:
        """Fetch match results for *match_date*.

        Tries football-data.org API first; falls back to scraping
        football-data.co.uk CSVs if the API key is absent or the call fails.

        Returns
        -------
        list[dict]
            Each element: {home_team, away_team, home_goals, away_goals, status}
        """
        api_key = settings.football_data_api_key
        if api_key:
            try:
                return self._fetch_results_api(match_date, api_key)
            except Exception:
                logger.warning(
                    "live_resolver: football-data.org API failed for %s — falling back to CSV",
                    match_date,
                )

        return self._fetch_results_csv_fallback(match_date)

    @staticmethod
    def compute_clv(pick_odds: float, closing_odds: float) -> float:
        """Compute Closing Line Value percentage.

        CLV% = ((1 / closing_odds) / (1 / pick_odds) - 1) * 100

        Positive CLV means we got a better price than market close.
        Negative CLV means the market tightened against us.

        Parameters
        ----------
        pick_odds : decimal odds at time of pick
        closing_odds : decimal odds at kick-off (market close)

        Returns
        -------
        float  CLV in percent (e.g. 3.2 means +3.2%)
        """
        if pick_odds <= 1.0 or closing_odds <= 1.0:
            return 0.0
        implied_pick = 1.0 / pick_odds
        implied_close = 1.0 / closing_odds
        return round((implied_close / implied_pick - 1.0) * 100.0, 4)

    def update_bankroll(
        self,
        pick_id: int,
        result: str,
        profit_loss: float,
        stake: float,
        session: Optional[Session] = None,
    ) -> None:
        """Write an immutable entry to BankrollLedger.

        Parameters
        ----------
        pick_id : DailyPick.id
        result  : "win" / "loss" / "void"
        profit_loss : net P&L for the pick
        stake   : amount staked
        session : if provided, reuses the existing session (no commit);
                  otherwise opens a new session and commits.
        """
        own_session = session is None
        if own_session:
            session = get_session()

        try:
            last_entry = (
                session.execute(
                    select(BankrollLedger)
                    .order_by(BankrollLedger.id.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            bankroll_before = last_entry.bankroll_after if last_entry else settings.bankroll_initial
            bankroll_after = bankroll_before + profit_loss

            # Compute drawdown relative to running peak
            peak_query = session.execute(
                select(BankrollLedger.bankroll_after)
                .order_by(BankrollLedger.id.asc())
            ).scalars().all()
            bankroll_initial = settings.bankroll_initial
            peak = max(peak_query, default=bankroll_initial)
            peak = max(peak, bankroll_initial)
            drawdown_pct = max(0.0, (peak - bankroll_after) / peak) if peak > 0 else 0.0

            # Determine circuit breaker level
            cb_level = 0
            if drawdown_pct >= 0.20:
                cb_level = 4
            elif drawdown_pct >= 0.15:
                cb_level = 3
            elif drawdown_pct >= 0.10:
                cb_level = 2
            elif drawdown_pct >= 0.05:
                cb_level = 1

            event_type = "void" if result == "void" else ("payout" if result == "win" else "stake")

            entry = BankrollLedger(
                event_date=date.today(),
                event_type=event_type,
                pick_id=pick_id,
                amount=profit_loss,
                bankroll_before=round(bankroll_before, 4),
                bankroll_after=round(bankroll_after, 4),
                drawdown_pct=round(drawdown_pct, 6),
                circuit_breaker_level=cb_level,
                notes=f"pick_id={pick_id} result={result} stake={stake:.4f} pl={profit_loss:.4f}",
            )
            session.add(entry)

            if own_session:
                session.commit()

            logger.debug(
                "live_resolver: bankroll %s -> %s (pick=%d, result=%s)",
                bankroll_before,
                bankroll_after,
                pick_id,
                result,
            )
        except Exception:
            if own_session:
                session.rollback()
            raise
        finally:
            if own_session:
                session.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_results_api(self, match_date: date, api_key: str) -> list[dict]:
        """Fetch results from football-data.org /v4/matches endpoint."""
        date_str = match_date.isoformat()
        url = f"{_FD_ORG_BASE}/matches?dateFrom={date_str}&dateTo={date_str}"
        headers = {"X-Auth-Token": api_key}

        logger.debug("live_resolver: GET %s", url)
        with httpx.Client(timeout=self._http_timeout) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        matches = data.get("matches", [])
        results: list[dict] = []
        for m in matches:
            home = m.get("homeTeam", {}).get("name", "")
            away = m.get("awayTeam", {}).get("name", "")
            score = m.get("score", {})
            full_time = score.get("fullTime", {})
            status = m.get("status", "")

            home_goals: Optional[int] = full_time.get("home")
            away_goals: Optional[int] = full_time.get("away")

            if not home or not away:
                continue

            results.append(
                {
                    "home_team": home,
                    "away_team": away,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "status": status,
                }
            )

        logger.info(
            "live_resolver: API returned %d matches for %s", len(results), match_date
        )
        return results

    def _fetch_results_csv_fallback(self, match_date: date) -> list[dict]:
        """Fallback: download football-data.co.uk CSV and filter by date.

        Iterates over all known leagues for the current season and
        searches for rows matching *match_date*.
        """
        # Derive season code from match_date (seasons start in July)
        year = match_date.year
        if match_date.month >= 7:
            season_code = f"{str(year)[2:]}{str(year + 1)[2:]}"  # e.g. "2425"
        else:
            season_code = f"{str(year - 1)[2:]}{str(year)[2:]}"

        results: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for league_name, league_code in _FD_UK_LEAGUE_CODES.items():
            url = f"{_FD_UK_BASE}/{season_code}/{league_code}.csv"
            try:
                with httpx.Client(timeout=self._http_timeout) as client:
                    resp = client.get(url)
                    resp.raise_for_status()

                import pandas as pd

                df = pd.read_csv(io.StringIO(resp.text))
                df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

                # Parse the Date column — football-data.co.uk uses DD/MM/YY or DD/MM/YYYY
                for fmt in ("%d/%m/%y", "%d/%m/%Y"):
                    try:
                        df["_parsed_date"] = pd.to_datetime(df["Date"], format=fmt).dt.date
                        break
                    except Exception:
                        continue
                else:
                    logger.warning(
                        "live_resolver: could not parse dates from %s CSV", league_name
                    )
                    continue

                day_df = df[df["_parsed_date"] == match_date]
                for _, row in day_df.iterrows():
                    home = str(row["HomeTeam"]).strip()
                    away = str(row["AwayTeam"]).strip()
                    key = (home.lower(), away.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(
                        {
                            "home_team": home,
                            "away_team": away,
                            "home_goals": int(row["FTHG"]),
                            "away_goals": int(row["FTAG"]),
                            "status": "FINISHED",
                        }
                    )

            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    logger.debug(
                        "live_resolver: CSV not found for %s / %s", league_name, season_code
                    )
                else:
                    logger.warning(
                        "live_resolver: CSV fetch error %s for %s: %s",
                        exc.response.status_code,
                        league_name,
                        exc,
                    )
            except Exception:
                logger.warning(
                    "live_resolver: CSV fallback failed for %s", league_name, exc_info=True
                )

        logger.info(
            "live_resolver: CSV fallback found %d matches for %s", len(results), match_date
        )
        return results

    @staticmethod
    def _match_pick_to_result(
        pick: DailyPick, results: list[dict]
    ) -> Optional[dict]:
        """Fuzzy-match a pick's home/away team names to a result row.

        Returns the best-matching result dict, or None if no match meets
        the threshold.
        """
        best_score = 0
        best_result: Optional[dict] = None

        pick_home = pick.home_team.lower().strip()
        pick_away = pick.away_team.lower().strip()

        for r in results:
            r_home = r["home_team"].lower().strip()
            r_away = r["away_team"].lower().strip()

            home_score = fuzz.ratio(pick_home, r_home)
            away_score = fuzz.ratio(pick_away, r_away)
            combined = (home_score + away_score) / 2

            if combined > best_score:
                best_score = combined
                best_result = r

        if best_score >= _FUZZY_THRESHOLD:
            logger.debug(
                "live_resolver: matched '%s vs %s' -> '%s vs %s' (score=%.1f)",
                pick.home_team,
                pick.away_team,
                best_result["home_team"] if best_result else "?",
                best_result["away_team"] if best_result else "?",
                best_score,
            )
            return best_result

        logger.debug(
            "live_resolver: no match for '%s vs %s' (best_score=%.1f < threshold=%d)",
            pick.home_team,
            pick.away_team,
            best_score,
            _FUZZY_THRESHOLD,
        )
        return None

    @staticmethod
    def _extract_closing_odds(pick: DailyPick) -> Optional[float]:
        """Extract closing odds from pick metadata if available.

        Checks pick.risk_flags["closing_odds"] first, then falls back to
        deriving a rough closing price from the implied_prob stored at
        pick creation time (which already incorporates vig).

        Returns decimal closing odds, or None if unavailable.
        """
        flags = pick.risk_flags or {}

        # Explicit closing odds stored during pre-match pipeline
        if "closing_odds" in flags:
            val = flags["closing_odds"]
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

        # Fallback: compute from implied_prob (the pick's own implied_prob is
        # the fair price at pick-time; we can't reconstruct true closing odds
        # without fresh market data, so we return None to skip CLV computation)
        return None

    def _trigger_agent_performance(self, target_date: date, summary: dict) -> None:
        """Write / update AgentPerformance snapshot for the resolved date.

        We record a lightweight snapshot keyed to the special "live_resolver"
        agent name so the system knows resolution was completed for this date.
        Additional per-agent snapshots should be computed by the full
        AgentPerformance pipeline that has access to agent IDs.
        """
        if summary["resolved"] == 0:
            return

        clv_values: list[float] = summary.get("clv_values", [])
        avg_clv = sum(clv_values) / len(clv_values) if clv_values else None

        total = summary["resolved"]
        wins = summary["wins"]
        roi_pct = (
            (summary["profit_loss"] / (total * 1.0)) * 100.0 if total > 0 else None
        )

        with get_session() as session:
            try:
                snapshot = AgentPerformance(
                    agent_id=None,  # type: ignore[arg-type]  — no agent row for resolver
                    sport_slug="football",
                    date=target_date,
                    window_days=1,
                    total_bets=total,
                    wins=wins,
                    roi_pct=round(roi_pct, 4) if roi_pct is not None else None,
                    clv_pct=round(avg_clv, 4) if avg_clv is not None else None,
                    avg_confidence=None,
                    brier_score=None,
                )
                session.add(snapshot)
                session.commit()
                logger.debug(
                    "live_resolver: AgentPerformance snapshot written for %s", target_date
                )
            except Exception:
                session.rollback()
                logger.warning(
                    "live_resolver: could not write AgentPerformance snapshot for %s",
                    target_date,
                    exc_info=True,
                )
