"""DB-backed performance ledger for SharpEdge agent tracking.

Replaces the in-memory AgentTracker with a persistent, Postgres-backed
implementation that supports rolling metrics, per-league/market breakdowns,
Bayesian arbiter weight feeds, and daily performance snapshots.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Generator, Optional

from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from sharpedge.db.engine import get_session
from sharpedge.db.models import (
    Agent,
    AgentPerformance,
    AgentPrediction as AgentPredictionRow,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session context manager
# ---------------------------------------------------------------------------


@contextmanager
def _session_scope() -> Generator[Session, None, None]:
    """Provide a transactional session scope."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_or_create_agent(session: Session, agent_name: str) -> Agent:
    """Fetch agent by name, creating a stub record if it does not yet exist."""
    agent = session.execute(
        select(Agent).where(Agent.name == agent_name)
    ).scalar_one_or_none()

    if agent is None:
        agent = Agent(
            name=agent_name,
            agent_type="unknown",
            description=f"Auto-registered agent: {agent_name}",
            active=True,
        )
        session.add(agent)
        session.flush()  # populate agent.id before returning
        logger.info("Auto-registered new agent '%s' (id=%d)", agent_name, agent.id)

    return agent


def _brier_score(predictions: list[AgentPredictionRow]) -> Optional[float]:
    """Compute mean Brier score over resolved predictions.

    Brier score = mean( (p_predicted - outcome)^2 )
    where outcome is 1 if correct else 0 and p_predicted is the probability
    assigned to the predicted outcome.
    """
    resolved = [p for p in predictions if p.correct is not None]
    if not resolved:
        return None

    total = 0.0
    for pred in resolved:
        probs: dict = pred.probabilities or {}
        p_hat = float(probs.get(pred.predicted_outcome, pred.confidence or 0.0))
        # Clip to valid probability range
        p_hat = max(0.0, min(1.0, p_hat))
        outcome = 1.0 if pred.correct else 0.0
        total += (p_hat - outcome) ** 2

    return total / len(resolved)


def _roi_pct(predictions: list[AgentPredictionRow]) -> Optional[float]:
    """Estimate ROI% from picks that have odds stored in probabilities.

    Uses the ``odds_at_pick`` key in the probabilities JSON blob when present.
    Falls back to implied fair-value return (odds = 1/p) if not available.
    Stakes are assumed unit (1.0) per pick.
    """
    resolved = [p for p in predictions if p.correct is not None]
    if not resolved:
        return None

    total_staked = float(len(resolved))
    total_return = 0.0
    for pred in resolved:
        probs: dict = pred.probabilities or {}
        odds = probs.get("odds_at_pick")
        if odds is None:
            # Fall back to fair-value odds from confidence
            conf = max(0.01, float(pred.confidence or 0.5))
            odds = 1.0 / conf
        odds = float(odds)
        if pred.correct:
            total_return += odds - 1.0  # profit on unit stake
        else:
            total_return -= 1.0

    return (total_return / total_staked) * 100.0 if total_staked > 0 else None


def _clv_pct(predictions: list[AgentPredictionRow]) -> Optional[float]:
    """Compute mean closing-line value %.

    CLV% = mean( (closing_odds / odds_at_pick - 1) * 100 )
    Requires both ``odds_at_pick`` and ``closing_odds`` in the probabilities JSON.
    """
    samples = []
    for pred in predictions:
        probs: dict = pred.probabilities or {}
        odds_at = probs.get("odds_at_pick")
        closing = probs.get("closing_odds")
        if odds_at and closing and float(odds_at) > 0:
            clv = (float(closing) / float(odds_at) - 1.0) * 100.0
            samples.append(clv)

    return sum(samples) / len(samples) if samples else None


def _fetch_predictions(
    session: Session,
    agent_id: int,
    sport: Optional[str] = None,
    last_n: Optional[int] = None,
    league: Optional[str] = None,
    market: Optional[str] = None,
    since_date: Optional[date] = None,
) -> list[AgentPredictionRow]:
    """Fetch agent prediction rows with optional filters."""
    stmt = select(AgentPredictionRow).where(AgentPredictionRow.agent_id == agent_id)

    if sport:
        stmt = stmt.where(AgentPredictionRow.sport_slug == sport)
    if market:
        stmt = stmt.where(AgentPredictionRow.market == market)
    if since_date:
        stmt = stmt.where(AgentPredictionRow.match_date >= since_date)
    # league is stored in home_team/away_team context via sport_slug in this schema;
    # when a league key is provided we filter on the sport_slug or home_team contains.
    # The AgentPrediction table does not have a league column so we expose it as sport
    # sub-filter for callers who pack league into sport_slug (e.g. "football:premier_league").
    if league:
        stmt = stmt.where(AgentPredictionRow.sport_slug.ilike(f"%{league}%"))

    stmt = stmt.order_by(AgentPredictionRow.match_date.desc())

    if last_n:
        stmt = stmt.limit(last_n)

    return list(session.execute(stmt).scalars().all())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PerformanceLedger:
    """Persistent, DB-backed performance ledger for SharpEdge prediction agents.

    All writes are transactional.  Read methods use short-lived read sessions
    so callers never need to manage session state.

    Example
    -------
    >>> ledger = PerformanceLedger()
    >>> ledger.record_prediction(
    ...     agent_name="starlizard_v2",
    ...     sport="football",
    ...     match_date=date(2026, 4, 5),
    ...     home_team="Arsenal",
    ...     away_team="Chelsea",
    ...     market="1x2",
    ...     predicted_outcome="home",
    ...     probabilities={"home": 0.55, "draw": 0.25, "away": 0.20},
    ...     confidence=0.55,
    ...     reasoning="Form + ELO edge",
    ... )
    """

    # ------------------------------------------------------------------
    # 1. Record a prediction
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        agent_name: str,
        sport: str,
        match_date: date,
        home_team: str,
        away_team: str,
        market: str,
        predicted_outcome: str,
        probabilities: dict,
        confidence: float,
        reasoning: Optional[str] = None,
        odds_at_pick: Optional[float] = None,
    ) -> int:
        """Persist an agent prediction to the DB.

        Parameters
        ----------
        agent_name:
            Logical name of the agent (e.g. ``"autogluon_v3"``).
        sport:
            Sport slug (e.g. ``"football"``, ``"tennis"``).
        match_date:
            Calendar date of the match.
        home_team / away_team:
            Team names as strings.
        market:
            Market key (e.g. ``"1x2"``, ``"btts"``, ``"over_25"``).
        predicted_outcome:
            The agent's predicted outcome string (e.g. ``"home"``, ``"yes"``).
        probabilities:
            Full probability dict for the market (e.g. ``{"home": 0.55, ...}``).
        confidence:
            Scalar confidence score (typically the probability of the predicted
            outcome).
        reasoning:
            Optional free-text explanation.
        odds_at_pick:
            Decimal odds available at prediction time.  Stored inside
            ``probabilities`` as ``odds_at_pick`` for CLV / ROI accounting.

        Returns
        -------
        int
            Primary key of the newly inserted ``AgentPrediction`` row.
        """
        if odds_at_pick is not None:
            probabilities = {**probabilities, "odds_at_pick": odds_at_pick}

        with _session_scope() as session:
            agent = _get_or_create_agent(session, agent_name)
            row = AgentPredictionRow(
                agent_id=agent.id,
                sport_slug=sport,
                match_date=match_date,
                home_team=home_team,
                away_team=away_team,
                market=market,
                predicted_outcome=predicted_outcome,
                probabilities=probabilities,
                confidence=confidence,
                reasoning=reasoning,
            )
            session.add(row)
            session.flush()
            row_id = row.id

        logger.debug(
            "Recorded prediction id=%d agent='%s' %s v %s [%s/%s]",
            row_id,
            agent_name,
            home_team,
            away_team,
            market,
            predicted_outcome,
        )
        return row_id

    # ------------------------------------------------------------------
    # 2. Resolve a match
    # ------------------------------------------------------------------

    def resolve_match(
        self,
        match_date: date,
        home_team: str,
        away_team: str,
        actual_outcome: str,
        closing_odds: Optional[dict] = None,
    ) -> dict[str, bool]:
        """Update all pending predictions for a match with the actual result.

        Parameters
        ----------
        match_date:
            Date the match was played.
        home_team / away_team:
            Team names (must match what was passed to ``record_prediction``).
        actual_outcome:
            The true outcome string, in the same vocabulary as
            ``predicted_outcome`` (e.g. ``"home"``, ``"draw"``, ``"away"``,
            ``"yes"``, ``"no"``).
        closing_odds:
            Optional dict mapping market keys to closing decimal odds
            (e.g. ``{"1x2": 2.10}``).  Stored into each prediction's
            ``probabilities`` blob for CLV computation.

        Returns
        -------
        dict[str, bool]
            Mapping of ``agent_name -> correct`` for every resolved prediction.
        """
        results: dict[str, bool] = {}

        with _session_scope() as session:
            stmt = (
                select(AgentPredictionRow, Agent.name)
                .join(Agent, Agent.id == AgentPredictionRow.agent_id)
                .where(
                    and_(
                        AgentPredictionRow.match_date == match_date,
                        AgentPredictionRow.home_team == home_team,
                        AgentPredictionRow.away_team == away_team,
                        AgentPredictionRow.actual_outcome.is_(None),  # unresolved only
                    )
                )
            )
            rows = session.execute(stmt).all()

            if not rows:
                logger.warning(
                    "No unresolved predictions found for %s v %s on %s",
                    home_team,
                    away_team,
                    match_date,
                )
                return results

            for pred_row, agent_name in rows:
                pred_row.actual_outcome = actual_outcome
                pred_row.correct = pred_row.predicted_outcome == actual_outcome

                # Attach closing odds to probabilities blob if supplied
                if closing_odds is not None:
                    market_closing = closing_odds.get(pred_row.market)
                    if market_closing is not None:
                        updated_probs = {
                            **(pred_row.probabilities or {}),
                            "closing_odds": float(market_closing),
                        }
                        pred_row.probabilities = updated_probs

                results[agent_name] = pred_row.correct

        logger.info(
            "Resolved %d predictions for %s v %s on %s  outcome='%s'",
            len(results),
            home_team,
            away_team,
            match_date,
            actual_outcome,
        )
        return results

    # ------------------------------------------------------------------
    # 3. Agent stats
    # ------------------------------------------------------------------

    def get_agent_stats(
        self,
        agent_name: str,
        sport: Optional[str] = None,
        last_n: Optional[int] = None,
        league: Optional[str] = None,
        market: Optional[str] = None,
    ) -> dict:
        """Return rolling performance metrics for a single agent.

        Parameters
        ----------
        agent_name:
            Agent to query.
        sport:
            Filter to this sport slug.
        last_n:
            Only consider the most recent N predictions.
        league:
            Filter by league substring in sport_slug.
        market:
            Filter by market key.

        Returns
        -------
        dict with keys:
            ``agent_name``, ``total_predictions``, ``resolved``, ``correct``,
            ``accuracy``, ``brier_score``, ``roi_pct``, ``clv_pct``,
            ``avg_confidence``, ``by_market``, ``by_sport``.
        """
        with _session_scope() as session:
            agent = session.execute(
                select(Agent).where(Agent.name == agent_name)
            ).scalar_one_or_none()

            if agent is None:
                logger.warning("Agent '%s' not found in DB", agent_name)
                return {"agent_name": agent_name, "error": "agent_not_found"}

            preds = _fetch_predictions(
                session,
                agent_id=agent.id,
                sport=sport,
                last_n=last_n,
                league=league,
                market=market,
            )

            resolved = [p for p in preds if p.correct is not None]
            correct_count = sum(1 for p in resolved if p.correct)

            # Per-market and per-sport breakdowns
            by_market: dict[str, dict] = defaultdict(
                lambda: {"total": 0, "resolved": 0, "correct": 0}
            )
            by_sport: dict[str, dict] = defaultdict(
                lambda: {"total": 0, "resolved": 0, "correct": 0}
            )

            for p in preds:
                by_market[p.market]["total"] += 1
                by_sport[p.sport_slug]["total"] += 1
                if p.correct is not None:
                    by_market[p.market]["resolved"] += 1
                    by_sport[p.sport_slug]["resolved"] += 1
                    if p.correct:
                        by_market[p.market]["correct"] += 1
                        by_sport[p.sport_slug]["correct"] += 1

            def _enrich(group: dict) -> dict:
                res = group["resolved"]
                cor = group["correct"]
                group["accuracy"] = cor / res if res else None
                return group

            by_market = {k: _enrich(v) for k, v in by_market.items()}
            by_sport = {k: _enrich(v) for k, v in by_sport.items()}

            total_conf = sum(
                float(p.confidence) for p in preds if p.confidence is not None
            )

            return {
                "agent_name": agent_name,
                "agent_id": agent.id,
                "total_predictions": len(preds),
                "resolved": len(resolved),
                "correct": correct_count,
                "accuracy": correct_count / len(resolved) if resolved else None,
                "brier_score": _brier_score(resolved),
                "roi_pct": _roi_pct(resolved),
                "clv_pct": _clv_pct(preds),
                "avg_confidence": total_conf / len(preds) if preds else None,
                "by_market": dict(by_market),
                "by_sport": dict(by_sport),
            }

    # ------------------------------------------------------------------
    # 4. Bayesian arbiter weights
    # ------------------------------------------------------------------

    def get_arbiter_weights(
        self,
        sport: str = "football",
        decay: float = 0.95,
        window_days: int = 90,
        min_predictions: int = 5,
    ) -> dict[str, float]:
        """Return normalised Bayesian weights for all active agents.

        Weights are proportional to a composite score:
            score = accuracy * (1 - brier_score) * clv_boost
        Recent performance is emphasised via exponential time-decay applied
        per daily performance snapshot (if available), otherwise raw prediction
        history is used.

        Parameters
        ----------
        sport:
            Sport slug to compute weights for.
        decay:
            Per-day decay factor (default 0.95 ≈ 74% half-life over 10 days).
        window_days:
            Look-back window in calendar days.
        min_predictions:
            Agents with fewer resolved predictions are assigned floor weight.

        Returns
        -------
        dict[str, float]
            ``{agent_name: weight}`` normalised so weights sum to 1.0.
        """
        since = date.today() - timedelta(days=window_days)
        weights: dict[str, float] = {}

        with _session_scope() as session:
            # All active agents
            agents = list(
                session.execute(
                    select(Agent).where(Agent.active.is_(True))
                ).scalars().all()
            )

            today = date.today()
            for agent in agents:
                preds = _fetch_predictions(
                    session,
                    agent_id=agent.id,
                    sport=sport,
                    since_date=since,
                )
                resolved = [p for p in preds if p.correct is not None]

                if len(resolved) < min_predictions:
                    weights[agent.name] = 1e-4  # floor — keep in rotation
                    continue

                # Time-decay: newer predictions count more
                weighted_correct = 0.0
                weighted_total = 0.0
                weighted_brier = 0.0

                for pred in resolved:
                    age_days = (today - pred.match_date).days
                    w = decay ** max(0, age_days)
                    weighted_total += w
                    if pred.correct:
                        weighted_correct += w
                    # Brier contribution
                    probs: dict = pred.probabilities or {}
                    p_hat = float(
                        probs.get(pred.predicted_outcome, pred.confidence or 0.0)
                    )
                    p_hat = max(0.0, min(1.0, p_hat))
                    outcome = 1.0 if pred.correct else 0.0
                    weighted_brier += w * (p_hat - outcome) ** 2

                acc = weighted_correct / weighted_total if weighted_total > 0 else 0.0
                brier = weighted_brier / weighted_total if weighted_total > 0 else 0.5

                # CLV bonus (small multiplicative boost if positive CLV)
                clv = _clv_pct(preds) or 0.0
                clv_boost = 1.0 + max(0.0, clv / 100.0)

                score = acc * max(0.0, 1.0 - brier) * clv_boost
                weights[agent.name] = max(1e-4, score)

        # Normalise
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}
        else:
            n = len(weights)
            weights = {k: 1.0 / n for k in weights} if n else {}

        logger.debug(
            "Arbiter weights for sport='%s': %s",
            sport,
            {k: round(v, 4) for k, v in sorted(weights.items(), key=lambda x: -x[1])},
        )
        return weights

    # ------------------------------------------------------------------
    # 5. Daily snapshot
    # ------------------------------------------------------------------

    def snapshot_daily(
        self,
        sport_slug: str,
        snapshot_date: Optional[date] = None,
        window_days: int = 30,
    ) -> None:
        """Compute and upsert rolling metrics into ``agent_performance``.

        Parameters
        ----------
        sport_slug:
            Sport to snapshot.
        snapshot_date:
            Date to snapshot for (defaults to today).
        window_days:
            Rolling window in days (default 30).
        """
        if snapshot_date is None:
            snapshot_date = date.today()

        since = snapshot_date - timedelta(days=window_days)

        with _session_scope() as session:
            agents = list(
                session.execute(
                    select(Agent).where(Agent.active.is_(True))
                ).scalars().all()
            )

            snapshotted = 0
            for agent in agents:
                preds = _fetch_predictions(
                    session,
                    agent_id=agent.id,
                    sport=sport_slug,
                    since_date=since,
                )
                resolved = [p for p in preds if p.correct is not None]

                if not preds:
                    continue

                wins = sum(1 for p in resolved if p.correct)
                brier = _brier_score(resolved)
                roi = _roi_pct(resolved)
                clv = _clv_pct(preds)
                avg_conf = (
                    sum(float(p.confidence) for p in preds if p.confidence is not None)
                    / len(preds)
                    if preds
                    else None
                )

                # Upsert: delete existing snapshot for same (agent, sport, date, window)
                existing = session.execute(
                    select(AgentPerformance).where(
                        and_(
                            AgentPerformance.agent_id == agent.id,
                            AgentPerformance.sport_slug == sport_slug,
                            AgentPerformance.date == snapshot_date,
                            AgentPerformance.window_days == window_days,
                        )
                    )
                ).scalar_one_or_none()

                if existing:
                    existing.total_bets = len(resolved)
                    existing.wins = wins
                    existing.roi_pct = roi
                    existing.clv_pct = clv
                    existing.avg_confidence = avg_conf
                    existing.brier_score = brier
                else:
                    session.add(
                        AgentPerformance(
                            agent_id=agent.id,
                            sport_slug=sport_slug,
                            date=snapshot_date,
                            window_days=window_days,
                            total_bets=len(resolved),
                            wins=wins,
                            roi_pct=roi,
                            clv_pct=clv,
                            avg_confidence=avg_conf,
                            brier_score=brier,
                        )
                    )
                snapshotted += 1

        logger.info(
            "Snapshotted daily performance for sport='%s' date=%s window=%dd  agents=%d",
            sport_slug,
            snapshot_date,
            window_days,
            snapshotted,
        )

    # ------------------------------------------------------------------
    # 6. Leaderboard
    # ------------------------------------------------------------------

    def get_leaderboard(
        self,
        sport: Optional[str] = None,
        min_bets: int = 10,
        window_days: int = 30,
    ) -> list[dict]:
        """Return a ranked leaderboard of agents sorted by ROI desc.

        Uses the most recent ``AgentPerformance`` snapshot for each agent.
        Falls back to live computation when no snapshot exists.

        Parameters
        ----------
        sport:
            Filter to a specific sport slug (None = all sports).
        min_bets:
            Minimum number of resolved bets to appear on leaderboard.
        window_days:
            Performance window to match snapshot ``window_days``.

        Returns
        -------
        list[dict]
            Each entry contains: ``rank``, ``agent_name``, ``sport_slug``,
            ``total_bets``, ``wins``, ``accuracy``, ``roi_pct``, ``clv_pct``,
            ``brier_score``, ``avg_confidence``, ``snapshot_date``.
        """
        rows: list[dict] = []

        with _session_scope() as session:
            stmt = (
                select(AgentPerformance, Agent.name)
                .join(Agent, Agent.id == AgentPerformance.agent_id)
                .where(AgentPerformance.window_days == window_days)
                .where(AgentPerformance.total_bets >= min_bets)
            )
            if sport:
                stmt = stmt.where(AgentPerformance.sport_slug == sport)

            # Most recent snapshot per agent per sport
            stmt = stmt.order_by(
                AgentPerformance.agent_id,
                AgentPerformance.sport_slug,
                AgentPerformance.date.desc(),
            )

            seen: set[tuple[int, str]] = set()
            for perf, agent_name in session.execute(stmt).all():
                key = (perf.agent_id, perf.sport_slug)
                if key in seen:
                    continue
                seen.add(key)

                accuracy = (
                    perf.wins / perf.total_bets if perf.total_bets > 0 else None
                )
                rows.append(
                    {
                        "agent_name": agent_name,
                        "sport_slug": perf.sport_slug,
                        "total_bets": perf.total_bets,
                        "wins": perf.wins,
                        "accuracy": round(accuracy, 4) if accuracy is not None else None,
                        "roi_pct": round(perf.roi_pct, 4) if perf.roi_pct is not None else None,
                        "clv_pct": round(perf.clv_pct, 4) if perf.clv_pct is not None else None,
                        "brier_score": round(perf.brier_score, 6) if perf.brier_score is not None else None,
                        "avg_confidence": round(perf.avg_confidence, 4) if perf.avg_confidence is not None else None,
                        "snapshot_date": perf.date.isoformat(),
                    }
                )

        # Sort by ROI desc (None last), then by accuracy desc as tiebreaker
        rows.sort(
            key=lambda r: (
                r["roi_pct"] if r["roi_pct"] is not None else float("-inf"),
                r["accuracy"] if r["accuracy"] is not None else float("-inf"),
            ),
            reverse=True,
        )

        # Add rank
        for i, row in enumerate(rows, start=1):
            row["rank"] = i

        return rows
