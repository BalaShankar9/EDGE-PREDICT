"""Agent Evolution Engine — the AI learning loop for SharpEdge.

Agents compete on real-money predictions, get evaluated on ROI / CLV / Brier,
and the system promotes winners, puts laggards on probation, and deprecates
consistent losers.  All decisions are logged to AgentEvolution + EdgeLog so
the audit trail is complete.

Public surface
--------------
AgentEvolutionEngine
    .evaluate_agents(sport_slug, window_days) -> list[AgentEvaluation]
    .discover_edges(sport_slug, window_days)  -> list[DiscoveredEdge]
    .run_natural_selection(sport_slug)         -> EvolutionReport
    .compute_specialization_matrix(sport_slug) -> dict[str, dict[str, float]]
    .suggest_agent_configs(sport_slug)         -> list[dict]
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import Integer, func

from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.models import (
    Agent,
    AgentEvolution,
    AgentPerformance,
    AgentPrediction as AgentPredictionRow,
    EdgeLog,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional scipy — fall back gracefully if not installed
# ---------------------------------------------------------------------------

try:
    from scipy.stats import binom_test as _scipy_binom_test  # type: ignore

    def _binom_p_value(successes: int, n: int, p0: float = 0.5) -> float:
        """Two-sided binomial p-value via scipy."""
        return float(_scipy_binom_test(successes, n, p0))

    _SCIPY_AVAILABLE = True

except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

    def _binom_p_value(successes: int, n: int, p0: float = 0.5) -> float:  # type: ignore[misc]
        """Two-sided z-test approximation when scipy is unavailable."""
        if n == 0:
            return 1.0
        p_hat = successes / n
        se = math.sqrt(p0 * (1 - p0) / n)
        if se == 0:
            return 1.0
        z = (p_hat - p0) / se
        # Abramowitz & Stegun rational approximation for the normal CDF tail
        abs_z = abs(z)
        t = 1.0 / (1.0 + 0.2316419 * abs_z)
        poly = t * (
            0.319381530
            + t * (-0.356563782
            + t * (1.781477937
            + t * (-1.821255978
            + t * 1.330274429)))
        )
        p_one_tail = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * abs_z ** 2) * poly
        return min(2.0 * p_one_tail, 1.0)

    logger.debug(
        "scipy not available — using z-test fallback for binomial significance tests"
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AgentEvaluation:
    """Full evaluation snapshot for a single agent within a sport."""

    agent_name: str
    total_bets: int
    wins: int
    win_rate: float
    roi_pct: float
    clv_pct: float
    brier_score: float
    rank: int  # 1 = best composite rank
    recommendation: str  # "promote" | "maintain" | "probation" | "deprecate"


@dataclass
class DiscoveredEdge:
    """A statistically significant (or noteworthy) edge found by an agent."""

    agent_name: str
    league: str
    market: str
    edge_type: str   # "roi" | "clv" | "accuracy"
    edge_value: float
    sample_size: int
    is_significant: bool  # p < 0.05 binomial test
    p_value: float = 1.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0


@dataclass
class EvolutionReport:
    """Summary of one full natural-selection cycle."""

    sport_slug: str
    run_date: date
    evaluations: list[AgentEvaluation] = field(default_factory=list)
    promotions: list[str] = field(default_factory=list)
    deprecations: list[str] = field(default_factory=list)
    probations: list[str] = field(default_factory=list)
    edges_discovered: list[DiscoveredEdge] = field(default_factory=list)
    arbiter_weights_updated: bool = False


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


def _rank_ascending(values: list[float]) -> list[int]:
    """Rank a list ascending (lower value = rank 1).  Handles ties with average rank."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0] * len(values)
    for rank_pos, (orig_idx, _) in enumerate(indexed, start=1):
        ranks[orig_idx] = rank_pos
    return ranks


def _rank_descending(values: list[float]) -> list[int]:
    """Rank a list descending (higher value = rank 1)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=True)
    ranks = [0] * len(values)
    for rank_pos, (orig_idx, _) in enumerate(indexed, start=1):
        ranks[orig_idx] = rank_pos
    return ranks


def _composite_score(
    roi_rank: int,
    clv_rank: int,
    brier_rank: int,
    vol_rank: int,
) -> float:
    """Lower composite score = better overall rank.  Weights from the spec."""
    return 0.4 * roi_rank + 0.3 * clv_rank + 0.2 * brier_rank + 0.1 * vol_rank


# ---------------------------------------------------------------------------
# Confidence interval helper (Wilson score)
# ---------------------------------------------------------------------------


def _wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    p_hat = wins / n
    center = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
    margin = (
        z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n)
        / (1 + z * z / n)
    )
    return max(0.0, center - margin), min(1.0, center + margin)


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class AgentEvolutionEngine:
    """Natural selection for prediction agents.

    Reads from AgentPerformance snapshots (written by the daily pipeline) and
    AgentPrediction rows (for granular league/market breakdowns).

    All mutations are persisted inside a single DB transaction per method so
    failures leave no partial state.
    """

    # Arbiter weight multipliers
    _PROMOTION_WEIGHT_BOOST: float = 1.25
    _PROBATION_WEIGHT_PENALTY: float = 0.80
    _DEPRECATION_WEIGHT: float = 0.0  # effectively removed from arbiter

    # Edge significance thresholds
    _EDGE_MIN_SAMPLES: int = 20
    _EDGE_SIGNIFICANCE_ALPHA: float = 0.05

    def __init__(self) -> None:
        self._min_bets: int = settings.agent_min_bets_for_eval
        self._deprecation_roi: float = settings.agent_deprecation_roi_threshold
        self._promotion_clv: float = settings.agent_promotion_clv_threshold
        self._eval_interval_days: int = settings.agent_evolution_interval_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_agents(
        self,
        sport_slug: str,
        window_days: int = 30,
    ) -> list[AgentEvaluation]:
        """Evaluate all active agents in a sport over the past *window_days*.

        Returns a ranked list of AgentEvaluation objects.  The composite rank
        uses:  0.4 × roi_rank + 0.3 × clv_rank + 0.2 × brier_rank + 0.1 × volume_rank

        Recommendations
        ---------------
        "promote"    CLV > promotion_threshold AND roi > 0
        "maintain"   roi >= 0  OR  insufficient data
        "probation"  roi < 0  AND  roi > deprecation_threshold
        "deprecate"  roi < deprecation_threshold AND total_bets >= min_bets
        """
        logger.info("evaluate_agents: sport=%s window=%d days", sport_slug, window_days)
        cutoff = date.today() - timedelta(days=window_days)

        session = get_session()
        try:
            # Most-recent performance snapshot per agent within the window
            subq = (
                session.query(
                    AgentPerformance.agent_id,
                    func.max(AgentPerformance.date).label("max_date"),
                )
                .filter(
                    AgentPerformance.sport_slug == sport_slug,
                    AgentPerformance.date >= cutoff,
                    AgentPerformance.window_days == window_days,
                )
                .group_by(AgentPerformance.agent_id)
                .subquery()
            )

            rows = (
                session.query(AgentPerformance, Agent)
                .join(Agent, Agent.id == AgentPerformance.agent_id)
                .join(
                    subq,
                    (subq.c.agent_id == AgentPerformance.agent_id)
                    & (subq.c.max_date == AgentPerformance.date),
                )
                .filter(
                    Agent.active.is_(True),
                    AgentPerformance.sport_slug == sport_slug,
                )
                .all()
            )
        finally:
            session.close()

        if not rows:
            logger.warning(
                "evaluate_agents: no AgentPerformance rows for sport=%s window=%d",
                sport_slug,
                window_days,
            )
            return []

        perfs, agents_ = zip(*rows)  # type: ignore[misc]

        # ----------------------------------------------------------------
        # Build ranking inputs — replace None with sentinel worst values
        # ----------------------------------------------------------------
        rois = [
            (p.roi_pct if p.roi_pct is not None else -999.0) for p in perfs
        ]
        clvs = [
            (p.clv_pct if p.clv_pct is not None else -999.0) for p in perfs
        ]
        briers = [
            (p.brier_score if p.brier_score is not None else 999.0)
            for p in perfs
        ]
        volumes = [p.total_bets for p in perfs]

        roi_ranks = _rank_descending(rois)
        clv_ranks = _rank_descending(clvs)
        brier_ranks = _rank_ascending(briers)   # lower Brier = better
        vol_ranks = _rank_descending(volumes)    # more volume = better signal

        composites = [
            _composite_score(
                roi_ranks[i], clv_ranks[i], brier_ranks[i], vol_ranks[i]
            )
            for i in range(len(perfs))
        ]
        final_ranks = _rank_ascending(composites)   # lower composite = rank 1

        evaluations: list[AgentEvaluation] = []
        for i, (perf, agent) in enumerate(zip(perfs, agents_)):
            roi = rois[i]
            clv = clvs[i]
            wins = perf.wins or 0
            total = perf.total_bets or 0
            win_rate = wins / total if total > 0 else 0.0
            brier = perf.brier_score if perf.brier_score is not None else 1.0

            evaluations.append(
                AgentEvaluation(
                    agent_name=agent.name,
                    total_bets=total,
                    wins=wins,
                    win_rate=round(win_rate, 4),
                    roi_pct=round(roi, 4),
                    clv_pct=round(clv, 4),
                    brier_score=round(brier, 4),
                    rank=final_ranks[i],
                    recommendation=self._recommend(roi, clv, total),
                )
            )

        evaluations.sort(key=lambda e: e.rank)
        logger.info(
            "evaluate_agents: %d agents evaluated for sport=%s",
            len(evaluations),
            sport_slug,
        )
        return evaluations

    # ------------------------------------------------------------------

    def discover_edges(
        self,
        sport_slug: str,
        window_days: int = 60,
    ) -> list[DiscoveredEdge]:
        """Find statistically significant edges per agent × league × market.

        Uses a two-sided binomial test (null: win_rate == 0.5) with p < 0.05.
        Significant edges are persisted to EdgeLog (upserted to avoid duplicates).

        Returns all cell-level discoveries so callers can apply their own
        significance filter.
        """
        logger.info("discover_edges: sport=%s window=%d days", sport_slug, window_days)
        cutoff = date.today() - timedelta(days=window_days)
        today = date.today()

        # ----------------------------------------------------------------
        # Pull resolved predictions, grouped by agent × league × market
        # We join AgentPrediction -> Agent and AgentPrediction -> Prediction
        # to get the league dimension (Prediction.league).
        # ----------------------------------------------------------------
        from sharpedge.db.models import Prediction  # avoid any potential circular

        session = get_session()
        try:
            raw_rows = (
                session.query(
                    Agent.name,
                    Prediction.league,
                    AgentPredictionRow.market,
                    func.count(AgentPredictionRow.id).label("total"),
                    func.sum(
                        func.cast(AgentPredictionRow.correct, Integer)
                    ).label("wins"),
                )
                .join(Agent, Agent.id == AgentPredictionRow.agent_id)
                .join(
                    Prediction,
                    (Prediction.home_team == AgentPredictionRow.home_team)
                    & (Prediction.away_team == AgentPredictionRow.away_team)
                    & (Prediction.match_date == AgentPredictionRow.match_date),
                )
                .filter(
                    AgentPredictionRow.sport_slug == sport_slug,
                    AgentPredictionRow.actual_outcome.isnot(None),
                    AgentPredictionRow.match_date >= cutoff,
                    Agent.active.is_(True),
                )
                .group_by(
                    Agent.name, Prediction.league, AgentPredictionRow.market
                )
                .having(
                    func.count(AgentPredictionRow.id) >= self._EDGE_MIN_SAMPLES
                )
                .all()
            )
        finally:
            session.close()

        discoveries: list[DiscoveredEdge] = []

        if not raw_rows:
            logger.info(
                "discover_edges: no qualifying rows (need >= %d samples per cell)",
                self._EDGE_MIN_SAMPLES,
            )
            return discoveries

        # ----------------------------------------------------------------
        # Compute significance and build DiscoveredEdge objects
        # ----------------------------------------------------------------
        significant: list[DiscoveredEdge] = []

        for row in raw_rows:
            agent_name, league, market, total, wins_raw = row
            n = int(total or 0)
            wins = int(wins_raw or 0)
            if n < self._EDGE_MIN_SAMPLES:
                continue

            win_rate = wins / n
            edge_value = round(win_rate - 0.5, 4)  # vs naive 50% baseline

            p_val = _binom_p_value(wins, n, p0=0.5)
            is_sig = p_val < self._EDGE_SIGNIFICANCE_ALPHA

            ci_lo, ci_hi = _wilson_ci(wins, n)

            edge = DiscoveredEdge(
                agent_name=agent_name,
                league=league or "unknown",
                market=market,
                edge_type="accuracy",
                edge_value=edge_value,
                sample_size=n,
                is_significant=is_sig,
                p_value=round(p_val, 4),
                ci_lo=round(ci_lo, 4),
                ci_hi=round(ci_hi, 4),
            )
            discoveries.append(edge)
            if is_sig:
                significant.append(edge)

        # ----------------------------------------------------------------
        # Persist significant edges to EdgeLog in a single transaction
        # ----------------------------------------------------------------
        if significant:
            session = get_session()
            try:
                for edge in significant:
                    self._upsert_edge_log(session, edge, sport_slug, today)
                session.commit()
                logger.info(
                    "discover_edges: %d edges discovered (%d significant) for sport=%s",
                    len(discoveries),
                    len(significant),
                    sport_slug,
                )
            except Exception:
                session.rollback()
                logger.exception(
                    "discover_edges: failed to persist EdgeLog — rolled back"
                )
                raise
            finally:
                session.close()

        return discoveries

    # ------------------------------------------------------------------

    def run_natural_selection(self, sport_slug: str) -> EvolutionReport:
        """One full evolution cycle for a sport.

        Steps
        -----
        1. Evaluate all agents
        2. Promote top performers (boost arbiter weight in config)
        3. Put under-performers on probation (reduce weight)
        4. Deprecate consistently poor agents (mark inactive, zero weight)
        5. Discover statistically significant edges
        6. Log every event to AgentEvolution
        7. Return EvolutionReport
        """
        today = date.today()
        logger.info("run_natural_selection: sport=%s date=%s", sport_slug, today)

        report = EvolutionReport(sport_slug=sport_slug, run_date=today)

        # ---- Step 1: Evaluate ----
        report.evaluations = self.evaluate_agents(sport_slug)
        if not report.evaluations:
            logger.warning("run_natural_selection: no evaluations — skipping cycle")
            return report

        # ---- Steps 2–4: Mutate agents + log to AgentEvolution ----
        session = get_session()
        try:
            for eval_ in report.evaluations:
                agent_row: Optional[Agent] = (
                    session.query(Agent)
                    .filter(Agent.name == eval_.agent_name)
                    .first()
                )
                if agent_row is None:
                    logger.warning(
                        "run_natural_selection: agent %s not found in DB",
                        eval_.agent_name,
                    )
                    continue

                config_before = dict(agent_row.config or {})

                if eval_.recommendation == "promote":
                    self._apply_promotion(agent_row)
                    report.promotions.append(eval_.agent_name)
                    event_type = "promoted"
                    reason = (
                        f"CLV={eval_.clv_pct:.2%} > threshold "
                        f"({self._promotion_clv:.2%}) and "
                        f"ROI={eval_.roi_pct:.2%} > 0"
                    )

                elif eval_.recommendation == "probation":
                    self._apply_probation(agent_row)
                    report.probations.append(eval_.agent_name)
                    event_type = "probation"
                    reason = (
                        f"ROI={eval_.roi_pct:.2%} < 0 but above "
                        f"deprecation threshold ({self._deprecation_roi:.2%})"
                    )

                elif eval_.recommendation == "deprecate":
                    self._apply_deprecation(agent_row)
                    report.deprecations.append(eval_.agent_name)
                    event_type = "deprecated"
                    reason = (
                        f"ROI={eval_.roi_pct:.2%} < deprecation threshold "
                        f"({self._deprecation_roi:.2%}) with "
                        f"{eval_.total_bets} bets"
                    )

                else:
                    # "maintain" — no config change, still audit-log
                    event_type = "evaluated"
                    reason = (
                        f"ROI={eval_.roi_pct:.2%} — no action required"
                    )

                evo_entry = AgentEvolution(
                    agent_name=eval_.agent_name,
                    event_type=event_type,
                    parent_agent=None,
                    event_date=today,
                    config_before=config_before,
                    config_after=dict(agent_row.config or {}),
                    reason=reason,
                    performance_at_event={
                        "total_bets": eval_.total_bets,
                        "wins": eval_.wins,
                        "win_rate": eval_.win_rate,
                        "roi_pct": eval_.roi_pct,
                        "clv_pct": eval_.clv_pct,
                        "brier_score": eval_.brier_score,
                        "rank": eval_.rank,
                    },
                )
                session.add(evo_entry)

            session.commit()
            report.arbiter_weights_updated = bool(
                report.promotions or report.deprecations or report.probations
            )
            logger.info(
                "run_natural_selection: promoted=%s deprecated=%s probation=%s",
                report.promotions,
                report.deprecations,
                report.probations,
            )
        except Exception:
            session.rollback()
            logger.exception(
                "run_natural_selection: DB transaction failed — rolled back"
            )
            raise
        finally:
            session.close()

        # ---- Step 5: Discover edges (non-fatal if it fails) ----
        try:
            report.edges_discovered = self.discover_edges(sport_slug)
        except Exception:
            logger.exception(
                "run_natural_selection: edge discovery failed — continuing"
            )

        logger.info(
            "run_natural_selection complete: sport=%s "
            "promotions=%d deprecations=%d edges=%d",
            sport_slug,
            len(report.promotions),
            len(report.deprecations),
            len(report.edges_discovered),
        )
        return report

    # ------------------------------------------------------------------

    def compute_specialization_matrix(
        self,
        sport_slug: str,
        window_days: int = 60,
    ) -> dict[str, dict[str, float]]:
        """Return ``{agent_name: {league: roi_pct}}`` for the past *window_days*.

        Shows where each agent has a genuine edge.  Only cells with at least
        _EDGE_MIN_SAMPLES resolved bets are included.

        The ROI value here is an accuracy-over-baseline proxy
        (win_rate − 0.5) — a positive number means the agent beats the naive
        50 % baseline in that league.
        """
        logger.info(
            "compute_specialization_matrix: sport=%s window=%d days",
            sport_slug,
            window_days,
        )
        cutoff = date.today() - timedelta(days=window_days)

        from sharpedge.db.models import Prediction  # avoid any potential circular

        session = get_session()
        try:
            raw_rows = (
                session.query(
                    Agent.name,
                    Prediction.league,
                    func.count(AgentPredictionRow.id).label("total"),
                    func.sum(
                        func.cast(AgentPredictionRow.correct, Integer)
                    ).label("wins"),
                )
                .join(Agent, Agent.id == AgentPredictionRow.agent_id)
                .join(
                    Prediction,
                    (Prediction.home_team == AgentPredictionRow.home_team)
                    & (Prediction.away_team == AgentPredictionRow.away_team)
                    & (Prediction.match_date == AgentPredictionRow.match_date),
                )
                .filter(
                    AgentPredictionRow.sport_slug == sport_slug,
                    AgentPredictionRow.actual_outcome.isnot(None),
                    AgentPredictionRow.match_date >= cutoff,
                    Agent.active.is_(True),
                )
                .group_by(Agent.name, Prediction.league)
                .having(
                    func.count(AgentPredictionRow.id) >= self._EDGE_MIN_SAMPLES
                )
                .all()
            )
        finally:
            session.close()

        matrix: dict[str, dict[str, float]] = {}
        for row in raw_rows:
            agent_name, league, total, wins_raw = row
            n = int(total or 0)
            w = int(wins_raw or 0)
            if n == 0:
                continue
            roi_proxy = round(w / n - 0.5, 4)
            league_key = league or "unknown"
            if agent_name not in matrix:
                matrix[agent_name] = {}
            matrix[agent_name][league_key] = roi_proxy

        logger.info(
            "compute_specialization_matrix: %d agent×league cells for sport=%s",
            sum(len(v) for v in matrix.values()),
            sport_slug,
        )
        return matrix

    # ------------------------------------------------------------------

    def suggest_agent_configs(self, sport_slug: str) -> list[dict]:
        """Suggest new agent configurations based on edge analysis.

        Logic
        -----
        1. Compute the specialization matrix to find strong league-specific edges
        2. For each agent with >= 5% excess win-rate in a league, suggest a
           league-focused variant config
        3. Agent-type tuning:
           - ContrarianAgent: lower fade_threshold in leagues where fading works
           - LeagueSpecialist: set primary_league
           - Form/Momentum agents: widen form_window in data-rich leagues
        4. For consistently weak leagues, suggest an exclusion list

        Returns a list of config dicts ready to be passed to ``Agent.config``.
        """
        logger.info("suggest_agent_configs: sport=%s", sport_slug)

        matrix = self.compute_specialization_matrix(sport_slug)
        if not matrix:
            logger.info("suggest_agent_configs: empty specialization matrix — no suggestions")
            return []

        # Pull current agent configs for context
        session = get_session()
        try:
            agent_rows: list[Agent] = (
                session.query(Agent).filter(Agent.active.is_(True)).all()
            )
            agent_meta: dict[str, tuple[str, dict]] = {
                a.name: (a.agent_type, dict(a.config or {}))
                for a in agent_rows
            }
        finally:
            session.close()

        suggestions: list[dict] = []

        for agent_name, league_rois in matrix.items():
            agent_type, current_config = agent_meta.get(
                agent_name, ("unknown", {})
            )

            strong_leagues = [
                (lg, roi)
                for lg, roi in league_rois.items()
                if roi >= 0.05  # >= 5% edge above 50% baseline
            ]
            weak_leagues = [
                (lg, roi)
                for lg, roi in league_rois.items()
                if roi <= -0.05
            ]

            # --- Variant suggestions for strong leagues ---
            for league, roi in strong_leagues:
                new_config: dict = {
                    **current_config,
                    "league_filter": [league],
                    "sport_slug": sport_slug,
                    "source_edge_roi": roi,
                }

                if "contrarian" in agent_name.lower():
                    # Tighter fade_threshold = higher selectivity in markets
                    # where going against heavy favourites works well
                    current_threshold = float(
                        current_config.get("fade_threshold", 0.65)
                    )
                    suggested_threshold = round(
                        max(0.55, current_threshold - 0.05), 2
                    )
                    new_config["fade_threshold"] = suggested_threshold
                    detail = (
                        f" Tighten fade_threshold "
                        f"{current_threshold} → {suggested_threshold}."
                    )

                elif "league_specialist" in agent_name.lower():
                    new_config["primary_league"] = league
                    detail = f" Set primary_league={league}."

                elif any(
                    kw in agent_name.lower()
                    for kw in ("form", "momentum", "gradient")
                ):
                    current_window = int(current_config.get("form_window", 5))
                    new_window = min(current_window + 2, 10)
                    new_config["form_window"] = new_window
                    detail = (
                        f" Extend form_window "
                        f"{current_window} → {new_window} for deeper data."
                    )

                else:
                    detail = ""

                suggestions.append(
                    {
                        "action": "spawn_variant",
                        "parent_agent": agent_name,
                        "agent_type": agent_type,
                        "reason": (
                            f"{agent_name} shows {roi:+.1%} edge in "
                            f"{league}. Suggest a league-focused variant."
                            + detail
                        ),
                        "config": new_config,
                    }
                )

            # --- Exclusion list for consistently weak leagues ---
            if weak_leagues and current_config.get("league_filter") is None:
                exclusion_leagues = [lg for lg, _ in weak_leagues]
                suggestions.append(
                    {
                        "action": "update_config",
                        "agent_name": agent_name,
                        "reason": (
                            f"{agent_name} underperforms in "
                            f"{', '.join(exclusion_leagues[:3])}"
                            + (" ..." if len(exclusion_leagues) > 3 else "")
                            + ". Suggest adding league_exclude list."
                        ),
                        "config": {
                            **current_config,
                            "league_exclude": exclusion_leagues,
                        },
                    }
                )

        logger.info(
            "suggest_agent_configs: %d suggestions generated for sport=%s",
            len(suggestions),
            sport_slug,
        )
        return suggestions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recommend(self, roi: float, clv: float, total_bets: int) -> str:
        """Classify an agent into one of four recommendation buckets."""
        if total_bets < self._min_bets:
            # Insufficient data — do not act
            return "maintain"

        if roi < self._deprecation_roi:
            return "deprecate"

        if roi < 0:
            return "probation"

        if clv >= self._promotion_clv and roi > 0:
            return "promote"

        return "maintain"

    def _apply_promotion(self, agent: Agent) -> None:
        """Boost the agent's arbiter weight in its config JSON."""
        cfg = dict(agent.config or {})
        current = float(cfg.get("arbiter_weight", 1.0))
        new_weight = round(min(current * self._PROMOTION_WEIGHT_BOOST, 3.0), 4)
        cfg["arbiter_weight"] = new_weight
        cfg["status"] = "promoted"
        agent.config = cfg
        logger.debug(
            "Promoted %s — arbiter_weight %.3f -> %.3f",
            agent.name,
            current,
            new_weight,
        )

    def _apply_probation(self, agent: Agent) -> None:
        """Reduce the agent's arbiter weight in its config JSON."""
        cfg = dict(agent.config or {})
        current = float(cfg.get("arbiter_weight", 1.0))
        new_weight = round(max(current * self._PROBATION_WEIGHT_PENALTY, 0.1), 4)
        cfg["arbiter_weight"] = new_weight
        cfg["status"] = "probation"
        agent.config = cfg
        logger.debug(
            "Probation %s — arbiter_weight %.3f -> %.3f",
            agent.name,
            current,
            new_weight,
        )

    def _apply_deprecation(self, agent: Agent) -> None:
        """Mark the agent inactive and zero out its arbiter weight."""
        cfg = dict(agent.config or {})
        cfg["arbiter_weight"] = self._DEPRECATION_WEIGHT
        cfg["status"] = "deprecated"
        agent.config = cfg
        agent.active = False
        logger.warning("Deprecated %s — marked inactive", agent.name)

    def _upsert_edge_log(
        self,
        session,
        edge: DiscoveredEdge,
        sport_slug: str,
        today: date,
    ) -> None:
        """Insert or refresh a significant edge in EdgeLog.

        If an open row (expired_date IS NULL) with the same composite key
        already exists, refresh its metrics.  Otherwise insert a new row.
        """
        existing: Optional[EdgeLog] = (
            session.query(EdgeLog)
            .filter(
                EdgeLog.agent_name == edge.agent_name,
                EdgeLog.sport_slug == sport_slug,
                EdgeLog.league == edge.league,
                EdgeLog.market == edge.market,
                EdgeLog.edge_type == edge.edge_type,
                EdgeLog.expired_date.is_(None),
            )
            .first()
        )

        if existing is not None:
            existing.edge_value = edge.edge_value
            existing.sample_size = edge.sample_size
            existing.confidence_interval_lo = edge.ci_lo
            existing.confidence_interval_hi = edge.ci_hi
            existing.is_significant = edge.is_significant
            existing.metadata_json = {"p_value": edge.p_value}
            logger.debug(
                "EdgeLog refreshed: %s / %s / %s",
                edge.agent_name,
                edge.league,
                edge.market,
            )
        else:
            log_row = EdgeLog(
                agent_name=edge.agent_name,
                sport_slug=sport_slug,
                league=edge.league,
                market=edge.market,
                edge_type=edge.edge_type,
                edge_value=edge.edge_value,
                sample_size=edge.sample_size,
                confidence_interval_lo=edge.ci_lo,
                confidence_interval_hi=edge.ci_hi,
                is_significant=edge.is_significant,
                discovered_date=today,
                metadata_json={"p_value": edge.p_value},
            )
            session.add(log_row)
            logger.info(
                "EdgeLog NEW: agent=%s league=%s market=%s edge=%.3f p=%.4f",
                edge.agent_name,
                edge.league,
                edge.market,
                edge.edge_value,
                edge.p_value,
            )
