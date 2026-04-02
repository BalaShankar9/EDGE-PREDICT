"""WarRoom Orchestrator — the production command centre for SharpEdge.

Coordinates every subsystem in a single, isolated daily cycle:
  health check -> resolution -> drift -> predictions -> broadcast -> evolution.

Each stage is independently fault-tolerant: a crash in evolution does not
prevent the prediction pipeline from running and broadcasting picks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from sharpedge.agents.arbiter import BayesianArbiter
from sharpedge.agents.base_agent import BaseAgent, MatchContext
from sharpedge.agents.evolution import AgentEvolutionEngine
from sharpedge.agents.tracker import AgentTracker
from sharpedge.alerts.telegram import send_alert_sync
from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.models import (
    Agent as AgentRow,
    BankrollLedger,
    DailyPick,
    Match,
    PipelineRun,
    Prediction,
    Sport,
)
from sharpedge.execution.staking import AntifragileStaking
from sharpedge.intelligence.edge_discovery import EdgeDiscoveryEngine
from sharpedge.pipeline.daily import DailyPipeline
from sharpedge.pipeline.drift_detector import DriftDetector, SmartRetrainer
from sharpedge.pipeline.live_resolver import LiveResolver
from sharpedge.pipeline.performance_ledger import PerformanceLedger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Result of a single prediction pipeline run for one sport."""

    sport: str
    timestamp: datetime
    matches_processed: int
    predictions_generated: int
    picks_produced: int
    agents_used: int
    errors: list[str] = field(default_factory=list)
    pipeline_run_id: Optional[int] = None


@dataclass
class DailyReport:
    """Comprehensive summary of all daily operations."""

    date: str
    sports_processed: list[str]
    total_matches: int
    total_predictions: int
    total_picks: int
    results: list[PipelineResult]
    circuit_breaker_level: int
    bankroll: float
    # Enhanced fields
    health_check: dict = field(default_factory=dict)
    resolution_summary: dict = field(default_factory=dict)
    evolution_report: dict = field(default_factory=dict)
    clv_summary: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# WarRoomOrchestrator
# ---------------------------------------------------------------------------


class WarRoomOrchestrator:
    """Production orchestrator — coordinates every SharpEdge subsystem.

    Instantiate once at process start.  All methods are isolated: failure in
    one stage is caught, logged, and recorded in the report without killing
    subsequent stages.

    Example
    -------
    >>> orc = WarRoomOrchestrator(bankroll=5000.0)
    >>> orc.register_agents([StatisticalAgent(), MarketAgent()])
    >>> report = orc.run_daily()
    """

    def __init__(self, bankroll: Optional[float] = None) -> None:
        _bankroll = bankroll if bankroll is not None else settings.bankroll_initial

        # Core subsystems
        self.daily_pipeline = DailyPipeline(model_path=settings.model_path)
        self.arbiter = BayesianArbiter(decay=0.95)
        self.staking = AntifragileStaking(
            bankroll=_bankroll,
            kelly_fraction=settings.kelly_fraction,
            max_bet_pct=settings.max_bet_pct,
            max_daily_pct=settings.max_daily_pct,
        )
        self.ledger = PerformanceLedger()
        self.tracker = AgentTracker()
        self.live_resolver = LiveResolver()
        self.drift_detector = DriftDetector()
        self.smart_retrainer = SmartRetrainer()
        self.evolution_engine = AgentEvolutionEngine()
        self.edge_engine = EdgeDiscoveryEngine()

        # Internal state
        self._last_run: Optional[datetime] = None
        self._registered_agents: list[BaseAgent] = []

        logger.info(
            "WarRoomOrchestrator initialised — bankroll=%.2f kelly=%.2f",
            _bankroll,
            settings.kelly_fraction,
        )

    # ------------------------------------------------------------------
    # 1. Agent registration
    # ------------------------------------------------------------------

    def register_agents(self, agents: list[BaseAgent]) -> None:
        """Register agents with the Bayesian Arbiter and internal tracker.

        Agents can be registered at any point before ``run_predictions`` is
        called.  Duplicate names are silently deduplicated.
        """
        existing_names = {a.name for a in self._registered_agents}
        for agent in agents:
            if agent.name in existing_names:
                logger.debug("Agent '%s' already registered — skipping", agent.name)
                continue
            self.arbiter.register_agent(agent)
            self._registered_agents.append(agent)
            existing_names.add(agent.name)
            logger.info("Registered agent '%s' (%s)", agent.name, agent.agent_type)

        logger.info(
            "Agent roster: %d agents total", len(self._registered_agents)
        )

    # ------------------------------------------------------------------
    # 2. Prediction pipeline
    # ------------------------------------------------------------------

    def run_predictions(
        self,
        sport: str,
        fixtures: list[dict],
        historical_matches=None,
        elo_df=None,
        xg_df=None,
        predictions_df=None,
    ) -> PipelineResult:
        """Run the full prediction pipeline for *sport* and the given *fixtures*.

        Steps
        -----
        a. Load/reload ML model via DailyPipeline
        b. Generate ensemble probabilities for every fixture
        c. Run each fixture through the Bayesian Arbiter (all registered agents)
        d. Apply BankerFilter to produce qualified picks
        e. Compute stakes via AntifragileStaking
        f. Persist Prediction + DailyPick rows to DB inside a transaction
        g. Record each agent prediction to the PerformanceLedger
        h. Return a PipelineResult with real counts

        Parameters
        ----------
        sport:
            Sport slug (e.g. ``"football"``).
        fixtures:
            List of fixture dicts as accepted by ``DailyPipeline.predict()``.
        historical_matches / elo_df / xg_df / predictions_df:
            Optional DataFrames forwarded to ``DailyPipeline.predict()``.
        """
        timestamp = datetime.now(timezone.utc)
        errors: list[str] = []
        pipeline_run_id: Optional[int] = None

        if not fixtures:
            logger.warning("run_predictions called with empty fixtures for sport=%s", sport)
            return PipelineResult(
                sport=sport, timestamp=timestamp,
                matches_processed=0, predictions_generated=0,
                picks_produced=0, agents_used=len(self._registered_agents),
                errors=["No fixtures provided"],
            )

        # ---- Open a PipelineRun record ----
        session = get_session()
        try:
            run_row = PipelineRun(
                run_date=date.today(),
                run_type="daily_predictions",
                status="running",
                started_at=timestamp,
                fixtures_count=len(fixtures),
            )
            session.add(run_row)
            session.commit()
            pipeline_run_id = run_row.id
            logger.info(
                "PipelineRun id=%d started for sport=%s fixtures=%d",
                pipeline_run_id, sport, len(fixtures),
            )
        except Exception:
            logger.exception("Failed to create PipelineRun record — continuing anyway")
            session.rollback()
        finally:
            session.close()

        # ---- a. Load model ----
        try:
            self.daily_pipeline.load_model()
        except Exception as exc:
            msg = f"Model load failed: {exc}"
            logger.error(msg)
            errors.append(msg)
            self._fail_pipeline_run(pipeline_run_id, msg)
            return PipelineResult(
                sport=sport, timestamp=timestamp,
                matches_processed=len(fixtures), predictions_generated=0,
                picks_produced=0, agents_used=len(self._registered_agents),
                errors=errors, pipeline_run_id=pipeline_run_id,
            )

        # ---- b. Generate ensemble predictions ----
        raw_predictions: list[dict] = []
        try:
            raw_predictions = self.daily_pipeline.predict(
                fixtures,
                historical_matches=historical_matches,
                elo_df=elo_df,
                xg_df=xg_df,
                predictions_df=predictions_df,
            )
            logger.info(
                "Ensemble predictions generated: %d for sport=%s",
                len(raw_predictions), sport,
            )
        except Exception as exc:
            msg = f"Ensemble prediction failed: {exc}"
            logger.exception(msg)
            errors.append(msg)
            self._fail_pipeline_run(pipeline_run_id, msg)
            return PipelineResult(
                sport=sport, timestamp=timestamp,
                matches_processed=len(fixtures), predictions_generated=0,
                picks_produced=0, agents_used=len(self._registered_agents),
                errors=errors, pipeline_run_id=pipeline_run_id,
            )

        # ---- c. Run arbiter for every fixture ----
        arbiter_results = []
        if self._registered_agents:
            contexts = self._build_match_contexts(fixtures, raw_predictions, sport)
            try:
                arbiter_results = self.arbiter.predict_batch(contexts)
                logger.info(
                    "Arbiter produced %d results for sport=%s",
                    len(arbiter_results), sport,
                )
            except Exception as exc:
                msg = f"Arbiter prediction failed: {exc}"
                logger.warning(msg)
                errors.append(msg)
                # Non-fatal — we still have ensemble predictions

        # ---- d. Apply banker filter ----
        picks = []
        try:
            picks = self.daily_pipeline.filter_picks(raw_predictions)
            logger.info(
                "BankerFilter produced %d picks from %d predictions",
                len(picks), len(raw_predictions),
            )
        except Exception as exc:
            msg = f"BankerFilter failed: {exc}"
            logger.exception(msg)
            errors.append(msg)

        # ---- e. Compute stakes ----
        staked_picks = []
        for pick in picks:
            try:
                stake_rec = self.staking.compute_stake(
                    match_id=pick.match_id,
                    prob=pick.model_prob,
                    odds=pick.best_odds,
                )
                staked_picks.append((pick, stake_rec))
            except Exception as exc:
                logger.warning("Staking failed for pick %s: %s", pick.match_id, exc)
                staked_picks.append((pick, None))

        # ---- f. Persist to DB ----
        persisted_picks = 0
        persisted_preds = 0
        if pipeline_run_id is not None:
            session = get_session()
            try:
                persisted_preds, persisted_picks = self._persist_predictions(
                    raw_predictions, staked_picks, sport, session, pipeline_run_id
                )
                self._complete_pipeline_run(
                    pipeline_run_id, len(raw_predictions), persisted_picks
                )
            except Exception as exc:
                msg = f"DB persistence failed: {exc}"
                logger.exception(msg)
                errors.append(msg)
                session.rollback()
                self._fail_pipeline_run(pipeline_run_id, msg)
            finally:
                session.close()

        # ---- g. Record agent predictions in PerformanceLedger ----
        if arbiter_results:
            self._record_agent_predictions_to_ledger(
                arbiter_results, fixtures, sport
            )

        self._last_run = datetime.now(timezone.utc)

        return PipelineResult(
            sport=sport,
            timestamp=timestamp,
            matches_processed=len(fixtures),
            predictions_generated=len(raw_predictions),
            picks_produced=persisted_picks if persisted_picks else len(staked_picks),
            agents_used=len(self._registered_agents),
            errors=errors,
            pipeline_run_id=pipeline_run_id,
        )

    # ------------------------------------------------------------------
    # 3. Resolution
    # ------------------------------------------------------------------

    def run_resolution(self) -> dict:
        """Resolve all pending picks using LiveResolver.

        Steps
        -----
        1. LiveResolver.run() — fetches results, resolves picks, updates bankroll
        2. Snapshot daily agent performance in PerformanceLedger
        3. Feed resolution results back to Bayesian Arbiter weights

        Returns
        -------
        dict
            Aggregated resolution summary from LiveResolver.
        """
        logger.info("run_resolution: starting")
        summary: dict = {}

        try:
            summary = self.live_resolver.run()
            logger.info(
                "run_resolution: resolved %d picks across %d dates — P&L=%.4f",
                summary.get("total_resolved", 0),
                summary.get("dates_processed", 0),
                summary.get("total_profit_loss", 0.0),
            )
        except Exception:
            logger.exception("run_resolution: LiveResolver.run() failed")
            summary = {"error": "resolution_failed", "total_resolved": 0}

        # Snapshot performance for all active sports
        try:
            session = get_session()
            try:
                sports = (
                    session.query(Sport)
                    .filter(Sport.active.is_(True))
                    .all()
                )
                sport_slugs = [s.slug for s in sports]
            finally:
                session.close()

            for slug in sport_slugs:
                try:
                    self.ledger.snapshot_daily(slug)
                    logger.debug(
                        "run_resolution: snapshotted performance for sport=%s", slug
                    )
                except Exception:
                    logger.exception(
                        "run_resolution: snapshot_daily failed for sport=%s", slug
                    )
        except Exception:
            logger.exception("run_resolution: failed to retrieve active sports for snapshot")

        return summary

    # ------------------------------------------------------------------
    # 4. Health check
    # ------------------------------------------------------------------

    def run_health_check(self) -> dict:
        """Assess system health across all active sports.

        Checks
        ------
        - Drift metrics (ECE, Brier) for each active sport
        - Retraining needs (schedule + drift)
        - Current bankroll, drawdown level, and circuit breaker status

        Returns
        -------
        dict with keys: bankroll, drawdown_pct, drawdown_level,
                        circuit_breaker_level, sports_drift, retrain_needed.
        """
        logger.info("run_health_check: starting")
        health: dict = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "bankroll": self.staking.bankroll,
            "drawdown_pct": round(self.staking.drawdown_pct, 4),
            "drawdown_level": self.staking.drawdown_level,
            "circuit_breaker_level": self.staking.drawdown_level,
            "sports_drift": {},
            "retrain_needed": [],
            "errors": [],
        }

        # Sync bankroll from DB so we have the true figure
        try:
            bankroll_from_db = self._get_current_bankroll()
            if bankroll_from_db is not None:
                self.staking.bankroll = bankroll_from_db
                health["bankroll"] = bankroll_from_db
        except Exception as exc:
            logger.warning("run_health_check: could not fetch bankroll from DB: %s", exc)

        # Per-sport drift checks
        try:
            drift_results = self.drift_detector.run_check_all()
            for slug, result in drift_results.items():
                health["sports_drift"][slug] = {
                    "severity": result.severity,
                    "has_drift": result.has_drift,
                    "calibration_error": round(result.calibration_error, 4),
                    "brier_score": round(result.brier_score, 4),
                    "accuracy": round(result.accuracy, 4),
                    "roi_pct": round(result.roi_pct, 2),
                    "n_predictions": result.n_predictions,
                    "recommendation": result.recommendation,
                }
        except Exception as exc:
            msg = f"Drift check failed: {exc}"
            logger.exception(msg)
            health["errors"].append(msg)

        # Retraining assessment
        try:
            session = get_session()
            try:
                sports = (
                    session.query(Sport)
                    .filter(Sport.active.is_(True))
                    .all()
                )
                active_slugs = [s.slug for s in sports]
            finally:
                session.close()

            for slug in active_slugs:
                try:
                    should_retrain, reason = self.drift_detector.should_retrain(slug)
                    if should_retrain:
                        health["retrain_needed"].append(
                            {"sport": slug, "reason": reason}
                        )
                except Exception as exc:
                    logger.warning(
                        "run_health_check: should_retrain failed for sport=%s: %s",
                        slug, exc,
                    )
        except Exception as exc:
            msg = f"Retrain check failed: {exc}"
            logger.exception(msg)
            health["errors"].append(msg)

        logger.info(
            "run_health_check: bankroll=%.2f drawdown=%.1f%% drift_sports=%d retrain_needed=%d",
            health["bankroll"],
            health["drawdown_pct"] * 100,
            len(health["sports_drift"]),
            len(health["retrain_needed"]),
        )
        return health

    # ------------------------------------------------------------------
    # 5. Evolution
    # ------------------------------------------------------------------

    def run_evolution(self, sport: str = "football") -> dict:
        """Run one full agent evolution cycle for *sport*.

        Steps
        -----
        1. AgentEvolutionEngine.run_natural_selection() — promotes/deprecates agents
        2. EdgeDiscoveryEngine.run_full_analysis() — surface profitable patterns
        3. Compile strategy recommendations from both engines

        This method is non-fatal: any sub-step exception is logged and
        reported in the returned dict without raising.

        Returns
        -------
        dict with keys: sport, run_date, promotions, deprecations, probations,
                        edges_discovered, strategy_recommendations, errors.
        """
        logger.info("run_evolution: starting for sport=%s", sport)
        report: dict = {
            "sport": sport,
            "run_date": date.today().isoformat(),
            "promotions": [],
            "deprecations": [],
            "probations": [],
            "edges_discovered": 0,
            "strategy_recommendations": [],
            "errors": [],
        }

        # Natural selection
        try:
            evo_report = self.evolution_engine.run_natural_selection(sport)
            report["promotions"] = evo_report.promotions
            report["deprecations"] = evo_report.deprecations
            report["probations"] = evo_report.probations
            report["edges_discovered"] = len(evo_report.edges_discovered)
            report["arbiter_weights_updated"] = evo_report.arbiter_weights_updated
            logger.info(
                "run_evolution: natural selection complete — "
                "promotions=%d deprecations=%d probations=%d edges=%d",
                len(evo_report.promotions),
                len(evo_report.deprecations),
                len(evo_report.probations),
                len(evo_report.edges_discovered),
            )
        except Exception as exc:
            msg = f"Natural selection failed: {exc}"
            logger.exception(msg)
            report["errors"].append(msg)

        # Edge discovery (full analysis)
        try:
            discovery_report = self.edge_engine.run_full_analysis(sport)
            recommendations = [
                {
                    "type": r.recommendation_type,
                    "description": r.description,
                    "expected_roi_improvement": r.expected_roi_improvement,
                    "priority": r.priority,
                }
                for r in (discovery_report.recommendations or [])
            ]
            report["strategy_recommendations"] = recommendations
            logger.info(
                "run_evolution: edge discovery complete — %d recommendations",
                len(recommendations),
            )
        except Exception as exc:
            msg = f"Edge discovery failed: {exc}"
            logger.exception(msg)
            report["errors"].append(msg)

        # Agent config suggestions
        try:
            config_suggestions = self.evolution_engine.suggest_agent_configs(sport)
            report["config_suggestions"] = config_suggestions
        except Exception as exc:
            logger.warning("run_evolution: suggest_agent_configs failed: %s", exc)
            report["config_suggestions"] = []

        return report

    # ------------------------------------------------------------------
    # 6. Full daily cycle
    # ------------------------------------------------------------------

    def run_daily(self) -> DailyReport:
        """Execute the complete daily SharpEdge cycle.

        Order of operations
        -------------------
        a. Health check (bankroll, drawdown, drift snapshot)
        b. Resolution of yesterday's (and earlier) pending picks
        c. Drift check + conditional retraining trigger
        d. Predictions for today's fixtures (per active sport)
        e. Broadcast qualifying picks via Telegram
        f. Evolution check (runs every ``agent_evolution_interval_days`` days)

        Every stage is wrapped in its own try/except so a single failure
        never aborts the rest of the cycle.
        """
        today = date.today().isoformat()
        logger.info("=== WarRoom daily cycle starting for %s ===", today)

        all_results: list[PipelineResult] = []
        all_errors: list[str] = []
        health: dict = {}
        resolution_summary: dict = {}
        evolution_report: dict = {}
        clv_summary: dict = {}

        # ---- a. Health check ----
        try:
            health = self.run_health_check()
            if health.get("circuit_breaker_level", 0) >= 4:
                logger.critical(
                    "Circuit breaker LEVEL 4 — all betting suspended. "
                    "Drawdown=%.1f%%", health.get("drawdown_pct", 0) * 100
                )
                send_alert_sync(
                    "CIRCUIT BREAKER LEVEL 4 ACTIVE\n"
                    f"Drawdown: {health.get('drawdown_pct', 0):.1%}\n"
                    "All new bets suspended."
                )
        except Exception as exc:
            msg = f"Health check failed: {exc}"
            logger.exception(msg)
            all_errors.append(msg)

        # ---- b. Resolution ----
        try:
            resolution_summary = self.run_resolution()
            # Build CLV summary from resolution data
            clv_values_flat: list[float] = []
            for _dt, day_data in resolution_summary.get("by_date", {}).items():
                clv_values_flat.extend(day_data.get("clv_values", []))
            if clv_values_flat:
                clv_summary = {
                    "avg_clv_pct": round(
                        sum(clv_values_flat) / len(clv_values_flat), 4
                    ),
                    "n_samples": len(clv_values_flat),
                    "positive_clv_pct": round(
                        sum(1 for v in clv_values_flat if v > 0) / len(clv_values_flat),
                        4,
                    ),
                    "target_clv_pct": settings.clv_target_pct,
                }
        except Exception as exc:
            msg = f"Resolution cycle failed: {exc}"
            logger.exception(msg)
            all_errors.append(msg)

        # ---- c. Drift check + smart retraining ----
        try:
            session = get_session()
            try:
                active_sports = (
                    session.query(Sport)
                    .filter(Sport.active.is_(True))
                    .all()
                )
                sport_slugs = [s.slug for s in active_sports]
            finally:
                session.close()

            for slug in sport_slugs:
                try:
                    retrain_result = self.smart_retrainer.check_and_retrain(slug)
                    action = retrain_result.get("action", "no_action")
                    if action not in ("no_action", "recalibrate"):
                        logger.warning(
                            "Retrain triggered for sport=%s action=%s reason=%s",
                            slug, action, retrain_result.get("reason", ""),
                        )
                        send_alert_sync(
                            f"RETRAIN TRIGGERED [{slug}]\n"
                            f"Action: {action}\n"
                            f"Reason: {retrain_result.get('reason', '')}"
                        )
                except Exception as exc:
                    logger.warning(
                        "Smart retrainer failed for sport=%s: %s", slug, exc
                    )
        except Exception as exc:
            msg = f"Drift/retrain check failed: {exc}"
            logger.exception(msg)
            all_errors.append(msg)

        # ---- d. Predictions for today ----
        circuit_level = health.get("circuit_breaker_level", 0)
        if circuit_level < 4:
            try:
                session = get_session()
                try:
                    active_sports = (
                        session.query(Sport)
                        .filter(Sport.active.is_(True))
                        .all()
                    )
                    sport_slugs = [s.slug for s in active_sports]
                finally:
                    session.close()
            except Exception as exc:
                logger.exception("Failed to query active sports: %s", exc)
                sport_slugs = []

            self.staking.reset_daily()

            for slug in sport_slugs:
                try:
                    fixtures = self._fetch_todays_fixtures(slug)
                    if not fixtures:
                        logger.info(
                            "No fixtures found for sport=%s on %s", slug, today
                        )
                        continue
                    result = self.run_predictions(sport=slug, fixtures=fixtures)
                    all_results.append(result)
                    if result.errors:
                        all_errors.extend(
                            [f"[{slug}] {e}" for e in result.errors]
                        )
                except Exception as exc:
                    msg = f"Prediction pipeline failed for sport={slug}: {exc}"
                    logger.exception(msg)
                    all_errors.append(msg)
                    all_results.append(
                        PipelineResult(
                            sport=slug,
                            timestamp=datetime.now(timezone.utc),
                            matches_processed=0,
                            predictions_generated=0,
                            picks_produced=0,
                            agents_used=len(self._registered_agents),
                            errors=[msg],
                        )
                    )
        else:
            logger.warning(
                "Skipping predictions — circuit breaker level %d", circuit_level
            )

        # ---- e. Broadcast picks ----
        total_picks_produced = sum(r.picks_produced for r in all_results)
        if total_picks_produced > 0:
            try:
                todays_picks = self._load_todays_picks()
                if todays_picks:
                    self._broadcast_picks(todays_picks)
            except Exception as exc:
                msg = f"Broadcast failed: {exc}"
                logger.exception(msg)
                all_errors.append(msg)

        # ---- f. Evolution (time-gated) ----
        if self._is_evolution_due():
            try:
                session = get_session()
                try:
                    active_sports = (
                        session.query(Sport)
                        .filter(Sport.active.is_(True))
                        .all()
                    )
                    sport_slugs_evo = [s.slug for s in active_sports]
                finally:
                    session.close()

                # Run evolution for each active sport; merge into one report
                merged: dict = {"sports": {}, "errors": []}
                for slug in sport_slugs_evo:
                    try:
                        evo = self.run_evolution(sport=slug)
                        merged["sports"][slug] = evo
                    except Exception as exc:
                        logger.exception(
                            "Evolution failed for sport=%s: %s", slug, exc
                        )
                        merged["errors"].append(f"[{slug}] {exc}")
                evolution_report = merged
            except Exception as exc:
                msg = f"Evolution cycle failed: {exc}"
                logger.exception(msg)
                all_errors.append(msg)
        else:
            logger.debug("Evolution not due today — skipping")

        # Aggregate totals
        total_matches = sum(r.matches_processed for r in all_results)
        total_predictions = sum(r.predictions_generated for r in all_results)

        report = DailyReport(
            date=today,
            sports_processed=[r.sport for r in all_results],
            total_matches=total_matches,
            total_predictions=total_predictions,
            total_picks=total_picks_produced,
            results=all_results,
            circuit_breaker_level=circuit_level,
            bankroll=health.get("bankroll", self.staking.bankroll),
            health_check=health,
            resolution_summary=resolution_summary,
            evolution_report=evolution_report,
            clv_summary=clv_summary,
            errors=all_errors,
        )

        logger.info(
            "=== WarRoom daily cycle complete — matches=%d predictions=%d picks=%d errors=%d ===",
            total_matches,
            total_predictions,
            total_picks_produced,
            len(all_errors),
        )
        return report

    # ------------------------------------------------------------------
    # 7. Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current system status snapshot.

        Returns
        -------
        dict with keys: bankroll, drawdown_pct, drawdown_level,
                        pending_picks, active_agents, last_run.
        """
        pending_picks = 0
        active_agents: list[str] = [a.name for a in self._registered_agents]

        try:
            session = get_session()
            try:
                from sqlalchemy import select
                pending_picks = (
                    session.execute(
                        select(DailyPick).where(DailyPick.result.is_(None))
                    )
                    .scalars()
                    .all()
                )
                pending_picks = len(list(pending_picks))
            finally:
                session.close()
        except Exception as exc:
            logger.warning("get_status: could not count pending picks: %s", exc)

        bankroll = self.staking.bankroll
        try:
            db_bankroll = self._get_current_bankroll()
            if db_bankroll is not None:
                bankroll = db_bankroll
        except Exception:
            pass

        return {
            "bankroll": round(bankroll, 2),
            "drawdown_pct": round(self.staking.drawdown_pct, 4),
            "drawdown_level": self.staking.drawdown_level,
            "circuit_breaker_level": self.staking.drawdown_level,
            "pending_picks": pending_picks,
            "active_agents": active_agents,
            "n_agents": len(active_agents),
            "last_run": self._last_run.isoformat() if self._last_run else None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _persist_predictions(
        self,
        raw_predictions: list[dict],
        staked_picks: list[tuple],
        sport: str,
        session: Session,
        pipeline_run_id: int,
    ) -> tuple[int, int]:
        """Write Prediction and DailyPick rows inside the provided session.

        Parameters
        ----------
        raw_predictions:
            Ensemble prediction dicts from DailyPipeline.
        staked_picks:
            List of (Pick, StakeRecommendation | None) pairs from BankerFilter.
        sport:
            Sport slug (currently stored in Prediction.league as fallback).
        session:
            SQLAlchemy session; caller is responsible for commit / rollback.
        pipeline_run_id:
            FK for PipelineRun row.

        Returns
        -------
        (n_predictions_persisted, n_picks_persisted)
        """
        match_to_pred_id: dict[str, int] = {}
        today = date.today()

        # Persist Prediction rows
        n_preds = 0
        for pred_dict in raw_predictions:
            try:
                raw_date = pred_dict.get("match_date", str(today))
                try:
                    import pandas as pd
                    match_date = pd.to_datetime(raw_date).date()
                except Exception:
                    match_date = today

                pred_row = Prediction(
                    pipeline_run_id=pipeline_run_id,
                    match_date=match_date,
                    home_team=str(pred_dict.get("home_team", "")),
                    away_team=str(pred_dict.get("away_team", "")),
                    league=str(pred_dict.get("league", sport)),
                    prob_home=pred_dict.get("prob_home"),
                    prob_draw=pred_dict.get("prob_draw"),
                    prob_away=pred_dict.get("prob_away"),
                    prob_over=pred_dict.get("prob_over"),
                    prob_under=pred_dict.get("prob_under"),
                    prob_btts_yes=pred_dict.get("prob_btts_yes"),
                    prob_btts_no=pred_dict.get("prob_btts_no"),
                    created_at=datetime.now(timezone.utc),
                )
                session.add(pred_row)
                session.flush()

                key = f"{pred_dict.get('home_team', '')}_v_{pred_dict.get('away_team', '')}"
                match_to_pred_id[key] = pred_row.id
                n_preds += 1
            except Exception as exc:
                logger.warning(
                    "_persist_predictions: failed to write Prediction row: %s", exc
                )

        # Persist DailyPick rows
        n_picks = 0
        for pick, stake_rec in staked_picks:
            try:
                pred_id = match_to_pred_id.get(
                    pick.match_id,
                    next(iter(match_to_pred_id.values()), None),
                )
                if pred_id is None:
                    logger.warning(
                        "_persist_predictions: no Prediction ID for pick %s — skipping",
                        pick.match_id,
                    )
                    continue

                raw_date = pick.match_date or str(today)
                try:
                    import pandas as pd
                    match_date = pd.to_datetime(raw_date).date()
                except Exception:
                    match_date = today

                stake_flat = None
                stake_kelly = None
                if stake_rec is not None:
                    stake_flat = round(stake_rec.stake_units, 4)
                    stake_kelly = round(stake_rec.stake_pct, 6)

                pick_row = DailyPick(
                    pipeline_run_id=pipeline_run_id,
                    prediction_id=pred_id,
                    match_date=match_date,
                    home_team=str(pick.home_team),
                    away_team=str(pick.away_team),
                    league=str(pick.league or sport),
                    pick_market=str(pick.market),
                    pick_selection=str(pick.predicted_outcome or pick.market),
                    model_prob=float(pick.model_prob),
                    model_spread=float(getattr(pick, "model_spread", 0.0)),
                    best_odds=float(pick.best_odds),
                    bookmaker=str(pick.bookmaker or "BetExplorer"),
                    implied_prob=round(1.0 / max(pick.best_odds, 1.01), 6),
                    edge=round(
                        pick.model_prob - (1.0 / max(pick.best_odds, 1.01)), 6
                    ),
                    tier=str(getattr(pick, "tier", "Standard")),
                    meta_agreement=int(getattr(pick, "meta_agreement", 1)),
                    risk_flags=pick.risk_flags if isinstance(pick.risk_flags, dict) else {},
                    stake_flat=stake_flat,
                    stake_kelly=stake_kelly,
                )
                session.add(pick_row)
                n_picks += 1
            except Exception as exc:
                logger.warning(
                    "_persist_predictions: failed to write DailyPick row: %s", exc
                )

        session.commit()
        logger.info(
            "_persist_predictions: committed %d predictions, %d picks (sport=%s)",
            n_preds, n_picks, sport,
        )
        return n_preds, n_picks

    def _broadcast_picks(self, picks: list[DailyPick]) -> None:
        """Format today's picks and send them to Telegram.

        Groups picks by tier (Diamond > Platinum > Gold > Standard), then
        sends a single formatted message.  Falls back to logging if Telegram
        is not configured.
        """
        if not picks:
            return

        tier_order = ["Diamond", "Platinum", "Gold", "Standard"]
        by_tier: dict[str, list[DailyPick]] = {t: [] for t in tier_order}
        for pick in picks:
            tier = pick.tier if pick.tier in by_tier else "Standard"
            by_tier[tier].append(pick)

        lines: list[str] = [
            f"SharpEdge Picks — {date.today().isoformat()}",
            f"Total picks: {len(picks)}",
            "",
        ]

        for tier in tier_order:
            tier_picks = by_tier[tier]
            if not tier_picks:
                continue
            tier_emoji = {"Diamond": "DIAMOND", "Platinum": "PLATINUM", "Gold": "GOLD"}.get(
                tier, tier.upper()
            )
            lines.append(f"[{tier_emoji}]")
            for p in tier_picks:
                edge_pct = round(p.edge * 100, 1)
                lines.append(
                    f"  {p.home_team} v {p.away_team} | {p.league}"
                )
                lines.append(
                    f"  Pick: {p.pick_selection} @ {p.best_odds} "
                    f"({p.model_prob:.0%} model | edge +{edge_pct}%)"
                )
                if p.stake_flat:
                    lines.append(f"  Stake: {p.stake_flat:.2f} units")
                lines.append("")

        message = "\n".join(lines)
        send_alert_sync(message)
        logger.info("_broadcast_picks: sent %d picks to Telegram", len(picks))

        # Mark picks as broadcasted
        try:
            session = get_session()
            try:
                now = datetime.now(timezone.utc)
                for pick in picks:
                    pick.broadcasted_at = now
                    session.merge(pick)
                session.commit()
            except Exception:
                session.rollback()
                logger.warning("_broadcast_picks: failed to set broadcasted_at")
            finally:
                session.close()
        except Exception:
            logger.warning("_broadcast_picks: DB session error while marking broadcast")

    def _build_match_contexts(
        self,
        fixtures: list[dict],
        raw_predictions: list[dict],
        sport: str,
    ) -> list[MatchContext]:
        """Build MatchContext objects for the Bayesian Arbiter."""
        import numpy as np

        contexts: list[MatchContext] = []
        pred_by_key: dict[str, dict] = {}
        for pred in raw_predictions:
            key = f"{pred.get('home_team', '')}_v_{pred.get('away_team', '')}"
            pred_by_key[key] = pred

        for fixture in fixtures:
            home = str(fixture.get("home_team_id", fixture.get("home_team", "")))
            away = str(fixture.get("away_team_id", fixture.get("away_team", "")))
            key = f"{home}_v_{away}"
            pred = pred_by_key.get(key, {})

            odds = {}
            for odds_key, field_key in [
                ("home", "B365H"), ("draw", "B365D"), ("away", "B365A")
            ]:
                val = fixture.get(field_key)
                if val is not None:
                    try:
                        odds[odds_key] = float(val)
                    except (TypeError, ValueError):
                        pass

            # Build feature vector from probabilities
            features = np.array([
                pred.get("prob_home", 1/3),
                pred.get("prob_draw", 1/3),
                pred.get("prob_away", 1/3),
                pred.get("prob_over", 0.5),
                pred.get("prob_btts_yes", 0.5),
            ], dtype=np.float64)

            match_id = key
            context = MatchContext(
                match_id=match_id,
                sport=sport,
                league=str(fixture.get("league", "")),
                match_date=str(fixture.get("match_date", str(date.today()))),
                home_team=home,
                away_team=away,
                features=features,
                odds=odds,
                market="1x2",
                outcomes=("home", "draw", "away"),
                metadata=fixture,
            )
            contexts.append(context)

        return contexts

    def _record_agent_predictions_to_ledger(
        self,
        arbiter_results: list,
        fixtures: list[dict],
        sport: str,
    ) -> None:
        """Persist individual agent predictions to the PerformanceLedger (DB)."""
        today = date.today()

        for arbiter_result in arbiter_results:
            # Find the matching fixture for home/away team names
            home = arbiter_result.match_id.split("_v_")[0] if "_v_" in arbiter_result.match_id else ""
            away = arbiter_result.match_id.split("_v_")[1] if "_v_" in arbiter_result.match_id else ""

            for agent_pred in arbiter_result.agent_predictions:
                try:
                    prob_dict = {
                        outcome: float(prob)
                        for outcome, prob in zip(
                            agent_pred.outcomes, agent_pred.probabilities
                        )
                    }
                    self.ledger.record_prediction(
                        agent_name=agent_pred.agent_name,
                        sport=sport,
                        match_date=today,
                        home_team=home,
                        away_team=away,
                        market=agent_pred.market,
                        predicted_outcome=agent_pred.predicted_outcome,
                        probabilities=prob_dict,
                        confidence=float(agent_pred.confidence),
                        reasoning=agent_pred.reasoning,
                    )
                except Exception as exc:
                    logger.debug(
                        "_record_agent_predictions: failed for agent %s match %s: %s",
                        agent_pred.agent_name, arbiter_result.match_id, exc,
                    )

    def _fetch_todays_fixtures(self, sport_slug: str) -> list[dict]:
        """Query the DB for unplayed matches scheduled for today.

        Returns minimal fixture dicts compatible with DailyPipeline.predict().
        """
        today = date.today()
        fixtures: list[dict] = []

        session = get_session()
        try:
            from sharpedge.db.models import Match as MatchRow, Team, League, Season
            from sqlalchemy import select

            rows = (
                session.execute(
                    select(MatchRow, Team, Team, League)
                    .join(
                        Season,
                        MatchRow.season_id == Season.id,
                    )
                    .join(
                        League,
                        Season.league_id == League.id,
                    )
                    .join(
                        Team,
                        MatchRow.home_team_id == Team.id,
                        isouter=True,
                    )
                    .where(
                        MatchRow.match_date == today,
                        MatchRow.result.is_(None),
                    )
                )
                .all()
            )

            # Simpler alternative — avoid multi-join complexity
            matches = (
                session.query(MatchRow)
                .join(Season, MatchRow.season_id == Season.id)
                .join(League, Season.league_id == League.id)
                .filter(
                    MatchRow.match_date == today,
                    MatchRow.result.is_(None),
                )
                .all()
            )

            for match in matches:
                try:
                    home_name = (
                        match.home_team.canonical_name
                        if match.home_team
                        else str(match.home_team_id)
                    )
                    away_name = (
                        match.away_team.canonical_name
                        if match.away_team
                        else str(match.away_team_id)
                    )
                    league_name = ""
                    try:
                        league_name = match.season.league.name
                    except Exception:
                        pass

                    fixtures.append({
                        "home_team": home_name,
                        "home_team_id": home_name,
                        "away_team": away_name,
                        "away_team_id": away_name,
                        "league": league_name,
                        "match_date": str(today),
                    })
                except Exception as exc:
                    logger.warning(
                        "_fetch_todays_fixtures: failed to build fixture dict: %s", exc
                    )

        except Exception as exc:
            logger.warning(
                "_fetch_todays_fixtures: DB query failed for sport=%s: %s",
                sport_slug, exc,
            )
        finally:
            session.close()

        logger.info(
            "_fetch_todays_fixtures: found %d fixtures for sport=%s on %s",
            len(fixtures), sport_slug, today,
        )
        return fixtures

    def _load_todays_picks(self) -> list[DailyPick]:
        """Fetch all unbroadcasted picks created today from the DB."""
        today = date.today()
        picks: list[DailyPick] = []

        session = get_session()
        try:
            from sqlalchemy import select
            picks = list(
                session.execute(
                    select(DailyPick).where(
                        DailyPick.match_date == today,
                        DailyPick.broadcasted_at.is_(None),
                        DailyPick.result.is_(None),
                    )
                )
                .scalars()
                .all()
            )
            # Detach from session before returning
            session.expunge_all()
        except Exception as exc:
            logger.warning("_load_todays_picks: DB query failed: %s", exc)
        finally:
            session.close()

        return picks

    def _get_current_bankroll(self) -> Optional[float]:
        """Query the DB for the latest bankroll figure from BankrollLedger."""
        session = get_session()
        try:
            from sqlalchemy import select
            last_entry = (
                session.execute(
                    select(BankrollLedger)
                    .order_by(BankrollLedger.id.desc())
                    .limit(1)
                )
                .scalars()
                .first()
            )
            if last_entry is not None:
                return float(last_entry.bankroll_after)
            return None
        except Exception as exc:
            logger.warning("_get_current_bankroll: DB query failed: %s", exc)
            return None
        finally:
            session.close()

    def _fail_pipeline_run(
        self, pipeline_run_id: Optional[int], error_message: str
    ) -> None:
        """Mark a PipelineRun row as failed."""
        if pipeline_run_id is None:
            return
        session = get_session()
        try:
            run_row = session.get(PipelineRun, pipeline_run_id)
            if run_row:
                run_row.status = "failed"
                run_row.error_message = error_message[:1000]
                run_row.completed_at = datetime.now(timezone.utc)
                session.commit()
        except Exception:
            session.rollback()
            logger.warning("_fail_pipeline_run: could not update PipelineRun")
        finally:
            session.close()

    def _complete_pipeline_run(
        self,
        pipeline_run_id: Optional[int],
        predictions_count: int,
        picks_count: int,
    ) -> None:
        """Mark a PipelineRun row as completed with counts."""
        if pipeline_run_id is None:
            return
        session = get_session()
        try:
            run_row = session.get(PipelineRun, pipeline_run_id)
            if run_row:
                run_row.status = "completed"
                run_row.predictions_count = predictions_count
                run_row.picks_count = picks_count
                run_row.completed_at = datetime.now(timezone.utc)
                session.commit()
        except Exception:
            session.rollback()
            logger.warning("_complete_pipeline_run: could not update PipelineRun")
        finally:
            session.close()

    def _is_evolution_due(self) -> bool:
        """Return True if today is an evolution day.

        Uses ``agent_evolution_interval_days`` from settings.  Checks the day
        number modulo the interval so it fires on the same weekday every N days
        without persistent state.
        """
        today = date.today()
        # Day-of-year modulo interval — fires on day 0, interval, 2*interval, ...
        day_of_year = today.timetuple().tm_yday
        interval = settings.agent_evolution_interval_days
        return (day_of_year % interval) == 0
