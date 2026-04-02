"""SharpEdge Production Runner — fully wired end-to-end pipeline.

TWO MODES:
  --scheduler   Persistent APScheduler daemon (Railway worker)
  (default)     Single daily cycle (cron / manual)

Single daily cycle phases
-------------------------
  0  COLLECT DATA
  1  RESOLVE YESTERDAY
  2  HEALTH CHECK
  3  LOAD CONTEXT DATA
  4  BOOTSTRAP AGENTS
  5  PREDICT
  6  BROADCAST
  7  EVOLUTION (if due)
  8  REPORT

CLI flags
---------
  --scheduler       Run as persistent daemon
  --date DATE       Override target date (ISO 8601)
  --collect-only    Only run data collection
  --resolve-only    Only resolve yesterday's picks
  --predict-only    Skip collection, go straight to prediction
  --evolve          Force evolution cycle regardless of schedule
  --check-only      Only run health / drift checks
"""
from __future__ import annotations

import argparse
import importlib
import logging
import signal
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure src is on path when run as a script / Railway entrypoint
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Logging — structured, UTC, always to stdout for Railway
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sharpedge.run_daily")

# ---------------------------------------------------------------------------
# Core imports — fail fast so Railway logs the actual error
# ---------------------------------------------------------------------------

from sharpedge.agents.arbiter import BayesianArbiter
from sharpedge.agents.contrarian_agent import ContrarianAgent
from sharpedge.agents.edge_value_agent import EdgeValueAgent
from sharpedge.agents.evolution import AgentEvolutionEngine, EvolutionReport
from sharpedge.agents.form_momentum_agent import FormMomentumAgent
from sharpedge.agents.gradient_agent import GradientAgent
from sharpedge.agents.h2h_venue_agent import H2HVenueAgent
from sharpedge.agents.market_agent import MarketAgent
from sharpedge.agents.statistical_agent import StatisticalAgent
from sharpedge.alerts.telegram import send_alert_sync
from sharpedge.collectors.orchestrator import ScraperOrchestrator
from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.ingest import ingest_dataframe
from sharpedge.db.models import (
    DailyPick,
    League,
    Match,
    MatchOdds,
    PipelineRun,
    Prediction,
    RawStagingRecord,
    Season,
)
from sharpedge.execution.accumulator import AccumulatorBuilder
from sharpedge.execution.staking import AntifragileStaking
from sharpedge.intelligence.edge_discovery import EdgeDiscoveryEngine
from sharpedge.pipeline.daily import DailyPipeline
from sharpedge.pipeline.drift_detector import DriftDetector, DriftResult, SmartRetrainer
from sharpedge.pipeline.live_resolver import LiveResolver
from sharpedge.pipeline.performance_ledger import PerformanceLedger
from sharpedge.warroom.health_monitor import HealthMonitor, HealthStatus
from sharpedge.warroom.orchestrator import WarRoomOrchestrator

# Optional agents — guarded imports
try:
    from sharpedge.agents.regime_agent import RegimeDetectionAgent  # type: ignore
    _REGIME_AGENT_AVAILABLE = True
except ImportError:
    _REGIME_AGENT_AVAILABLE = False
    logger.warning("RegimeDetectionAgent not found — skipping")

try:
    from sharpedge.agents.tabpfn_agent import TabPFNAgent  # type: ignore
    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False

try:
    from sharpedge.agents.league_specialist_agent import LeagueSpecialistAgent  # type: ignore
    _LEAGUE_SPECIALIST_AVAILABLE = True
except ImportError:
    _LEAGUE_SPECIALIST_AVAILABLE = False

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_SHUTDOWN_REQUESTED = False


def _handle_signal(signum: int, _frame) -> None:
    global _SHUTDOWN_REQUESTED
    logger.warning("Signal %d received — requesting graceful shutdown", signum)
    _SHUTDOWN_REQUESTED = True


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TG_LENGTH = 4096
_HISTORY_DAYS = 2 * 365  # ~2 seasons of lookback


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

def _send_telegram(message: str) -> None:
    """Best-effort Telegram broadcast. Never raises."""
    try:
        if len(message) > _MAX_TG_LENGTH:
            for i in range(0, len(message), _MAX_TG_LENGTH):
                send_alert_sync(message[i : i + _MAX_TG_LENGTH])
        else:
            send_alert_sync(message)
    except Exception:
        logger.exception("Telegram broadcast failed (non-fatal)")


def _fmt_elapsed(start_ts: datetime) -> str:
    secs = (datetime.now(timezone.utc) - start_ts).total_seconds()
    mins, s = divmod(int(secs), 60)
    return f"{mins}m{s:02d}s" if mins else f"{s}s"


def _phase_banner(n: int, name: str) -> None:
    logger.info("=" * 60)
    logger.info("  PHASE %d: %s", n, name)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _open_pipeline_run(target_date: date, run_type: str) -> Optional[int]:
    session = get_session()
    try:
        run = PipelineRun(
            run_date=target_date,
            run_type=run_type,
            status="running",
            started_at=datetime.now(timezone.utc),
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id
        logger.info("PipelineRun created: id=%d type=%s", run_id, run_type)
        return run_id
    except Exception:
        session.rollback()
        logger.exception("Failed to create PipelineRun record")
        return None
    finally:
        session.close()


def _close_pipeline_run(
    run_id: Optional[int],
    status: str,
    *,
    fixtures_count: int = 0,
    predictions_count: int = 0,
    picks_count: int = 0,
    error_message: Optional[str] = None,
) -> None:
    if run_id is None:
        return
    session = get_session()
    try:
        run = session.query(PipelineRun).filter(PipelineRun.id == run_id).first()
        if run:
            run.status = status
            run.completed_at = datetime.now(timezone.utc)
            run.fixtures_count = fixtures_count
            run.predictions_count = predictions_count
            run.picks_count = picks_count
            run.error_message = error_message
            session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to update PipelineRun")
    finally:
        session.close()


def _should_run_evolution(target_date: date) -> bool:
    interval = settings.agent_evolution_interval_days
    session = get_session()
    try:
        from sharpedge.db.models import AgentEvolution  # type: ignore
        latest = (
            session.query(AgentEvolution.event_date)
            .order_by(AgentEvolution.event_date.desc())
            .first()
        )
        if latest is None:
            return True
        delta = (target_date - latest[0]).days
        logger.info("Last evolution: %s (%d days ago, interval=%d)", latest[0], delta, interval)
        return delta >= interval
    except Exception:
        logger.exception("Evolution schedule check failed — defaulting to skip")
        return False
    finally:
        session.close()


# ---------------------------------------------------------------------------
# PHASE 0: COLLECT DATA
# ---------------------------------------------------------------------------

def phase_collect(orchestrator: ScraperOrchestrator, target_date: date) -> dict:
    """Run all collectors and ingest collected data to DB.

    Returns a summary dict with counts per category.
    """
    _phase_banner(0, "COLLECT DATA")
    summary: dict = {
        "full_report": None,
        "odds_rows": 0,
        "intelligence_rows": 0,
        "errors": [],
    }

    # 0a. Full collection (all 31+ collectors)
    try:
        logger.info("[Phase 0] Running full collection (all collectors)...")
        report = orchestrator.run_all(date=target_date.isoformat())
        summary["full_report"] = report
        logger.info(
            "[Phase 0] Full collection: %d/%d sources ok, %d rows",
            report.sources_succeeded,
            report.sources_attempted,
            report.total_rows,
        )
        # Ingest each source's data frame
        for source_name, src_info in (report.by_source or {}).items():
            df = src_info.get("df") if isinstance(src_info, dict) else None
            if df is not None and not df.empty:
                try:
                    rows = ingest_dataframe(df, source_name)
                    logger.info("[Phase 0] Ingested %d rows from %s", rows, source_name)
                except Exception as ingest_err:
                    msg = f"Ingest failed for {source_name}: {ingest_err}"
                    logger.warning("[Phase 0] %s", msg)
                    summary["errors"].append(msg)
    except Exception as exc:
        msg = f"Full collection crashed: {exc}"
        logger.exception("[Phase 0] %s", msg)
        summary["errors"].append(msg)
        _send_telegram(f"CRITICAL: Full data collection failed.\n{exc}")

    # 0b. Fresh odds
    try:
        logger.info("[Phase 0] Running odds collection...")
        odds_df = orchestrator.run_odds_collection(date=target_date.isoformat())
        if odds_df is not None and not odds_df.empty:
            summary["odds_rows"] = ingest_dataframe(odds_df, "odds_collection")
        logger.info("[Phase 0] Odds ingested: %d rows", summary["odds_rows"])
    except Exception as exc:
        msg = f"Odds collection failed: {exc}"
        logger.warning("[Phase 0] %s", msg)
        summary["errors"].append(msg)

    # 0c. Intelligence: referee, manager, wages, fatigue, team news, advanced metrics
    intelligence_collectors = [
        ("referee_stats",    "sharpedge.collectors.referee_stats",    "RefereeStatsCollector"),
        ("manager_records",  "sharpedge.collectors.manager_records",  "ManagerRecordsCollector"),
        ("capology_wages",   "sharpedge.collectors.capology",         "CapologyCollector"),
        ("european_fatigue", "sharpedge.collectors.european_fatigue", "EuropeanFatigueCollector"),
        ("team_news",        "sharpedge.collectors.team_news",        "TeamNewsCollector"),
        ("fbref_advanced",   "sharpedge.collectors.fbref",            "FBrefCollector"),
        ("advanced_metrics", "sharpedge.collectors.advanced_metrics", "AdvancedMetricsCollector"),
    ]
    date_str = target_date.isoformat()
    for record_type, module_path, class_name in intelligence_collectors:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            df = cls().collect(date=date_str)
            if df is not None and not df.empty:
                rows = ingest_dataframe(df, record_type)
                summary["intelligence_rows"] += rows
                logger.info("[Phase 0] Intelligence '%s': %d rows", record_type, rows)
        except Exception as exc:
            msg = f"Intelligence collector '{record_type}' failed: {exc}"
            logger.warning("[Phase 0] %s", msg)
            summary["errors"].append(msg)

    logger.info(
        "[Phase 0] Collection complete. Errors: %d", len(summary["errors"])
    )
    return summary


# ---------------------------------------------------------------------------
# PHASE 1: RESOLVE YESTERDAY
# ---------------------------------------------------------------------------

def phase_resolve(target_date: date) -> dict:
    """Fetch results, resolve picks, update bankroll, snapshot agent performance."""
    _phase_banner(1, "RESOLVE YESTERDAY")
    result: dict = {
        "resolved": 0,
        "won": 0,
        "lost": 0,
        "voided": 0,
        "pnl": 0.0,
        "avg_clv_pct": None,
        "picks": [],
        "errors": [],
    }

    # 1a. LiveResolver fetches results from API and resolves DailyPick rows
    try:
        resolver = LiveResolver()
        resolution = resolver.run()
        result["resolved"] = getattr(resolution, "resolved_count", 0)
        result["won"] = getattr(resolution, "won_count", 0)
        result["lost"] = getattr(resolution, "lost_count", 0)
        result["voided"] = getattr(resolution, "voided_count", 0)
        result["pnl"] = float(getattr(resolution, "profit", 0.0))
        result["avg_clv_pct"] = getattr(resolution, "avg_clv_pct", None)
        result["picks"] = getattr(resolution, "resolved_picks", [])
        logger.info(
            "[Phase 1] Resolved %d picks — W:%d L:%d V:%d P&L:%.2f",
            result["resolved"], result["won"], result["lost"],
            result["voided"], result["pnl"],
        )
    except Exception as exc:
        msg = f"LiveResolver failed: {exc}"
        logger.exception("[Phase 1] %s", msg)
        result["errors"].append(msg)

    # 1b. CLV is computed inside LiveResolver; log summary
    if result["avg_clv_pct"] is not None:
        logger.info("[Phase 1] Average CLV: %+.2f%%", result["avg_clv_pct"])

    # 1c. Snapshot agent performance
    try:
        ledger = PerformanceLedger()
        ledger.snapshot_daily()
        logger.info("[Phase 1] Agent performance snapshot saved")
    except Exception as exc:
        msg = f"Performance snapshot failed: {exc}"
        logger.warning("[Phase 1] %s", msg)
        result["errors"].append(msg)

    return result


# ---------------------------------------------------------------------------
# PHASE 2: HEALTH CHECK
# ---------------------------------------------------------------------------

def phase_health_check() -> tuple[HealthStatus, list, int]:
    """Run all health checks and determine circuit breaker level.

    Returns (overall_status, components, circuit_level)
    where circuit_level: 0=normal, 1=caution, 2=reduced, 3=STOP
    """
    _phase_banner(2, "HEALTH CHECK")
    monitor = HealthMonitor()
    health = monitor.run_checks()

    for comp in health.components:
        level = logging.WARNING if comp.status != HealthStatus.HEALTHY else logging.DEBUG
        logger.log(level, "  [%s] %s — %s", comp.status.value, comp.name, comp.message)

    # Determine circuit breaker level
    down_count = sum(1 for c in health.components if c.status == HealthStatus.DOWN)
    degraded_count = sum(1 for c in health.components if c.status == HealthStatus.DEGRADED)

    if health.overall == HealthStatus.DOWN or down_count >= 2:
        circuit_level = 3  # STOP
    elif down_count == 1 or degraded_count >= 3:
        circuit_level = 2  # Reduced staking
    elif degraded_count >= 1:
        circuit_level = 1  # Caution
    else:
        circuit_level = 0  # Normal

    logger.info(
        "[Phase 2] Health: %s | Circuit breaker: level %d",
        health.overall.value.upper(),
        circuit_level,
    )
    return health.overall, health.components, circuit_level


# ---------------------------------------------------------------------------
# PHASE 3: LOAD CONTEXT DATA
# ---------------------------------------------------------------------------

def phase_load_context(target_date: date) -> dict:
    """Load all context data from DB / staging into memory.

    Returns dict with keys:
      historical_matches, fixtures, elo_df, xg_df, competitor_df,
      referee_df, manager_df, wage_df, fatigue_df, lineup_df, advanced_df
    """
    _phase_banner(3, "LOAD CONTEXT DATA")
    ctx: dict = {
        "historical_matches": [],
        "fixtures": [],
        "elo_df": pd.DataFrame(),
        "xg_df": pd.DataFrame(),
        "competitor_df": pd.DataFrame(),
        "referee_df": pd.DataFrame(),
        "manager_df": pd.DataFrame(),
        "wage_df": pd.DataFrame(),
        "fatigue_df": pd.DataFrame(),
        "lineup_df": pd.DataFrame(),
        "advanced_df": pd.DataFrame(),
        "errors": [],
    }

    cutoff = target_date - timedelta(days=_HISTORY_DAYS)
    tomorrow = target_date + timedelta(days=1)

    session = get_session()
    try:
        # 3a. Historical matches (last 2 seasons, resolved)
        try:
            hist_rows = (
                session.query(Match)
                .filter(Match.match_date >= cutoff, Match.result.isnot(None))
                .order_by(Match.match_date)
                .all()
            )
            for m in hist_rows:
                home = m.home_team.canonical_name if m.home_team else str(m.home_team_id)
                away = m.away_team.canonical_name if m.away_team else str(m.away_team_id)
                league = (m.season.league.name if m.season and m.season.league else "")
                ctx["historical_matches"].append({
                    "match_id": str(m.id),
                    "home_team_id": home, "away_team_id": away,
                    "home_team": home, "away_team": away,
                    "league": league,
                    "match_date": str(m.match_date),
                    "FTHG": m.home_goals, "FTAG": m.away_goals,
                    "FTR": m.result or "",
                    "season": m.season.label if m.season else "",
                    "venue": m.venue or "",
                    "referee": m.referee or "",
                })
            logger.info("[Phase 3] Historical matches loaded: %d", len(ctx["historical_matches"]))
        except Exception as exc:
            ctx["errors"].append(f"Historical matches: {exc}")
            logger.exception("[Phase 3] Failed to load historical matches")

        # 3b. Today's fixtures (today + tomorrow)
        try:
            fixture_rows = (
                session.query(Match)
                .filter(
                    Match.match_date.in_([target_date, tomorrow]),
                    Match.status == "scheduled",
                )
                .order_by(Match.match_date, Match.kick_off_time)
                .all()
            )
            BK_PRIORITY = ["Pinnacle", "Bet365", "William Hill", "Betfair"]
            for m in fixture_rows:
                home = m.home_team.canonical_name if m.home_team else str(m.home_team_id)
                away = m.away_team.canonical_name if m.away_team else str(m.away_team_id)
                league = (m.season.league.name if m.season and m.season.league else "")
                odds_map: dict = {}
                for o in (m.odds or []):
                    k = f"{o.bookmaker}_{o.market}_{o.odds_type}"
                    odds_map[k] = o
                fixture: dict = {
                    "match_id": str(m.id),
                    "home_team_id": home, "away_team_id": away,
                    "home_team": home, "away_team": away,
                    "league": league,
                    "match_date": str(m.match_date),
                    "kick_off_time": m.kick_off_time or "",
                    "season": m.season.label if m.season else "",
                    "venue": m.venue or "",
                    "referee": m.referee or "",
                }
                # Attach Bet365 and Pinnacle odds columns
                for col, attr, bk_tag in [
                    ("B365H", "odds_home", "Bet365"),
                    ("B365D", "odds_draw", "Bet365"),
                    ("B365A", "odds_away", "Bet365"),
                    ("PSH",   "odds_home", "Pinnacle"),
                    ("PSD",   "odds_draw", "Pinnacle"),
                    ("PSA",   "odds_away", "Pinnacle"),
                ]:
                    for bk_key in [k for k in odds_map if bk_tag in k]:
                        row = odds_map[bk_key]
                        val = getattr(row, attr, None)
                        if val is not None:
                            fixture[col] = float(val)
                            break
                ctx["fixtures"].append(fixture)
            logger.info("[Phase 3] Fixtures loaded: %d", len(ctx["fixtures"]))
        except Exception as exc:
            ctx["errors"].append(f"Fixtures: {exc}")
            logger.exception("[Phase 3] Failed to load fixtures")

        # 3c. ELO ratings
        try:
            elo_rows = (
                session.query(RawStagingRecord)
                .filter(
                    RawStagingRecord.record_type == "elo_rating",
                    RawStagingRecord.status == "ingested",
                )
                .all()
            )
            if elo_rows:
                ctx["elo_df"] = pd.DataFrame([r.raw_data for r in elo_rows if r.raw_data])
            logger.info("[Phase 3] ELO rows: %d", len(ctx["elo_df"]))
        except Exception as exc:
            ctx["errors"].append(f"ELO: {exc}")
            logger.warning("[Phase 3] ELO load failed: %s", exc)

        # 3d. xG data
        try:
            xg_rows = (
                session.query(RawStagingRecord)
                .filter(
                    RawStagingRecord.record_type == "xg_data",
                    RawStagingRecord.status == "ingested",
                )
                .order_by(RawStagingRecord.created_at.desc())
                .limit(50_000)
                .all()
            )
            if xg_rows:
                ctx["xg_df"] = pd.DataFrame([r.raw_data for r in xg_rows if r.raw_data])
            logger.info("[Phase 3] xG rows: %d", len(ctx["xg_df"]))
        except Exception as exc:
            ctx["errors"].append(f"xG: {exc}")
            logger.warning("[Phase 3] xG load failed: %s", exc)

        # 3e. Competitor predictions
        try:
            comp_rows = (
                session.query(RawStagingRecord)
                .filter(
                    RawStagingRecord.record_type.in_([
                        "forebet_prediction", "predictz_prediction",
                        "windrawwin_prediction", "competitor_prediction",
                    ]),
                    RawStagingRecord.status == "ingested",
                )
                .all()
            )
            if comp_rows:
                ctx["competitor_df"] = pd.DataFrame(
                    [r.raw_data for r in comp_rows if r.raw_data]
                )
            logger.info("[Phase 3] Competitor predictions: %d rows", len(ctx["competitor_df"]))
        except Exception as exc:
            ctx["errors"].append(f"Competitor predictions: {exc}")
            logger.warning("[Phase 3] Competitor predictions load failed: %s", exc)

        # 3f-3j. Intelligence data from staging
        _staging_map = {
            "referee_df":  "referee_stats",
            "manager_df":  "manager_records",
            "wage_df":     "capology_wages",
            "fatigue_df":  "european_fatigue",
            "lineup_df":   "lineup",
            "advanced_df": ("fbref_advanced", "advanced_metrics"),
        }
        for ctx_key, record_types in _staging_map.items():
            if isinstance(record_types, str):
                record_types = (record_types,)
            try:
                rows = (
                    session.query(RawStagingRecord)
                    .filter(
                        RawStagingRecord.record_type.in_(record_types),
                        RawStagingRecord.status == "ingested",
                    )
                    .all()
                )
                if rows:
                    ctx[ctx_key] = pd.DataFrame(
                        [r.raw_data for r in rows if r.raw_data]
                    )
                logger.info(
                    "[Phase 3] %s: %d rows (types=%s)",
                    ctx_key, len(ctx[ctx_key]), record_types,
                )
            except Exception as exc:
                ctx["errors"].append(f"{ctx_key}: {exc}")
                logger.warning("[Phase 3] %s load failed: %s", ctx_key, exc)

    finally:
        session.close()

    logger.info(
        "[Phase 3] Context loaded. Fixtures=%d  History=%d  Errors=%d",
        len(ctx["fixtures"]),
        len(ctx["historical_matches"]),
        len(ctx["errors"]),
    )
    return ctx


# ---------------------------------------------------------------------------
# PHASE 4: BOOTSTRAP AGENTS
# ---------------------------------------------------------------------------

def phase_bootstrap_agents(ctx: dict) -> tuple[list, BayesianArbiter]:
    """Initialise all agents, fit those that require training data, register with arbiter."""
    _phase_banner(4, "BOOTSTRAP AGENTS")
    agents = []
    hist_df = pd.DataFrame(ctx["historical_matches"]) if ctx["historical_matches"] else pd.DataFrame()
    historical_matches = ctx["historical_matches"]

    # No-training agents
    for AgentCls in [MarketAgent, ContrarianAgent, EdgeValueAgent]:
        try:
            agents.append(AgentCls())
            logger.info("[Phase 4] %s ready (no training)", AgentCls.__name__)
        except Exception as exc:
            logger.warning("[Phase 4] %s init failed: %s", AgentCls.__name__, exc)

    # Agents that require historical match data
    fitting_agents = [
        ("StatisticalAgent",   StatisticalAgent,   "fit", historical_matches),
        ("FormMomentumAgent",  FormMomentumAgent,  "fit", hist_df),
        ("H2HVenueAgent",      H2HVenueAgent,      "fit", historical_matches),
    ]
    for name, AgentCls, fit_method, data in fitting_agents:
        try:
            agent = AgentCls()
            getattr(agent, fit_method)(data)
            agents.append(agent)
            logger.info("[Phase 4] %s fitted (%d rows)", name, len(data))
        except Exception as exc:
            logger.warning("[Phase 4] %s fit failed — skipping: %s", name, exc)

    # GradientAgent — needs model training data
    try:
        from sharpedge.ml.training.trainer import ModelTrainer  # type: ignore
        trainer = ModelTrainer.load(settings.model_path)
        if trainer.X_train is not None and trainer.y_train is not None:
            grad = GradientAgent()
            grad.fit(trainer.X_train, trainer.y_train)
            agents.append(grad)
            logger.info("[Phase 4] GradientAgent fitted from saved model")
        else:
            logger.warning("[Phase 4] GradientAgent skipped — no cached training data")
    except Exception as exc:
        logger.warning("[Phase 4] GradientAgent init failed: %s", exc)

    # RegimeDetectionAgent (optional)
    if _REGIME_AGENT_AVAILABLE:
        try:
            regime = RegimeDetectionAgent()  # type: ignore
            regime.fit(historical_matches)
            agents.append(regime)
            logger.info("[Phase 4] RegimeDetectionAgent fitted")
        except Exception as exc:
            logger.warning("[Phase 4] RegimeDetectionAgent failed: %s", exc)

    # TabPFN agent (optional)
    if _TABPFN_AVAILABLE:
        try:
            tab = TabPFNAgent()  # type: ignore
            if not hist_df.empty:
                tab.fit(hist_df)
            agents.append(tab)
            logger.info("[Phase 4] TabPFNAgent ready")
        except Exception as exc:
            logger.warning("[Phase 4] TabPFNAgent failed: %s", exc)

    # LeagueSpecialist (optional)
    if _LEAGUE_SPECIALIST_AVAILABLE:
        try:
            league_agent = LeagueSpecialistAgent()  # type: ignore
            agents.append(league_agent)
            logger.info("[Phase 4] LeagueSpecialistAgent ready")
        except Exception as exc:
            logger.warning("[Phase 4] LeagueSpecialistAgent failed: %s", exc)

    # Register all agents with the arbiter
    arbiter = BayesianArbiter()
    for agent in agents:
        try:
            arbiter.register_agent(agent)
        except Exception as exc:
            logger.warning("[Phase 4] Arbiter registration failed for %s: %s", agent, exc)

    logger.info("[Phase 4] Bootstrap complete: %d agents registered", len(agents))
    return agents, arbiter


# ---------------------------------------------------------------------------
# PHASE 5: PREDICT
# ---------------------------------------------------------------------------

def phase_predict(
    target_date: date,
    ctx: dict,
    agents: list,
    arbiter: BayesianArbiter,
    run_id: Optional[int],
    circuit_level: int,
) -> dict:
    """Run the full prediction pipeline.

    Returns dict with picks, accumulators, predictions, errors.
    """
    _phase_banner(5, "PREDICT")
    result: dict = {
        "predictions": [],
        "picks": [],
        "accumulators": [],
        "n_fixtures": len(ctx["fixtures"]),
        "errors": [],
    }

    fixtures = ctx["fixtures"]
    if not fixtures:
        logger.warning("[Phase 5] No fixtures — skipping prediction")
        return result

    try:
        # 5a. Load DailyPipeline (core ML ensemble)
        pipeline = DailyPipeline(model_path=settings.model_path)
        pipeline.load_model()

        hist_df = (
            pd.DataFrame(ctx["historical_matches"])
            if ctx["historical_matches"]
            else None
        )

        # Attach context DataFrames for feature enrichment
        for attr, key in [
            ("elo_df",        "elo_df"),
            ("xg_df",         "xg_df"),
            ("competitor_df", "competitor_df"),
            ("referee_df",    "referee_df"),
            ("manager_df",    "manager_df"),
            ("wage_df",       "wage_df"),
            ("fatigue_df",    "fatigue_df"),
            ("lineup_df",     "lineup_df"),
            ("advanced_df",   "advanced_df"),
        ]:
            df = ctx.get(key)
            if df is not None and not df.empty:
                try:
                    setattr(pipeline, attr, df)
                except Exception:
                    pass  # not all pipeline versions expose these attributes

        # 5b. Run ensemble predictions
        predictions_raw = pipeline.predict(fixtures, historical_matches=hist_df)
        result["predictions"] = predictions_raw or []
        logger.info("[Phase 5] Predictions generated: %d", len(result["predictions"]))

        # 5c. Run agent swarm through arbiter
        try:
            from sharpedge.agents.base_agent import MatchContext
            for fix in fixtures[:len(result["predictions"])]:
                match_ctx = MatchContext(
                    match_id=fix.get("match_id", ""),
                    home_team=fix.get("home_team", ""),
                    away_team=fix.get("away_team", ""),
                    league=fix.get("league", ""),
                    match_date=fix.get("match_date", str(target_date)),
                    home_odds=fix.get("PSH") or fix.get("B365H"),
                    draw_odds=fix.get("PSD") or fix.get("B365D"),
                    away_odds=fix.get("PSA") or fix.get("B365A"),
                )
                arbiter.evaluate(match_ctx)
        except Exception as exc:
            logger.warning("[Phase 5] Arbiter swarm evaluation failed: %s", exc)
            result["errors"].append(f"Arbiter swarm: {exc}")

        # 5d. Apply banker filter
        picks_raw = pipeline.filter_picks(result["predictions"])
        logger.info("[Phase 5] Picks after banker filter: %d", len(picks_raw))

        # 5e. Compute stakes via AntifragileStaking
        staker = AntifragileStaking()
        # Reduce stakes if circuit breaker is elevated
        stake_multiplier = max(0.0, 1.0 - circuit_level * 0.33)
        if circuit_level > 0:
            logger.warning(
                "[Phase 5] Circuit breaker level %d — stake multiplier %.2f",
                circuit_level, stake_multiplier,
            )

        picks: list[dict] = []
        for p in picks_raw:
            pick_dict = {
                "home_team":       getattr(p, "home_team", ""),
                "away_team":       getattr(p, "away_team", ""),
                "league":          getattr(p, "league", ""),
                "match_date":      getattr(p, "match_date", str(target_date)),
                "pick_market":     getattr(p, "market", ""),
                "pick_selection":  getattr(p, "selection", ""),
                "model_prob":      float(getattr(p, "model_prob", 0)),
                "model_spread":    float(getattr(p, "model_spread", 0)),
                "best_odds":       float(getattr(p, "best_odds", 1.0)),
                "bookmaker":       getattr(p, "bookmaker", ""),
                "implied_prob": (
                    1.0 / float(getattr(p, "best_odds", 2.0))
                    if getattr(p, "best_odds", 0) > 1.0 else 0.0
                ),
                "edge":            float(getattr(p, "edge", 0)),
                "tier":            getattr(p, "tier", "standard"),
                "meta_agreement":  int(getattr(p, "meta_agreement", 0)),
                "risk_flags":      getattr(p, "risk_flags", []),
                "stake_flat":      float(getattr(p, "stake_flat", 0) or 0) * stake_multiplier,
                "stake_kelly":     float(getattr(p, "stake_kelly", 0) or 0) * stake_multiplier,
            }
            # Ask staker for refined stake if it supports that interface
            try:
                refined = staker.compute_stake(
                    edge=pick_dict["edge"],
                    odds=pick_dict["best_odds"],
                    model_prob=pick_dict["model_prob"],
                )
                pick_dict["stake_flat"] = float(refined) * stake_multiplier
            except Exception:
                pass
            picks.append(pick_dict)

        result["picks"] = picks

        # 5f. Build accumulators
        try:
            builder = AccumulatorBuilder()
            accas = builder.build_daily_accas(picks)
            result["accumulators"] = accas or []
            logger.info("[Phase 5] Accumulators built: %d", len(result["accumulators"]))
        except Exception as exc:
            logger.warning("[Phase 5] Accumulator build failed: %s", exc)
            result["errors"].append(f"Accumulators: {exc}")

        # 5g. Persist picks to DB
        _persist_picks(picks, run_id, target_date)

        # 5h. Record in PerformanceLedger
        try:
            ledger = PerformanceLedger()
            for pick in picks:
                ledger.record_prediction(
                    agent_name="ensemble",
                    match_id=pick.get("match_id", ""),
                    market=pick.get("pick_market", ""),
                    selection=pick.get("pick_selection", ""),
                    model_prob=pick.get("model_prob", 0.0),
                    odds=pick.get("best_odds", 1.0),
                    stake=pick.get("stake_flat", 0.0),
                )
        except Exception as exc:
            logger.warning("[Phase 5] PerformanceLedger recording failed: %s", exc)
            result["errors"].append(f"Ledger: {exc}")

    except Exception as exc:
        msg = f"Prediction pipeline crashed: {exc}"
        logger.exception("[Phase 5] %s", msg)
        result["errors"].append(msg)
        _send_telegram(f"CRITICAL: Prediction pipeline failed.\n{exc}")

    logger.info(
        "[Phase 5] Predict complete — picks=%d accas=%d errors=%d",
        len(result["picks"]),
        len(result["accumulators"]),
        len(result["errors"]),
    )
    return result


def _persist_picks(
    picks: list[dict],
    run_id: Optional[int],
    target_date: Optional[date] = None,
) -> None:
    """Write DailyPick rows to DB, skipping duplicates."""
    if not picks or run_id is None:
        return

    today = target_date or date.today()
    session = get_session()
    inserted = 0
    try:
        for p in picks:
            raw_date = p.get("match_date", str(today))
            try:
                match_date_val = date.fromisoformat(str(raw_date)[:10])
            except Exception:
                match_date_val = today

            existing = (
                session.query(DailyPick)
                .join(PipelineRun, DailyPick.pipeline_run_id == PipelineRun.id)
                .filter(
                    DailyPick.home_team == p["home_team"],
                    DailyPick.away_team == p["away_team"],
                    DailyPick.match_date == match_date_val,
                    DailyPick.pick_market == p["pick_market"],
                    PipelineRun.run_date == today,
                )
                .first()
            )
            if existing:
                continue

            pred_row = Prediction(
                pipeline_run_id=run_id,
                match_date=match_date_val,
                home_team=p["home_team"],
                away_team=p["away_team"],
                league=p.get("league", ""),
                prob_home=p.get("model_prob") if "home" in p.get("pick_market", "") else None,
                prob_draw=p.get("model_prob") if "draw" in p.get("pick_market", "") else None,
                prob_away=p.get("model_prob") if "away" in p.get("pick_market", "") else None,
                created_at=datetime.now(timezone.utc),
            )
            session.add(pred_row)
            session.flush()

            pick_row = DailyPick(
                pipeline_run_id=run_id,
                prediction_id=pred_row.id,
                match_date=match_date_val,
                home_team=p["home_team"],
                away_team=p["away_team"],
                league=p.get("league", ""),
                pick_market=p["pick_market"],
                pick_selection=p.get("pick_selection", ""),
                model_prob=p.get("model_prob", 0.0),
                model_spread=p.get("model_spread", 0.0),
                best_odds=p.get("best_odds", 1.0),
                bookmaker=p.get("bookmaker", ""),
                implied_prob=p.get("implied_prob", 0.0),
                edge=p.get("edge", 0.0),
                tier=p.get("tier", "standard"),
                meta_agreement=p.get("meta_agreement", 0),
                risk_flags=p.get("risk_flags"),
                stake_flat=p.get("stake_flat"),
                stake_kelly=p.get("stake_kelly"),
            )
            session.add(pick_row)
            inserted += 1

        session.commit()
        logger.info("Persisted %d picks to DB (run_id=%d)", inserted, run_id)
    except Exception:
        session.rollback()
        logger.exception("Failed to persist picks — rolled back")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# PHASE 6: BROADCAST
# ---------------------------------------------------------------------------

def phase_broadcast(
    target_date: date,
    predict_result: dict,
    resolution: dict,
    health_status: HealthStatus,
    health_components: list,
    drift_result: Optional[DriftResult],
    evolution_report: Optional[EvolutionReport],
    elapsed: str,
) -> None:
    """Format all outputs and send to Telegram."""
    _phase_banner(6, "BROADCAST")

    picks = predict_result.get("picks", [])
    accumulators = predict_result.get("accumulators", [])
    lines: list[str] = []

    env_tag = f" [{settings.environment.upper()}]" if settings.environment != "production" else ""
    lines.append(f"<b>SharpEdge Daily{env_tag} — {target_date}</b>")
    lines.append(f"Run time: {elapsed}")
    lines.append("")

    # Pipeline summary
    health_icon = {
        HealthStatus.HEALTHY: "✅",
        HealthStatus.DEGRADED: "⚠️",
        HealthStatus.DOWN: "🚨",
    }.get(health_status, "❓")
    lines.append(f"<b>Pipeline</b>")
    lines.append(f"  Fixtures: {predict_result.get('n_fixtures', 0)}")
    lines.append(f"  Picks: {len(picks)}")
    lines.append(f"  Accumulators: {len(accumulators)}")
    lines.append(f"  Health: {health_icon} {health_status.value.upper()}")
    lines.append("")

    # Today's picks by tier
    if picks:
        lines.append("<b>Today's Picks</b>")
        by_tier: dict[str, list] = {}
        for p in picks:
            by_tier.setdefault(p.get("tier", "standard"), []).append(p)
        for tier in ("diamond", "platinum", "gold", "standard"):
            tier_picks = by_tier.get(tier, [])
            if not tier_picks:
                continue
            lines.append(f"  <i>{tier.capitalize()}</i> ({len(tier_picks)})")
            for p in tier_picks[:5]:
                edge_s = f" +{p['edge']*100:.1f}%" if p.get("edge") else ""
                odds_s = f" @ {p['best_odds']:.2f}" if p.get("best_odds") else ""
                stake_s = f" [{p['stake_flat']:.2f}u]" if p.get("stake_flat") else ""
                lines.append(
                    f"    • {p['home_team']} v {p['away_team']} "
                    f"| {p.get('pick_selection','')}{odds_s}{edge_s}{stake_s}"
                )
        lines.append("")
    else:
        lines.append("<b>Picks:</b> none today")
        lines.append("")

    # Accumulators
    if accumulators:
        lines.append("<b>Accumulators</b>")
        for i, acca in enumerate(accumulators[:3], 1):
            legs = getattr(acca, "legs", []) or (acca if isinstance(acca, list) else [])
            combined_odds = getattr(acca, "combined_odds", None)
            stake = getattr(acca, "stake", None)
            odds_s = f" @ {combined_odds:.2f}" if combined_odds else ""
            stake_s = f" [{stake:.2f}u]" if stake else ""
            lines.append(f"  Acca {i} ({len(legs)} legs){odds_s}{stake_s}")
        lines.append("")

    # Yesterday's results
    lines.append("<b>Yesterday's Results</b>")
    w = resolution.get("won", resolution.get("wins", 0))
    lo = resolution.get("lost", resolution.get("losses", 0))
    v = resolution.get("voided", resolution.get("voids", 0))
    pnl = resolution.get("pnl", 0.0)
    clv = resolution.get("avg_clv_pct")
    if w + lo + v == 0:
        lines.append("  No resolved picks")
    else:
        total = w + lo + v
        win_rate = w / (w + lo) * 100 if (w + lo) > 0 else 0.0
        pnl_sign = "+" if pnl >= 0 else ""
        lines.append(f"  W/L/V: {w}/{lo}/{v}  ({total} total)")
        lines.append(f"  Win rate: {win_rate:.1f}%")
        lines.append(f"  P&L: {pnl_sign}{pnl:.2f}u")
        if clv is not None:
            lines.append(f"  Avg CLV: {clv:+.2f}%")
        resolved_picks = resolution.get("picks", [])
        if isinstance(resolved_picks, list):
            for p in resolved_picks[:8]:
                result_s = (
                    p.get("result", "") if isinstance(p, dict)
                    else getattr(p, "result", "")
                )
                match_s = (
                    p.get("match", "") if isinstance(p, dict)
                    else f"{getattr(p,'home_team','')} v {getattr(p,'away_team','')}"
                )
                icon = "✅" if result_s == "win" else ("❌" if result_s == "loss" else "⬛")
                pnl_p = p.get("pnl", 0.0) if isinstance(p, dict) else getattr(p, "profit_loss", 0.0)
                lines.append(f"  {icon} {match_s}: {pnl_p:+.2f}u")
    lines.append("")

    # CLV tracking line
    clv_target = getattr(settings, "clv_target_pct", 2.0)
    lines.append(f"<b>CLV target:</b> +{clv_target:.1f}% vs closing line")
    lines.append("")

    # Drift
    if drift_result:
        has_drift = getattr(drift_result, "has_drift", False) or getattr(drift_result, "drift_detected", False)
        if has_drift:
            sev = getattr(drift_result, "severity", "unknown")
            ece = getattr(drift_result, "calibration_error", 0.0)
            brier = getattr(drift_result, "brier_score", 0.0)
            lines.append(f"⚠️ <b>Drift</b>: severity={sev}  ECE={ece:.4f}  Brier={brier:.4f}")
            lines.append("")

    # Evolution
    if evolution_report:
        promoted = getattr(evolution_report, "promotions", [])
        deprecated = getattr(evolution_report, "deprecations", [])
        edges = getattr(evolution_report, "edges_discovered", [])
        lines.append("<b>Evolution</b>")
        if promoted:
            lines.append(f"  Promoted: {', '.join(promoted)}")
        if deprecated:
            lines.append(f"  Deprecated: {', '.join(deprecated)}")
        sig_edges = [e for e in edges if getattr(e, "is_significant", False)]
        if sig_edges:
            lines.append(f"  Significant edges: {len(sig_edges)}")
        lines.append("")

    # Degraded components
    degraded = [c for c in health_components if c.status != HealthStatus.HEALTHY]
    if degraded:
        lines.append("<b>Component Warnings</b>")
        for c in degraded:
            icon = "🚨" if c.status == HealthStatus.DOWN else "⚠️"
            lines.append(f"  {icon} {c.name}: {c.message}")
        lines.append("")

    lines.append(f"<i>SharpEdge · {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>")
    report = "\n".join(lines)

    logger.info("=" * 60)
    logger.info("BROADCAST DIGEST\n%s", report)
    logger.info("=" * 60)
    _send_telegram(report)


# ---------------------------------------------------------------------------
# PHASE 7: EVOLUTION
# ---------------------------------------------------------------------------

def phase_evolution(target_date: date) -> Optional[EvolutionReport]:
    """Run agent natural selection and edge discovery."""
    _phase_banner(7, "EVOLUTION")
    try:
        engine = AgentEvolutionEngine()
        report = engine.run_natural_selection("football")
        logger.info(
            "[Phase 7] Evolution complete: promoted=%s deprecated=%s edges=%d",
            getattr(report, "promotions", []),
            getattr(report, "deprecations", []),
            len(getattr(report, "edges_discovered", [])),
        )
        try:
            edge_engine = EdgeDiscoveryEngine()
            edge_report = edge_engine.discover()
            logger.info(
                "[Phase 7] Edge discovery: %d edges found",
                getattr(edge_report, "edges_found", 0),
            )
        except Exception as exc:
            logger.warning("[Phase 7] EdgeDiscovery failed: %s", exc)
        return report
    except Exception as exc:
        logger.exception("[Phase 7] Evolution failed: %s", exc)
        _send_telegram(f"WARNING: Agent evolution failed.\n{exc}")
        return None


# ---------------------------------------------------------------------------
# PHASE 8: REPORT
# ---------------------------------------------------------------------------

def phase_report(
    *,
    target_date: date,
    start_ts: datetime,
    collection_summary: dict,
    resolution: dict,
    predict_result: dict,
    drift_result: Optional[DriftResult],
    evolution_report: Optional[EvolutionReport],
    circuit_level: int,
    errors: list[str],
) -> None:
    """Write comprehensive summary to logs."""
    _phase_banner(8, "REPORT")
    elapsed = _fmt_elapsed(start_ts)

    total_errors = (
        len(collection_summary.get("errors", []))
        + len(resolution.get("errors", []))
        + len(predict_result.get("errors", []))
        + len(errors)
    )

    logger.info("SharpEdge Run Summary")
    logger.info("  Date:         %s", target_date)
    logger.info("  Elapsed:      %s", elapsed)
    logger.info("  Circuit:      level %d", circuit_level)
    logger.info("  Collection:   odds=%d  intelligence=%d",
                collection_summary.get("odds_rows", 0),
                collection_summary.get("intelligence_rows", 0))
    logger.info("  Resolution:   W=%d L=%d V=%d P&L=%.2f",
                resolution.get("won", 0), resolution.get("lost", 0),
                resolution.get("voided", 0), resolution.get("pnl", 0.0))
    logger.info("  Predictions:  fixtures=%d  picks=%d  accas=%d",
                predict_result.get("n_fixtures", 0),
                len(predict_result.get("picks", [])),
                len(predict_result.get("accumulators", [])))
    logger.info("  Total errors: %d", total_errors)

    if total_errors > 0:
        logger.warning("Errors encountered during run:")
        for e in (
            collection_summary.get("errors", [])
            + resolution.get("errors", [])
            + predict_result.get("errors", [])
            + errors
        ):
            logger.warning("  - %s", e)


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SharpEdge production pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scheduler",
        action="store_true",
        help="Run as a persistent APScheduler daemon (Railway worker mode)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override target date (ISO 8601, e.g. 2026-04-02)",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only run data collection phases — skip prediction and resolution",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve yesterday's picks",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip data collection; run prediction pipeline only",
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Force agent evolution cycle regardless of schedule",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run health and drift checks",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full daily pipeline or start the scheduler daemon.

    Returns 0 on success, 1 on critical failure.
    """
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # --scheduler mode: launch persistent APScheduler daemon
    # ------------------------------------------------------------------
    if args.scheduler:
        logger.info("Starting SharpEdge Scheduler daemon...")
        from sharpedge.warroom.scheduler import SharpEdgeScheduler
        SharpEdgeScheduler().start()
        return 0  # only reached after KeyboardInterrupt / SIGTERM

    # ------------------------------------------------------------------
    # Single daily cycle
    # ------------------------------------------------------------------
    if args.date:
        try:
            target_date = date.fromisoformat(args.date)
        except ValueError:
            logger.error("--date must be YYYY-MM-DD, got: %s", args.date)
            return 1
    else:
        target_date = date.today()

    start_ts = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info("  SharpEdge Daily Pipeline  —  %s", target_date)
    logger.info("  Environment: %s", settings.environment)
    logger.info("=" * 70)

    # State accumulators
    collection_summary: dict = {"errors": [], "odds_rows": 0, "intelligence_rows": 0}
    resolution: dict = {"won": 0, "lost": 0, "voided": 0, "pnl": 0.0, "errors": []}
    predict_result: dict = {"picks": [], "accumulators": [], "n_fixtures": 0, "errors": []}
    drift_result: Optional[DriftResult] = None
    evolution_report: Optional[EvolutionReport] = None
    health_status = HealthStatus.HEALTHY
    health_components: list = []
    circuit_level = 0
    run_id: Optional[int] = None
    pipeline_errors: list[str] = []
    exit_code = 0

    try:
        orchestrator = ScraperOrchestrator()

        # ----------------------------------------------------------------
        # PHASE 0: COLLECT DATA
        # ----------------------------------------------------------------
        if not args.predict_only and not args.resolve_only and not args.check_only:
            collection_summary = phase_collect(orchestrator, target_date)

        if args.collect_only:
            elapsed = _fmt_elapsed(start_ts)
            logger.info("collect-only mode complete in %s", elapsed)
            _send_telegram(
                f"SharpEdge: Data collection complete — {target_date}\n"
                f"Elapsed: {elapsed}\n"
                f"Errors: {len(collection_summary.get('errors', []))}"
            )
            return 0

        # ----------------------------------------------------------------
        # PHASE 1: RESOLVE YESTERDAY
        # ----------------------------------------------------------------
        if not args.predict_only and not args.check_only:
            resolution = phase_resolve(target_date)

        if args.resolve_only:
            elapsed = _fmt_elapsed(start_ts)
            pnl = resolution.get("pnl", 0.0)
            _send_telegram(
                f"SharpEdge Resolution — {target_date}\n"
                f"W/L/V: {resolution.get('won',0)}/{resolution.get('lost',0)}/{resolution.get('voided',0)}\n"
                f"P&L: {'+' if pnl >= 0 else ''}{pnl:.2f}u  |  {elapsed}"
            )
            return 0

        # ----------------------------------------------------------------
        # PHASE 2: HEALTH CHECK
        # ----------------------------------------------------------------
        health_status, health_components, circuit_level = phase_health_check()

        if circuit_level == 3:
            msg = (
                f"STOP signal — circuit breaker level 3 ({target_date}).\n"
                "Pipeline aborted. Investigate immediately."
            )
            logger.error(msg)
            _send_telegram(msg)
            if not args.predict_only:
                return 1

        if args.check_only:
            # Also run drift check
            try:
                detector = DriftDetector()
                results = detector.run_check_all()
                drift_result = next(iter(results.values()), None) if results else None
            except Exception as exc:
                logger.warning("Drift check failed: %s", exc)
            elapsed = _fmt_elapsed(start_ts)
            _send_telegram(
                f"SharpEdge Health Check — {target_date}\n"
                f"Status: {health_status.value.upper()}\n"
                f"Circuit: level {circuit_level}\n"
                f"Elapsed: {elapsed}"
            )
            return 0

        # ----------------------------------------------------------------
        # PHASE 3: LOAD CONTEXT DATA
        # ----------------------------------------------------------------
        ctx = phase_load_context(target_date)
        pipeline_errors.extend(ctx.pop("errors", []))

        # ----------------------------------------------------------------
        # PHASE 4: BOOTSTRAP AGENTS
        # ----------------------------------------------------------------
        agents, arbiter = phase_bootstrap_agents(ctx)

        # ----------------------------------------------------------------
        # PHASE 5: PREDICT
        # ----------------------------------------------------------------
        run_id = _open_pipeline_run(target_date, "daily")
        predict_result = phase_predict(
            target_date, ctx, agents, arbiter, run_id, circuit_level
        )
        pipeline_errors.extend(predict_result.get("errors", []))

        _close_pipeline_run(
            run_id,
            "completed",
            fixtures_count=predict_result["n_fixtures"],
            predictions_count=len(predict_result["predictions"]),
            picks_count=len(predict_result["picks"]),
        )

        if args.predict_only:
            elapsed = _fmt_elapsed(start_ts)
            _send_telegram(
                f"SharpEdge: Predict-only — {target_date}\n"
                f"Fixtures: {predict_result['n_fixtures']} | "
                f"Picks: {len(predict_result['picks'])} | {elapsed}"
            )
            return 0

        # ----------------------------------------------------------------
        # PHASE 6: BROADCAST
        # ----------------------------------------------------------------
        elapsed = _fmt_elapsed(start_ts)
        # Run drift check for broadcast
        try:
            detector = DriftDetector()
            all_drift = detector.run_check_all()
            drift_result = next(iter(all_drift.values()), None) if all_drift else None
        except Exception as exc:
            logger.warning("Drift check failed (non-fatal): %s", exc)

        phase_broadcast(
            target_date=target_date,
            predict_result=predict_result,
            resolution=resolution,
            health_status=health_status,
            health_components=health_components,
            drift_result=drift_result,
            evolution_report=None,  # evolution hasn't run yet
            elapsed=elapsed,
        )

        # ----------------------------------------------------------------
        # PHASE 7: EVOLUTION (if due or forced)
        # ----------------------------------------------------------------
        if args.evolve or _should_run_evolution(target_date):
            evolution_report = phase_evolution(target_date)
            if evolution_report:
                # Send brief evolution update if anything changed
                promoted = getattr(evolution_report, "promotions", [])
                deprecated = getattr(evolution_report, "deprecations", [])
                if promoted or deprecated:
                    _send_telegram(
                        f"SharpEdge Evolution — {target_date}\n"
                        f"Promoted: {', '.join(promoted) or 'none'}\n"
                        f"Deprecated: {', '.join(deprecated) or 'none'}"
                    )
        else:
            logger.info("[Phase 7] Evolution not due — skipping")

        # ----------------------------------------------------------------
        # PHASE 8: REPORT
        # ----------------------------------------------------------------
        phase_report(
            target_date=target_date,
            start_ts=start_ts,
            collection_summary=collection_summary,
            resolution=resolution,
            predict_result=predict_result,
            drift_result=drift_result,
            evolution_report=evolution_report,
            circuit_level=circuit_level,
            errors=pipeline_errors,
        )

    except Exception:
        tb = traceback.format_exc()
        logger.error("Unhandled exception in daily pipeline:\n%s", tb)
        _close_pipeline_run(run_id, "failed", error_message=tb[:2000])
        _send_telegram(
            f"CRITICAL: SharpEdge daily pipeline CRASHED — {target_date}\n"
            f"<pre>{tb[:1200]}</pre>"
        )
        exit_code = 1

    elapsed_total = _fmt_elapsed(start_ts)
    logger.info("=" * 70)
    logger.info("  Pipeline finished in %s — exit code %d", elapsed_total, exit_code)
    logger.info("=" * 70)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
