"""Production daily pipeline runner for Railway worker.

Phases
------
1. Bootstrap      — init all agents, load model
2. Load history   — query DB for last-2-season matches
3. Load fixtures  — today + tomorrow scheduled matches with odds
4. Run predictions
5. Resolve yesterday
6. Health + drift check
7. Evolution check (every N days)
8. Report + Telegram broadcast

CLI
---
  --date DATE          Override target date (ISO, default: today)
  --resolve-only       Only resolve yesterday's picks
  --predict-only       Only run predictions
  --check-only         Only run health/drift checks
  --evolve             Force agent evolution cycle regardless of schedule
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Ensure src is on path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sharpedge.run_daily")

# ---------------------------------------------------------------------------
# Core imports — fail fast so Railway logs the real error
# ---------------------------------------------------------------------------

from sharpedge.warroom.orchestrator import WarRoomOrchestrator
from sharpedge.warroom.health_monitor import HealthMonitor, HealthStatus
from sharpedge.agents.market_agent import MarketAgent
from sharpedge.agents.statistical_agent import StatisticalAgent
from sharpedge.agents.form_momentum_agent import FormMomentumAgent
from sharpedge.agents.contrarian_agent import ContrarianAgent
from sharpedge.agents.h2h_venue_agent import H2HVenueAgent
from sharpedge.agents.gradient_agent import GradientAgent
from sharpedge.agents.edge_value_agent import EdgeValueAgent
from sharpedge.alerts.telegram import send_alert_sync
from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.models import (
    DailyPick,
    League,
    Match,
    MatchOdds,
    PipelineRun,
    Season,
)

# RegimeDetectionAgent is not yet on disk; guard so the rest still boots
try:
    from sharpedge.agents.regime_agent import RegimeDetectionAgent  # type: ignore

    _REGIME_AGENT_AVAILABLE = True
except ImportError:
    _REGIME_AGENT_AVAILABLE = False
    logger.warning("RegimeDetectionAgent not found — skipping (import guard active)")

# Evolution engine
from sharpedge.agents.evolution import AgentEvolutionEngine, EvolutionReport

# Drift detector
from sharpedge.pipeline.drift_detector import DriftDetector, DriftResult  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_SHUTDOWN_REQUESTED = False


def _handle_signal(signum: int, _frame) -> None:
    global _SHUTDOWN_REQUESTED
    logger.warning("Signal %d received — requesting graceful shutdown", signum)
    _SHUTDOWN_REQUESTED = False  # set True to abort mid-run if desired


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

_MAX_TG_LENGTH = 4096  # Telegram message hard limit


def _send_telegram(message: str) -> None:
    """Best-effort Telegram broadcast; never raises."""
    try:
        # Split oversized messages so nothing gets silently truncated
        if len(message) > _MAX_TG_LENGTH:
            chunks = [
                message[i : i + _MAX_TG_LENGTH]
                for i in range(0, len(message), _MAX_TG_LENGTH)
            ]
            for chunk in chunks:
                send_alert_sync(chunk)
        else:
            send_alert_sync(message)
    except Exception:
        logger.exception("Telegram broadcast failed (non-fatal)")


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------


def _load_historical_matches(session, cutoff_date: date) -> list[dict]:
    """Return resolved matches since *cutoff_date* as plain dicts."""
    rows = (
        session.query(Match)
        .filter(
            Match.match_date >= cutoff_date,
            Match.result.isnot(None),
        )
        .order_by(Match.match_date)
        .all()
    )
    matches = []
    for m in rows:
        home_team = m.home_team.canonical_name if m.home_team else str(m.home_team_id)
        away_team = m.away_team.canonical_name if m.away_team else str(m.away_team_id)
        season_label = m.season.label if m.season else ""
        league_name = (m.season.league.name if m.season and m.season.league else "")
        matches.append(
            {
                "match_id": str(m.id),
                "home_team_id": home_team,
                "away_team_id": away_team,
                "home_team": home_team,
                "away_team": away_team,
                "league": league_name,
                "match_date": str(m.match_date),
                "FTHG": m.home_goals,
                "FTAG": m.away_goals,
                "FTR": m.result or "",
                "season": season_label,
                "venue": m.venue or "",
                "referee": m.referee or "",
            }
        )
    logger.info("Loaded %d historical matches since %s", len(matches), cutoff_date)
    return matches


def _load_fixtures(session, target_date: date) -> list[dict]:
    """Return scheduled fixtures for target_date and the following day."""
    tomorrow = target_date + timedelta(days=1)
    rows = (
        session.query(Match)
        .filter(
            Match.match_date.in_([target_date, tomorrow]),
            Match.status == "scheduled",
        )
        .order_by(Match.match_date, Match.kick_off_time)
        .all()
    )

    fixtures: list[dict] = []
    for m in rows:
        home_team = m.home_team.canonical_name if m.home_team else str(m.home_team_id)
        away_team = m.away_team.canonical_name if m.away_team else str(m.away_team_id)
        season_label = m.season.label if m.season else ""
        league_name = (m.season.league.name if m.season and m.season.league else "")

        # Collect odds — prefer Pinnacle, fall back to B365, then first available
        odds_map: dict[str, dict] = {}
        for o in (m.odds or []):
            key = f"{o.bookmaker}_{o.market}_{o.odds_type}"
            odds_map[key] = o

        def _best_odds(bk_priority: list[str], suffix: str) -> Optional[float]:
            for bk in bk_priority:
                key = f"{bk}_1x2_opening"
                row = odds_map.get(key)
                if row:
                    val = getattr(row, suffix, None)
                    if val is not None:
                        return float(val)
            return None

        BK = ["Pinnacle", "Bet365", "William Hill", "Betfair"]
        fixture: dict = {
            "match_id": str(m.id),
            "home_team_id": home_team,
            "away_team_id": away_team,
            "home_team": home_team,
            "away_team": away_team,
            "league": league_name,
            "match_date": str(m.match_date),
            "kick_off_time": m.kick_off_time or "",
            "season": season_label,
            "venue": m.venue or "",
            "referee": m.referee or "",
        }
        # Attach odds columns expected by DailyPipeline.predict()
        for col, attr in [
            ("B365H", "odds_home"), ("B365D", "odds_draw"), ("B365A", "odds_away"),
        ]:
            for bk_key in [k for k in odds_map if "Bet365" in k or "B365" in k]:
                row = odds_map[bk_key]
                if getattr(row, attr, None) is not None:
                    fixture[col] = float(getattr(row, attr))
                    break

        for col, attr in [
            ("PSH", "odds_home"), ("PSD", "odds_draw"), ("PSA", "odds_away"),
        ]:
            for bk_key in [k for k in odds_map if "Pinnacle" in k]:
                row = odds_map[bk_key]
                if getattr(row, attr, None) is not None:
                    fixture[col] = float(getattr(row, attr))
                    break

        fixtures.append(fixture)

    logger.info("Loaded %d fixtures for %s / %s", len(fixtures), target_date, tomorrow)
    return fixtures


def _bootstrap_agents(historical_matches: list[dict]) -> list:
    """Initialise and fit all agents. Returns list of fitted agent instances."""
    import pandas as pd

    agents = []
    hist_df = pd.DataFrame(historical_matches)

    # --- No-training agents ---
    agents.append(MarketAgent())
    agents.append(ContrarianAgent())
    agents.append(EdgeValueAgent())
    logger.info("Market / Contrarian / EdgeValue agents initialised (no training needed)")

    # --- Statistical agent ---
    try:
        stat = StatisticalAgent()
        stat.fit(historical_matches)
        agents.append(stat)
        logger.info("StatisticalAgent fitted on %d matches", len(historical_matches))
    except Exception:
        logger.exception("StatisticalAgent fit failed — skipping")

    # --- Form momentum agent ---
    try:
        form = FormMomentumAgent()
        form.fit(hist_df)
        agents.append(form)
        logger.info("FormMomentumAgent fitted")
    except Exception:
        logger.exception("FormMomentumAgent fit failed — skipping")

    # --- H2H / venue agent ---
    try:
        h2h = H2HVenueAgent()
        h2h.fit(historical_matches)
        agents.append(h2h)
        logger.info("H2HVenueAgent fitted")
    except Exception:
        logger.exception("H2HVenueAgent fit failed — skipping")

    # --- Gradient agent (needs feature matrix from model) ---
    try:
        from sharpedge.ml.training.trainer import ModelTrainer

        trainer = ModelTrainer.load(settings.model_path)
        grad = GradientAgent()
        # GradientAgent.fit expects (X_train, y_train_1x2)
        if trainer.X_train is not None and trainer.y_train is not None:
            grad.fit(trainer.X_train, trainer.y_train)
            agents.append(grad)
            logger.info("GradientAgent fitted from loaded model training data")
        else:
            logger.warning("GradientAgent skipped — model has no cached training data")
    except Exception:
        logger.exception("GradientAgent fit failed — skipping")

    # --- Regime detection agent (optional) ---
    if _REGIME_AGENT_AVAILABLE:
        try:
            regime = RegimeDetectionAgent()  # type: ignore[name-defined]
            regime.fit(historical_matches)
            agents.append(regime)
            logger.info("RegimeDetectionAgent fitted")
        except Exception:
            logger.exception("RegimeDetectionAgent fit failed — skipping")

    logger.info("Bootstrap complete: %d agents ready", len(agents))
    return agents


def _resolve_yesterday(target_date: date) -> dict:
    """Resolve DailyPick rows for yesterday against actual results.

    Returns summary dict: {wins, losses, voids, pnl}.
    """
    from sharpedge.pipeline.resolver import resolve_pick

    yesterday = target_date - timedelta(days=1)
    session = get_session()
    summary = {"wins": 0, "losses": 0, "voids": 0, "pnl": 0.0, "picks": []}
    try:
        picks = (
            session.query(DailyPick)
            .filter(
                DailyPick.match_date == yesterday,
                DailyPick.result.is_(None),
            )
            .all()
        )
        logger.info("Resolving %d unresolved picks for %s", len(picks), yesterday)

        for pick in picks:
            # Find the match result from the DB
            match_row = (
                session.query(Match)
                .filter(
                    Match.home_team.has(canonical_name=pick.home_team),
                    Match.away_team.has(canonical_name=pick.away_team),
                    Match.match_date == yesterday,
                )
                .first()
            )
            if match_row is None or match_row.home_goals is None:
                summary["voids"] += 1
                continue

            outcome = resolve_pick(
                pick_market=pick.pick_market,
                home_goals=match_row.home_goals,
                away_goals=match_row.away_goals,
                best_odds=pick.best_odds,
                stake=pick.stake_flat or 1.0,
            )
            pick.result = outcome["result"]
            pick.profit_loss = outcome["profit_loss"]
            pick.resolved_at = datetime.now(timezone.utc)

            if outcome["result"] == "win":
                summary["wins"] += 1
            elif outcome["result"] == "loss":
                summary["losses"] += 1
            else:
                summary["voids"] += 1

            summary["pnl"] += outcome["profit_loss"]
            summary["picks"].append(
                {
                    "match": f"{pick.home_team} v {pick.away_team}",
                    "market": pick.pick_market,
                    "result": outcome["result"],
                    "pnl": outcome["profit_loss"],
                    "odds": pick.best_odds,
                    "tier": pick.tier,
                }
            )

        session.commit()
        logger.info(
            "Resolution complete: W=%d L=%d V=%d P&L=%.2f",
            summary["wins"],
            summary["losses"],
            summary["voids"],
            summary["pnl"],
        )
    except Exception:
        session.rollback()
        logger.exception("Resolution DB transaction failed — rolled back")
    finally:
        session.close()

    return summary


def _run_drift_check(target_date: date) -> Optional[DriftResult]:
    """Run the drift detector for the football sport slug."""
    try:
        detector = DriftDetector()
        result = detector.check("football", window_days=settings.drift_window_days)
        logger.info(
            "Drift check: severity=%s ECE=%.4f Brier=%.4f acc=%.3f n=%d",
            result.severity,
            result.calibration_error,
            result.brier_score,
            result.accuracy,
            result.n_predictions,
        )
        return result
    except Exception:
        logger.exception("Drift check failed (non-fatal)")
        return None


def _should_run_evolution(target_date: date) -> bool:
    """Return True if the evolution interval has elapsed since last run."""
    interval = settings.agent_evolution_interval_days
    session = get_session()
    try:
        from sharpedge.db.models import AgentEvolution

        latest = (
            session.query(AgentEvolution.event_date)
            .order_by(AgentEvolution.event_date.desc())
            .first()
        )
        if latest is None:
            return True
        last_run: date = latest[0]
        delta = (target_date - last_run).days
        logger.info(
            "Last evolution run: %s (%d days ago, interval=%d)",
            last_run,
            delta,
            interval,
        )
        return delta >= interval
    except Exception:
        logger.exception("Evolution schedule check failed — defaulting to skip")
        return False
    finally:
        session.close()


def _run_evolution_cycle() -> Optional[EvolutionReport]:
    """Execute one full evolution cycle for football."""
    try:
        engine = AgentEvolutionEngine()
        report = engine.run_natural_selection("football")
        logger.info(
            "Evolution cycle complete: promotions=%s deprecations=%s probations=%s edges=%d",
            report.promotions,
            report.deprecations,
            report.probations,
            len(report.edges_discovered),
        )
        return report
    except Exception:
        logger.exception("Evolution cycle failed (non-fatal)")
        return None


# ---------------------------------------------------------------------------
# Telegram report builder
# ---------------------------------------------------------------------------

def _build_report(
    *,
    target_date: date,
    elapsed_s: float,
    health_status: HealthStatus,
    health_components: list,
    n_fixtures: int,
    pipeline_report,
    resolution: dict,
    drift: Optional[DriftResult],
    evolution: Optional[EvolutionReport],
    picks: list[dict],
) -> str:
    """Compose the full Telegram daily digest."""
    lines: list[str] = []

    # --- Header ---
    env_tag = f" [{settings.environment.upper()}]" if settings.environment != "production" else ""
    lines.append(f"<b>SharpEdge Daily Digest{env_tag} — {target_date}</b>")
    lines.append(f"Run time: {elapsed_s:.0f}s")
    lines.append("")

    # --- 📊 Pipeline summary ---
    lines.append("📊 <b>Pipeline</b>")
    lines.append(f"  Fixtures loaded: {n_fixtures}")
    if pipeline_report:
        lines.append(f"  Matches processed: {pipeline_report.total_matches}")
        lines.append(f"  Predictions: {pipeline_report.total_predictions}")
        lines.append(f"  Picks generated: {pipeline_report.total_picks}")
    health_emoji = "✅" if health_status == HealthStatus.HEALTHY else ("⚠️" if health_status == HealthStatus.DEGRADED else "🚨")
    lines.append(f"  Health: {health_emoji} {health_status.value.upper()}")
    lines.append("")

    # --- 🏆 Top picks by tier ---
    if picks:
        lines.append("🏆 <b>Today's Picks</b>")
        by_tier: dict[str, list[dict]] = {}
        for p in picks:
            by_tier.setdefault(p.get("tier", "standard"), []).append(p)
        for tier in ("diamond", "platinum", "gold", "standard"):
            tier_picks = by_tier.get(tier, [])
            if not tier_picks:
                continue
            lines.append(f"  <i>{tier.capitalize()}</i> ({len(tier_picks)})")
            # Show top 5 per tier to keep message size manageable
            for p in tier_picks[:5]:
                edge_str = f"+{p['edge']*100:.1f}%" if p.get("edge") else ""
                odds_str = f"@ {p['best_odds']:.2f}" if p.get("best_odds") else ""
                lines.append(
                    f"    • {p['home_team']} v {p['away_team']} | "
                    f"{p['pick_selection']} {odds_str} {edge_str}"
                )
        lines.append("")
    else:
        lines.append("🏆 <b>Picks:</b> none today")
        lines.append("")

    # --- 💰 Yesterday's results ---
    lines.append("💰 <b>Yesterday's Results</b>")
    if not resolution.get("picks") and resolution.get("wins", 0) + resolution.get("losses", 0) + resolution.get("voids", 0) == 0:
        lines.append("  No resolved picks")
    else:
        wins = resolution.get("wins", 0)
        losses = resolution.get("losses", 0)
        voids = resolution.get("voids", 0)
        pnl = resolution.get("pnl", 0.0)
        total = wins + losses + voids
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
        pnl_sign = "+" if pnl >= 0 else ""
        lines.append(f"  W/L/V: {wins}/{losses}/{voids} (total: {total})")
        lines.append(f"  Win rate: {win_rate:.1f}%")
        lines.append(f"  P&L: {pnl_sign}{pnl:.2f}u")
        # Show resolved picks detail (top 10)
        for p in resolution.get("picks", [])[:10]:
            icon = "✅" if p["result"] == "win" else ("❌" if p["result"] == "loss" else "⬛")
            pnl_s = f"{'+' if p['pnl'] >= 0 else ''}{p['pnl']:.2f}u"
            lines.append(f"  {icon} {p['match']} [{p['market']}] {pnl_s}")
    lines.append("")

    # --- 📈 CLV tracking ---
    lines.append("📈 <b>CLV Tracking</b>")
    clv_target = settings.clv_target_pct
    lines.append(f"  Target: +{clv_target:.1f}% vs closing line")
    lines.append("  (CLV per-pick tracked in DB — query daily_picks.clv)")
    lines.append("")

    # --- ⚠️ Drift warnings ---
    if drift:
        if drift.has_drift:
            lines.append(f"⚠️ <b>Drift Detected</b>: severity={drift.severity.upper()}")
            lines.append(f"  ECE={drift.calibration_error:.4f} | Brier={drift.brier_score:.4f}")
            lines.append(f"  Accuracy={drift.accuracy:.3f} | n={drift.n_predictions}")
            lines.append(f"  Recommendation: {drift.recommendation}")
        else:
            lines.append(f"✅ <b>No Drift</b>: ECE={drift.calibration_error:.4f} | Brier={drift.brier_score:.4f} | n={drift.n_predictions}")
        lines.append("")

    # --- 🧬 Evolution events ---
    if evolution:
        has_events = any(
            [evolution.promotions, evolution.deprecations, evolution.probations]
        )
        lines.append("🧬 <b>Agent Evolution</b>")
        if evolution.promotions:
            lines.append(f"  Promoted: {', '.join(evolution.promotions)}")
        if evolution.deprecations:
            lines.append(f"  Deprecated: {', '.join(evolution.deprecations)}")
        if evolution.probations:
            lines.append(f"  Probation: {', '.join(evolution.probations)}")
        sig_edges = [e for e in evolution.edges_discovered if e.is_significant]
        if sig_edges:
            lines.append(f"  Significant edges: {len(sig_edges)}")
            for e in sig_edges[:3]:
                lines.append(
                    f"    • {e.agent_name} | {e.league} | {e.market}: "
                    f"+{e.edge_value*100:.1f}% (p={e.p_value:.3f})"
                )
        if not has_events and not sig_edges:
            lines.append("  No changes this cycle")
        lines.append("")
    elif _should_run_evolution(target_date):
        # Mentioned but not run (only triggered when flag passed or interval due)
        pass

    # --- Degraded component warnings ---
    degraded = [
        c for c in health_components if c.status != HealthStatus.HEALTHY
    ]
    if degraded:
        lines.append("⚠️ <b>Component Warnings</b>")
        for c in degraded:
            icon = "🚨" if c.status == HealthStatus.DOWN else "⚠️"
            lines.append(f"  {icon} {c.name}: {c.message}")
        lines.append("")

    lines.append(f"<i>SharpEdge · {datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DB persistence helpers
# ---------------------------------------------------------------------------

def _open_pipeline_run(target_date: date, run_type: str) -> Optional[int]:
    """Insert a PipelineRun row and return its ID."""
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
        logger.info("PipelineRun created: id=%d run_type=%s", run_id, run_type)
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
    """Update PipelineRun to completed / failed."""
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
        logger.exception("Failed to update PipelineRun record")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Main phases
# ---------------------------------------------------------------------------


def phase_bootstrap_and_predict(
    target_date: date,
    fixtures: list[dict],
    historical_matches: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Run bootstrap + prediction phase.

    Returns (predictions, picks) where both are plain dicts.
    """
    agents = _bootstrap_agents(historical_matches)

    # DailyPipeline owns the core ML prediction stack
    from sharpedge.pipeline.daily import DailyPipeline
    import pandas as pd

    pipeline = DailyPipeline(model_path=settings.model_path)
    pipeline.load_model()

    hist_df = pd.DataFrame(historical_matches) if historical_matches else None
    predictions = pipeline.predict(fixtures, historical_matches=hist_df)
    logger.info("Predictions generated: %d", len(predictions))

    picks_raw = pipeline.filter_picks(predictions)
    logger.info("Picks after banker filter: %d", len(picks_raw))

    # Serialise Pick objects to plain dicts for later use / DB writes
    picks: list[dict] = []
    for p in picks_raw:
        picks.append(
            {
                "home_team": getattr(p, "home_team", ""),
                "away_team": getattr(p, "away_team", ""),
                "league": getattr(p, "league", ""),
                "match_date": getattr(p, "match_date", str(target_date)),
                "pick_market": getattr(p, "market", ""),
                "pick_selection": getattr(p, "selection", ""),
                "model_prob": float(getattr(p, "model_prob", 0)),
                "model_spread": float(getattr(p, "model_spread", 0)),
                "best_odds": float(getattr(p, "best_odds", 1.0)),
                "bookmaker": getattr(p, "bookmaker", ""),
                "implied_prob": (
                    1.0 / float(getattr(p, "best_odds", 2.0))
                    if getattr(p, "best_odds", 0) > 1.0
                    else 0.0
                ),
                "edge": float(getattr(p, "edge", 0)),
                "tier": getattr(p, "tier", "standard"),
                "meta_agreement": int(getattr(p, "meta_agreement", 0)),
                "risk_flags": getattr(p, "risk_flags", []),
                "stake_flat": float(getattr(p, "stake_flat", 0) or 0),
                "stake_kelly": float(getattr(p, "stake_kelly", 0) or 0),
            }
        )

    return predictions, picks


def _persist_picks(picks: list[dict], run_id: Optional[int]) -> None:
    """Write DailyPick rows to DB for today's selections."""
    if not picks or run_id is None:
        return
    session = get_session()
    try:
        # Need a Prediction FK — use a placeholder approach: insert one
        # aggregate Prediction per fixture (simplified) or reuse existing.
        # Here we do a minimal insert without a Prediction row to keep this
        # self-contained; a full implementation would insert Prediction rows
        # separately in phase_bootstrap_and_predict.
        from sharpedge.db.models import Prediction
        from datetime import date as date_type

        inserted = 0
        for p in picks:
            raw_date = p.get("match_date", str(date.today()))
            try:
                match_date_val = (
                    date_type.fromisoformat(str(raw_date)[:10])
                )
            except Exception:
                match_date_val = date.today()

            # Upsert-style: skip if a pick for this fixture+market already exists
            existing = (
                session.query(DailyPick)
                .join(PipelineRun, DailyPick.pipeline_run_id == PipelineRun.id)
                .filter(
                    DailyPick.home_team == p["home_team"],
                    DailyPick.away_team == p["away_team"],
                    DailyPick.match_date == match_date_val,
                    DailyPick.pick_market == p["pick_market"],
                    PipelineRun.run_date == date.today(),
                )
                .first()
            )
            if existing:
                continue

            # Create a minimal Prediction placeholder
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
        logger.info("Persisted %d new picks to DB (run_id=%d)", inserted, run_id)
    except Exception:
        session.rollback()
        logger.exception("Failed to persist picks — rolled back")
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SharpEdge daily prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Override target date (ISO 8601, e.g. 2026-04-02)",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve yesterday's picks — skip predictions",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only run predictions — skip resolution and evolution",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only run health and drift checks",
    )
    parser.add_argument(
        "--evolve",
        action="store_true",
        help="Force an agent evolution cycle regardless of schedule",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the full daily pipeline. Returns 0 on success, 1 on failure."""
    args = parse_args(argv)

    # Parse target date
    if args.date:
        try:
            target_date = date.fromisoformat(args.date)
        except ValueError:
            logger.error("--date must be ISO format (YYYY-MM-DD), got: %s", args.date)
            return 1
    else:
        target_date = date.today()

    start_ts = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info("  SharpEdge Daily Pipeline  —  %s", target_date)
    logger.info("  Environment: %s", settings.environment)
    logger.info("=" * 70)

    # State collectors
    pipeline_report = None
    resolution: dict = {}
    drift_result: Optional[DriftResult] = None
    evolution_report: Optional[EvolutionReport] = None
    picks: list[dict] = []
    n_fixtures = 0
    run_id: Optional[int] = None
    exit_code = 0

    try:
        # ------------------------------------------------------------------
        # 1. Health check (always runs unless --predict-only)
        # ------------------------------------------------------------------
        monitor = HealthMonitor()
        health = monitor.run_checks()
        logger.info(
            "Health: %s (%d components checked)",
            health.overall.value,
            len(health.components),
        )
        for comp in health.components:
            level = logging.WARNING if comp.status != HealthStatus.HEALTHY else logging.DEBUG
            logger.log(level, "  [%s] %s — %s", comp.status.value, comp.name, comp.message)

        if health.overall == HealthStatus.DOWN and not args.predict_only:
            msg = (
                f"🚨 SharpEdge health check FAILED ({target_date})\n"
                "Pipeline aborted — check logs immediately."
            )
            logger.error(msg)
            _send_telegram(msg)
            return 1

        if args.check_only:
            drift_result = _run_drift_check(target_date)
            elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()
            report = _build_report(
                target_date=target_date,
                elapsed_s=elapsed,
                health_status=health.overall,
                health_components=health.components,
                n_fixtures=0,
                pipeline_report=None,
                resolution={},
                drift=drift_result,
                evolution=None,
                picks=[],
            )
            logger.info("Check-only mode complete")
            _send_telegram(report)
            return 0

        # ------------------------------------------------------------------
        # 2. Resolve yesterday (skip if --predict-only)
        # ------------------------------------------------------------------
        if not args.predict_only:
            resolution = _resolve_yesterday(target_date)

        if args.resolve_only:
            elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()
            pnl = resolution.get("pnl", 0.0)
            msg = (
                f"💰 Resolution only — {target_date}\n"
                f"W/L/V: {resolution.get('wins',0)}/{resolution.get('losses',0)}/{resolution.get('voids',0)}\n"
                f"P&L: {'+' if pnl >= 0 else ''}{pnl:.2f}u  |  {elapsed:.0f}s"
            )
            logger.info(msg)
            _send_telegram(msg)
            return 0

        # ------------------------------------------------------------------
        # 3. Load data
        # ------------------------------------------------------------------
        cutoff = target_date - timedelta(days=2 * 365)  # last ~2 seasons
        session = get_session()
        try:
            historical_matches = _load_historical_matches(session, cutoff)
            fixtures = _load_fixtures(session, target_date)
            n_fixtures = len(fixtures)
        finally:
            session.close()

        # ------------------------------------------------------------------
        # 4. Predictions
        # ------------------------------------------------------------------
        run_id = _open_pipeline_run(target_date, run_type="daily")

        if fixtures:
            predictions, picks = phase_bootstrap_and_predict(
                target_date, fixtures, historical_matches
            )
            _persist_picks(picks, run_id)
            _close_pipeline_run(
                run_id,
                "completed",
                fixtures_count=n_fixtures,
                predictions_count=len(predictions),
                picks_count=len(picks),
            )

            # Also wire into WarRoomOrchestrator for future multi-sport extension
            orchestrator = WarRoomOrchestrator()
            orchestrator.register_sport_pipeline("football", {"agents": []})
            pipeline_report = orchestrator.run_daily()
        else:
            logger.warning("No fixtures found for %s / %s — skipping predictions", target_date, target_date + timedelta(days=1))
            _close_pipeline_run(run_id, "completed", fixtures_count=0)

        if args.predict_only:
            elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()
            msg = (
                f"🔮 Predict-only — {target_date}\n"
                f"Fixtures: {n_fixtures} | Picks: {len(picks)} | {elapsed:.0f}s"
            )
            _send_telegram(msg)
            return 0

        # ------------------------------------------------------------------
        # 5. Drift check
        # ------------------------------------------------------------------
        drift_result = _run_drift_check(target_date)

        # ------------------------------------------------------------------
        # 6. Evolution
        # ------------------------------------------------------------------
        if args.evolve or _should_run_evolution(target_date):
            logger.info("Running agent evolution cycle")
            evolution_report = _run_evolution_cycle()
        else:
            logger.info("Agent evolution skipped (not due yet)")

        # ------------------------------------------------------------------
        # 7. Build + send final report
        # ------------------------------------------------------------------
        elapsed = (datetime.now(timezone.utc) - start_ts).total_seconds()
        report = _build_report(
            target_date=target_date,
            elapsed_s=elapsed,
            health_status=health.overall,
            health_components=health.components,
            n_fixtures=n_fixtures,
            pipeline_report=pipeline_report,
            resolution=resolution,
            drift=drift_result,
            evolution=evolution_report,
            picks=picks,
        )
        logger.info("=" * 70)
        logger.info("DAILY DIGEST\n%s", report)
        logger.info("=" * 70)
        _send_telegram(report)

    except Exception:
        tb = traceback.format_exc()
        logger.error("Unhandled exception in daily pipeline:\n%s", tb)
        _close_pipeline_run(run_id, "failed", error_message=tb[:2000])
        _send_telegram(
            f"🚨 SharpEdge daily pipeline CRASHED — {target_date}\n"
            f"<pre>{tb[:1200]}</pre>"
        )
        exit_code = 1

    elapsed_total = (datetime.now(timezone.utc) - start_ts).total_seconds()
    logger.info("Pipeline finished in %.1fs — exit code %d", elapsed_total, exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
