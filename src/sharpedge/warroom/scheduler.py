"""SharpEdge Auto-Scheduler — persistent daemon for Railway.

Runs every subsystem on its own cadence using APScheduler.
Never crashes: every job catches all exceptions and alerts via Telegram.

Start as a daemon:
    python -m sharpedge.warroom.run_daily --scheduler
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure src on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from sharpedge.alerts.telegram import send_alert_sync
from sharpedge.collectors.orchestrator import ScraperOrchestrator
from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.ingest import ingest_dataframe
from sharpedge.db.models import Match, RawStagingRecord
from sharpedge.pipeline.drift_detector import DriftDetector, SmartRetrainer
from sharpedge.pipeline.live_resolver import LiveResolver
from sharpedge.pipeline.performance_ledger import PerformanceLedger
from sharpedge.warroom.orchestrator import WarRoomOrchestrator

logger = logging.getLogger("sharpedge.scheduler")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _fmt_elapsed(start: float) -> str:
    secs = time.perf_counter() - start
    mins, s = divmod(int(secs), 60)
    return f"{mins}m{s:02d}s" if mins else f"{s}s"


def _today_has_matches() -> bool:
    """Return True if the DB has matches scheduled for today UTC."""
    today = _utcnow().date()
    try:
        with get_session() as session:
            count = (
                session.query(Match)
                .filter(Match.match_date == today)
                .count()
            )
            return count > 0
    except Exception as exc:
        logger.warning(f"[scheduler] _today_has_matches check failed: {exc}")
        return True  # safe default: assume there are matches


def _new_lineups_since(last_check: Optional[datetime]) -> list[int]:
    """Return match IDs with newly confirmed lineups since *last_check*."""
    if last_check is None:
        return []
    try:
        with get_session() as session:
            rows = (
                session.query(RawStagingRecord)
                .filter(
                    RawStagingRecord.record_type == "lineup",
                    RawStagingRecord.status == "ingested",
                    RawStagingRecord.created_at > last_check,
                )
                .all()
            )
            match_ids: list[int] = []
            for row in rows:
                if row.raw_data and "match_id" in row.raw_data:
                    match_ids.append(int(row.raw_data["match_id"]))
            return list(set(match_ids))
    except Exception as exc:
        logger.warning(f"[scheduler] _new_lineups_since check failed: {exc}")
        return []


# ---------------------------------------------------------------------------
# SharpEdgeScheduler
# ---------------------------------------------------------------------------

class SharpEdgeScheduler:
    """APScheduler-based daemon that drives the full SharpEdge pipeline.

    Schedule (all times UTC):
      - Full data collection       06:00, 10:00, 14:00, 18:00, 22:00
      - Odds collection            Every 30 min on match days
      - Predictions                07:00 daily
      - Lineup check               Every 15 min, 10:00-21:00 UTC
      - Result resolution          01:00 daily
      - Drift check                03:00 daily
      - Evolution                  Every 14 days at 04:00 UTC
      - Intelligence collection    06:00 daily
    """

    def __init__(self) -> None:
        self._scheduler = BlockingScheduler(timezone="UTC")
        self._orchestrator = ScraperOrchestrator()
        self._warroom = WarRoomOrchestrator()
        self._drift_detector = DriftDetector()
        self._retrainer = SmartRetrainer()
        self._ledger = PerformanceLedger()
        self._resolver = LiveResolver()
        # Track last lineup check timestamp for delta detection
        self._last_lineup_check: Optional[datetime] = None

    # -----------------------------------------------------------------------
    # Job implementations
    # -----------------------------------------------------------------------

    def _job_collect_full(self) -> None:
        """Full data collection across all 31+ collectors, then ingest to DB."""
        start = time.perf_counter()
        logger.info("[scheduler] JOB START: full data collection")
        try:
            report = self._orchestrator.run_all()
            # Ingest each source's DataFrame into the DB
            for source_name, source_info in report.by_source.items():
                df = source_info.get("df")
                if df is not None and not df.empty:
                    try:
                        rows_ingested = ingest_dataframe(df, source_name)
                        logger.info(
                            f"[scheduler] Ingested {rows_ingested} rows from {source_name}"
                        )
                    except Exception as ingest_err:
                        logger.warning(
                            f"[scheduler] Ingest failed for {source_name}: {ingest_err}"
                        )
            elapsed = _fmt_elapsed(start)
            summary = (
                f"Full collection complete in {elapsed}.\n"
                f"Sources: {report.sources_succeeded}/{report.sources_attempted} ok, "
                f"{report.total_rows} rows total."
            )
            logger.info(f"[scheduler] JOB END: {summary}")
            if report.sources_failed > 0:
                send_alert_sync(
                    f"SharpEdge: Full collection — {report.sources_failed} sources failed.\n{summary}"
                )
        except Exception as exc:
            elapsed = _fmt_elapsed(start)
            logger.exception(f"[scheduler] JOB FAILED: full collection after {elapsed}: {exc}")
            send_alert_sync(f"CRITICAL: Full data collection crashed after {elapsed}.\n{exc}")

    def _job_collect_odds(self) -> None:
        """Odds-only collection (BetExplorer + OddsPortal + Betfair exchange)."""
        if not _today_has_matches():
            logger.info("[scheduler] Skipping odds collection — no matches today")
            return

        start = time.perf_counter()
        logger.info("[scheduler] JOB START: odds collection")
        try:
            odds_df = self._orchestrator.run_odds_collection()

            # Exchange liquidity via Betfair
            exchange_df = None
            try:
                from sharpedge.collectors.betfair_exchange import BetfairExchangeCollector  # type: ignore
                exchange = BetfairExchangeCollector()
                exchange_df = exchange.collect()
            except Exception as exc_ex:
                logger.warning(f"[scheduler] Betfair exchange collection skipped: {exc_ex}")

            rows = 0
            if odds_df is not None and not odds_df.empty:
                rows += ingest_dataframe(odds_df, "odds_collection")
            if exchange_df is not None and not exchange_df.empty:
                rows += ingest_dataframe(exchange_df, "betfair_exchange")

            elapsed = _fmt_elapsed(start)
            logger.info(
                f"[scheduler] JOB END: odds collection — {rows} rows ingested in {elapsed}"
            )
        except Exception as exc:
            elapsed = _fmt_elapsed(start)
            logger.exception(f"[scheduler] JOB FAILED: odds collection after {elapsed}: {exc}")
            send_alert_sync(f"WARNING: Odds collection failed after {elapsed}.\n{exc}")

    def _job_predict(self) -> None:
        """Run the full prediction pipeline and broadcast picks via Telegram."""
        start = time.perf_counter()
        logger.info("[scheduler] JOB START: predictions")
        try:
            today = _utcnow().date()
            report = self._warroom.run_full_cycle(target_date=today)
            elapsed = _fmt_elapsed(start)

            summary = (
                f"Predictions complete in {elapsed}.\n"
                f"Matches: {report.total_matches} | "
                f"Picks: {report.total_picks} | "
                f"Bankroll: {report.bankroll:.2f}"
            )
            logger.info(f"[scheduler] JOB END: {summary}")
            send_alert_sync(f"SharpEdge Daily Picks\n\n{summary}")
        except Exception as exc:
            elapsed = _fmt_elapsed(start)
            logger.exception(f"[scheduler] JOB FAILED: predictions after {elapsed}: {exc}")
            send_alert_sync(f"CRITICAL: Prediction pipeline crashed after {elapsed}.\n{exc}")

    def _job_check_lineups(self) -> None:
        """Check for newly confirmed lineups; re-run predictions for affected matches."""
        start = time.perf_counter()
        logger.info("[scheduler] JOB START: lineup check")
        try:
            # Only active in the 10:00-21:00 UTC window (APScheduler cron handles this)
            lineup_df = self._orchestrator.run_fixtures_collection()

            new_match_ids = _new_lineups_since(self._last_lineup_check)
            self._last_lineup_check = _utcnow()

            if lineup_df is not None and not lineup_df.empty:
                ingest_dataframe(lineup_df, "lineup_check")

            elapsed = _fmt_elapsed(start)
            if new_match_ids:
                logger.info(
                    f"[scheduler] New lineups for match IDs {new_match_ids} — "
                    f"triggering re-prediction"
                )
                try:
                    today = _utcnow().date()
                    self._warroom.run_full_cycle(
                        target_date=today,
                        match_ids=new_match_ids,
                    )
                    send_alert_sync(
                        f"Lineup update: re-ran predictions for "
                        f"{len(new_match_ids)} match(es)."
                    )
                except Exception as repredict_exc:
                    logger.warning(
                        f"[scheduler] Re-prediction after lineup update failed: {repredict_exc}"
                    )
                    send_alert_sync(
                        f"WARNING: Lineup re-prediction failed.\n{repredict_exc}"
                    )
            else:
                logger.info(
                    f"[scheduler] JOB END: lineup check ({elapsed}) — no new lineups"
                )
        except Exception as exc:
            elapsed = _fmt_elapsed(start)
            logger.exception(f"[scheduler] JOB FAILED: lineup check after {elapsed}: {exc}")
            send_alert_sync(f"WARNING: Lineup check crashed after {elapsed}.\n{exc}")

    def _job_resolve(self) -> None:
        """Resolve yesterday's picks, update bankroll, snapshot agent performance."""
        start = time.perf_counter()
        logger.info("[scheduler] JOB START: result resolution")
        try:
            resolution_result = self._resolver.run()
            elapsed = _fmt_elapsed(start)

            resolved = getattr(resolution_result, "resolved_count", 0)
            won = getattr(resolution_result, "won_count", 0)
            profit = getattr(resolution_result, "profit", 0.0)
            clv = getattr(resolution_result, "avg_clv_pct", None)

            summary_lines = [
                f"Resolution complete in {elapsed}.",
                f"Resolved: {resolved} picks | Won: {won} | P&L: {profit:+.2f}",
            ]
            if clv is not None:
                summary_lines.append(f"Avg CLV: {clv:+.2f}%")

            summary = "\n".join(summary_lines)
            logger.info(f"[scheduler] JOB END: {summary}")

            # Snapshot agent performance
            try:
                self._ledger.snapshot_daily()
            except Exception as snap_exc:
                logger.warning(f"[scheduler] Performance snapshot failed: {snap_exc}")

            send_alert_sync(f"SharpEdge Results\n\n{summary}")
        except Exception as exc:
            elapsed = _fmt_elapsed(start)
            logger.exception(f"[scheduler] JOB FAILED: resolution after {elapsed}: {exc}")
            send_alert_sync(f"CRITICAL: Result resolution crashed after {elapsed}.\n{exc}")

    def _job_drift_check(self) -> None:
        """Check model calibration and trigger retrain if needed."""
        start = time.perf_counter()
        logger.info("[scheduler] JOB START: drift check")
        try:
            drift_results = self._drift_detector.run_check_all()
            elapsed = _fmt_elapsed(start)

            drifted = [k for k, v in drift_results.items() if getattr(v, "drift_detected", False)]
            if drifted:
                logger.warning(f"[scheduler] Drift detected in models: {drifted}")
                send_alert_sync(
                    f"Model drift detected in: {', '.join(drifted)}.\n"
                    f"Triggering retraining."
                )
                try:
                    self._retrainer.retrain_all()
                    logger.info("[scheduler] Retraining triggered successfully")
                except Exception as retrain_exc:
                    logger.error(f"[scheduler] Retraining failed: {retrain_exc}")
                    send_alert_sync(f"WARNING: Auto-retrain failed.\n{retrain_exc}")
            else:
                logger.info(
                    f"[scheduler] JOB END: drift check ({elapsed}) — all models calibrated"
                )
        except Exception as exc:
            elapsed = _fmt_elapsed(start)
            logger.exception(f"[scheduler] JOB FAILED: drift check after {elapsed}: {exc}")
            send_alert_sync(f"WARNING: Drift check crashed after {elapsed}.\n{exc}")

    def _job_evolution(self) -> None:
        """Run agent natural selection and edge discovery."""
        start = time.perf_counter()
        logger.info("[scheduler] JOB START: agent evolution")
        try:
            from sharpedge.agents.evolution import AgentEvolutionEngine
            from sharpedge.intelligence.edge_discovery import EdgeDiscoveryEngine

            evolution_engine = AgentEvolutionEngine()
            evolution_report = evolution_engine.run_natural_selection()

            edge_engine = EdgeDiscoveryEngine()
            edge_report = edge_engine.discover()

            elapsed = _fmt_elapsed(start)
            promoted = getattr(evolution_report, "promoted", [])
            culled = getattr(evolution_report, "culled", [])
            edges_found = getattr(edge_report, "edges_found", 0)

            summary = (
                f"Evolution complete in {elapsed}.\n"
                f"Promoted: {len(promoted)} agents | Culled: {len(culled)} agents\n"
                f"New edges discovered: {edges_found}"
            )
            logger.info(f"[scheduler] JOB END: {summary}")
            send_alert_sync(f"SharpEdge Evolution\n\n{summary}")
        except Exception as exc:
            elapsed = _fmt_elapsed(start)
            logger.exception(f"[scheduler] JOB FAILED: evolution after {elapsed}: {exc}")
            send_alert_sync(f"WARNING: Agent evolution crashed after {elapsed}.\n{exc}")

    def _job_intelligence(self) -> None:
        """Collect intelligence data: referee stats, manager records, wages, fatigue."""
        start = time.perf_counter()
        logger.info("[scheduler] JOB START: intelligence collection")
        try:
            today_str = _utcnow().date().isoformat()
            rows_total = 0

            intelligence_collectors = [
                ("referee_stats", "sharpedge.collectors.referee_stats", "RefereeStatsCollector"),
                ("manager_records", "sharpedge.collectors.manager_records", "ManagerRecordsCollector"),
                ("capology_wages", "sharpedge.collectors.capology", "CapologyCollector"),
                ("european_fatigue", "sharpedge.collectors.european_fatigue", "EuropeanFatigueCollector"),
                ("team_news", "sharpedge.collectors.team_news", "TeamNewsCollector"),
                ("fbref_advanced", "sharpedge.collectors.fbref", "FBrefCollector"),
                ("advanced_metrics", "sharpedge.collectors.advanced_metrics", "AdvancedMetricsCollector"),
            ]

            for record_type, module_path, class_name in intelligence_collectors:
                try:
                    import importlib
                    module = importlib.import_module(module_path)
                    cls = getattr(module, class_name)
                    collector = cls()
                    df = collector.collect(date=today_str)
                    if df is not None and not df.empty:
                        rows = ingest_dataframe(df, record_type)
                        rows_total += rows
                        logger.info(
                            f"[scheduler] Intelligence: {record_type} → {rows} rows"
                        )
                except Exception as col_exc:
                    logger.warning(
                        f"[scheduler] Intelligence collector '{record_type}' failed: {col_exc}"
                    )

            elapsed = _fmt_elapsed(start)
            logger.info(
                f"[scheduler] JOB END: intelligence collection ({elapsed}) — "
                f"{rows_total} rows ingested"
            )
        except Exception as exc:
            elapsed = _fmt_elapsed(start)
            logger.exception(
                f"[scheduler] JOB FAILED: intelligence collection after {elapsed}: {exc}"
            )
            send_alert_sync(
                f"WARNING: Intelligence collection crashed after {elapsed}.\n{exc}"
            )

    # -----------------------------------------------------------------------
    # Scheduler setup
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Register all jobs and start the blocking scheduler."""
        logger.info("[scheduler] Registering jobs...")

        # 1. Full data collection — 02:00, 08:00, 14:00, 20:00 UTC
        self._scheduler.add_job(
            self._job_collect_full,
            CronTrigger(hour="2,8,14,20", minute=0, timezone="UTC"),
            id="collect_full",
            name="Full Data Collection",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=600,
        )

        # 2. Odds collection — every 30 minutes (all day; job skips if no matches)
        self._scheduler.add_job(
            self._job_collect_odds,
            IntervalTrigger(minutes=30, timezone="UTC"),
            id="collect_odds",
            name="Odds Collection (30-min)",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
        )

        # 3. Predictions — daily 07:00 UTC
        self._scheduler.add_job(
            self._job_predict,
            CronTrigger(hour=7, minute=0, timezone="UTC"),
            id="predict",
            name="Daily Predictions",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=1800,
        )

        # 4. Lineup check — every 15 minutes between 10:00-21:00 UTC
        self._scheduler.add_job(
            self._job_check_lineups,
            CronTrigger(hour="10-21", minute="*/15", timezone="UTC"),
            id="check_lineups",
            name="Lineup Check (15-min)",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
        )

        # 5. Result resolution — daily 01:00 UTC
        self._scheduler.add_job(
            self._job_resolve,
            CronTrigger(hour=1, minute=0, timezone="UTC"),
            id="resolve",
            name="Result Resolution",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=1800,
        )

        # 6. Drift check — daily 03:00 UTC
        self._scheduler.add_job(
            self._job_drift_check,
            CronTrigger(hour=3, minute=0, timezone="UTC"),
            id="drift_check",
            name="Drift Check",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=900,
        )

        # 7. Evolution — every 14 days at 04:00 UTC
        self._scheduler.add_job(
            self._job_evolution,
            CronTrigger(day="1,15", hour=4, minute=0, timezone="UTC"),
            id="evolution",
            name="Agent Evolution (bi-monthly)",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=3600,
        )

        # 8. Intelligence collection — daily 06:00 UTC
        self._scheduler.add_job(
            self._job_intelligence,
            CronTrigger(hour=6, minute=0, timezone="UTC"),
            id="intelligence",
            name="Intelligence Collection",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=1800,
        )

        logger.info("[scheduler] All jobs registered. Starting scheduler...")
        send_alert_sync("SharpEdge Scheduler started. All 8 jobs active.")

        try:
            self._scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("[scheduler] Shutdown signal received — stopping.")
            send_alert_sync("SharpEdge Scheduler shutting down.")
            self._scheduler.shutdown(wait=True)

    def get_schedule(self) -> list[dict]:
        """Return current schedule for display/API."""
        jobs = []
        for job in self._scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append(
                {
                    "id": job.id,
                    "name": job.name,
                    "next_run_utc": next_run.isoformat() if next_run else None,
                    "trigger": str(job.trigger),
                }
            )
        return jobs
