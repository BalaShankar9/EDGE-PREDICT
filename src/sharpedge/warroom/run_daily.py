"""Production daily pipeline runner for Railway worker.

Runs the full SharpEdge prediction pipeline:
1. Health checks
2. Data collection for all sports
3. Feature building + agent predictions
4. Arbiter combination + banker filter
5. Publish picks to DB + Telegram broadcast
6. Check retraining schedule
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sharpedge.warroom.orchestrator import WarRoomOrchestrator
from sharpedge.warroom.health_monitor import HealthMonitor, HealthStatus
from sharpedge.warroom.retrainer import AutoRetrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def send_telegram(message: str) -> None:
    """Send Telegram alert (best-effort)."""
    try:
        from sharpedge.alerts.telegram import send_alert_sync
        send_alert_sync(message)
    except Exception as e:
        logger.warning(f"Telegram alert failed: {e}")


def main() -> None:
    """Run the full daily prediction pipeline."""
    start = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("  SharpEdge Daily Pipeline — Starting")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {start.isoformat()}")

    # 1. Health checks
    monitor = HealthMonitor()
    health = monitor.run_checks()
    logger.info(f"Health: {health.overall.value}")

    if health.overall == HealthStatus.DOWN:
        msg = "SharpEdge health check FAILED. Pipeline aborted."
        logger.error(msg)
        send_telegram(f"🚨 {msg}")
        return

    # 2. Run orchestrator
    orchestrator = WarRoomOrchestrator()

    # Register football pipeline (primary sport)
    # Additional sports registered when their pipelines are production-ready
    try:
        import sharpedge.sports.football
        orchestrator.register_sport_pipeline("football", {
            "agents": [],  # agents are registered separately
        })
    except Exception as e:
        logger.warning(f"Failed to register football: {e}")

    report = orchestrator.run_daily()

    # 3. Check retraining needs
    retrainer = AutoRetrainer()
    for sport in report.sports_processed:
        retrainer.register_sport(sport)
    due_sports = retrainer.get_due_sports()
    if due_sports:
        logger.info(f"Retraining due for: {due_sports}")

    # 4. Summary
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    summary = (
        f"📊 SharpEdge Daily Pipeline Complete\n"
        f"Time: {elapsed:.0f}s\n"
        f"Sports: {', '.join(report.sports_processed) or 'none'}\n"
        f"Matches: {report.total_matches}\n"
        f"Predictions: {report.total_predictions}\n"
        f"Picks: {report.total_picks}\n"
    )
    if due_sports:
        summary += f"⚠️ Retrain due: {', '.join(due_sports)}\n"

    logger.info(summary)
    send_telegram(summary)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
