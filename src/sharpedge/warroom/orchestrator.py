"""Multi-Sport Orchestrator — coordinates all departments.

Discovers registered sports, runs their collectors, builds features,
generates agent predictions, combines via arbiter, and produces picks.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sharpedge.core.sport import sport_registry

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a single pipeline run."""
    sport: str
    timestamp: datetime
    matches_processed: int
    predictions_generated: int
    picks_produced: int
    agents_used: int
    errors: list[str] = field(default_factory=list)


@dataclass
class DailyReport:
    """Summary of all daily operations."""
    date: str
    sports_processed: list[str]
    total_matches: int
    total_predictions: int
    total_picks: int
    results: list[PipelineResult]
    circuit_breaker_level: int
    bankroll: float


class WarRoomOrchestrator:
    """Top-level orchestrator for all SharpEdge operations."""

    def __init__(self):
        self._sport_pipelines: dict[str, dict] = {}  # sport -> {collector, features, trainer, agents}
        self._daily_results: list[DailyReport] = []

    def register_sport_pipeline(self, sport: str, pipeline_config: dict) -> None:
        """Register a sport's complete pipeline configuration.

        pipeline_config should contain:
        - collectors: list of collector instances
        - feature_pipeline: FeaturePipeline instance
        - trainer: trainer instance
        - agents: list of BaseAgent instances
        """
        self._sport_pipelines[sport] = pipeline_config
        logger.info(f"Registered pipeline for {sport}")

    def get_active_sports(self) -> list[str]:
        """Return list of sports with registered pipelines."""
        return list(self._sport_pipelines.keys())

    def run_sport_pipeline(self, sport: str) -> PipelineResult:
        """Run the full prediction pipeline for a single sport.

        Steps:
        1. Collect data from all collectors
        2. Build features
        3. Run all agents
        4. Combine via arbiter
        5. Apply execution desk (staking, portfolio, timing)
        """
        timestamp = datetime.now(timezone.utc)
        errors = []

        if sport not in self._sport_pipelines:
            return PipelineResult(
                sport=sport, timestamp=timestamp,
                matches_processed=0, predictions_generated=0,
                picks_produced=0, agents_used=0,
                errors=[f"No pipeline registered for {sport}"],
            )

        config = self._sport_pipelines[sport]
        n_agents = len(config.get("agents", []))

        # In a real implementation, each step would:
        # 1. Call collectors to get match data
        # 2. Build feature matrix
        # 3. Run each agent's predict_batch()
        # 4. Run arbiter.predict() for each match
        # 5. Apply staking + portfolio filters

        # For now, return a placeholder result
        # (full implementation requires live data flow)
        logger.info(f"Pipeline run for {sport}: {n_agents} agents registered")

        return PipelineResult(
            sport=sport, timestamp=timestamp,
            matches_processed=0,
            predictions_generated=0,
            picks_produced=0,
            agents_used=n_agents,
            errors=errors,
        )

    def run_daily(self) -> DailyReport:
        """Run all sport pipelines for the day."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        results = []
        total_matches = 0
        total_preds = 0
        total_picks = 0

        for sport in self.get_active_sports():
            try:
                result = self.run_sport_pipeline(sport)
                results.append(result)
                total_matches += result.matches_processed
                total_preds += result.predictions_generated
                total_picks += result.picks_produced
            except Exception as e:
                logger.error(f"Pipeline failed for {sport}: {e}")
                results.append(PipelineResult(
                    sport=sport, timestamp=datetime.now(timezone.utc),
                    matches_processed=0, predictions_generated=0,
                    picks_produced=0, agents_used=0, errors=[str(e)],
                ))

        report = DailyReport(
            date=today,
            sports_processed=[r.sport for r in results],
            total_matches=total_matches,
            total_predictions=total_preds,
            total_picks=total_picks,
            results=results,
            circuit_breaker_level=0,
            bankroll=0.0,
        )

        self._daily_results.append(report)
        logger.info(
            f"Daily run complete: {len(results)} sports, "
            f"{total_matches} matches, {total_picks} picks"
        )
        return report
