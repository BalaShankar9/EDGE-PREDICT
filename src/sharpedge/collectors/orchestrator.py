"""
ScraperOrchestrator — master coordinator for all 31 SharpEdge data collectors.

Runs collectors in priority-tiered groups, isolates failures, and returns
structured CollectionReports.  Every public method is safe to call from the
daily pipeline: no exception ever propagates out of this class.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import pandas as pd

from sharpedge.alerts.telegram import send_alert_sync
from sharpedge.collectors.advanced_metrics import AdvancedMetricsCollector
from sharpedge.collectors.base import BaseCollector
from sharpedge.collectors.betexplorer import BetExplorerCollector
from sharpedge.collectors.betfair_exchange import BetfairExchangeCollector
from sharpedge.collectors.capology import CapologyCollector
from sharpedge.collectors.club_elo import ClubELOCollector
from sharpedge.collectors.crowd_sentiment import CrowdSentimentCollector
from sharpedge.collectors.european_fatigue import EuropeanFatigueCollector
from sharpedge.collectors.fbref import FBrefCollector
from sharpedge.collectors.flashscore import FlashScoreCollector
from sharpedge.collectors.football_data_org import FootballDataOrgCollector
from sharpedge.collectors.football_data_uk import FootballDataUKCollector
from sharpedge.collectors.footystats import FootyStatsCollector
from sharpedge.collectors.forebet import ForebetCollector
from sharpedge.collectors.fotmob import FotMobCollector
from sharpedge.collectors.lineup_scraper import LineupScraper
from sharpedge.collectors.manager_records import ManagerRecordsCollector
from sharpedge.collectors.oddsportal import OddsPortalCollector
from sharpedge.collectors.open_meteo import OpenMeteoCollector
from sharpedge.collectors.prediction_aggregator import PredictionAggregator
from sharpedge.collectors.predictz import PredictZCollector
from sharpedge.collectors.referee_stats import RefereeStatsCollector
from sharpedge.collectors.soccerway import SoccerwayCollector
from sharpedge.collectors.sofascore import SofascoreCollector
from sharpedge.collectors.team_news import TeamNewsCollector
from sharpedge.collectors.transfermarkt import TransfermarktCollector
from sharpedge.collectors.travel_calculator import TravelCalculator
from sharpedge.collectors.understat import UnderstatCollector
from sharpedge.collectors.whoscored import WhoScoredCollector
from sharpedge.collectors.windrawwin import WinDrawWinCollector
from sharpedge.collectors.world_football import WorldFootballCollector
from sharpedge.config import settings
from sharpedge.db.ingest import ingest_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority constants
# ---------------------------------------------------------------------------

PRIORITY_CRITICAL = "CRITICAL"
PRIORITY_HIGH = "HIGH"
PRIORITY_MEDIUM = "MEDIUM"
PRIORITY_LOW = "LOW"

_PRIORITY_ORDER = [PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW]


# ---------------------------------------------------------------------------
# CollectionReport dataclass
# ---------------------------------------------------------------------------

@dataclass
class CollectionReport:
    """Summary produced by ScraperOrchestrator.run_all()."""

    date: str
    sources_attempted: int = 0
    sources_succeeded: int = 0
    sources_failed: int = 0
    total_rows: int = 0
    by_source: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    errors: dict = field(default_factory=dict)

    # Holds the actual DataFrames keyed by source name; populated during run
    dataframes: dict = field(default_factory=dict)

    # Convenience ----------------------------------------------------------

    @property
    def success_rate(self) -> float:
        """Fraction of attempted sources that succeeded (0.0–1.0)."""
        if self.sources_attempted == 0:
            return 0.0
        return self.sources_succeeded / self.sources_attempted

    def __str__(self) -> str:
        return (
            f"CollectionReport({self.date}) "
            f"{self.sources_succeeded}/{self.sources_attempted} sources OK, "
            f"{self.total_rows} rows, "
            f"{self.elapsed_seconds:.1f}s"
        )


# ---------------------------------------------------------------------------
# ScraperOrchestrator
# ---------------------------------------------------------------------------

class ScraperOrchestrator:
    """Coordinates all 31 data collectors with priority-tiered execution.

    Priorities
    ----------
    CRITICAL  FootballDataUK, BetExplorer, Sofascore, FotMob
    HIGH      Forebet, PredictZ, WinDrawWin, OddsPortal, Transfermarkt,
              BetfairExchange, LineupScraper
    MEDIUM    FBref, Understat, ClubElo, FootyStats, FlashScore, Soccerway,
              AdvancedMetrics, WhoScored
    LOW       WorldFootball, PredictionAggregator, CrowdSentiment, OpenMeteo,
              Capology, RefereeStats, ManagerRecords, TeamNews, EuropeanFatigue

    TravelCalculator is exposed as a utility via ``self.travel_calculator``
    and is not run as a collector.
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # CRITICAL
        # ------------------------------------------------------------------
        self._football_data_uk = FootballDataUKCollector()
        self._betexplorer = BetExplorerCollector()
        self._sofascore = SofascoreCollector()
        self._fotmob = FotMobCollector()

        # ------------------------------------------------------------------
        # HIGH
        # ------------------------------------------------------------------
        self._forebet = ForebetCollector()
        self._predictz = PredictZCollector()
        self._windrawwin = WinDrawWinCollector()
        self._oddsportal = OddsPortalCollector()
        self._transfermarkt = TransfermarktCollector()
        self._betfair_exchange = BetfairExchangeCollector()
        self._lineup_scraper = LineupScraper()

        # ------------------------------------------------------------------
        # MEDIUM
        # ------------------------------------------------------------------
        self._fbref = FBrefCollector()
        self._understat = UnderstatCollector()
        self._club_elo = ClubELOCollector()
        self._footystats = FootyStatsCollector()
        self._flashscore = FlashScoreCollector()
        self._soccerway = SoccerwayCollector()
        self._advanced_metrics = AdvancedMetricsCollector()
        self._whoscored = WhoScoredCollector()

        # ------------------------------------------------------------------
        # LOW
        # ------------------------------------------------------------------
        self._world_football = WorldFootballCollector()
        self._prediction_aggregator = PredictionAggregator()
        self._crowd_sentiment = CrowdSentimentCollector()
        self._open_meteo = OpenMeteoCollector()
        self._capology = CapologyCollector()
        self._referee_stats = RefereeStatsCollector()
        self._manager_records = ManagerRecordsCollector()
        self._team_news = TeamNewsCollector()
        self._european_fatigue = EuropeanFatigueCollector()

        # ------------------------------------------------------------------
        # Utility (not a collector — not in registry)
        # ------------------------------------------------------------------
        self.travel_calculator = TravelCalculator()

        # Priority-ordered registry: {priority: [(name, collector)]}
        self._registry: dict[str, list[tuple[str, BaseCollector]]] = {
            PRIORITY_CRITICAL: [
                ("football_data_uk", self._football_data_uk),
                ("betexplorer", self._betexplorer),
                ("sofascore", self._sofascore),
                ("fotmob", self._fotmob),
            ],
            PRIORITY_HIGH: [
                ("forebet", self._forebet),
                ("predictz", self._predictz),
                ("windrawwin", self._windrawwin),
                ("oddsportal", self._oddsportal),
                ("transfermarkt", self._transfermarkt),
                ("betfair_exchange", self._betfair_exchange),
                ("lineup_scraper", self._lineup_scraper),
            ],
            PRIORITY_MEDIUM: [
                ("fbref", self._fbref),
                ("understat", self._understat),
                ("club_elo", self._club_elo),
                ("footystats", self._footystats),
                ("flashscore", self._flashscore),
                ("soccerway", self._soccerway),
                ("advanced_metrics", self._advanced_metrics),
                ("whoscored", self._whoscored),
            ],
            PRIORITY_LOW: [
                ("world_football", self._world_football),
                ("prediction_aggregator", self._prediction_aggregator),
                ("crowd_sentiment", self._crowd_sentiment),
                ("open_meteo", self._open_meteo),
                ("capology", self._capology),
                ("referee_stats", self._referee_stats),
                ("manager_records", self._manager_records),
                ("team_news", self._team_news),
                ("european_fatigue", self._european_fatigue),
            ],
        }

        # Fast lookup by source name
        self._collector_by_name: dict[str, BaseCollector] = {
            name: collector
            for group in self._registry.values()
            for name, collector in group
        }

        # Health tracking: {source_name: {last_success, last_rows, last_error}}
        self._health: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _resolve_date(self, date_val: Optional[str | date | datetime]) -> str:
        """Normalise date argument to YYYY-MM-DD string."""
        if date_val is None:
            return self._today()
        if isinstance(date_val, datetime):
            return date_val.strftime("%Y-%m-%d")
        if isinstance(date_val, date):
            return date_val.strftime("%Y-%m-%d")
        return str(date_val)

    def _run_collector(
        self,
        name: str,
        collector: BaseCollector,
        kwargs: dict,
    ) -> tuple[pd.DataFrame, Optional[str]]:
        """Run a single collector safely.

        Returns (DataFrame, error_str | None).
        Empty DataFrame + error_str means failure.
        """
        try:
            logger.info(f"[orchestrator] Starting {name}")
            df = collector.collect(**kwargs)
            error: Optional[str] = None
            rows = len(df) if df is not None else 0
            self._health[name] = {
                "last_success": self._today(),
                "last_rows": rows,
                "last_error": None,
            }
            logger.info(f"[orchestrator] {name} OK — {rows} rows")
            return df if df is not None else pd.DataFrame(), error
        except Exception as exc:
            error_str = f"{type(exc).__name__}: {exc}"
            logger.error(f"[orchestrator] {name} FAILED: {error_str}")
            self._health[name] = {
                "last_success": self._health.get(name, {}).get("last_success"),
                "last_rows": 0,
                "last_error": error_str,
            }
            return pd.DataFrame(), error_str

    def _run_group(
        self,
        priority: str,
        collectors: list[tuple[str, BaseCollector]],
        kwargs: dict,
        report: CollectionReport,
    ) -> dict[str, pd.DataFrame]:
        """Run a priority group and accumulate results into report.

        Returns {source_name: DataFrame} for this group.
        """
        results: dict[str, pd.DataFrame] = {}

        for name, collector in collectors:
            report.sources_attempted += 1
            df, error = self._run_collector(name, collector, kwargs)

            rows = len(df) if df is not None and not df.empty else 0
            report.by_source[name] = {
                "priority": priority,
                "rows": rows,
                "success": error is None,
                "error": error,
            }

            if error is None:
                report.sources_succeeded += 1
                report.total_rows += rows
                results[name] = df
                report.dataframes[name] = df
            else:
                report.sources_failed += 1
                report.errors[name] = error
                results[name] = pd.DataFrame()
                report.dataframes[name] = pd.DataFrame()

                if priority == PRIORITY_CRITICAL:
                    msg = (
                        f"CRITICAL source <b>{name}</b> failed.\n"
                        f"Error: {error}\n"
                        f"Date: {report.date}"
                    )
                    try:
                        send_alert_sync(msg)
                    except Exception as alert_err:
                        logger.warning(f"[orchestrator] Alert send failed: {alert_err}")

        return results

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_all(
        self, date: Optional[str | date | datetime] = None
    ) -> CollectionReport:
        """Run all 31 collectors grouped by priority.

        CRITICAL failures trigger a Telegram alert but execution continues.
        No single failure can kill the run.

        Parameters
        ----------
        date:
            Target date (YYYY-MM-DD string, date, or datetime).
            Defaults to today UTC.

        Returns
        -------
        CollectionReport
            Comprehensive summary of the entire run.  ``report.dataframes``
            holds every collector's DataFrame keyed by source name.
        """
        date_str = self._resolve_date(date)
        report = CollectionReport(date=date_str)
        kwargs = {"date": date_str}
        wall_start = time.perf_counter()

        logger.info(f"[orchestrator] run_all starting for {date_str}")

        for priority in _PRIORITY_ORDER:
            collectors = self._registry[priority]
            logger.info(
                f"[orchestrator] Running {priority} group "
                f"({len(collectors)} collectors)"
            )
            self._run_group(priority, collectors, kwargs, report)

        report.elapsed_seconds = time.perf_counter() - wall_start
        logger.info(f"[orchestrator] run_all complete: {report}")
        return report

    # -----------------------------------------------------------------------
    # Specialised collection methods
    # -----------------------------------------------------------------------

    def run_predictions_collection(
        self, date: Optional[str | date | datetime] = None
    ) -> pd.DataFrame:
        """Collect from prediction-source collectors and return a unified DataFrame.

        Sources: Forebet, PredictZ, WinDrawWin, PredictionAggregator.

        Adds consensus columns:
          - consensus_pick      ('H' | 'D' | 'A')
          - consensus_sources   (number of sources agreeing with pick)
          - consensus_pct       (fraction of sources agreeing with pick)

        Parameters
        ----------
        date:
            Target date. Defaults to today UTC.

        Returns
        -------
        pd.DataFrame
            Unified predictions DataFrame (empty if all sources fail).
        """
        date_str = self._resolve_date(date)
        kwargs = {"date": date_str}

        prediction_collectors: list[tuple[str, BaseCollector]] = [
            ("forebet", self._forebet),
            ("predictz", self._predictz),
            ("windrawwin", self._windrawwin),
            ("prediction_aggregator", self._prediction_aggregator),
        ]

        frames: list[pd.DataFrame] = []
        for name, collector in prediction_collectors:
            df, error = self._run_collector(name, collector, kwargs)
            if not df.empty:
                df["_source"] = name
                frames.append(df)

        if not frames:
            logger.warning("[orchestrator] run_predictions_collection: all sources failed")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = self._add_consensus_scores(combined)
        return combined

    def run_odds_collection(
        self, date: Optional[str | date | datetime] = None
    ) -> pd.DataFrame:
        """Collect odds from BetExplorer and OddsPortal.

        Returns a unified odds DataFrame with the best (highest) available
        price per outcome (home / draw / away) per match.

        Parameters
        ----------
        date:
            Target date. Defaults to today UTC.

        Returns
        -------
        pd.DataFrame
            Unified odds DataFrame with best_odds_home, best_odds_draw,
            best_odds_away columns added.
        """
        date_str = self._resolve_date(date)
        kwargs = {"date": date_str}

        odds_collectors: list[tuple[str, BaseCollector]] = [
            ("betexplorer", self._betexplorer),
            ("oddsportal", self._oddsportal),
        ]

        frames: list[pd.DataFrame] = []
        for name, collector in odds_collectors:
            df, error = self._run_collector(name, collector, kwargs)
            if not df.empty:
                df["_source"] = name
                frames.append(df)

        if not frames:
            logger.warning("[orchestrator] run_odds_collection: all sources failed")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined = self._compute_best_odds(combined)
        return combined

    def run_fixtures_collection(
        self, date: Optional[str | date | datetime] = None
    ) -> pd.DataFrame:
        """Collect fixtures from Sofascore, FlashScore, Soccerway, Transfermarkt.

        Cross-references and deduplicates fixtures across sources using
        normalised team names and kick-off date.

        Parameters
        ----------
        date:
            Target date. Defaults to today UTC.

        Returns
        -------
        pd.DataFrame
            Deduplicated fixture list.  A ``source_count`` column records how
            many sources confirmed each fixture.
        """
        date_str = self._resolve_date(date)
        kwargs = {"date": date_str}

        fixture_collectors: list[tuple[str, BaseCollector]] = [
            ("sofascore", self._sofascore),
            ("flashscore", self._flashscore),
            ("soccerway", self._soccerway),
            ("transfermarkt", self._transfermarkt),
        ]

        frames: list[pd.DataFrame] = []
        for name, collector in fixture_collectors:
            df, error = self._run_collector(name, collector, kwargs)
            if not df.empty:
                df["_source"] = name
                frames.append(df)

        if not frames:
            logger.warning("[orchestrator] run_fixtures_collection: all sources failed")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        deduped = self._deduplicate_fixtures(combined)
        return deduped

    def run_results_collection(
        self, date: Optional[str | date | datetime] = None
    ) -> pd.DataFrame:
        """Collect results from FootballDataUK, FlashScore, Soccerway, BetExplorer.

        A result is considered confirmed only when it appears in at least
        two independent sources.  The ``confirmed`` boolean column signals this.

        Parameters
        ----------
        date:
            Target date. Defaults to today UTC.

        Returns
        -------
        pd.DataFrame
            Cross-referenced results with a ``confirmed`` column and a
            ``source_count`` column (number of sources reporting each result).
        """
        date_str = self._resolve_date(date)
        kwargs = {"date": date_str}

        result_collectors: list[tuple[str, BaseCollector]] = [
            ("football_data_uk", self._football_data_uk),
            ("flashscore", self._flashscore),
            ("soccerway", self._soccerway),
            ("betexplorer", self._betexplorer),
        ]

        frames: list[pd.DataFrame] = []
        for name, collector in result_collectors:
            df, error = self._run_collector(name, collector, kwargs)
            if not df.empty:
                df["_source"] = name
                frames.append(df)

        if not frames:
            logger.warning("[orchestrator] run_results_collection: all sources failed")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        verified = self._cross_reference_results(combined)
        return verified

    def run_lineup_collection(
        self,
        match_ids: Optional[list] = None,
    ) -> pd.DataFrame:
        """Collect lineups from LineupScraper and FotMob.

        Both sources are queried; results are concatenated and deduplicated
        on (match_id, team_id, player_id) where those columns exist.

        Parameters
        ----------
        match_ids:
            Optional list of match IDs to scope the query.  When ``None``
            both collectors use their default behaviour (today's matches).

        Returns
        -------
        pd.DataFrame
            Unified lineup DataFrame (empty if both sources fail).
        """
        kwargs: dict = {}
        if match_ids is not None:
            kwargs["match_ids"] = match_ids

        frames: list[pd.DataFrame] = []
        for name, collector in [
            ("lineup_scraper", self._lineup_scraper),
            ("fotmob", self._fotmob),
        ]:
            df, error = self._run_collector(name, collector, kwargs)
            if not df.empty:
                df["_source"] = name
                frames.append(df)

        if not frames:
            logger.warning("[orchestrator] run_lineup_collection: all sources failed")
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)

        # Deduplicate on natural key if columns exist
        dedup_cols = [c for c in ("match_id", "team_id", "player_id") if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols, keep="first")

        return combined.drop(columns=["_source"], errors="ignore")

    def run_intelligence_collection(self) -> dict[str, pd.DataFrame]:
        """Collect contextual intelligence from enrichment sources.

        Sources queried (all LOW-priority):
          - referee_stats
          - manager_records
          - capology (wages)
          - team_news
          - european_fatigue
          - advanced_metrics
          - whoscored
          - crowd_sentiment

        No date argument: these sources are typically not date-scoped.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of source_name → DataFrame.  Failed sources map to an
            empty DataFrame so callers can always iterate safely.
        """
        intelligence_collectors: list[tuple[str, BaseCollector]] = [
            ("referee_stats", self._referee_stats),
            ("manager_records", self._manager_records),
            ("capology", self._capology),
            ("team_news", self._team_news),
            ("european_fatigue", self._european_fatigue),
            ("advanced_metrics", self._advanced_metrics),
            ("whoscored", self._whoscored),
            ("crowd_sentiment", self._crowd_sentiment),
        ]

        results: dict[str, pd.DataFrame] = {}
        for name, collector in intelligence_collectors:
            df, error = self._run_collector(name, collector, {})
            results[name] = df if df is not None else pd.DataFrame()

        succeeded = sum(1 for df in results.values() if not df.empty)
        logger.info(
            f"[orchestrator] run_intelligence_collection complete: "
            f"{succeeded}/{len(intelligence_collectors)} sources OK"
        )
        return results

    def run_exchange_collection(
        self, date: Optional[str | date | datetime] = None
    ) -> pd.DataFrame:
        """Collect Betfair exchange odds.

        Parameters
        ----------
        date:
            Target date. Defaults to today UTC.

        Returns
        -------
        pd.DataFrame
            Exchange odds DataFrame (empty on failure).
        """
        date_str = self._resolve_date(date)
        kwargs = {"date": date_str}

        df, error = self._run_collector("betfair_exchange", self._betfair_exchange, kwargs)
        if error:
            logger.warning(f"[orchestrator] run_exchange_collection failed: {error}")
            return pd.DataFrame()
        return df

    def ingest_all(self, collection_report: CollectionReport) -> dict:
        """Persist every DataFrame in *collection_report* to the database.

        Iterates ``collection_report.dataframes`` and calls
        ``ingest_dataframe(df, source_name)`` for each non-empty frame.

        Parameters
        ----------
        collection_report:
            The CollectionReport returned by ``run_all()``.  Its
            ``dataframes`` dict must be populated (it is when produced by
            this class).

        Returns
        -------
        dict
            Summary mapping source_name → {"rows_inserted": int, "error": str|None}.
        """
        summary: dict[str, dict] = {}

        for source_name, df in collection_report.dataframes.items():
            if df is None or df.empty:
                summary[source_name] = {"rows_inserted": 0, "error": None}
                continue

            try:
                rows_inserted = ingest_dataframe(df, source_name)
                summary[source_name] = {"rows_inserted": rows_inserted, "error": None}
                logger.info(
                    f"[orchestrator] ingest_all: {source_name} → {rows_inserted} rows"
                )
            except Exception as exc:
                error_str = f"{type(exc).__name__}: {exc}"
                logger.error(
                    f"[orchestrator] ingest_all: {source_name} FAILED: {error_str}"
                )
                summary[source_name] = {"rows_inserted": 0, "error": error_str}

        total_inserted = sum(v["rows_inserted"] for v in summary.values())
        total_failed = sum(1 for v in summary.values() if v["error"])
        logger.info(
            f"[orchestrator] ingest_all complete: "
            f"{total_inserted} total rows inserted, "
            f"{total_failed} sources failed"
        )
        return summary

    def get_health(self) -> dict:
        """Return health status for every collector.

        Keys are source names.  Each value is a dict with:
          - last_success: YYYY-MM-DD of last successful run, or None
          - last_rows: row count from last successful run
          - last_error: error string from last failure, or None

        Collectors not yet run appear with last_success=None.
        """
        all_names = [
            name
            for group in self._registry.values()
            for name, _ in group
        ]
        return {
            name: self._health.get(
                name,
                {"last_success": None, "last_rows": 0, "last_error": None},
            )
            for name in all_names
        }

    # -----------------------------------------------------------------------
    # Post-processing helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _add_consensus_scores(df: pd.DataFrame) -> pd.DataFrame:
        """Add consensus pick columns to a unified predictions DataFrame.

        Requires columns: home_team_id (or home_team), away_team_id (or
        away_team), and at least one of: prob_home, prob_draw, prob_away.

        Adds:
          consensus_pick      — most frequent raw pick across sources ('H'/'D'/'A')
          consensus_sources   — count of sources picking that outcome
          consensus_pct       — fraction of sources picking that outcome
          avg_prob_home/draw/away — mean probabilities across sources
        """
        if df.empty:
            return df

        # Determine pick per row from probabilities if no explicit pick column
        if "pick" not in df.columns:
            prob_cols = {"H": "prob_home", "D": "prob_draw", "A": "prob_away"}
            available = {k: v for k, v in prob_cols.items() if v in df.columns}
            if available:
                prob_df = pd.DataFrame(
                    {k: df[v] for k, v in available.items()}
                )
                df = df.copy()
                df["pick"] = prob_df.idxmax(axis=1)
            else:
                df = df.copy()
                df["pick"] = None

        # Match key: normalised team names + date
        key_cols = []
        for col in ("home_team_id", "home_team"):
            if col in df.columns:
                key_cols.append(col)
                break
        for col in ("away_team_id", "away_team"):
            if col in df.columns:
                key_cols.append(col)
                break
        if "match_date" in df.columns or "date" in df.columns:
            date_col = "match_date" if "match_date" in df.columns else "date"
            key_cols.append(date_col)

        if len(key_cols) < 2:
            logger.warning(
                "[orchestrator] Cannot add consensus: team columns not found"
            )
            return df

        df = df.copy()
        df["_match_key"] = df[key_cols].astype(str).agg("__".join, axis=1)

        # Per-match consensus aggregation
        consensus_rows: list[dict] = []
        for match_key, grp in df.groupby("_match_key"):
            picks = grp["pick"].dropna()
            total = len(picks)

            if total > 0:
                best_pick = picks.value_counts().idxmax()
                best_count = int(picks.value_counts().iloc[0])
                best_pct = round(best_count / total, 3)
            else:
                best_pick, best_count, best_pct = None, 0, 0.0

            row: dict = {
                "_match_key": match_key,
                "consensus_pick": best_pick,
                "consensus_sources": best_count,
                "consensus_pct": best_pct,
            }
            for col, label in [
                ("prob_home", "avg_prob_home"),
                ("prob_draw", "avg_prob_draw"),
                ("prob_away", "avg_prob_away"),
            ]:
                if col in grp.columns:
                    row[label] = round(float(grp[col].dropna().mean()), 3)
            consensus_rows.append(row)

        if not consensus_rows:
            return df.drop(columns=["_match_key"], errors="ignore")

        consensus_df = pd.DataFrame(consensus_rows)
        df = df.merge(consensus_df, on="_match_key", how="left")
        df = df.drop(columns=["_match_key"], errors="ignore")
        return df

    @staticmethod
    def _compute_best_odds(df: pd.DataFrame) -> pd.DataFrame:
        """Compute best (highest) available decimal odds per outcome per match.

        Requires columns: home_team (or home_team_id), away_team (or
        away_team_id), and at least one of odds_home / odds_draw / odds_away.

        Adds: best_odds_home, best_odds_draw, best_odds_away.
        """
        if df.empty:
            return df

        df = df.copy()

        # Match key
        key_cols = []
        for col in ("home_team_id", "home_team"):
            if col in df.columns:
                key_cols.append(col)
                break
        for col in ("away_team_id", "away_team"):
            if col in df.columns:
                key_cols.append(col)
                break
        if not key_cols:
            return df

        df["_match_key"] = df[key_cols].astype(str).agg("__".join, axis=1)

        best_rows: list[dict] = []
        for match_key, grp in df.groupby("_match_key"):
            row: dict = {"_match_key": match_key}
            for col, best_col in [
                ("odds_home", "best_odds_home"),
                ("odds_draw", "best_odds_draw"),
                ("odds_away", "best_odds_away"),
            ]:
                if col in grp.columns:
                    valid = grp[col].dropna()
                    row[best_col] = float(valid.max()) if not valid.empty else None
            best_rows.append(row)

        if not best_rows:
            return df.drop(columns=["_match_key"], errors="ignore")

        best_df = pd.DataFrame(best_rows)
        df = df.merge(best_df, on="_match_key", how="left")
        df = df.drop(columns=["_match_key"], errors="ignore")
        return df

    @staticmethod
    def _deduplicate_fixtures(df: pd.DataFrame) -> pd.DataFrame:
        """Deduplicate fixtures by normalised match key.

        Collapses rows for the same fixture into one, preserving all
        non-null fields from any source.  Adds ``source_count`` column.
        """
        if df.empty:
            return df

        df = df.copy()

        key_cols = []
        for col in ("home_team_id", "home_team"):
            if col in df.columns:
                key_cols.append(col)
                break
        for col in ("away_team_id", "away_team"):
            if col in df.columns:
                key_cols.append(col)
                break
        if "match_date" in df.columns:
            key_cols.append("match_date")
        elif "date" in df.columns:
            key_cols.append("date")

        if len(key_cols) < 2:
            return df

        df["_match_key"] = df[key_cols].astype(str).agg("__".join, axis=1)

        deduped_rows: list[dict] = []
        for match_key, grp in df.groupby("_match_key"):
            merged: dict = {"_match_key": match_key}
            merged["source_count"] = grp["_source"].nunique() if "_source" in grp else 1
            merged["sources"] = (
                ",".join(sorted(grp["_source"].unique()))
                if "_source" in grp
                else ""
            )
            # For each non-key column take the first non-null value
            for col in grp.columns:
                if col in ("_match_key", "_source"):
                    continue
                non_null = grp[col].dropna()
                merged[col] = non_null.iloc[0] if not non_null.empty else None
            deduped_rows.append(merged)

        if not deduped_rows:
            return df.drop(columns=["_match_key", "_source"], errors="ignore")

        result = pd.DataFrame(deduped_rows)
        result = result.drop(columns=["_match_key"], errors="ignore")
        return result

    @staticmethod
    def _cross_reference_results(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-reference results across sources.

        A result is 'confirmed' when at least two independent sources report
        the same home_goals / away_goals for the same match.

        Adds:
          source_count  — how many sources reported this match
          confirmed     — True if 2+ sources agree on the score
        """
        if df.empty:
            return df

        df = df.copy()

        key_cols = []
        for col in ("home_team_id", "home_team"):
            if col in df.columns:
                key_cols.append(col)
                break
        for col in ("away_team_id", "away_team"):
            if col in df.columns:
                key_cols.append(col)
                break
        if "match_date" in df.columns:
            key_cols.append("match_date")
        elif "date" in df.columns:
            key_cols.append("date")

        if len(key_cols) < 2:
            return df

        df["_match_key"] = df[key_cols].astype(str).agg("__".join, axis=1)

        verified_rows: list[dict] = []
        for match_key, grp in df.groupby("_match_key"):
            source_count = grp["_source"].nunique() if "_source" in grp else 1

            # Build canonical score from majority vote if available
            confirmed = False
            canonical_home: Optional[int] = None
            canonical_away: Optional[int] = None

            if "home_goals" in grp.columns and "away_goals" in grp.columns:
                score_pairs = (
                    grp[["home_goals", "away_goals"]]
                    .dropna()
                    .apply(lambda r: (int(r["home_goals"]), int(r["away_goals"])), axis=1)
                )
                if not score_pairs.empty:
                    score_counts = score_pairs.value_counts()
                    top_score = score_counts.index[0]
                    top_count = int(score_counts.iloc[0])
                    if top_count >= 2:
                        confirmed = True
                    canonical_home, canonical_away = top_score

            row: dict = {}
            for col in grp.columns:
                if col in ("_source",):
                    continue
                non_null = grp[col].dropna()
                row[col] = non_null.iloc[0] if not non_null.empty else None

            row["source_count"] = source_count
            row["confirmed"] = confirmed
            if canonical_home is not None:
                row["home_goals"] = canonical_home
            if canonical_away is not None:
                row["away_goals"] = canonical_away
            row["sources"] = (
                ",".join(sorted(grp["_source"].unique()))
                if "_source" in grp
                else ""
            )
            verified_rows.append(row)

        if not verified_rows:
            return df.drop(columns=["_match_key", "_source"], errors="ignore")

        result = pd.DataFrame(verified_rows)
        result = result.drop(columns=["_match_key"], errors="ignore")
        return result
