"""Drift detection and auto-retraining system for SharpEdge.

Monitors model calibration quality by computing Expected Calibration Error (ECE)
and Brier Score over a rolling window of resolved picks. When metrics degrade past
configurable thresholds a DriftResult is produced and, if severe enough, a retrain
request is logged and persisted to DriftSnapshot.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from sharpedge.config import settings
from sharpedge.db.engine import get_session
from sharpedge.db.models import AgentPerformance, DailyPick, DriftSnapshot
# Lazy import to avoid circular dependency with warroom
# from sharpedge.warroom.retrainer import AutoRetrainer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

_SEVERITY_RANK: dict[str, int] = {
    "none": 0,
    "warning": 1,
    "critical": 2,
    "emergency": 3,
}

_RECOMMENDATION_FOR_SEVERITY: dict[str, str] = {
    "none": "none",
    "warning": "recalibrate",
    "critical": "retrain",
    "emergency": "emergency_retrain",
}


@dataclass
class DriftResult:
    """Full calibration-drift assessment for a single sport / window."""

    sport_slug: str
    has_drift: bool
    severity: str  # "none" | "warning" | "critical" | "emergency"
    calibration_error: float  # ECE
    brier_score: float
    accuracy: float
    roi_pct: float
    n_predictions: int
    recommendation: str  # "none" | "recalibrate" | "retrain" | "emergency_retrain"
    window_days: int = settings.drift_window_days
    assessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict = field(default_factory=dict)

    # Threshold snapshots (for auditability)
    calibration_threshold: float = settings.drift_calibration_threshold
    brier_threshold: float = settings.drift_brier_threshold
    accuracy_floor: float = settings.drift_accuracy_floor

    def __str__(self) -> str:
        return (
            f"DriftResult(sport={self.sport_slug!r}, severity={self.severity!r}, "
            f"ECE={self.calibration_error:.4f}, Brier={self.brier_score:.4f}, "
            f"accuracy={self.accuracy:.4f}, roi={self.roi_pct:.2f}%, "
            f"n={self.n_predictions}, rec={self.recommendation!r})"
        )


@dataclass
class RetrainRequest:
    """A logged-but-not-yet-executed retrain request."""

    sport: str
    reason: str
    severity: str
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    drift_result: Optional[DriftResult] = None


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Computes calibration metrics on resolved DailyPick rows and detects drift.

    All heavy DB work is done inside context managers so sessions are always
    released cleanly.
    """

    N_BINS: int = 10  # ECE bins

    def __init__(self) -> None:
        self._min_predictions = settings.drift_check_min_predictions

    # ------------------------------------------------------------------
    # Core metric computation (pure, no I/O)
    # ------------------------------------------------------------------

    def compute_calibration_error(self, predictions: list[dict]) -> float:
        """Expected Calibration Error (ECE) with 10 equally-spaced bins.

        Each prediction dict must contain:
            ``model_prob`` (float, predicted win probability)
            ``outcome``    (int/float, 1 = correct, 0 = incorrect)

        Returns ECE in [0, 1].  Lower is better.
        """
        if not predictions:
            return 0.0

        bin_edges = [i / self.N_BINS for i in range(self.N_BINS + 1)]
        bin_confidences: list[list[float]] = [[] for _ in range(self.N_BINS)]
        bin_outcomes: list[list[float]] = [[] for _ in range(self.N_BINS)]

        for pred in predictions:
            prob: float = float(pred["model_prob"])
            outcome: float = float(pred["outcome"])
            # Clamp to [0, 1] defensively
            prob = max(0.0, min(1.0, prob))
            # Determine bin index; last edge is inclusive
            bin_idx = min(int(prob * self.N_BINS), self.N_BINS - 1)
            bin_confidences[bin_idx].append(prob)
            bin_outcomes[bin_idx].append(outcome)

        n_total = len(predictions)
        ece = 0.0
        for b in range(self.N_BINS):
            n_b = len(bin_confidences[b])
            if n_b == 0:
                continue
            avg_conf = sum(bin_confidences[b]) / n_b
            avg_acc = sum(bin_outcomes[b]) / n_b
            ece += (n_b / n_total) * abs(avg_acc - avg_conf)

        return ece

    def compute_brier_score(self, predictions: list[dict]) -> float:
        """Standard Brier Score: mean((predicted_prob - outcome)^2).

        Each prediction dict must contain:
            ``model_prob`` (float)
            ``outcome``    (int/float, 1 = correct, 0 = incorrect)

        Returns Brier Score in [0, 1].  Lower is better.
        """
        if not predictions:
            return 0.0

        total = sum(
            (float(p["model_prob"]) - float(p["outcome"])) ** 2
            for p in predictions
        )
        return total / len(predictions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_resolved_picks(
        self, sport_slug: str, window_days: int
    ) -> list[DailyPick]:
        """Fetch resolved DailyPick rows for *sport_slug* in the last *window_days*."""
        cutoff_date = date.today() - timedelta(days=window_days)
        with get_session() as session:
            rows = (
                session.query(DailyPick)
                .filter(
                    DailyPick.match_date >= cutoff_date,
                    DailyPick.result.isnot(None),
                    DailyPick.resolved_at.isnot(None),
                )
                .all()
            )
            # Detach from session so callers can use them freely
            session.expunge_all()
            return rows

    @staticmethod
    def _picks_to_prediction_dicts(picks: list[DailyPick]) -> list[dict]:
        """Convert ORM rows to the lightweight dicts used by metric methods."""
        result = []
        for pick in picks:
            outcome = 1.0 if pick.result == "win" else 0.0
            result.append(
                {
                    "model_prob": pick.model_prob,
                    "outcome": outcome,
                    "profit_loss": pick.profit_loss or 0.0,
                    "stake_flat": pick.stake_flat or 1.0,
                    "pick_id": pick.id,
                    "match_date": pick.match_date,
                }
            )
        return result

    @staticmethod
    def _compute_roi(prediction_dicts: list[dict]) -> float:
        """ROI % = total_profit_loss / total_staked * 100."""
        if not prediction_dicts:
            return 0.0
        total_pl = sum(p["profit_loss"] for p in prediction_dicts)
        total_staked = sum(p["stake_flat"] for p in prediction_dicts)
        if total_staked == 0:
            return 0.0
        return (total_pl / total_staked) * 100.0

    @staticmethod
    def _classify_severity(
        calibration_error: float,
        brier_score: float,
        accuracy: float,
    ) -> tuple[str, str]:
        """Return (severity, recommendation).

        Priority order (highest wins):
          emergency → accuracy < floor
          critical  → ECE > threshold AND Brier > threshold
          warning   → ECE > threshold OR  Brier > threshold
          none      → all healthy
        """
        ece_breach = calibration_error > settings.drift_calibration_threshold
        brier_breach = brier_score > settings.drift_brier_threshold
        acc_emergency = accuracy < settings.drift_accuracy_floor

        if acc_emergency:
            severity = "emergency"
        elif ece_breach and brier_breach:
            severity = "critical"
        elif ece_breach or brier_breach:
            severity = "warning"
        else:
            severity = "none"

        return severity, _RECOMMENDATION_FOR_SEVERITY[severity]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_drift(
        self, sport_slug: str, window_days: Optional[int] = None
    ) -> DriftResult:
        """Compute drift metrics for *sport_slug* over *window_days*.

        Returns a :class:`DriftResult` regardless of whether enough data
        is available; if predictions < min threshold the result carries
        ``severity="none"`` and a note in ``details``.
        """
        window = window_days if window_days is not None else settings.drift_window_days

        logger.info(
            "Checking drift for sport=%r window=%d days", sport_slug, window
        )

        picks = self._fetch_resolved_picks(sport_slug, window)
        prediction_dicts = self._picks_to_prediction_dicts(picks)
        n = len(prediction_dicts)

        if n < self._min_predictions:
            logger.warning(
                "Insufficient predictions for drift check: sport=%r n=%d min=%d",
                sport_slug,
                n,
                self._min_predictions,
            )
            return DriftResult(
                sport_slug=sport_slug,
                has_drift=False,
                severity="none",
                calibration_error=0.0,
                brier_score=0.0,
                accuracy=0.0,
                roi_pct=0.0,
                n_predictions=n,
                recommendation="none",
                window_days=window,
                details={"insufficient_data": True, "min_required": self._min_predictions},
            )

        ece = self.compute_calibration_error(prediction_dicts)
        brier = self.compute_brier_score(prediction_dicts)
        wins = sum(1 for p in prediction_dicts if p["outcome"] == 1.0)
        accuracy = wins / n if n > 0 else 0.0
        roi_pct = self._compute_roi(prediction_dicts)

        severity, recommendation = self._classify_severity(ece, brier, accuracy)
        has_drift = severity != "none"

        result = DriftResult(
            sport_slug=sport_slug,
            has_drift=has_drift,
            severity=severity,
            calibration_error=ece,
            brier_score=brier,
            accuracy=accuracy,
            roi_pct=roi_pct,
            n_predictions=n,
            recommendation=recommendation,
            window_days=window,
            details={
                "ece_threshold": settings.drift_calibration_threshold,
                "brier_threshold": settings.drift_brier_threshold,
                "accuracy_floor": settings.drift_accuracy_floor,
                "ece_breach": ece > settings.drift_calibration_threshold,
                "brier_breach": brier > settings.drift_brier_threshold,
                "accuracy_breach": accuracy < settings.drift_accuracy_floor,
                "wins": wins,
                "losses": n - wins,
            },
        )

        logger.info("Drift check result: %s", result)
        return result

    def snapshot(self, sport_slug: str, window_days: Optional[int] = None) -> None:
        """Compute drift metrics and persist a :class:`DriftSnapshot` row.

        Idempotent within the same calendar day — if a snapshot for
        (sport_slug, today) already exists it is updated in-place.
        """
        window = window_days if window_days is not None else settings.drift_window_days
        result = self.check_drift(sport_slug, window_days=window)
        today = date.today()

        with get_session() as session:
            existing = (
                session.query(DriftSnapshot)
                .filter(
                    DriftSnapshot.sport_slug == sport_slug,
                    DriftSnapshot.snapshot_date == today,
                    DriftSnapshot.window_days == window,
                )
                .first()
            )

            if existing is not None:
                snap = existing
                logger.debug(
                    "Updating existing DriftSnapshot id=%d for sport=%r date=%s",
                    snap.id,
                    sport_slug,
                    today,
                )
            else:
                snap = DriftSnapshot(
                    sport_slug=sport_slug,
                    snapshot_date=today,
                    window_days=window,
                )
                session.add(snap)

            snap.calibration_error = result.calibration_error
            snap.brier_score = result.brier_score
            snap.log_loss = _safe_log_loss(result)
            snap.n_predictions = result.n_predictions
            snap.accuracy = result.accuracy
            snap.roi_pct = result.roi_pct
            snap.drift_detected = result.has_drift
            snap.drift_severity = result.severity
            snap.retrain_triggered = result.recommendation in (
                "retrain",
                "emergency_retrain",
            )
            snap.details = result.details

            session.commit()
            logger.info(
                "DriftSnapshot persisted: sport=%r date=%s severity=%r",
                sport_slug,
                today,
                result.severity,
            )

    def should_retrain(self, sport_slug: str) -> tuple[bool, str]:
        """Return (should_retrain, reason).

        Combines drift-based check with schedule-based check from
        :class:`~sharpedge.warroom.retrainer.AutoRetrainer`.

        Drift takes priority; schedule is the fallback.
        """
        # --- Drift check ---
        result = self.check_drift(sport_slug)

        if result.severity == "emergency":
            return (
                True,
                f"Emergency drift detected: accuracy={result.accuracy:.3f} < "
                f"floor={settings.drift_accuracy_floor}",
            )

        if result.severity == "critical":
            return (
                True,
                f"Critical drift detected: ECE={result.calibration_error:.4f}, "
                f"Brier={result.brier_score:.4f}",
            )

        # --- Schedule check (fallback) ---
        from sharpedge.warroom.retrainer import AutoRetrainer
        retrainer = AutoRetrainer()
        retrainer.register_sport(sport_slug)
        if retrainer.check_due(sport_slug):
            return (
                True,
                f"Scheduled retrain due for {sport_slug!r} "
                f"(frequency={retrainer._schedules[sport_slug].frequency_days}d)",
            )

        if result.severity == "warning":
            return (
                False,
                f"Warning drift only (ECE={result.calibration_error:.4f} or "
                f"Brier={result.brier_score:.4f}): recalibration recommended, "
                f"retrain not yet triggered",
            )

        return False, f"No drift detected for {sport_slug!r}; schedule is current"

    def run_check_all(self) -> dict[str, DriftResult]:
        """Check drift for all active sports, snapshot each, return summary dict."""
        with get_session() as session:
            from sharpedge.db.models import Sport

            active_sports = (
                session.query(Sport).filter(Sport.active.is_(True)).all()
            )
            slugs = [s.slug for s in active_sports]
            session.expunge_all()

        if not slugs:
            logger.warning("No active sports found — drift check skipped")
            return {}

        results: dict[str, DriftResult] = {}
        for slug in slugs:
            try:
                self.snapshot(slug)
                result = self.check_drift(slug)
                results[slug] = result
                logger.info(
                    "run_check_all: sport=%r severity=%r has_drift=%s",
                    slug,
                    result.severity,
                    result.has_drift,
                )
            except Exception:
                logger.exception("Drift check failed for sport=%r", slug)

        n_drifted = sum(1 for r in results.values() if r.has_drift)
        logger.info(
            "run_check_all complete: %d sports checked, %d with drift",
            len(results),
            n_drifted,
        )
        return results


# ---------------------------------------------------------------------------
# SmartRetrainer
# ---------------------------------------------------------------------------


class SmartRetrainer:
    """Schedule-aware retrainer augmented with drift-based emergency triggering.

    Uses :class:`~sharpedge.warroom.retrainer.AutoRetrainer` for schedule management
    and adds:
    - Drift-based emergency triggering via :class:`DriftDetector`
    - ``check_and_retrain`` — unified decision + action method
    - ``execute_retrain``    — lightweight placeholder that logs a
      :class:`RetrainRequest` (actual model training is asynchronous/heavy
      and must be triggered outside this process)
    """

    def __init__(self) -> None:
        from sharpedge.warroom.retrainer import AutoRetrainer
        self._retrainer = AutoRetrainer()
        self.detector = DriftDetector()
        self._pending_requests: list[RetrainRequest] = []

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------

    def check_and_retrain(self, sport: str) -> dict:
        """Evaluate schedule + drift, execute retrain if warranted.

        Returns a summary dict with the action taken and supporting detail.

        Decision priority:
        1. Emergency drift (accuracy below floor) → emergency_retrain
        2. Critical drift (ECE + Brier both breached) → retrain
        3. Schedule due → scheduled_retrain
        4. Warning drift → recalibrate (no full retrain)
        5. Healthy → no_action
        """
        logger.info("SmartRetrainer.check_and_retrain: sport=%r", sport)

        # Ensure sport is registered
        if sport not in self._schedules:
            self.register_sport(sport)

        drift_result = self.detector.check_drift(sport)

        # 1. Emergency
        if drift_result.severity == "emergency":
            reason = (
                f"Emergency: accuracy={drift_result.accuracy:.3f} < "
                f"floor={settings.drift_accuracy_floor} "
                f"(n={drift_result.n_predictions})"
            )
            executed = self.execute_retrain(sport, reason=reason, severity="emergency")
            return {
                "action": "emergency_retrain",
                "sport": sport,
                "triggered_by": "drift",
                "reason": reason,
                "executed": executed,
                "drift": drift_result,
            }

        # 2. Critical drift
        if drift_result.severity == "critical":
            reason = (
                f"Critical drift: ECE={drift_result.calibration_error:.4f} "
                f"Brier={drift_result.brier_score:.4f}"
            )
            executed = self.execute_retrain(sport, reason=reason, severity="critical")
            return {
                "action": "retrain",
                "sport": sport,
                "triggered_by": "drift",
                "reason": reason,
                "executed": executed,
                "drift": drift_result,
            }

        # 3. Schedule due
        if self.check_due(sport):
            freq = self._schedules[sport].frequency_days
            reason = f"Scheduled retrain due (every {freq} days)"
            executed = self.execute_retrain(sport, reason=reason, severity="scheduled")
            return {
                "action": "scheduled_retrain",
                "sport": sport,
                "triggered_by": "schedule",
                "reason": reason,
                "executed": executed,
                "drift": drift_result,
            }

        # 4. Warning drift — recommend recalibration only
        if drift_result.severity == "warning":
            logger.warning(
                "Warning drift for sport=%r — recalibration recommended but "
                "retrain not triggered. ECE=%.4f Brier=%.4f",
                sport,
                drift_result.calibration_error,
                drift_result.brier_score,
            )
            return {
                "action": "recalibrate",
                "sport": sport,
                "triggered_by": "drift",
                "reason": (
                    f"Warning drift: ECE={drift_result.calibration_error:.4f} "
                    f"or Brier={drift_result.brier_score:.4f} breached threshold"
                ),
                "executed": False,
                "drift": drift_result,
            }

        # 5. All healthy
        logger.info(
            "SmartRetrainer: no action needed for sport=%r (severity=none)", sport
        )
        return {
            "action": "no_action",
            "sport": sport,
            "triggered_by": None,
            "reason": "All metrics healthy and schedule is current",
            "executed": False,
            "drift": drift_result,
        }

    def execute_retrain(
        self,
        sport: str,
        reason: str = "unspecified",
        severity: str = "unknown",
    ) -> bool:
        """Log a retrain request and return True.

        This is an intentional placeholder.  Full model training is expensive
        (minutes to hours) and must be dispatched asynchronously by an
        orchestrator (e.g. a Celery task, a cron job, or a CI pipeline trigger).

        What this method does:
        - Creates a :class:`RetrainRequest` and appends it to ``_pending_requests``
        - Emits a structured log at WARNING level so monitoring can detect it
        - Returns True to signal "request was accepted"

        It does NOT actually call any training code.
        """
        request = RetrainRequest(
            sport=sport,
            reason=reason,
            severity=severity,
        )
        self._pending_requests.append(request)

        logger.warning(
            "RETRAIN_REQUEST sport=%r reason=%r severity=%r requested_at=%s",
            sport,
            reason,
            severity,
            request.requested_at.isoformat(),
        )

        # Persist intent to DriftSnapshot so the orchestrator can poll it
        try:
            self._persist_retrain_request(request)
        except Exception:
            logger.exception(
                "Failed to persist retrain request for sport=%r", sport
            )
            # Do not raise — logging the intent is still valuable

        return True

    def get_pending_requests(self) -> list[RetrainRequest]:
        """Return copy of pending retrain requests (not yet cleared)."""
        return list(self._pending_requests)

    def clear_pending(self, sport: Optional[str] = None) -> int:
        """Clear pending requests.  Pass sport to clear only that sport.

        Returns the number of requests removed.
        """
        before = len(self._pending_requests)
        if sport is None:
            self._pending_requests.clear()
        else:
            self._pending_requests = [
                r for r in self._pending_requests if r.sport != sport
            ]
        removed = before - len(self._pending_requests)
        logger.info("Cleared %d pending retrain request(s) for sport=%r", removed, sport)
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_retrain_request(self, request: RetrainRequest) -> None:
        """Update the latest DriftSnapshot to mark retrain_triggered=True."""
        today = date.today()
        with get_session() as session:
            snap = (
                session.query(DriftSnapshot)
                .filter(
                    DriftSnapshot.sport_slug == request.sport,
                    DriftSnapshot.snapshot_date == today,
                )
                .order_by(DriftSnapshot.id.desc())
                .first()
            )
            if snap is not None:
                snap.retrain_triggered = True
                details = snap.details or {}
                details["retrain_reason"] = request.reason
                details["retrain_severity"] = request.severity
                details["retrain_requested_at"] = request.requested_at.isoformat()
                snap.details = details
                session.commit()
                logger.debug(
                    "Updated DriftSnapshot id=%d retrain_triggered=True", snap.id
                )
            else:
                logger.warning(
                    "No DriftSnapshot found for sport=%r date=%s — "
                    "retrain request not persisted to snapshot",
                    request.sport,
                    today,
                )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _safe_log_loss(result: DriftResult) -> Optional[float]:
    """Approximate log-loss from accuracy (rough proxy when full dist unavailable)."""
    if result.accuracy <= 0.0 or result.accuracy >= 1.0:
        return None
    try:
        return -(
            result.accuracy * math.log(result.accuracy)
            + (1.0 - result.accuracy) * math.log(1.0 - result.accuracy)
        )
    except (ValueError, ZeroDivisionError):
        return None
