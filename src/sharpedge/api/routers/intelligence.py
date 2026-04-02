"""Intelligence endpoints — edges, drift, calibration, and market matrix."""
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import DailyPick, DriftSnapshot, EdgeLog

router = APIRouter(tags=["intelligence"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_edge(e: EdgeLog) -> dict:
    return {
        "id": e.id,
        "agent_name": e.agent_name,
        "sport_slug": e.sport_slug,
        "league": e.league,
        "market": e.market,
        "edge_type": e.edge_type,
        "edge_value": e.edge_value,
        "sample_size": e.sample_size,
        "confidence_interval_lo": e.confidence_interval_lo,
        "confidence_interval_hi": e.confidence_interval_hi,
        "is_significant": e.is_significant,
        "discovered_date": str(e.discovered_date),
        "expired_date": str(e.expired_date) if e.expired_date else None,
        "metadata_json": e.metadata_json,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/intelligence/edges")
async def get_edges(
    sport: Optional[str] = Query(None, description="Filter by sport slug"),
    league: Optional[str] = Query(None, description="Filter by league name"),
    agent: Optional[str] = Query(None, description="Filter by agent name"),
    is_significant: Optional[bool] = Query(None, description="Only return statistically significant edges"),
    active_only: bool = Query(True, description="Only return edges that have not yet expired"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Return all discovered edges, with optional filters."""
    q = db.query(EdgeLog)

    if sport:
        q = q.filter(EdgeLog.sport_slug == sport)
    if league:
        q = q.filter(EdgeLog.league == league)
    if agent:
        q = q.filter(EdgeLog.agent_name == agent)
    if is_significant is not None:
        q = q.filter(EdgeLog.is_significant == is_significant)
    if active_only:
        q = q.filter(EdgeLog.expired_date.is_(None))

    total = q.count()
    edges = q.order_by(EdgeLog.discovered_date.desc()).offset(offset).limit(limit).all()

    return {
        "status": "ok",
        "data": [_serialize_edge(e) for e in edges],
        "meta": {
            "count": len(edges),
            "total": total,
            "offset": offset,
            "generated_at": datetime.now().isoformat(),
        },
    }


@router.get("/intelligence/drift")
async def get_drift_status(db: Session = Depends(get_db)):
    """Return the latest drift snapshot for each sport."""
    # Subquery: latest snapshot_date per sport
    latest_subq = (
        db.query(
            DriftSnapshot.sport_slug,
            func.max(DriftSnapshot.snapshot_date).label("max_date"),
        )
        .group_by(DriftSnapshot.sport_slug)
        .subquery()
    )

    rows = (
        db.query(DriftSnapshot)
        .join(
            latest_subq,
            (DriftSnapshot.sport_slug == latest_subq.c.sport_slug)
            & (DriftSnapshot.snapshot_date == latest_subq.c.max_date),
        )
        .all()
    )

    result: dict = {}
    for r in rows:
        result[r.sport_slug] = {
            "snapshot_date": str(r.snapshot_date),
            "window_days": r.window_days,
            "calibration_error": r.calibration_error,
            "brier_score": r.brier_score,
            "log_loss": r.log_loss,
            "accuracy": r.accuracy,
            "roi_pct": r.roi_pct,
            "n_predictions": r.n_predictions,
            "drift_detected": r.drift_detected,
            "severity": r.drift_severity,
            "retrain_triggered": r.retrain_triggered,
        }

    return {
        "status": "ok",
        "data": result,
        "meta": {"sports_tracked": len(result), "generated_at": datetime.now().isoformat()},
    }


@router.get("/intelligence/strategy-recommendations")
async def get_strategy_recommendations(
    min_sample: int = Query(20, ge=1, description="Minimum sample size to surface a recommendation"),
    db: Session = Depends(get_db),
):
    """Aggregate significant active edges into actionable strategy recommendations."""
    edges = (
        db.query(EdgeLog)
        .filter(
            EdgeLog.is_significant == True,  # noqa: E712
            EdgeLog.expired_date.is_(None),
            EdgeLog.sample_size >= min_sample,
        )
        .order_by(EdgeLog.edge_value.desc())
        .all()
    )

    # Group by (sport, league, market) to de-duplicate overlapping signals
    seen: set = set()
    recommendations = []
    for e in edges:
        key = (e.sport_slug, e.league, e.market)
        if key in seen:
            continue
        seen.add(key)
        recommendations.append(
            {
                "sport_slug": e.sport_slug,
                "league": e.league,
                "market": e.market,
                "edge_type": e.edge_type,
                "edge_value": e.edge_value,
                "sample_size": e.sample_size,
                "confidence_interval": [e.confidence_interval_lo, e.confidence_interval_hi],
                "leading_agent": e.agent_name,
                "discovered_date": str(e.discovered_date),
                "recommendation": (
                    f"Focus on {e.market} bets in {e.league} "
                    f"(edge={e.edge_value:.1%}, n={e.sample_size})"
                ),
            }
        )

    return {
        "status": "ok",
        "data": recommendations,
        "meta": {"count": len(recommendations), "generated_at": datetime.now().isoformat()},
    }


@router.get("/intelligence/calibration")
async def get_calibration(
    n_bins: int = Query(10, ge=2, le=20, description="Number of probability bins"),
    db: Session = Depends(get_db),
):
    """
    Return calibration curve data for resolved picks.
    Bins picks by model_prob and computes predicted_avg vs actual_win_rate per bin.
    """
    resolved = (
        db.query(DailyPick)
        .filter(DailyPick.result.isnot(None))
        .all()
    )

    if not resolved:
        return {
            "status": "ok",
            "data": {"bins": []},
            "meta": {"total_resolved": 0, "generated_at": datetime.now().isoformat()},
        }

    # Build bins
    bin_size = 1.0 / n_bins
    buckets: dict[int, dict] = {
        i: {"probs": [], "wins": 0, "count": 0, "profit_loss": 0.0}
        for i in range(n_bins)
    }

    for pick in resolved:
        bin_idx = min(int(pick.model_prob / bin_size), n_bins - 1)
        buckets[bin_idx]["probs"].append(pick.model_prob)
        buckets[bin_idx]["count"] += 1
        if pick.result == "win":
            buckets[bin_idx]["wins"] += 1
        buckets[bin_idx]["profit_loss"] += pick.profit_loss or 0.0

    bins = []
    for i, bucket in buckets.items():
        lo = round(i * bin_size, 4)
        hi = round((i + 1) * bin_size, 4)
        count = bucket["count"]
        if count == 0:
            continue
        predicted_avg = sum(bucket["probs"]) / count
        actual_win_rate = bucket["wins"] / count
        roi = bucket["profit_loss"] / count if count else 0.0
        bins.append(
            {
                "confidence_lo": lo,
                "confidence_hi": hi,
                "predicted_avg": round(predicted_avg, 4),
                "actual_win_rate": round(actual_win_rate, 4),
                "count": count,
                "roi": round(roi, 4),
            }
        )

    return {
        "status": "ok",
        "data": {"bins": bins},
        "meta": {
            "total_resolved": len(resolved),
            "n_bins": n_bins,
            "generated_at": datetime.now().isoformat(),
        },
    }


@router.get("/intelligence/league-market-matrix")
async def get_league_market_matrix(
    min_picks: int = Query(5, ge=1, description="Minimum resolved picks to include a cell"),
    db: Session = Depends(get_db),
):
    """
    Return a league x market performance matrix for resolved picks.
    Returns {matrix: {league: {market: {picks, wins, roi}}}}.
    """
    resolved = (
        db.query(DailyPick)
        .filter(DailyPick.result.isnot(None))
        .all()
    )

    # Aggregate in Python (avoids complex SQL across backends)
    matrix: dict[str, dict[str, dict]] = {}
    for pick in resolved:
        lg = pick.league or "Unknown"
        mk = pick.pick_market or "Unknown"
        matrix.setdefault(lg, {}).setdefault(mk, {"picks": 0, "wins": 0, "profit_loss": 0.0})
        cell = matrix[lg][mk]
        cell["picks"] += 1
        if pick.result == "win":
            cell["wins"] += 1
        cell["profit_loss"] += pick.profit_loss or 0.0

    # Compute ROI and prune sparse cells
    pruned: dict[str, dict] = {}
    for lg, markets in matrix.items():
        for mk, cell in markets.items():
            if cell["picks"] < min_picks:
                continue
            roi = cell["profit_loss"] / cell["picks"]
            pruned.setdefault(lg, {})[mk] = {
                "picks": cell["picks"],
                "wins": cell["wins"],
                "win_rate": round(cell["wins"] / cell["picks"], 4),
                "roi": round(roi, 4),
            }

    return {
        "status": "ok",
        "data": {"matrix": pruned},
        "meta": {
            "leagues": len(pruned),
            "min_picks_threshold": min_picks,
            "generated_at": datetime.now().isoformat(),
        },
    }
