"""Evolution endpoints — agent leaderboard, specialization, bankroll tracking."""
from datetime import date, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from sharpedge.api.deps import get_db
from sharpedge.db.models import (
    Agent,
    AgentEvolution,
    AgentPerformance,
    AgentPrediction,
    BankrollLedger,
)

router = APIRouter(tags=["evolution"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perf_to_dict(p: AgentPerformance, agent_name: str) -> dict:
    return {
        "agent_name": agent_name,
        "sport_slug": p.sport_slug,
        "date": str(p.date),
        "window_days": p.window_days,
        "total_bets": p.total_bets,
        "wins": p.wins,
        "win_rate": round(p.wins / p.total_bets, 4) if p.total_bets else None,
        "roi_pct": p.roi_pct,
        "clv_pct": p.clv_pct,
        "avg_confidence": p.avg_confidence,
        "brier_score": p.brier_score,
    }


def _evo_to_dict(e: AgentEvolution) -> dict:
    return {
        "id": e.id,
        "agent_name": e.agent_name,
        "event_type": e.event_type,
        "parent_agent": e.parent_agent,
        "event_date": str(e.event_date),
        "reason": e.reason,
        "config_before": e.config_before,
        "config_after": e.config_after,
        "performance_at_event": e.performance_at_event,
    }


def _ledger_to_dict(l: BankrollLedger) -> dict:
    return {
        "id": l.id,
        "event_date": str(l.event_date),
        "event_type": l.event_type,
        "pick_id": l.pick_id,
        "amount": l.amount,
        "bankroll_before": l.bankroll_before,
        "bankroll_after": l.bankroll_after,
        "drawdown_pct": l.drawdown_pct,
        "circuit_breaker_level": l.circuit_breaker_level,
        "notes": l.notes,
    }


# ---------------------------------------------------------------------------
# Agent evolution endpoints
# ---------------------------------------------------------------------------

@router.get("/agents/evolution/leaderboard")
async def agent_leaderboard(
    sport: Optional[str] = Query(None, description="Filter by sport slug"),
    window_days: int = Query(30, ge=1, le=365, description="Performance window in days"),
    min_bets: int = Query(5, ge=1, description="Minimum bets to include an agent"),
    db: Session = Depends(get_db),
):
    """Return agents ranked by ROI from their latest performance snapshot."""
    # Latest snapshot date per agent + sport combo
    latest_subq = (
        db.query(
            AgentPerformance.agent_id,
            AgentPerformance.sport_slug,
            func.max(AgentPerformance.date).label("max_date"),
        )
        .filter(AgentPerformance.window_days == window_days)
        .group_by(AgentPerformance.agent_id, AgentPerformance.sport_slug)
        .subquery()
    )

    q = (
        db.query(AgentPerformance, Agent.name)
        .join(Agent, AgentPerformance.agent_id == Agent.id)
        .join(
            latest_subq,
            (AgentPerformance.agent_id == latest_subq.c.agent_id)
            & (AgentPerformance.sport_slug == latest_subq.c.sport_slug)
            & (AgentPerformance.date == latest_subq.c.max_date),
        )
        .filter(AgentPerformance.total_bets >= min_bets)
    )

    if sport:
        q = q.filter(AgentPerformance.sport_slug == sport)

    rows = q.all()

    leaderboard = [_perf_to_dict(perf, name) for perf, name in rows]
    # Sort by ROI descending (None treated as worst)
    leaderboard.sort(key=lambda x: (x["roi_pct"] is None, -(x["roi_pct"] or 0)))

    # Add rank
    for i, entry in enumerate(leaderboard, start=1):
        entry["rank"] = i

    return {
        "status": "ok",
        "data": leaderboard,
        "meta": {
            "count": len(leaderboard),
            "window_days": window_days,
            "generated_at": datetime.now().isoformat(),
        },
    }


@router.get("/agents/evolution/history")
async def evolution_history(
    agent: Optional[str] = Query(None, description="Filter by agent name"),
    event_type: Optional[str] = Query(None, description="Filter by event type (spawned, mutated, promoted, deprecated)"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Return agent evolution event history."""
    q = db.query(AgentEvolution)

    if agent:
        q = q.filter(AgentEvolution.agent_name == agent)
    if event_type:
        q = q.filter(AgentEvolution.event_type == event_type)

    total = q.count()
    events = q.order_by(AgentEvolution.event_date.desc()).offset(offset).limit(limit).all()

    return {
        "status": "ok",
        "data": [_evo_to_dict(e) for e in events],
        "meta": {
            "count": len(events),
            "total": total,
            "offset": offset,
            "generated_at": datetime.now().isoformat(),
        },
    }


@router.get("/agents/evolution/specialization")
async def agent_specialization(
    min_count: int = Query(10, ge=1, description="Minimum predictions to include a cell"),
    db: Session = Depends(get_db),
):
    """
    Return agent x league specialization matrix.
    Returns {agent: {league: {accuracy, roi, count}}}.

    NOTE: ROI is not directly stored on AgentPrediction — we approximate it
    as (wins / count) relative to a flat-bet baseline where win=+1, loss=-1.
    """
    # Pull all resolved agent predictions
    rows = (
        db.query(
            Agent.name,
            AgentPrediction.sport_slug,
            AgentPrediction.market,
            AgentPrediction.correct,
        )
        .join(Agent, AgentPrediction.agent_id == Agent.id)
        .filter(AgentPrediction.correct.isnot(None))
        .all()
    )

    # Aggregate: agent -> sport+market -> {count, wins}
    matrix: dict[str, dict[str, dict]] = {}
    for name, sport_slug, market, correct in rows:
        key = f"{sport_slug}:{market}"
        matrix.setdefault(name, {}).setdefault(key, {"count": 0, "wins": 0})
        matrix[name][key]["count"] += 1
        if correct:
            matrix[name][key]["wins"] += 1

    # Prune sparse cells and compute stats
    pruned: dict[str, dict] = {}
    for agent_name, cells in matrix.items():
        for key, cell in cells.items():
            if cell["count"] < min_count:
                continue
            accuracy = cell["wins"] / cell["count"]
            # Approximate flat-bet ROI: +1 per win, -1 per loss (normalised to per-bet)
            roi = (2 * cell["wins"] - cell["count"]) / cell["count"]
            pruned.setdefault(agent_name, {})[key] = {
                "accuracy": round(accuracy, 4),
                "roi": round(roi, 4),
                "count": cell["count"],
            }

    return {
        "status": "ok",
        "data": pruned,
        "meta": {
            "agents": len(pruned),
            "min_count_threshold": min_count,
            "generated_at": datetime.now().isoformat(),
        },
    }


@router.get("/agents/{agent_name}/performance")
async def agent_performance(
    agent_name: str,
    sport: Optional[str] = Query(None, description="Filter by sport slug"),
    db: Session = Depends(get_db),
):
    """Return detailed performance breakdown for a single agent."""
    agent = db.query(Agent).filter(Agent.name == agent_name).first()
    if not agent:
        return {
            "status": "error",
            "data": {"error": f"Agent '{agent_name}' not found"},
            "meta": {},
        }

    # Latest performance snapshot
    perf_q = (
        db.query(AgentPerformance)
        .filter(AgentPerformance.agent_id == agent.id)
        .order_by(AgentPerformance.date.desc())
    )
    if sport:
        perf_q = perf_q.filter(AgentPerformance.sport_slug == sport)
    latest_perf = perf_q.first()

    # Recent predictions (last 20 resolved)
    pred_q = (
        db.query(AgentPrediction)
        .filter(
            AgentPrediction.agent_id == agent.id,
            AgentPrediction.correct.isnot(None),
        )
        .order_by(AgentPrediction.match_date.desc())
    )
    if sport:
        pred_q = pred_q.filter(AgentPrediction.sport_slug == sport)
    recent_preds = pred_q.limit(20).all()

    # Per-league breakdown
    all_resolved = (
        db.query(AgentPrediction)
        .filter(
            AgentPrediction.agent_id == agent.id,
            AgentPrediction.correct.isnot(None),
        )
    )
    if sport:
        all_resolved = all_resolved.filter(AgentPrediction.sport_slug == sport)

    league_stats: dict[str, dict] = {}
    for pred in all_resolved.all():
        lg = pred.sport_slug  # league not on AgentPrediction; use sport as proxy
        league_stats.setdefault(lg, {"count": 0, "wins": 0})
        league_stats[lg]["count"] += 1
        if pred.correct:
            league_stats[lg]["wins"] += 1

    per_league = {
        lg: {
            "count": v["count"],
            "accuracy": round(v["wins"] / v["count"], 4) if v["count"] else None,
        }
        for lg, v in league_stats.items()
    }

    return {
        "status": "ok",
        "data": {
            "agent": {
                "name": agent.name,
                "agent_type": agent.agent_type,
                "description": agent.description,
                "active": agent.active,
            },
            "latest_performance": (
                _perf_to_dict(latest_perf, agent.name) if latest_perf else None
            ),
            "recent_predictions": [
                {
                    "match_date": str(p.match_date),
                    "home_team": p.home_team,
                    "away_team": p.away_team,
                    "market": p.market,
                    "predicted_outcome": p.predicted_outcome,
                    "actual_outcome": p.actual_outcome,
                    "correct": p.correct,
                    "confidence": p.confidence,
                }
                for p in recent_preds
            ],
            "per_league": per_league,
        },
        "meta": {"generated_at": datetime.now().isoformat()},
    }


# ---------------------------------------------------------------------------
# Bankroll endpoints
# ---------------------------------------------------------------------------

@router.get("/bankroll/status")
async def bankroll_status(db: Session = Depends(get_db)):
    """Return the current bankroll status from the latest ledger entry."""
    latest = (
        db.query(BankrollLedger)
        .order_by(BankrollLedger.event_date.desc(), BankrollLedger.id.desc())
        .first()
    )

    if not latest:
        return {
            "status": "ok",
            "data": {"message": "No bankroll entries found"},
            "meta": {"generated_at": datetime.now().isoformat()},
        }

    # Compute daily P&L (sum of amounts for today)
    today = date.today()
    daily_rows = (
        db.query(BankrollLedger)
        .filter(BankrollLedger.event_date == today)
        .all()
    )
    daily_pnl = sum(r.amount for r in daily_rows)

    # Weekly P&L — sum across last 7 days
    from datetime import timedelta
    week_ago = today - timedelta(days=7)
    weekly_rows = (
        db.query(BankrollLedger)
        .filter(BankrollLedger.event_date >= week_ago)
        .all()
    )
    weekly_pnl = sum(r.amount for r in weekly_rows)

    # Peak bankroll (all-time high of bankroll_after)
    peak_row = (
        db.query(func.max(BankrollLedger.bankroll_after))
        .scalar()
    )

    return {
        "status": "ok",
        "data": {
            "bankroll": latest.bankroll_after,
            "peak": peak_row,
            "drawdown_pct": latest.drawdown_pct,
            "circuit_breaker_level": latest.circuit_breaker_level,
            "daily_pnl": daily_pnl,
            "weekly_pnl": weekly_pnl,
            "last_event_date": str(latest.event_date),
            "last_event_type": latest.event_type,
        },
        "meta": {"generated_at": datetime.now().isoformat()},
    }


@router.get("/bankroll/history")
async def bankroll_history(
    start_date: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    event_type: Optional[str] = Query(None, description="Filter by event type (stake, payout, deposit, withdrawal)"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """Return bankroll ledger history for charting."""
    q = db.query(BankrollLedger)

    if start_date:
        try:
            q = q.filter(BankrollLedger.event_date >= date.fromisoformat(start_date))
        except ValueError:
            return {"status": "error", "data": {"error": "Invalid start_date format"}, "meta": {}}

    if end_date:
        try:
            q = q.filter(BankrollLedger.event_date <= date.fromisoformat(end_date))
        except ValueError:
            return {"status": "error", "data": {"error": "Invalid end_date format"}, "meta": {}}

    if event_type:
        q = q.filter(BankrollLedger.event_type == event_type)

    total = q.count()
    rows = q.order_by(BankrollLedger.event_date.asc(), BankrollLedger.id.asc()).offset(offset).limit(limit).all()

    return {
        "status": "ok",
        "data": [_ledger_to_dict(r) for r in rows],
        "meta": {
            "count": len(rows),
            "total": total,
            "offset": offset,
            "generated_at": datetime.now().isoformat(),
        },
    }
