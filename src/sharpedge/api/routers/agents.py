"""Agent API endpoints — agent swarm monitoring and leaderboard."""
from fastapi import APIRouter

router = APIRouter(tags=["agents"])

# Global tracker instance (would be initialized at app startup in production)
_tracker = None


def set_tracker(tracker):
    global _tracker
    _tracker = tracker


@router.get("/agents")
async def list_agents():
    """List all registered agents."""
    agents = [
        {"name": "statistical_agent", "type": "statistical", "description": "Dixon-Coles + BVP blend"},
        {"name": "gradient_agent", "type": "ml", "description": "XGBoost + CatBoost + LightGBM stacking"},
        {"name": "market_agent", "type": "market", "description": "Odds-only devigged probability model"},
        {"name": "form_momentum_agent", "type": "context", "description": "Last-10-match rolling form"},
        {"name": "ovr_specialist_agent", "type": "ml", "description": "One-vs-Rest binary classifiers"},
        {"name": "contrarian_agent", "type": "market", "description": "Fades heavy favorites"},
        {"name": "meta_consensus_agent", "type": "context", "description": "External prediction aggregator"},
        {"name": "h2h_venue_agent", "type": "context", "description": "Head-to-head specialist"},
        {"name": "league_specialist", "type": "niche", "description": "Per-league calibrated predictions"},
    ]

    return {"status": "ok", "data": agents, "meta": {"count": len(agents)}}


@router.get("/agents/leaderboard")
async def agent_leaderboard():
    """Get agents ranked by performance."""
    if _tracker is None:
        return {"status": "ok", "data": [], "meta": {"message": "No tracker initialized"}}

    leaderboard = _tracker.get_leaderboard(min_bets=5)
    return {"status": "ok", "data": leaderboard, "meta": {"count": len(leaderboard)}}


@router.get("/agents/{agent_name}")
async def get_agent_stats(agent_name: str):
    """Get detailed stats for a specific agent."""
    if _tracker is None:
        return {"status": "ok", "data": {"agent_name": agent_name, "total_bets": 0}}

    stats = _tracker.get_agent_stats(agent_name)
    return {"status": "ok", "data": {"agent_name": agent_name, **stats}}
