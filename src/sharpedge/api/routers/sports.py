"""Sports API endpoints — multi-sport discovery and navigation."""
from fastapi import APIRouter

router = APIRouter(tags=["sports"])


@router.get("/sports")
async def list_sports():
    """List all registered sports with their configs."""
    from sharpedge.core.sport import sport_registry

    sports = []
    for slug in sport_registry.list_sports():
        cfg = sport_registry.get(slug)
        sports.append({
            "name": cfg.name,
            "slug": cfg.slug,
            "markets": [
                {"name": m.name, "outcomes": list(m.outcomes), "description": m.description}
                for m in cfg.markets
            ],
            "n_markets": len(cfg.markets),
        })

    return {"status": "ok", "data": sports, "meta": {"count": len(sports)}}


@router.get("/sports/{sport}")
async def get_sport(sport: str):
    """Get details for a specific sport."""
    from sharpedge.core.sport import sport_registry

    if not sport_registry.is_registered(sport):
        return {"status": "error", "message": f"Sport '{sport}' not found"}

    cfg = sport_registry.get(sport)
    return {
        "status": "ok",
        "data": {
            "name": cfg.name,
            "slug": cfg.slug,
            "markets": [
                {"name": m.name, "outcomes": list(m.outcomes), "description": m.description}
                for m in cfg.markets
            ],
            "min_train_seasons": cfg.min_train_seasons,
            "default_mc_sims": cfg.default_mc_sims,
            "default_min_consensus": cfg.default_min_consensus,
        },
    }
