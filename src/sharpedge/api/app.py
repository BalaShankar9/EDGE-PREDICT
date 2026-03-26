"""FastAPI application factory."""
from fastapi import FastAPI
from sharpedge.api.routers import health, leagues, matches, pipeline, predictions, picks, track_record, sports, agents


def create_app() -> FastAPI:
    # Register all sports on app startup
    import sharpedge.sports.football  # noqa: F401
    import sharpedge.sports.tennis  # noqa: F401
    import sharpedge.sports.basketball  # noqa: F401
    import sharpedge.sports.ice_hockey  # noqa: F401
    import sharpedge.sports.american_football  # noqa: F401
    import sharpedge.sports.baseball  # noqa: F401

    app = FastAPI(title="SharpEdge AI", description="AI-powered multi-sport prediction API", version="0.4.0")
    app.include_router(health.router, prefix="/api")
    app.include_router(pipeline.router, prefix="/api")
    app.include_router(predictions.router, prefix="/api")
    app.include_router(picks.router, prefix="/api")
    app.include_router(track_record.router, prefix="/api")
    app.include_router(leagues.router, prefix="/api")
    app.include_router(matches.router, prefix="/api")
    app.include_router(sports.router, prefix="/api")
    app.include_router(agents.router, prefix="/api")
    return app
