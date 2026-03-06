"""FastAPI application factory."""
from fastapi import FastAPI
from sharpedge.api.routers import health, leagues, matches, pipeline, predictions, picks, track_record


def create_app() -> FastAPI:
    app = FastAPI(title="SharpEdge AI", description="AI-powered football prediction API", version="0.3.0")
    app.include_router(health.router, prefix="/api")
    app.include_router(pipeline.router, prefix="/api")
    app.include_router(predictions.router, prefix="/api")
    app.include_router(picks.router, prefix="/api")
    app.include_router(track_record.router, prefix="/api")
    app.include_router(leagues.router, prefix="/api")
    app.include_router(matches.router, prefix="/api")
    return app
