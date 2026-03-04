"""Pipeline trigger endpoints (API-key protected)."""
import logging
from datetime import datetime, date

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from sharpedge.api.auth import require_api_key
from sharpedge.api.deps import get_db
from sharpedge.db.models import PipelineRun, DailyPick
from sharpedge.pipeline.daily import DailyPipeline
from sharpedge.pipeline.resolver import resolve_pick

logger = logging.getLogger(__name__)
router = APIRouter(tags=["pipeline"])


@router.post("/pipeline/run", dependencies=[Depends(require_api_key)])
async def run_pipeline(db: Session = Depends(get_db)):
    run = PipelineRun(
        run_date=date.today(), run_type="predict",
        status="running", started_at=datetime.now(),
    )
    db.add(run)
    db.commit()
    try:
        pipeline = DailyPipeline()
        pipeline.load_model()
        fixtures = []  # TODO: Replace with actual fixture collection
        predictions = pipeline.predict(fixtures)
        picks = pipeline.filter_picks(predictions)
        run.predictions_count = len(predictions)
        run.picks_count = len(picks)
        run.status = "success"
        run.completed_at = datetime.now()
        db.commit()
        return {"status": "ok", "data": {"run_id": run.id, "predictions": len(predictions), "picks": len(picks)}, "meta": {"generated_at": datetime.now().isoformat()}}
    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        run.completed_at = datetime.now()
        db.commit()
        logger.error(f"Pipeline failed: {e}")
        return {"status": "error", "data": {"error": str(e)}, "meta": {}}


@router.post("/pipeline/resolve", dependencies=[Depends(require_api_key)])
async def resolve_results(db: Session = Depends(get_db)):
    unresolved = db.query(DailyPick).filter(DailyPick.result.is_(None)).all()
    resolved_count = 0
    for pick in unresolved:
        pass  # TODO: Fetch actual scores
    return {"status": "ok", "data": {"resolved": resolved_count}, "meta": {"generated_at": datetime.now().isoformat()}}


@router.post("/pipeline/retrain", dependencies=[Depends(require_api_key)])
async def retrain_model():
    return {"status": "ok", "data": {"message": "Retrain not yet implemented"}, "meta": {}}


@router.get("/pipeline/status", dependencies=[Depends(require_api_key)])
async def pipeline_status(db: Session = Depends(get_db)):
    last_run = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).first()
    if not last_run:
        return {"status": "ok", "data": {"last_run": None}, "meta": {}}
    return {"status": "ok", "data": {"last_run": {
        "id": last_run.id, "run_date": str(last_run.run_date),
        "run_type": last_run.run_type, "status": last_run.status,
        "predictions_count": last_run.predictions_count,
        "picks_count": last_run.picks_count,
        "started_at": last_run.started_at.isoformat() if last_run.started_at else None,
        "completed_at": last_run.completed_at.isoformat() if last_run.completed_at else None,
    }}, "meta": {}}
