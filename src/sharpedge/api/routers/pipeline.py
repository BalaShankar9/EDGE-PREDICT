"""Pipeline trigger endpoints (API-key protected)."""
import logging
from datetime import datetime, date

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from sharpedge.api.auth import require_api_key
from sharpedge.api.deps import get_db
from sharpedge.db.models import PipelineRun, Prediction, DailyPick
from sharpedge.pipeline.daily import DailyPipeline
from sharpedge.pipeline.enrichment import enrich_fixtures
from sharpedge.pipeline.resolver import resolve_pick
from sharpedge.collectors.football_data_org import FootballDataOrgCollector
from sharpedge.collectors.odds_api import OddsAPICollector
from sharpedge.collectors.forebet import ForebetCollector
from sharpedge.collectors.predictz import PredictZCollector
from sharpedge.collectors.windrawwin import WinDrawWinCollector
from sharpedge.collectors.footystats import FootyStatsCollector
from sharpedge.collectors.fpl import FPLCollector
from sharpedge.collectors.vitibet import VitibetCollector
from sharpedge.collectors.betstudy import BetStudyCollector
from sharpedge.collectors.transfermarkt import TransfermarktCollector

logger = logging.getLogger(__name__)
router = APIRouter(tags=["pipeline"])


def _load_historical(db: Session):
    """Load historical matches and ELO data for feature computation."""
    matches_df = pd.read_sql(
        text("""
        SELECT m.id, m.match_date, m.home_team_id, m.away_team_id,
            m.home_goals AS "FTHG", m.away_goals AS "FTAG",
            m.home_goals_ht AS "HTHG", m.away_goals_ht AS "HTAG",
            CASE WHEN m.home_goals > m.away_goals THEN 'H'
                 WHEN m.home_goals = m.away_goals THEN 'D' ELSE 'A' END AS "FTR",
            m.referee AS "Referee",
            ht.canonical_name AS home_team_name, at.canonical_name AS away_team_name,
            l.name AS league, s.label AS season,
            ms.home_shots AS "HS", ms.away_shots AS "AS",
            ms.home_shots_on_target AS "HST", ms.away_shots_on_target AS "AST",
            ms.home_fouls AS "HF", ms.away_fouls AS "AF",
            ms.home_corners AS "HC", ms.away_corners AS "AC",
            ms.home_yellow_cards AS "HY", ms.away_yellow_cards AS "AY",
            ms.home_red_cards AS "HR", ms.away_red_cards AS "AR",
            b365.odds_home AS "B365H", b365.odds_draw AS "B365D", b365.odds_away AS "B365A",
            ps.odds_home AS "PSH", ps.odds_draw AS "PSD", ps.odds_away AS "PSA",
            wh.odds_home AS "WHH", wh.odds_draw AS "WHD", wh.odds_away AS "WHA",
            mx.odds_home AS "MaxH", mx.odds_draw AS "MaxD", mx.odds_away AS "MaxA",
            av.odds_home AS "AvgH", av.odds_draw AS "AvgD", av.odds_away AS "AvgA"
        FROM matches m
        JOIN teams ht ON m.home_team_id = ht.id
        JOIN teams at ON m.away_team_id = at.id
        JOIN seasons s ON m.season_id = s.id
        JOIN leagues l ON s.league_id = l.id
        LEFT JOIN match_stats ms ON ms.match_id = m.id
        LEFT JOIN match_odds b365 ON b365.match_id = m.id AND b365.bookmaker = 'Bet365'
        LEFT JOIN match_odds ps ON ps.match_id = m.id AND ps.bookmaker = 'Pinnacle'
        LEFT JOIN match_odds wh ON wh.match_id = m.id AND wh.bookmaker = 'WilliamHill'
        LEFT JOIN match_odds mx ON mx.match_id = m.id AND mx.bookmaker = 'MarketMax'
        LEFT JOIN match_odds av ON av.match_id = m.id AND av.bookmaker = 'MarketAvg'
        WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
        ORDER BY m.match_date
    """),
        db.bind,
    )
    elo_df = pd.read_sql(
        text("SELECT team_id, rating_date AS date, elo, source FROM elo_ratings ORDER BY rating_date"),
        db.bind,
    )
    return matches_df, elo_df


@router.post("/pipeline/run", dependencies=[Depends(require_api_key)])
async def run_pipeline(db: Session = Depends(get_db)):
    run = PipelineRun(
        run_date=date.today(), run_type="predict",
        status="running", started_at=datetime.now(),
    )
    db.add(run)
    db.commit()
    try:
        # 1. Collect scheduled fixtures
        collector = FootballDataOrgCollector()
        fixtures_df = collector.collect(data_type="fixtures")

        if fixtures_df.empty:
            run.fixtures_count = 0
            run.predictions_count = 0
            run.picks_count = 0
            run.status = "success"
            run.completed_at = datetime.now()
            db.commit()
            return {
                "status": "ok",
                "data": {"run_id": run.id, "predictions": 0, "picks": 0, "message": "No fixtures found"},
                "meta": {"generated_at": datetime.now().isoformat()},
            }

        # Convert fixtures DataFrame to list of dicts for predict()
        fixtures = fixtures_df.to_dict(orient="records")
        run.fixtures_count = len(fixtures)

        # 2. Collect bookmaker odds
        odds_df = None
        try:
            odds_collector = OddsAPICollector()
            odds_df = odds_collector.collect()
            logger.info(f"Collected {len(odds_df)} odds records")
        except Exception as e:
            logger.warning(f"Odds collection failed (proceeding without): {e}")

        # 3. Collect competitor predictions
        predictions_dfs = {}
        for name, cls in [
            ("forebet", ForebetCollector),
            ("predictz", PredictZCollector),
            ("windrawwin", WinDrawWinCollector),
            ("footystats", FootyStatsCollector),
            ("vitibet", VitibetCollector),
            ("betstudy", BetStudyCollector),
        ]:
            try:
                df = cls().collect()
                if not df.empty:
                    predictions_dfs[name] = df
                    logger.info(f"Collected {len(df)} {name} predictions")
            except Exception as e:
                logger.warning(f"{name} collection failed (proceeding without): {e}")

        # 4. Collect team intel (FPL — EPL injuries/form + Transfermarkt — all leagues)
        intel_frames = []
        injuries_df = None
        try:
            fpl_collector = FPLCollector()
            fpl_df = fpl_collector.collect()
            if not fpl_df.empty:
                intel_frames.append(fpl_df)
                logger.info(f"Collected FPL intel for {len(fpl_df)} teams")
        except Exception as e:
            logger.warning(f"FPL collection failed (proceeding without): {e}")

        try:
            tm_collector = TransfermarktCollector()
            tm_df = tm_collector.collect()
            if not tm_df.empty:
                intel_frames.append(tm_df)
                injuries_df = tm_df  # Also pass to InjuryFeatures
                logger.info(f"Collected Transfermarkt injuries for {len(tm_df)} teams")
        except Exception as e:
            logger.warning(f"Transfermarkt collection failed (proceeding without): {e}")

        team_intel_df = (
            pd.concat(intel_frames, ignore_index=True) if intel_frames else None
        )

        # 5. Enrich fixtures with odds + predictions + team intel
        fixtures = enrich_fixtures(
            fixtures,
            odds_df=odds_df,
            predictions_dfs=predictions_dfs if predictions_dfs else None,
            team_intel_df=team_intel_df,
        )

        # 5. Load historical data for rolling features
        matches_df, elo_df = _load_historical(db)

        # 5b. Build unified predictions_df for MetaPredictionFeatures
        all_pred_frames = []
        for source_name, pred_df in (predictions_dfs or {}).items():
            if pred_df is not None and not pred_df.empty:
                df_copy = pred_df.copy()
                if "source" not in df_copy.columns:
                    df_copy["source"] = source_name
                all_pred_frames.append(df_copy)
        combined_predictions_df = (
            pd.concat(all_pred_frames, ignore_index=True) if all_pred_frames else None
        )

        # 6. Run predictions
        pipeline = DailyPipeline()
        pipeline.load_model()
        predictions = pipeline.predict(
            fixtures,
            historical_matches=matches_df if not matches_df.empty else None,
            elo_df=elo_df if not elo_df.empty else None,
            predictions_df=combined_predictions_df,
            injuries_df=injuries_df,
        )

        # 7. Filter picks
        picks = pipeline.filter_picks(predictions)

        # 8. Save predictions to DB
        prediction_map = {}  # index -> Prediction record
        for i, pred in enumerate(predictions):
            match_date_str = pred.get("match_date", str(date.today()))
            if isinstance(match_date_str, str) and "T" in match_date_str:
                match_date_str = match_date_str.split("T")[0]
            try:
                match_dt = pd.to_datetime(match_date_str).date()
            except Exception:
                match_dt = date.today()

            # Use original fixture name as fallback if normalized ID is None
            fixture = fixtures[i] if i < len(fixtures) else {}
            home_team = pred.get("home_team") or fixture.get("home_team", "unknown")
            away_team = pred.get("away_team") or fixture.get("away_team", "unknown")

            prediction_record = Prediction(
                pipeline_run_id=run.id,
                match_date=match_dt,
                home_team=home_team,
                away_team=away_team,
                league=pred.get("league", "") or fixture.get("league", ""),
                prob_home=pred.get("prob_home"),
                prob_draw=pred.get("prob_draw"),
                prob_away=pred.get("prob_away"),
                prob_over=pred.get("prob_over"),
                prob_under=pred.get("prob_under"),
                prob_btts_yes=pred.get("prob_btts_yes"),
                prob_btts_no=pred.get("prob_btts_no"),
                created_at=datetime.now(),
            )
            db.add(prediction_record)
            db.flush()  # Get the ID
            prediction_map[f"{home_team}_{away_team}"] = prediction_record

        # 9. Save picks to DB
        for pick in picks:
            pick_key = f"{pick.home_team}_{pick.away_team}"
            prediction_record = prediction_map.get(pick_key)
            if not prediction_record:
                continue

            match_date_str = pick.match_date
            if isinstance(match_date_str, str) and "T" in match_date_str:
                match_date_str = match_date_str.split("T")[0]
            try:
                match_dt = pd.to_datetime(match_date_str).date()
            except Exception:
                match_dt = date.today()

            daily_pick = DailyPick(
                pipeline_run_id=run.id,
                prediction_id=prediction_record.id,
                match_date=match_dt,
                home_team=pick.home_team,
                away_team=pick.away_team,
                league=pick.league,
                pick_market=pick.market,
                pick_selection=pick.market.split("_")[-1],  # "home", "draw", "away", etc.
                model_prob=pick.model_prob,
                model_spread=pick.model_spread,
                best_odds=pick.best_odds,
                bookmaker=pick.bookmaker,
                implied_prob=pick.implied_prob,
                edge=pick.edge,
                tier=pick.tier,
                meta_agreement=pick.meta_agreement,
                risk_flags=pick.risk_flags,
            )
            db.add(daily_pick)

        run.predictions_count = len(predictions)
        run.picks_count = len(picks)
        run.status = "success"
        run.completed_at = datetime.now()
        db.commit()

        return {
            "status": "ok",
            "data": {
                "run_id": run.id,
                "fixtures": len(fixtures),
                "predictions": len(predictions),
                "picks": len(picks),
            },
            "meta": {"generated_at": datetime.now().isoformat()},
        }
    except Exception as e:
        db.rollback()
        run.status = "failed"
        run.error_message = str(e)[:500]
        run.completed_at = datetime.now()
        db.add(run)
        db.commit()
        logger.error(f"Pipeline failed: {e}")
        return {"status": "error", "data": {"error": str(e)[:200]}, "meta": {}}


@router.post("/pipeline/resolve", dependencies=[Depends(require_api_key)])
async def resolve_results(db: Session = Depends(get_db)):
    """Resolve unresolved daily picks against actual match results."""
    unresolved = db.query(DailyPick).filter(DailyPick.result.is_(None)).all()
    resolved_count = 0

    for pick in unresolved:
        # Look up actual result: match by team names + date where goals are not null
        result_row = db.execute(
            text("""
                SELECT m.home_goals, m.away_goals
                FROM matches m
                JOIN teams ht ON m.home_team_id = ht.id
                JOIN teams at ON m.away_team_id = at.id
                WHERE ht.canonical_name = :home_team
                  AND at.canonical_name = :away_team
                  AND m.match_date = :match_date
                  AND m.home_goals IS NOT NULL
                LIMIT 1
            """),
            {"home_team": pick.home_team, "away_team": pick.away_team, "match_date": pick.match_date},
        ).fetchone()

        if result_row is None:
            continue

        home_goals, away_goals = result_row[0], result_row[1]
        resolution = resolve_pick(pick.pick_market, home_goals, away_goals, pick.best_odds)

        pick.result = resolution["result"]
        pick.profit_loss = resolution["profit_loss"]
        pick.resolved_at = datetime.now()
        resolved_count += 1

    db.commit()
    return {
        "status": "ok",
        "data": {"resolved": resolved_count, "remaining": len(unresolved) - resolved_count},
        "meta": {"generated_at": datetime.now().isoformat()},
    }


@router.post("/pipeline/retrain", dependencies=[Depends(require_api_key)])
async def retrain_model(db: Session = Depends(get_db)):
    """Retrain the model on all historical data and save to models/latest.pkl."""
    from sharpedge.ml.training.trainer import ModelTrainer

    run = PipelineRun(
        run_date=date.today(), run_type="retrain",
        status="running", started_at=datetime.now(),
    )
    db.add(run)
    db.commit()

    try:
        # Load training data
        matches_df, elo_df = _load_historical(db)

        if len(matches_df) < 100:
            raise ValueError(f"Not enough matches to train ({len(matches_df)}). Run backfill first.")

        # Train
        trainer = ModelTrainer(n_splits=2, min_train_seasons=3)
        result = trainer.train(
            matches_df,
            elo_df=elo_df if not elo_df.empty else None,
        )

        # Save
        save_path = "models/latest.pkl"
        trainer.save(save_path)

        run.status = "success"
        run.completed_at = datetime.now()
        db.commit()

        return {
            "status": "ok",
            "data": {
                "run_id": run.id,
                "matches_trained": len(matches_df),
                "mean_rps": result.aggregate_metrics.get("mean_rps"),
                "mean_accuracy": result.aggregate_metrics.get("mean_accuracy"),
                "features": len(result.feature_names),
                "model_path": save_path,
            },
            "meta": {"generated_at": datetime.now().isoformat()},
        }
    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        run.completed_at = datetime.now()
        db.commit()
        logger.error(f"Retrain failed: {e}")
        return {"status": "error", "data": {"error": str(e)}, "meta": {}}


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
