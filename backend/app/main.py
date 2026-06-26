from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import get_settings
from app.database import close_db, init_db
from app.groq_service import analyze_prediction_with_groq, build_system_prompt, call_groq
from app.ml_pipeline import predict_patient, train_from_csv
from app.models import PredictionRecord, TrainingRun
from app.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluationStats,
    FeatureEngineeringStats,
    ModelSelectionStats,
    PatientInput,
    PipelineStatsResponse,
    PredictionAnalysisResponse,
    PredictionRecordResponse,
    PredictionResponse,
    PreprocessingStats,
    TrainingRunResponse,
)
from app.storage import get_storage


settings = get_settings()
app = FastAPI(title=settings.app_name)

ROOT_LOADER_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Heart Disease ML API</title>
    <style>
      :root {
        color-scheme: dark;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      * {
        box-sizing: border-box;
      }

      body {
        min-height: 100vh;
        margin: 0;
        display: grid;
        place-items: center;
        overflow: hidden;
        background:
          radial-gradient(circle at 20% 20%, rgba(59, 130, 246, 0.2), transparent 32rem),
          radial-gradient(circle at 80% 10%, rgba(16, 185, 129, 0.16), transparent 28rem),
          linear-gradient(135deg, #050507, #101116 48%, #07080b);
        color: #f8fafc;
      }

      body::before {
        content: "";
        position: fixed;
        inset: -40px;
        background:
          linear-gradient(rgba(255, 255, 255, 0.04) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255, 255, 255, 0.04) 1px, transparent 1px);
        background-size: 42px 42px;
        filter: blur(1px);
        mask-image: radial-gradient(circle, black, transparent 78%);
      }

      .shell {
        position: relative;
        width: min(92vw, 480px);
        padding: 1px;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(255,255,255,0.22), rgba(255,255,255,0.06));
        box-shadow: 0 28px 90px rgba(0, 0, 0, 0.48);
      }

      .card {
        border-radius: 27px;
        padding: 34px;
        background: rgba(10, 10, 14, 0.72);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
      }

      .badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid rgba(96, 165, 250, 0.28);
        border-radius: 999px;
        padding: 7px 11px;
        color: #bfdbfe;
        background: rgba(37, 99, 235, 0.12);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
      }

      .pulse {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #60a5fa;
        box-shadow: 0 0 0 0 rgba(96, 165, 250, 0.65);
        animation: pulse 1.4s infinite;
      }

      h1 {
        margin: 22px 0 10px;
        font-size: clamp(28px, 6vw, 42px);
        line-height: 1.04;
        letter-spacing: -0.03em;
      }

      p {
        margin: 0;
        color: #a1a1aa;
        font-size: 15px;
        line-height: 1.7;
      }

      .loader {
        margin-top: 26px;
        height: 10px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.08);
      }

      .loader span {
        display: block;
        width: 42%;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #60a5fa, #34d399);
        animation: slide 1.15s ease-in-out infinite;
      }

      .links {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 24px;
      }

      a {
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 10px 13px;
        color: #e5e7eb;
        text-decoration: none;
        font-size: 13px;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.06);
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
      }

      a:hover {
        transform: translateY(-1px);
        border-color: rgba(96, 165, 250, 0.45);
        background: rgba(96, 165, 250, 0.12);
      }

      @keyframes slide {
        0% { transform: translateX(-105%); }
        100% { transform: translateX(245%); }
      }

      @keyframes pulse {
        70% { box-shadow: 0 0 0 10px rgba(96, 165, 250, 0); }
        100% { box-shadow: 0 0 0 0 rgba(96, 165, 250, 0); }
      }
    </style>
  </head>
  <body>
    <main class="shell" aria-live="polite">
      <section class="card">
        <span class="badge"><span class="pulse"></span> API server</span>
        <h1>Taking a moment to activate the server</h1>
        <p>
          FastAPI is waking the ML service and database connection. Keep this tab open;
          the dashboard can connect as soon as the health check returns ok.
        </p>
        <div class="loader" aria-label="Loading"><span></span></div>
        <div class="links">
          <a href="/docs">Open API docs</a>
          <a href="http://localhost:5173/">Open dashboard</a>
        </div>
      </section>
    </main>
  </body>
</html>
"""

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup() -> None:
    await init_db()


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_db()


@app.get("/", response_class=HTMLResponse)
async def root_loader() -> HTMLResponse:
    return HTMLResponse(ROOT_LOADER_HTML)


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": settings.app_name,
        "docs": "/docs",
    }


@app.post("/api/train", response_model=TrainingRunResponse)
async def train_model(file: UploadFile = File(...)) -> TrainingRunResponse:
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    run = await TrainingRun.create(source_filename=file.filename, status="running")
    try:
        with NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await file.read())

        stats = await run_in_threadpool(train_from_csv, temp_path, run.id)
        await run.update_from_dict({"status": "completed", **stats}).save()
        return TrainingRunResponse(**await serialize_run(run))
    except Exception as exc:
        await run.update_from_dict({"status": "failed", "error_message": str(exc)}).save()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()


@app.get("/api/runs", response_model=list[TrainingRunResponse])
async def list_runs() -> list[TrainingRunResponse]:
    runs = await TrainingRun.all().limit(20)
    return [TrainingRunResponse(**await serialize_run(run)) for run in runs]


@app.get("/api/runs/latest", response_model=TrainingRunResponse)
async def latest_run() -> TrainingRunResponse:
    run = await TrainingRun.filter(status="completed").first()
    if not run:
        raise HTTPException(status_code=404, detail="No completed training run found.")
    return TrainingRunResponse(**await serialize_run(run))


@app.get("/api/runs/{run_id}", response_model=TrainingRunResponse)
async def get_run(run_id: str) -> TrainingRunResponse:
    run = await TrainingRun.get_or_none(id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return TrainingRunResponse(**await serialize_run(run))


@app.get("/api/runs/{run_id}/stats", response_model=PipelineStatsResponse)
async def get_pipeline_stats(run_id: str) -> PipelineStatsResponse:
    run = await TrainingRun.get_or_none(id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return PipelineStatsResponse(
        run_id=str(run.id),
        preprocessing=PreprocessingStats(**(run.preprocessing_stats or {})),
        feature_engineering=FeatureEngineeringStats(**(run.feature_engineering_stats or {})),
        model_selection=ModelSelectionStats(**(run.model_selection_stats or {})),
        evaluation=EvaluationStats(**(run.evaluation_stats or {})),
    )


@app.get("/api/runs/{run_id}/stats/preprocessing", response_model=PreprocessingStats)
async def get_preprocessing_stats(run_id: str) -> PreprocessingStats:
    run = await TrainingRun.get_or_none(id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return PreprocessingStats(**(run.preprocessing_stats or {}))


@app.get("/api/runs/{run_id}/stats/feature-engineering", response_model=FeatureEngineeringStats)
async def get_feature_engineering_stats(run_id: str) -> FeatureEngineeringStats:
    run = await TrainingRun.get_or_none(id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return FeatureEngineeringStats(**(run.feature_engineering_stats or {}))


@app.get("/api/runs/{run_id}/stats/model-selection", response_model=ModelSelectionStats)
async def get_model_selection_stats(run_id: str) -> ModelSelectionStats:
    run = await TrainingRun.get_or_none(id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return ModelSelectionStats(**(run.model_selection_stats or {}))


@app.get("/api/runs/{run_id}/stats/evaluation", response_model=EvaluationStats)
async def get_evaluation_stats(run_id: str) -> EvaluationStats:
    run = await TrainingRun.get_or_none(id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found.")
    return EvaluationStats(**(run.evaluation_stats or {}))


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(input_data: PatientInput) -> PredictionResponse:
    if input_data.run_id:
        run = await TrainingRun.get_or_none(id=input_data.run_id, status="completed")
    else:
        run = await TrainingRun.filter(status="completed").first()

    if not run:
        raise HTTPException(status_code=404, detail="Train a model before prediction.")

    storage = get_storage()
    artifacts = {
        "model": storage.download_to_cache(run.model_artifact_path),
        "scaler": storage.download_to_cache(run.scaler_artifact_path),
        "feature_columns": storage.download_to_cache(run.feature_columns_artifact_path),
        "threshold": storage.download_to_cache(run.threshold_artifact_path),
    }
    patient_payload = input_data.model_dump(exclude={"run_id"})
    result = await run_in_threadpool(
        predict_patient,
        patient_payload,
        artifacts,
    )
    record = await PredictionRecord.create(
        training_run=run,
        input_data=patient_payload,
        probability=result["probability"],
        threshold=result["threshold"],
        prediction=result["prediction"],
        label=result["label"],
    )
    return PredictionResponse(
        id=str(record.id),
        run_id=str(run.id),
        created_at=record.created_at.isoformat() if record.created_at else None,
        **result,
    )


@app.get("/api/predictions", response_model=list[PredictionRecordResponse])
async def list_predictions(
    run_id: str | None = Query(default=None),
    limit: int = Query(default=25, ge=1, le=100),
) -> list[PredictionRecordResponse]:
    query = PredictionRecord.all().select_related("training_run")
    if run_id:
        query = query.filter(training_run_id=run_id)
    records = await query.limit(limit)
    return [serialize_prediction(record) for record in records]


@app.get("/api/predictions/{prediction_id}", response_model=PredictionRecordResponse)
async def get_prediction(prediction_id: str) -> PredictionRecordResponse:
    record = await PredictionRecord.get_or_none(id=prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return serialize_prediction(record)


@app.delete("/api/predictions/{prediction_id}")
async def delete_prediction(prediction_id: str) -> dict:
    deleted = await PredictionRecord.filter(id=prediction_id).delete()
    if not deleted:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return {"deleted": True, "prediction_id": prediction_id}


@app.post("/api/predictions/{prediction_id}/analysis", response_model=PredictionAnalysisResponse)
async def analyze_prediction(prediction_id: str) -> PredictionAnalysisResponse:
    record = await PredictionRecord.get_or_none(id=prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    result = await analyze_prediction_with_groq(record)
    return PredictionAnalysisResponse(**result)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    record = await PredictionRecord.get_or_none(id=request.prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    system_prompt = build_system_prompt(record)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        {
            "role": message.role if message.role in {"user", "assistant"} else "user",
            "content": message.content,
        }
        for message in request.messages
    )
    reply = await call_groq(messages)
    return ChatResponse(
        prediction_id=request.prediction_id,
        system_prompt=system_prompt,
        reply=reply,
    )


async def serialize_run(run: TrainingRun) -> dict:
    return {
        "id": str(run.id),
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
        "source_filename": run.source_filename,
        "status": run.status,
        "error_message": run.error_message,
        "rows_before": run.rows_before,
        "rows_after": run.rows_after,
        "feature_count": run.feature_count,
        "champion_model": run.champion_model,
        "optimal_threshold": run.optimal_threshold,
        "preprocessing_stats": run.preprocessing_stats,
        "feature_engineering_stats": run.feature_engineering_stats,
        "model_selection_stats": run.model_selection_stats,
        "evaluation_stats": run.evaluation_stats,
        "model_artifact_path": run.model_artifact_path,
        "scaler_artifact_path": run.scaler_artifact_path,
        "feature_columns_artifact_path": run.feature_columns_artifact_path,
        "threshold_artifact_path": run.threshold_artifact_path,
    }


def serialize_prediction(record: PredictionRecord) -> PredictionRecordResponse:
    return PredictionRecordResponse(
        id=str(record.id),
        run_id=str(record.training_run_id),
        created_at=record.created_at.isoformat() if record.created_at else None,
        input_data=record.input_data,
        probability=record.probability,
        threshold=record.threshold,
        prediction=record.prediction,
        label=record.label,
    )
