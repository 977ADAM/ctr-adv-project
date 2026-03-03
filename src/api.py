from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import CTRInferenceService

REQUIRED_ARTIFACT_FILES = (
    "best.pt",
    "meta.json",
    "scaler.joblib",
    "cat_encoder.joblib",
    "preprocessing_meta.json",
)


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(
        ...,
        description="Rows for CTR prediction.",
        min_length=1,
    )


class PredictResponse(BaseModel):
    probabilities: list[float]


def resolve_artifacts_dir() -> Path:
    env_path = os.getenv("MODEL_ARTIFACTS_DIR")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
        raise RuntimeError(
            f"MODEL_ARTIFACTS_DIR points to missing path: {p}"
        )

    base = Path("artifacts") / "click_model"
    if not base.exists():
        raise RuntimeError(
            "Artifacts base path does not exist: artifacts/click_model. "
            "Run training first or set MODEL_ARTIFACTS_DIR."
        )

    runs = sorted([p for p in base.iterdir() if p.is_dir()], reverse=True)
    for run in runs:
        if all((run / fname).exists() for fname in REQUIRED_ARTIFACT_FILES):
            return run

    raise RuntimeError(
        "No valid artifacts run found under artifacts/click_model. "
        "Run training first or set MODEL_ARTIFACTS_DIR."
    )


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.artifacts_dir = None
        app.state.service = None
        app.state.startup_error = None
        try:
            artifacts_dir = resolve_artifacts_dir()
            service = CTRInferenceService(artifacts_dir=artifacts_dir)
            app.state.artifacts_dir = artifacts_dir
            app.state.service = service
        except Exception as exc:  # noqa: BLE001
            app.state.startup_error = str(exc)
        yield

    app = FastAPI(title="CTR Demo API", version="1.0.0", lifespan=lifespan)

    def _get_service_or_503() -> CTRInferenceService:
        service = app.state.service
        if service is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model is not loaded. "
                    f"Startup error: {app.state.startup_error}"
                ),
            )
        return service

    @app.get("/health")
    def health() -> dict[str, Any]:
        service = app.state.service
        return {
            "status": "ok" if service is not None else "degraded",
            "artifacts_dir": (
                str(app.state.artifacts_dir)
                if app.state.artifacts_dir is not None
                else None
            ),
            "model_loaded": service is not None and service.model is not None,
            "startup_error": app.state.startup_error,
        }

    @app.get("/model-info")
    def model_info() -> dict[str, Any]:
        service = _get_service_or_503()
        meta = service.meta or {}
        return {
            "artifacts_dir": str(app.state.artifacts_dir),
            "numerical_cols": meta.get("numerical_cols", []),
            "categorical_cols": meta.get("categorical_cols", []),
            "cat_cardinalities": meta.get("cat_cardinalities", []),
            "best_val_auc": meta.get("best_val_auc"),
            "feature_schema_version": meta.get("feature_schema_version"),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        service = _get_service_or_503()
        try:
            df = pd.DataFrame(payload.rows)
            probs = service.predict_proba(df)
            return PredictResponse(probabilities=probs.astype(float).tolist())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
