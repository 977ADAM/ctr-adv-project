from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator

from .inference import CTRInferenceService

REQUIRED_ARTIFACT_FILES = (
    "best.pt",
    "meta.json",
    "scaler.joblib",
    "cat_encoder.joblib",
    "preprocessing_meta.json",
)


class PredictResponse(BaseModel):
    probabilities: list[float]


class _DynamicPredictRowBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    @field_validator("DateTime", check_fields=False)
    @classmethod
    def validate_datetime(cls, value: str) -> str:
        if value is None or value.strip() == "":
            raise ValueError("DateTime must be a valid datetime string")
        try:
            pd.to_datetime(value, errors="raise")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("DateTime must be a valid datetime string") from exc
        return value


DERIVED_COLUMNS = {"hour", "dayofweek"}


def _build_required_predict_fields(meta: dict[str, Any]) -> list[str]:
    numerical_cols = meta.get("numerical_cols")
    categorical_cols = meta.get("categorical_cols")
    if not isinstance(numerical_cols, list) or not isinstance(categorical_cols, list):
        raise RuntimeError("meta.json does not contain valid numerical/categorical columns")

    fields = ["DateTime"]
    for col in numerical_cols + categorical_cols:
        if col in DERIVED_COLUMNS:
            continue
        if col not in fields:
            fields.append(col)
    return fields


def _build_predict_row_model(meta: dict[str, Any]) -> type[BaseModel]:
    categorical_cols = meta.get("categorical_cols", [])
    categorical_set = set(categorical_cols)

    required_fields = _build_required_predict_fields(meta)
    model_fields: dict[str, tuple[Any, Any]] = {"DateTime": (str, ...)}

    for col in required_fields:
        if col == "DateTime":
            continue
        if col in categorical_set:
            model_fields[col] = (str | int | float | bool | None, ...)
        else:
            model_fields[col] = (float | int | None, ...)

    return cast(
        type[BaseModel],
        create_model(  # type: ignore[call-overload]
            "PredictRowDynamic",
            __base__=_DynamicPredictRowBase,
            **model_fields,
        ),
    )


def _build_predict_request_model() -> type[BaseModel]:
    return cast(
        type[BaseModel],
        create_model(
        "PredictRequestDynamic",
        rows=(list[dict[str, Any]], Field(..., min_length=1)),
        ),
    )


def _rows_to_dataframe(rows: list[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump() for row in rows])


def _web_ui_path() -> Path:
    return Path(__file__).resolve().parent / "web" / "index.html"


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
        app.state.predict_request_model = None
        app.state.predict_row_model = None
        app.state.required_predict_fields = []
        app.state.startup_error = None
        try:
            artifacts_dir = resolve_artifacts_dir()
            service = CTRInferenceService(artifacts_dir=artifacts_dir)
            predict_row_model = _build_predict_row_model(service.meta or {})
            app.state.predict_request_model = _build_predict_request_model()
            app.state.predict_row_model = predict_row_model
            app.state.required_predict_fields = _build_required_predict_fields(
                service.meta or {}
            )
            app.state.artifacts_dir = artifacts_dir
            app.state.service = service
        except Exception as exc:  # noqa: BLE001
            app.state.startup_error = str(exc)
        yield

    app = FastAPI(title="CTR Demo API", version="1.0.0", lifespan=lifespan)
    app.state.artifacts_dir = None
    app.state.service = None
    app.state.predict_request_model = None
    app.state.predict_row_model = None
    app.state.required_predict_fields = []
    app.state.startup_error = None

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

    def _get_predict_request_model_or_503() -> type[BaseModel]:
        model = app.state.predict_request_model
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Predict schema is not loaded. "
                    f"Startup error: {app.state.startup_error}"
                ),
            )
        return model

    def _get_predict_row_model_or_503() -> type[BaseModel]:
        model = app.state.predict_row_model
        if model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Predict row schema is not loaded. "
                    f"Startup error: {app.state.startup_error}"
                ),
            )
        return model

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
            "schema_loaded": app.state.predict_request_model is not None,
            "startup_error": app.state.startup_error,
        }

    def _validate_and_predict(payload: dict[str, Any]) -> PredictResponse:
        service = _get_service_or_503()
        request_model = _get_predict_request_model_or_503()
        row_model = _get_predict_row_model_or_503()

        validated_payload = request_model.model_validate(payload)
        validated_rows: list[BaseModel] = []
        row_errors: list[dict[str, Any]] = []

        payload_rows = cast(Any, validated_payload).rows
        for idx, row in enumerate(payload_rows):
            try:
                validated_rows.append(row_model.model_validate(row))
            except ValidationError as exc:
                for err in exc.errors():
                    loc = tuple(err.get("loc", ()))
                    row_errors.append(
                        {
                            **err,
                            "loc": ("rows", idx, *loc),
                        }
                    )

        if row_errors:
            raise HTTPException(status_code=422, detail=row_errors)

        df = _rows_to_dataframe(validated_rows)
        probs = service.predict_proba(df)
        return PredictResponse(probabilities=probs.astype(float).tolist())

    @app.get("/")
    def web_ui() -> FileResponse:
        ui = _web_ui_path()
        if not ui.exists():
            raise HTTPException(status_code=404, detail="Web UI file not found")
        return FileResponse(ui)

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
            "required_predict_fields": app.state.required_predict_fields,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: dict[str, Any]) -> PredictResponse:
        try:
            return _validate_and_predict(payload)
        except HTTPException:
            raise
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                msg_type = message.get("type")

                if msg_type == "health":
                    await websocket.send_json(
                        {"type": "health", "data": health()}
                    )
                    continue

                if msg_type == "model_info":
                    try:
                        data = model_info()
                        await websocket.send_json(
                            {"type": "model_info", "data": data}
                        )
                    except HTTPException as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "status": exc.status_code,
                                "detail": exc.detail,
                            }
                        )
                    continue

                if msg_type == "predict":
                    payload = message.get("payload")
                    if not isinstance(payload, dict):
                        await websocket.send_json(
                            {
                                "type": "error",
                                "status": 400,
                                "detail": "predict message must include object payload",
                            }
                        )
                        continue
                    try:
                        result = _validate_and_predict(payload)
                        await websocket.send_json(
                            {
                                "type": "predict_result",
                                "data": result.model_dump(),
                            }
                        )
                    except HTTPException as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "status": exc.status_code,
                                "detail": exc.detail,
                            }
                        )
                    except ValidationError as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "status": 422,
                                "detail": exc.errors(),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        await websocket.send_json(
                            {
                                "type": "error",
                                "status": 500,
                                "detail": str(exc),
                            }
                        )
                    continue

                await websocket.send_json(
                    {
                        "type": "error",
                        "status": 400,
                        "detail": "Unknown message type",
                    }
                )
        except WebSocketDisconnect:
            return

    return app


app = create_app()
