from __future__ import annotations

import os
import time
import uuid
import logging
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, field_validator

from .inference import CTRInferenceService

REQUIRED_ARTIFACT_FILES = (
    "best.pt",
    "meta.json",
    "scaler.joblib",
    "cat_encoder.joblib",
    "preprocessing_meta.json",
)
logger = logging.getLogger("ctr_api")


class PredictResponse(BaseModel):
    probabilities: list[float]


class PredictRowIn(BaseModel):
    model_config = ConfigDict(extra="allow")

    DateTime: str
    user_id: int | None = None
    gender: str | None = None
    product: str | None = None
    campaign_id: int | None = None
    webpage_id: int | None = None
    user_group_id: float | int | None = None
    product_category_1: float | int | None = None
    product_category_2: float | int | None = None
    age_level: float | int | None = None
    user_depth: float | int | None = None
    city_development_index: float | int | None = None
    var_1: float | int | None = None

    @field_validator("DateTime")
    @classmethod
    def validate_datetime(cls, value: str) -> str:
        if value is None or value.strip() == "":
            raise ValueError("DateTime must be a valid datetime string")
        try:
            pd.to_datetime(value, errors="raise")
        except Exception as exc:  # noqa: BLE001
            raise ValueError("DateTime must be a valid datetime string") from exc
        return value


class PredictRequest(BaseModel):
    rows: list[PredictRowIn] = Field(
        ...,
        description="Rows for CTR prediction",
        min_length=1,
    )


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

def _rows_to_dataframe(rows: list[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([row.model_dump() for row in rows])


def _web_ui_path() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "dist" / "index.html"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


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
        app.state.predict_row_model = None
        app.state.required_predict_fields = []
        app.state.startup_error = None
        app.state.api_key = os.getenv("API_KEY")
        app.state.max_batch_size = _env_int("MAX_BATCH_SIZE", 100)
        app.state.rate_limit_rpm = _env_int("RATE_LIMIT_RPM", 120)
        app.state.rate_limit_window_sec = 60.0
        app.state.rate_limit_hits = defaultdict(deque)
        try:
            artifacts_dir = resolve_artifacts_dir()
            service = CTRInferenceService(artifacts_dir=artifacts_dir)
            predict_row_model = _build_predict_row_model(service.meta or {})
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
    app.state.predict_row_model = None
    app.state.required_predict_fields = []
    app.state.startup_error = None
    app.state.api_key = os.getenv("API_KEY")
    app.state.max_batch_size = _env_int("MAX_BATCH_SIZE", 100)
    app.state.rate_limit_rpm = _env_int("RATE_LIMIT_RPM", 120)
    app.state.rate_limit_window_sec = 60.0
    app.state.rate_limit_hits = defaultdict(deque)
    web_dist_dir = Path(__file__).resolve().parents[1] / "web" / "dist"
    app.mount(
        "/assets",
        StaticFiles(directory=web_dist_dir / "assets", check_dir=False),
        name="web-assets",
    )

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

    def _enforce_api_key_or_401(api_key: str | None) -> None:
        expected = app.state.api_key
        if not expected:
            return
        if api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def _enforce_rate_limit_or_429(client_id: str) -> None:
        hits = app.state.rate_limit_hits[client_id]
        now = time.monotonic()
        window = float(app.state.rate_limit_window_sec)
        limit = int(app.state.rate_limit_rpm)

        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        hits.append(now)

    @app.get("/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        model_ready = app.state.service is not None
        schema_ready = app.state.predict_row_model is not None
        if not (model_ready and schema_ready):
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "model_loaded": model_ready,
                    "schema_loaded": schema_ready,
                    "startup_error": app.state.startup_error,
                },
            )
        return {
            "status": "ready",
            "model_loaded": True,
            "schema_loaded": True,
        }

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
            "schema_loaded": app.state.predict_row_model is not None,
            "startup_error": app.state.startup_error,
        }

    def _validate_and_predict(payload: PredictRequest) -> PredictResponse:
        service = _get_service_or_503()
        row_model = _get_predict_row_model_or_503()

        payload_rows = payload.rows
        if len(payload_rows) > int(app.state.max_batch_size):
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Batch size {len(payload_rows)} exceeds max_batch_size "
                    f"{app.state.max_batch_size}"
                ),
            )

        validated_rows: list[BaseModel] = []
        row_errors: list[dict[str, Any]] = []

        for idx, row in enumerate(payload_rows):
            try:
                validated_rows.append(
                    row_model.model_validate(row.model_dump(exclude_unset=True))
                )
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
    def model_info(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        _enforce_api_key_or_401(x_api_key)
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
    def predict(
        payload: PredictRequest,
        request: Request,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> PredictResponse:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        started = time.perf_counter()
        _enforce_api_key_or_401(x_api_key)
        client_host = request.client.host if request.client is not None else "unknown"
        _enforce_rate_limit_or_429(f"http:{client_host}")
        try:
            result = _validate_and_predict(payload)
            latency_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "predict_ok request_id=%s client=%s latency_ms=%.2f batch_size=%d",
                request_id,
                client_host,
                latency_ms,
                len(payload.rows),
            )
            return result
        except HTTPException:
            latency_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "predict_error request_id=%s client=%s latency_ms=%.2f",
                request_id,
                client_host,
                latency_ms,
            )
            raise
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.websocket("/ws")
    async def websocket_handler(websocket: WebSocket) -> None:
        client_host = websocket.client.host if websocket.client else "unknown"
        request_id_prefix = str(uuid.uuid4())[:8]
        try:
            _enforce_api_key_or_401(websocket.query_params.get("api_key"))
        except HTTPException:
            await websocket.close(code=1008)
            return

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
                        started = time.perf_counter()
                        request_id = (
                            message.get("request_id")
                            or f"{request_id_prefix}-{uuid.uuid4().hex[:8]}"
                        )
                        _enforce_rate_limit_or_429(f"ws:{client_host}")
                        result = _validate_and_predict(
                            PredictRequest.model_validate(payload)
                        )
                        latency_ms = (time.perf_counter() - started) * 1000
                        logger.info(
                            "ws_predict_ok request_id=%s client=%s latency_ms=%.2f",
                            request_id,
                            client_host,
                            latency_ms,
                        )
                        await websocket.send_json(
                            {
                                "type": "predict_result",
                                "request_id": request_id,
                                "data": result.model_dump(),
                            }
                        )
                    except HTTPException as exc:
                        latency_ms = (time.perf_counter() - started) * 1000
                        logger.info(
                            "ws_predict_error request_id=%s client=%s latency_ms=%.2f status=%d",
                            request_id,
                            client_host,
                            latency_ms,
                            exc.status_code,
                        )
                        await websocket.send_json(
                            {
                                "type": "error",
                                "request_id": request_id,
                                "status": exc.status_code,
                                "detail": exc.detail,
                            }
                        )
                    except ValidationError as exc:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "request_id": request_id,
                                "status": 422,
                                "detail": exc.errors(),
                            }
                        )
                    except Exception as exc:  # noqa: BLE001
                        await websocket.send_json(
                            {
                                "type": "error",
                                "request_id": request_id,
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
