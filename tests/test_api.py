from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from starlette.requests import Request

from src import api


class StubService:
    def __init__(self) -> None:
        self.model = object()
        self.meta: dict[str, list[str]] = {
            "numerical_cols": [
                "user_id",
                "age_level",
                "user_depth",
                "city_development_index",
                "var_1",
                "hour",
                "dayofweek",
            ],
            "categorical_cols": [
                "gender",
                "product",
                "campaign_id",
                "webpage_id",
                "user_group_id",
                "product_category_1",
                "product_category_2",
            ],
        }
        self.last_df = None

    def predict_proba(self, df):
        self.last_df = df.copy()
        return np.full(len(df), 0.42, dtype=np.float32)


def _valid_row() -> dict[str, Any]:
    return {
        "DateTime": "2017-07-08 00:00",
        "user_id": 732573,
        "gender": "Male",
        "product": "J",
        "campaign_id": 404347,
        "webpage_id": 53587,
        "user_group_id": 5,
        "product_category_1": 1,
        "product_category_2": None,
        "age_level": 5,
        "user_depth": 3,
        "city_development_index": None,
        "var_1": 0,
    }


def _endpoint(app, path: str) -> Callable[..., Any]:
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"Endpoint not found: {path}")


def _fake_request() -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "method": "POST",
            "path": "/predict",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }
    )


def _app_with_stub() -> tuple[Any, StubService]:
    app = api.create_app()
    stub = StubService()
    app.state.service = stub
    app.state.startup_error = None
    row_model = api._build_predict_row_model(stub.meta)
    app.state.predict_row_model = row_model
    app.state.required_predict_fields = api._build_required_predict_fields(stub.meta)
    app.state.api_key = None
    app.state.max_batch_size = 100
    app.state.rate_limit_rpm = 120
    app.state.rate_limit_window_sec = 60.0
    app.state.rate_limit_hits.clear()
    return app, stub


def test_predict_returns_422_when_datetime_invalid() -> None:
    row = _valid_row()
    row["DateTime"] = "not-a-date"

    with pytest.raises(ValidationError):
        api.PredictRequest.model_validate({"rows": [row]})


def test_predict_returns_422_when_required_field_missing_from_meta() -> None:
    app, _ = _app_with_stub()
    row = _valid_row()
    row.pop("age_level")

    predict = _endpoint(app, "/predict")
    with pytest.raises(HTTPException) as exc:
        predict(
            api.PredictRequest.model_validate({"rows": [row]}),
            request=_fake_request(),
        )

    assert exc.value.status_code == 422


def test_predict_accepts_null_in_non_datetime_core_fields() -> None:
    app, stub = _app_with_stub()
    row = _valid_row()
    row["user_group_id"] = None
    row["product_category_2"] = None

    predict = _endpoint(app, "/predict")
    resp = predict(
        api.PredictRequest.model_validate({"rows": [row]}),
        request=_fake_request(),
    )

    assert resp.probabilities == [0.41999998688697815]
    assert stub.last_df is not None


def test_predict_allows_extra_fields() -> None:
    app, stub = _app_with_stub()
    row = _valid_row()
    row["custom_feature_x"] = "value"

    predict = _endpoint(app, "/predict")
    predict(
        api.PredictRequest.model_validate({"rows": [row]}),
        request=_fake_request(),
    )

    assert stub.last_df is not None
    assert "custom_feature_x" in stub.last_df.columns


def test_predict_fails_whole_request_on_one_invalid_row() -> None:
    good = _valid_row()
    bad = _valid_row()
    bad["DateTime"] = "bad"

    with pytest.raises(ValidationError):
        api.PredictRequest.model_validate({"rows": [good, bad]})


def test_predict_returns_503_when_model_not_loaded() -> None:
    app = api.create_app()
    app.state.service = None
    app.state.predict_row_model = None
    app.state.startup_error = "artifacts missing"

    predict = _endpoint(app, "/predict")
    health = _endpoint(app, "/health")

    with pytest.raises(HTTPException) as exc:
        predict(
            api.PredictRequest.model_validate({"rows": [_valid_row()]}),
            request=_fake_request(),
        )

    assert exc.value.status_code == 503
    assert "Model is not loaded" in str(exc.value.detail)

    health_resp = health()
    assert health_resp["status"] == "degraded"


def test_model_info_exposes_required_predict_fields() -> None:
    app, _ = _app_with_stub()
    app.state.artifacts_dir = "artifacts/click_model/fake"

    model_info = _endpoint(app, "/model-info")
    resp = model_info()

    assert "required_predict_fields" in resp
    assert "DateTime" in resp["required_predict_fields"]
    assert "hour" not in resp["required_predict_fields"]
    assert "dayofweek" not in resp["required_predict_fields"]


def test_live_and_ready_endpoints() -> None:
    app, _ = _app_with_stub()

    live = _endpoint(app, "/live")
    ready = _endpoint(app, "/ready")

    assert live()["status"] == "alive"
    assert ready()["status"] == "ready"


def test_ready_returns_503_when_model_or_schema_missing() -> None:
    app = api.create_app()
    app.state.service = None
    app.state.startup_error = "not loaded"

    ready = _endpoint(app, "/ready")
    with pytest.raises(HTTPException) as exc:
        ready()
    assert exc.value.status_code == 503


def test_predict_enforces_max_batch_size() -> None:
    app, _ = _app_with_stub()
    app.state.max_batch_size = 1
    predict = _endpoint(app, "/predict")

    with pytest.raises(HTTPException) as exc:
        predict(
            api.PredictRequest.model_validate(
                {"rows": [_valid_row(), _valid_row()]}
            ),
            request=_fake_request(),
        )
    assert exc.value.status_code == 413


def test_predict_requires_api_key_when_configured() -> None:
    app, _ = _app_with_stub()
    app.state.api_key = "secret"
    predict = _endpoint(app, "/predict")

    with pytest.raises(HTTPException) as exc:
        predict(
            api.PredictRequest.model_validate({"rows": [_valid_row()]}),
            request=_fake_request(),
            x_api_key=None,
        )
    assert exc.value.status_code == 401

    resp = predict(
        api.PredictRequest.model_validate({"rows": [_valid_row()]}),
        request=_fake_request(),
        x_api_key="secret",
    )
    assert len(resp.probabilities) == 1
