from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src import api


class StubService:
    def __init__(self) -> None:
        self.model = object()
        self.meta: dict[str, list[str]] = {
            "numerical_cols": [],
            "categorical_cols": [],
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
    }


def _endpoint(app, path: str) -> Callable[..., Any]:
    for route in app.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"Endpoint not found: {path}")


def test_predict_returns_422_when_datetime_invalid() -> None:
    row = _valid_row()
    row["DateTime"] = "not-a-date"

    with pytest.raises(ValidationError):
        api.PredictRequest.model_validate({"rows": [row]})


def test_predict_returns_422_when_required_core_field_missing() -> None:
    row = _valid_row()
    row.pop("campaign_id")

    with pytest.raises(ValidationError):
        api.PredictRequest.model_validate({"rows": [row]})


def test_predict_accepts_null_in_non_datetime_core_fields() -> None:
    app = api.create_app()
    stub = StubService()
    app.state.service = stub
    app.state.startup_error = None

    row = _valid_row()
    row["user_group_id"] = None
    row["product_category_2"] = None
    payload = api.PredictRequest.model_validate({"rows": [row]})

    predict = _endpoint(app, "/predict")
    resp = predict(payload)

    assert resp.probabilities == [0.41999998688697815]
    assert stub.last_df is not None


def test_predict_allows_extra_fields() -> None:
    app = api.create_app()
    stub = StubService()
    app.state.service = stub
    app.state.startup_error = None

    row = _valid_row()
    row["custom_feature_x"] = "value"
    payload = api.PredictRequest.model_validate({"rows": [row]})

    predict = _endpoint(app, "/predict")
    predict(payload)

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
    app.state.startup_error = "artifacts missing"

    predict = _endpoint(app, "/predict")
    health = _endpoint(app, "/health")
    payload = api.PredictRequest.model_validate({"rows": [_valid_row()]})

    with pytest.raises(HTTPException) as exc:
        predict(payload)

    assert exc.value.status_code == 503
    assert "Model is not loaded" in str(exc.value.detail)

    health_resp = health()
    assert health_resp["status"] == "degraded"
