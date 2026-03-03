from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import pytest
from typing import Any, cast

from src.inference import CTRInferenceService


class DummyPreprocessor:
    def __init__(self) -> None:
        self.called = False

    def transform(self, df: pd.DataFrame):
        self.called = True
        n = len(df)
        x_num = np.zeros((n, 2), dtype=np.float32)
        x_cat = np.ones((n, 2), dtype=np.int64)
        return x_num, x_cat


class DummyModel(torch.nn.Module):
    def forward(self, x):
        x_num, _ = x
        return torch.zeros((x_num.shape[0], 1), dtype=torch.float32)


def test_predict_proba_returns_empty_array_for_empty_input() -> None:
    svc = CTRInferenceService.__new__(CTRInferenceService)
    svc_any = cast(Any, svc)
    svc_any.preprocessor = DummyPreprocessor()
    svc_any.model = DummyModel()
    svc_any.device = torch.device("cpu")
    svc_any.batch_size = 8
    svc_any.num_workers = 0

    probs = svc.predict_proba(pd.DataFrame())

    assert probs.shape == (0,)
    assert probs.dtype == np.float32
    assert cast(Any, svc.preprocessor).called is False


def test_predict_proba_returns_probabilities_for_non_empty_input() -> None:
    svc = CTRInferenceService.__new__(CTRInferenceService)
    svc_any = cast(Any, svc)
    svc_any.preprocessor = DummyPreprocessor()
    svc_any.model = DummyModel()
    svc_any.device = torch.device("cpu")
    svc_any.batch_size = 2
    svc_any.num_workers = 0

    probs = svc.predict_proba(pd.DataFrame({"x": [1, 2, 3]}))

    assert probs.shape == (3,)
    assert np.allclose(probs, 0.5)
    assert cast(Any, svc.preprocessor).called is True


def test_predict_proba_raises_when_model_not_loaded() -> None:
    svc = CTRInferenceService.__new__(CTRInferenceService)
    svc_any = cast(Any, svc)
    svc_any.preprocessor = DummyPreprocessor()
    svc_any.model = None
    svc_any.device = torch.device("cpu")
    svc_any.batch_size = 2
    svc_any.num_workers = 0

    with pytest.raises(RuntimeError, match="Model is not loaded"):
        svc.predict_proba(pd.DataFrame({"x": [1]}))


def test_predict_to_csv_writes_click_proba_column(tmp_path) -> None:
    svc = CTRInferenceService.__new__(CTRInferenceService)
    svc_any = cast(Any, svc)
    svc_any.preprocessor = DummyPreprocessor()
    svc_any.model = DummyModel()
    svc_any.device = torch.device("cpu")
    svc_any.batch_size = 2
    svc_any.num_workers = 0

    output_path = tmp_path / "preds.csv"
    svc.predict_to_csv(pd.DataFrame({"x": [1, 2]}), output_path)

    out = pd.read_csv(output_path)
    assert list(out.columns) == ["click_proba"]
    assert len(out) == 2
    assert np.allclose(out["click_proba"].to_numpy(), 0.5)
