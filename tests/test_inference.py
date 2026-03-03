from __future__ import annotations

import numpy as np
import pandas as pd
import torch
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
