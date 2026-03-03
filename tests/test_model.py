from __future__ import annotations

import numpy as np
import torch

from src.config import Config
from src.model import ClickDataset, ClickModel, EarlyStopping


def test_click_dataset_shapes() -> None:
    x_num = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    x_cat = np.array([[1, 2], [2, 1]], dtype=np.int64)
    y = np.array([0.0, 1.0], dtype=np.float32)

    ds = ClickDataset(x_num, x_cat, y)

    assert len(ds) == 2
    xn, xc, yt = ds[0]
    assert xn.shape == (2,)
    assert xc.shape == (2,)
    assert yt.shape == (1,)


def test_click_model_forward_shape() -> None:
    cfg = Config(hidden_dims=(8,), dropout=0.0, emb_max_dim=8)
    model = ClickModel(num_features=2, cat_cardinalities=[4, 5], config=cfg)

    x_num = torch.randn(3, 2)
    x_cat = torch.tensor([[1, 2], [2, 3], [3, 1]], dtype=torch.long)

    out = model((x_num, x_cat))
    assert out.shape == (3, 1)


def test_early_stopping_behavior() -> None:
    stopper = EarlyStopping(patience=2, mode="max")

    assert stopper.step(0.5) is False
    assert stopper.step(0.4) is False
    assert stopper.step(0.3) is True
