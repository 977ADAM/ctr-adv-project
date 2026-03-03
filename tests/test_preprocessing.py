from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import CTRPreprocessor


def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "DateTime": ["2024-01-01 10:00:00", "2024-01-02 11:00:00", "2024-01-03 12:00:00", "2024-01-04 13:00:00"],
            "user_id": [1, 2, 3, 4],
            "gender": ["M", "F", None, "M"],
            "product": ["A", "B", "A", None],
            "age": [20.0, 30.0, np.nan, 40.0],
            "score": [0.1, 0.2, 0.3, np.nan],
            "is_click": [0, 1, 0, 1],
        }
    )


def test_fit_transform_produces_numeric_and_categorical_arrays() -> None:
    df = _make_df()
    p = CTRPreprocessor()

    x_num, x_cat = p.fit_transform(df)

    assert x_num.shape[0] == len(df)
    assert x_cat.shape == (len(df), 2)
    assert x_num.dtype == np.float64
    assert x_cat.dtype == np.int64


def test_transform_handles_unknown_categories_as_zero_or_positive() -> None:
    train_df = _make_df()
    test_df = train_df.copy()
    test_df.loc[0, "gender"] = "UNKNOWN_GENDER"
    test_df.loc[1, "product"] = "UNKNOWN_PRODUCT"

    p = CTRPreprocessor().fit(train_df)
    _, x_cat = p.transform(test_df)

    assert (x_cat >= 0).all()


def test_save_and_load_roundtrip(tmp_path) -> None:
    df = _make_df()

    p1 = CTRPreprocessor().fit(df)
    p1.save(tmp_path)

    p2 = CTRPreprocessor()
    p2.load(tmp_path)

    x_num_1, x_cat_1 = p1.transform(df)
    x_num_2, x_cat_2 = p2.transform(df)

    assert np.allclose(x_num_1, x_num_2)
    assert np.array_equal(x_cat_1, x_cat_2)


def test_make_splits_prevents_time_and_user_leakage() -> None:
    df = pd.DataFrame(
        {
            "DateTime": [
                "2024-01-01 10:00:00",
                "2024-01-02 10:00:00",
                "2024-01-03 10:00:00",
                "2024-01-04 10:00:00",
                "2024-01-05 10:00:00",
                "2024-01-06 10:00:00",
            ],
            "user_id": [1, 2, 3, 1, 4, 5],
            "gender": ["M", "F", "F", "M", "F", "M"],
            "product": ["A", "A", "B", "B", "A", "B"],
            "age": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "is_click": [0, 1, 0, 1, 0, 1],
        }
    )

    p = CTRPreprocessor()
    x_train, x_val, _, _ = p.make_splits(df, test_size=0.33, seed=42)

    train_users = set(x_train["user_id"].tolist())
    val_users = set(x_val["user_id"].tolist())
    assert train_users.isdisjoint(val_users)

    train_ts = pd.to_datetime(x_train["DateTime"])
    val_ts = pd.to_datetime(x_val["DateTime"])
    assert train_ts.max() < val_ts.min()
