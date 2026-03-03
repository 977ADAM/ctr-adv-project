from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


class CTRPreprocessor:
    """
    Responsible for:
    - feature engineering
    - train/val splitting
    - scaling numerical features
    - encoding categorical features
    - saving/loading preprocessing artifacts
    """

    TARGET = "is_click"

    def __init__(self):
        self.scaler = StandardScaler()
        self.cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.medians = None

        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []

    # -------------------- Feature engineering --------------------

    def _prepare_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df["hour"] = df["DateTime"].dt.hour.fillna(0).astype(int)
        df["dayofweek"] = df["DateTime"].dt.dayofweek.fillna(0).astype(int)
        return df.drop(columns=["DateTime"])

    def _build_feature_lists(self, df: pd.DataFrame):
        self.categorical_cols = ["gender", "product"]
        self.numerical_cols = [
            c for c in df.columns
            if c not in self.categorical_cols + [self.TARGET]
        ]

    # -------------------- Split --------------------

    def make_splits(
        self,
        df: pd.DataFrame,
        test_size: float,
        seed: int,
    ):
        X = df.drop(columns=[self.TARGET])
        y = df[self.TARGET].astype(np.float32).values

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            stratify=y,
        )

    # -------------------- Fit / Transform --------------------

    def fit(self, df: pd.DataFrame):
        df = self._prepare_datetime(df)
        self._build_feature_lists(df)

        self.medians = df[self.numerical_cols].median(numeric_only=True)
        df[self.numerical_cols] = df[self.numerical_cols].fillna(self.medians)

        # 🔥 добавляем это
        df[self.categorical_cols] = df[self.categorical_cols].fillna("UNK")

        self.scaler.fit(df[self.numerical_cols])
        self.cat_encoder.fit(df[self.categorical_cols])

        return self

    def transform(self, df: pd.DataFrame):
        df = self._prepare_datetime(df)

        # числовые
        df[self.numerical_cols] = df[self.numerical_cols].fillna(self.medians)

        # 🔥 ВАЖНО: заполняем NaN в категориальных
        df[self.categorical_cols] = df[self.categorical_cols].fillna("UNK")

        # encoder
        X_num = self.scaler.transform(df[self.numerical_cols])

        X_cat_raw = self.cat_encoder.transform(df[self.categorical_cols])

        # безопасная обработка nan
        X_cat_raw = np.nan_to_num(X_cat_raw, nan=-1)

        X_cat = X_cat_raw.astype(np.int64) + 1

        return X_num, X_cat

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    # -------------------- Persistence --------------------

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path / "scaler.joblib")
        joblib.dump(self.cat_encoder, path / "cat_encoder.joblib")

        meta = {
            "categorical_cols": self.categorical_cols,
            "numerical_cols": self.numerical_cols,
            "medians": (
                self.medians.to_dict() if self.medians is not None else None
            ),
        }
        (path / "preprocessing_meta.json").write_text(
            json.dumps(meta, indent=2)
        )

    def load(self, path: Path):
        self.scaler = joblib.load(path / "scaler.joblib")
        self.cat_encoder = joblib.load(path / "cat_encoder.joblib")

        meta = json.loads(
            (path / "preprocessing_meta.json").read_text()
        )
        self.categorical_cols = meta["categorical_cols"]
        self.numerical_cols = meta["numerical_cols"]
        medians = meta.get("medians")
        self.medians = (
            pd.Series(medians, dtype=float) if medians is not None else None
        )
