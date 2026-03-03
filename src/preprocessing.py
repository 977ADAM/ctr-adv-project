from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import joblib

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
    DEFAULT_CATEGORICAL_COLS = ("gender", "product")

    def __init__(self):
        self.scaler = StandardScaler()
        self.cat_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        self.medians = None

        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []

    @staticmethod
    def _require_columns(
        df: pd.DataFrame,
        required_cols: Sequence[str],
        stage: str,
    ) -> None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"{stage}: missing required columns: {missing}"
            )

    # -------------------- Feature engineering --------------------

    def _prepare_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, ["DateTime"], "datetime preprocessing")
        df = df.copy()
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
        df["hour"] = df["DateTime"].dt.hour.fillna(0).astype(int)
        df["dayofweek"] = df["DateTime"].dt.dayofweek.fillna(0).astype(int)
        return df.drop(columns=["DateTime"])

    def _build_feature_lists(self, df: pd.DataFrame):
        self._require_columns(
            df,
            list(self.DEFAULT_CATEGORICAL_COLS),
            "feature schema validation",
        )
        self.categorical_cols = list(self.DEFAULT_CATEGORICAL_COLS)
        self.numerical_cols = [
            c for c in df.columns
            if c not in self.categorical_cols + [self.TARGET]
        ]

    # -------------------- Split --------------------

    def make_splits(
        self,
        df: pd.DataFrame,
        test_size: float,
    ):
        if not 0.0 < test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        self._require_columns(
            df,
            ["DateTime", "user_id", self.TARGET],
            "train/validation split",
        )

        split_df = df.copy()
        split_df["DateTime"] = pd.to_datetime(
            split_df["DateTime"], errors="coerce"
        )
        split_df = split_df.dropna(subset=["DateTime"])
        if split_df.empty:
            raise ValueError("train/validation split: no valid DateTime values")

        # Time holdout: take the latest time window as validation.
        cutoff = split_df["DateTime"].quantile(1.0 - test_size)
        val_time_mask = split_df["DateTime"] >= cutoff
        val_user_ids = set(split_df.loc[val_time_mask, "user_id"].tolist())

        # Group holdout: users from validation window are excluded from train.
        train_mask = (~val_time_mask) & (~split_df["user_id"].isin(val_user_ids))
        val_mask = val_time_mask

        if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
            raise ValueError(
                "Split produced an empty train/validation set. "
                "Adjust test_size or check DateTime/user_id distribution."
            )

        train_df = split_df.loc[train_mask].copy()
        val_df = split_df.loc[val_mask].copy()

        X_train = train_df.drop(columns=[self.TARGET])
        y_train = train_df[self.TARGET].astype(np.float32).values
        X_val = val_df.drop(columns=[self.TARGET])
        y_val = val_df[self.TARGET].astype(np.float32).values

        return X_train, X_val, y_train, y_val

    # -------------------- Fit / Transform --------------------

    def fit(self, df: pd.DataFrame):
        df = self._prepare_datetime(df)
        self._require_columns(df, [self.TARGET], "fit")
        self._build_feature_lists(df)

        self.medians = df[self.numerical_cols].median(numeric_only=True)
        df[self.numerical_cols] = df[self.numerical_cols].fillna(self.medians)

        # 🔥 добавляем это
        df[self.categorical_cols] = df[self.categorical_cols].fillna("UNK")

        self.scaler.fit(df[self.numerical_cols])
        self.cat_encoder.fit(df[self.categorical_cols])

        return self

    def transform(self, df: pd.DataFrame):
        if not self.numerical_cols or not self.categorical_cols:
            raise ValueError(
                "transform: preprocessor is not fitted/loaded "
                "(feature lists are empty)"
            )
        df = self._prepare_datetime(df)
        self._require_columns(
            df,
            self.numerical_cols + self.categorical_cols,
            "transform",
        )
        if self.medians is None:
            raise ValueError(
                "transform: preprocessor is not fitted/loaded (medians are missing)"
            )

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
