import pandas as pd
import numpy as np


class TimeFeaturesTransformer:

    def __init__(
        self,
        user_col: str,
        time_col: str,
        target_col: str,
        lags=(1,7,30),
        event_id_col: str | None = None,
    ):
        self.user_col = user_col
        self.time_col = time_col
        self.target_col = target_col
        self.lags = lags
        self.event_id_col = event_id_col
        self.feature_columns = [
            "year", "month", "day", "dayofweek",
            "hour", "weekofyear", "is_weekend",
            "hour_sin", "hour_cos",
            "dow_sin", "dow_cos",
        ]

    def _ensure_utc(self, s: pd.Series) -> pd.Series:
        dt = pd.to_datetime(s, errors="coerce", utc=True)

        if dt.isna().any(): # проверка на пропуски
            raise ValueError("Не удалось обработать некоторые временные метки.")
        
        return dt

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df[self.time_col] = self._ensure_utc(df[self.time_col])

        sort_cols = [self.user_col, self.time_col]
        if self.event_id_col is not None and self.event_id_col in df.columns:
            sort_cols.append(self.event_id_col)
        df = df.sort_values(sort_cols, kind="mergesort")

        dupli = df.duplicated([self.user_col, self.time_col], keep=False)
        if dupli.any() and self.event_id_col is None:
            raise ValueError(
                "Обнаружен дубликат (пользователь, время). Укажите event_id_col для детерминированного упорядочивания"
                "или удалите дубликат из вышестоящего источника."
            )

        df["year"] = df[self.time_col].dt.year
        df["month"] = df[self.time_col].dt.month
        df["day"] = df[self.time_col].dt.day
        df["dayofweek"] = df[self.time_col].dt.dayofweek
        df["hour"] = df[self.time_col].dt.hour
        df["weekofyear"] = df[self.time_col].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

        # cyclic
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

        feature_cols = list(self.feature_columns)

        grouped = df.groupby(self.user_col)[self.target_col]

        # lags
        for lag in self.lags:
            col = f"lag_{lag}"
            df[col] = grouped.shift(lag)
            feature_cols.append(col)

        
        shifted = grouped.shift(1)

        df["rolling_mean_7"] = (
            shifted.rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
        )

        df["rolling_std_30"] = (
            shifted.rolling(30, min_periods=1).std().reset_index(level=0, drop=True)
        )

        df["ewm_7"] = (
            shifted
                .ewm(span=7, adjust=False)
                .mean()
                .reset_index(level=0, drop=True)
        )
        feature_cols += [
            "rolling_mean_7",
            "rolling_std_30",
            "ewm_7",
        ]

        self.feature_columns = feature_cols

        df = df.reset_index(drop=True)
        
        return df
    
    def get_feature_columns(self):
        return self.feature_columns