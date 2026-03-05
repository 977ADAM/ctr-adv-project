from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def add_time_features(
    df: pd.DataFrame,
    datetime_col: str = "DateTime",
    drop_datetime: bool = False,
) -> pd.DataFrame:
    if datetime_col not in df.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")

    frame = df.copy()
    dt = pd.to_datetime(frame[datetime_col], errors="coerce")
    invalid_count = int(dt.isna().sum())
    if invalid_count > 0:
        raise ValueError(
            f"{datetime_col} has {invalid_count} invalid timestamp values; "
            "fix input data before training."
        )
    frame[datetime_col] = dt

    frame["year"] = dt.dt.year.astype("int16")
    frame["month"] = dt.dt.month.astype("int8")
    frame["day"] = dt.dt.day.astype("int8")
    frame["dayofweek"] = dt.dt.dayofweek.astype("int8")
    frame["hour"] = dt.dt.hour.astype("int8")
    frame["weekofyear"] = dt.dt.isocalendar().week.astype("int16")
    frame["dayofyear"] = dt.dt.dayofyear.astype("int16")
    frame["quarter"] = dt.dt.quarter.astype("int8")

    frame["is_weekend"] = frame["dayofweek"].isin([5, 6]).astype("int8")
    frame["is_work_hour"] = frame["hour"].between(9, 18).astype("int8")
    frame["is_night"] = ((frame["hour"] <= 5) | (frame["hour"] >= 23)).astype("int8")
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["dayofweek"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["dayofweek"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    frame["hour_bin"] = pd.cut(
        frame["hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "day", "evening"],
    ).astype("string")

    if "user_id" in frame.columns:
        user_group = frame.groupby("user_id", sort=False)
        frame["user_event_count"] = user_group["user_id"].transform("size").astype("int32")
        if "campaign_id" in frame.columns:
            frame["user_unique_campaigns"] = user_group["campaign_id"].transform("nunique").astype("int16")
        if "product" in frame.columns:
            frame["user_unique_products"] = user_group["product"].transform("nunique").astype("int16")

        max_dt = user_group[datetime_col].transform("max")
        min_dt = user_group[datetime_col].transform("min")
        span_hours = (max_dt - min_dt).dt.total_seconds().div(3600).astype("float32")
        frame["user_activity_span_hours"] = span_hours

        sorted_idx = frame.sort_values(["user_id", datetime_col]).index
        dt_sorted = frame.loc[sorted_idx, datetime_col]
        user_sorted = frame.loc[sorted_idx, "user_id"]
        prev_gap = (
            dt_sorted.groupby(user_sorted).diff().dt.total_seconds().div(3600).fillna(-1.0).clip(-1.0, 24.0 * 30.0)
        )
        frame.loc[sorted_idx, "user_prev_gap_hours"] = prev_gap.astype("float32")
    else:
        frame["user_event_count"] = 1.0
        frame["user_unique_campaigns"] = 1.0
        frame["user_unique_products"] = 1.0
        frame["user_activity_span_hours"] = 0.0
        frame["user_prev_gap_hours"] = -1.0

    if drop_datetime:
        frame = frame.drop(columns=[datetime_col])
    return frame


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()

    def _str_cat(col_name: str) -> pd.Series:
        if col_name not in frame.columns:
            return pd.Series(["NA"] * len(frame), index=frame.index, dtype="string")
        return frame[col_name].astype("string").fillna("NA")

    frame["campaign_webpage"] = _str_cat("campaign_id") + "_" + _str_cat("webpage_id")
    frame["campaign_product"] = _str_cat("campaign_id") + "_" + _str_cat("product")
    frame["user_group_product"] = _str_cat("user_group_id") + "_" + _str_cat("product")
    frame["gender_age"] = _str_cat("gender") + "_" + _str_cat("age_level")
    frame["product_cat1"] = _str_cat("product") + "_" + _str_cat("product_category_1")
    frame["campaign_hour_bin"] = _str_cat("campaign_id") + "_" + _str_cat("hour_bin")
    frame["webpage_hour_bin"] = _str_cat("webpage_id") + "_" + _str_cat("hour_bin")
    frame["product_hour_bin"] = _str_cat("product") + "_" + _str_cat("hour_bin")

    for col in (
        "campaign_id",
        "webpage_id",
        "product",
        "product_category_1",
        "product_category_2",
        "user_group_id",
        "gender",
        "campaign_webpage",
        "campaign_product",
        "user_group_product",
        "campaign_hour_bin",
        "webpage_hour_bin",
        "product_hour_bin",
    ):
        if col in frame.columns:
            freq = frame[col].value_counts(dropna=False)
            frame[f"{col}_freq_norm"] = frame[col].map(freq).fillna(0).astype("float32") / len(frame)

    return frame


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    datetime_col: str = "DateTime",
    drop_datetime: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = add_time_features(train_df, datetime_col=datetime_col, drop_datetime=drop_datetime)
    test_frame = add_time_features(test_df, datetime_col=datetime_col, drop_datetime=drop_datetime)
    train_frame = add_interaction_features(train_frame)
    test_frame = add_interaction_features(test_frame)
    return train_frame, test_frame


def resolve_feature_columns(
    train_df: pd.DataFrame,
    numeric_candidates: Iterable[str],
    categorical_candidates: Iterable[str],
) -> tuple[list[str], list[str]]:
    cols = set(train_df.columns)
    numeric_features = [col for col in numeric_candidates if col in cols]
    categorical_features = [col for col in categorical_candidates if col in cols]

    if not numeric_features:
        raise ValueError("No numeric features were found in train data after feature engineering.")
    if not categorical_features:
        raise ValueError("No categorical features were found in train data after feature engineering.")
    return numeric_features, categorical_features
