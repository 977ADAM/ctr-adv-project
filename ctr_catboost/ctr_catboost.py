from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier



TARGET_COL = "is_click"
DEFAULT_CATEGORICAL_COLS = (
    "gender",
    "product",
    "user_id",
    "campaign_id",
    "webpage_id",
    "user_group_id",
    "product_category_1",
    "product_category_2",
    "age_level",
    "user_depth",
    "city_development_index",
    "var_1",
)
INTERACTION_FEATURE_PAIRS = (
    ("campaign_id", "webpage_id"),
    ("product", "campaign_id"),
    ("product", "webpage_id"),
    ("product", "gender"),
    ("user_group_id", "product_category_1"),
)
MISSING_FLAG_COLS = (
    "product_category_2",
    "gender",
    "user_group_id",
    "age_level",
    "user_depth",
    "city_development_index",
)
FREQUENCY_FEATURE_COLS = (
    "user_id",
    "campaign_id",
    "webpage_id",
    "product",
    "product_category_1",
    "product_category_2",
    "campaign_id__webpage_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost model to predict whether user clicks or not (is_click)."
    )
    parser.add_argument("--train-path", type=Path, default=Path("data/dataset_train.csv"))
    parser.add_argument("--test-path", type=Path, default=Path("data/dataset_test.csv"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/catboost/catboost_bundle.joblib"),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/catboost/metrics.json"),
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=Path("artifacts/catboost/test_predictions.csv"),
    )
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=8)
    return parser.parse_args()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        df["hour"] = dt.dt.hour.fillna(0).astype(int)
        df["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        df["day"] = dt.dt.day.fillna(0).astype(int)
        df["month"] = dt.dt.month.fillna(0).astype(int)
        df["weekofyear"] = dt.dt.isocalendar().week.fillna(0).astype(int)
        df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
        df["hour_sin"] = np.sin((2.0 * np.pi * df["hour"]) / 24.0)
        df["hour_cos"] = np.cos((2.0 * np.pi * df["hour"]) / 24.0)
        df["dayofweek_sin"] = np.sin((2.0 * np.pi * df["dayofweek"]) / 7.0)
        df["dayofweek_cos"] = np.cos((2.0 * np.pi * df["dayofweek"]) / 7.0)
        df = df.drop(columns=["DateTime"])
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for left, right in INTERACTION_FEATURE_PAIRS:
        if left in df.columns and right in df.columns:
            left_val = df[left].astype("string").fillna("MISSING")
            right_val = df[right].astype("string").fillna("MISSING")
            df[f"{left}__{right}"] = (left_val + "_" + right_val).astype(str)
    return df


def add_missing_value_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MISSING_FLAG_COLS:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(np.int8)
    return df


def build_frequency_maps(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    frequency_maps: dict[str, dict[str, float]] = {}
    for col in FREQUENCY_FEATURE_COLS:
        if col not in df.columns:
            continue
        values = df[col].astype("string").fillna("MISSING").astype(str)
        frequency_maps[col] = values.value_counts(normalize=True).to_dict()
    return frequency_maps


def add_frequency_features(
    df: pd.DataFrame, frequency_maps: dict[str, dict[str, float]]
) -> pd.DataFrame:
    df = df.copy()
    for col, col_freq_map in frequency_maps.items():
        feat_col = f"{col}_freq"
        if col in df.columns:
            values = df[col].astype("string").fillna("MISSING").astype(str)
            df[feat_col] = values.map(col_freq_map).fillna(0.0).astype(np.float32)
        else:
            df[feat_col] = np.zeros(len(df), dtype=np.float32)
    return df


def build_feature_frame(
    df: pd.DataFrame,
    frequency_maps: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    x = add_time_features(df)
    x = add_missing_value_flags(x)
    x = add_interaction_features(x)
    if frequency_maps is None:
        frequency_maps = build_frequency_maps(x)
    x = add_frequency_features(x, frequency_maps)
    return x, frequency_maps


def choose_categorical_columns(df: pd.DataFrame) -> list[str]:
    explicit_categorical = [c for c in DEFAULT_CATEGORICAL_COLS if c in df.columns]
    inferred_categorical = [
        c
        for c in df.columns
        if c not in explicit_categorical
        and (
            pd.api.types.is_object_dtype(df[c])
            or pd.api.types.is_string_dtype(df[c])
            or pd.api.types.is_bool_dtype(df[c])
            or isinstance(df[c].dtype, pd.CategoricalDtype)
        )
    ]
    return list(dict.fromkeys(explicit_categorical + inferred_categorical))


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    frequency_maps: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, dict[str, float]]]:
    x, frequency_maps = build_feature_frame(df, frequency_maps=frequency_maps)

    if feature_columns is None:
        feature_columns = list(x.columns)
    else:
        for col in feature_columns:
            if col not in x.columns:
                x[col] = np.nan
        x = x[feature_columns]

    if categorical_cols is None:
        categorical_cols = choose_categorical_columns(x)
    else:
        categorical_cols = [c for c in categorical_cols if c in x.columns]

    for col in categorical_cols:
        x[col] = x[col].astype("string").fillna("MISSING").astype(str)

    return x, feature_columns, categorical_cols, frequency_maps


def time_aware_split(df: pd.DataFrame, val_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be in (0, 1)")

    if "DateTime" in df.columns:
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
        
        if dt.notna().sum() >= 2:
            tmp = df.copy()
            tmp["_dt"] = dt
            tmp = tmp.sort_values("_dt", na_position="first")
            split_idx = int(len(tmp) * (1.0 - val_size))
            split_idx = min(max(1, split_idx), len(tmp) - 1)
            train_df = tmp.iloc[:split_idx].drop(columns=["_dt"]).copy()
            val_df = tmp.iloc[split_idx:].drop(columns=["_dt"]).copy()
            return train_df, val_df

    stratify = df[TARGET_COL] if df[TARGET_COL].nunique(dropna=False) > 1 else None
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=42,
        stratify=stratify,
    )
    return train_df.copy(), val_df.copy()


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def train(
    train_path: Path,
    val_size: float,
    iterations: int,
    learning_rate: float,
    depth: int,
) -> tuple[dict[str, object], dict[str, float]]:
    if CatBoostClassifier is None:
        raise ImportError(
            "CatBoost is not installed. Install it with: pip install catboost"
        )

    df = pd.read_csv(train_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column not found in {train_path}")

    train_df, val_df = time_aware_split(df, val_size)
    y_train = train_df[TARGET_COL].astype(np.int32).to_numpy()
    y_val = val_df[TARGET_COL].astype(np.int32).to_numpy()

    x_train, feature_columns, categorical_cols, frequency_maps = prepare_features(
        train_df.drop(columns=[TARGET_COL])
    )
    x_val, _, _, _ = prepare_features(
        val_df.drop(columns=[TARGET_COL]),
        feature_columns=feature_columns,
        categorical_cols=categorical_cols,
        frequency_maps=frequency_maps,
    )
    cat_feature_indices = [x_train.columns.get_loc(col) for col in categorical_cols]

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        auto_class_weights="Balanced",
        depth=depth,
        random_seed=42,
        verbose=False,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=cat_feature_indices if cat_feature_indices else None,
        eval_set=(x_val, y_val),
        use_best_model=True,
    )

    val_proba = model.predict_proba(x_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(np.int32)
    metrics = {
        "roc_auc": safe_roc_auc(y_val, val_proba),
        "pr_auc": float(average_precision_score(y_val, val_proba)),
        "log_loss": float(log_loss(y_val, val_proba, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "best_iteration": float(max(model.get_best_iteration(), 0)),
    }
    bundle: dict[str, object] = {
        "model": model,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_cols,
        "frequency_maps": frequency_maps,
    }
    return bundle, metrics


def save_artifacts(
    bundle: dict[str, object],
    metrics: dict[str, float],
    model_path: Path,
    metrics_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))


def make_test_predictions(
    bundle: dict[str, object],
    test_path: Path,
    predictions_path: Path,
) -> Path | None:
    if not test_path.exists():
        return None

    test_df = pd.read_csv(test_path)
    session_col = (
        test_df["session_id"]
        if "session_id" in test_df.columns
        else pd.Series(np.arange(len(test_df)), name="session_id")
    )

    feature_columns = bundle["feature_columns"]
    categorical_cols = bundle["categorical_columns"]
    frequency_maps = bundle.get("frequency_maps", {})
    model = bundle["model"]
    assert isinstance(feature_columns, list)
    assert isinstance(categorical_cols, list)
    assert isinstance(frequency_maps, dict)

    x_test, _, _, _ = prepare_features(
        test_df,
        feature_columns=feature_columns,
        categorical_cols=categorical_cols,
        frequency_maps=frequency_maps,
    )
    proba = model.predict_proba(x_test)[:, 1]

    out_df = pd.DataFrame(
        {
            "session_id": session_col,
            "click_probability": proba,
            "is_click_pred": (proba >= 0.5).astype(np.int32),
        }
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(predictions_path, index=False)
    return predictions_path


def main() -> None:
    args = parse_args()
    bundle, metrics = train(
        train_path=args.train_path,
        val_size=args.val_size,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
    )
    save_artifacts(bundle, metrics, args.model_path, args.metrics_path)
    pred_path = make_test_predictions(bundle, args.test_path, args.predictions_path)

    print(f"Model saved to: {args.model_path}")
    print(f"Metrics saved to: {args.metrics_path}")
    if pred_path is not None:
        print(f"Test predictions saved to: {pred_path}")
    print(f"Validation metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()
