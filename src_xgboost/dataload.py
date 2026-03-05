from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_csv(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    dataset_name: str,
) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing_cols)}"
        )


def validate_train_test_schema(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> None:
    train_features = [c for c in train_df.columns if c != target_col]
    missing_in_test = sorted(set(train_features) - set(test_df.columns))
    if missing_in_test:
        raise ValueError(
            f"test is missing columns present in train: {', '.join(missing_in_test)}"
        )


def stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    sample_n: int | None,
    random_state: int,
) -> pd.DataFrame:
    if sample_n is None or sample_n >= len(df):
        return df.reset_index(drop=True)

    n_classes = int(df[target_col].nunique())
    if sample_n < n_classes * 2:
        return df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_n, random_state=random_state)
    sampled_idx, _ = next(splitter.split(df, df[target_col]))
    sampled_df = df.iloc[sampled_idx]
    sampled_df = sampled_df.sample(frac=1.0, random_state=random_state)
    return sampled_df.reset_index(drop=True)
