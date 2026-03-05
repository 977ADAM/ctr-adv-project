import pandas as pd
import numpy as np

def get_feature_cols(df: pd.DataFrame, target_col="clicked", id_col="ID") -> list[str]:
    exclude = {target_col, id_col}
    return [c for c in df.columns if c not in exclude]

def safe_group_features(df: pd.DataFrame) -> pd.DataFrame:
    # hour_group, day_group (컬럼이 있을 때만)
    if "hour" in df.columns and np.issubdtype(df["hour"].dtype, np.number):
        df["hour_group"] = (df["hour"] // 6).astype("int32")
    if "day_of_week" in df.columns and np.issubdtype(df["day_of_week"].dtype, np.number):
        df["day_group"] = (df["day_of_week"] // 2).astype("int32")
    return df

def history_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    history_cols = [c for c in df.columns if c.startswith("history")]
    if len(history_cols) == 0:
        return df
    # 결측 처리 후 집계
    temp = df[history_cols].fillna(0)
    df["history_sum"] = temp.sum(axis=1).astype("float32")
    df["history_mean"] = temp.mean(axis=1).astype("float32")
    df["history_max"] = temp.max(axis=1).astype("float32")
    return df

def target_encode_inventory(train_df: pd.DataFrame, test_df: pd.DataFrame,
                            target_col="clicked", col="inventory_id") -> tuple[pd.DataFrame, pd.DataFrame]:
    if col not in train_df.columns or col not in test_df.columns:
        return train_df, test_df

    te = train_df.groupby(col)[target_col].mean()
    global_mean = float(train_df[target_col].mean())

    train_df[f"{col}_te"] = train_df[col].map(te).fillna(global_mean).astype("float32")
    test_df[f"{col}_te"] = test_df[col].map(te).fillna(global_mean).astype("float32")
    return train_df, test_df

def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   target_col="clicked") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = safe_group_features(train_df)
    test_df = safe_group_features(test_df)

    train_df = history_aggregate_features(train_df)
    test_df = history_aggregate_features(test_df)

    # target encoding (inventory_id)
    train_df, test_df = target_encode_inventory(train_df, test_df, target_col=target_col, col="inventory_id")
    return train_df, test_df
