import pandas as pd
import numpy as np

def get_feature_cols(df: pd.DataFrame, target_col="is_click", session_id_col="session_id") -> list[str]:
    exclude = {target_col, session_id_col}
    return [c for c in df.columns if c not in exclude]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df["DateTime"] = pd.to_datetime(df["DateTime"])

    df = df.sort_values("DateTime")

    dt = df["DateTime"]

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["hour"] = dt.dt.hour
    df["weekofyear"] = dt.dt.isocalendar().week.astype("int16")

    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype("int8")

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["dow_sin"] = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dayofweek"]/7)

    return df

def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   target_col="is_click") -> tuple[pd.DataFrame, pd.DataFrame]:
    
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df = add_time_features(train_df)
    test_df = add_time_features(test_df)

    return train_df, test_df
