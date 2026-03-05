import pandas as pd

TARGET_COL = "clicked"

def load_data(path):
    return pd.read_parquet(path)

def get_feature_columns(df, target_col=TARGET_COL):
    exclude_cols = ["ID", target_col]
    return [c for c in df.columns if c not in exclude_cols]
