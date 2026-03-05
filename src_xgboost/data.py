import os
import pandas as pd

def load_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {path}\n"
            f"data/ 폴더에 train.parquet, test.parquet를 두고 실행하세요."
        )
    return pd.read_parquet(path, engine="pyarrow")

def maybe_sample(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    if n is None:
        return df
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)
