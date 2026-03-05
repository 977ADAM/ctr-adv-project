import os
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("=========!!!!!!!!!!!!!!===========")
    return pd.read_csv(path)