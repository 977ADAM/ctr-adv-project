import os
import pandas as pd

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_submission(ids, preds, out_path: str, id_col="ID", target_col="clicked"):
    df = pd.DataFrame({id_col: ids, target_col: preds})
    df.to_csv(out_path, index=False)
    return out_path
