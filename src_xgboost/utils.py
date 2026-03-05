from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def save_submission(
    ids: pd.Series | np.ndarray,
    preds: np.ndarray,
    out_path: str | Path,
    id_col: str,
    target_col: str,
) -> Path:
    out_file = Path(out_path)
    df = pd.DataFrame({id_col: ids, target_col: preds})
    df.to_csv(out_file, index=False)
    return out_file


def save_json(data: dict[str, Any], out_path: str | Path) -> Path:
    out_file = Path(out_path)
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out_file


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
