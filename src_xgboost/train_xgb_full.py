import os
import numpy as np
from xgboost import XGBClassifier

from .config import Config
from .data import load_parquet
from .features import build_features, get_feature_cols
from .utils import ensure_dir, save_submission


def main():
    cfg = Config()
    ensure_dir(cfg.output_dir)

    train_df = load_parquet(cfg.train_path)
    test_df = load_parquet(cfg.test_path)

    train_df, test_df = build_features(train_df, test_df, target_col=cfg.target_col)

    feature_cols = get_feature_cols(train_df, target_col=cfg.target_col, id_col=cfg.id_col)
    X_train = train_df[feature_cols]
    y_train = train_df[cfg.target_col].astype(int)

    X_test = test_df[feature_cols]
    test_ids = test_df[cfg.id_col].values

    model = XGBClassifier(**cfg.xgb_params)
    model.fit(X_train, y_train)

    test_pred = model.predict_proba(X_test)[:, 1]

    out_path = os.path.join(cfg.output_dir, cfg.submission_name)
    save_path = save_submission(test_ids, test_pred, out_path, id_col=cfg.id_col, target_col=cfg.target_col)
    print(f"✅ Saved submission: {save_path}")


if __name__ == "__main__":
    main()
