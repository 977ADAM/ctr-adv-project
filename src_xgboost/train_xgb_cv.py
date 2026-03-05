import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .config import Config
from .data import load_parquet, maybe_sample
from .features import build_features, get_feature_cols
from .utils import ensure_dir


def main():
    cfg = Config()
    ensure_dir(cfg.output_dir)

    train_df = load_parquet(cfg.train_path)
    test_df = load_parquet(cfg.test_path)

    # 빠른 재현 샘플링 (원하면 cfg.train_sample_n 설정)
    train_df = maybe_sample(train_df, cfg.train_sample_n, cfg.random_state)

    # Feature engineering
    train_df, test_df = build_features(train_df, test_df, target_col=cfg.target_col)

    feature_cols = get_feature_cols(train_df, target_col=cfg.target_col, id_col=cfg.id_col)
    X = train_df[feature_cols]
    y = train_df[cfg.target_col].astype(int)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = XGBClassifier(**cfg.xgb_params)
        model.fit(X_tr, y_tr)

        pred = model.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, pred)
        aucs.append(auc)
        print(f"[Fold {fold}] AUC: {auc:.6f}")

    print("=" * 40)
    print(f"Mean CV AUC: {np.mean(aucs):.6f}  |  Std: {np.std(aucs):.6f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
