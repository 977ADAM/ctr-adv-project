import numpy as np
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from xgboost import XGBClassifier

from .config import Config
from .dataload import load_csv
from .features import build_features, get_feature_cols
from .utils import ensure_dir


def main():
    cfg = Config() # Конфигурация проекта
    ensure_dir(cfg.output_dir) # создание папки для выходных данных

    train_df = load_csv(cfg.train_path) # загрузка тренировачного датасета
    test_df = load_csv(cfg.test_path) # загрузка тестового датасета

    train_df, test_df = build_features(train_df, test_df, target_col=cfg.target_col)

    feature_cols = get_feature_cols(train_df, target_col=cfg.target_col, id_col=cfg.id_col)

    X = train_df[feature_cols]
    y = train_df[cfg.target_col].astype(int)

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    aucs = []

    for fold, (train_idx, valid_idx) in enumerate(sgkf.split(X, y, groups=train_df["user_id"])):

        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        params = cfg.xgb_params.copy()
        params["scale_pos_weight"] = scale_pos_weight

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_valid)[:, 1]

        auc = roc_auc_score(y_valid, proba)
        ll = log_loss(y_valid, proba)
        pr_auc = average_precision_score(y_valid, proba)

        aucs.append(auc)

        print(f"===============Fold {fold+1} ROC AUC: {auc:.6f} PR AUC: {pr_auc:.6f} LogLoss: {ll:.6f} ================")

    print("=" * 40)
    print(f"Mean CV AUC: {np.mean(aucs):.6f}  |  Std: {np.std(aucs):.6f}")
    print("=" * 40)

    print("\nTraining final model on full data...")

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    params = cfg.xgb_params.copy()
    params["scale_pos_weight"] = scale_pos_weight

    model = XGBClassifier(**params)

    model.fit(X, y)

    joblib.dump(model, "ctr_model.pkl")


if __name__ == "__main__":
    main()
