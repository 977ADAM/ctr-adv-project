import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def train_xgb_cv(train_df, feature_cols, target_col="clicked"):

    X = train_df[feature_cols]
    y = train_df[target_col]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)

        print(f"Fold {fold+1} AUC: {auc:.6f}")
        scores.append(auc)

    print("Mean CV AUC:", np.mean(scores))
    return model
