from __future__ import annotations

import argparse
from dataclasses import asdict, fields
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from src_xgboost.config import Config
from src_xgboost.dataload import (
    load_csv,
    stratified_sample,
    validate_required_columns,
    validate_train_test_schema,
)
from src_xgboost.features import build_features, resolve_feature_columns
from src_xgboost.preprocess import build_preprocessor
from src_xgboost.utils import ensure_dir, save_json, save_submission, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost pipeline with CV and artifacts.")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--submission_name", type=str, default=None)
    parser.add_argument("--n_splits", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=None)
    parser.add_argument("--train_sample_n", type=int, default=None)
    parser.add_argument("--early_stopping_rounds", type=int, default=None)
    parser.add_argument("--scale_pos_weight_power", type=float, default=0.5)
    parser.add_argument(
        "--drop_datetime",
        action="store_true",
        help="Drop DateTime column after feature engineering.",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    base = Config()
    cfg_kwargs = {
        field_.name: getattr(base, field_.name)
        for field_ in fields(Config)
        if field_.name != "xgb_params"
    }
    cfg_kwargs["xgb_params"] = {}
    for name in ("train_path", "test_path", "output_dir", "submission_name"):
        value = getattr(args, name)
        if value is not None:
            cfg_kwargs[name] = value
    for name in (
        "n_splits",
        "random_state",
        "train_sample_n",
        "early_stopping_rounds",
        "scale_pos_weight_power",
    ):
        value = getattr(args, name)
        if value is not None:
            cfg_kwargs[name] = float(value) if name == "scale_pos_weight_power" else int(value)
    return Config(**cfg_kwargs)


def compute_metrics(y_true: pd.Series, preds: np.ndarray) -> dict[str, float]:
    clipped = np.clip(preds, 1e-7, 1 - 1e-7)
    unique_classes = int(pd.Series(y_true).nunique())
    roc_auc = float("nan")
    if unique_classes > 1:
        roc_auc = float(roc_auc_score(y_true, clipped))

    return {
        "roc_auc": roc_auc,
        "pr_auc": float(average_precision_score(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def build_splitter(
    cfg: Config,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series | None,
) -> tuple[Any, str]:
    if groups is not None and groups.nunique() >= cfg.n_splits:
        splitter = StratifiedGroupKFold(
            n_splits=cfg.n_splits,
            shuffle=True,
            random_state=cfg.random_state,
        )
        return splitter.split(X, y, groups=groups), "StratifiedGroupKFold"

    splitter = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    return splitter.split(X, y), "StratifiedKFold"


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    set_global_seed(cfg.random_state)

    output_dir = ensure_dir(cfg.output_path)
    train_df = load_csv(cfg.train_path)
    test_df = load_csv(cfg.test_path)

    validate_required_columns(
        train_df,
        required_cols=[cfg.target_col, cfg.datetime_col],
        dataset_name="train",
    )
    validate_required_columns(test_df, required_cols=[cfg.datetime_col], dataset_name="test")
    validate_train_test_schema(train_df, test_df, target_col=cfg.target_col)

    train_df = stratified_sample(
        train_df,
        target_col=cfg.target_col,
        sample_n=cfg.train_sample_n,
        random_state=cfg.random_state,
    )

    train_df, test_df = build_features(
        train_df,
        test_df,
        datetime_col=cfg.datetime_col,
        drop_datetime=args.drop_datetime,
    )

    numeric_features, categorical_features = resolve_feature_columns(
        train_df,
        numeric_candidates=cfg.numeric_features,
        categorical_candidates=cfg.categorical_features,
    )
    model_features = numeric_features + categorical_features

    X = train_df[model_features].copy()
    y = train_df[cfg.target_col].astype(int).copy()
    X_test = test_df[model_features].copy()

    groups = None
    if cfg.group_col in train_df.columns:
        groups = train_df[cfg.group_col]

    split_iter, split_name = build_splitter(cfg, X, y, groups)
    print(f"Using splitter: {split_name} | rows={len(X)} | features={len(model_features)}")

    oof_preds = np.zeros(len(X), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float64)
    fold_metrics: list[dict[str, float | int]] = []
    best_n_estimators: list[int] = []
    fold_count = 0

    for fold, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            random_state=cfg.random_state,
        )
        X_train_proc = preprocessor.fit_transform(X_train, y_train)
        X_valid_proc = preprocessor.transform(X_valid)
        X_test_proc = preprocessor.transform(X_test)

        pos_count = int((y_train == 1).sum())
        neg_count = int((y_train == 0).sum())
        base_spw = float(neg_count / max(pos_count, 1))
        scale_pos_weight = float(base_spw ** cfg.scale_pos_weight_power)

        fold_params = cfg.xgb_params.copy()
        fold_params["scale_pos_weight"] = scale_pos_weight
        fold_params["early_stopping_rounds"] = cfg.early_stopping_rounds

        model = XGBClassifier(**fold_params)
        model.fit(
            X_train_proc,
            y_train,
            eval_set=[(X_valid_proc, y_valid)],
            verbose=False,
        )

        valid_pred = model.predict_proba(X_valid_proc)[:, 1]
        oof_preds[valid_idx] = valid_pred
        test_preds += model.predict_proba(X_test_proc)[:, 1]

        metrics = compute_metrics(y_valid, valid_pred)
        n_estimators_used = int(getattr(model, "best_iteration", fold_params["n_estimators"] - 1)) + 1
        best_n_estimators.append(n_estimators_used)

        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_idx),
                "valid_rows": len(valid_idx),
                "best_n_estimators": n_estimators_used,
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "log_loss": metrics["log_loss"],
            }
        )
        fold_count += 1
        print(
            f"Fold {fold} | AUC={metrics['roc_auc']:.6f} "
            f"PR_AUC={metrics['pr_auc']:.6f} LogLoss={metrics['log_loss']:.6f} "
            f"best_trees={n_estimators_used}"
        )

    if fold_count == 0:
        raise RuntimeError("No folds were created. Check split configuration and input data.")

    test_preds /= fold_count
    overall_metrics = compute_metrics(y, oof_preds)
    print(
        f"OOF | AUC={overall_metrics['roc_auc']:.6f} "
        f"PR_AUC={overall_metrics['pr_auc']:.6f} LogLoss={overall_metrics['log_loss']:.6f}"
    )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_path = output_dir / cfg.fold_metrics_name
    fold_metrics_df.to_csv(fold_metrics_path, index=False)

    best_trees_median = int(np.median(best_n_estimators))
    global_pos = int((y == 1).sum())
    global_neg = int((y == 0).sum())
    final_params = cfg.xgb_params.copy()
    final_params["n_estimators"] = max(best_trees_median, 100)
    global_spw = float(global_neg / max(global_pos, 1))
    final_params["scale_pos_weight"] = float(global_spw ** cfg.scale_pos_weight_power)
    final_model = XGBClassifier(**final_params)

    final_preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=cfg.random_state,
    )
    X_full_proc = final_preprocessor.fit_transform(X, y)
    final_model.fit(X_full_proc, y, verbose=False)

    model_artifact = {
        "preprocessor": final_preprocessor,
        "model": final_model,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "config": asdict(cfg),
        "metrics": overall_metrics,
    }
    joblib.dump(model_artifact, cfg.model_path)

    oof_path = output_dir / "oof_predictions.csv"
    oof_id_col = cfg.session_id_col if cfg.session_id_col in train_df.columns else "row_id"
    oof_ids = train_df[cfg.session_id_col] if cfg.session_id_col in train_df.columns else np.arange(len(train_df))
    pd.DataFrame(
        {
            oof_id_col: oof_ids,
            cfg.target_col: y.values,
            "prediction": oof_preds,
        }
    ).to_csv(oof_path, index=False)

    if cfg.session_id_col in test_df.columns:
        sub_ids = test_df[cfg.session_id_col]
        sub_id_col = cfg.session_id_col
    else:
        sub_ids = np.arange(len(test_df))
        sub_id_col = "row_id"
    submission_path = save_submission(
        ids=sub_ids,
        preds=test_preds,
        out_path=output_dir / cfg.submission_name,
        id_col=sub_id_col,
        target_col=cfg.target_col,
    )

    summary = {
        "splitter": split_name,
        "n_rows_train": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "n_features": int(len(model_features)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "folds": fold_metrics,
        "oof_metrics": overall_metrics,
        "best_trees_median": best_trees_median,
        "output_files": {
            "model_path": str(cfg.model_path),
            "submission_path": str(submission_path),
            "oof_path": str(oof_path),
            "fold_metrics_path": str(fold_metrics_path),
        },
    }
    summary_path = save_json(summary, output_dir / cfg.metrics_name)

    print(f"Saved model: {cfg.model_path}")
    print(f"Saved submission: {submission_path}")
    print(f"Saved metrics: {summary_path}")


if __name__ == "__main__":
    main()
