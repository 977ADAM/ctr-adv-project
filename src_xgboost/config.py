from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Config:
    train_path: str = "./data/dataset_train.csv"
    test_path: str = "./data/dataset_test.csv"
    output_dir: str = "outputs/xgboost"
    model_name: str = "ctr_xgb_model.joblib"
    submission_name: str = "submission_xgb.csv"
    metrics_name: str = "cv_metrics.json"
    fold_metrics_name: str = "cv_fold_metrics.csv"

    target_col: str = "is_click"
    session_id_col: str = "session_id"
    group_col: str = "user_id"
    datetime_col: str = "DateTime"

    numeric_features: list[str] = field(
        default_factory=lambda: [
            "age_level",
            "user_depth",
            "city_development_index",
            "var_1",
            "hour",
            "dayofweek",
            "weekofyear",
            "day",
            "dayofyear",
            "month",
            "quarter",
            "year",
            "is_weekend",
            "is_work_hour",
            "is_night",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "user_event_count",
            "user_unique_campaigns",
            "user_unique_products",
            "user_activity_span_hours",
            "user_prev_gap_hours",
            "campaign_id_freq_norm",
            "webpage_id_freq_norm",
            "product_freq_norm",
            "product_category_1_freq_norm",
            "product_category_2_freq_norm",
            "user_group_id_freq_norm",
            "gender_freq_norm",
            "campaign_webpage_freq_norm",
            "campaign_product_freq_norm",
            "user_group_product_freq_norm",
            "campaign_hour_bin_freq_norm",
            "webpage_hour_bin_freq_norm",
            "product_hour_bin_freq_norm",
        ]
    )
    categorical_features: list[str] = field(
        default_factory=lambda: [
            "campaign_id",
            "webpage_id",
            "product",
            "product_category_1",
            "product_category_2",
            "user_group_id",
            "gender",
            "hour_bin",
            "campaign_webpage",
            "campaign_product",
            "user_group_product",
            "gender_age",
            "product_cat1",
            "campaign_hour_bin",
            "webpage_hour_bin",
            "product_hour_bin",
        ]
    )

    n_splits: int = 5
    random_state: int = 42
    train_sample_n: int | None = None
    early_stopping_rounds: int = 100
    scale_pos_weight_power: float = 0.0

    xgb_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.train_sample_n is not None and self.train_sample_n <= 0:
            raise ValueError("train_sample_n must be positive when provided")
        if self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive")
        if self.scale_pos_weight_power < 0:
            raise ValueError("scale_pos_weight_power must be >= 0")

        default_xgb_params: dict[str, Any] = {
            "n_estimators": 3000,
            "learning_rate": 0.02,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "colsample_bylevel": 0.75,
            "gamma": 0.0,
            "reg_lambda": 6.0,
            "reg_alpha": 1.0,
            "random_state": self.random_state,
            "eval_metric": "auc",
            "tree_method": "hist",
            "n_jobs": -1,
            "enable_categorical": False,
            "max_bin": 256,
            "max_delta_step": 1,
        }
        if self.xgb_params:
            default_xgb_params.update(self.xgb_params)
        self.xgb_params = default_xgb_params

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def model_path(self) -> Path:
        return self.output_path / self.model_name
