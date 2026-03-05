from dataclasses import dataclass

@dataclass
class Config:
    # paths
    train_path: str = "./data/dataset_train.csv"
    test_path: str = "./data/dataset_test.csv"
    output_dir: str = "outputs"
    submission_name: str = "submission_xgb.csv"

    # columns
    target_col: str = "is_click"
    session_id_col: str = "session_id"
    seq_col: str = "seq"

    numeric_features = [
        "age_level",
        "user_depth",
        "city_development_index",
        "var_1",
        "hour",
        "dayofweek",
        "weekofyear",
        "day",
        "month",
        "year",
        "is_weekend",
        'hour_sin',
        'hour_cos',
        'dow_sin',
        'dow_cos',
    ]

    categorical_features = [
        "user_id",
        "campaign_id",
        "webpage_id",
        "product",
        "product_category_1",
        "product_category_2",
        "user_group_id",
        "gender"
    ]

    # CV
    n_splits: int = 5
    random_state: int = 42

    # sampling (빠른 재현용)
    # None이면 전체 사용
    train_sample_n: int | None = None  # 예: 10000, 20000

    # XGBoost params (기본값, 필요시 수정)
    xgb_params: dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "n_estimators": 1000,
                "learning_rate": 0.02,
                "max_depth": 8,
                "colsample_bytree": 0.8,
                "colsample_bylevel": 0.8,
                "gamma": 0.1,
                "reg_lambda": 1.0,
                "reg_alpha": 5.0,
                "random_state": self.random_state,
                "eval_metric": "auc",
            }
