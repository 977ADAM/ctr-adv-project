from dataclasses import dataclass

@dataclass
class Config:
    # paths
    train_path: str = "data/train.parquet"
    test_path: str = "data/test.parquet"
    output_dir: str = "outputs"
    submission_name: str = "submission_xgb.csv"

    # columns
    target_col: str = "clicked"
    id_col: str = "ID"
    seq_col: str = "seq"

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
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "learning_rate": 0.05,
                "max_depth": 7,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_lambda": 1.2,
                "reg_alpha": 0.4,
                "n_estimators": 800,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
