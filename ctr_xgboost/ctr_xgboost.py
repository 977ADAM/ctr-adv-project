import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score
from category_encoders import CatBoostEncoder
from xgboost import XGBClassifier

np.random.seed(42)



@dataclass
class Config:
    # paths
    train_path: str = "./data/dataset_train.csv"
    test_path: str = "./data/dataset_test.csv"
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

cfg = Config()

df = pd.read_csv(cfg.train_path)
df_test = pd.read_csv(cfg.test_path)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values("DateTime")
    dt = df["DateTime"]

    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["dayofweek"] = dt.dt.dayofweek
    df["hour"] = dt.dt.hour
    df["weekofyear"] = dt.dt.isocalendar().week.astype("int16")

    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype("int8")

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["dow_sin"] = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dayofweek"]/7)

    return df



df = add_time_features(df)
df_test = add_time_features(df_test)


df_test = df_test.drop(columns=["session_id","DateTime"])

X = df.drop(columns=["is_click", "session_id", "DateTime"])
y = df["is_click"]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []

for fold, (train_idx, valid_idx) in enumerate(
    sgkf.split(X, y, groups=df["user_id"])
):

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = XGBClassifier(
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42,
        
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_valid)[:, 1]

    ll = log_loss(y_valid, proba)
    auc = roc_auc_score(y_valid, proba)
    pr_auc = average_precision_score(y_valid, proba)
    aucs.append(auc)

    print(f"===============Fold {fold+1} ROC AUC: {auc:.6f} PR AUC: {pr_auc:.6f} LogLoss: {ll:.6f} ================")
    
# =================================================================================================

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

features = numeric_features + categorical_features

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder",  CatBoostEncoder(sigma=0.01))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(

    n_estimators=4000,
    learning_rate=0.02,


    max_depth=8,
    min_child_weight=5,

    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.8,

    gamma=0.1,

    reg_alpha=1.0,
    reg_lambda=5.0,

    scale_pos_weight=scale_pos_weight,

    eval_metric="auc",
    tree_method="hist",
    
    random_state=42,
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_valid)[:,1]

ll = log_loss(y_valid, proba)
auc = roc_auc_score(y_valid, proba)
pr_auc = average_precision_score(y_valid, proba)


print(f"===============PR AUC: {pr_auc} =====================")
print(f"===============ROC AUC: {auc} =====================")
print(f"===============LogLoss: {ll} =====================")


joblib.dump(pipeline, "ctr_model.pkl")

pipeline = joblib.load("ctr_model.pkl")

df_test = df_test[features]
pred = pipeline.predict_proba(df_test)[:,1]