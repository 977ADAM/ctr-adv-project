import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

np.random.seed(42)

df = pd.read_csv("./data/dataset_train.csv")
df_test = pd.read_csv("./data/dataset_test.csv")


df["DateTime"] = pd.to_datetime(df["DateTime"])
dt = df["DateTime"]
df["year"] = dt.dt.year
df["month"] = dt.dt.month
df["day"] = dt.dt.day
df["dayofweek"] = dt.dt.dayofweek
df["hour"] = dt.dt.hour
df["weekofyear"] = dt.dt.isocalendar().week.astype(np.int16)
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(np.int8)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)


df_test["DateTime"] = pd.to_datetime(df_test["DateTime"])
dt = df_test["DateTime"]
df_test["year"] = dt.dt.year
df_test["month"] = dt.dt.month
df_test["day"] = dt.dt.day
df_test["dayofweek"] = dt.dt.dayofweek
df_test["hour"] = dt.dt.hour
df_test["weekofyear"] = dt.dt.isocalendar().week.astype(np.int16)
df_test["is_weekend"] = df_test["dayofweek"].isin([5,6]).astype(np.int8)
df_test["hour_sin"] = np.sin(2*np.pi*df_test["hour"]/24)
df_test["hour_cos"] = np.cos(2*np.pi*df_test["hour"]/24)
df_test["dow_sin"] = np.sin(2*np.pi*df_test["dayofweek"]/7)
df_test["dow_cos"] = np.cos(2*np.pi*df_test["dayofweek"]/7)

df_test = df_test.drop(columns=["session_id","DateTime"])

X = df.drop(columns=["is_click", "session_id", "DateTime"])
y = df["is_click"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

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
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

scale_pos_weight = (y == 0).sum() / (y == 1).sum()

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_valid)[:,1]

auc = roc_auc_score(y_valid, proba)
print("ROC AUC:", auc)

joblib.dump(pipeline, "ctr_model.pkl")

pipeline = joblib.load("ctr_model.pkl")

df_test = df_test[features]
pred = pipeline.predict_proba(df_test)[:,1]