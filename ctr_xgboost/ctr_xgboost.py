import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from category_encoders import CatBoostEncoder, TargetEncoder

from xgboost import XGBClassifier

np.random.seed(42)

df = pd.read_csv("./data/dataset_train.csv")
df_test = pd.read_csv("./data/dataset_test.csv")


def oof_ctr_feature(df, col, target, n_splits=5):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof = np.zeros(len(df))

    global_ctr = df[target].mean()

    for train_idx, valid_idx in kf.split(df):

        train_fold = df.iloc[train_idx]
        valid_fold = df.iloc[valid_idx]

        ctr = train_fold.groupby(col)[target].mean()

        oof[valid_idx] = valid_fold[col].map(ctr)

    oof = pd.Series(oof, index=df.index)

    oof.fillna(global_ctr, inplace=True)

    return oof


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

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

train_df = X_train.copy()
train_df["is_click"] = y_train

X_train["user_ctr"] = oof_ctr_feature(
    train_df,
    col="user_id",
    target="is_click"
)

X_train["product_ctr"] = oof_ctr_feature(
    train_df,
    col="product",
    target="is_click"
)
train_df["user_product"] = (
    train_df["user_id"].astype(str)
    + "_"
    + train_df["product"].astype(str)
)

X_train["user_product_ctr"] = oof_ctr_feature(
    train_df,
    col="user_product",
    target="is_click"
)

user_ctr = train_df.groupby("user_id")["is_click"].mean()

product_ctr = train_df.groupby("product")["is_click"].mean()

user_product_ctr = train_df.groupby("user_product")["is_click"].mean()


X_valid["user_ctr"] = X_valid["user_id"].map(user_ctr)

X_valid["product_ctr"] = X_valid["product"].map(product_ctr)

X_valid["user_product"] = (
    X_valid["user_id"].astype(str)
    + "_"
    + X_valid["product"].astype(str)
)

X_valid["user_product_ctr"] = X_valid["user_product"].map(user_product_ctr)

# =================================================================================================

df_test["user_ctr"] = df_test["user_id"].map(user_ctr)

df_test["product_ctr"] = df_test["product"].map(product_ctr)

df_test["user_product"] = (
    df_test["user_id"].astype(str)
    + "_"
    + df_test["product"].astype(str)
)

df_test["user_product_ctr"] = df_test["user_product"].map(user_product_ctr)

fill_value = y_train.mean()

for col in ["user_ctr", "product_ctr", "user_product_ctr"]:
    X_train[col].fillna(fill_value, inplace=True)
    X_valid[col].fillna(fill_value, inplace=True)
    df_test[col].fillna(fill_value, inplace=True)

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
    "user_ctr",
    "product_ctr",
    "user_product_ctr"
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
    ("encoder",  TargetEncoder(smoothing=10))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    tree_method="auto",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
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