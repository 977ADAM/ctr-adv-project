from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import CatBoostEncoder
from sklearn.pipeline import Pipeline

from config import Config

cfg = Config()

def prepro():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder",  CatBoostEncoder(random_state=cfg.random_state, sigma=0.01))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, cfg.numeric_features),
            ("cat", categorical_pipeline, cfg.categorical_features)
        ]
    )

    return preprocessor