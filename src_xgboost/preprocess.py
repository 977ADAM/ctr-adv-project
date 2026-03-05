from __future__ import annotations

from collections.abc import Sequence

from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, Sequence[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    CatBoostEncoder(
                        random_state=random_state,
                        sigma=0.01,
                        handle_unknown="value",
                        handle_missing="value",
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No transformers configured. Check numeric/categorical feature lists.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
