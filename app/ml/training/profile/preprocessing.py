"""Training-only preprocessing for Profile baseline models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.training.artifacts import prevent_overwrite
from app.ml.training.profile.schemas import ProfilePreprocessorResult


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _split_feature_types(df: pd.DataFrame, features: Iterable[str]) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            numeric.append(feature)
        else:
            categorical.append(feature)
    return numeric, categorical


def detect_all_null_features(df: pd.DataFrame, features: Iterable[str]) -> list[str]:
    return [feature for feature in features if df[feature].isna().all()]


def detect_constant_features(df: pd.DataFrame, features: Iterable[str]) -> list[str]:
    constants: list[str] = []
    for feature in features:
        if df[feature].nunique(dropna=False) <= 1:
            constants.append(feature)
    return constants


def build_profile_preprocessor(
    train_df: pd.DataFrame,
    features: list[str],
    *,
    estimator_type: str,
    scale_numeric: bool | None = None,
) -> ProfilePreprocessorResult:
    X_train = train_df[features].copy()
    numeric_features, categorical_features = _split_feature_types(X_train, features)
    if scale_numeric is None:
        scale_numeric = estimator_type == "logistic_regression"

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)
    preprocessor.fit(X_train)
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return ProfilePreprocessorResult(
        preprocessor=preprocessor,
        feature_names=names,
        feature_count=len(names),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        constant_features=detect_constant_features(X_train, features),
        all_null_features=detect_all_null_features(X_train, features),
    )


def transform_profile_features(preprocessor, df: pd.DataFrame, features: list[str]):
    return preprocessor.transform(df[features].copy())


def save_profile_preprocessor(preprocessor, path: str | Path, *, overwrite: bool = False) -> Path:
    output_path = Path(path)
    prevent_overwrite(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.tmp")
    joblib.dump(preprocessor, tmp)
    tmp.replace(output_path)
    return output_path
