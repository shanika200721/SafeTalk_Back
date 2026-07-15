"""Train-only preprocessing for Speech acoustic features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.training.speech.schemas import SpeechPreprocessorResult


def _finite_numeric_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    frame = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return frame


def detect_constant_features(train_df: pd.DataFrame, features: list[str]) -> list[str]:
    frame = _finite_numeric_frame(train_df, features)
    return [column for column in features if frame[column].nunique(dropna=False) <= 1]


def validate_numeric_features(df: pd.DataFrame, features: list[str]) -> None:
    missing = sorted(set(features) - set(df.columns))
    if missing:
        raise ValueError(f"missing Speech feature columns: {missing}")
    frame = _finite_numeric_frame(df, features)
    bad = [column for column in features if np.isinf(frame[column].to_numpy(dtype=float, na_value=np.nan)).any()]
    if bad:
        raise ValueError(f"infinite Speech feature values in columns: {bad}")


def build_speech_preprocessor(
    train_df: pd.DataFrame,
    features: list[str],
    *,
    estimator_type: str,
    scale_numeric: bool | None = None,
) -> SpeechPreprocessorResult:
    validate_numeric_features(train_df, features)
    frame = _finite_numeric_frame(train_df, features)
    removed = detect_constant_features(train_df, features)
    retained = [feature for feature in features if feature not in removed]
    if not retained:
        raise ValueError("all Speech features are constant in training data")
    if scale_numeric is None:
        scale_numeric = estimator_type in {"logistic_regression", "linear_svm", "rbf_svm"}
    missing_report = {
        "train_missing_by_feature": {feature: int(frame[feature].isna().sum()) for feature in retained},
        "imputation_strategy": "median",
        "fit_scope": "train_only",
    }
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[("numeric", Pipeline(steps), retained)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    preprocessor.fit(frame[retained])
    names = [str(name) for name in preprocessor.get_feature_names_out()]
    return SpeechPreprocessorResult(
        preprocessor=preprocessor,
        feature_names=names,
        removed_constant_features=removed,
        missing_value_report=missing_report,
        scale_numeric=bool(scale_numeric),
    )


def transform_speech_features(preprocessor, df: pd.DataFrame, features: list[str]):
    frame = _finite_numeric_frame(df, features)
    matrix = preprocessor.transform(frame)
    if not np.isfinite(matrix).all():
        raise ValueError("Speech preprocessing produced NaN or infinite values")
    return matrix
