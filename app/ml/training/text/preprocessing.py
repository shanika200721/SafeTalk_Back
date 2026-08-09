"""Training-only TF-IDF vectorization for Text baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from app.ml.common import hashing
from app.ml.training.artifacts import prevent_overwrite
from app.ml.training.text.constants import TEXT_VECTORIZER_VERSION
from app.ml.training.text.schemas import TextVectorizerResult


def _tuple_range(value) -> tuple[int, int]:
    if value is None:
        return (1, 1)
    return (int(value[0]), int(value[1]))


def _word_vectorizer(config: Mapping[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=_tuple_range(config.get("ngram_range", [1, 2])),
        min_df=config.get("min_df", 2),
        max_df=config.get("max_df", 0.95),
        max_features=config.get("max_features", 20000),
        sublinear_tf=bool(config.get("sublinear_tf", True)),
        strip_accents=None,
        lowercase=False,
        token_pattern=r"(?u)<[^>\s]+>|[\w'-]+",
    )


def _char_vectorizer(config: Mapping[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=_tuple_range(config.get("ngram_range", [3, 5])),
        min_df=config.get("min_df", 2),
        max_df=config.get("max_df", 1.0),
        max_features=config.get("max_features", 30000),
        sublinear_tf=bool(config.get("sublinear_tf", True)),
        lowercase=False,
    )


def create_text_vectorizer(config: Mapping[str, Any]):
    kind = str(config.get("kind", "word"))
    if kind == "word":
        return _word_vectorizer(config)
    if kind == "char":
        return _char_vectorizer(config)
    if kind == "combined":
        word_config = dict(config.get("word") or {})
        char_config = dict(config.get("char") or {})
        return FeatureUnion(
            [
                ("word", _word_vectorizer(word_config)),
                ("char", _char_vectorizer(char_config)),
            ],
            n_jobs=1,
        )
    raise ValueError(f"unsupported Text vectorizer kind: {kind}")


def get_text_feature_names(vectorizer) -> list[str]:
    if hasattr(vectorizer, "get_feature_names_out"):
        return [str(name) for name in vectorizer.get_feature_names_out()]
    names: list[str] = []
    for prefix, transformer in vectorizer.transformer_list:
        names.extend(f"{prefix}__{name}" for name in transformer.get_feature_names_out())
    return names


def hash_vocabulary(feature_names: list[str]) -> str:
    return hashing.hash_json_data({"feature_names": feature_names})


def fit_text_vectorizer(train_texts, config: Mapping[str, Any]) -> TextVectorizerResult:
    vectorizer = create_text_vectorizer(config)
    vectorizer.fit([str(value) for value in train_texts])
    feature_names = get_text_feature_names(vectorizer)
    return TextVectorizerResult(
        vectorizer=vectorizer,
        feature_names=feature_names,
        feature_count=len(feature_names),
        vocabulary_hash=hash_vocabulary(feature_names),
        vectorizer_config={**dict(config), "vectorizer_version": TEXT_VECTORIZER_VERSION},
    )


def transform_text_features(vectorizer, texts):
    return vectorizer.transform([str(value) for value in texts])


def save_text_vectorizer(vectorizer, path: str | Path, *, overwrite: bool = False) -> Path:
    output_path = Path(path)
    prevent_overwrite(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.tmp")
    joblib.dump(vectorizer, tmp)
    tmp.replace(output_path)
    return output_path


def vectorizer_summary(result: TextVectorizerResult, *, expose_vocabulary: bool = False) -> dict[str, Any]:
    payload = {
        "vectorizer_version": TEXT_VECTORIZER_VERSION,
        "vectorizer_config": result.vectorizer_config,
        "feature_count": result.feature_count,
        "vocabulary_hash": result.vocabulary_hash,
        "complete_vocabulary_exposed": False,
    }
    if expose_vocabulary:
        payload["complete_vocabulary_exposed"] = True
        payload["vocabulary"] = result.feature_names
    return payload
