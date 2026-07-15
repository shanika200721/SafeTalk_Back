"""Future text feature extraction contracts without fitting anything."""

from __future__ import annotations

from typing import List, Optional

from pydantic.v1 import BaseModel, Field, root_validator, validator

from app.ml.common.schemas import FeatureDefinition, FeatureSchema, Modality
from app.ml.preprocessing.text.constants import DATASET_NAME, DATASET_VERSION, TEXT_FEATURE_SCHEMA_VERSION, TEXT_PREPROCESSING_VERSION


class TextFeatureExtractionConfig(BaseModel):
    extractor_name: str
    enabled: bool = False
    ngram_range: Optional[List[int]] = None
    analyzer: Optional[str] = None
    max_features: Optional[int] = None
    pretrained_model_name: Optional[str] = None
    fit_allowed: bool = False
    download_allowed: bool = False

    class Config:
        extra = "forbid"

    @validator("extractor_name")
    def non_blank(cls, value: str) -> str:
        if not str(value).strip():
            raise ValueError("extractor_name cannot be blank")
        return str(value).strip()

    @root_validator
    def no_fitting_or_downloads(cls, values):
        if values.get("fit_allowed"):
            raise ValueError("Text feature config is schema-only in this phase; fitting is not allowed")
        if values.get("download_allowed"):
            raise ValueError("Transformer/tokenizer downloads are not allowed in this phase")
        return values


def default_future_feature_contracts() -> list[TextFeatureExtractionConfig]:
    return [
        TextFeatureExtractionConfig(extractor_name="tfidf_word_ngrams", ngram_range=[1, 2], analyzer="word", max_features=None),
        TextFeatureExtractionConfig(extractor_name="tfidf_character_ngrams", ngram_range=[3, 5], analyzer="char", max_features=None),
        TextFeatureExtractionConfig(extractor_name="transformer_tokenization", pretrained_model_name=None),
    ]


def build_text_feature_schema(*, dataset_name: str = DATASET_NAME, dataset_version: str = DATASET_VERSION) -> FeatureSchema:
    features = [
        FeatureDefinition(
            name="normalized_text",
            dtype="string",
            description="Privacy-safe normalized text; future training may derive features from this field only.",
            source_columns=["text"],
            nullable=False,
            preprocessing_step="normalize_text; no tokenization or model fitting",
        ),
        FeatureDefinition(
            name="character_count",
            dtype="integer",
            description="Metadata length count after privacy-safe normalization.",
            source_columns=["text"],
            nullable=False,
            minimum=0,
            preprocessing_step="metadata_count",
        ),
        FeatureDefinition(
            name="word_count",
            dtype="integer",
            description="Whitespace-delimited word count after privacy-safe normalization.",
            source_columns=["text"],
            nullable=False,
            minimum=0,
            preprocessing_step="metadata_count",
        ),
        FeatureDefinition(
            name="line_count",
            dtype="integer",
            description="Line count after line-break normalization.",
            source_columns=["text"],
            nullable=False,
            minimum=0,
            preprocessing_step="metadata_count",
        ),
        FeatureDefinition(
            name="placeholder_count",
            dtype="integer",
            description="Count of privacy placeholder tokens in normalized text.",
            source_columns=["text"],
            nullable=False,
            minimum=0,
            preprocessing_step="privacy_replacement",
        ),
    ]
    return FeatureSchema(
        schema_name="text-canonical-feature-contract",
        feature_schema_version=TEXT_FEATURE_SCHEMA_VERSION,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        preprocessing_version=TEXT_PREPROCESSING_VERSION,
        modality=Modality.TEXT,
        features=features,
        target_columns=["canonical_label"],
        excluded_columns=["record_id", "text_hash", "source_name", "source_file", "source_row_index", "original_id", "Unique_ID", "status"],
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        notes="Schema records canonical text and metadata only. TF-IDF, vocabularies, sparse matrices, and transformer token IDs are not created in Phase 2.",
    )
