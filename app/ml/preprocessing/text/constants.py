"""Constants for research-only text dataset preprocessing."""

from __future__ import annotations

TEXT_PREPROCESSING_VERSION = "1.0.0"
TEXT_FEATURE_SCHEMA_VERSION = "1.0.0"
TEXT_LABEL_MAPPING_VERSION = "1.0.0"
TEXT_PRIVACY_RULESET_VERSION = "1.0.0"

DATASET_NAME = "mental-health-text"
DATASET_VERSION = "v1"
RECORD_ID_PREFIX = "mental-health-text-v1"

AUTHORITATIVE_SOURCE_FILE = "mental_heath_unbanlanced.csv"
FEATURE_ENGINEERED_SOURCE_FILE = "mental_heath_feature_engineered.csv"
REFERENCE_TEST_SOURCE_FILE = "mental_health_combined_test.csv"

TEXT_COLUMN = "text"
LABEL_COLUMN = "status"
OPTIONAL_ID_COLUMN = "Unique_ID"

RAW_SOURCE_COLUMNS = (OPTIONAL_ID_COLUMN, TEXT_COLUMN, LABEL_COLUMN)
REFERENCE_SOURCE_COLUMNS = (TEXT_COLUMN, LABEL_COLUMN)
CONFIRMED_LABELS = ("Anxiety", "Depression", "Normal", "Suicidal")

PRIVACY_PLACEHOLDERS = ("<URL>", "<EMAIL>", "<PHONE>", "<USER>", "<IP>", "<COMMUNITY>")

CANONICAL_COLUMNS = (
    "record_id",
    "normalized_text",
    "canonical_label",
    "source_name",
    "text_hash",
    "url_count",
    "email_count",
    "phone_count",
    "username_count",
    "ip_address_count",
    "community_count",
    "possible_person_identifier_count",
    "character_count",
    "word_count",
    "line_count",
    "placeholder_count",
    "validation_warnings",
)

ENGINEERED_LEAKAGE_CANDIDATES = (
    "text_length",
    "word_count",
    "num_urls",
    "num_emojis",
    "num_special_chars",
    "num_excess_punct",
    "avg_word_length",
    "stopword_ratio",
    "type_token_ratio",
    "polarity",
    "subjectivity",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
    "has_suicidal_keyword",
    "has_stress_keyword",
    "has_help_keyword",
)
