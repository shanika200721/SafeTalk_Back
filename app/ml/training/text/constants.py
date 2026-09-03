"""Constants for the Phase 3D Text classification baseline."""

TEXT_MODEL_FAMILY_VERSION = "1.0.0"
TEXT_BASELINE_EXPERIMENT_VERSION = "1.0.0"
TEXT_VECTORIZER_VERSION = "1.0.0"

TEXT_MODALITY = "text"
TEXT_DATASET_NAME = "mental-health-text"
TEXT_DATASET_VERSION = "v1"
TEXT_TEXT_COLUMN = "normalized_text"
TEXT_TARGET_COLUMN = "canonical_label"
TEXT_RECORD_ID_COLUMN = "record_id"
TEXT_HASH_COLUMN = "text_hash"

TEXT_LABELS = ["anxiety", "depression", "normal", "suicidal"]
SUICIDAL_LABEL = "suicidal"
NORMAL_LABEL = "normal"
DEPRESSION_LABEL = "depression"

TEXT_FEATURE_SET = "normalized_text_tfidf"

PROHIBITED_TEXT_FEATURES = {
    TEXT_RECORD_ID_COLUMN,
    TEXT_HASH_COLUMN,
    "source_name",
    "source_file",
    "source_row_index",
    "original_id",
    "Unique_ID",
    "split",
    TEXT_TARGET_COLUMN,
    "status",
    "url_count",
    "email_count",
    "phone_count",
    "username_count",
    "ip_address_count",
    "community_count",
    "possible_person_identifier_count",
    "placeholder_count",
}

DEFAULT_CANONICAL_DATA = "generated/preprocessing/text/v1/canonical_text.csv"
DEFAULT_FEATURE_SCHEMA = "generated/preprocessing/text/v1/text_feature_schema.json"
DEFAULT_PREPROCESSING_REPORT = "generated/preprocessing/text/v1/text_preprocessing_report.json"
DEFAULT_RECORD_MANIFEST = "generated/preprocessing/text/v1/text_record_manifest.json"
DEFAULT_DUPLICATE_MANIFEST = "generated/preprocessing/text/v1/text_duplicate_manifest.json"
DEFAULT_CONFLICT_QUARANTINE = "generated/preprocessing/text/v1/text_conflict_quarantine.csv"
DEFAULT_SOURCE_OVERLAP_REPORT = "generated/preprocessing/text/v1/text_source_overlap_report.json"
DEFAULT_SPLIT_MANIFEST = "generated/manifests/splits/text/v1/text_split_manifest.json"
DEFAULT_SPLIT_ASSIGNMENTS = "generated/manifests/splits/text/v1/text_split_assignments.csv"
DEFAULT_SOURCE_FINGERPRINT = "generated/manifests/fingerprints/mental-health-text-v1.json"
DEFAULT_REPORT_DIR = "generated/reports/text_baseline/v1"
DEFAULT_MODEL_ROOT = "ml_models"

REQUIRED_MODEL_CARD_DISCLAIMER = (
    "This model is a research prototype and is not a clinical diagnostic or "
    "autonomous suicide-prevention system."
)

