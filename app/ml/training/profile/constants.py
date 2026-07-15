"""Constants for the Profile depression research baseline."""

PROFILE_MODEL_FAMILY_VERSION = "1.0.0"
PROFILE_BASELINE_EXPERIMENT_VERSION = "1.0.0"

PROFILE_MODALITY = "profile"
PROFILE_DATASET_NAME = "student-profile"
PROFILE_DATASET_VERSION = "v1"
PROFILE_TARGET_COLUMN = "target_depression"
PROFILE_RECORD_ID_COLUMN = "record_id"
PROFILE_POSITIVE_LABEL = "yes"
PROFILE_NEGATIVE_LABEL = "no"

MINIMAL_CONTEXTUAL_FEATURE_SET = "minimal_contextual"
EXTENDED_SELF_REPORT_FEATURE_SET = "extended_self_report"
SENSITIVE_CONTEXT_FEATURE_SET = "sensitive_context_exploratory"

FEATURE_SETS = {
    MINIMAL_CONTEXTUAL_FEATURE_SET: ["year_of_study"],
    EXTENDED_SELF_REPORT_FEATURE_SET: [
        "year_of_study",
        "self_reported_anxiety",
        "self_reported_panic_attack",
    ],
    SENSITIVE_CONTEXT_FEATURE_SET: [
        "year_of_study",
        "age",
        "gender",
        "course",
        "cgpa_band",
        "marital_status",
    ],
}

OUTCOME_LIKE_FEATURES = {"self_reported_anxiety", "self_reported_panic_attack"}
SENSITIVE_CONTEXT_FEATURES = {"age", "gender", "course", "cgpa_band", "marital_status"}
PROHIBITED_FEATURES = {
    PROFILE_TARGET_COLUMN,
    "source_timestamp",
    "Timestamp",
    "sought_specialist_treatment",
    "Did you seek any specialist for a treatment?",
}

DEFAULT_CANONICAL_DATA = "generated/preprocessing/profile/v1/canonical_profile.csv"
DEFAULT_FEATURE_SCHEMA = "generated/preprocessing/profile/v1/profile_feature_schema.json"
DEFAULT_PREPROCESSING_REPORT = "generated/preprocessing/profile/v1/profile_preprocessing_report.json"
DEFAULT_RECORD_MANIFEST = "generated/preprocessing/profile/v1/profile_record_manifest.json"
DEFAULT_SPLIT_MANIFEST = "generated/manifests/splits/profile/v1/profile_split_manifest.json"
DEFAULT_SPLIT_ASSIGNMENTS = "generated/manifests/splits/profile/v1/profile_split_assignments.csv"
DEFAULT_SOURCE_FINGERPRINT = "generated/manifests/fingerprints/student-profile-v1.json"
DEFAULT_REPORT_DIR = "generated/reports/profile_baseline/v1"
DEFAULT_MODEL_ROOT = "ml_models"

REQUIRED_MODEL_CARD_DISCLAIMER = (
    "This model is a research prototype and is not a clinical diagnostic or "
    "autonomous suicide-prevention system."
)

DEFAULT_SECONDARY_METRICS = [
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
]

