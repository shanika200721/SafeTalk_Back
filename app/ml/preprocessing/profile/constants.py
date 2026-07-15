"""Constants for Student Profile canonical preprocessing."""

from __future__ import annotations

PROFILE_PREPROCESSING_VERSION = "1.0.0"
PROFILE_FEATURE_SCHEMA_VERSION = "1.0.0"
PROFILE_MAPPING_VERSION = "1.0.0"

DATASET_NAME = "student-profile"
DATASET_VERSION = "v1"
RECORD_ID_PREFIX = "student-profile-v1"

TIMESTAMP_COLUMN = "Timestamp"
TARGET_COLUMN = "Do you have Depression?"
TREATMENT_COLUMN = "Did you seek any specialist for a treatment?"

SOURCE_COLUMNS = (
    TIMESTAMP_COLUMN,
    "Choose your gender",
    "Age",
    "What is your course?",
    "Your current year of Study",
    "What is your CGPA?",
    "Marital status",
    TARGET_COLUMN,
    "Do you have Anxiety?",
    "Do you have Panic attack?",
    TREATMENT_COLUMN,
)

CANONICAL_COLUMNS = {
    TIMESTAMP_COLUMN: "source_timestamp",
    "Choose your gender": "gender",
    "Age": "age",
    "What is your course?": "course",
    "Your current year of Study": "year_of_study",
    "What is your CGPA?": "cgpa_band",
    "Marital status": "marital_status",
    TARGET_COLUMN: "target_depression",
    "Do you have Anxiety?": "self_reported_anxiety",
    "Do you have Panic attack?": "self_reported_panic_attack",
    TREATMENT_COLUMN: "sought_specialist_treatment",
}

TARGET_CANONICAL_COLUMN = CANONICAL_COLUMNS[TARGET_COLUMN]
METADATA_COLUMNS = (TIMESTAMP_COLUMN,)
SENSITIVE_CONTEXT_COLUMNS = (
    "Choose your gender",
    "Age",
    "What is your course?",
    "What is your CGPA?",
    "Marital status",
)
LEAKAGE_CANDIDATE_COLUMNS = (TIMESTAMP_COLUMN, TREATMENT_COLUMN)
DEFAULT_EXCLUDED_SOURCE_COLUMNS = (TIMESTAMP_COLUMN, TREATMENT_COLUMN)

BINARY_LABEL_VALUES = {
    "yes": "yes",
    "y": "yes",
    "no": "no",
    "n": "no",
}

TARGET_ALLOWED_VALUES = ("no", "yes")
GENDER_ALLOWED_VALUES = ("female", "male")
YEAR_ALLOWED_VALUES = ("year 1", "year 2", "year 3", "year 4")
CGPA_ALLOWED_VALUES = (
    "0 - 1.99",
    "2.00 - 2.49",
    "2.50 - 2.99",
    "3.00 - 3.49",
    "3.50 - 4.00",
)

BASELINE_FEATURE_SOURCE_COLUMNS = (
    "Your current year of Study",
    "Do you have Anxiety?",
    "Do you have Panic attack?",
)

OPTIONAL_SENSITIVE_CONTEXT_SOURCE_COLUMNS = SENSITIVE_CONTEXT_COLUMNS

PRODUCTION_PROFILE_FIELDS = (
    "gpa",
    "repeated_subjects",
    "attendance",
    "academic_difficulty",
    "family_relationship_score",
    "income_level",
    "parents_employment",
    "family_support",
    "living_arrangement",
    "employment_status",
    "financial_stress",
    "communication_skills",
    "social_connection",
    "sleep_pattern",
    "exercise_frequency",
    "substance_use",
    "department",
    "year_of_study",
)
