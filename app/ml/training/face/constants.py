"""Constants for Phase 3I Face research baselines."""

from __future__ import annotations

FACE_MODEL_FAMILY_VERSION = "1.0.0"
FACE_BASELINE_EXPERIMENT_VERSION = "1.0.0"
FACE_IMAGE_PIPELINE_VERSION = "1.0.0"

FACE_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
FACE_SPLITS = ("train", "validation", "test")
FACE_IMAGE_SIZE = (48, 48)
FACE_FLATTENED_PIXEL_COUNT = 2304

DEFAULT_DEDUPLICATED_MANIFEST = "generated/remediation/face/v1/face_deduplicated_manifest.csv"
DEFAULT_SPLIT_MANIFEST = "generated/manifests/splits/face/v2/face_split_manifest.json"
DEFAULT_SPLIT_ASSIGNMENTS = "generated/manifests/splits/face/v2/face_split_assignments.csv"
DEFAULT_QUARANTINE = "generated/remediation/face/v1/face_cross_label_quarantine.json"
DEFAULT_DUPLICATE_DECISIONS = "generated/remediation/face/v1/face_remediation_decisions.csv"
DEFAULT_SOURCE_FINGERPRINT = "generated/manifests/fingerprints/face/facial-emotion-v1.json"
DEFAULT_FEATURE_SCHEMA = "generated/preprocessing/face/v1/face_feature_schema.json"
DEFAULT_REPORT_DIR = "generated/reports/face_baseline/v1"
DEFAULT_MODEL_ROOT = "ml_models"

REQUIRED_MODEL_CARD_DISCLAIMER = (
    "This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system."
)

FORBIDDEN_PREDICTIVE_FIELDS = {
    "record_id",
    "source_split",
    "original_split",
    "original_label",
    "canonical_emotion_label",
    "image_relative_path",
    "image_hash",
    "duplicate_group_id",
    "remediation_action",
    "remediation_policy_version",
}

