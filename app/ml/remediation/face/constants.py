"""Constants for Phase 3G facial emotion remediation."""

from __future__ import annotations

FACE_REMEDIATION_VERSION = "1.0.0"
FACE_DEDUPLICATED_VIEW_VERSION = "1.0.0"
FACE_REVISED_SPLIT_VERSION = "1.0.0"
FACE_DUPLICATE_POLICY_VERSION = "1.0.0"

FACE_DATASET_NAME = "facial-emotion"
FACE_MODALITY = "face"
FACE_REVISED_SPLIT_ID = "v2"
FACE_REMEDIATION_OUTPUT_VERSION = "v1"
FACE_DEFAULT_RANDOM_SEED = 43107
FACE_SPLIT_NAMES = ("train", "validation", "test")
FACE_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

FACE_REQUIRED_CANONICAL_COLUMNS = (
    "record_id",
    "source_split",
    "original_label",
    "canonical_emotion_label",
    "image_relative_path",
    "image_hash",
    "readable",
)

FACE_DEDUPLICATED_MANIFEST_COLUMNS = (
    "record_id",
    "source_split",
    "original_label",
    "canonical_emotion_label",
    "image_relative_path",
    "image_hash",
    "readable",
    "duplicate_group_id",
    "remediation_action",
    "remediation_policy_version",
)

FACE_DECISION_COLUMNS = (
    "record_id",
    "action",
    "representative_id",
    "group_id",
    "reason",
    "policy_version",
)

