"""Constants for read-only facial emotion preprocessing."""

from __future__ import annotations

FACE_PREPROCESSING_VERSION = "1.0.0"
FACE_FEATURE_SCHEMA_VERSION = "1.0.0"
FACE_LABEL_MAPPING_VERSION = "1.0.0"
FACE_IMAGE_POLICY_VERSION = "1.0.0"

DATASET_NAME = "facial-emotion"
DATASET_VERSION = "v1"

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PREDEFINED_SPLITS = ("train", "test")
CANONICAL_EMOTION_LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

RECORD_ID_PREFIX = "face-v1-rec"
SAFE_SUBJECT_KEY_PREFIX = "face-v1-sub"

FACE_STATISTIC_COLUMNS = (
    "mean_intensity",
    "std_intensity",
    "contrast",
    "edge_density",
    "entropy",
)

