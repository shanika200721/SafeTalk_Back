"""Constants for Phase 2 validation and readiness reporting."""

from __future__ import annotations

from pathlib import Path


PHASE2_VALIDATION_VERSION = "1.0.0"
READINESS_POLICY_VERSION = "1.0.0"

KNOWN_MODALITIES = ("dass21", "profile", "mood", "text", "speech", "face", "behavioral")
FUSION_MODALITY = "fusion"

DEFAULT_REPORT_SUBDIR = Path("phase2_validation")
DEFAULT_VALIDATION_OUTPUT_FILES = {
    "report_json": "phase2_validation_report.json",
    "report_markdown": "phase2_validation_report.md",
    "readiness_matrix": "phase2_readiness_matrix.csv",
    "artifact_inventory": "phase2_artifact_inventory.json",
    "blockers": "phase2_blockers.json",
    "next_actions": "phase2_next_actions.json",
}

MODEL_ARTIFACT_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".h5",
    ".joblib",
    ".keras",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".sav",
}
MODEL_ARTIFACT_NAMES = {
    "model",
    "models",
    "checkpoint",
    "checkpoints",
    "encoder",
    "encoders",
    "scaler",
    "scalers",
    "tokenizer",
    "tfidf",
    "vocabulary",
}
SPLIT_MANIFEST_HINTS = ("split", "train", "validation", "test")
DATABASE_WRITE_HINTS = ("migration", "alembic", "postgres", "database_write", "db_write")
PREDICTION_OUTPUT_HINTS = ("prediction", "predictions", "inference", "alert")

TEXT_PRIVACY_PATTERNS = {
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "phone": r"(?<![\w])(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-])\d{3}[\s.-]\d{4}(?!\w)",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "url": r"https?://\S+",
    "windows_absolute_path": r"\b[A-Za-z]:[\\/][^\s\"']+",
    "posix_absolute_path": r"(?<![A-Za-z0-9_])/(?:home|Users|mnt|var|tmp|Final Dataset)[^\s\"']*",
}

RAW_DATA_HINTS = (
    "raw_text",
    "message_text",
    "transcript",
    "image_bytes",
    "audio_bytes",
    "telemetry_payload",
    "participant_name",
    "email",
    "phone",
)
