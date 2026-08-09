"""Version and policy constants for Phase 4A pilot protocol design."""

from __future__ import annotations

PILOT_PROTOCOL_VERSION = "1.0.0"
PILOT_DATA_SCHEMA_VERSION = "1.0.0"
PILOT_CONSENT_VERSION = "1.0.0"
PILOT_SAFETY_POLICY_VERSION = "1.0.0"
PILOT_RETENTION_POLICY_VERSION = "1.0.0"

SYNTHETIC_MARKER = "PHASE4A_SYNTHETIC_PROTOCOL_ONLY"

MODALITIES = ("dass21", "profile", "mood", "text", "speech", "face", "behavioral")
SEPARATE_CONSENT_MODALITIES = ("speech", "face", "behavioral")
BIOMETRIC_MODALITIES = ("speech", "face")
DEFAULT_DISABLED_REAL_COLLECTION = ("face", "behavioral")

READINESS_STATES = (
    "protocol_draft",
    "governance_review_required",
    "ethics_approval_required",
    "technical_schema_ready",
    "synthetic_validation_complete",
    "real_collection_prohibited",
    "pilot_ready_after_approval",
)

EXPECTED_FINAL_READINESS = (
    "technical_schema_ready",
    "synthetic_validation_complete",
    "ethics_approval_required",
    "governance_review_required",
    "real_collection_prohibited",
)

PROHIBITED_ARTIFACT_TERMS = (
    "modality_prediction",
    "risk_assessment",
    "model_output",
    "prediction",
    "alert",
    "fusion_score",
)
