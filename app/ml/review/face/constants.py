"""Constants for Phase 3H face human-review workflows."""

from __future__ import annotations

FACE_REVIEW_WORKFLOW_VERSION = "1.0.0"
FACE_REVIEW_DECISION_SCHEMA_VERSION = "1.0.0"
FACE_RECONCILIATION_POLICY_VERSION = "1.0.0"

FACE_REVIEW_ITEM_TYPES = ("cross_label_conflict", "perceptual_duplicate_candidate")
FACE_REVIEW_CONFIDENCE_LEVELS = ("low", "medium", "high")
FACE_FINAL_ACTIONS = (
    "keep_quarantined",
    "retain_existing_representative",
    "restore_record",
    "exclude_all",
    "unresolved",
    "additional_review",
)
FACE_DEFAULT_REVIEW_OUTPUT = "generated/review/face/v1"
FACE_DEFAULT_REVIEW_AUDIT_OUTPUT = "generated/review/face/v1/audit"

FACE_ALLOWED_REVIEW_DECISIONS = (
    "confirm_exact_duplicate",
    "confirm_near_duplicate",
    "not_duplicate",
    "label_conflict_unresolved",
    "source_label_likely_incorrect",
    "ambiguous",
    "corrupted",
    "retain_quarantine",
    "recommend_restore_same_label_representative",
    "requires_additional_review",
)

FACE_DEFAULT_REASON_CODES = (
    "visually_identical",
    "near_identical",
    "not_duplicate",
    "label_conflict",
    "likely_source_label_error",
    "ambiguous_visual_evidence",
    "corrupted_or_unreadable",
    "privacy_or_policy_limit",
    "insufficient_consensus",
    "requires_additional_review",
    "synthetic_smoke_test",
)

