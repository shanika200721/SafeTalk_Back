"""Policy loading for Phase 3H face review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.hashing import hash_json_data
from app.ml.review.face.constants import (
    FACE_ALLOWED_REVIEW_DECISIONS,
    FACE_DEFAULT_REASON_CODES,
    FACE_RECONCILIATION_POLICY_VERSION,
)


DEFAULT_FACE_REVIEW_POLICY: dict[str, Any] = {
    "policy_version": FACE_RECONCILIATION_POLICY_VERSION,
    "minimum_reviewer_count": 2,
    "double_review_required": True,
    "conflict_resolution_rule": "unanimous_consensus_required_no_majority_vote",
    "allowed_decisions": list(FACE_ALLOWED_REVIEW_DECISIONS),
    "reason_codes": list(FACE_DEFAULT_REASON_CODES),
    "unresolved_policy": "remain_quarantined_or_flagged",
    "restore_policy": {
        "reviewer_consensus_required": True,
        "allowed_for_cross_label_conflicts": True,
        "requires_source_label_invalid_reason": True,
        "no_label_overwrite": True,
    },
    "relabel_policy": {
        "reviewers_may_recommend_label_review": True,
        "reviewers_must_not_overwrite_canonical_labels": True,
        "automatic_relabeling_allowed": False,
    },
    "duplicate_policy": {
        "confirmed_duplicates_grouped_for_future_exclusion": True,
        "automatic_exclusion_allowed": False,
    },
    "perceptual_candidate_policy": {
        "diagnostic_only_until_human_reviewed": True,
        "not_duplicate_keeps_records_unchanged": True,
        "ambiguous_remains_flagged": True,
    },
    "reviewer_privacy_rules": [
        "do not identify people",
        "do not infer race, ethnicity, health condition, identity, suicide risk, or clinical state",
        "inspect only whether images appear visually identical, near-identical, incorrectly labeled, corrupted, or ambiguous",
    ],
    "audit_retention_policy": "append_only_generated_review_audit_no_personal_reviewer_details",
    "deterministic_reconciliation_rule": "sort_by_review_item_id_then_reviewer_alias; unanimous_decision_required",
    "notes": [
        "No raw image files are modified, copied by default, deleted, resized, cropped, or overwritten.",
        "Human review improves data consistency only and cannot establish clinical truth.",
    ],
}


def load_review_policy(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_FACE_REVIEW_POLICY))
    policy_path = Path(path)
    if not policy_path.is_absolute():
        policy_path = paths.get_repository_root() / policy_path
    with policy_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("face review policy must be a JSON object")
    if payload.get("policy_version") != FACE_RECONCILIATION_POLICY_VERSION:
        raise ValueError("unsupported face review policy version")
    return payload


def hash_review_policy(policy: dict[str, Any]) -> str:
    return hash_json_data(policy)
