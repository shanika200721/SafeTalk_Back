"""Policy loading for Phase 3G facial duplicate remediation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ml.common.hashing import hash_json_data
from app.ml.remediation.face.constants import FACE_DEFAULT_RANDOM_SEED, FACE_DUPLICATE_POLICY_VERSION


DEFAULT_FACE_DUPLICATE_POLICY: dict[str, Any] = {
    "policy_version": FACE_DUPLICATE_POLICY_VERSION,
    "exact_duplicate_policy": {
        "definition": "Records with identical SHA-256 image_hash values in the canonical face manifest.",
        "uses_image_content_beyond_hash": False,
    },
    "same_label_duplicate_policy": {
        "action": "keep_one_deterministic_representative_exclude_other_copies",
        "record_exclusions": True,
    },
    "cross_label_conflict_policy": {
        "action": "quarantine_entire_hash_group",
        "automatic_label_resolution": False,
        "majority_vote_allowed": False,
    },
    "cross_split_duplicate_policy": {
        "action": "ignore_original_split_for_v2_keep_one_hash_in_one_revised_split",
        "original_split_is_metadata_only": True,
    },
    "representative_selection_rule": [
        "prefer readable valid records",
        "prefer lexicographically smallest repository-relative path",
        "then prefer lexicographically smallest record_id",
        "do not prefer original train over original test",
        "do not use label frequency",
    ],
    "original_split_treatment": "metadata_only_not_final_leakage_safe_split",
    "perceptual_near_duplicate_policy": {
        "diagnostic_only": True,
        "automatic_exclusion": False,
        "threshold": 6,
        "method": "Pillow average hash with bounded bucket comparison",
    },
    "class_balance_safeguards": {
        "strategy": "deterministic_stratified_by_canonical_emotion_label",
        "minimum_records_per_class_per_split": 1,
    },
    "random_seed": FACE_DEFAULT_RANDOM_SEED,
    "deterministic_tie_break_rule": "readable desc, repository-relative path asc, record_id asc",
    "minimum_records_per_class_per_split": 1,
    "notes": [
        "No raw image is deleted or modified.",
        "Cross-label exact conflicts are not relabeled.",
        "Subject-independent splitting is impossible because subject identifiers are unavailable.",
    ],
}


def load_duplicate_policy(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_FACE_DUPLICATE_POLICY))
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("duplicate policy config must be a JSON object")
    if payload.get("policy_version") != FACE_DUPLICATE_POLICY_VERSION:
        raise ValueError("unsupported face duplicate policy version")
    return payload


def hash_duplicate_policy_payload(policy: dict[str, Any]) -> str:
    return hash_json_data(policy)

