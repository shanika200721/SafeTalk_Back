"""Retention policy validation helpers."""

from __future__ import annotations

from typing import Any, Dict, List

REQUIRED_RETENTION_CATEGORIES = (
    "raw_audio",
    "raw_face_images",
    "text",
    "questionnaire_responses",
    "behavioral_telemetry",
    "derived_features",
    "pseudonymous_ids",
    "linkage_files",
    "model_outputs",
    "audit_logs",
)


def validate_retention_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    categories = policy.get("categories", {})
    for category in REQUIRED_RETENTION_CATEGORIES:
        if category not in categories:
            errors.append(f"missing retention category: {category}")
            continue
        item = categories[category]
        for field in ("retention_period", "access", "withdrawal_handling", "export_restrictions"):
            if not item.get(field):
                errors.append(f"{category}: missing {field}")
    if categories.get("model_outputs", {}).get("retention_period") != "not collected in Phase 4A":
        errors.append("model_outputs must be marked not collected in Phase 4A")
    return {"valid": not errors, "errors": errors}
