"""Pilot modality policy helpers."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from app.ml.pilot.constants import BIOMETRIC_MODALITIES, DEFAULT_DISABLED_REAL_COLLECTION, MODALITIES


def modality_matrix(scope: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for modality in MODALITIES:
        item = scope.get("modalities", {}).get(modality, {})
        rows.append(
            {
                "modality": modality,
                "enabled_for_protocol_design": bool(item.get("enabled_for_protocol_design")),
                "enabled_for_real_collection": bool(item.get("enabled_for_real_collection")),
                "collection_frequency": item.get("collection_frequency", ""),
                "consent_required": bool(item.get("consent_required", True)),
                "optional_or_required": item.get("optional_or_required", "optional"),
                "raw_data_retention": item.get("raw_data_retention", ""),
                "derived_feature_retention": item.get("derived_feature_retention", ""),
                "storage_duration": item.get("storage_duration", ""),
                "withdrawal_behavior": item.get("withdrawal_behavior", ""),
                "safety_notes": item.get("safety_notes", ""),
                "exclusion_reasons": "; ".join(item.get("exclusion_reasons", [])),
                "biometric": modality in BIOMETRIC_MODALITIES,
            }
        )
    return rows


def disabled_real_collection_modalities(scope: Dict[str, Any]) -> List[str]:
    return [
        modality
        for modality in MODALITIES
        if not scope.get("modalities", {}).get(modality, {}).get("enabled_for_real_collection", False)
    ]


def validate_modality_scope(scope: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    modalities = scope.get("modalities", {})
    for modality in MODALITIES:
        if modality not in modalities:
            errors.append(f"missing modality scope: {modality}")
            continue
        item = modalities[modality]
        for field in ("enabled_for_protocol_design", "enabled_for_real_collection", "collection_frequency", "consent_required"):
            if field not in item:
                errors.append(f"{modality}: missing {field}")
    for modality in DEFAULT_DISABLED_REAL_COLLECTION:
        if modalities.get(modality, {}).get("enabled_for_real_collection"):
            errors.append(f"{modality}: real collection must remain disabled in Phase 4A")
    return {"valid": not errors, "errors": errors, "matrix": modality_matrix(scope)}
