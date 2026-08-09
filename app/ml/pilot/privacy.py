"""Privacy checks for pilot protocol exports."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from app.ml.pilot.constants import BIOMETRIC_MODALITIES
from app.ml.pilot.participant import validate_no_production_user_id_leakage, validate_pilot_participant_id

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
NAME_KEYS = {"name", "full_name", "email", "address", "student_id", "registration_number"}


def validate_no_direct_identifiers(payload: Any) -> None:
    text = str(payload)
    if EMAIL_RE.search(text):
        raise ValueError("email-like direct identifier found")
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in NAME_KEYS and value:
                raise ValueError(f"direct identifier field found: {key}")
            validate_no_direct_identifiers(value)
    elif isinstance(payload, list):
        for item in payload:
            validate_no_direct_identifiers(item)


def validate_raw_artifact_reference(record: Any) -> Dict[str, Any]:
    ref = record.raw_artifact_reference
    if not ref:
        return {"valid": True}
    path = Path(ref)
    if path.is_absolute():
        return {"valid": False, "reason": "raw artifact references must be repository-relative placeholders"}
    if record.modality in BIOMETRIC_MODALITIES and not path.as_posix().startswith("generated/pilot-protocol-smoke/"):
        return {"valid": False, "reason": "raw biometric references must stay in synthetic smoke output scope"}
    return {"valid": True}


def validate_privacy(records: Sequence[Any], participant_ids: Iterable[str]) -> Dict[str, Any]:
    errors = []
    for pid in participant_ids:
        if not validate_pilot_participant_id(pid):
            errors.append(f"invalid pseudonymous participant ID: {pid}")
        try:
            validate_no_production_user_id_leakage([pid])
        except ValueError as exc:
            errors.append(str(exc))
    for record in records:
        result = validate_raw_artifact_reference(record)
        if not result["valid"]:
            errors.append(f"{record.record_id}: {result['reason']}")
        for ref in (record.raw_artifact_reference, record.derived_artifact_reference):
            if ref:
                try:
                    validate_no_direct_identifiers({"artifact_reference": ref})
                except ValueError as exc:
                    errors.append(f"{record.record_id}: {exc}")
    return {"valid": not errors, "errors": errors}
