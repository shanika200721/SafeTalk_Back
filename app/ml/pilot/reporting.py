"""Readiness report generation for Phase 4A pilot protocol work."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.ml.common import paths
from app.ml.pilot.constants import EXPECTED_FINAL_READINESS, READINESS_STATES
from app.ml.pilot.modalities import modality_matrix
from app.ml.pilot.retention import validate_retention_policy
from app.ml.pilot.safety import safety_summary
from app.ml.pilot.schemas import to_plain_dict


def _write_json(path: Path, payload: Any, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing pilot report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing pilot report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_csv(path: Path, rows: List[Dict[str, Any]], overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing pilot report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def artifact_hashes(files: Iterable[Path]) -> Dict[str, str]:
    hashes = {}
    root = paths.get_repository_root()
    for file_path in files:
        if file_path.exists() and file_path.is_file():
            try:
                key = file_path.resolve().relative_to(root).as_posix()
            except ValueError:
                key = file_path.name
            hashes[key] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return hashes


def readiness_classification(validation: Dict[str, Any]) -> List[str]:
    states = list(EXPECTED_FINAL_READINESS)
    if not validation.get("valid"):
        states.append("protocol_draft")
    return states


def build_readiness_report(
    output_dir: str | Path,
    dataset: Any,
    validation: Dict[str, Any],
    modality_scope: Dict[str, Any],
    alignment_validation: Dict[str, Any],
    retention_policy: Dict[str, Any],
    privacy_validation: Dict[str, Any],
    overwrite: bool = False,
) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = paths.get_repository_root() / out_dir
    out_dir = paths.assert_not_raw_dataset_path(out_dir)
    states = readiness_classification(validation)
    blockers = {
        "governance_blockers": ["Institutional governance review has not approved real pilot collection."],
        "ethics_blockers": ["Ethics/IRB or institutional approval is required before recruitment or data collection."],
        "technical_blockers": [] if validation.get("valid") else [item["message"] for item in validation.get("errors", [])],
    }
    next_actions = {
        "required_before_real_collection": [
            "Obtain ethics/IRB or institutional approval.",
            "Finalize participant information sheet and consent form through institutional review.",
            "Finalize local crisis/hotline escalation contacts through qualified institutional staff.",
            "Confirm storage, access control, encryption, deletion, and backup procedures.",
            "Run a supervised dry run with synthetic data only.",
        ]
    }
    summary = {
        "readiness_states": states,
        "all_possible_states": list(READINESS_STATES),
        "protocol_version": dataset.manifest.protocol_version,
        "schema_version": dataset.manifest.schema_version,
        "participant_count": len(dataset.participants),
        "study_duration_weeks": dataset.metadata["weeks"],
        "session_count": len(dataset.sessions),
        "modality_record_count": len(dataset.modality_records),
        "withdrawal_count": len(dataset.withdrawals),
        "safety_event_count": len(dataset.safety_events),
        "real_collection_prohibited": True,
        "synthetic_validation_complete": validation.get("valid", False),
        "ethics_approval_required": True,
        "governance_review_required": True,
    }
    summary_md = "\n".join(
        [
            "# Phase 4A Pilot Readiness Summary",
            "",
            f"Readiness: {', '.join(states)}",
            f"Participants: {len(dataset.participants)} synthetic",
            f"Sessions: {len(dataset.sessions)}",
            f"Modality records: {len(dataset.modality_records)}",
            "",
            "Real participant collection remains prohibited until ethics and governance approvals are complete.",
            "",
        ]
    )
    files = [
        _write_json(out_dir / "pilot_readiness_summary.json", summary, overwrite),
        _write_text(out_dir / "pilot_readiness_summary.md", summary_md, overwrite),
        _write_json(out_dir / "pilot_protocol_validation.json", validation, overwrite),
        _write_csv(out_dir / "pilot_consent_matrix.csv", _consent_rows(dataset.consents), overwrite),
        _write_csv(out_dir / "pilot_modality_matrix.csv", modality_matrix(modality_scope), overwrite),
        _write_json(out_dir / "pilot_alignment_validation.json", alignment_validation, overwrite),
        _write_json(out_dir / "pilot_retention_validation.json", validate_retention_policy(retention_policy), overwrite),
        _write_json(out_dir / "pilot_safety_validation.json", safety_summary(dataset.safety_events), overwrite),
        _write_json(out_dir / "pilot_privacy_validation.json", privacy_validation, overwrite),
        _write_json(out_dir / "pilot_blockers.json", blockers, overwrite),
        _write_json(out_dir / "pilot_next_actions.json", next_actions, overwrite),
    ]
    inventory = {"files": [file.resolve().relative_to(paths.get_repository_root()).as_posix() for file in files]}
    inventory["hashes"] = artifact_hashes(files)
    files.append(_write_json(out_dir / "pilot_artifact_inventory.json", inventory, overwrite))
    return {"output_dir": out_dir, "summary": summary, "blockers": blockers, "next_actions": next_actions, "files": files}


def _consent_rows(consents: Iterable[Any]) -> List[Dict[str, Any]]:
    from app.ml.pilot.consent import consent_matrix

    rows: List[Dict[str, Any]] = []
    for consent in consents:
        rows.extend(consent_matrix(consent))
    return rows
