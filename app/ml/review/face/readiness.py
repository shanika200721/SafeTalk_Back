"""Phase 3H review-readiness validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.remediation.face.reporting import artifact_inventory, write_csv, write_json


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with _resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def classify_review_readiness(*, total_items: int, reviewed_items: int, unresolved_items: int, integrity_failures: list[str]) -> str:
    if integrity_failures:
        return "review_failed_integrity"
    if reviewed_items == 0:
        return "review_not_started"
    if reviewed_items < total_items:
        return "review_in_progress"
    if unresolved_items:
        return "review_complete_with_unresolved_items"
    return "review_complete_ready_with_restrictions"


def validate_phase3h_review_readiness(
    *,
    review_package_path: str | Path,
    phase3g_quarantine_path: str | Path,
    source_fingerprint_path: str | Path,
    output_dir: str | Path,
    reconciliation_manifest_path: str | Path | None = None,
    reviewed_view_report_path: str | Path | None = None,
    split_report_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    package = _load_json(review_package_path)
    source = _load_json(source_fingerprint_path)
    items = package.get("review_items", [])
    cross_items = [item for item in items if item.get("item_type") == "cross_label_conflict"]
    integrity_failures: list[str] = []
    if source.get("combined_sha256") != package.get("source_fingerprint"):
        integrity_failures.append("source_fingerprint_mismatch")
    reviewed_items = 0
    unresolved_items = len(cross_items)
    consensus_items = 0
    disagreement_items = 0
    restored_count = 0
    retained_quarantine_count = len(_load_json(phase3g_quarantine_path).get("quarantined_records", []))
    if reconciliation_manifest_path and _resolve(reconciliation_manifest_path).exists():
        rec = _load_json(reconciliation_manifest_path).get("reconciled_decisions", [])
        reviewed_items = sum(1 for item in rec if item.get("consensus_reached"))
        consensus_items = reviewed_items
        disagreement_items = sum(1 for item in rec if item.get("final_status") == "disagreement")
        unresolved_items = sum(1 for item in rec if item.get("final_action") in {"keep_quarantined", "unresolved", "additional_review"})
        restored_count = sum(len(item.get("retained_record_ids", [])) for item in rec if item.get("final_action") == "restore_record")
        retained_quarantine_count = sum(len(item.get("quarantined_record_ids", [])) for item in rec)
    if reviewed_view_report_path and _resolve(reviewed_view_report_path).exists():
        view = _load_json(reviewed_view_report_path)
        if view.get("source_fingerprint") != source.get("combined_sha256"):
            integrity_failures.append("reviewed_view_source_fingerprint_mismatch")
        if view.get("label_change_applied") is not False:
            integrity_failures.append("automatic_label_change_detected")
    classification = classify_review_readiness(
        total_items=len(items),
        reviewed_items=reviewed_items,
        unresolved_items=unresolved_items,
        integrity_failures=integrity_failures,
    )
    report = {
        "readiness_classification": classification,
        "source_fingerprint_verified": not any("source_fingerprint" in item for item in integrity_failures),
        "source_fingerprint": source.get("combined_sha256"),
        "total_review_items": len(items),
        "cross_label_review_item_count": len(cross_items),
        "perceptual_review_item_count": len(items) - len(cross_items),
        "reviewed_items": reviewed_items,
        "pending_items": max(len(items) - reviewed_items, 0),
        "consensus_items": consensus_items,
        "disagreement_items": disagreement_items,
        "unresolved_items": unresolved_items,
        "restored_record_count": restored_count,
        "retained_quarantine_count": retained_quarantine_count,
        "integrity_failures": integrity_failures,
        "no_automatic_relabeling": True,
        "no_raw_image_content": True,
        "subject_independence": "unavailable_subject_ids_not_present",
        "split_integrity": "not_regenerated_v2_authoritative" if not split_report_path else "see_split_report",
    }
    output = _resolve(output_dir)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_reports_root(), output):
        raise ValueError("phase3h readiness output must be under generated/reports/")
    output.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    outputs["validation_json"] = write_json(output / "face_review_validation.json", report, overwrite=overwrite)
    md = output / "face_review_validation.md"
    if md.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {md}")
    md.write_text(
        "# Phase 3H Face Review Validation\n\n"
        f"- Readiness classification: {classification}\n"
        f"- Total review items: {len(items)}\n"
        f"- Reviewed items: {reviewed_items}\n"
        f"- Pending items: {max(len(items) - reviewed_items, 0)}\n"
        f"- Integrity failures: {len(integrity_failures)}\n\n"
        "Human review cannot establish clinical truth and does not authorize deployment.\n",
        encoding="utf-8",
    )
    outputs["validation_md"] = md
    matrix_rows = [
        {
            "review_item_id": item.get("review_item_id"),
            "item_type": item.get("item_type"),
            "required_reviewers": item.get("required_reviewers"),
            "status": "pending" if reviewed_items == 0 else "see_reconciliation_manifest",
        }
        for item in items
    ]
    outputs["matrix"] = write_csv(
        output / "face_review_completion_matrix.csv",
        matrix_rows,
        fieldnames=["review_item_id", "item_type", "required_reviewers", "status"],
        overwrite=overwrite,
    )
    blockers = {
        "blockers": (
            integrity_failures
            or [
                "actual human decisions are pending",
                "unresolved conflicts remain quarantined",
                "subject identifiers are unavailable",
            ]
        )
    }
    outputs["blockers"] = write_json(output / "face_review_blockers.json", blockers, overwrite=overwrite)
    outputs["next_actions"] = write_json(
        output / "face_review_next_actions.json",
        {"next_actions": ["collect two independent human-review decisions per item", "run import and reconciliation", "rerun readiness validation"]},
        overwrite=overwrite,
    )
    outputs["inventory"] = write_json(output / "face_review_artifact_inventory.json", {"artifacts": artifact_inventory(outputs)}, overwrite=overwrite)
    report["artifact_hashes"] = {name: sha256_file(path, allow_outside_project=True) for name, path in outputs.items()}
    return report

