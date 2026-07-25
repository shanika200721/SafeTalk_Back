"""Generate a reviewed canonical view without overwriting Phase 3G artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.hashing import hash_json_data, sha256_file
from app.ml.remediation.face.reporting import artifact_inventory, write_csv, write_json


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with _resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = _resolve(output_dir)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_root() / "remediation" / "face", output):
        raise ValueError("reviewed face view output must be under generated/remediation/face/")
    output.mkdir(parents=True, exist_ok=True)
    return output


def create_reviewed_canonical_view(
    *,
    phase3g_manifest_path: str | Path,
    phase3g_decisions_path: str | Path,
    phase3g_quarantine_path: str | Path,
    reconciliation_manifest_path: str | Path,
    canonical_manifest_path: str | Path,
    source_fingerprint: str,
    output_dir: str | Path,
    overwrite: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    retained_rows = _load_csv(phase3g_manifest_path)
    canonical_by_id = {row["record_id"]: row for row in _load_csv(canonical_manifest_path)}
    phase3g_decisions = _load_csv(phase3g_decisions_path)
    phase3g_quarantine = _load_json(phase3g_quarantine_path).get("quarantined_records", [])
    reconciled = _load_json(reconciliation_manifest_path).get("reconciled_decisions", [])
    retained_by_id = {row["record_id"]: dict(row) for row in retained_rows}
    restorations: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    quarantine_by_id = {item["record_id"]: dict(item) for item in phase3g_quarantine}
    for decision in reconciled:
        if decision.get("final_action") == "restore_record":
            for record_id in decision.get("retained_record_ids", []):
                if record_id in canonical_by_id and record_id not in retained_by_id:
                    restored = dict(canonical_by_id[record_id])
                    restored["duplicate_group_id"] = decision.get("review_item_id", "")
                    restored["remediation_action"] = "review_restored_without_label_change"
                    restored["remediation_policy_version"] = "phase3h_review_1.0.0"
                    retained_by_id[record_id] = restored
                    restorations.append({"record_id": record_id, "review_item_id": decision["review_item_id"], "label_changed": False})
                    quarantine_by_id.pop(record_id, None)
        elif decision.get("final_action") == "exclude_all":
            for record_id in decision.get("excluded_record_ids", []):
                retained_by_id.pop(record_id, None)
                exclusions.append({"record_id": record_id, "review_item_id": decision["review_item_id"]})
        for record_id in decision.get("quarantined_record_ids", []):
            quarantine_by_id.setdefault(record_id, {"record_id": record_id, "reason": decision.get("reconciliation_reason", "")})
    reviewed_rows = [retained_by_id[record_id] for record_id in sorted(retained_by_id)]
    view_hash = hash_json_data(reviewed_rows)
    report = {
        "source_fingerprint": source_fingerprint,
        "phase3g_manifest_hash": sha256_file(phase3g_manifest_path, allow_outside_project=True),
        "phase3g_decisions_hash": sha256_file(phase3g_decisions_path, allow_outside_project=True),
        "reconciliation_manifest_hash": sha256_file(reconciliation_manifest_path, allow_outside_project=True),
        "reviewed_view_hash": view_hash,
        "retained_record_count": len(reviewed_rows),
        "restored_record_count": len(restorations),
        "excluded_record_count": len(exclusions),
        "quarantined_record_count": len(quarantine_by_id),
        "label_change_applied": False,
        "raw_images_changed": False,
        "retained_label_distribution": dict(sorted(Counter(row["canonical_emotion_label"] for row in reviewed_rows).items())),
        "phase3g_linkage": "generated/remediation/face/v1",
    }
    if validate_only:
        return {"report": report, "reviewed_rows": reviewed_rows, "restorations": restorations, "quarantine": list(quarantine_by_id.values())}
    output = _ensure_output_dir(output_dir)
    outputs: dict[str, Path] = {}
    fieldnames = list(reviewed_rows[0].keys()) if reviewed_rows else []
    outputs["manifest"] = write_csv(output / "face_reviewed_deduplicated_manifest.csv", reviewed_rows, fieldnames=fieldnames, overwrite=overwrite)
    outputs["decisions"] = write_csv(output / "face_reviewed_remediation_decisions.csv", phase3g_decisions, fieldnames=list(phase3g_decisions[0].keys()), overwrite=overwrite)
    outputs["quarantine"] = write_json(output / "face_reviewed_quarantine.json", {"quarantined_records": list(quarantine_by_id.values()), "count": len(quarantine_by_id)}, overwrite=overwrite)
    outputs["restorations"] = write_json(output / "face_reviewed_restorations.json", {"restored_records": restorations, "count": len(restorations)}, overwrite=overwrite)
    outputs["exclusions"] = write_json(output / "face_reviewed_exclusions.json", {"excluded_records": exclusions, "count": len(exclusions)}, overwrite=overwrite)
    outputs["report_json"] = write_json(output / "face_reviewed_view_report.json", report, overwrite=overwrite)
    md = output / "face_reviewed_view_report.md"
    if md.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {md}")
    md.write_text(
        "# Phase 3H Reviewed Face View\n\n"
        f"- Retained records: {report['retained_record_count']}\n"
        f"- Restored records: {report['restored_record_count']}\n"
        f"- Quarantined records: {report['quarantined_record_count']}\n"
        "- Label changes applied: False\n- Raw images changed: False\n",
        encoding="utf-8",
    )
    outputs["report_md"] = md
    outputs["inventory"] = write_json(output / "face_reviewed_artifact_inventory.json", {"artifacts": artifact_inventory(outputs)}, overwrite=overwrite)
    return {"report": report, "outputs": outputs}

