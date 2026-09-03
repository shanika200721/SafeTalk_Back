"""Reviewed split generation for Phase 3H face review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.remediation.face.reporting import artifact_inventory, write_json
from app.ml.remediation.face.splitting import create_face_v2_split, replay_face_v2_split


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


def reviewed_records_changed(reviewed_view_report_path: str | Path) -> bool:
    report = _load_json(reviewed_view_report_path)
    return int(report.get("restored_record_count", 0)) > 0 or int(report.get("excluded_record_count", 0)) > 0


def create_face_reviewed_split(
    *,
    reviewed_manifest_path: str | Path,
    reviewed_view_report_path: str | Path,
    source_fingerprint_path: str | Path,
    policy_config_path: str | Path,
    output_dir: str | Path,
    seed: int | None = None,
    validate_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    changed = reviewed_records_changed(reviewed_view_report_path)
    output = _resolve(output_dir)
    paths.assert_not_raw_dataset_path(output)
    if not changed:
        report = {
            "reviewed_split_generated": False,
            "reason": "no_reconciled_decision_changed_retained_records",
            "authoritative_split": "generated/manifests/splits/face/v2",
            "reviewed_manifest_hash": sha256_file(reviewed_manifest_path, allow_outside_project=True),
            "subject_independence": "unavailable_subject_ids_not_present",
        }
        if validate_only:
            return {"report": report, "outputs": {}}
        output.mkdir(parents=True, exist_ok=True)
        outputs = {
            "report_json": write_json(output / "face_reviewed_split_report.json", report, overwrite=overwrite),
        }
        md = output / "face_reviewed_split_report.md"
        if md.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {md}")
        md.write_text(
            "# Phase 3H Reviewed Split\n\n"
            "No reviewed split regeneration was required. The Phase 3G v2 split remains authoritative.\n",
            encoding="utf-8",
        )
        outputs["report_md"] = md
        outputs["inventory"] = write_json(output / "face_reviewed_split_artifact_inventory.json", {"artifacts": artifact_inventory(outputs)}, overwrite=overwrite)
        return {"report": report, "outputs": outputs}

    # Reuse the verified v2 deterministic splitter, then rename outputs to reviewed names.
    result = create_face_v2_split(
        deduplicated_manifest_path=reviewed_manifest_path,
        remediation_report_path=reviewed_view_report_path,
        source_fingerprint_path=source_fingerprint_path,
        policy_config_path=policy_config_path,
        output_dir=output,
        seed=seed,
        validate_only=validate_only,
        overwrite=overwrite,
    )
    if validate_only:
        return result
    rename_map = {
        "face_split_manifest.json": "face_reviewed_split_manifest.json",
        "face_split_assignments.csv": "face_reviewed_split_assignments.csv",
        "face_split_report.json": "face_reviewed_split_report.json",
        "face_split_report.md": "face_reviewed_split_report.md",
        "face_split_exclusions.json": "face_reviewed_split_exclusions.json",
        "face_hash_isolation_report.json": "face_reviewed_hash_isolation_report.json",
        "face_class_distribution.json": "face_reviewed_class_distribution.json",
    }
    for old_name, new_name in rename_map.items():
        old_path = output / old_name
        new_path = output / new_name
        if old_path.exists():
            if new_path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing file: {new_path}")
            old_path.replace(new_path)
    replay = replay_face_v2_split(
        manifest_path=output / "face_reviewed_split_manifest.json",
        deduplicated_manifest_path=reviewed_manifest_path,
        remediation_report_path=reviewed_view_report_path,
        source_fingerprint_path=source_fingerprint_path,
        policy_config_path=policy_config_path,
    )
    outputs = {key: output / name for key, name in rename_map.items()}
    outputs["replay"] = write_json(output / "face_reviewed_replay_report.json", {"deterministic_replay_passed": replay}, overwrite=overwrite)
    outputs["inventory"] = write_json(output / "face_reviewed_split_artifact_inventory.json", {"artifacts": artifact_inventory(outputs)}, overwrite=overwrite)
    result["outputs"] = outputs
    return result

