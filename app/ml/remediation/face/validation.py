"""Validation and readiness reporting for Phase 3G face remediation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.remediation.face.constants import FACE_REVISED_SPLIT_VERSION
from app.ml.remediation.face.reporting import artifact_inventory, readiness_markdown, write_json
from app.ml.remediation.face.splitting import replay_face_v2_split


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def validate_face_remediation_artifacts(
    *,
    canonical_manifest_path: str | Path,
    deduplicated_manifest_path: str | Path,
    remediation_report_path: str | Path,
    source_fingerprint_path: str | Path,
    policy_config_path: str | Path,
    split_manifest_path: str | Path,
    split_assignments_path: str | Path,
) -> dict[str, Any]:
    canonical_manifest_path = _resolve_path(canonical_manifest_path)
    deduplicated_manifest_path = _resolve_path(deduplicated_manifest_path)
    remediation_report_path = _resolve_path(remediation_report_path)
    source_fingerprint_path = _resolve_path(source_fingerprint_path)
    policy_config_path = _resolve_path(policy_config_path)
    split_manifest_path = _resolve_path(split_manifest_path)
    split_assignments_path = _resolve_path(split_assignments_path)
    remediation_report = _load_json(remediation_report_path)
    source_fingerprint = _load_json(source_fingerprint_path)
    split_manifest = _load_json(split_manifest_path)
    source_verified = source_fingerprint.get("combined_sha256") == remediation_report.get("source_fingerprint") == split_manifest.get("source_fingerprint")
    if not source_verified:
        raise ValueError("source fingerprint mismatch")
    canonical_hash = sha256_file(canonical_manifest_path, allow_outside_project=True)
    dedup_hash = sha256_file(deduplicated_manifest_path, allow_outside_project=True)
    if canonical_hash != remediation_report.get("canonical_manifest_hash") or canonical_hash != split_manifest.get("canonical_manifest_hash"):
        raise ValueError("canonical manifest hash mismatch")
    if dedup_hash != split_manifest.get("deduplicated_view_hash"):
        raise ValueError("deduplicated view hash mismatch")
    replay_passed = replay_face_v2_split(
        manifest_path=split_manifest_path,
        deduplicated_manifest_path=deduplicated_manifest_path,
        remediation_report_path=remediation_report_path,
        source_fingerprint_path=source_fingerprint_path,
        policy_config_path=policy_config_path,
    )
    if not replay_passed:
        raise ValueError("deterministic replay failed")
    if int(split_manifest.get("image_hash_overlap_count", 0)) != 0:
        raise ValueError("exact image hash overlap remains")
    if int(split_manifest.get("duplicate_overlap_count", 0)) != 0:
        raise ValueError("duplicate group overlap remains")
    required_labels = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}
    distributions = split_manifest.get("label_distributions", {})
    missing_labels = [
        f"{split}:{label}"
        for split in ("train", "validation", "test")
        for label in required_labels
        if int(distributions.get(split, {}).get(label, 0)) < 1
    ]
    if missing_labels:
        raise ValueError(f"label coverage failed: {missing_labels}")
    return {
        "source_fingerprint_verified": True,
        "canonical_manifest_hash": canonical_hash,
        "deduplicated_view_hash": dedup_hash,
        "duplicate_policy_hash": remediation_report["duplicate_policy_hash"],
        "deterministic_replay_passed": replay_passed,
        "exact_hash_overlap_count": int(split_manifest.get("image_hash_overlap_count", 0)),
        "duplicate_group_overlap_count": int(split_manifest.get("duplicate_overlap_count", 0)),
        "record_overlap_count": int(split_manifest.get("record_overlap_count", 0)),
        "all_labels_present": True,
        "split_version": FACE_REVISED_SPLIT_VERSION,
    }


def generate_face_readiness_report(
    *,
    canonical_manifest_path: str | Path,
    deduplicated_manifest_path: str | Path,
    remediation_report_path: str | Path,
    source_fingerprint_path: str | Path,
    policy_config_path: str | Path,
    split_manifest_path: str | Path,
    split_assignments_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    validation = validate_face_remediation_artifacts(
        canonical_manifest_path=canonical_manifest_path,
        deduplicated_manifest_path=deduplicated_manifest_path,
        remediation_report_path=remediation_report_path,
        source_fingerprint_path=source_fingerprint_path,
        policy_config_path=policy_config_path,
        split_manifest_path=split_manifest_path,
        split_assignments_path=split_assignments_path,
    )
    report = {
        **validation,
        "readiness_classification": "ready_with_restrictions",
        "research_only": True,
        "deployable": False,
        "remaining_blockers": [
            "Subject-level leakage cannot be eliminated because subject identifiers are unavailable.",
            "Perceptual near duplicates may remain outside the bounded diagnostic set.",
            "Demographic composition is unknown or incomplete.",
            "Facial emotion is not depression and not suicidal ideation.",
            "No autonomous decision should rely on facial emotion.",
        ],
        "recommended_next_actions": [
            "Have a human review quarantined cross-label exact conflicts before any future use.",
            "Review bounded perceptual duplicate candidates and decide whether a stricter image-level policy is needed.",
            "Only then consider a research baseline using the v2 split; do not deploy it.",
        ],
    }
    output = Path(output_dir)
    if not output.is_absolute():
        output = paths.get_repository_root() / output
    output = output.resolve(strict=False)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_reports_root(), output):
        raise ValueError("face readiness report must be under generated/reports/")
    output.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    outputs["validation_json"] = write_json(output / "face_readiness_validation.json", report, overwrite=overwrite)
    md_path = output / "face_readiness_validation.md"
    if md_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {md_path}")
    md_path.write_text(readiness_markdown(report), encoding="utf-8")
    outputs["validation_md"] = md_path
    outputs["blockers"] = write_json(output / "face_blockers_remaining.json", {"blockers": report["remaining_blockers"]}, overwrite=overwrite)
    outputs["next_actions"] = write_json(output / "face_next_actions.json", {"next_actions": report["recommended_next_actions"]}, overwrite=overwrite)
    outputs["inventory"] = write_json(output / "face_artifact_inventory.json", {"artifacts": artifact_inventory(outputs)}, overwrite=overwrite)
    report["outputs"] = {key: str(value) for key, value in outputs.items()}
    return report
