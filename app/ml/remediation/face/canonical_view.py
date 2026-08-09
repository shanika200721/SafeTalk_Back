"""Build a leakage-safe deduplicated facial emotion development view."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.remediation.face.constants import (
    FACE_DECISION_COLUMNS,
    FACE_DEDUPLICATED_MANIFEST_COLUMNS,
    FACE_DUPLICATE_POLICY_VERSION,
    FACE_REMEDIATION_VERSION,
)
from app.ml.remediation.face.duplicates import (
    build_face_remediation_decisions,
    decision_rows,
    duplicate_group_summary,
)
from app.ml.remediation.face.reporting import artifact_inventory, remediation_markdown, write_csv, write_json
from app.ml.remediation.face.schemas import (
    FaceDeduplicatedViewReport,
    FaceRemediationAction,
)


def _load_canonical_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_source_fingerprint(path: str | Path) -> str:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = str(payload.get("combined_sha256") or payload.get("source_fingerprint_hash") or "").lower()
    if len(value) != 64:
        raise ValueError("source fingerprint file does not contain combined_sha256")
    return value


def _verify_source_fingerprint(source_fingerprint_path: str | Path, expected: str | None) -> str:
    actual = _load_source_fingerprint(source_fingerprint_path)
    if expected and actual != expected.lower():
        raise ValueError("source fingerprint mismatch")
    return actual


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if not output.is_absolute():
        output = paths.get_repository_root() / output
    output = output.resolve(strict=False)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_root(), output):
        raise ValueError("face remediation output must be under generated/")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _repo_file_exists(relative_path: str) -> bool:
    return (paths.get_repository_root() / relative_path).exists()


def build_face_deduplicated_view(
    *,
    canonical_manifest_path: str | Path,
    duplicate_manifest_path: str | Path,
    source_fingerprint_path: str | Path,
    policy_config_path: str | Path | None,
    output_dir: str | Path,
    validate_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    remediation = build_face_remediation_decisions(canonical_manifest_path, duplicate_manifest_path, policy_config_path)
    source_fingerprint = _verify_source_fingerprint(source_fingerprint_path, None)
    canonical_hash = sha256_file(canonical_manifest_path, allow_outside_project=True)
    rows = _load_canonical_rows(canonical_manifest_path)
    records_by_id = remediation["records_by_id"]
    decisions_by_id = remediation["decisions_by_id"]
    groups = remediation["duplicate_groups"]
    group_by_record_id = {record_id: group.group_id for group in groups for record_id in group.record_ids}

    retained_rows: list[dict[str, Any]] = []
    excluded_same_label: list[dict[str, Any]] = []
    cross_label_quarantine: list[dict[str, Any]] = []
    excluded_label_counter: Counter[str] = Counter()
    retained_label_counter: Counter[str] = Counter()

    for row in rows:
        record_id = row["record_id"]
        decision = decisions_by_id[record_id]
        if decision.action == FaceRemediationAction.KEEP:
            if not _repo_file_exists(row["image_relative_path"]):
                raise FileNotFoundError(f"retained source image missing: {row['image_relative_path']}")
            retained = dict(row)
            retained["duplicate_group_id"] = group_by_record_id.get(record_id, "")
            retained["remediation_action"] = decision.action
            retained["remediation_policy_version"] = FACE_DUPLICATE_POLICY_VERSION
            retained_rows.append(retained)
            retained_label_counter[row["canonical_emotion_label"]] += 1
        elif decision.action == FaceRemediationAction.EXCLUDE_DUPLICATE:
            excluded_label_counter[row["canonical_emotion_label"]] += 1
            excluded_same_label.append(
                {
                    "record_id": record_id,
                    "representative_id": decision.representative_id,
                    "group_id": decision.group_id,
                    "reason": decision.reason,
                    "canonical_emotion_label": row["canonical_emotion_label"],
                }
            )
        elif decision.action == FaceRemediationAction.QUARANTINE_CROSS_LABEL:
            excluded_label_counter[row["canonical_emotion_label"]] += 1
            cross_label_quarantine.append(
                {
                    "record_id": record_id,
                    "group_id": decision.group_id,
                    "reason": decision.reason,
                    "canonical_emotion_label": row["canonical_emotion_label"],
                }
            )
        else:
            excluded_label_counter[row["canonical_emotion_label"]] += 1

    quarantined_ids = {item["record_id"] for item in cross_label_quarantine}
    retained_ids = {row["record_id"] for row in retained_rows}
    if retained_ids & quarantined_ids:
        raise ValueError("quarantined records cannot appear in retained view")

    summary = duplicate_group_summary(groups)
    report = FaceDeduplicatedViewReport(
        source_fingerprint=source_fingerprint,
        canonical_manifest_hash=canonical_hash,
        duplicate_policy_hash=remediation["policy_hash"],
        source_record_count=len(rows),
        retained_record_count=len(retained_rows),
        excluded_same_label_duplicate_count=len(excluded_same_label),
        quarantined_cross_label_record_count=len(cross_label_quarantine),
        duplicate_group_count=summary["duplicate_group_count"],
        same_label_group_count=summary["same_label_group_count"],
        cross_split_group_count=summary["cross_split_group_count"],
        cross_label_group_count=summary["cross_label_group_count"],
        records_in_duplicate_groups=summary["records_in_duplicate_groups"],
        retained_label_distribution=dict(sorted(retained_label_counter.items())),
        excluded_label_distribution=dict(sorted(excluded_label_counter.items())),
        warnings=[
            "Exact deduplication does not guarantee subject independence.",
            "Subject identifiers are unavailable; subject-level leakage cannot be eliminated.",
            "Perceptual near duplicates are diagnostic candidates only and are not automatically excluded.",
            "Facial emotion is not depression or suicidal ideation; this view is research-only.",
            "No raw images were copied, modified, deleted, or relabeled.",
        ],
    )

    if validate_only:
        return {
            "report": report,
            "retained_rows": retained_rows,
            "decisions_by_id": decisions_by_id,
            "duplicate_groups": groups,
            "policy_hash": remediation["policy_hash"],
            "outputs": {},
        }

    output = _ensure_output_dir(output_dir)
    outputs: dict[str, Path] = {}
    outputs["deduplicated_manifest"] = write_csv(
        output / "face_deduplicated_manifest.csv",
        retained_rows,
        fieldnames=list(FACE_DEDUPLICATED_MANIFEST_COLUMNS),
        overwrite=overwrite,
    )
    outputs["decisions"] = write_csv(
        output / "face_remediation_decisions.csv",
        decision_rows(decisions_by_id),
        fieldnames=list(FACE_DECISION_COLUMNS),
        overwrite=overwrite,
    )
    outputs["report_json"] = write_json(output / "face_remediation_report.json", report.to_safe_dict(), overwrite=overwrite)
    outputs["same_label_duplicate_exclusions"] = write_json(
        output / "face_same_label_duplicate_exclusions.json",
        {"excluded_records": excluded_same_label, "count": len(excluded_same_label)},
        overwrite=overwrite,
    )
    outputs["cross_label_quarantine"] = write_json(
        output / "face_cross_label_quarantine.json",
        {"quarantined_records": cross_label_quarantine, "count": len(cross_label_quarantine)},
        overwrite=overwrite,
    )
    outputs["duplicate_group_summary"] = write_json(output / "face_duplicate_group_summary.json", summary, overwrite=overwrite)
    outputs["report_md"] = output / "face_remediation_report.md"
    if outputs["report_md"].exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {outputs['report_md']}")
    outputs["report_md"].write_text(remediation_markdown(report.to_safe_dict()), encoding="utf-8")
    inventory = artifact_inventory(outputs)
    outputs["artifact_inventory"] = write_json(
        output / "face_remediation_artifact_inventory.json",
        {"artifacts": inventory, "remediation_version": FACE_REMEDIATION_VERSION},
        overwrite=overwrite,
    )
    return {
        "report": report,
        "retained_rows": retained_rows,
        "decisions_by_id": decisions_by_id,
        "duplicate_groups": groups,
        "policy_hash": remediation["policy_hash"],
        "outputs": outputs,
    }


def _average_hash(path: Path, *, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    average = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
    return int(bits, 2)


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def run_perceptual_duplicate_diagnostics(
    *,
    deduplicated_manifest_path: str | Path,
    output_path: str | Path,
    limit: int = 1000,
    threshold: int = 6,
    overwrite: bool = False,
) -> dict[str, Any]:
    rows = _load_canonical_rows(deduplicated_manifest_path)
    selected = rows if limit <= 0 else rows[:limit]
    hashed: list[dict[str, Any]] = []
    repo = paths.get_repository_root()
    for row in selected:
        image_path = repo / row["image_relative_path"]
        if not image_path.exists():
            continue
        try:
            digest = _average_hash(image_path)
        except Exception:
            continue
        hashed.append({"record_id": row["record_id"], "image_hash": row["image_hash"], "perceptual_hash": digest})

    candidates = []
    comparisons = 0
    for index, left in enumerate(hashed):
        for right in hashed[index + 1 :]:
            if left["image_hash"] == right["image_hash"]:
                continue
            comparisons += 1
            distance = _hamming(left["perceptual_hash"], right["perceptual_hash"])
            if distance <= threshold:
                candidates.append(
                    {
                        "record_ids": [left["record_id"], right["record_id"]],
                        "hamming_distance": distance,
                        "threshold": threshold,
                        "automatic_exclusion": False,
                    }
                )
    payload = {
        "method": "bounded Pillow average-hash pairwise comparison over retained manifest order",
        "limit": limit,
        "threshold": threshold,
        "records_considered": len(selected),
        "hashes_computed": len(hashed),
        "comparisons": comparisons,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "limitations": [
            "Average hash is a lightweight perceptual diagnostic, not identity recognition.",
            "Candidates are not automatically excluded.",
            "Bounded comparisons may miss candidates outside the selected limit.",
        ],
    }
    write_json(output_path, payload, overwrite=overwrite)
    return payload

