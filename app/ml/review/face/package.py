"""Create privacy-safe human-review packages for Phase 3H."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from datetime import timezone
from pathlib import Path
from typing import Any, Iterable

from app.ml.common import paths
from app.ml.common.hashing import hash_json_data, sha256_file
from app.ml.remediation.face.reporting import artifact_inventory, write_csv, write_json
from app.ml.review.face.constants import (
    FACE_REVIEW_DECISION_SCHEMA_VERSION,
    FACE_REVIEW_WORKFLOW_VERSION,
)
from app.ml.review.face.policy import hash_review_policy, load_review_policy
from app.ml.review.face.schemas import FaceReviewItem, utc_now, validate_repo_relative_path

ITEM_COLUMNS = (
    "review_item_id",
    "item_type",
    "group_id",
    "record_ids",
    "current_labels",
    "original_splits",
    "safe_image_references",
    "image_hashes",
    "perceptual_distance",
    "review_status",
    "required_reviewers",
    "policy_version",
    "created_at",
)

TEMPLATE_COLUMNS = (
    "review_item_id",
    "reviewer_alias",
    "decision",
    "reason_code",
    "confidence",
    "notes",
    "reviewed_at",
)


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


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = _resolve(output_dir)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_root() / "review", output):
        raise ValueError("face review package output must be under generated/review/")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _canonical_index(canonical_manifest_path: str | Path) -> dict[str, dict[str, str]]:
    rows = _load_csv(canonical_manifest_path)
    index = {row["record_id"]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError("canonical manifest contains duplicate record IDs")
    return index


def load_cross_label_quarantine(path: str | Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    records = payload.get("quarantined_records", [])
    if not isinstance(records, list):
        raise ValueError("cross-label quarantine missing quarantined_records")
    return records


def load_perceptual_candidates(path: str | Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("perceptual candidates missing candidates list")
    return candidates


def create_safe_image_reference(record: dict[str, str]) -> str:
    return validate_repo_relative_path(record["image_relative_path"])


def generate_review_item_id(item_type: str, group_id: str, record_ids: Iterable[str], distance: int | None = None) -> str:
    digest = hash_json_data(
        {
            "workflow_version": FACE_REVIEW_WORKFLOW_VERSION,
            "item_type": item_type,
            "group_id": group_id,
            "record_ids": sorted(record_ids),
            "perceptual_distance": distance,
        }
    )[:16]
    return f"face-review-{item_type.replace('_', '-')}-{digest}"


def _item_from_records(
    *,
    item_type: str,
    group_id: str,
    record_ids: list[str],
    canonical_by_id: dict[str, dict[str, str]],
    required_reviewers: int,
    policy_version: str,
    perceptual_distance: int | None = None,
) -> FaceReviewItem:
    missing = sorted(record_id for record_id in record_ids if record_id not in canonical_by_id)
    if missing:
        raise ValueError(f"review item references records missing from canonical manifest: {missing[:5]}")
    records = {record_id: canonical_by_id[record_id] for record_id in record_ids}
    return FaceReviewItem(
        review_item_id=generate_review_item_id(item_type, group_id, record_ids, perceptual_distance),
        item_type=item_type,
        group_id=group_id,
        record_ids=sorted(record_ids),
        current_labels={record_id: records[record_id]["canonical_emotion_label"] for record_id in sorted(records)},
        original_splits={record_id: records[record_id]["source_split"] for record_id in sorted(records)},
        safe_image_references={record_id: create_safe_image_reference(records[record_id]) for record_id in sorted(records)},
        image_hashes={record_id: records[record_id]["image_hash"].lower() for record_id in sorted(records)},
        perceptual_distance=perceptual_distance,
        required_reviewers=required_reviewers,
        policy_version=policy_version,
    )


def build_review_items(
    *,
    cross_label_quarantine_path: str | Path,
    perceptual_candidates_path: str | Path,
    canonical_manifest_path: str | Path,
    policy_config_path: str | Path | None = None,
) -> list[FaceReviewItem]:
    policy = load_review_policy(policy_config_path)
    required_reviewers = int(policy.get("minimum_reviewer_count", 2))
    canonical_by_id = _canonical_index(canonical_manifest_path)
    items: list[FaceReviewItem] = []

    grouped: dict[str, list[str]] = defaultdict(list)
    for record in load_cross_label_quarantine(cross_label_quarantine_path):
        grouped[str(record["group_id"])].append(str(record["record_id"]))
    for group_id, record_ids in sorted(grouped.items()):
        items.append(
            _item_from_records(
                item_type="cross_label_conflict",
                group_id=group_id,
                record_ids=record_ids,
                canonical_by_id=canonical_by_id,
                required_reviewers=required_reviewers,
                policy_version=policy["policy_version"],
            )
        )

    for index, candidate in enumerate(load_perceptual_candidates(perceptual_candidates_path)):
        record_ids = [str(record_id) for record_id in candidate.get("record_ids", [])]
        distance = int(candidate.get("hamming_distance", candidate.get("perceptual_distance", 0)))
        group_id = f"face-perceptual-{index:06d}"
        items.append(
            _item_from_records(
                item_type="perceptual_duplicate_candidate",
                group_id=group_id,
                record_ids=record_ids,
                canonical_by_id=canonical_by_id,
                required_reviewers=required_reviewers,
                policy_version=policy["policy_version"],
                perceptual_distance=distance,
            )
        )

    return sorted(items, key=lambda item: item.review_item_id)


def validate_review_package(items: list[FaceReviewItem]) -> dict[str, Any]:
    ids = [item.review_item_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("review package contains duplicate review item IDs")
    for item in items:
        for reference in item.safe_image_references.values():
            validate_repo_relative_path(reference)
    return {
        "valid": True,
        "review_item_count": len(items),
        "cross_label_review_item_count": sum(1 for item in items if item.item_type == "cross_label_conflict"),
        "perceptual_review_item_count": sum(1 for item in items if item.item_type == "perceptual_duplicate_candidate"),
    }


def _instructions(policy: dict[str, Any]) -> str:
    decisions = "\n".join(f"- {item}" for item in policy.get("allowed_decisions", []))
    reasons = "\n".join(f"- {item}" for item in policy.get("reason_codes", []))
    return (
        "# Phase 3H Face Human Review Instructions\n\n"
        "Reviewers must not identify people, infer demographics, infer health state, infer suicide risk, "
        "or make clinical judgments. Inspect only whether listed records appear visually identical, "
        "near-identical, incorrectly labeled, corrupted, or ambiguous.\n\n"
        "Do not modify, copy, crop, resize, relabel, delete, or overwrite raw image files. Every submitted "
        "decision requires a reason code. Unresolved conflicts remain quarantined.\n\n"
        "## Allowed Decisions\n\n"
        f"{decisions}\n\n"
        "## Reason Codes\n\n"
        f"{reasons}\n"
    )


def _html_index(items: list[FaceReviewItem]) -> str:
    rows = []
    for item in items:
        refs = "<br>".join(
            f"{html.escape(record_id)}: <a href=\"../../../{html.escape(path)}\">image</a>"
            for record_id, path in sorted(item.safe_image_references.items())
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.review_item_id)}</td>"
            f"<td>{html.escape(str(item.item_type))}</td>"
            f"<td>{html.escape(item.group_id)}</td>"
            f"<td>{html.escape(', '.join(item.record_ids))}</td>"
            f"<td>{refs}</td>"
            "</tr>"
        )
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Face Review Index</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:6px;vertical-align:top;font-size:12px}</style></head><body>"
        "<h1>Local Face Review Index</h1><p>Local-only index. Image bytes are not embedded.</p>"
        "<table><thead><tr><th>Item</th><th>Type</th><th>Group</th><th>Records</th><th>References</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></body></html>"
    )


def save_review_package(
    *,
    items: list[FaceReviewItem],
    output_dir: str | Path,
    source_fingerprint: str,
    policy_config_path: str | Path | None = None,
    include_html_index: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    validation = validate_review_package(items)
    policy = load_review_policy(policy_config_path)
    output = _ensure_output_dir(output_dir)
    now = utc_now()
    item_payload = [item.to_safe_dict() for item in items]
    outputs: dict[str, Path] = {}
    outputs["items_json"] = write_json(
        output / "face_review_items.json",
        {
            "workflow_version": FACE_REVIEW_WORKFLOW_VERSION,
            "decision_schema_version": FACE_REVIEW_DECISION_SCHEMA_VERSION,
            "source_fingerprint": source_fingerprint,
            "review_policy_hash": hash_review_policy(policy),
            "review_items": item_payload,
        },
        overwrite=overwrite,
    )
    csv_rows = [{key: _json_cell(value) if isinstance(value, (dict, list)) else value for key, value in item.items()} for item in item_payload]
    outputs["items_csv"] = write_csv(output / "face_review_items.csv", csv_rows, fieldnames=list(ITEM_COLUMNS), overwrite=overwrite)
    template_rows = [
        {
            "review_item_id": item.review_item_id,
            "reviewer_alias": "",
            "decision": "",
            "reason_code": "",
            "confidence": "",
            "notes": "",
            "reviewed_at": "",
        }
        for item in items
    ]
    outputs["template_csv"] = write_csv(
        output / "face_review_template.csv",
        template_rows,
        fieldnames=list(TEMPLATE_COLUMNS),
        overwrite=overwrite,
    )
    instructions_path = output / "face_review_instructions.md"
    if instructions_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {instructions_path}")
    instructions_path.write_text(_instructions(policy), encoding="utf-8")
    outputs["instructions"] = instructions_path
    status = {
        **validation,
        "workflow_version": FACE_REVIEW_WORKFLOW_VERSION,
        "decision_schema_version": FACE_REVIEW_DECISION_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "review_status": "review_not_started",
        "reviewed_items": 0,
        "pending_items": len(items),
        "required_reviewers": int(policy.get("minimum_reviewer_count", 2)),
        "generated_at": now.astimezone(timezone.utc).isoformat(),
    }
    outputs["status"] = write_json(output / "face_review_status.json", status, overwrite=overwrite)
    if include_html_index:
        index_path = output / "face_review_index.html"
        if index_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {index_path}")
        index_path.write_text(_html_index(items), encoding="utf-8")
        outputs["html_index"] = index_path
    outputs["inventory"] = write_json(
        output / "face_review_artifact_inventory.json",
        {"artifacts": artifact_inventory(outputs), "workflow_version": FACE_REVIEW_WORKFLOW_VERSION},
        overwrite=overwrite,
    )
    status["outputs"] = {key: str(value) for key, value in outputs.items()}
    status["package_hash"] = sha256_file(outputs["items_json"], allow_outside_project=True)
    return status


def create_review_package(
    *,
    cross_label_quarantine_path: str | Path,
    perceptual_candidates_path: str | Path,
    canonical_manifest_path: str | Path,
    source_fingerprint: str,
    policy_config_path: str | Path | None,
    output_dir: str | Path,
    include_html_index: bool = False,
    validate_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    items = build_review_items(
        cross_label_quarantine_path=cross_label_quarantine_path,
        perceptual_candidates_path=perceptual_candidates_path,
        canonical_manifest_path=canonical_manifest_path,
        policy_config_path=policy_config_path,
    )
    validation = validate_review_package(items)
    if validate_only:
        return {**validation, "items": items, "source_fingerprint": source_fingerprint}
    return save_review_package(
        items=items,
        output_dir=output_dir,
        source_fingerprint=source_fingerprint,
        policy_config_path=policy_config_path,
        include_html_index=include_html_index,
        overwrite=overwrite,
    )

