"""Import and validate reviewer decisions for Phase 3H."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.remediation.face.reporting import artifact_inventory, write_json
from app.ml.review.face.constants import FACE_ALLOWED_REVIEW_DECISIONS, FACE_DEFAULT_REASON_CODES
from app.ml.review.face.policy import load_review_policy
from app.ml.review.face.schemas import FaceReviewerDecision


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_json(path: str | Path) -> Any:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_review_item_ids(review_package: str | Path) -> set[str]:
    package_path = _resolve(review_package)
    if package_path.is_dir():
        package_path = package_path / "face_review_items.json"
    payload = _load_json(package_path)
    return {str(item["review_item_id"]) for item in payload.get("review_items", [])}


def validate_allowed_decision(decision: str, policy: dict[str, Any]) -> str:
    allowed = set(policy.get("allowed_decisions") or FACE_ALLOWED_REVIEW_DECISIONS)
    if decision not in allowed:
        raise ValueError(f"invalid review decision: {decision}")
    return decision


def validate_reason_code(reason_code: str, policy: dict[str, Any]) -> str:
    allowed = set(policy.get("reason_codes") or FACE_DEFAULT_REASON_CODES)
    if reason_code not in allowed:
        raise ValueError(f"invalid reason code: {reason_code}")
    return reason_code


def validate_reviewer_decision(
    row: dict[str, Any],
    *,
    known_review_item_ids: set[str],
    policy: dict[str, Any],
    reviewer_alias: str | None = None,
) -> FaceReviewerDecision:
    payload = dict(row)
    if reviewer_alias:
        payload["reviewer_alias"] = reviewer_alias
    if payload.get("review_item_id") not in known_review_item_ids:
        raise ValueError(f"unknown review item ID: {payload.get('review_item_id')}")
    validate_allowed_decision(str(payload.get("decision", "")), policy)
    validate_reason_code(str(payload.get("reason_code", "")), policy)
    return FaceReviewerDecision(**payload)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({str(key).strip().strip('"'): value for key, value in row.items()})
        return rows


def _decision_rows_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if "reviewer_decisions" in payload:
            return list(payload["reviewer_decisions"])
        if "decisions" in payload:
            return list(payload["decisions"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("decision JSON must be a list or object with reviewer_decisions")


def detect_duplicate_reviewer_submission(decisions: list[FaceReviewerDecision]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    for decision in decisions:
        key = (decision.review_item_id, decision.reviewer_alias)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def detect_missing_review_items(decisions: list[FaceReviewerDecision], review_item_ids: set[str]) -> set[str]:
    reviewed = {decision.review_item_id for decision in decisions}
    return set(review_item_ids) - reviewed


def load_reviewer_decisions(
    *,
    review_package: str | Path,
    decision_file: str | Path,
    policy_config_path: str | Path | None = None,
    reviewer_alias: str | None = None,
) -> list[FaceReviewerDecision]:
    policy = load_review_policy(policy_config_path)
    known_ids = _load_review_item_ids(review_package)
    source = _resolve(decision_file)
    if source.suffix.lower() == ".csv":
        rows = _read_csv(source)
    else:
        rows = _decision_rows_from_json(_load_json(source))
    decisions = [
        validate_reviewer_decision(row, known_review_item_ids=known_ids, policy=policy, reviewer_alias=reviewer_alias)
        for row in rows
        if any(str(value).strip() for value in row.values())
    ]
    duplicates = detect_duplicate_reviewer_submission(decisions)
    if duplicates:
        raise ValueError(f"duplicate reviewer submissions: {duplicates[:5]}")
    return sorted(decisions, key=lambda item: (item.review_item_id, item.reviewer_alias, item.reviewed_at.isoformat()))


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = _resolve(output_dir)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_root() / "review", output):
        raise ValueError("face review decisions output must be under generated/review/")
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_validated_decisions(
    *,
    decisions: list[FaceReviewerDecision],
    output_dir: str | Path,
    decision_file: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = _ensure_output_dir(output_dir)
    payload = {
        "decision_count": len(decisions),
        "decision_source_hash": sha256_file(decision_file, allow_outside_project=True) if decision_file else None,
        "reviewer_decisions": [decision.to_safe_dict() for decision in decisions],
    }
    outputs: dict[str, Path] = {}
    outputs["validated_decisions"] = write_json(output / "reviewer_decisions_validated.json", payload, overwrite=overwrite)
    outputs["inventory"] = write_json(output / "reviewer_decisions_artifact_inventory.json", {"artifacts": artifact_inventory(outputs)}, overwrite=overwrite)
    return {"payload": payload, "outputs": outputs}
