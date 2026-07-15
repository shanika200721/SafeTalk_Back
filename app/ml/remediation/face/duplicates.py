"""Deterministic exact-duplicate remediation for facial emotion records."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from app.ml.common.hashing import hash_json_data
from app.ml.remediation.face.constants import (
    FACE_DECISION_COLUMNS,
    FACE_DUPLICATE_POLICY_VERSION,
    FACE_REQUIRED_CANONICAL_COLUMNS,
)
from app.ml.remediation.face.policy import hash_duplicate_policy_payload, load_duplicate_policy
from app.ml.remediation.face.schemas import (
    FaceDuplicateGroup,
    FaceDuplicateRecord,
    FaceRemediationAction,
    FaceRemediationDecision,
)


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_canonical_records(canonical_manifest_path: str | Path) -> dict[str, FaceDuplicateRecord]:
    path = Path(canonical_manifest_path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(set(FACE_REQUIRED_CANONICAL_COLUMNS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"canonical manifest missing columns: {missing}")
        records: dict[str, FaceDuplicateRecord] = {}
        for row in reader:
            group_id = f"hash:{row['image_hash'].lower()}"
            record = FaceDuplicateRecord(
                record_id=row["record_id"],
                image_hash=row["image_hash"],
                canonical_label=row["canonical_emotion_label"],
                original_split=row["source_split"],
                relative_path=row["image_relative_path"],
                group_id=group_id,
                readable=_bool_value(row.get("readable", True)),
            )
            if record.record_id in records:
                raise ValueError(f"duplicate record_id in canonical manifest: {record.record_id}")
            records[record.record_id] = record
    return records


def load_face_duplicate_groups(
    duplicate_manifest_path: str | Path,
    canonical_manifest_path: str | Path | None = None,
) -> list[FaceDuplicateGroup]:
    with Path(duplicate_manifest_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    groups_payload = payload.get("duplicate_image_hash_groups")
    if not isinstance(groups_payload, list):
        raise ValueError("duplicate manifest missing duplicate_image_hash_groups")

    records = load_canonical_records(canonical_manifest_path) if canonical_manifest_path else {}
    groups: list[FaceDuplicateGroup] = []
    for index, item in enumerate(groups_payload):
        record_ids = sorted(str(record_id) for record_id in item.get("record_ids", []))
        missing = sorted(record_id for record_id in record_ids if records and record_id not in records)
        if missing:
            raise ValueError(f"duplicate group contains IDs missing from canonical manifest: {missing[:5]}")
        if records:
            labels = sorted({records[record_id].canonical_label for record_id in record_ids})
            splits = sorted({records[record_id].original_split for record_id in record_ids})
        else:
            labels = sorted(str(label) for label in item.get("labels", []))
            splits = sorted(str(split) for split in item.get("source_splits") or item.get("original_splits") or [])
        image_hash = str(item.get("image_hash") or item.get("duplicate_hash") or "").lower()
        if not image_hash:
            raise ValueError(f"duplicate group {index} missing image_hash")
        groups.append(
            FaceDuplicateGroup(
                group_id=f"face-dup-{index:06d}-{image_hash[:12]}",
                image_hash=image_hash,
                record_ids=record_ids,
                labels=labels,
                original_splits=splits,
                same_label=len(labels) == 1,
                cross_label=len(labels) > 1,
                cross_split=len(splits) > 1,
            )
        )
    return groups


def classify_duplicate_group(group: FaceDuplicateGroup) -> FaceDuplicateGroup:
    labels = sorted(set(group.labels))
    splits = sorted(set(group.original_splits))
    return group.copy(
        update={
            "labels": labels,
            "original_splits": splits,
            "same_label": len(labels) == 1,
            "cross_label": len(labels) > 1,
            "cross_split": len(splits) > 1,
        }
    )


def select_deterministic_representative(
    group: FaceDuplicateGroup,
    records_by_id: Mapping[str, FaceDuplicateRecord],
) -> str:
    candidates = [records_by_id[record_id] for record_id in group.record_ids]
    if not candidates:
        raise ValueError("duplicate group has no candidate records")
    candidates.sort(key=lambda item: (not item.readable, item.relative_path, item.record_id))
    return candidates[0].record_id


def quarantine_cross_label_group(group: FaceDuplicateGroup) -> tuple[FaceDuplicateGroup, list[FaceRemediationDecision]]:
    decisions = [
        FaceRemediationDecision(
            record_id=record_id,
            action=FaceRemediationAction.QUARANTINE_CROSS_LABEL,
            representative_id=None,
            group_id=group.group_id,
            reason="exact_hash_cross_label_conflict_quarantine_entire_group",
            policy_version=FACE_DUPLICATE_POLICY_VERSION,
        )
        for record_id in group.record_ids
    ]
    reasons = {record_id: "cross_label_conflict_quarantined" for record_id in group.record_ids}
    return (
        group.copy(
            update={
                "quarantined": True,
                "decision": "quarantine_entire_cross_label_group",
                "selected_representative_id": None,
                "exclusion_reasons": reasons,
            }
        ),
        decisions,
    )


def remediate_same_label_group(
    group: FaceDuplicateGroup,
    records_by_id: Mapping[str, FaceDuplicateRecord],
) -> tuple[FaceDuplicateGroup, list[FaceRemediationDecision]]:
    representative_id = select_deterministic_representative(group, records_by_id)
    decisions: list[FaceRemediationDecision] = []
    reasons: dict[str, str] = {}
    for record_id in group.record_ids:
        if record_id == representative_id:
            decisions.append(
                FaceRemediationDecision(
                    record_id=record_id,
                    action=FaceRemediationAction.KEEP,
                    representative_id=representative_id,
                    group_id=group.group_id,
                    reason="deterministic_representative_for_same_label_exact_duplicate_group",
                )
            )
        else:
            reasons[record_id] = "same_label_exact_duplicate_excluded_representative_retained"
            decisions.append(
                FaceRemediationDecision(
                    record_id=record_id,
                    action=FaceRemediationAction.EXCLUDE_DUPLICATE,
                    representative_id=representative_id,
                    group_id=group.group_id,
                    reason="same_label_exact_duplicate_excluded_representative_retained",
                )
            )
    return (
        group.copy(
            update={
                "quarantined": False,
                "decision": "keep_one_exclude_same_label_duplicates",
                "selected_representative_id": representative_id,
                "exclusion_reasons": reasons,
            }
        ),
        decisions,
    )


def build_face_remediation_decisions(
    canonical_manifest_path: str | Path,
    duplicate_manifest_path: str | Path,
    policy_config_path: str | Path | None = None,
) -> dict[str, Any]:
    policy = load_duplicate_policy(policy_config_path)
    records_by_id = load_canonical_records(canonical_manifest_path)
    groups = load_face_duplicate_groups(duplicate_manifest_path, canonical_manifest_path)
    decisions_by_id: dict[str, FaceRemediationDecision] = {}
    remediated_groups: list[FaceDuplicateGroup] = []

    for raw_group in groups:
        group = classify_duplicate_group(raw_group)
        if group.cross_label:
            remediated, group_decisions = quarantine_cross_label_group(group)
        elif group.same_label:
            remediated, group_decisions = remediate_same_label_group(group, records_by_id)
        else:
            raise ValueError(f"duplicate group has invalid label classification: {group.group_id}")
        remediated_groups.append(remediated)
        for decision in group_decisions:
            if decision.record_id in decisions_by_id:
                raise ValueError(f"record receives multiple remediation decisions: {decision.record_id}")
            decisions_by_id[decision.record_id] = decision

    for record_id in sorted(records_by_id):
        if record_id not in decisions_by_id:
            decisions_by_id[record_id] = FaceRemediationDecision(
                record_id=record_id,
                action=FaceRemediationAction.KEEP,
                representative_id=record_id,
                group_id=None,
                reason="unique_exact_hash_record_retained",
            )

    validate_duplicate_group_coverage(records_by_id, remediated_groups, decisions_by_id)
    return {
        "policy": policy,
        "policy_hash": hash_duplicate_policy_payload(policy),
        "records_by_id": records_by_id,
        "duplicate_groups": remediated_groups,
        "decisions_by_id": decisions_by_id,
    }


def validate_duplicate_group_coverage(
    records_by_id: Mapping[str, FaceDuplicateRecord],
    groups: list[FaceDuplicateGroup],
    decisions_by_id: Mapping[str, FaceRemediationDecision],
) -> None:
    if set(records_by_id) != set(decisions_by_id):
        missing = sorted(set(records_by_id) - set(decisions_by_id))
        extra = sorted(set(decisions_by_id) - set(records_by_id))
        raise ValueError(f"remediation decisions do not cover canonical records: missing={len(missing)} extra={len(extra)}")
    seen_group_records: set[str] = set()
    for group in groups:
        for record_id in group.record_ids:
            if record_id not in records_by_id:
                raise ValueError(f"duplicate group references missing record: {record_id}")
            seen_group_records.add(record_id)
            decision = decisions_by_id[record_id]
            if group.cross_label and decision.action != FaceRemediationAction.QUARANTINE_CROSS_LABEL:
                raise ValueError("cross-label group was not fully quarantined")
            if not group.cross_label and decision.action == FaceRemediationAction.QUARANTINE_CROSS_LABEL:
                raise ValueError("same-label group incorrectly quarantined")
        if group.cross_label:
            actions = {decisions_by_id[record_id].action for record_id in group.record_ids}
            if actions != {FaceRemediationAction.QUARANTINE_CROSS_LABEL}:
                raise ValueError("cross-label group partially retained")
        elif group.same_label:
            kept = [record_id for record_id in group.record_ids if decisions_by_id[record_id].action == FaceRemediationAction.KEEP]
            if len(kept) != 1:
                raise ValueError(f"same-label duplicate group must retain exactly one record: {group.group_id}")
    for record_id, decision in decisions_by_id.items():
        if decision.action != FaceRemediationAction.KEEP and not decision.reason:
            raise ValueError(f"excluded record missing reason: {record_id}")


def hash_duplicate_policy(policy_config_path: str | Path | None = None) -> str:
    return hash_duplicate_policy_payload(load_duplicate_policy(policy_config_path))


def decision_rows(decisions_by_id: Mapping[str, FaceRemediationDecision]) -> list[dict[str, Any]]:
    return [
        {column: decision.to_safe_dict().get(column, "") for column in FACE_DECISION_COLUMNS}
        for _, decision in sorted(decisions_by_id.items())
    ]


def duplicate_group_summary(groups: list[FaceDuplicateGroup]) -> dict[str, Any]:
    records_in_duplicate_groups = sorted({record_id for group in groups for record_id in group.record_ids})
    size_distribution = Counter(len(group.record_ids) for group in groups)
    return {
        "duplicate_group_count": len(groups),
        "records_in_duplicate_groups": len(records_in_duplicate_groups),
        "groups_with_more_than_two_records": sum(1 for group in groups if len(group.record_ids) > 2),
        "same_label_group_count": sum(1 for group in groups if group.same_label),
        "cross_split_group_count": sum(1 for group in groups if group.cross_split),
        "cross_label_group_count": sum(1 for group in groups if group.cross_label),
        "both_cross_split_and_cross_label_group_count": sum(1 for group in groups if group.cross_split and group.cross_label),
        "max_group_size": max((len(group.record_ids) for group in groups), default=0),
        "group_size_distribution": dict(sorted((str(size), count) for size, count in size_distribution.items())),
        "groups": [group.to_safe_dict() for group in groups],
        "summary_hash": hash_json_data([group.to_safe_dict() for group in groups]),
    }

