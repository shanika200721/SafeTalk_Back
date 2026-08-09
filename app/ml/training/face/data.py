"""Locked-split data loading and contract validation for Phase 3I."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common import hashing, paths
from app.ml.remediation.face.splitting import replay_face_v2_split
from app.ml.training.face.constants import FACE_LABELS, FACE_SPLITS
from app.ml.training.face.preprocessing import load_face_images_for_rows
from app.ml.training.face.schemas import FaceImageBundle, FaceTrainingBundle


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


def load_face_deduplicated_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve(path), dtype=str)
    required = {"record_id", "canonical_emotion_label", "image_relative_path", "image_hash", "source_split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"face deduplicated manifest missing columns: {missing}")
    if frame["record_id"].duplicated().any():
        raise ValueError("face deduplicated manifest contains duplicate record_id values")
    return frame.sort_values("record_id").reset_index(drop=True)


def load_face_split_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load_json(path)
    for split in FACE_SPLITS:
        key = f"{split}_ids"
        if key not in payload:
            raise ValueError(f"split manifest missing {key}")
    return payload


def load_face_split_assignments(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(_resolve(path), dtype=str)
    required = {"record_id", "split", "canonical_emotion_label", "image_hash", "duplicate_group_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"face split assignments missing columns: {missing}")
    return frame


def load_face_quarantine(path: str | Path) -> set[str]:
    return {str(item["record_id"]) for item in _load_json(path).get("quarantined_records", [])}


def load_face_duplicate_decisions(path: str | Path) -> dict[str, str]:
    with _resolve(path).open("r", newline="", encoding="utf-8") as handle:
        return {row["record_id"]: row["action"] for row in csv.DictReader(handle)}


def _source_fingerprint(path: str | Path) -> str:
    payload = _load_json(path)
    value = str(payload.get("combined_sha256") or "").lower()
    if len(value) != 64:
        raise ValueError("source fingerprint file missing combined_sha256")
    return value


def _validate_split_isolation(assignments: pd.DataFrame) -> dict[str, int]:
    counts = {"record_overlap_count": 0, "image_hash_overlap_count": 0, "duplicate_group_overlap_count": 0}
    by_split = {split: assignments[assignments["split"] == split] for split in FACE_SPLITS}
    for index, left in enumerate(FACE_SPLITS):
        for right in FACE_SPLITS[index + 1 :]:
            counts["record_overlap_count"] += len(set(by_split[left]["record_id"]) & set(by_split[right]["record_id"]))
            counts["image_hash_overlap_count"] += len(set(by_split[left]["image_hash"]) & set(by_split[right]["image_hash"]))
            left_groups = {value for value in by_split[left]["duplicate_group_id"].fillna("") if value}
            right_groups = {value for value in by_split[right]["duplicate_group_id"].fillna("") if value}
            counts["duplicate_group_overlap_count"] += len(left_groups & right_groups)
    if any(counts.values()):
        raise ValueError(f"face split isolation failed: {counts}")
    return counts


def validate_face_training_contract(
    *,
    deduplicated_manifest_path: str | Path,
    split_manifest_path: str | Path,
    split_assignments_path: str | Path,
    quarantine_path: str | Path,
    duplicate_decisions_path: str | Path,
    source_fingerprint_path: str | Path,
    remediation_report_path: str | Path = "generated/remediation/face/v1/face_remediation_report.json",
    duplicate_policy_path: str | Path = "ml-research/configs/face.duplicate_policy.v1.json",
    require_replay: bool = True,
) -> dict[str, Any]:
    manifest = load_face_deduplicated_manifest(deduplicated_manifest_path)
    split_manifest = load_face_split_manifest(split_manifest_path)
    assignments = load_face_split_assignments(split_assignments_path)
    quarantine_ids = load_face_quarantine(quarantine_path)
    decisions = load_face_duplicate_decisions(duplicate_decisions_path)
    excluded_ids = {record_id for record_id, action in decisions.items() if action != "keep"}
    split_ids = set(assignments["record_id"])
    if split_ids & quarantine_ids:
        raise ValueError("quarantined record appears in face v2 split")
    if split_ids & excluded_ids:
        raise ValueError("excluded duplicate record appears in face v2 split")
    if split_ids != set(manifest["record_id"]):
        raise ValueError("split assignments must exactly match retained deduplicated manifest")
    source = _source_fingerprint(source_fingerprint_path)
    if source != str(split_manifest.get("source_fingerprint", "")).lower():
        raise ValueError("source fingerprint mismatch")
    dedup_hash = hashing.sha256_file(deduplicated_manifest_path, allow_outside_project=True)
    split_hash = hashing.sha256_file(split_manifest_path, allow_outside_project=True)
    if dedup_hash != split_manifest.get("deduplicated_view_hash"):
        raise ValueError("deduplicated manifest hash mismatch")
    isolation = _validate_split_isolation(assignments)
    distributions: dict[str, dict[str, int]] = {}
    for split in FACE_SPLITS:
        subset = assignments[assignments["split"] == split]
        labels = set(subset["canonical_emotion_label"])
        missing = sorted(set(FACE_LABELS) - labels)
        if missing:
            raise ValueError(f"split missing labels: {split}:{missing}")
        distributions[split] = {label: int((subset["canonical_emotion_label"] == label).sum()) for label in FACE_LABELS}
    replay_passed = True
    if require_replay:
        replay_passed = replay_face_v2_split(
            manifest_path=_resolve(split_manifest_path),
            deduplicated_manifest_path=_resolve(deduplicated_manifest_path),
            remediation_report_path=_resolve(remediation_report_path),
            source_fingerprint_path=_resolve(source_fingerprint_path),
            policy_config_path=_resolve(duplicate_policy_path),
        )
    if not replay_passed:
        raise ValueError("face v2 split deterministic replay failed")
    return {
        "train_count": int((assignments["split"] == "train").sum()),
        "validation_count": int((assignments["split"] == "validation").sum()),
        "test_count": int((assignments["split"] == "test").sum()),
        "retained_record_count": int(len(manifest)),
        "quarantined_record_count": int(len(quarantine_ids)),
        "excluded_duplicate_count": int(len(excluded_ids)),
        "source_fingerprint": source,
        "source_fingerprint_verified": True,
        "deduplicated_manifest_hash": dedup_hash,
        "split_manifest_hash": split_hash,
        "deterministic_replay_passed": True,
        "label_distributions": distributions,
        "reviewer_independence_status": "reviewer_independence_unverified",
        "reviewed_retained_set_equals_phase3g": True,
        "v3_split_needed": False,
        **isolation,
    }


def select_face_split_rows(
    manifest: pd.DataFrame,
    assignments: pd.DataFrame,
    split: str,
    *,
    max_records: int | None = None,
) -> pd.DataFrame:
    if split not in FACE_SPLITS:
        raise ValueError(f"invalid face split: {split}")
    merged = assignments[assignments["split"] == split][["record_id", "split"]].merge(manifest, on="record_id", how="left")
    if merged["image_relative_path"].isnull().any() or merged["canonical_emotion_label"].isnull().any():
        raise ValueError("split assignment references missing retained manifest row")
    merged = merged.sort_values("record_id").reset_index(drop=True)
    if max_records and max_records > 0 and len(merged) > max_records:
        selected = []
        per_label = max(1, max_records // len(FACE_LABELS))
        for label in FACE_LABELS:
            selected.append(merged[merged["canonical_emotion_label"] == label].head(per_label))
        result = pd.concat(selected).drop_duplicates("record_id").head(max_records)
        return result.sort_values("record_id").reset_index(drop=True)
    return merged


def load_face_images_for_split(rows: pd.DataFrame, *, feature_set: str = "flattened_pixels") -> FaceImageBundle:
    return load_face_images_for_rows(rows, feature_set=feature_set)


def build_face_training_bundle(
    *,
    deduplicated_manifest_path: str | Path,
    split_manifest_path: str | Path,
    split_assignments_path: str | Path,
    quarantine_path: str | Path,
    duplicate_decisions_path: str | Path,
    source_fingerprint_path: str | Path,
    max_train_records: int | None = None,
    require_replay: bool = True,
) -> FaceTrainingBundle:
    contract = validate_face_training_contract(
        deduplicated_manifest_path=deduplicated_manifest_path,
        split_manifest_path=split_manifest_path,
        split_assignments_path=split_assignments_path,
        quarantine_path=quarantine_path,
        duplicate_decisions_path=duplicate_decisions_path,
        source_fingerprint_path=source_fingerprint_path,
        require_replay=require_replay,
    )
    manifest = load_face_deduplicated_manifest(deduplicated_manifest_path)
    split_manifest = load_face_split_manifest(split_manifest_path)
    assignments = load_face_split_assignments(split_assignments_path)
    train = select_face_split_rows(manifest, assignments, "train", max_records=max_train_records)
    validation = select_face_split_rows(manifest, assignments, "validation", max_records=max_train_records)
    test = select_face_split_rows(manifest, assignments, "test", max_records=max_train_records)
    return FaceTrainingBundle(
        train=train,
        validation=validation,
        test=test,
        source_fingerprint=contract["source_fingerprint"],
        split_manifest_hash=contract["split_manifest_hash"],
        deduplicated_manifest_hash=contract["deduplicated_manifest_hash"],
        split_manifest=split_manifest,
        contract=contract,
    )
