"""Create deterministic leakage-safe v2 splits for remediated face data."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.hashing import hash_json_data, sha256_file
from app.ml.remediation.face.constants import (
    FACE_DEFAULT_RANDOM_SEED,
    FACE_LABELS,
    FACE_REVISED_SPLIT_ID,
    FACE_REVISED_SPLIT_VERSION,
    FACE_SPLIT_NAMES,
)
from app.ml.remediation.face.policy import load_duplicate_policy
from app.ml.remediation.face.reporting import artifact_inventory, split_markdown, write_csv, write_json
from app.ml.remediation.face.schemas import FaceRevisedSplitManifest


def _load_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = Path(output_dir)
    if not output.is_absolute():
        output = paths.get_repository_root() / output
    output = output.resolve(strict=False)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_manifests_root(), output):
        raise ValueError("face v2 split output must be under generated/manifests/")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _stable_key(record_id: str, seed: int) -> str:
    return hash_json_data({"seed": seed, "record_id": record_id})


def _targets(count: int, proportions: tuple[float, float, float], *, minimum_per_split: int) -> dict[str, int]:
    if count >= minimum_per_split * len(FACE_SPLIT_NAMES):
        result = {split: minimum_per_split for split in FACE_SPLIT_NAMES}
        remaining = count - sum(result.values())
    else:
        result = {split: 0 for split in FACE_SPLIT_NAMES}
        remaining = count
    raw = {split: remaining * proportion for split, proportion in zip(FACE_SPLIT_NAMES, proportions)}
    floors = {split: int(raw[split]) for split in FACE_SPLIT_NAMES}
    for split in FACE_SPLIT_NAMES:
        result[split] += floors[split]
    remainder = count - sum(result.values())
    order = sorted(FACE_SPLIT_NAMES, key=lambda split: (raw[split] - floors[split], split), reverse=True)
    for split in order[:remainder]:
        result[split] += 1
    return result


def deterministic_stratified_face_split(
    rows: list[dict[str, str]],
    *,
    seed: int,
    train: float = 0.70,
    validation: float = 0.15,
    test: float = 0.15,
    minimum_records_per_class_per_split: int = 1,
) -> dict[str, str]:
    if abs(train + validation + test - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to 1")
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["canonical_emotion_label"]].append(row)
    missing_labels = sorted(set(FACE_LABELS) - set(by_label))
    if missing_labels:
        raise ValueError(f"deduplicated view missing labels: {missing_labels}")
    assignments: dict[str, str] = {}
    for label in FACE_LABELS:
        label_rows = sorted(by_label[label], key=lambda row: row["record_id"])
        if len(label_rows) < minimum_records_per_class_per_split * 3:
            raise ValueError(f"label has too few records for all revised splits: {label}")
        ordered = sorted(label_rows, key=lambda row: (_stable_key(row["record_id"], seed), row["record_id"]))
        label_targets = _targets(
            len(ordered),
            (train, validation, test),
            minimum_per_split=minimum_records_per_class_per_split,
        )
        start = 0
        for split in FACE_SPLIT_NAMES:
            selected = ordered[start : start + label_targets[split]]
            start += label_targets[split]
            for row in selected:
                assignments[row["record_id"]] = split
    return assignments


def _distributions(rows: list[dict[str, str]], assignments: dict[str, str]) -> dict[str, dict[str, int]]:
    counters = {split: Counter() for split in FACE_SPLIT_NAMES}
    for row in rows:
        counters[assignments[row["record_id"]]][row["canonical_emotion_label"]] += 1
    return {split: dict(sorted(counters[split].items())) for split in FACE_SPLIT_NAMES}


def _ids_by_split(assignments: dict[str, str]) -> dict[str, list[str]]:
    return {split: sorted(record_id for record_id, assigned in assignments.items() if assigned == split) for split in FACE_SPLIT_NAMES}


def _overlap_count(values_by_split: dict[str, set[str]]) -> int:
    count = 0
    splits = list(FACE_SPLIT_NAMES)
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            count += len(values_by_split[left] & values_by_split[right])
    return count


def _original_split_comparison(rows: list[dict[str, str]], assignments: dict[str, str]) -> dict[str, Any]:
    matrix: dict[str, Counter[str]] = {split: Counter() for split in FACE_SPLIT_NAMES}
    for row in rows:
        matrix[assignments[row["record_id"]]][row.get("source_split", "unknown")] += 1
    return {
        "policy": "original predefined train/test retained only as metadata",
        "matrix": {split: dict(sorted(counter.items())) for split, counter in matrix.items()},
    }


def _validate_split_rows(
    rows: list[dict[str, str]],
    assignments: dict[str, str],
    excluded_ids: dict[str, str],
    quarantined_ids: dict[str, str],
    *,
    minimum_records_per_class_per_split: int,
) -> dict[str, Any]:
    row_ids = {row["record_id"] for row in rows}
    if set(assignments) != row_ids:
        raise ValueError("all retained records must be covered by exactly one revised split")
    if set(assignments) & set(quarantined_ids):
        raise ValueError("quarantined record included in revised split")
    if set(assignments) & set(excluded_ids):
        raise ValueError("excluded duplicate included in revised split")
    by_split = _ids_by_split(assignments)
    record_overlap = _overlap_count({split: set(by_split[split]) for split in FACE_SPLIT_NAMES})
    hash_by_split = {split: set() for split in FACE_SPLIT_NAMES}
    group_by_split = {split: set() for split in FACE_SPLIT_NAMES}
    record_by_id = {row["record_id"]: row for row in rows}
    for record_id, split in assignments.items():
        row = record_by_id[record_id]
        hash_by_split[split].add(row["image_hash"])
        group_id = row.get("duplicate_group_id")
        if group_id:
            group_by_split[split].add(group_id)
    image_hash_overlap = _overlap_count(hash_by_split)
    duplicate_overlap = _overlap_count(group_by_split)
    if record_overlap:
        raise ValueError("record IDs overlap across revised splits")
    if image_hash_overlap:
        raise ValueError("exact image hash appears in multiple revised splits")
    if duplicate_overlap:
        raise ValueError("duplicate group appears in multiple revised splits")
    distributions = _distributions(rows, assignments)
    missing = [
        f"{split}:{label}"
        for split in FACE_SPLIT_NAMES
        for label in FACE_LABELS
        if distributions[split].get(label, 0) < minimum_records_per_class_per_split
    ]
    if missing:
        raise ValueError(f"class presence requirement failed: {missing}")
    return {
        "record_overlap_count": record_overlap,
        "image_hash_overlap_count": image_hash_overlap,
        "duplicate_overlap_count": duplicate_overlap,
        "label_distributions": distributions,
    }


def create_face_v2_split(
    *,
    deduplicated_manifest_path: str | Path,
    remediation_report_path: str | Path,
    source_fingerprint_path: str | Path,
    policy_config_path: str | Path,
    output_dir: str | Path,
    seed: int | None = None,
    validate_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    seed = FACE_DEFAULT_RANDOM_SEED if seed is None else int(seed)
    rows = _load_csv(deduplicated_manifest_path)
    if not rows:
        raise ValueError("deduplicated manifest is empty")
    remediation_report = _load_json(remediation_report_path)
    source_fingerprint = _load_json(source_fingerprint_path)
    policy = load_duplicate_policy(policy_config_path)
    if source_fingerprint.get("combined_sha256") != remediation_report.get("source_fingerprint"):
        raise ValueError("source fingerprint mismatch")
    if sha256_file(deduplicated_manifest_path, allow_outside_project=True) == "0" * 64:
        raise ValueError("invalid deduplicated view hash")

    excluded_ids: dict[str, str] = {}
    quarantined_ids: dict[str, str] = {}
    # The sibling remediation artifacts are optional for direct unit tests, but present in real runs.
    rem_dir = Path(remediation_report_path).parent
    exclusion_path = rem_dir / "face_same_label_duplicate_exclusions.json"
    quarantine_path = rem_dir / "face_cross_label_quarantine.json"
    if exclusion_path.exists():
        excluded_ids = {item["record_id"]: item["reason"] for item in _load_json(exclusion_path).get("excluded_records", [])}
    if quarantine_path.exists():
        quarantined_ids = {item["record_id"]: item["reason"] for item in _load_json(quarantine_path).get("quarantined_records", [])}

    minimum = int(policy.get("minimum_records_per_class_per_split", 1))
    assignments = deterministic_stratified_face_split(rows, seed=seed, minimum_records_per_class_per_split=minimum)
    validation = _validate_split_rows(
        rows,
        assignments,
        excluded_ids,
        quarantined_ids,
        minimum_records_per_class_per_split=minimum,
    )
    ids = _ids_by_split(assignments)
    manifest = FaceRevisedSplitManifest(
        source_fingerprint=source_fingerprint["combined_sha256"],
        canonical_manifest_hash=remediation_report["canonical_manifest_hash"],
        duplicate_policy_hash=remediation_report["duplicate_policy_hash"],
        deduplicated_view_hash=sha256_file(deduplicated_manifest_path, allow_outside_project=True),
        random_seed=seed,
        strategy="deterministic_stratified_by_canonical_emotion_label_70_15_15",
        train_ids=ids["train"],
        validation_ids=ids["validation"],
        test_ids=ids["test"],
        excluded_ids=excluded_ids,
        quarantined_ids=quarantined_ids,
        label_distributions=validation["label_distributions"],
        duplicate_overlap_count=validation["duplicate_overlap_count"],
        image_hash_overlap_count=validation["image_hash_overlap_count"],
        record_overlap_count=validation["record_overlap_count"],
        original_split_overlap_summary=_original_split_comparison(rows, assignments),
        warnings=[
            "Subject identifiers are unavailable; revised split is exact-hash leakage-safe but not subject-independent.",
            "Perceptual near-duplicate leakage risk remains bounded only by diagnostics.",
            "The split is research-only and does not authorize face model deployment.",
        ],
    )
    report = {
        "split_version": FACE_REVISED_SPLIT_VERSION,
        "split_id": FACE_REVISED_SPLIT_ID,
        "strategy": manifest.strategy,
        "random_seed": seed,
        "split_counts": {split: len(ids[split]) for split in FACE_SPLIT_NAMES},
        "label_distributions": validation["label_distributions"],
        "record_overlap_count": validation["record_overlap_count"],
        "image_hash_overlap_count": validation["image_hash_overlap_count"],
        "duplicate_overlap_count": validation["duplicate_overlap_count"],
        "original_split_overlap_summary": manifest.original_split_overlap_summary,
        "warnings": manifest.warnings,
    }

    if validate_only:
        return {"manifest": manifest, "report": report, "assignments": assignments, "outputs": {}}

    output = _ensure_output_dir(output_dir)
    outputs: dict[str, Path] = {}
    outputs["manifest"] = write_json(output / "face_split_manifest.json", manifest.to_safe_dict(), overwrite=overwrite)
    assignment_rows = []
    row_by_id = {row["record_id"]: row for row in rows}
    for record_id, split in sorted(assignments.items()):
        row = row_by_id[record_id]
        assignment_rows.append(
            {
                "record_id": record_id,
                "split": split,
                "canonical_emotion_label": row["canonical_emotion_label"],
                "image_hash": row["image_hash"],
                "duplicate_group_id": row.get("duplicate_group_id", ""),
                "original_split": row.get("source_split", ""),
            }
        )
    outputs["assignments"] = write_csv(
        output / "face_split_assignments.csv",
        assignment_rows,
        fieldnames=["record_id", "split", "canonical_emotion_label", "image_hash", "duplicate_group_id", "original_split"],
        overwrite=overwrite,
    )
    outputs["report_json"] = write_json(output / "face_split_report.json", report, overwrite=overwrite)
    outputs["report_md"] = output / "face_split_report.md"
    if outputs["report_md"].exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {outputs['report_md']}")
    outputs["report_md"].write_text(split_markdown(report), encoding="utf-8")
    outputs["exclusions"] = write_json(
        output / "face_split_exclusions.json",
        {"excluded_ids": excluded_ids, "quarantined_ids": quarantined_ids},
        overwrite=overwrite,
    )
    outputs["hash_isolation"] = write_json(
        output / "face_hash_isolation_report.json",
        {
            "record_overlap_count": validation["record_overlap_count"],
            "image_hash_overlap_count": validation["image_hash_overlap_count"],
            "duplicate_group_overlap_count": validation["duplicate_overlap_count"],
        },
        overwrite=overwrite,
    )
    outputs["original_split_comparison"] = write_json(
        output / "face_original_split_comparison.json",
        manifest.original_split_overlap_summary,
        overwrite=overwrite,
    )
    outputs["class_distribution"] = write_json(
        output / "face_class_distribution.json",
        validation["label_distributions"],
        overwrite=overwrite,
    )
    outputs["artifact_inventory"] = write_json(
        output / "face_artifact_inventory.json",
        {"artifacts": artifact_inventory(outputs)},
        overwrite=overwrite,
    )
    return {"manifest": manifest, "report": report, "assignments": assignments, "outputs": outputs}


def replay_face_v2_split(
    *,
    manifest_path: str | Path,
    deduplicated_manifest_path: str | Path,
    remediation_report_path: str | Path,
    source_fingerprint_path: str | Path,
    policy_config_path: str | Path,
) -> bool:
    manifest = _load_json(manifest_path)
    replay = create_face_v2_split(
        deduplicated_manifest_path=deduplicated_manifest_path,
        remediation_report_path=remediation_report_path,
        source_fingerprint_path=source_fingerprint_path,
        policy_config_path=policy_config_path,
        output_dir=Path(manifest_path).parent,
        seed=int(manifest["random_seed"]),
        validate_only=True,
    )["manifest"].to_safe_dict()
    return (
        replay["train_ids"] == manifest["train_ids"]
        and replay["validation_ids"] == manifest["validation_ids"]
        and replay["test_ids"] == manifest["test_ids"]
    )
