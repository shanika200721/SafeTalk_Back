"""Reusable deterministic split helpers for Phase 3A."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from app.ml.common.hashing import hash_json_data, sha256_file
from app.ml.splitting.constants import SPLIT_DESIGN_VERSION, SPLIT_MANIFEST_VERSION, SPLIT_NAMES
from app.ml.splitting.schemas import ModalitySplitManifest, SplitRecord, SplitStrategy, SplitValidationSummary


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def save_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def validate_split_proportions(train: float, validation: float, test: float) -> None:
    values = {"train": train, "validation": validation, "test": test}
    for name, value in values.items():
        if value < 0 or value > 1:
            raise ValueError(f"{name} proportion must be between 0 and 1")
    if train <= 0 or test <= 0:
        raise ValueError("train and test proportions must be greater than 0")
    if abs((train + validation + test) - 1.0) > 1e-9:
        raise ValueError("split proportions must sum to 1.0")


def calculate_split_targets(total_count: int, train: float, validation: float, test: float) -> dict[str, int]:
    validate_split_proportions(train, validation, test)
    if total_count <= 0:
        raise ValueError("total_count must be positive")
    proportions = {"train": train, "validation": validation, "test": test}
    raw = {name: total_count * proportion for name, proportion in proportions.items()}
    targets = {name: int(value) for name, value in raw.items()}
    remainder = total_count - sum(targets.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (raw[name] - targets[name], name), reverse=True)
    for name in order[:remainder]:
        targets[name] += 1
    for name in ("train", "test"):
        if targets[name] == 0:
            raise ValueError(f"{name} split would be empty")
    return targets


def _stable_key(value: Any, seed: int) -> str:
    return hash_json_data({"seed": seed, "value": str(value)})


def deterministic_shuffle(values: Iterable[Any], seed: int) -> list[Any]:
    return sorted([str(value) for value in values], key=lambda value: (_stable_key(value, seed), value))


def _targets_for_labels(label_counts: Counter[str], train: float, validation: float, test: float) -> dict[str, dict[str, int]]:
    targets = {split: {} for split in SPLIT_NAMES}
    for label in sorted(label_counts):
        label_targets = calculate_split_targets(label_counts[label], train, validation, test)
        for split in SPLIT_NAMES:
            targets[split][label] = label_targets[split]
    return targets


def _records_from_frame(
    df: pd.DataFrame,
    *,
    record_id_column: str,
    label_column: str,
    group_column: str | None = None,
    duplicate_column: str | None = None,
    source_column: str | None = None,
) -> list[dict[str, str | None]]:
    records: list[dict[str, str | None]] = []
    for row in df.sort_values(record_id_column, kind="mergesort").to_dict("records"):
        record = {
            "record_id": str(row[record_id_column]),
            "label": str(row[label_column]),
            "group_id": str(row[group_column]) if group_column and pd.notna(row.get(group_column)) else None,
            "duplicate_group_id": str(row[duplicate_column]) if duplicate_column and pd.notna(row.get(duplicate_column)) else None,
            "source_name": str(row[source_column]) if source_column and pd.notna(row.get(source_column)) else None,
        }
        records.append(record)
    return records


def stratified_split(
    df: pd.DataFrame,
    *,
    record_id_column: str,
    label_column: str,
    train_proportion: float,
    validation_proportion: float,
    test_proportion: float,
    seed: int,
    minimum_class_count_per_split: int = 1,
) -> list[SplitRecord]:
    validate_split_proportions(train_proportion, validation_proportion, test_proportion)
    records = _records_from_frame(df, record_id_column=record_id_column, label_column=label_column)
    by_label: dict[str, list[dict[str, str | None]]] = defaultdict(list)
    for record in records:
        by_label[str(record["label"])].append(record)

    assignments: list[SplitRecord] = []
    for label in sorted(by_label):
        label_records = sorted(by_label[label], key=lambda item: str(item["record_id"]))
        if len(label_records) < minimum_class_count_per_split * 3:
            raise ValueError(f"Label {label} has too few records for all three splits")
        shuffled_ids = deterministic_shuffle([record["record_id"] for record in label_records], seed)
        record_by_id = {record["record_id"]: record for record in label_records}
        targets = calculate_split_targets(len(label_records), train_proportion, validation_proportion, test_proportion)
        start = 0
        for split in SPLIT_NAMES:
            selected = shuffled_ids[start : start + targets[split]]
            start += targets[split]
            for record_id in selected:
                record = record_by_id[record_id]
                assignments.append(SplitRecord(record_id=str(record_id), split=split, label=label))
    return sorted(assignments, key=lambda item: item.record_id)


def assign_duplicate_groups(
    duplicate_manifest: Mapping[str, Any] | None,
    *,
    duplicate_groups_key: str = "exact_duplicate_groups",
) -> dict[str, str]:
    if not duplicate_manifest:
        return {}
    mapping: dict[str, str] = {}
    groups = duplicate_manifest.get(duplicate_groups_key, [])
    if isinstance(groups, dict):
        iterable = [{"duplicate_hash": key, "record_ids": value} for key, value in groups.items()]
    else:
        iterable = groups
    for index, group in enumerate(iterable):
        if not isinstance(group, Mapping):
            continue
        group_id = str(group.get("duplicate_hash") or group.get("hash") or f"duplicate-group-{index:06d}")
        for record_id in group.get("record_ids", []) or []:
            mapping[str(record_id)] = group_id
    return mapping


def _build_group_components(records: list[dict[str, str | None]]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    for record in records:
        record_node = f"record:{record['record_id']}"
        find(record_node)
        for column in ("group_id", "duplicate_group_id"):
            value = record.get(column)
            if value:
                union(record_node, f"{column}:{value}")

    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[find(f"record:{record['record_id']}")].append(str(record["record_id"]))
    group_for_record: dict[str, str] = {}
    for index, record_ids in enumerate(sorted(groups.values(), key=lambda ids: (min(ids), len(ids)))):
        group_id = f"split-group-{index:06d}-{hash_json_data(sorted(record_ids))[:12]}"
        for record_id in record_ids:
            group_for_record[record_id] = group_id
    return group_for_record


def _grouped_records(
    df: pd.DataFrame,
    *,
    record_id_column: str,
    label_column: str,
    group_column: str | None,
    duplicate_column: str | None,
    source_column: str | None,
) -> list[dict[str, Any]]:
    records = _records_from_frame(
        df,
        record_id_column=record_id_column,
        label_column=label_column,
        group_column=group_column,
        duplicate_column=duplicate_column,
        source_column=source_column,
    )
    if group_column or duplicate_column:
        component_map = _build_group_components(records)
        for record in records:
            record["split_group_id"] = component_map[str(record["record_id"])]
    else:
        for record in records:
            record["split_group_id"] = str(record["record_id"])
    return records


def grouped_stratified_split(
    df: pd.DataFrame,
    *,
    record_id_column: str,
    label_column: str,
    group_column: str | None,
    duplicate_column: str | None = None,
    source_column: str | None = None,
    train_proportion: float,
    validation_proportion: float,
    test_proportion: float,
    seed: int,
    retry_limit: int = 25,
    minimum_class_count_per_split: int = 1,
) -> list[SplitRecord]:
    validate_split_proportions(train_proportion, validation_proportion, test_proportion)
    if retry_limit <= 0:
        raise ValueError("retry_limit must be positive")

    records = _grouped_records(
        df,
        record_id_column=record_id_column,
        label_column=label_column,
        group_column=group_column,
        duplicate_column=duplicate_column,
        source_column=source_column,
    )
    label_counts = Counter(str(record["label"]) for record in records)
    impossible = [label for label, count in label_counts.items() if count < minimum_class_count_per_split * 3]
    if impossible:
        raise ValueError(f"Labels have too few records for all splits: {sorted(impossible)}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["split_group_id"])].append(record)
    groups = []
    for group_id, group_records in grouped.items():
        groups.append(
            {
                "group_id": group_id,
                "records": sorted(group_records, key=lambda item: str(item["record_id"])),
                "label_counts": Counter(str(item["label"]) for item in group_records),
                "source_counts": Counter(str(item.get("source_name") or "") for item in group_records),
                "size": len(group_records),
            }
        )

    global_targets = calculate_split_targets(len(records), train_proportion, validation_proportion, test_proportion)
    label_targets = _targets_for_labels(label_counts, train_proportion, validation_proportion, test_proportion)
    best_assignments: dict[str, str] | None = None
    best_score: tuple[float, int] | None = None
    last_error = ""

    for attempt in range(retry_limit):
        attempt_seed = seed + attempt
        ordered = sorted(
            groups,
            key=lambda group: (-group["size"], _stable_key(group["group_id"], attempt_seed), group["group_id"]),
        )
        split_counts = Counter()
        split_label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_NAMES}
        assignments: dict[str, str] = {}

        for group in ordered:
            scores = []
            for split in SPLIT_NAMES:
                before_total_deficit = abs(global_targets[split] - split_counts[split])
                total_after = split_counts[split] + group["size"]
                total_delta = abs(global_targets[split] - total_after) - before_total_deficit
                label_delta = 0.0
                for label in sorted(label_counts):
                    before = split_label_counts[split][label]
                    after = split_label_counts[split][label] + group["label_counts"].get(label, 0)
                    label_delta += abs(label_targets[split][label] - after) - abs(label_targets[split][label] - before)
                capacity_penalty = max(0, total_after - global_targets[split]) * 2.0
                empty_label_bonus = 0.0
                for label, count in group["label_counts"].items():
                    if count and split_label_counts[split][label] == 0:
                        empty_label_bonus -= 0.25
                scores.append((total_delta + label_delta + capacity_penalty + empty_label_bonus, split))
            _, chosen_split = min(scores, key=lambda item: (item[0], item[1]))
            assignments[group["group_id"]] = chosen_split
            split_counts[chosen_split] += group["size"]
            for label, count in group["label_counts"].items():
                split_label_counts[chosen_split][label] += count

        missing_labels = [
            f"{split}:{label}"
            for split in SPLIT_NAMES
            for label in sorted(label_counts)
            if split_label_counts[split][label] < minimum_class_count_per_split
        ]
        score_value = sum(abs(split_counts[split] - global_targets[split]) for split in SPLIT_NAMES)
        score_value += sum(
            abs(split_label_counts[split][label] - label_targets[split][label])
            for split in SPLIT_NAMES
            for label in sorted(label_counts)
        )
        score = (float(score_value), len(missing_labels))
        if best_score is None or score < best_score:
            best_score = score
            best_assignments = dict(assignments)
        if not missing_labels:
            best_assignments = dict(assignments)
            break
        last_error = f"missing minimum class presence in {missing_labels[:10]}"

    if best_assignments is None:
        raise ValueError(f"Could not create grouped split: {last_error}")

    split_records: list[SplitRecord] = []
    for group in groups:
        split = best_assignments[group["group_id"]]
        for record in group["records"]:
            split_records.append(
                SplitRecord(
                    record_id=str(record["record_id"]),
                    split=split,
                    label=str(record["label"]),
                    group_id=str(record.get("group_id") or record.get("split_group_id") or ""),
                    source_name=record.get("source_name"),
                    duplicate_group_id=record.get("duplicate_group_id"),
                )
            )
    return sorted(split_records, key=lambda item: item.record_id)


def ids_by_split(assignments: Sequence[SplitRecord]) -> dict[str, list[str]]:
    result = {split: [] for split in SPLIT_NAMES}
    for assignment in sorted(assignments, key=lambda item: item.record_id):
        result[assignment.split].append(assignment.record_id)
    return result


def label_distribution(assignments: Sequence[SplitRecord]) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = {split: Counter() for split in SPLIT_NAMES}
    for assignment in assignments:
        result[assignment.split][assignment.label] += 1
    return {split: dict(sorted(counts.items())) for split, counts in result.items()}


def validate_no_overlap(assignments: Sequence[SplitRecord]) -> int:
    split_to_ids = {split: set(ids) for split, ids in ids_by_split(assignments).items()}
    overlap = 0
    for index, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[index + 1 :]:
            overlap += len(split_to_ids[left] & split_to_ids[right])
    if overlap:
        raise ValueError(f"record IDs overlap across splits: {overlap}")
    return overlap


def _overlap_count_by_field(assignments: Sequence[SplitRecord], field_name: str) -> int:
    values: dict[str, set[str]] = defaultdict(set)
    for assignment in assignments:
        value = getattr(assignment, field_name)
        if value:
            values[str(value)].add(assignment.split)
    return sum(1 for splits in values.values() if len(splits) > 1)


def validate_group_isolation(assignments: Sequence[SplitRecord]) -> int:
    overlap = _overlap_count_by_field(assignments, "group_id")
    if overlap:
        raise ValueError(f"group IDs overlap across splits: {overlap}")
    return overlap


def validate_duplicate_isolation(assignments: Sequence[SplitRecord]) -> int:
    overlap = _overlap_count_by_field(assignments, "duplicate_group_id")
    if overlap:
        raise ValueError(f"duplicate groups overlap across splits: {overlap}")
    return overlap


def validate_manifest_coverage(
    assignments: Sequence[SplitRecord],
    expected_record_ids: Iterable[str],
    excluded_ids: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    expected = {str(record_id) for record_id in expected_record_ids}
    assigned = {assignment.record_id for assignment in assignments}
    excluded = set(str(record_id) for record_id in (excluded_ids or {}))
    missing = expected - assigned - excluded
    unexpected = assigned - expected
    if missing:
        raise ValueError(f"split assignments missing {len(missing)} records")
    if unexpected:
        raise ValueError(f"split assignments contain {len(unexpected)} unexpected records")
    return len(missing), len(unexpected)


def validate_label_distribution(assignments: Sequence[SplitRecord], *, minimum_class_count_per_split: int = 1) -> None:
    labels = sorted({assignment.label for assignment in assignments})
    distributions = label_distribution(assignments)
    missing = [
        f"{split}:{label}"
        for split in SPLIT_NAMES
        for label in labels
        if distributions[split].get(label, 0) < minimum_class_count_per_split
    ]
    if missing:
        raise ValueError(f"class presence requirement failed: {missing}")


def make_validation_summary(
    assignments: Sequence[SplitRecord],
    *,
    expected_record_ids: Iterable[str],
    excluded_ids: Mapping[str, str] | None = None,
    deterministic_replay_passed: bool = False,
    source_overlap_count: int = 0,
    warnings: Sequence[str] | None = None,
    blockers: Sequence[str] | None = None,
) -> SplitValidationSummary:
    missing, unexpected = validate_manifest_coverage(assignments, expected_record_ids, excluded_ids)
    validate_no_overlap(assignments)
    group_overlap = validate_group_isolation(assignments)
    duplicate_overlap = validate_duplicate_isolation(assignments)
    distributions = label_distribution(assignments)
    split_ids = ids_by_split(assignments)
    return SplitValidationSummary(
        train_count=len(split_ids["train"]),
        validation_count=len(split_ids["validation"]),
        test_count=len(split_ids["test"]),
        total_count=len(assignments),
        train_distribution=distributions["train"],
        validation_distribution=distributions["validation"],
        test_distribution=distributions["test"],
        group_overlap_count=group_overlap,
        duplicate_overlap_count=duplicate_overlap,
        source_overlap_count=source_overlap_count,
        missing_record_count=missing,
        unexpected_record_count=unexpected,
        deterministic_replay_passed=deterministic_replay_passed,
        warnings=list(warnings or []),
        blockers=list(blockers or []),
    )


def build_split_manifest(
    *,
    modality: str,
    dataset_name: str,
    dataset_version: str,
    preprocessing_version: str,
    feature_schema_version: str,
    source_fingerprint: str,
    preprocessing_artifact_hash: str,
    config_hash: str,
    random_seed: int,
    split_strategy: SplitStrategy,
    assignments: Sequence[SplitRecord],
    excluded_ids: Mapping[str, str] | None,
    grouping_column: str | None,
    stratify_column: str,
    source_split_policy: str | None,
    duplicate_policy: str,
    validation_summary: SplitValidationSummary,
    notes: Sequence[str] | None = None,
) -> ModalitySplitManifest:
    split_ids = ids_by_split(assignments)
    return ModalitySplitManifest(
        manifest_version=SPLIT_MANIFEST_VERSION,
        split_design_version=SPLIT_DESIGN_VERSION,
        modality=modality,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        preprocessing_version=preprocessing_version,
        feature_schema_version=feature_schema_version,
        source_fingerprint=source_fingerprint,
        preprocessing_artifact_hash=preprocessing_artifact_hash,
        config_hash=config_hash,
        random_seed=random_seed,
        split_strategy=split_strategy,
        train_ids=split_ids["train"],
        validation_ids=split_ids["validation"],
        test_ids=split_ids["test"],
        excluded_ids=dict(sorted((excluded_ids or {}).items())),
        grouping_column=grouping_column,
        stratify_column=stratify_column,
        source_split_policy=source_split_policy,
        duplicate_policy=duplicate_policy,
        created_at=utc_now(),
        validation_summary=validation_summary,
        notes=list(notes or []),
    )


def compute_split_artifact_hash(payload: Any) -> str:
    if isinstance(payload, (str, Path)):
        path = Path(payload)
        if path.exists() and path.is_file():
            return sha256_file(path)
    if hasattr(payload, "to_safe_dict"):
        payload = payload.to_safe_dict()
    if isinstance(payload, dict) and "created_at" in payload:
        payload = dict(payload)
        payload.pop("created_at", None)
    return hash_json_data(payload)


def replay_split_from_manifest(manifest: ModalitySplitManifest, assignments: Sequence[SplitRecord]) -> bool:
    split_ids = ids_by_split(assignments)
    return (
        split_ids["train"] == manifest.train_ids
        and split_ids["validation"] == manifest.validation_ids
        and split_ids["test"] == manifest.test_ids
    )


def assignments_to_frame(assignments: Sequence[SplitRecord]) -> pd.DataFrame:
    rows = [assignment.dict(exclude_none=True) for assignment in sorted(assignments, key=lambda item: item.record_id)]
    return pd.DataFrame(rows, columns=["record_id", "split", "label", "group_id", "source_name", "source_split", "duplicate_group_id", "notes"])


def write_assignments_csv(assignments: Sequence[SplitRecord], output_path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing assignments: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    assignments_to_frame(assignments).to_csv(path, index=False)
    return path
