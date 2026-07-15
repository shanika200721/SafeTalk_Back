"""Duplicate analysis for facial emotion preprocessing."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from PIL import Image

from app.ml.preprocessing.face.schemas import FaceCanonicalRecord, FaceDuplicateGroup


def detect_exact_duplicate_groups(records: list[FaceCanonicalRecord]) -> list[FaceDuplicateGroup]:
    grouped: dict[str, list[FaceCanonicalRecord]] = defaultdict(list)
    for record in records:
        grouped[record.image_hash].append(record)
    groups: list[FaceDuplicateGroup] = []
    for digest, items in sorted(grouped.items()):
        if len(items) <= 1:
            continue
        splits = sorted({item.source_split for item in items})
        labels = sorted({item.canonical_emotion_label for item in items})
        groups.append(
            FaceDuplicateGroup(
                image_hash=digest,
                record_ids=sorted(item.record_id for item in items),
                source_splits=splits,
                labels=labels,
                cross_split=len(splits) > 1,
                cross_label=len(labels) > 1,
            )
        )
    return groups


def duplicate_manifest(records: list[FaceCanonicalRecord]) -> dict[str, object]:
    groups = detect_exact_duplicate_groups(records)
    return {
        "duplicate_image_hash_groups": [group.to_safe_dict() for group in groups],
        "duplicate_image_hash_group_count": len(groups),
        "cross_split_duplicate_hash_groups": [group.to_safe_dict() for group in groups if group.cross_split],
        "cross_split_duplicate_hash_group_count": sum(1 for group in groups if group.cross_split),
        "cross_label_duplicate_hash_groups": [group.to_safe_dict() for group in groups if group.cross_label],
        "cross_label_duplicate_hash_group_count": sum(1 for group in groups if group.cross_label),
        "privacy_note": "Duplicate manifests contain record IDs, labels, splits, and hashes only; no image pixels or thumbnails are included.",
    }


def _average_hash(path: Path, *, hash_size: int = 8) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        pixels = list(gray.getdata())
    average = sum(pixels) / len(pixels)
    bits = "".join("1" if pixel >= average else "0" for pixel in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def find_near_duplicate_candidates(paths_by_record_id: dict[str, Path], *, limit: int = 0) -> list[dict[str, object]]:
    if limit <= 0:
        return []
    selected = sorted(paths_by_record_id.items())[:limit]
    buckets: dict[str, list[str]] = defaultdict(list)
    for record_id, path in selected:
        try:
            buckets[_average_hash(path)].append(record_id)
        except Exception:
            continue
    return [
        {"perceptual_hash": digest, "record_ids": sorted(record_ids), "candidate_count": len(record_ids)}
        for digest, record_ids in sorted(buckets.items())
        if len(record_ids) > 1
    ]

