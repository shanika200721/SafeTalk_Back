"""Validation helpers for facial emotion preprocessing."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from app.ml.preprocessing.face.constants import SUPPORTED_IMAGE_EXTENSIONS
from app.ml.preprocessing.face.duplicates import detect_exact_duplicate_groups
from app.ml.preprocessing.face.image_io import extract_image_metadata
from app.ml.preprocessing.face.mapping import normalize_face_label, validate_face_source_structure
from app.ml.preprocessing.face.schemas import FaceCanonicalRecord, FaceFeatureRecord, FaceLabelMappingConfig, FaceSourceStructureConfig


def validate_face_source_paths(source_structure: FaceSourceStructureConfig) -> dict[str, object]:
    return validate_face_source_structure(source_structure)


def validate_image_extensions(files: Iterable[Path]) -> dict[str, int]:
    extensions = Counter(path.suffix.lower() or "<none>" for path in files)
    unsupported = sorted(ext for ext in extensions if ext not in SUPPORTED_IMAGE_EXTENSIONS)
    if unsupported:
        raise ValueError(f"Unsupported facial image extensions: {unsupported}")
    return dict(sorted(extensions.items()))


def validate_image_dimensions(records: Iterable[FaceCanonicalRecord]) -> None:
    for record in records:
        if record.metadata.readable and (record.metadata.width <= 0 or record.metadata.height <= 0):
            raise ValueError(f"Invalid facial image dimensions: {record.record_id}")


def validate_color_modes(records: Iterable[FaceCanonicalRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.metadata.color_mode for record in records if record.metadata.readable).items()))


def validate_labels(records, label_mapping: FaceLabelMappingConfig) -> None:
    for record in records:
        normalize_face_label(record.original_label, label_mapping)


def detect_corrupt_images(files: Iterable[Path]) -> list[str]:
    corrupt = []
    for path in files:
        metadata = extract_image_metadata(path)
        if not metadata.readable and path.stat().st_size > 0:
            corrupt.append(path.name)
    return corrupt


def detect_zero_byte_images(files: Iterable[Path]) -> list[str]:
    return [path.name for path in files if path.exists() and path.stat().st_size == 0]


def detect_exact_duplicates(records: list[FaceCanonicalRecord]):
    return detect_exact_duplicate_groups(records)


def detect_cross_split_duplicates(records: list[FaceCanonicalRecord]):
    return [group for group in detect_exact_duplicate_groups(records) if group.cross_split]


def detect_cross_label_duplicates(records: list[FaceCanonicalRecord]):
    return [group for group in detect_exact_duplicate_groups(records) if group.cross_label]


def detect_subject_leakage_risk(records: Iterable[FaceCanonicalRecord]) -> dict[str, object]:
    keys = [record.safe_subject_key for record in records if record.safe_subject_key]
    return {
        "subject_identifiers_available": bool(keys),
        "subject_independent_splitting_possible": len(set(keys)) >= 2,
        "subject_count": len(set(keys)) if keys else None,
        "risk": "unknown_without_subject_ids",
        "notes": "Current folder/filename structure does not document subject IDs; predefined train/test cannot be assumed subject-independent.",
    }


def validate_feature_values(records: Iterable[FaceFeatureRecord]) -> None:
    for record in records:
        for name, value in record.feature_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Face feature contains NaN or infinity: {record.record_id}:{name}")


def validate_predefined_split_integrity(records: Iterable[FaceCanonicalRecord]) -> dict[str, object]:
    by_split = Counter(record.source_split for record in records)
    cross_split = detect_cross_split_duplicates(list(records))
    return {
        "split_distribution": dict(sorted(by_split.items())),
        "cross_split_duplicate_count": len(cross_split),
        "leakage_note": "Predefined split folders are retained as metadata only; duplicate overlap is flagged and no new split is created.",
    }

