"""Canonical read-only preprocessing for facial emotion datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common import paths
from app.ml.common.schemas import DatasetFingerprint
from app.ml.preprocessing.face.constants import (
    FACE_FEATURE_SCHEMA_VERSION,
    FACE_IMAGE_POLICY_VERSION,
    FACE_LABEL_MAPPING_VERSION,
    FACE_PREPROCESSING_VERSION,
    FACE_STATISTIC_COLUMNS,
    RECORD_ID_PREFIX,
    SAFE_SUBJECT_KEY_PREFIX,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from app.ml.preprocessing.face.duplicates import duplicate_manifest, find_near_duplicate_candidates
from app.ml.preprocessing.face.features import extract_lightweight_image_statistics, face_feature_schema_payload
from app.ml.preprocessing.face.image_io import convert_image_deterministic, extract_image_metadata
from app.ml.preprocessing.face.mapping import (
    normalize_face_label,
    parse_face_source_path,
    validate_face_source_structure,
)
from app.ml.preprocessing.face.reporting import create_face_preprocessing_markdown
from app.ml.preprocessing.face.schemas import (
    FaceCanonicalRecord,
    FaceFeatureRecord,
    FaceLabelMappingConfig,
    FacePreprocessingReport,
    FaceSourceStructureConfig,
)
from app.ml.preprocessing.face.validation import (
    detect_subject_leakage_risk,
    validate_feature_values,
    validate_image_dimensions,
)


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (paths.get_repository_root() / candidate).resolve(strict=False)


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(paths.get_repository_root()).as_posix()
    except ValueError:
        return path.name


def _fingerprint_value(source_fingerprint: DatasetFingerprint | str | None) -> str:
    if source_fingerprint is None:
        return "0" * 64
    return source_fingerprint.combined_sha256 if isinstance(source_fingerprint, DatasetFingerprint) else str(source_fingerprint)


def discover_face_images(
    source_structure: FaceSourceStructureConfig,
    *,
    max_files: int | None = None,
    source_split: str | None = None,
) -> list[Path]:
    root = _resolve_project_path(source_structure.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Facial dataset root does not exist: {source_structure.dataset_root}")
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in source_structure.supported_image_extensions}
    files: list[Path] = []
    splits = [source_split] if source_split else source_structure.predefined_split_folders
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise ValueError(f"Unknown or missing facial source split: {split}")
        files.extend(path for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in extensions)
    files = sorted(files)
    return files[:max_files] if max_files is not None else files


def parse_face_record(path: str | Path, source_structure: FaceSourceStructureConfig):
    return parse_face_source_path(path, source_structure)


def extract_face_metadata(path: str | Path):
    return extract_image_metadata(path)


def generate_face_record_id(source_relative_path: str, source_fingerprint: str) -> str:
    if len(source_fingerprint) < 12:
        raise ValueError("source_fingerprint must be available for deterministic face record IDs")
    digest = hashlib.sha256(f"{RECORD_ID_PREFIX}:{source_relative_path}:{source_fingerprint}".encode("utf-8")).hexdigest()[:16]
    return f"{RECORD_ID_PREFIX}-{digest}"


def generate_safe_subject_key(subject_id: str, source_fingerprint: str) -> str:
    if not str(subject_id).strip():
        raise ValueError("subject_id is required")
    digest = hashlib.sha256(f"{SAFE_SUBJECT_KEY_PREFIX}:{subject_id}:{source_fingerprint}".encode("utf-8")).hexdigest()[:16]
    return f"{SAFE_SUBJECT_KEY_PREFIX}-{digest}"


def apply_face_duplicate_policy(records: list[FaceCanonicalRecord]) -> dict[str, object]:
    return duplicate_manifest(records)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    series = pd.Series(values, dtype="float64")
    return {
        "min": round(float(series.min()), 6),
        "max": round(float(series.max()), 6),
        "mean": round(float(series.mean()), 6),
        "median": round(float(series.median()), 6),
        "p25": round(float(series.quantile(0.25)), 6),
        "p75": round(float(series.quantile(0.75)), 6),
        "p95": round(float(series.quantile(0.95)), 6),
    }


def _write_json(payload: dict[str, Any], output_path: Path, *, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(output_path)
    return output_path


def _write_csv(rows: list[dict[str, Any]], output_path: Path, *, overwrite: bool, fieldnames: list[str] | None = None) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _canonical_row(record: FaceCanonicalRecord) -> dict[str, Any]:
    metadata = record.metadata
    return {
        "record_id": record.record_id,
        "source_split": record.source_split,
        "original_label": record.original_label,
        "canonical_emotion_label": record.canonical_emotion_label,
        "image_relative_path": record.image_relative_path,
        "image_hash": record.image_hash,
        "safe_subject_key": record.safe_subject_key,
        "width": metadata.width,
        "height": metadata.height,
        "color_mode": metadata.color_mode,
        "file_format": metadata.file_format,
        "file_size_bytes": metadata.file_size_bytes,
        "readable": metadata.readable,
        "validation_warnings": "|".join(record.validation_warnings + metadata.validation_warnings),
    }


def _feature_row(record: FaceFeatureRecord) -> dict[str, Any]:
    row = {
        "record_id": record.record_id,
        "canonical_emotion_label": record.canonical_emotion_label,
        "feature_extraction_warnings": "|".join(record.feature_extraction_warnings),
    }
    row.update(record.feature_values)
    return row


def build_face_canonical_manifest(records: list[FaceCanonicalRecord]) -> list[dict[str, Any]]:
    return [_canonical_row(record) for record in records]


def optionally_write_generated_image(
    path: Path,
    record: FaceCanonicalRecord,
    *,
    target_width: int,
    target_height: int,
    color_mode: str,
    overwrite: bool,
) -> dict[str, object]:
    generated_root = paths.get_generated_preprocessing_root() / "face" / "v1" / "images"
    target = generated_root / f"{record.record_id}.png"
    manifest = convert_image_deterministic(
        path,
        target,
        target_width=target_width,
        target_height=target_height,
        color_mode=color_mode,
        overwrite=overwrite,
        center_crop=False,
    )
    manifest.update(
        {
            "record_id": record.record_id,
            "source_image_hash": record.image_hash,
            "generated_image_relative_path": _relative_to_repo(target),
            "source_not_overwritten": True,
        }
    )
    return manifest


def preprocess_face_dataset(
    source_structure_config: FaceSourceStructureConfig,
    label_mapping_config: FaceLabelMappingConfig,
    *,
    source_fingerprint: DatasetFingerprint | str | None = None,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    max_files: int | None = None,
    source_split: str | None = None,
    compute_image_statistics: bool = False,
    write_normalized_images: bool = False,
    target_width: int = 48,
    target_height: int = 48,
    color_mode: str = "L",
    near_duplicate_limit: int = 0,
) -> dict[str, Any]:
    validate_face_source_structure(source_structure_config, label_mapping_config)
    files = discover_face_images(source_structure_config, max_files=max_files, source_split=source_split)
    source_digest = _fingerprint_value(source_fingerprint)
    canonical_records: list[FaceCanonicalRecord] = []
    feature_records: list[FaceFeatureRecord] = []
    corrupt_files: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    normalized_images: list[dict[str, object]] = []
    paths_by_record_id: dict[str, Path] = {}

    seen_paths: set[str] = set()
    for image_path in files:
        relative_path = _relative_to_repo(image_path)
        if relative_path in seen_paths:
            raise ValueError(f"Duplicate source image path discovered: {relative_path}")
        seen_paths.add(relative_path)
        try:
            source = parse_face_record(image_path, source_structure_config)
            canonical_label = normalize_face_label(source.original_label, label_mapping_config)
        except Exception as exc:
            excluded.append({"source_path_hash": hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16], "reason": f"parse_or_mapping_error:{exc.__class__.__name__}"})
            continue
        metadata = extract_face_metadata(image_path)
        record_id = generate_face_record_id(relative_path, source_digest)
        safe_subject_key = generate_safe_subject_key(source.subject_id, source_digest) if source.subject_id else None
        if not metadata.readable:
            corrupt_files.append(
                {
                    "record_path_hash": hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16],
                    "source_split": source.source_split,
                    "original_label": source.original_label,
                    "file_size_bytes": metadata.file_size_bytes,
                    "warnings": metadata.validation_warnings,
                }
            )
            excluded.append({"source_path_hash": hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16], "reason": "unreadable_or_corrupt_image"})
            continue
        record = FaceCanonicalRecord(
            record_id=record_id,
            source_split=source.source_split,
            original_label=source.original_label,
            canonical_emotion_label=canonical_label,
            image_relative_path=relative_path,
            image_hash=metadata.image_hash,
            metadata=metadata,
            safe_subject_key=safe_subject_key,
            validation_warnings=[],
        )
        canonical_records.append(record)
        paths_by_record_id[record.record_id] = image_path
        if compute_image_statistics:
            feature_values, warnings = extract_lightweight_image_statistics(image_path)
            feature_records.append(
                FaceFeatureRecord(
                    record_id=record.record_id,
                    canonical_emotion_label=record.canonical_emotion_label,
                    feature_values=feature_values,
                    feature_extraction_warnings=warnings,
                )
            )
        if write_normalized_images:
            normalized_images.append(
                optionally_write_generated_image(
                    image_path,
                    record,
                    target_width=target_width,
                    target_height=target_height,
                    color_mode=color_mode,
                    overwrite=overwrite,
                )
            )

    canonical_records.sort(key=lambda item: (item.source_split, item.canonical_emotion_label, item.image_relative_path, item.record_id))
    feature_records.sort(key=lambda item: item.record_id)
    validate_preprocessed_face(canonical_records, feature_records)
    duplicates = apply_face_duplicate_policy(canonical_records)
    near_candidates = find_near_duplicate_candidates(paths_by_record_id, limit=near_duplicate_limit)
    widths = [float(record.metadata.width) for record in canonical_records if record.metadata.readable]
    heights = [float(record.metadata.height) for record in canonical_records if record.metadata.readable]
    split_distribution = dict(sorted(Counter(record.source_split for record in canonical_records).items()))
    label_distribution = dict(sorted(Counter(record.canonical_emotion_label for record in canonical_records).items()))
    color_modes = dict(sorted(Counter(record.metadata.color_mode for record in canonical_records).items()))
    formats = dict(sorted(Counter(record.metadata.file_format for record in canonical_records).items()))
    subject_risk = detect_subject_leakage_risk(canonical_records)
    feature_missing_summary = {name: len(canonical_records) for name in FACE_STATISTIC_COLUMNS}
    if compute_image_statistics:
        feature_missing_summary = {name: int(sum(1 for record in feature_records if name not in record.feature_values)) for name in FACE_STATISTIC_COLUMNS}
    report = FacePreprocessingReport(
        preprocessing_version=FACE_PREPROCESSING_VERSION,
        feature_schema_version=FACE_FEATURE_SCHEMA_VERSION,
        label_mapping_version=FACE_LABEL_MAPPING_VERSION,
        image_policy_version=FACE_IMAGE_POLICY_VERSION,
        source_fingerprint=source_digest,
        source_file_count=len(files),
        readable_file_count=len(canonical_records),
        unreadable_file_count=len(corrupt_files),
        output_record_count=len(canonical_records),
        excluded_record_count=len(excluded),
        split_distribution=split_distribution,
        label_distribution=label_distribution,
        width_summary=_summary(widths),
        height_summary=_summary(heights),
        color_mode_distribution=color_modes,
        format_distribution=formats,
        duplicate_group_count=int(duplicates["duplicate_image_hash_group_count"]),
        cross_split_duplicate_count=int(duplicates["cross_split_duplicate_hash_group_count"]),
        cross_label_duplicate_count=int(duplicates["cross_label_duplicate_hash_group_count"]),
        corrupt_file_count=len([item for item in corrupt_files if item["file_size_bytes"] > 0]),
        zero_byte_file_count=len([item for item in corrupt_files if item["file_size_bytes"] == 0]),
        subject_count=subject_risk["subject_count"],
        warnings=[
            "Facial emotion recognition is not depression diagnosis or suicide-risk prediction.",
            "Facial expressions are culturally and individually variable; acted expressions may not reflect natural distress.",
            "Lighting, pose, camera quality, occlusion, skin tone, and demographics can affect performance.",
            "Facial data are biometric and sensitive; explicit consent is required for webcam use.",
            "Predefined train/test folders are retained as metadata only and are not assumed leakage-free.",
            "Subject identifiers are not documented in this folder structure; subject-independent splitting is not currently ready.",
            "No raw image files were modified and no train/validation/test split, scaler, model, embedding, facial recognition, live inference, alert, or treatment recommendation was created.",
        ],
    )

    outputs: dict[str, str] = {}
    if not validate_only:
        resolved_output_dir = output_dir.resolve(strict=False)
        paths.assert_not_raw_dataset_path(resolved_output_dir)
        if not paths.is_path_inside(paths.get_generated_root(), resolved_output_dir):
            raise ValueError("Face preprocessing outputs must be under generated/")
        canonical_rows = build_face_canonical_manifest(canonical_records)
        feature_rows = [_feature_row(record) for record in feature_records] if compute_image_statistics else []
        feature_fieldnames = ["record_id", "canonical_emotion_label", *FACE_STATISTIC_COLUMNS, "feature_extraction_warnings"]
        cross_split = {"cross_split_duplicate_hash_groups": duplicates["cross_split_duplicate_hash_groups"]}
        cross_label = {
            "cross_label_duplicate_hash_groups": duplicates["cross_label_duplicate_hash_groups"],
            "severity": "critical" if duplicates["cross_label_duplicate_hash_group_count"] else "none",
            "policy": "quarantine/report only; no source images deleted or relabeled",
        }
        record_manifest = {
            "dataset": "facial-emotion",
            "record_count": len(canonical_records),
            "record_id_strategy": "hash(source-relative path, source fingerprint); filenames and hashes excluded from predictive features",
            "records": [
                {
                    "record_id": record.record_id,
                    "source_split": record.source_split,
                    "canonical_emotion_label": record.canonical_emotion_label,
                    "image_hash": record.image_hash,
                }
                for record in canonical_records
            ],
        }
        outputs = {
            "canonical_manifest_csv": str(_write_csv(canonical_rows, resolved_output_dir / "face_canonical_manifest.csv", overwrite=overwrite)),
            "feature_schema_json": str(_write_json(face_feature_schema_payload(), resolved_output_dir / "face_feature_schema.json", overwrite=overwrite)),
            "report_json": str(_write_json(report.to_safe_dict(), resolved_output_dir / "face_preprocessing_report.json", overwrite=overwrite)),
            "record_manifest_json": str(_write_json(record_manifest, resolved_output_dir / "face_record_manifest.json", overwrite=overwrite)),
            "corrupt_files_json": str(_write_json({"corrupt_or_unreadable_files": corrupt_files}, resolved_output_dir / "face_corrupt_files.json", overwrite=overwrite)),
            "duplicate_manifest_json": str(_write_json({**duplicates, "near_duplicate_candidates": near_candidates}, resolved_output_dir / "face_duplicate_manifest.json", overwrite=overwrite)),
            "cross_split_overlap_json": str(_write_json(cross_split, resolved_output_dir / "face_cross_split_overlap.json", overwrite=overwrite)),
            "cross_label_conflicts_json": str(_write_json(cross_label, resolved_output_dir / "face_cross_label_conflicts.json", overwrite=overwrite)),
            "label_distribution_json": str(_write_json(label_distribution, resolved_output_dir / "face_label_distribution.json", overwrite=overwrite)),
        }
        if compute_image_statistics:
            outputs["image_statistics_csv"] = str(_write_csv(feature_rows, resolved_output_dir / "face_image_statistics.csv", overwrite=overwrite, fieldnames=feature_fieldnames))
        md_path = resolved_output_dir / "face_preprocessing_report.md"
        if md_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {md_path}")
        md_path.write_text(create_face_preprocessing_markdown(report), encoding="utf-8")
        outputs["report_markdown"] = str(md_path)
        if write_normalized_images:
            outputs["normalized_images_manifest_json"] = str(
                _write_json({"normalized_images": normalized_images}, resolved_output_dir / "face_normalized_images_manifest.json", overwrite=overwrite)
            )

    return {
        "valid": True,
        "critical_conflicts": int(duplicates["cross_label_duplicate_hash_group_count"]),
        "validate_only": validate_only,
        "compute_image_statistics": compute_image_statistics,
        "write_normalized_images": write_normalized_images,
        "source_files": len(files),
        "output_records": len(canonical_records),
        "excluded_records": len(excluded),
        "readable_files": report.readable_file_count,
        "unreadable_files": report.unreadable_file_count,
        "split_distribution": split_distribution,
        "label_distribution": label_distribution,
        "width_summary": report.width_summary,
        "height_summary": report.height_summary,
        "color_mode_distribution": color_modes,
        "format_distribution": formats,
        "duplicate_summary": {
            "duplicate_group_count": report.duplicate_group_count,
            "cross_split_duplicate_count": report.cross_split_duplicate_count,
            "cross_label_duplicate_count": report.cross_label_duplicate_count,
            "near_duplicate_candidate_count": len(near_candidates),
            "subject_leakage_risk": subject_risk,
        },
        "feature_missing_summary": feature_missing_summary,
        "report": report,
        "outputs": outputs,
    }


def validate_preprocessed_face(canonical_records: list[FaceCanonicalRecord], feature_records: list[FaceFeatureRecord] | None = None) -> None:
    validate_image_dimensions(canonical_records)
    for record in canonical_records:
        if record.safe_subject_key and record.safe_subject_key in record.image_relative_path:
            raise ValueError("safe subject keys must not expose raw subject identifiers")
        if Path(record.image_relative_path).is_absolute():
            raise ValueError("face reports and manifests must use relative paths")
    if feature_records:
        validate_feature_values(feature_records)
        blocked = {"record_id", "source_split", "image_relative_path", "image_hash", "safe_subject_key", "canonical_emotion_label"}
        if blocked & set(FACE_STATISTIC_COLUMNS):
            raise ValueError("Face statistic columns contain metadata leakage")

