"""Dataset fingerprint service for Phase 2 read-only source manifests."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from pydantic.v1 import ValidationError

from app.ml.common import paths
from app.ml.common.hashing import (
    DATASET_FINGERPRINT_VERSION,
    create_directory_fingerprint,
    create_file_fingerprint,
    hash_json_data,
)
from app.ml.common.schemas import DatasetConfig, DatasetFingerprint, SupportedFileFormat
from app.ml.common.serialization import load_dataset_fingerprint as _load_dataset_fingerprint
from app.ml.common.serialization import save_schema_json


_FILE_FORMAT_EXTENSIONS = {
    SupportedFileFormat.CSV: {".csv"},
    SupportedFileFormat.TSV: {".tsv", ".tab", ".csv"},
    SupportedFileFormat.JSON: {".json"},
    SupportedFileFormat.JSONL: {".jsonl"},
    SupportedFileFormat.XLSX: {".xlsx"},
    SupportedFileFormat.TXT: {".txt"},
    SupportedFileFormat.WAV: {".wav"},
    SupportedFileFormat.MP3: {".mp3"},
    SupportedFileFormat.FLAC: {".flac"},
    SupportedFileFormat.JPG: {".jpg"},
    SupportedFileFormat.JPEG: {".jpeg", ".jpg"},
    SupportedFileFormat.PNG: {".png"},
}


def _allow_test_paths(dataset_config: DatasetConfig) -> bool:
    return getattr(dataset_config, "validation_context", None) == "test"


def _source_report_root(source_path: Path, *, allow_outside_project: bool) -> Path:
    repository_root = paths.get_repository_root().resolve(strict=False)
    if paths.is_path_inside(repository_root, source_path):
        return repository_root
    if allow_outside_project:
        return source_path if source_path.is_dir() else source_path.parent
    return repository_root


def _operational_config_payload(dataset_config: DatasetConfig) -> dict:
    payload = dataset_config.to_safe_dict()
    payload.pop("notes", None)
    return payload


def dataset_config_hash(dataset_config: DatasetConfig) -> str:
    """Return a deterministic hash of operational dataset config fields."""
    return hash_json_data(_operational_config_payload(dataset_config))


def _validate_source_compatible(dataset_config: DatasetConfig, source_path: Path) -> None:
    file_format = dataset_config.file_format
    if file_format == SupportedFileFormat.FOLDER:
        if not source_path.is_dir():
            raise ValueError(f"file_format=folder requires a directory source: {source_path}")
        return

    if not source_path.is_file():
        raise ValueError(f"file_format={file_format.value} requires a file source: {source_path}")

    allowed_extensions = _FILE_FORMAT_EXTENSIONS[file_format]
    extension = source_path.suffix.lower()
    if extension not in allowed_extensions:
        expected = ", ".join(sorted(allowed_extensions))
        raise ValueError(f"Source extension {extension or '<none>'} is not compatible with {file_format.value}; expected {expected}")


def fingerprint_source_path(
    dataset_config: DatasetConfig,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
    include_modified_time: bool = False,
    allow_empty: bool = False,
) -> dict:
    """Fingerprint only the source path described by ``dataset_config``."""
    source_path = dataset_config.validate_source_exists()
    _validate_source_compatible(dataset_config, source_path)
    allow_outside_project = _allow_test_paths(dataset_config)
    report_root = _source_report_root(source_path, allow_outside_project=allow_outside_project)

    if dataset_config.file_format == SupportedFileFormat.FOLDER:
        return create_directory_fingerprint(
            source_path,
            allowed_extensions=allowed_extensions,
            root=report_root,
            include_modified_time=include_modified_time,
            allow_empty=allow_empty,
            allow_outside_project=allow_outside_project,
        )

    file_entry = create_file_fingerprint(
        source_path,
        root=report_root,
        include_modified_time=include_modified_time,
        allow_outside_project=allow_outside_project,
    )
    return {
        "source_relative_path": file_entry["relative_path"],
        "source_type": "file",
        "file_count": 1,
        "total_bytes": file_entry["size_bytes"],
        "combined_sha256": file_entry["sha256"],
        "files": [file_entry],
        "skipped_files": [],
        "skipped_file_count": 0,
    }


def fingerprint_dataset(
    dataset_config: DatasetConfig,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
    include_modified_time: bool = False,
    allow_empty: bool = False,
    generated_at: Optional[datetime] = None,
) -> DatasetFingerprint:
    """Create a typed read-only dataset fingerprint from a validated config."""
    source_fingerprint = fingerprint_source_path(
        dataset_config,
        allowed_extensions=allowed_extensions,
        include_modified_time=include_modified_time,
        allow_empty=allow_empty,
    )
    timestamp = generated_at or datetime.now(timezone.utc)
    return DatasetFingerprint(
        dataset_name=dataset_config.dataset_name,
        dataset_version=dataset_config.dataset_version,
        modality=dataset_config.modality,
        source_relative_path=source_fingerprint["source_relative_path"],
        source_type=source_fingerprint["source_type"],
        file_count=source_fingerprint["file_count"],
        total_bytes=source_fingerprint["total_bytes"],
        combined_sha256=source_fingerprint["combined_sha256"],
        files=source_fingerprint["files"],
        skipped_files=source_fingerprint["skipped_files"],
        generated_at=timestamp,
        fingerprint_version=DATASET_FINGERPRINT_VERSION,
        config_hash=dataset_config_hash(dataset_config),
        notes=dataset_config.notes,
        allow_empty=allow_empty,
    )


def _default_fingerprint_output_path(fingerprint: DatasetFingerprint) -> Path:
    filename = f"{fingerprint.dataset_name}-{fingerprint.dataset_version}.json"
    return paths.get_generated_manifests_root() / "fingerprints" / filename


def _resolve_fingerprint_output_path(output_path: str | os.PathLike[str] | Path) -> Path:
    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    resolved = candidate.resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    allowed_roots = (
        paths.get_generated_manifests_root(),
        paths.get_generated_manifests_root() / "fingerprints",
        paths.get_ml_research_root() / "manifests",
    )
    if not any(paths.is_path_inside(root, resolved) for root in allowed_roots):
        raise ValueError("Dataset fingerprint reports must be saved under generated/manifests/ or ml-research/manifests/")
    return resolved


def save_dataset_fingerprint(
    fingerprint: DatasetFingerprint,
    output_path: str | os.PathLike[str] | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a dataset fingerprint as deterministic JSON."""
    resolved = _default_fingerprint_output_path(fingerprint) if output_path is None else _resolve_fingerprint_output_path(output_path)
    return save_schema_json(fingerprint, resolved, overwrite=overwrite)


def load_dataset_fingerprint(path: str | os.PathLike[str] | Path) -> DatasetFingerprint:
    """Load a saved dataset fingerprint report."""
    return _load_dataset_fingerprint(path)


def _fingerprint_identity_payload(fingerprint: DatasetFingerprint) -> dict:
    return {
        "dataset_name": fingerprint.dataset_name,
        "dataset_version": fingerprint.dataset_version,
        "modality": fingerprint.modality,
        "source_relative_path": fingerprint.source_relative_path,
        "source_type": fingerprint.source_type,
        "file_count": fingerprint.file_count,
        "total_bytes": fingerprint.total_bytes,
        "combined_sha256": fingerprint.combined_sha256,
        "files": [file.to_safe_dict() for file in fingerprint.files],
        "skipped_files": [skipped.to_safe_dict() for skipped in fingerprint.skipped_files],
        "fingerprint_version": fingerprint.fingerprint_version,
        "config_hash": fingerprint.config_hash,
    }


def verify_dataset_fingerprint(fingerprint: DatasetFingerprint, dataset_config: DatasetConfig) -> bool:
    """Return True when the saved fingerprint matches current source files."""
    try:
        current = fingerprint_dataset(dataset_config)
    except (FileNotFoundError, ValueError, ValidationError):
        return False
    return _fingerprint_identity_payload(current) == _fingerprint_identity_payload(fingerprint)


def verify_fingerprint_against_path(fingerprint: DatasetFingerprint, source_root: str | os.PathLike[str] | Path) -> bool:
    """Verify a fingerprint against a direct source path without a config."""
    source_path = Path(source_root).resolve(strict=False)
    try:
        if fingerprint.source_type == "directory":
            source_payload = create_directory_fingerprint(
                source_path,
                root=paths.get_repository_root() if paths.is_path_inside(paths.get_repository_root(), source_path) else source_path,
                allow_outside_project=not paths.is_path_inside(paths.get_repository_root(), source_path),
            )
            current_payload = fingerprint.to_safe_dict()
            current_payload.update(
                {
                    "source_relative_path": source_payload["source_relative_path"],
                    "source_type": "directory",
                    "file_count": source_payload["file_count"],
                    "total_bytes": source_payload["total_bytes"],
                    "combined_sha256": source_payload["combined_sha256"],
                    "files": source_payload["files"],
                    "skipped_files": source_payload["skipped_files"],
                }
            )
            current = DatasetFingerprint(**current_payload)
        else:
            file_payload = create_file_fingerprint(
                source_path,
                root=paths.get_repository_root() if paths.is_path_inside(paths.get_repository_root(), source_path) else source_path.parent,
                allow_outside_project=not paths.is_path_inside(paths.get_repository_root(), source_path),
            )
            current_payload = fingerprint.to_safe_dict()
            current_payload.update(
                {
                    "source_relative_path": file_payload["relative_path"],
                    "source_type": "file",
                    "file_count": 1,
                    "total_bytes": file_payload["size_bytes"],
                    "combined_sha256": file_payload["sha256"],
                    "files": [file_payload],
                    "skipped_files": [],
                }
            )
            current = DatasetFingerprint(**current_payload)
    except (FileNotFoundError, ValueError, ValidationError):
        return False

    return _fingerprint_identity_payload(current) == _fingerprint_identity_payload(fingerprint)
