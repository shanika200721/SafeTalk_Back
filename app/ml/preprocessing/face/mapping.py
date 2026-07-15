"""Folder and label mapping helpers for facial emotion preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.common import paths
from app.ml.preprocessing.face.constants import CANONICAL_EMOTION_LABELS, PREDEFINED_SPLITS, SUPPORTED_IMAGE_EXTENSIONS
from app.ml.preprocessing.face.schemas import FaceLabelMappingConfig, FaceLabelMappingEntry, FaceSourceRecord, FaceSourceStructureConfig


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


def default_face_label_mapping_config() -> FaceLabelMappingConfig:
    return FaceLabelMappingConfig(
        canonical_labels=list(CANONICAL_EMOTION_LABELS),
        entries=[
            FaceLabelMappingEntry(original_label=label, canonical_label=label, notes="Confirmed class folder; retained without merging.")
            for label in CANONICAL_EMOTION_LABELS
        ],
        notes="Actual FER-style folder labels are retained one-to-one. No class merging is performed.",
    )


def load_face_label_mapping(path: str | Path | None) -> FaceLabelMappingConfig:
    if path is None:
        return default_face_label_mapping_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return FaceLabelMappingConfig(**payload)


def load_face_source_structure(path: str | Path) -> FaceSourceStructureConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return FaceSourceStructureConfig(**payload)


def normalize_face_label(original_label: str, mapping_config: FaceLabelMappingConfig) -> str:
    for entry in mapping_config.entries:
        if entry.original_label == original_label:
            if entry.excluded or not entry.retained:
                raise ValueError(f"Face label is configured as excluded: {original_label}")
            return entry.canonical_label
    raise ValueError(f"Unknown face emotion label: {original_label}")


def identify_predefined_split(path: str | Path, dataset_root: str | Path, split_names: list[str] | tuple[str, ...] = PREDEFINED_SPLITS) -> str:
    root = _resolve_project_path(dataset_root)
    candidate = Path(path).resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image path is outside facial dataset root: {path}") from exc
    if len(relative.parts) < 3:
        raise ValueError(f"Malformed facial dataset path: {relative.as_posix()}")
    split = relative.parts[0]
    allowed = {name.lower(): name for name in split_names}
    if split.lower() not in allowed:
        raise ValueError(f"Unknown facial source split: {split}")
    return allowed[split.lower()]


def identify_class_folder(path: str | Path, dataset_root: str | Path, class_folder_depth: int = 1) -> str:
    root = _resolve_project_path(dataset_root)
    candidate = Path(path).resolve(strict=False)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Image path is outside facial dataset root: {path}") from exc
    if len(relative.parts) < class_folder_depth + 2:
        raise ValueError(f"Malformed facial dataset path: {relative.as_posix()}")
    return relative.parts[class_folder_depth]


def parse_face_source_path(path: str | Path, source_structure: FaceSourceStructureConfig) -> FaceSourceRecord:
    dataset_root = source_structure.dataset_root
    split = identify_predefined_split(path, dataset_root, source_structure.predefined_split_folders)
    label = identify_class_folder(path, dataset_root, source_structure.class_folder_depth)
    return FaceSourceRecord(
        source_file=_relative_to_repo(Path(path)),
        source_split=split,
        original_label=label,
        subject_id=None,
        original_id=Path(path).stem,
    )


def validate_face_source_structure(source_structure: FaceSourceStructureConfig, label_mapping: FaceLabelMappingConfig | None = None) -> dict[str, object]:
    root = _resolve_project_path(source_structure.dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Facial dataset root does not exist: {source_structure.dataset_root}")
    if not root.is_dir():
        raise ValueError(f"Facial dataset root must be a directory: {source_structure.dataset_root}")
    missing_splits = [split for split in source_structure.predefined_split_folders if not (root / split).is_dir()]
    if missing_splits:
        raise ValueError(f"Missing facial predefined split folders: {missing_splits}")
    labels = sorted({entry.original_label for entry in (label_mapping or default_face_label_mapping_config()).entries})
    missing_labels = {
        split: [label for label in labels if not (root / split / label).is_dir()]
        for split in source_structure.predefined_split_folders
    }
    missing_labels = {split: values for split, values in missing_labels.items() if values}
    if missing_labels:
        raise ValueError(f"Missing facial class folders: {missing_labels}")
    extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in source_structure.supported_image_extensions}
    unsupported_declared = sorted(extensions - SUPPORTED_IMAGE_EXTENSIONS)
    if unsupported_declared:
        raise ValueError(f"Unsupported declared facial image extensions: {unsupported_declared}")
    return {"dataset_root": source_structure.dataset_root, "splits": list(source_structure.predefined_split_folders), "labels": labels}

