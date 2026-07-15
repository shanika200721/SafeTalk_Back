from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image

from app.ml.common import paths
from app.ml.common.fingerprinting import fingerprint_dataset
from app.ml.common.schemas import DatasetConfig
from app.ml.preprocessing.face.duplicates import detect_exact_duplicate_groups, find_near_duplicate_candidates
from app.ml.preprocessing.face.features import build_face_feature_schema, extract_lightweight_image_statistics
from app.ml.preprocessing.face.image_io import convert_image_deterministic, extract_image_metadata, image_sha256
from app.ml.preprocessing.face.mapping import normalize_face_label, parse_face_source_path
from app.ml.preprocessing.face.preprocessor import generate_face_record_id, generate_safe_subject_key, preprocess_face_dataset
from app.ml.preprocessing.face.schemas import FaceLabelMappingConfig, FaceLabelMappingEntry, FaceSourceStructureConfig


def write_image(path: Path, *, mode: str = "RGB", size=(12, 10), color=(20, 100, 180)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fill = 120 if mode == "L" else color
    Image.new(mode, size, fill).save(path)
    return path


def label_mapping(labels=("angry", "happy", "sad")) -> FaceLabelMappingConfig:
    return FaceLabelMappingConfig(
        canonical_labels=list(labels),
        entries=[
            FaceLabelMappingEntry(original_label=label, canonical_label=label, notes="fixture")
            for label in labels
        ],
    )


def source_structure(root: Path, labels=("angry", "happy", "sad")) -> FaceSourceStructureConfig:
    for split in ("train", "test"):
        for label in labels:
            (root / split / label).mkdir(parents=True, exist_ok=True)
    return FaceSourceStructureConfig(
        dataset_root=str(root),
        predefined_split_folders=["train", "test"],
        class_folder_depth=1,
        supported_image_extensions=[".png", ".jpg", ".jpeg"],
        subject_id_available=False,
        filename_parsing_rule="fixture <split>/<label>/<file>",
        duplicate_policy="report only",
        corruption_policy="report unreadable",
        inclusion_exclusion_notes="fixture",
        license_note="fixture",
    )


def dataset_config(root: Path) -> DatasetConfig:
    return DatasetConfig(
        dataset_name="face-fixture",
        dataset_version="v1",
        modality="face",
        source_path=root,
        file_format="folder",
        label_columns=[],
        feature_columns=[],
        identifier_columns=[],
        sensitive_columns=[],
        excluded_columns=[],
        expected_columns=[],
        missing_value_policy="preserve",
        duplicate_policy="report_only",
        is_raw_source=True,
        validation_context="test",
    )


def dataset_config_payload(root: Path) -> dict:
    payload = dataset_config(root).to_safe_dict()
    payload["validation_context"] = "test"
    return payload


def test_source_parsing_valid_unknown_split_unknown_label_and_malformed(tmp_path):
    root = tmp_path / "face"
    structure = source_structure(root, labels=("angry", "happy"))
    train = write_image(root / "train" / "angry" / "a.png")
    test = write_image(root / "test" / "happy" / "b.png")
    assert parse_face_source_path(train, structure).source_split == "train"
    assert parse_face_source_path(test, structure).original_label == "happy"
    with pytest.raises(ValueError, match="Unknown facial source split"):
        parse_face_source_path(write_image(root / "dev" / "angry" / "x.png"), structure)
    unknown = write_image(root / "train" / "surprise" / "x.png")
    record = parse_face_source_path(unknown, structure)
    with pytest.raises(ValueError, match="Unknown face emotion label"):
        normalize_face_label(record.original_label, label_mapping(labels=("angry", "happy")))
    with pytest.raises(ValueError, match="Malformed"):
        parse_face_source_path(root / "train" / "orphan.png", structure)


def test_metadata_rgb_grayscale_dimensions_corrupt_zero_and_hash(tmp_path):
    rgb = write_image(tmp_path / "rgb.png", mode="RGB", size=(13, 11))
    gray = write_image(tmp_path / "gray.jpg", mode="L", size=(8, 9), color=(0, 0, 0))
    zero = tmp_path / "zero.png"
    zero.write_bytes(b"")
    corrupt = tmp_path / "bad.png"
    corrupt.write_bytes(b"not image data")
    first = extract_image_metadata(rgb)
    second = extract_image_metadata(rgb)
    assert first == second
    assert first.width == 13 and first.height == 11
    assert first.color_mode == "RGB"
    assert extract_image_metadata(gray).color_mode == "L"
    assert not extract_image_metadata(zero).readable
    assert not extract_image_metadata(corrupt).readable
    assert image_sha256(rgb) == first.image_hash


def test_labels_ids_safe_subject_key_and_feature_contract():
    mapping = label_mapping(labels=("angry", "sad"))
    assert normalize_face_label("angry", mapping) == "angry"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_face_label("fear", mapping)
    with pytest.raises(ValueError, match="must not silently merge"):
        FaceLabelMappingConfig(
            canonical_labels=["angry"],
            entries=[FaceLabelMappingEntry(original_label="mad", canonical_label="angry", merged=True, notes="bad")],
        )
    rid1 = generate_face_record_id("Final Dataset/Facial Emotion/train/angry/a.png", "a" * 64)
    rid2 = generate_face_record_id("Final Dataset/Facial Emotion/train/angry/a.png", "a" * 64)
    rid3 = generate_face_record_id("Final Dataset/Facial Emotion/train/angry/a.png", "b" * 64)
    assert rid1 == rid2
    assert rid1 != rid3
    key = generate_safe_subject_key("student-001", "a" * 64)
    assert "student-001" not in key
    schema = build_face_feature_schema()
    assert "canonical_emotion_label" in schema.target_columns
    assert "canonical_emotion_label" not in schema.feature_names()


def test_duplicates_cross_split_cross_label_and_near_candidates(tmp_path):
    root = tmp_path / "face"
    structure = source_structure(root)
    a = write_image(root / "train" / "angry" / "a.png", color=(255, 0, 0))
    b = root / "train" / "angry" / "b.png"
    b.write_bytes(a.read_bytes())
    c = root / "test" / "sad" / "c.png"
    c.write_bytes(a.read_bytes())
    result = preprocess_face_dataset(
        structure,
        label_mapping(),
        source_fingerprint="a" * 64,
        output_dir=paths.get_generated_root() / "temporary" / f"pytest-face-dups-{uuid.uuid4().hex}",
        overwrite=True,
        near_duplicate_limit=10,
    )
    assert result["duplicate_summary"]["duplicate_group_count"] == 1
    assert result["duplicate_summary"]["cross_split_duplicate_count"] == 1
    assert result["duplicate_summary"]["cross_label_duplicate_count"] == 1
    groups = detect_exact_duplicate_groups(result["report"] and [])
    assert groups == []
    assert find_near_duplicate_candidates({}, limit=10) == []


def test_conversion_statistics_outputs_privacy_overwrite_and_no_model_artifacts(tmp_path):
    root = tmp_path / "face"
    structure = source_structure(root, labels=("angry", "happy"))
    source = write_image(root / "train" / "angry" / "a.png", color=(10, 80, 200))
    write_image(root / "test" / "happy" / "b.png", mode="L", size=(9, 9))
    original_hash = image_sha256(source)
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-face-{uuid.uuid4().hex}"
    try:
        result = preprocess_face_dataset(
            structure,
            label_mapping(labels=("angry", "happy")),
            source_fingerprint="a" * 64,
            output_dir=output_dir,
            overwrite=False,
            compute_image_statistics=True,
        )
        assert result["output_records"] == 2
        assert (output_dir / "face_canonical_manifest.csv").exists()
        assert (output_dir / "face_feature_schema.json").exists()
        assert (output_dir / "face_image_statistics.csv").exists()
        assert not (output_dir / "face_normalized_images_manifest.json").exists()
        assert not any((output_dir / name).exists() for name in ["train.csv", "validation.csv", "test.csv", "model.pkl", "scaler.pkl"])
        report_text = (output_dir / "face_preprocessing_report.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in report_text
        assert "data:image" not in report_text
        with pytest.raises(FileExistsError):
            preprocess_face_dataset(
                structure,
                label_mapping(labels=("angry", "happy")),
                source_fingerprint="a" * 64,
                output_dir=output_dir,
                overwrite=False,
            )
        converted = tmp_path / "converted.png"
        convert_image_deterministic(source, converted, target_width=6, target_height=6, color_mode="L")
        assert image_sha256(source) == original_hash
        stats1, warnings = extract_lightweight_image_statistics(source)
        stats2, _ = extract_lightweight_image_statistics(source)
        assert not warnings
        assert stats1 == stats2
        assert all(math.isfinite(value) for value in stats1.values())
        assert 0 <= stats1["edge_density"] <= 1
        normalized = preprocess_face_dataset(
            structure,
            label_mapping(labels=("angry", "happy")),
            source_fingerprint="a" * 64,
            output_dir=output_dir,
            overwrite=True,
            write_normalized_images=True,
            target_width=6,
            target_height=6,
        )
        assert normalized["write_normalized_images"]
        assert (output_dir / "face_normalized_images_manifest.json").exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(paths.get_generated_preprocessing_root() / "face" / "v1" / "images", ignore_errors=True)


def test_cli_validate_metadata_missing_mapping_fingerprint_corrupt_cross_label_and_overwrite(tmp_path):
    root = tmp_path / "face"
    structure = source_structure(root, labels=("angry", "happy", "sad"))
    good = write_image(root / "train" / "angry" / "a.png")
    write_image(root / "test" / "happy" / "b.png", color=(5, 180, 90))
    corrupt = root / "train" / "sad" / "bad.png"
    corrupt.write_bytes(b"bad")
    structure_path = tmp_path / "structure.json"
    structure_path.write_text(json.dumps(structure.to_safe_dict()), encoding="utf-8")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(label_mapping().to_safe_dict()), encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset_config_payload(root)), encoding="utf-8")
    prep_path = tmp_path / "prep.json"
    prep_path.write_text(json.dumps({"preprocessing_version": "1.0.0"}), encoding="utf-8")
    fingerprint_path = tmp_path / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint_dataset(dataset_config(root)).to_safe_dict()), encoding="utf-8")
    script = paths.get_backend_root() / "scripts" / "preprocess_face_dataset.py"
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-face-cli-{uuid.uuid4().hex}"
    base = [
        sys.executable,
        str(script),
        "--dataset-config",
        str(dataset_path),
        "--preprocessing-config",
        str(prep_path),
        "--label-mapping-config",
        str(labels_path),
        "--source-structure-config",
        str(structure_path),
        "--fingerprint",
        str(fingerprint_path),
        "--output-dir",
        str(output_dir),
    ]
    validate = subprocess.run(base + ["--validate-only"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert validate.returncode == 0, validate.stderr
    metadata = subprocess.run(base + ["--overwrite"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert metadata.returncode == 0, metadata.stderr
    assert (output_dir / "face_corrupt_files.json").exists()
    no_overwrite = subprocess.run(base, cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert no_overwrite.returncode != 0
    stats = subprocess.run(base + ["--overwrite", "--compute-image-statistics"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert stats.returncode == 0, stats.stderr
    assert (output_dir / "face_image_statistics.csv").exists()
    normalized = subprocess.run(base + ["--overwrite", "--write-normalized-images", "--target-width", "8", "--target-height", "8"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert normalized.returncode == 0, normalized.stderr
    assert (output_dir / "face_normalized_images_manifest.json").exists()
    missing_base = list(base)
    missing_base[missing_base.index("--label-mapping-config") + 1] = str(tmp_path / "missing.json")
    missing_mapping = subprocess.run(missing_base, cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert missing_mapping.returncode != 0
    good.write_bytes(good.read_bytes() + b"x")
    mismatch = subprocess.run(base + ["--validate-only"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert mismatch.returncode != 0
    assert "fingerprint mismatch" in mismatch.stderr.lower()
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(paths.get_generated_preprocessing_root() / "face" / "v1" / "images", ignore_errors=True)


def test_cli_cross_label_duplicate_returns_failure_after_quarantine_manifest(tmp_path):
    root = tmp_path / "face"
    structure = source_structure(root, labels=("angry", "sad"))
    angry = write_image(root / "train" / "angry" / "a.png", color=(40, 40, 40))
    sad = root / "test" / "sad" / "same.png"
    sad.write_bytes(angry.read_bytes())
    structure_path = tmp_path / "structure.json"
    structure_path.write_text(json.dumps(structure.to_safe_dict()), encoding="utf-8")
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(label_mapping(labels=("angry", "sad")).to_safe_dict()), encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset_config_payload(root)), encoding="utf-8")
    prep_path = tmp_path / "prep.json"
    prep_path.write_text(json.dumps({"preprocessing_version": "1.0.0"}), encoding="utf-8")
    fingerprint_path = tmp_path / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint_dataset(dataset_config(root)).to_safe_dict()), encoding="utf-8")
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-face-conflict-{uuid.uuid4().hex}"
    script = paths.get_backend_root() / "scripts" / "preprocess_face_dataset.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset-config",
        str(dataset_path),
        "--preprocessing-config",
        str(prep_path),
        "--label-mapping-config",
        str(labels_path),
        "--source-structure-config",
        str(structure_path),
        "--fingerprint",
        str(fingerprint_path),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    result = subprocess.run(cmd, cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert result.returncode != 0
    conflict_path = output_dir / "face_cross_label_conflicts.json"
    assert conflict_path.exists()
    conflicts = json.loads(conflict_path.read_text(encoding="utf-8"))
    assert conflicts["severity"] == "critical"
    assert len(conflicts["cross_label_duplicate_hash_groups"]) == 1
    shutil.rmtree(output_dir, ignore_errors=True)
