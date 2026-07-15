from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from app.ml.common import paths
from app.ml.common.fingerprinting import fingerprint_dataset
from app.ml.common.schemas import DatasetConfig, PreprocessingConfig
from app.ml.preprocessing.profile.constants import (
    PROFILE_FEATURE_SCHEMA_VERSION,
    PROFILE_MAPPING_VERSION,
    PROFILE_PREPROCESSING_VERSION,
    SOURCE_COLUMNS,
    TARGET_CANONICAL_COLUMN,
    TARGET_COLUMN,
    TREATMENT_COLUMN,
)
from app.ml.preprocessing.profile.mapping import default_profile_mapping_config
from app.ml.preprocessing.profile.preprocessor import (
    build_profile_feature_table,
    canonicalize_profile_dataframe,
    generate_record_id,
    normalize_binary_label,
    parse_age,
    parse_cgpa,
    preprocess_profile_dataset,
)
from app.ml.preprocessing.profile.validation import (
    detect_profile_leakage,
    validate_profile_categories,
    validate_profile_mapping,
    validate_profile_missing_values,
    validate_profile_numeric_ranges,
    validate_profile_source_columns,
    validate_profile_target_values,
)


def fixture_rows() -> list[dict[str, object]]:
    return [
        {
            "Timestamp": "8/7/2020 12:00",
            "Choose your gender": " Female ",
            "Age": 20,
            "What is your course?": "Secret Course",
            "Your current year of Study": " Year 1 ",
            "What is your CGPA?": "3.50 - 4.00 ",
            "Marital status": "No",
            "Do you have Depression?": "Yes",
            "Do you have Anxiety?": "No",
            "Do you have Panic attack?": "Yes",
            "Did you seek any specialist for a treatment?": "No",
        },
        {
            "Timestamp": "8/7/2020 13:00",
            "Choose your gender": "Male",
            "Age": None,
            "What is your course?": "Another Course",
            "Your current year of Study": "year 2",
            "What is your CGPA?": "3.00 - 3.49",
            "Marital status": "Yes",
            "Do you have Depression?": "No",
            "Do you have Anxiety?": " yes ",
            "Do you have Panic attack?": " no ",
            "Did you seek any specialist for a treatment?": "Yes",
        },
    ]


def write_source(path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    pd.DataFrame(rows or fixture_rows(), columns=list(SOURCE_COLUMNS)).to_csv(path, index=False)
    return path


def dataset_config(source_path: Path) -> DatasetConfig:
    return DatasetConfig(
        dataset_name="student-profile",
        dataset_version="v1",
        modality="profile",
        source_path=source_path,
        file_format="csv",
        label_columns=[TARGET_COLUMN],
        feature_columns=[
            "Choose your gender",
            "Age",
            "What is your course?",
            "Your current year of Study",
            "What is your CGPA?",
            "Marital status",
            "Do you have Anxiety?",
            "Do you have Panic attack?",
        ],
        identifier_columns=[],
        sensitive_columns=[
            "Choose your gender",
            "Age",
            "What is your course?",
            "What is your CGPA?",
            "Marital status",
        ],
        excluded_columns=["Timestamp", TREATMENT_COLUMN],
        expected_columns=list(SOURCE_COLUMNS),
        missing_value_policy="preserve",
        duplicate_policy="report_only",
        is_raw_source=True,
        validation_context="test",
    )


def preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        preprocessing_name="profile-canonical-preprocessing",
        preprocessing_version=PROFILE_PREPROCESSING_VERSION,
        dataset_name="student-profile",
        dataset_version="v1",
        modality="profile",
        random_seed=42,
        test_size=0.2,
        validation_size=0.0,
        stratify_column=TARGET_CANONICAL_COLUMN,
        group_column=None,
        normalization_method="none",
        categorical_encoding="none",
        text_cleaning_options={},
        audio_options={},
        image_options={},
        output_format="csv",
        feature_schema_version=PROFILE_FEATURE_SCHEMA_VERSION,
        output_subdirectory="profile/v1",
    )


def run_preprocess(tmp_path: Path, *, overwrite: bool = True, include_sensitive_context: bool = False):
    source = write_source(tmp_path / "source.csv")
    config = dataset_config(source)
    fingerprint = fingerprint_dataset(config)
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-profile-{uuid.uuid4().hex}"
    result = preprocess_profile_dataset(
        config,
        preprocessing_config(),
        default_profile_mapping_config(),
        fingerprint,
        output_dir=output_dir,
        overwrite=overwrite,
        include_sensitive_context=include_sensitive_context,
    )
    return result, output_dir


def test_source_validation_valid_missing_required_and_target_values(tmp_path):
    source = write_source(tmp_path / "source.csv")
    df = pd.read_csv(source)

    assert validate_profile_source_columns(df.columns)["valid"]
    with pytest.raises(ValueError, match="Missing required"):
        validate_profile_source_columns([column for column in df.columns if column != "Age"])
    with pytest.raises(ValueError, match="unrecognized"):
        validate_profile_target_values(df.assign(**{TARGET_COLUMN: ["Maybe", "No"]}))


def test_missing_age_unexpected_category_invalid_age_and_malformed_cgpa(tmp_path):
    df = pd.read_csv(write_source(tmp_path / "source.csv"))

    missing = validate_profile_missing_values(df)
    assert missing["age_missing_count"] == 1

    categories = validate_profile_categories(df.assign(**{"Your current year of Study": ["Year 5", "year 1"]}))
    assert categories["unexpected_categories"]["Your current year of Study"] == ["year 5"]

    with pytest.raises(ValueError, match="numeric ranges"):
        validate_profile_numeric_ranges(df.assign(Age=[20, 999]))
    with pytest.raises(ValueError, match="CGPA"):
        parse_cgpa("3.75")


def test_timestamp_target_and_treatment_leakage_detection():
    leakage = detect_profile_leakage(["year_of_study", "source_timestamp", TARGET_CANONICAL_COLUMN, "sought_specialist_treatment"])

    assert leakage["has_leakage"]
    reasons = " ".join(item["reason"] for item in leakage["leakage_columns"])
    assert "timestamp" in reasons
    assert "target" in reasons
    assert "post-outcome" in reasons


def test_mapping_versions_and_required_exclusions():
    mapping = default_profile_mapping_config()
    result = validate_profile_mapping(mapping)

    assert result["mapping_version"] == PROFILE_MAPPING_VERSION
    assert "Timestamp" in result["excluded_columns"]
    assert TREATMENT_COLUMN in result["excluded_columns"]


def test_canonicalization_normalization_ids_and_missing_preservation(tmp_path):
    source = write_source(tmp_path / "source.csv")
    df = pd.read_csv(source)
    fingerprint = fingerprint_dataset(dataset_config(source))
    canonical = canonicalize_profile_dataframe(df, default_profile_mapping_config(), source_fingerprint=fingerprint.combined_sha256)

    assert canonical.loc[0, "gender"] == "female"
    assert canonical.loc[0, "year_of_study"] == "year 1"
    assert canonical.loc[0, "cgpa_band"] == "3.50 - 4.00"
    assert pd.isna(canonical.loc[1, "age"])
    assert canonical.loc[0, TARGET_CANONICAL_COLUMN] == "yes"
    assert canonical.loc[1, "self_reported_anxiety"] == "yes"
    assert canonical.loc[0, "record_id"] == generate_record_id(0, fingerprint.combined_sha256)
    assert generate_record_id(0, fingerprint.combined_sha256) != generate_record_id(0, "f" * 64)


def test_binary_and_numeric_normalization_fail_clearly():
    assert normalize_binary_label(" YES ") == "yes"
    assert parse_age("21") == 21
    assert parse_age(None) is None
    with pytest.raises(ValueError, match="unrecognized"):
        normalize_binary_label("sometimes")
    with pytest.raises(ValueError, match="whole number"):
        parse_age("20.5")


def test_feature_table_excludes_target_metadata_treatment_and_sensitive_by_default(tmp_path):
    source = write_source(tmp_path / "source.csv")
    df = pd.read_csv(source)
    fingerprint = fingerprint_dataset(dataset_config(source))
    canonical = canonicalize_profile_dataframe(df, default_profile_mapping_config(), source_fingerprint=fingerprint.combined_sha256)
    feature_df, features, excluded, sensitive = build_profile_feature_table(canonical, default_profile_mapping_config())

    assert features == ["year_of_study", "self_reported_anxiety", "self_reported_panic_attack"]
    assert TARGET_CANONICAL_COLUMN not in features
    assert "source_timestamp" not in features
    assert "sought_specialist_treatment" not in features
    assert "gender" in excluded
    assert sensitive == []
    assert list(feature_df.columns) == ["record_id", *features, TARGET_CANONICAL_COLUMN]


def test_optional_sensitive_context_behavior(tmp_path):
    result, output_dir = run_preprocess(tmp_path, include_sensitive_context=True)
    try:
        assert {"gender", "age", "course", "cgpa_band", "marital_status"} <= set(result["feature_columns"])
        assert set(result["sensitive_context_columns"]) == {"gender", "age", "course", "cgpa_band", "marital_status"}
        canonical = pd.read_csv(output_dir / "canonical_profile.csv")
        assert "age" in canonical.columns
        assert canonical["age"].isna().sum() == 1
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_feature_schema_contract_and_no_duplicate_features(tmp_path):
    result, output_dir = run_preprocess(tmp_path)
    try:
        schema = result["feature_schema"]
        names = schema.feature_names()
        assert names == ["year_of_study", "self_reported_anxiety", "self_reported_panic_attack"]
        assert TARGET_CANONICAL_COLUMN not in names
        assert "source_timestamp" not in names
        assert len(names) == len(set(names))
        assert schema.required_source_columns() == [
            "Your current year of Study",
            "Do you have Anxiety?",
            "Do you have Panic attack?",
        ]
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_outputs_written_outside_raw_root_overwrite_and_privacy(tmp_path):
    result, output_dir = run_preprocess(tmp_path)
    try:
        assert paths.is_path_inside(paths.get_generated_root(), output_dir)
        assert not paths.is_path_inside(paths.get_raw_dataset_root(), output_dir)
        assert (output_dir / "canonical_profile.csv").exists()
        assert (output_dir / "profile_preprocessing_report.json").exists()
        assert (output_dir / "profile_preprocessing_report.md").exists()
        assert (output_dir / "profile_record_manifest.json").exists()
        assert "Secret Course" not in (output_dir / "profile_preprocessing_report.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in (output_dir / "profile_preprocessing_report.md").read_text(encoding="utf-8")

        with pytest.raises(FileExistsError):
            preprocess_profile_dataset(
                dataset_config(write_source(tmp_path / "source.csv")),
                preprocessing_config(),
                default_profile_mapping_config(),
                fingerprint_dataset(dataset_config(tmp_path / "source.csv")),
                output_dir=output_dir,
                overwrite=False,
            )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_report_versions_counts_warnings_and_no_splits_or_artifacts(tmp_path):
    result, output_dir = run_preprocess(tmp_path)
    try:
        report = result["report"]
        assert report.preprocessing_version == PROFILE_PREPROCESSING_VERSION
        assert report.feature_schema_version == PROFILE_FEATURE_SCHEMA_VERSION
        assert report.mapping_version == PROFILE_MAPPING_VERSION
        assert report.source_row_count == 2
        assert report.output_row_count == 2
        assert report.excluded_row_count == 0
        assert report.target_distribution == {"no": 1, "yes": 1}
        assert report.generated_at.tzinfo is not None
        assert any("Treatment-seeking is excluded" in warning for warning in report.warnings)
        assert not any(path.name.startswith(("train", "validation", "test")) for path in output_dir.iterdir())
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def cli_config_payload(source_path: Path) -> dict:
    payload = dataset_config(source_path).to_safe_dict()
    payload["validation_context"] = "test"
    return payload


def test_cli_validate_only_success_preprocess_success_and_overwrite_failure(tmp_path):
    source = write_source(tmp_path / "source.csv")
    config = dataset_config(source)
    fingerprint = fingerprint_dataset(config)
    dataset_path = write_json(tmp_path / "dataset.json", cli_config_payload(source))
    preprocessing_path = write_json(tmp_path / "preprocessing.json", preprocessing_config().to_safe_dict())
    mapping_path = write_json(tmp_path / "mapping.json", default_profile_mapping_config().to_safe_dict())
    fingerprint_path = write_json(tmp_path / "fingerprint.json", fingerprint.to_safe_dict())
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-profile-cli-{uuid.uuid4().hex}"
    script = paths.get_backend_root() / "scripts" / "preprocess_profile_dataset.py"
    try:
        validate = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--mapping-config",
                str(mapping_path),
                "--fingerprint",
                str(fingerprint_path),
                "--output-dir",
                str(output_dir),
                "--validate-only",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert validate.returncode == 0, validate.stderr
        assert not output_dir.exists()

        preprocess = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--mapping-config",
                str(mapping_path),
                "--fingerprint",
                str(fingerprint_path),
                "--output-dir",
                str(output_dir),
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert preprocess.returncode == 0, preprocess.stderr
        assert (output_dir / "canonical_profile.csv").exists()
        assert "sought_specialist_treatment" not in pd.read_csv(output_dir / "canonical_profile.csv").columns

        overwrite = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--mapping-config",
                str(mapping_path),
                "--fingerprint",
                str(fingerprint_path),
                "--output-dir",
                str(output_dir),
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert overwrite.returncode != 0
        assert "overwrite" in overwrite.stderr.lower()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_cli_missing_fingerprint_mismatch_unknown_target_and_sensitive_flag(tmp_path):
    source = write_source(tmp_path / "source.csv")
    config = dataset_config(source)
    fingerprint = fingerprint_dataset(config)
    dataset_path = write_json(tmp_path / "dataset.json", cli_config_payload(source))
    preprocessing_path = write_json(tmp_path / "preprocessing.json", preprocessing_config().to_safe_dict())
    mapping_path = write_json(tmp_path / "mapping.json", default_profile_mapping_config().to_safe_dict())
    fingerprint_path = write_json(tmp_path / "fingerprint.json", fingerprint.to_safe_dict())
    script = paths.get_backend_root() / "scripts" / "preprocess_profile_dataset.py"

    missing_fp = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-config",
            str(dataset_path),
            "--preprocessing-config",
            str(preprocessing_path),
            "--mapping-config",
            str(mapping_path),
            "--fingerprint",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(paths.get_generated_root() / "temporary" / f"pytest-profile-cli-{uuid.uuid4().hex}"),
            "--validate-only",
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_fp.returncode != 0

    source.write_text(source.read_text(encoding="utf-8").replace("Secret Course", "Changed Course"), encoding="utf-8")
    mismatch = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-config",
            str(dataset_path),
            "--preprocessing-config",
            str(preprocessing_path),
            "--mapping-config",
            str(mapping_path),
            "--fingerprint",
            str(fingerprint_path),
            "--output-dir",
            str(paths.get_generated_root() / "temporary" / f"pytest-profile-cli-{uuid.uuid4().hex}"),
            "--validate-only",
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert mismatch.returncode != 0
    assert "fingerprint mismatch" in mismatch.stderr.lower()

    bad_source = write_source(tmp_path / "bad.csv", [{**fixture_rows()[0], TARGET_COLUMN: "Unknown"}])
    bad_config_path = write_json(tmp_path / "bad_dataset.json", cli_config_payload(bad_source))
    bad_fp = write_json(tmp_path / "bad_fingerprint.json", fingerprint_dataset(dataset_config(bad_source)).to_safe_dict())
    unknown = subprocess.run(
        [
            sys.executable,
            str(script),
            "--dataset-config",
            str(bad_config_path),
            "--preprocessing-config",
            str(preprocessing_path),
            "--mapping-config",
            str(mapping_path),
            "--fingerprint",
            str(bad_fp),
            "--output-dir",
            str(paths.get_generated_root() / "temporary" / f"pytest-profile-cli-{uuid.uuid4().hex}"),
            "--validate-only",
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert unknown.returncode != 0
    assert "unrecognized" in unknown.stderr.lower()

    restored = write_source(tmp_path / "source.csv")
    sensitive_fp = write_json(tmp_path / "sensitive_fingerprint.json", fingerprint_dataset(dataset_config(restored)).to_safe_dict())
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-profile-cli-{uuid.uuid4().hex}"
    try:
        sensitive = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--mapping-config",
                str(mapping_path),
                "--fingerprint",
                str(sensitive_fp),
                "--output-dir",
                str(output_dir),
                "--include-sensitive-context",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert sensitive.returncode == 0, sensitive.stderr
        assert "gender" in pd.read_csv(output_dir / "canonical_profile.csv").columns
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
