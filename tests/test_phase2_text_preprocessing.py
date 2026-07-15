from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest

from app.ml.common import paths
from app.ml.common.fingerprinting import fingerprint_dataset
from app.ml.common.schemas import DatasetConfig, PreprocessingConfig
from app.ml.preprocessing.text.constants import (
    TEXT_FEATURE_SCHEMA_VERSION,
    TEXT_LABEL_MAPPING_VERSION,
    TEXT_PREPROCESSING_VERSION,
    TEXT_PRIVACY_RULESET_VERSION,
)
from app.ml.preprocessing.text.duplicates import bounded_near_duplicate_candidates, exact_duplicate_groups
from app.ml.preprocessing.text.features import TextFeatureExtractionConfig, build_text_feature_schema
from app.ml.preprocessing.text.mapping import default_text_label_mapping_config, normalize_label
from app.ml.preprocessing.text.normalization import normalize_text
from app.ml.preprocessing.text.preprocessor import (
    canonicalize_text_dataframe,
    generate_text_record_id,
    preprocess_text_dataset,
    synthetic_fingerprint,
    synthetic_text_fixture,
)
from app.ml.preprocessing.text.privacy import replace_privacy_identifiers
from app.ml.preprocessing.text.schemas import TextSourceSelectionConfig
from app.ml.preprocessing.text.validation import (
    detect_engineered_feature_leakage,
    detect_privacy_pattern_leakage,
    validate_predefined_test_overlap,
    validate_source_selection,
    validate_text_source_columns,
)


def source_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Unique_ID": 1, "text": "I am not okay.\nPlease do not remove punctuation!!!", "status": "Depression"},
            {"Unique_ID": 2, "text": "Contact https://example.com or me@example.com @helper 192.168.1.1", "status": "Anxiety"},
            {"Unique_ID": 3, "text": "Normal day with friends :)", "status": "Normal"},
            {"Unique_ID": 4, "text": "Normal day with friends :)", "status": "Suicidal"},
        ]
    )


def write_source(path: Path, df: pd.DataFrame | None = None) -> Path:
    (df if df is not None else source_df()).to_csv(path, index=False)
    return path


def dataset_config(source_path: Path) -> DatasetConfig:
    return DatasetConfig(
        dataset_name="mental-health-text",
        dataset_version="v1",
        modality="text",
        source_path=source_path,
        file_format="csv",
        label_columns=["status"],
        feature_columns=["text"],
        identifier_columns=["Unique_ID"],
        sensitive_columns=[],
        excluded_columns=["Unique_ID"],
        expected_columns=["Unique_ID", "text", "status"],
        missing_value_policy="preserve",
        duplicate_policy="report_only",
        is_raw_source=False,
        validation_context="test",
    )


def preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        preprocessing_name="mental-health-text-canonical-preprocessing",
        preprocessing_version=TEXT_PREPROCESSING_VERSION,
        dataset_name="mental-health-text",
        dataset_version="v1",
        modality="text",
        random_seed=42,
        test_size=0.2,
        validation_size=0.0,
        stratify_column="status",
        group_column=None,
        normalization_method="none",
        categorical_encoding="none",
        text_cleaning_options={"privacy_replacement": True},
        audio_options={},
        image_options={},
        output_format="csv",
        feature_schema_version=TEXT_FEATURE_SCHEMA_VERSION,
        output_subdirectory="text/v1",
    )


def source_selection_payload(authoritative: str = "source.csv") -> dict:
    return {
        "selection_version": "1.0.0",
        "dataset_name": "mental-health-text",
        "dataset_version": "v1",
        "authoritative_source_file": authoritative,
        "duplicate_check_policy": "report cross-file overlap; do not combine files",
        "conflict_policy": "quarantine",
        "sources": [
            {
                "filename": authoritative,
                "role": "authoritative_raw",
                "include_in_canonical": True,
                "reason": "raw source",
                "text_column": "text",
                "label_column": "status",
            },
            {
                "filename": "derived.csv",
                "role": "excluded_derived",
                "include_in_canonical": False,
                "reason": "feature engineered",
                "text_column": "text",
                "label_column": "status",
            },
        ],
    }


def source_selection(authoritative: str = "source.csv") -> TextSourceSelectionConfig:
    return TextSourceSelectionConfig.parse_obj(source_selection_payload(authoritative))


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def test_source_selection_columns_derived_excluded_overlap_and_conflicting_policy():
    config = source_selection()
    assert validate_source_selection(config)["authoritative_source_file"] == "source.csv"
    bad = source_selection_payload()
    bad["sources"][1]["include_in_canonical"] = True
    with pytest.raises(ValueError, match="exactly one"):
        TextSourceSelectionConfig.parse_obj(bad)
    with pytest.raises(ValueError, match="Missing required"):
        validate_text_source_columns(["text"], "text", "status")
    assert detect_engineered_feature_leakage(["text", "has_suicidal_keyword", "polarity"]) == ["polarity", "has_suicidal_keyword"]

    raw = pd.DataFrame({"comparison_text": ["a", "b"]})
    test = pd.DataFrame({"comparison_text": ["b", "c"]})
    assert validate_predefined_test_overlap(raw, test)["exact_overlap_count"] == 1


def test_privacy_replacement_unicode_false_positive_resistance_and_no_report_leakage():
    text = "Visit https://x.test, email me@example.com, call +94 77 123 4567, @helper, r/support, ip 192.168.0.1. Date 2025-03-01 Ω"
    safe, summary = replace_privacy_identifiers(text)
    assert "<URL>" in safe and "<EMAIL>" in safe and "<PHONE>" in safe and "<USER>" in safe and "<IP>" in safe and "<COMMUNITY>" in safe
    assert "me@example.com" not in safe and "192.168.0.1" not in safe
    assert "2025-03-01" in safe
    assert summary.url_count == 1
    assert summary.email_count == 1
    assert summary.phone_count == 1
    assert summary.username_count == 1
    assert summary.ip_address_count == 1
    assert summary.community_count == 1
    assert detect_privacy_pattern_leakage([safe]) == {"url_count": 0, "email_count": 0, "phone_count": 0, "username_count": 0, "ip_address_count": 0}


def test_normalization_preserves_negation_punctuation_emoji_unicode_and_is_deterministic():
    first = normalize_text("I can\u2019t do this!!!\r\nBut I will not remove emoji 😊 &amp; spaces")
    second = normalize_text("I can\u2019t do this!!!\r\nBut I will not remove emoji 😊 &amp; spaces")
    assert first == second
    assert "can't" in first.display_text
    assert "not" in first.display_text
    assert "!!!" in first.display_text
    assert "😊" in first.display_text
    assert "\r" not in first.display_text
    assert "  " not in first.display_text


def test_labels_no_silent_merging_unknown_and_excluded_from_features():
    mapping = default_text_label_mapping_config()
    assert normalize_label("Suicidal", mapping) == "suicidal"
    assert all(not entry.merged for entry in mapping.entries)
    with pytest.raises(ValueError, match="Unknown"):
        normalize_label("Stress", mapping)
    schema = build_text_feature_schema()
    assert "canonical_label" in schema.target_columns
    assert "canonical_label" not in schema.feature_names()
    with pytest.raises(ValueError, match="fitting"):
        TextFeatureExtractionConfig(extractor_name="tfidf", fit_allowed=True)


def test_duplicate_handling_record_ids_quarantine_and_no_raw_text_in_manifest():
    df = source_df()
    canonical = canonicalize_text_dataframe(df, default_text_label_mapping_config(), source_fingerprint=synthetic_fingerprint(df), source_name="source.csv")
    groups = exact_duplicate_groups(canonical)
    assert len(groups) == 1
    assert groups[0].conflict is True
    assert groups[0].duplicate_type.value == "exact"
    assert "Normal day with friends" not in json.dumps(groups[0].to_safe_dict())
    near = bounded_near_duplicate_candidates(canonical, max_records=4, threshold=0.5)
    assert isinstance(near, list)

    fp = synthetic_fingerprint(df)
    rid1 = generate_text_record_id(dataset_version="v1", source_file_identity="source.csv", source_row_index=0, source_fingerprint=fp)
    rid2 = generate_text_record_id(dataset_version="v1", source_file_identity="source.csv", source_row_index=0, source_fingerprint=fp)
    rid3 = generate_text_record_id(dataset_version="v1", source_file_identity="source.csv", source_row_index=0, source_fingerprint="a" * 64)
    assert rid1 == rid2
    assert rid1 != rid3


def test_preprocessing_outputs_safe_schema_overwrite_and_no_splits_or_fitting(tmp_path):
    source = write_source(tmp_path / "source.csv")
    config = dataset_config(source)
    fingerprint = fingerprint_dataset(config)
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-text-{uuid.uuid4().hex}"
    try:
        result = preprocess_text_dataset(
            config,
            preprocessing_config(),
            default_text_label_mapping_config(),
            source_selection("source.csv"),
            fingerprint,
            output_dir=output_dir,
        )
        assert result["source_rows"] == 4
        assert result["output_rows"] == 2
        assert result["excluded_rows"] == 2
        assert result["conflicting_duplicate_groups"] == 1
        canonical = pd.read_csv(output_dir / "canonical_text.csv")
        assert "normalized_text" in canonical.columns
        assert "source_row_index" not in canonical.columns
        assert "comparison_text" not in canonical.columns
        assert not canonical["normalized_text"].astype(str).str.contains("me@example.com|https://x.test|192.168", regex=True).any()
        assert (output_dir / "text_feature_schema.json").exists()
        assert not any((output_dir / name).exists() for name in ["train.csv", "validation.csv", "test.csv", "tfidf.npz"])
        report_text = (output_dir / "text_preprocessing_report.json").read_text(encoding="utf-8")
        assert "me@example.com" not in report_text
        assert "SafeTalk" in report_text
        with pytest.raises(FileExistsError):
            preprocess_text_dataset(config, preprocessing_config(), default_text_label_mapping_config(), source_selection("source.csv"), fingerprint, output_dir=output_dir)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_cli_validate_success_preprocess_fingerprint_unknown_label_and_no_raw_console(tmp_path):
    source = write_source(tmp_path / "source.csv")
    config = dataset_config(source)
    fingerprint = fingerprint_dataset(config)
    dataset_payload = config.to_safe_dict()
    dataset_payload["validation_context"] = "test"
    dataset_path = write_json(tmp_path / "dataset.json", dataset_payload)
    preprocessing_path = write_json(tmp_path / "preprocessing.json", preprocessing_config().to_safe_dict())
    mapping_path = write_json(tmp_path / "mapping.json", default_text_label_mapping_config().to_safe_dict())
    source_selection_path = write_json(tmp_path / "source_selection.json", source_selection("source.csv").to_safe_dict())
    fingerprint_path = write_json(tmp_path / "fingerprint.json", fingerprint.to_safe_dict())
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-text-cli-{uuid.uuid4().hex}"
    script = paths.get_backend_root() / "scripts" / "preprocess_text_dataset.py"
    try:
        validate = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--label-mapping-config",
                str(mapping_path),
                "--source-selection-config",
                str(source_selection_path),
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
        assert "validation: passed" in validate.stdout
        assert "me@example.com" not in validate.stdout
        assert not output_dir.exists()

        preprocess = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--label-mapping-config",
                str(mapping_path),
                "--source-selection-config",
                str(source_selection_path),
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
        assert "conflicting duplicate groups: 1" in preprocess.stdout

        source.write_text(source.read_text(encoding="utf-8").replace("Depression", "Stress", 1), encoding="utf-8")
        unknown = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--label-mapping-config",
                str(mapping_path),
                "--source-selection-config",
                str(source_selection_path),
                "--fingerprint",
                str(fingerprint_path),
                "--output-dir",
                str(paths.get_generated_root() / "temporary" / f"pytest-text-cli-{uuid.uuid4().hex}"),
                "--validate-only",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert unknown.returncode != 0
        assert "fingerprint mismatch" in unknown.stderr.lower()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_versions_and_no_postgresql_or_safetalk_access():
    assert TEXT_PREPROCESSING_VERSION == "1.0.0"
    assert TEXT_FEATURE_SCHEMA_VERSION == "1.0.0"
    assert TEXT_LABEL_MAPPING_VERSION == "1.0.0"
    assert TEXT_PRIVACY_RULESET_VERSION == "1.0.0"
    text_root = paths.get_backend_root() / "app" / "ml" / "preprocessing" / "text"
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_root.glob("*.py"))
    assert "SessionLocal" not in combined
    assert "get_db" not in combined
    assert "psycopg" not in combined.lower()
    assert "fit_transform" not in combined
    assert "train_test_split" not in combined
