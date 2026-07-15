import json
import subprocess
import sys
import uuid
import wave
from pathlib import Path

import pytest
from PIL import Image

from app.ml.audit import DATASET_AUDIT_VERSION, audit_dataset
from app.ml.audit.reporting import (
    create_markdown_summary,
    load_audit_report,
    save_audit_json,
    save_audit_markdown,
)
from app.ml.common import paths
from app.ml.common.fingerprinting import fingerprint_dataset
from app.ml.common.schemas import DatasetConfig


def dataset_config(source_path, **overrides):
    payload = {
        "dataset_name": "fixture",
        "dataset_version": "v1",
        "modality": "profile",
        "source_path": str(source_path),
        "file_format": "csv",
        "label_columns": [],
        "feature_columns": [],
        "identifier_columns": [],
        "sensitive_columns": [],
        "excluded_columns": [],
        "expected_columns": [],
        "missing_value_policy": "preserve",
        "duplicate_policy": "report_only",
        "is_raw_source": True,
        "validation_context": "test",
    }
    payload.update(overrides)
    return DatasetConfig(**payload)


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_wav(path: Path, sample_rate=8000, channels=1, frames=800):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames * channels)
    return path


def write_png(path: Path, size=(8, 6), color=(20, 40, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)
    return path


def test_tabular_audit_reports_counts_missing_duplicates_and_candidates(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text(
        "student_id,email,gender,label,target_score,constant,value\n"
        "1,a@example.com,F,yes,1,x,10\n"
        "1,a@example.com,F,yes,1,x,10\n"
        "2,b@example.com,M,no,0,x,\n",
        encoding="utf-8",
    )
    config = dataset_config(
        source,
        label_columns=["label"],
        identifier_columns=["student_id"],
        sensitive_columns=["gender"],
    )
    report = audit_dataset(config, fingerprint_dataset(config))
    result = report.tabular_result

    assert report.audit_version == DATASET_AUDIT_VERSION
    assert result.row_count == 3
    assert result.column_count == 7
    assert result.duplicate_row_count == 1
    assert result.class_distribution["label"][0].count == 2
    assert "student_id" in result.possible_identifier_columns
    assert "email" in result.possible_identifier_columns
    assert "gender" in result.possible_sensitive_columns
    assert "target_score" in result.possible_leakage_columns
    assert any(column.column_name == "constant" and column.unique_count == 1 for column in result.columns)
    assert "a@example.com" not in json.dumps(report.to_safe_dict())


def test_tabular_tsv_parsing_and_range_issue(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("label\tage\nA\t22\nB\t130\n", encoding="utf-8")
    config = dataset_config(source, file_format="tsv", label_columns=["label"])
    report = audit_dataset(
        config,
        fingerprint_dataset(config),
        options={"expected_numeric_ranges": {"age": {"minimum": 0, "maximum": 120}}},
    )

    assert report.tabular_result.row_count == 2
    assert report.tabular_result.invalid_range_counts == {"age": 1}


def test_tabular_malformed_and_unsupported_formats(tmp_path):
    malformed = tmp_path / "bad.csv"
    malformed.write_bytes(b"\xff\xfe\x00")
    config = dataset_config(malformed)
    with pytest.raises(ValueError, match="decode|parse"):
        audit_dataset(config, fingerprint_dataset(config))

    txt = tmp_path / "source.txt"
    txt.write_text("x", encoding="utf-8")
    unsupported = dataset_config(txt, file_format="txt")
    with pytest.raises(ValueError, match="Unsupported"):
        audit_dataset(unsupported, fingerprint_dataset(unsupported))


def test_text_audit_detects_missing_duplicates_conflicts_and_privacy_patterns(tmp_path):
    source = tmp_path / "text.csv"
    source.write_text(
        "text,status\n"
        "\"Please contact me at a@example.com or +1 555 222 3333\",risk\n"
        "\"please contact me at a@example.com or +1 555 222 3333\",safe\n"
        "\"Visit https://example.com @helper\",risk\n"
        "\"\",safe\n",
        encoding="utf-8",
    )
    config = dataset_config(source, modality="text", feature_columns=["text"], label_columns=["status"])
    report = audit_dataset(config, fingerprint_dataset(config), options={"text_column": "text"})
    result = report.text_result

    assert result.record_count == 4
    assert result.missing_text_count == 1
    assert result.exact_duplicate_text_count == 1
    assert result.duplicate_text_conflicting_labels_count == 1
    assert result.email_occurrence_count == 2
    assert result.phone_occurrence_count == 2
    assert result.url_occurrence_count == 1
    assert result.username_occurrence_count == 1
    assert "a@example.com" not in json.dumps(report.to_safe_dict())


def test_text_audit_requires_explicit_text_column(tmp_path):
    source = tmp_path / "text.csv"
    source.write_text("body,status\nhello,ok\n", encoding="utf-8")
    config = dataset_config(source, modality="text", label_columns=["status"])

    with pytest.raises(ValueError, match="text column"):
        audit_dataset(config, fingerprint_dataset(config))


def test_text_near_duplicate_detection_is_deterministic(tmp_path):
    source = tmp_path / "text.csv"
    source.write_text(
        "text,status\n"
        "\"alpha beta gamma delta one two three four\",a\n"
        "\"alpha beta gamma delta changed words three four\",a\n"
        "\"completely different sentence for control\",b\n",
        encoding="utf-8",
    )
    config = dataset_config(source, modality="text", feature_columns=["text"], label_columns=["status"])
    fingerprint = fingerprint_dataset(config)

    first = audit_dataset(config, fingerprint, options={"text_column": "text", "sample_seed": 7})
    second = audit_dataset(config, fingerprint, options={"text_column": "text", "sample_seed": 7})

    assert first.text_result.near_duplicate_candidate_count == second.text_result.near_duplicate_candidate_count


def test_audio_audit_wav_corrupt_empty_duplicates_and_labels(tmp_path):
    root = tmp_path / "audio"
    wav_a = write_wav(root / "happy" / "a.wav", sample_rate=16000, channels=2)
    wav_b = root / "happy" / "b.wav"
    wav_b.write_bytes(wav_a.read_bytes())
    (root / "sad").mkdir(parents=True, exist_ok=True)
    (root / "sad" / "bad.wav").write_bytes(b"not a wav")
    (root / "sad" / "empty.wav").write_bytes(b"")
    config = dataset_config(root, modality="voice", file_format="folder")
    report = audit_dataset(config, fingerprint_dataset(config), options={"folder_label_depth": 1})
    result = report.audio_result

    assert result.file_count == 4
    assert result.readable_file_count == 2
    assert result.corrupt_file_count == 1
    assert result.empty_file_count == 1
    assert result.sample_rate_distribution == {"16000": 2}
    assert result.channel_distribution == {"2": 2}
    assert result.duplicate_hash_group_count == 1
    assert result.label_distribution["folder_label"][0].label == "happy"


def test_image_audit_dimensions_corrupt_duplicates_labels_and_split_overlap(tmp_path):
    root = tmp_path / "images"
    train = write_png(root / "train" / "happy" / "a.png", size=(10, 12))
    duplicate = root / "test" / "happy" / "b.png"
    duplicate.parent.mkdir(parents=True, exist_ok=True)
    duplicate.write_bytes(train.read_bytes())
    write_png(root / "test" / "sad" / "c.png", size=(4, 5))
    (root / "train" / "sad" / "bad.png").parent.mkdir(parents=True, exist_ok=True)
    (root / "train" / "sad" / "bad.png").write_bytes(b"bad")
    config = dataset_config(root, modality="face", file_format="folder")
    report = audit_dataset(config, fingerprint_dataset(config), options={"folder_label_depth": 1})
    result = report.image_result

    assert result.file_count == 4
    assert result.readable_file_count == 3
    assert result.corrupt_file_count == 1
    assert result.width_summary.maximum == 10
    assert result.height_summary.maximum == 12
    assert result.color_mode_distribution == {"RGB": 3}
    assert result.duplicate_hash_group_count == 1
    assert any(issue.code == "train_test_duplicate_hash_overlap" for issue in report.issues)
    assert "image_data" not in json.dumps(report.to_safe_dict())


def test_reporting_save_load_markdown_overwrite_and_path_policy(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("label,value\nA,1\n", encoding="utf-8")
    config = dataset_config(source, label_columns=["label"])
    report = audit_dataset(config, fingerprint_dataset(config))
    output_dir = paths.get_generated_root() / "audits" / f"pytest-{uuid.uuid4().hex}"
    json_path = output_dir / "audit.json"
    md_path = output_dir / "audit.md"
    try:
        saved_json = save_audit_json(report, json_path)
        saved_md = save_audit_markdown(report, md_path)
        assert load_audit_report(saved_json).dataset_name == "fixture"
        markdown = saved_md.read_text(encoding="utf-8")
        assert "Dataset Audit" in markdown
        assert str(tmp_path) not in markdown
        with pytest.raises(FileExistsError):
            save_audit_json(report, json_path)
        save_audit_json(report, json_path, overwrite=True)
        with pytest.raises(ValueError, match="raw dataset"):
            save_audit_markdown(report, paths.get_raw_dataset_root() / "audit.md")
    finally:
        json_path.unlink(missing_ok=True)
        md_path.unlink(missing_ok=True)


def test_markdown_has_no_sensitive_value_leakage(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("email,label\nsecret@example.com,A\n", encoding="utf-8")
    config = dataset_config(source, label_columns=["label"])
    report = audit_dataset(config, fingerprint_dataset(config))
    markdown = create_markdown_summary(report)

    assert "secret@example.com" not in json.dumps(report.to_safe_dict())
    assert "secret@example.com" not in markdown


def test_fingerprint_mismatch_and_missing_source_fail(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("label,value\nA,1\n", encoding="utf-8")
    config = dataset_config(source, label_columns=["label"])
    fingerprint = fingerprint_dataset(config)
    source.write_text("label,value\nA,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        audit_dataset(config, fingerprint)

    missing = dataset_config(tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        audit_dataset(missing)


def test_cli_valid_summary_only_fingerprint_mismatch_and_fail_on_critical(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("label,value\nA,1\n", encoding="utf-8")
    config = dataset_config(source, label_columns=["label"])
    config_payload = config.to_safe_dict()
    config_payload["validation_context"] = "test"
    config_path = write_json(tmp_path / "dataset.json", config_payload)
    fingerprint = fingerprint_dataset(config)
    fingerprint_path = tmp_path / "fingerprint.json"
    fingerprint_path.write_text(json.dumps(fingerprint.to_safe_dict()), encoding="utf-8")
    script = paths.get_backend_root() / "scripts" / "audit_dataset.py"

    summary = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--fingerprint",
            str(fingerprint_path),
            "--summary-only",
            "--fail-on-critical",
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert summary.returncode == 0, summary.stderr
    assert "summary only" in summary.stdout

    source.write_text("label,value\nA,2\n", encoding="utf-8")
    mismatch = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--fingerprint",
            str(fingerprint_path),
            "--summary-only",
        ],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert mismatch.returncode != 0
    assert "fingerprint mismatch" in mismatch.stderr.lower()
