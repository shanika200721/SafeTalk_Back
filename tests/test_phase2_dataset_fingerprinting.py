import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic.v1 import ValidationError

from app.ml.common import paths
from app.ml.common.fingerprinting import (
    dataset_config_hash,
    fingerprint_dataset,
    load_dataset_fingerprint,
    save_dataset_fingerprint,
    verify_dataset_fingerprint,
)
from app.ml.common.hashing import (
    DATASET_FINGERPRINT_VERSION,
    create_directory_fingerprint,
    create_file_fingerprint,
    hash_bytes,
    hash_directory,
    hash_json_data,
    hash_text,
    normalize_relative_path,
    sha256_file,
)
from app.ml.common.schemas import DatasetConfig, DatasetFingerprint


def dataset_config_data(source_path, **overrides):
    data = {
        "dataset_name": "fixture-dataset",
        "dataset_version": "v1",
        "modality": "text",
        "source_path": source_path,
        "file_format": "csv",
        "label_columns": [],
        "feature_columns": [],
        "identifier_columns": [],
        "sensitive_columns": [],
        "excluded_columns": [],
        "expected_columns": [],
        "missing_value_policy": "preserve",
        "duplicate_policy": "report_only",
        "notes": "test note",
        "is_raw_source": True,
        "validation_context": "test",
    }
    data.update(overrides)
    return data


def write_config(path: Path, source_path: Path, **overrides):
    payload = dataset_config_data(str(source_path), **overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def make_fingerprint(**overrides):
    file_hash = hashlib.sha256(b"abc").hexdigest()
    data = {
        "dataset_name": "fixture-dataset",
        "dataset_version": "v1",
        "modality": "text",
        "source_relative_path": "fixture/source.csv",
        "source_type": "file",
        "file_count": 1,
        "total_bytes": 3,
        "combined_sha256": file_hash,
        "files": [
            {
                "relative_path": "fixture/source.csv",
                "sha256": file_hash,
                "size_bytes": 3,
                "extension": "csv",
            }
        ],
        "skipped_files": [],
        "generated_at": datetime(2026, 7, 14, tzinfo=timezone.utc),
        "fingerprint_version": DATASET_FINGERPRINT_VERSION,
    }
    data.update(overrides)
    return DatasetFingerprint(**data)


def test_file_hash_same_and_different_content(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    third = tmp_path / "third.txt"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")
    third.write_text("different", encoding="utf-8")

    assert sha256_file(first, allow_outside_project=True) == sha256_file(second, allow_outside_project=True)
    assert sha256_file(first, allow_outside_project=True) != sha256_file(third, allow_outside_project=True)
    assert hash_bytes(b"same") == hash_text("same")
    assert hash_json_data({"b": 2, "a": 1}) == hash_json_data({"a": 1, "b": 2})


def test_changing_one_byte_changes_hash(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"abcdef")
    before = sha256_file(source, allow_outside_project=True)
    source.write_bytes(b"abcdeg")

    assert sha256_file(source, allow_outside_project=True) != before


def test_large_file_is_read_in_chunks(tmp_path, monkeypatch):
    source = tmp_path / "large.txt"
    source.write_bytes(b"x" * 1024)
    read_sizes = []
    original_open = Path.open

    class TrackingReader:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def read(self, size=-1):
            read_sizes.append(size)
            return self.handle.read(size)

    def tracking_open(self, *args, **kwargs):
        handle = original_open(self, *args, **kwargs)
        if self == source.resolve() and args and args[0] == "rb":
            return TrackingReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)
    sha256_file(source, chunk_size=17, allow_outside_project=True)

    assert 17 in read_sizes
    assert read_sizes.count(17) > 1


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        sha256_file(tmp_path / "missing.csv", allow_outside_project=True)


def test_unsupported_source_type_raises(tmp_path):
    source = tmp_path / "source.exe"
    source.write_text("x", encoding="utf-8")
    config = DatasetConfig(**dataset_config_data(source, file_format="csv"))

    with pytest.raises(ValueError, match="not compatible"):
        fingerprint_dataset(config)


def test_directory_hash_deterministic_regardless_creation_order(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.txt").write_text("b", encoding="utf-8")
    (first / "a.txt").write_text("a", encoding="utf-8")
    (second / "a.txt").write_text("a", encoding="utf-8")
    (second / "b.txt").write_text("b", encoding="utf-8")

    left = create_directory_fingerprint(first, root=first, allow_outside_project=True)
    right = create_directory_fingerprint(second, root=second, allow_outside_project=True)

    assert left["combined_sha256"] == right["combined_sha256"]


def test_directory_hash_changes_for_filename_content_and_size(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    item = source / "a.txt"
    item.write_text("abc", encoding="utf-8")
    baseline = create_directory_fingerprint(source, root=source, allow_outside_project=True)["combined_sha256"]
    item.rename(source / "b.txt")
    renamed = create_directory_fingerprint(source, root=source, allow_outside_project=True)["combined_sha256"]
    (source / "b.txt").write_text("abd", encoding="utf-8")
    content_changed = create_directory_fingerprint(source, root=source, allow_outside_project=True)["combined_sha256"]
    (source / "b.txt").write_text("abcd", encoding="utf-8")
    size_changed = create_directory_fingerprint(source, root=source, allow_outside_project=True)["combined_sha256"]

    assert renamed != baseline
    assert content_changed != renamed
    assert size_changed != content_changed


def test_nested_directories_ignored_files_counts_and_bytes(tmp_path):
    source = tmp_path / "source"
    nested = source / "nested"
    cache = source / "__pycache__"
    nested.mkdir(parents=True)
    cache.mkdir()
    (source / "a.txt").write_text("aa", encoding="utf-8")
    (nested / "b.csv").write_text("bbb", encoding="utf-8")
    (source / ".DS_Store").write_text("ignored", encoding="utf-8")
    (cache / "module.pyc").write_bytes(b"ignored")

    fingerprint = create_directory_fingerprint(source, root=source, allow_outside_project=True)

    assert fingerprint["file_count"] == 2
    assert fingerprint["total_bytes"] == 5
    assert [file["relative_path"] for file in fingerprint["files"]] == ["a.txt", "nested/b.csv"]
    assert fingerprint["skipped_file_count"] == 2


def test_empty_directory_rejected_unless_allowed(tmp_path):
    with pytest.raises(ValueError, match="no fingerprintable files"):
        create_directory_fingerprint(tmp_path, root=tmp_path, allow_outside_project=True)

    fingerprint = create_directory_fingerprint(tmp_path, root=tmp_path, allow_empty=True, allow_outside_project=True)
    assert fingerprint["file_count"] == 0


def test_duplicate_hash_groups_identified(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("same", encoding="utf-8")
    (source / "b.txt").write_text("same", encoding="utf-8")
    config = DatasetConfig(**dataset_config_data(source, file_format="folder"))

    fingerprint = fingerprint_dataset(config)
    groups = fingerprint.duplicate_hash_groups()

    assert list(groups.values()) == [["a.txt", "b.txt"]]


def test_symlink_escape_is_rejected_or_skipped(tmp_path):
    source = tmp_path / "source"
    outside = tmp_path / "outside.txt"
    source.mkdir()
    outside.write_text("outside", encoding="utf-8")
    link = source / "escape.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available")

    with pytest.raises(ValueError, match="escapes"):
        create_directory_fingerprint(source, root=source, allow_outside_project=True)


def test_reports_cannot_be_written_inside_final_dataset():
    fingerprint = make_fingerprint()

    with pytest.raises(ValueError, match="raw dataset"):
        save_dataset_fingerprint(fingerprint, paths.get_raw_dataset_root() / "unsafe.json")


def test_absolute_paths_do_not_appear_in_saved_json(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    config = DatasetConfig(**dataset_config_data(source))
    fingerprint = fingerprint_dataset(config)
    output = paths.get_generated_manifests_root() / "fingerprints" / f"pytest-{uuid.uuid4().hex}.json"
    try:
        saved = save_dataset_fingerprint(fingerprint, output)
        text = saved.read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "source.csv" in text
    finally:
        output.unlink(missing_ok=True)


def test_traversal_paths_rejected_and_windows_separators_normalized():
    with pytest.raises(ValueError, match="traversal"):
        make_fingerprint(source_relative_path="../source.csv")

    fingerprint = make_fingerprint(source_relative_path="folder\\source.csv")
    assert fingerprint.source_relative_path == "folder/source.csv"


def test_working_directory_changes_do_not_change_hashes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("abc", encoding="utf-8")
    before = create_directory_fingerprint(source, root=source, allow_outside_project=True)
    monkeypatch.chdir(tmp_path)
    after = create_directory_fingerprint(source, root=source, allow_outside_project=True)

    assert before["combined_sha256"] == after["combined_sha256"]
    assert normalize_relative_path(source / "a.txt", source) == "a.txt"
    assert hash_directory(source, allow_outside_project=True) == hash_directory(source, allow_outside_project=True)


def test_schema_validation_rejects_invalid_values():
    with pytest.raises(ValidationError, match="64 hexadecimal"):
        make_fingerprint(combined_sha256="bad")
    with pytest.raises(ValidationError, match="unique"):
        make_fingerprint(files=make_fingerprint().files * 2, file_count=2, total_bytes=6)
    with pytest.raises(ValidationError, match="file_count"):
        make_fingerprint(file_count=2)
    with pytest.raises(ValidationError, match="total_bytes"):
        make_fingerprint(total_bytes=4)
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_fingerprint(generated_at=datetime(2026, 7, 14))


def test_service_fingerprints_single_file_and_folder(tmp_path):
    file_source = tmp_path / "source.csv"
    folder_source = tmp_path / "folder"
    file_source.write_text("a,b\n1,2\n", encoding="utf-8")
    folder_source.mkdir()
    (folder_source / "one.txt").write_text("one", encoding="utf-8")

    file_fingerprint = fingerprint_dataset(DatasetConfig(**dataset_config_data(file_source)))
    folder_fingerprint = fingerprint_dataset(DatasetConfig(**dataset_config_data(folder_source, file_format="folder")))

    assert file_fingerprint.file_count == 1
    assert folder_fingerprint.file_count == 1
    assert folder_fingerprint.source_type == "directory"


def test_save_load_round_trip_and_overwrite_protection(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    fingerprint = fingerprint_dataset(DatasetConfig(**dataset_config_data(source)))
    output = paths.get_generated_manifests_root() / "fingerprints" / f"pytest-{uuid.uuid4().hex}.json"
    try:
        save_dataset_fingerprint(fingerprint, output)
        loaded = load_dataset_fingerprint(output)
        assert loaded.combined_sha256 == fingerprint.combined_sha256
        with pytest.raises(FileExistsError):
            save_dataset_fingerprint(fingerprint, output)
        save_dataset_fingerprint(fingerprint, output, overwrite=True)
    finally:
        output.unlink(missing_ok=True)


def test_verification_passes_and_fails_after_modification(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    config = DatasetConfig(**dataset_config_data(source))
    fingerprint = fingerprint_dataset(config)

    assert verify_dataset_fingerprint(fingerprint, config)
    assert fingerprint.verify_current_source(source)
    source.write_text("a,b\n1,3\n", encoding="utf-8")
    assert not verify_dataset_fingerprint(fingerprint, config)
    assert not fingerprint.verify_current_source(source)


def test_config_hash_is_deterministic_and_ignores_notes(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    first = DatasetConfig(**dataset_config_data(source, notes="first"))
    second = DatasetConfig(**dataset_config_data(source, notes="second"))

    assert dataset_config_hash(first) == dataset_config_hash(second)


def test_cli_success_verification_failure_and_missing_source(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    config_path = write_config(tmp_path / "dataset.json", source)
    output = paths.get_generated_manifests_root() / "fingerprints" / f"pytest-cli-{uuid.uuid4().hex}.json"
    missing_config = write_config(tmp_path / "missing.json", tmp_path / "missing.csv")
    backend_root = paths.get_backend_root()
    script = backend_root / "scripts" / "fingerprint_dataset.py"

    try:
        generated = subprocess.run(
            [sys.executable, str(script), "--config", str(config_path), "--output", str(output)],
            cwd=backend_root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert generated.returncode == 0, generated.stderr
        assert output.exists()

        verified = subprocess.run(
            [sys.executable, str(script), "--config", str(config_path), "--output", str(output), "--verify-existing"],
            cwd=backend_root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert verified.returncode == 0, verified.stderr
        assert "verification: unchanged" in verified.stdout

        source.write_text("a,b\n1,3\n", encoding="utf-8")
        changed = subprocess.run(
            [sys.executable, str(script), "--config", str(config_path), "--output", str(output), "--verify-existing"],
            cwd=backend_root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert changed.returncode != 0
        assert "verification: changed" in changed.stdout

        missing = subprocess.run(
            [sys.executable, str(script), "--config", str(missing_config), "--summary-only"],
            cwd=backend_root,
            text=True,
            capture_output=True,
            check=False,
        )
        assert missing.returncode != 0
        assert "does not exist" in missing.stderr
    finally:
        output.unlink(missing_ok=True)
