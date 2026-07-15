from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.ml.common import paths
from app.ml.common.fingerprinting import fingerprint_dataset
from app.ml.common.schemas import DatasetConfig
from app.ml.validation.checks import (
    check_conflict_quarantine_present,
    check_dataset_version_present,
    check_fingerprint_matches_source,
    check_identifiers_not_in_features,
    check_json_schema_loads,
    check_metadata_not_in_features,
    check_model_artifacts_absent,
    check_no_absolute_paths,
    check_no_nan_or_infinity,
    check_output_outside_raw_dataset,
    check_required_file_exists,
    check_split_manifest_absent,
    check_target_not_in_features,
    check_training_outputs_absent,
)
from app.ml.validation.constants import PHASE2_VALIDATION_VERSION, READINESS_POLICY_VERSION
from app.ml.validation.cross_modality import validate_phase2_cross_modality
from app.ml.validation.inventory import create_phase2_artifact_inventory
from app.ml.validation.readiness import classify_modality_readiness
from app.ml.validation.reporting import (
    create_phase2_markdown_summary,
    create_readiness_matrix,
    load_phase2_validation_report,
    save_phase2_validation_json,
    save_phase2_validation_markdown,
    save_supporting_reports,
)
from app.ml.validation.schemas import ModalityReadiness, ValidationStatus


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_common_checks_detect_artifact_schema_leakage_and_forbidden_outputs(tmp_path):
    missing = check_required_file_exists(tmp_path / "missing.json", modality="test")
    assert missing.status == ValidationStatus.FAILED

    malformed = tmp_path / "bad.json"
    malformed.write_text("{", encoding="utf-8")
    assert check_json_schema_loads(malformed, modality="test").status == ValidationStatus.FAILED
    assert check_dataset_version_present({}, modality="test").status == ValidationStatus.FAILED

    raw_violation = check_output_outside_raw_dataset(paths.get_raw_dataset_root() / "bad.csv", modality="test")
    assert raw_violation.status == ValidationStatus.BLOCKED
    assert check_no_absolute_paths("C:\\Users\\Name\\raw.csv", modality="test").status == ValidationStatus.FAILED

    csv_path = tmp_path / "features.csv"
    csv_path.write_text("record_id,target,feature\n1,yes,NaN\n", encoding="utf-8")
    assert check_no_nan_or_infinity(csv_path, modality="test").status == ValidationStatus.BLOCKED
    assert check_target_not_in_features(["target"], ["target"], modality="test").status == ValidationStatus.BLOCKED
    assert check_identifiers_not_in_features(["student_id"], ["student_id"], modality="test").status == ValidationStatus.BLOCKED
    assert check_metadata_not_in_features(["source_file"], ["source_file"], modality="test").status == ValidationStatus.FAILED

    model = tmp_path / "model.pkl"
    model.write_text("not a model", encoding="utf-8")
    split = tmp_path / "split_manifest.json"
    split.write_text("{}", encoding="utf-8")
    training = tmp_path / "training_metrics.json"
    training.write_text("{}", encoding="utf-8")
    assert check_model_artifacts_absent(tmp_path, modality="test").status == ValidationStatus.BLOCKED
    assert check_split_manifest_absent(tmp_path, modality="test").status == ValidationStatus.BLOCKED
    assert check_training_outputs_absent(tmp_path, modality="test").status == ValidationStatus.BLOCKED
    assert check_conflict_quarantine_present(tmp_path / "none.csv", modality="test").status == ValidationStatus.FAILED


def test_fingerprint_mismatch_and_verified_source(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    config = DatasetConfig(
        dataset_name="fixture",
        dataset_version="v1",
        modality="profile",
        source_path=source,
        file_format="csv",
        label_columns=["b"],
        feature_columns=["a"],
        identifier_columns=[],
        sensitive_columns=[],
        excluded_columns=[],
        expected_columns=["a", "b"],
        missing_value_policy="preserve",
        duplicate_policy="report_only",
        validation_context="test",
    )
    fingerprint = fingerprint_dataset(config)
    config_path = write_json(tmp_path / "dataset.json", config.to_safe_dict() | {"validation_context": "test"})
    fingerprint_path = write_json(tmp_path / "fingerprint.json", fingerprint.to_safe_dict())

    assert check_fingerprint_matches_source(fingerprint_path, config_path, modality="test").status == ValidationStatus.PASSED
    source.write_text("a,b\n9,2\n", encoding="utf-8")
    assert check_fingerprint_matches_source(fingerprint_path, config_path, modality="test").status == ValidationStatus.BLOCKED


def test_readiness_rules_are_deterministic():
    readiness, warnings, blockers = classify_modality_readiness(
        "face",
        {"critical_duplicate_conflicts": True},
        [],
    )
    assert readiness == ModalityReadiness.BLOCKED_PENDING_LEAKAGE_RESOLUTION
    assert blockers

    readiness, warnings, blockers = classify_modality_readiness("text", {"warnings": ["duplicate restrictions"]}, [])
    assert readiness == ModalityReadiness.READY_WITH_RESTRICTIONS
    assert not blockers

    readiness, warnings, blockers = classify_modality_readiness(
        "behavioral",
        {"synthetic_only": True, "real_source_exists": False},
        [],
    )
    assert readiness == ModalityReadiness.ENGINEERING_TESTS_ONLY
    assert blockers

    readiness, warnings, blockers = classify_modality_readiness("speech", {"missing_grouping_key": True}, [])
    assert readiness == ModalityReadiness.READY_WITH_RESTRICTIONS

    readiness, warnings, blockers = classify_modality_readiness("profile", {"fingerprint_mismatch": True}, [])
    assert readiness == ModalityReadiness.BLOCKED_PENDING_DATA

    readiness, warnings, blockers = classify_modality_readiness("profile", {}, [])
    assert readiness == ModalityReadiness.READY_FOR_SPLIT_DESIGN


def test_real_modality_readiness_and_cross_modality_findings():
    report, inventory = validate_phase2_cross_modality(skip_source_reverification=True)
    readiness = {result.modality: result.readiness_classification for result in report.modalities}

    assert report.validation_version == PHASE2_VALIDATION_VERSION
    assert report.readiness_policy_version == READINESS_POLICY_VERSION
    assert readiness["dass21"] == ModalityReadiness.SCORING_ONLY_NOT_ML
    assert readiness["profile"] == ModalityReadiness.READY_WITH_RESTRICTIONS
    assert readiness["mood"] == ModalityReadiness.BLOCKED_PENDING_DATA
    assert readiness["text"] == ModalityReadiness.READY_WITH_RESTRICTIONS
    assert readiness["speech"] == ModalityReadiness.READY_WITH_RESTRICTIONS
    assert readiness["face"] == ModalityReadiness.BLOCKED_PENDING_LEAKAGE_RESOLUTION
    assert readiness["behavioral"] == ModalityReadiness.ENGINEERING_TESTS_ONLY
    assert any("no common participant key" in finding for finding in report.global_findings)
    assert any("fusion training is not valid" in finding for finding in report.global_findings)
    assert any(item.expected_classification == "expected" for item in inventory)


def test_cross_modality_consistency_checks_are_reported():
    report, _ = validate_phase2_cross_modality(skip_source_reverification=True)
    matrix = create_readiness_matrix(report)
    assert any(row["modality"] == "fusion" and row["readiness"] == "blocked_pending_data" for row in matrix)
    assert "multimodal fusion" in create_phase2_markdown_summary(report).lower()
    assert report.blocked_checks >= 1
    assert "profile" in {result.modality for result in report.modalities}


def test_reporting_json_markdown_matrix_inventory_and_overwrite(tmp_path):
    report, inventory = validate_phase2_cross_modality(modalities=["dass21", "profile"], skip_source_reverification=True)
    json_path = save_phase2_validation_json(report, tmp_path)
    markdown_path = save_phase2_validation_markdown(report, tmp_path)
    support = save_supporting_reports(report, inventory, tmp_path)

    loaded = load_phase2_validation_report(json_path)
    assert loaded.validation_version == PHASE2_VALIDATION_VERSION
    assert markdown_path.read_text(encoding="utf-8").startswith("# Phase 2")
    assert support["readiness_matrix"].exists()
    assert support["artifact_inventory"].exists()
    assert support["blockers"].exists()
    assert support["next_actions"].exists()
    assert "C:\\Users" not in json_path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError):
        save_phase2_validation_json(report, tmp_path)


def test_artifact_inventory_expected_missing_and_unexpected_model_artifact(tmp_path):
    model_path = paths.get_generated_root() / "temporary" / "phase2-validation-test-model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        model_path.write_text("unexpected", encoding="utf-8")
        inventory = create_phase2_artifact_inventory(modalities=["behavioral"])
        assert any(item.artifact_type == "model artifact" and item.expected_classification == "unexpected" for item in inventory)
        assert any(item.artifact_type == "source fingerprint" and not item.exists for item in inventory) is False
    finally:
        model_path.unlink(missing_ok=True)


def run_cli(*args: str, timeout: int = 120):
    script = paths.get_backend_root() / "scripts" / "validate_phase2_preprocessing.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def test_cli_modes_summary_inventory_strict_warning_overwrite_and_source_reverification(tmp_path):
    common = [
        "--config-dir",
        "../ml-research/configs",
        "--generated-root",
        "../generated",
    ]
    summary = run_cli(*common, "--summary-only", "--skip-source-reverification")
    assert summary.returncode == 0, summary.stderr
    assert "profile: ready_with_restrictions" in summary.stdout

    strict = run_cli(*common, "--summary-only", "--skip-source-reverification", "--strict")
    assert strict.returncode == 1

    filtered = run_cli(*common, "--summary-only", "--modalities", "profile", "--skip-source-reverification")
    assert filtered.returncode == 0
    assert "profile: ready_with_restrictions" in filtered.stdout
    assert "text:" not in filtered.stdout

    inventory_dir = tmp_path / "inventory"
    inventory = run_cli(*common, "--output-dir", str(inventory_dir), "--inventory-only", "--modalities", "profile", "--skip-source-reverification")
    assert inventory.returncode == 0, inventory.stderr
    assert (inventory_dir / "phase2_artifact_inventory.json").exists()

    overwrite = run_cli(*common, "--output-dir", str(inventory_dir), "--inventory-only", "--modalities", "profile", "--skip-source-reverification")
    assert overwrite.returncode == 2
    assert "overwrite" in overwrite.stderr.lower()

    fail_on_warning = run_cli(*common, "--summary-only", "--modalities", "profile", "--skip-source-reverification", "--fail-on-warning")
    assert fail_on_warning.returncode == 1

    source_verify = run_cli(*common, "--summary-only", "--modalities", "profile")
    assert source_verify.returncode == 0, source_verify.stderr
