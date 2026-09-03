import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from app.ml.common import paths
from app.ml.governance.artifacts import (
    build_artifact_integrity_summary,
    discover_model_runs,
    load_artifact_manifest,
    verify_artifact_hashes,
    verify_artifact_manifest,
    verify_inactive_status,
)
from app.ml.governance.comparison import build_unimodal_comparison, comparison_metadata
from app.ml.governance.metrics import build_false_negative_safety_summary
from app.ml.governance.model_cards import validate_model_card
from app.ml.governance.readiness import assess_deployment_readiness, assess_evidence_strength
from app.ml.governance.reporting import run_governance_validation
from app.ml.governance.schemas import ArtifactIntegrityStatus


DISCLAIMER = "This model is a research prototype and is not a clinical diagnostic or autonomous suicide-prevention system."


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir(root: Path, modality: str = "text", model_name: str = "text-classification-linear-svm") -> Path:
    run = root / modality / model_name / "1.0.0" / f"{modality}-run"
    run.mkdir(parents=True)
    return run


def _manifest(run: Path, modality: str = "text", *, active: bool = False, synthetic: bool = False) -> Path:
    metrics = _write_json(run / "metrics.json", {"test": {"macro_f1": 0.1}})
    card = run / "model_card.md"
    card.write_text(_card_text(modality), encoding="utf-8")
    config = _write_json(run / "training_config.json", {"modality": modality})
    repro = _write_json(run / "reproducibility_report.json", {"created_at": "2026-07-15T00:00:00+00:00"})
    split = _write_json(run / "split_manifest_reference.json", {"split_manifest_hash": "a" * 64})
    files = [metrics, card, config, repro, split]
    rel_files = [_rel(path) for path in files]
    payload = {
        "active": active,
        "file_hashes": {_rel(path): _sha(path) for path in files},
        "files": rel_files,
        "manifest_version": "1.0.0",
        "modality": "synthetic" if synthetic else modality,
        "model_name": run.parent.parent.name,
        "model_version": "1.0.0",
        "run_id": run.name,
        "source_fingerprint": "b" * 64,
        "split_manifest_hash": "c" * 64,
    }
    return _write_json(run / "artifact_manifest.json", payload)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.get_repository_root()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _card_text(modality: str) -> str:
    extras = {
        "profile": "15-record test limitation. all-positive behavior.",
        "text": "448 suicidal false negatives. shortcut terms warning.",
        "speech": "LOCO domain-shift addendum.",
        "face": "bounded subset training. reviewer independence unverified. no subject IDs. no deployment use.",
    }.get(modality, "")
    return (
        "# Model Card\n"
        "Intended use: research only.\n"
        "Prohibited use: clinical or deployment use.\n"
        "Dataset origin and split strategy are documented.\n"
        "Target label meaning and performance metrics are documented.\n"
        "False negative concerns, privacy risks, fairness limitations, domain-shift risks, and human oversight are documented.\n"
        "Activation status: inactive. Registration status: not registered.\n"
        f"{DISCLAIMER}\n"
        f"{extras}\n"
    )


def _summaries(run: Path) -> Path:
    reports = paths.get_generated_root() / "temporary" / f"phase3j-test-reports-{uuid.uuid4().hex}"
    for modality in ("profile", "text", "speech", "face"):
        modality_run = run.parent.parent.parent.parent / modality / f"{modality}-model" / "1.0.0" / f"{modality}-run"
        modality_run.mkdir(parents=True, exist_ok=True)
        _manifest(modality_run, modality)
        summary_dir = reports / f"{modality}_baseline" / "v1"
        test_metrics = {"macro_f1": 0.2, "balanced_accuracy": 0.3}
        if modality == "profile":
            test_metrics = {"f1": 0.5, "balanced_accuracy": 0.5, "false_negatives": 0, "false_positives": 10}
        if modality == "text":
            test_metrics["suicidal_class"] = {
                "false_negatives": 448,
                "false_positives": 565,
                "suicidal_predicted_as_normal": 145,
                "suicidal_predicted_as_depression": 282,
            }
        if modality == "speech":
            test_metrics["class_false_negatives"] = {"fearful": 228, "surprised": 89}
        if modality == "face":
            test_metrics["false_negatives_by_class"] = {"fear": 96}
        _write_json(
            summary_dir / f"{modality}_baseline_summary.json",
            {
                "artifact_path": str(modality_run),
                "model_name": f"{modality}-model",
                "model_version": "1.0.0",
                "run_id": f"{modality}-run",
                "train_count": 10,
                "validation_count": 5,
                "test_count": 5,
                "selected_candidate": {"estimator_type": "random_forest"},
                "test_metrics": test_metrics,
                "candidate_registration_occurred": False,
                "model_became_active": False,
                "activation_status": "inactive",
                "registration_status": "not_registered",
            },
        )
    _write_json(
        reports / "speech_domain_shift" / "v1" / "speech_domain_shift_summary.json",
        {"corpus_gap_summary": {"pooled_test_macro_f1": 0.3569, "loco_mean_macro_f1": 0.1910}, "activation_status": "inactive"},
    )
    return reports


def test_artifact_integrity_valid_missing_hash_mismatch_malformed_and_active(tmp_path):
    run = _run_dir(tmp_path)
    manifest_path = _manifest(run)
    manifest = load_artifact_manifest(manifest_path)
    assert verify_artifact_manifest(manifest) == ArtifactIntegrityStatus.verified
    assert verify_artifact_hashes(manifest) == ArtifactIntegrityStatus.verified
    assert verify_inactive_status(manifest) == ArtifactIntegrityStatus.verified

    missing_manifest = dict(manifest)
    missing_manifest["files"] = [str(tmp_path / "missing.json")]
    missing_manifest["file_hashes"] = {str(tmp_path / "missing.json"): "0" * 64}
    assert verify_artifact_hashes(missing_manifest) == ArtifactIntegrityStatus.missing

    bad_hash = dict(manifest)
    bad_hash["file_hashes"] = dict(manifest["file_hashes"])
    bad_hash["file_hashes"][manifest["files"][0]] = "0" * 64
    assert verify_artifact_hashes(bad_hash) == ArtifactIntegrityStatus.hash_mismatch

    assert verify_artifact_manifest({"files": []}) == ArtifactIntegrityStatus.malformed
    active_manifest = dict(manifest, active=True)
    assert verify_inactive_status(active_manifest) == ArtifactIntegrityStatus.malformed


def test_discovery_excludes_synthetic_and_marks_evaluation_only(tmp_path):
    real_run = _run_dir(tmp_path, "text")
    _manifest(real_run, "text")
    synthetic_run = _run_dir(tmp_path, "synthetic", "pytest-model")
    _manifest(synthetic_run, "synthetic", synthetic=True)
    eval_run = tmp_path / "speech-domain-evaluation" / "hold_out_x" / "domain-run"
    eval_run.mkdir(parents=True)
    _manifest(eval_run, "speech")
    runs = discover_model_runs(model_root=tmp_path, reports_root=tmp_path, modalities=["text", "speech", "synthetic"])
    summary = build_artifact_integrity_summary(runs)
    assert any(item["classification"] == "synthetic_excluded" for item in summary["unexpected_artifacts"])
    assert any(item["classification"] == "evaluation_only" for item in summary["unexpected_artifacts"])


@pytest.mark.parametrize(
    ("modality", "expected_readiness", "expected_evidence"),
    [
        ("profile", "research_baseline_only", "very_low"),
        ("text", "research_evaluated_not_deployable", "low"),
        ("speech", "research_evaluated_not_deployable", "low"),
        ("face", "research_baseline_only", "very_low"),
        ("mood", "blocked_pending_data", "none"),
        ("behavioral", "engineering_only", "none"),
        ("fusion", "blocked_pending_data", "none"),
    ],
)
def test_readiness_no_deployable_model(modality, expected_readiness, expected_evidence):
    assert assess_deployment_readiness(modality).value == expected_readiness
    assert assess_evidence_strength(modality).value == expected_evidence
    assert expected_readiness != "deployable"


def test_model_cards_required_findings(tmp_path):
    for modality in ("profile", "text", "speech", "face"):
        path = tmp_path / f"{modality}.md"
        path.write_text(_card_text(modality), encoding="utf-8")
        result = validate_model_card(path, modality)
        assert result["clinical_disclaimer_present"]
        assert result["valid"]
    missing = tmp_path / "missing_privacy.md"
    missing.write_text(f"Intended use. Prohibited use. {DISCLAIMER}", encoding="utf-8")
    result = validate_model_card(missing, "text")
    assert "privacy_risks" in result["missing"]
    assert "false_negative_concerns" in result["missing"]


def test_comparison_has_no_invalid_ranking_and_reports_task_differences():
    rows = build_unimodal_comparison(
        {
            "profile": {"test_metrics": {"f1": 0.5}},
            "text": {"test_metrics": {"macro_f1": 0.7816, "suicidal_class": {"false_negatives": 448}}},
            "speech": {"test_metrics": {"macro_f1": 0.3569}},
            "face": {"test_metrics": {"macro_f1": 0.1645}},
        }
    )
    tasks = {row.modality: row.task for row in rows}
    assert "self-reported depression" in tasks["profile"]
    assert "acted speech emotion" in tasks["speech"]
    assert comparison_metadata()["ranking_prohibited"] is True
    assert "not directly comparable" in rows[0].comparability_warning


def test_safety_summaries_do_not_sum_cross_modality_false_negatives():
    summary = build_false_negative_safety_summary(
        {
            "profile": {"test_metrics": {"false_negatives": 0, "false_positives": 10}},
            "text": {"test_metrics": {"suicidal_class": {"false_negatives": 448}}},
            "speech": {"test_metrics": {"class_false_negatives": {"fearful": 228}}},
            "face": {"test_metrics": {"false_negatives_by_class": {"fear": 96}}},
        }
    )
    assert "All-positive" in summary["profile"]["safety_interpretation"]
    assert summary["text"]["suicidal_false_negatives"] == 448
    assert "not suicide-risk" in summary["speech"]["safety_interpretation"]
    assert "not suicide-risk" in summary["face"]["safety_interpretation"]
    assert summary["cross_modality_total"] is None


def test_reports_overwrite_protection_and_deterministic_outputs(tmp_path):
    base = _run_dir(tmp_path, "seed")
    reports = _summaries(base)
    output = paths.get_generated_root() / "temporary" / f"phase3j-output-{uuid.uuid4().hex}"
    result = run_governance_validation(model_root=tmp_path, reports_root=reports, output_dir=output, overwrite=True)
    assert result["exit_code"] == 0
    summary = json.loads((output / "model_governance_summary.json").read_text(encoding="utf-8"))
    assert summary["final_research_status"] == "research_models_complete_not_deployable"
    assert "D:\\" not in json.dumps(summary)
    assert "secret" not in json.dumps(summary).lower()
    with pytest.raises(FileExistsError):
        run_governance_validation(model_root=tmp_path, reports_root=reports, output_dir=output, overwrite=False)


def test_cli_normal_strict_inventory_and_filters(tmp_path):
    base = _run_dir(tmp_path, "seed")
    reports = _summaries(base)
    output = paths.get_generated_root() / "temporary" / f"phase3j-cli-{uuid.uuid4().hex}"
    script = paths.get_backend_root() / "scripts" / "validate_model_governance.py"
    normal = subprocess.run(
        [sys.executable, str(script), "--model-root", str(tmp_path), "--reports-root", str(reports), "--output-dir", str(output), "--overwrite", "--verify-hashes", "--validate-model-cards", "--modalities", "profile", "text", "speech", "face"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert normal.returncode == 0
    assert "active_models=0" in normal.stdout
    strict = subprocess.run(
        [sys.executable, str(script), "--model-root", str(tmp_path), "--reports-root", str(reports), "--output-dir", str(output), "--overwrite", "--strict"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert strict.returncode == 1
    assert "Strict mode failed as expected" in strict.stdout
    inventory = subprocess.run(
        [sys.executable, str(script), "--model-root", str(tmp_path), "--reports-root", str(reports), "--output-dir", str(output), "--overwrite", "--inventory-only"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert inventory.returncode == 0
