from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from app.ml.common import paths
from app.ml.training.face.constants import FACE_LABELS
from app.ml.training.face.data import (
    load_face_duplicate_decisions,
    load_face_images_for_split,
    validate_face_training_contract,
)
from app.ml.training.face.estimators import face_candidate_specs
from app.ml.training.face.evaluation import evaluate_face_predictions, selection_score, train_validation_gap
from app.ml.training.face.preprocessing import fit_train_only_scaler, load_48x48_grayscale, transform_with_scaler
from app.ml.training.face.runner import run_face_baseline

FINGERPRINT = "f" * 64


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_image(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (48, 48), color=value).save(path, format="PNG")
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def face_baseline_fixture():
    repo = paths.get_repository_root()
    root = repo / "generated" / "temporary" / "phase3i_unit"
    extra = [repo / "generated" / "reports" / "face_baseline" / "phase3i_unit", repo / "ml_models" / "face" / "phase3i-unit"]
    for path in [root, *extra]:
        if path.exists():
            shutil.rmtree(path)
    rows = []
    assignments = []
    split_ids = {"train": [], "validation": [], "test": []}
    counter = 0
    for split in ("train", "validation", "test"):
        for label_index, label in enumerate(FACE_LABELS):
            for rep in range(2):
                counter += 1
                record_id = f"{split}-{label}-{rep}"
                rel = f"generated/temporary/phase3i_unit/images/{record_id}.png"
                digest = _write_image(repo / rel, (20 + label_index * 20 + rep + counter) % 255)
                row = {
                    "record_id": record_id,
                    "source_split": "original_train" if split != "test" else "original_test",
                    "original_label": label,
                    "canonical_emotion_label": label,
                    "image_relative_path": rel,
                    "image_hash": digest,
                    "readable": "True",
                    "duplicate_group_id": "",
                    "remediation_action": "keep",
                    "remediation_policy_version": "1.0.0",
                }
                rows.append(row)
                assignments.append(
                    {
                        "record_id": record_id,
                        "split": split,
                        "canonical_emotion_label": label,
                        "image_hash": digest,
                        "duplicate_group_id": "",
                        "original_split": row["source_split"],
                    }
                )
                split_ids[split].append(record_id)
    manifest = root / "face_deduplicated_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    assignments_path = root / "face_split_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assignments[0].keys()))
        writer.writeheader()
        writer.writerows(assignments)
    decisions_path = root / "face_remediation_decisions.csv"
    with decisions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "action", "representative_id", "group_id", "reason", "policy_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"record_id": row["record_id"], "action": "keep", "representative_id": row["record_id"], "group_id": "", "reason": "unit", "policy_version": "1.0.0"})
        writer.writerow({"record_id": "quarantine-x", "action": "quarantine_cross_label", "representative_id": "", "group_id": "g", "reason": "unit", "policy_version": "1.0.0"})
        writer.writerow({"record_id": "excluded-x", "action": "exclude_duplicate", "representative_id": "", "group_id": "g2", "reason": "unit", "policy_version": "1.0.0"})
    quarantine = _write_json(root / "face_cross_label_quarantine.json", {"count": 1, "quarantined_records": [{"record_id": "quarantine-x"}]})
    fingerprint = _write_json(root / "fingerprint.json", {"combined_sha256": FINGERPRINT})
    import hashlib

    dedup_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    split_manifest = _write_json(
        root / "face_split_manifest.json",
        {
            "source_fingerprint": FINGERPRINT,
            "deduplicated_view_hash": dedup_hash,
            "canonical_manifest_hash": "a" * 64,
            "duplicate_policy_hash": "b" * 64,
            "random_seed": 1,
            "train_ids": split_ids["train"],
            "validation_ids": split_ids["validation"],
            "test_ids": split_ids["test"],
            "excluded_ids": {"excluded-x": "unit"},
            "quarantined_ids": {"quarantine-x": "unit"},
            "label_distributions": {},
            "duplicate_overlap_count": 0,
            "image_hash_overlap_count": 0,
            "record_overlap_count": 0,
        },
    )
    config = _write_json(
        root / "config.json",
        {
            "feature_set": "image_statistics",
            "max_candidate_count": 1,
            "hyperparameter_search": {"random_forest": {"n_estimators": [5], "max_depth": [3], "class_weight": [None], "random_state": 1, "n_jobs": 1}},
        },
    )
    yield {
        "root": root,
        "manifest": manifest,
        "assignments": assignments_path,
        "decisions": decisions_path,
        "quarantine": quarantine,
        "fingerprint": fingerprint,
        "split_manifest": split_manifest,
        "config": config,
    }
    for path in [root, *extra]:
        if path.exists():
            shutil.rmtree(path)


def _contract(fixture, **kwargs):
    args = {
        "deduplicated_manifest_path": fixture["manifest"],
        "split_manifest_path": fixture["split_manifest"],
        "split_assignments_path": fixture["assignments"],
        "quarantine_path": fixture["quarantine"],
        "duplicate_decisions_path": fixture["decisions"],
        "source_fingerprint_path": fixture["fingerprint"],
        "require_replay": False,
    }
    args.update(kwargs)
    return validate_face_training_contract(**args)


def test_data_contract_split_quarantine_duplicates_hash_and_fingerprint(face_baseline_fixture):
    contract = _contract(face_baseline_fixture)
    assert contract["train_count"] == 14
    assert contract["validation_count"] == 14
    assert contract["test_count"] == 14
    assert contract["image_hash_overlap_count"] == 0
    assert contract["duplicate_group_overlap_count"] == 0
    assert contract["reviewer_independence_status"] == "reviewer_independence_unverified"
    assert load_face_duplicate_decisions(face_baseline_fixture["decisions"])["excluded-x"] == "exclude_duplicate"

    bad_fingerprint = _write_json(face_baseline_fixture["root"] / "bad_fingerprint.json", {"combined_sha256": "e" * 64})
    with pytest.raises(ValueError, match="source fingerprint"):
        _contract(face_baseline_fixture, source_fingerprint_path=bad_fingerprint)

    assignments = pd.read_csv(face_baseline_fixture["assignments"], dtype=str)
    assignments.loc[assignments["split"] == "test", "image_hash"] = assignments.iloc[0]["image_hash"]
    bad_assignments = face_baseline_fixture["root"] / "bad_assignments.csv"
    assignments.to_csv(bad_assignments, index=False)
    with pytest.raises(ValueError, match="isolation"):
        _contract(face_baseline_fixture, split_assignments_path=bad_assignments)


def test_preprocessing_grayscale_dimensions_scaling_and_no_metadata_features(face_baseline_fixture):
    rows = pd.read_csv(face_baseline_fixture["manifest"], dtype=str)
    row = rows.iloc[0]
    array = load_48x48_grayscale(row["image_relative_path"])
    assert array.shape == (48, 48)
    assert 0.0 <= float(array.min()) <= float(array.max()) <= 1.0
    bundle = load_face_images_for_split(rows.head(4), feature_set="flattened_pixels")
    assert bundle.X.shape == (4, 2304)
    assert "record_id" not in bundle.feature_names
    scaler = fit_train_only_scaler(bundle.X.copy())
    transformed = transform_with_scaler(scaler, bundle.X.copy())
    assert transformed.shape == bundle.X.shape
    bad = paths.get_repository_root() / "generated" / "temporary" / "phase3i_unit" / "bad.png"
    Image.new("L", (47, 48), color=1).save(bad)
    with pytest.raises(ValueError, match="48x48"):
        load_48x48_grayscale(bad)


def test_estimators_evaluation_selection_and_governance(face_baseline_fixture):
    config = json.loads(face_baseline_fixture["config"].read_text(encoding="utf-8"))
    specs = face_candidate_specs(config, candidate="random_forest", feature_set="image_statistics")
    assert len(specs) == 1
    assert specs[0].estimator_type == "random_forest"
    metrics = evaluate_face_predictions(["angry", "disgust", "happy"], ["angry", "happy", "happy"], split_name="unit")
    assert "disgust_recall" in metrics
    assert metrics["false_negatives_by_class"]["disgust"] == 1
    gap = train_validation_gap({"macro_f1": 0.9, "macro_recall": 0.8, "balanced_accuracy": 0.8}, {"macro_f1": 0.5, "macro_recall": 0.4, "balanced_accuracy": 0.4})
    assert selection_score(metrics, gap, "random_forest")


def test_runner_artifacts_model_card_no_image_data_and_inactive(face_baseline_fixture):
    result = run_face_baseline(
        config_path=face_baseline_fixture["config"],
        deduplicated_manifest_path=face_baseline_fixture["manifest"],
        split_manifest_path=face_baseline_fixture["split_manifest"],
        split_assignments_path=face_baseline_fixture["assignments"],
        quarantine_path=face_baseline_fixture["quarantine"],
        duplicate_decisions_path=face_baseline_fixture["decisions"],
        source_fingerprint_path=face_baseline_fixture["fingerprint"],
        report_dir=paths.get_repository_root() / "generated" / "reports" / "face_baseline" / "phase3i_unit",
        model_root=paths.get_repository_root() / "ml_models" / "face" / "phase3i-unit",
        candidate="random_forest",
        feature_set="image_statistics",
        max_train_records=14,
        overwrite=True,
        require_replay=False,
    )
    assert result.registered is False
    assert result.artifact_manifest["active"] is False
    card = (result.run_dir / "model_card.md").read_text(encoding="utf-8")
    assert "research prototype" in card
    assert "reviewer_independence_unverified" in card
    assert not list(result.report_dir.rglob("*.png"))
    summary = json.loads((result.report_dir / "face_baseline_summary.json").read_text(encoding="utf-8"))
    assert summary["registration_status"] == "not_registered"
    assert summary["activation_status"] == "inactive"


def test_cli_dry_run_real_locked_contract():
    cmd = [sys.executable, "scripts/train_face_baseline.py", "--dry-run"]
    result = subprocess.run(cmd, cwd=paths.get_backend_root(), check=False, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["train_count"] == 23566
    assert payload["quarantined_record_count"] == 155
    assert payload["image_hash_overlap_count"] == 0
