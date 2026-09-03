from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from app.ml.common import paths
from app.ml.remediation.face.canonical_view import (
    build_face_deduplicated_view,
    run_perceptual_duplicate_diagnostics,
)
from app.ml.remediation.face.duplicates import (
    build_face_remediation_decisions,
    classify_duplicate_group,
    duplicate_group_summary,
    load_face_duplicate_groups,
    select_deterministic_representative,
)
from app.ml.remediation.face.schemas import FaceRemediationAction
from app.ml.remediation.face.splitting import create_face_v2_split, replay_face_v2_split
from app.ml.remediation.face.validation import generate_face_readiness_report


FINGERPRINT = "a" * 64
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _write_image(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=value).save(path, format="PNG")
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_fingerprint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"combined_sha256": FINGERPRINT}) + "\n", encoding="utf-8")
    return path


def _write_policy(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    policy = {
        "policy_version": "1.0.0",
        "exact_duplicate_policy": {"definition": "same SHA-256 image_hash", "uses_image_content_beyond_hash": False},
        "same_label_duplicate_policy": {"action": "keep_one_deterministic_representative_exclude_other_copies", "record_exclusions": True},
        "cross_label_conflict_policy": {"action": "quarantine_entire_hash_group", "automatic_label_resolution": False, "majority_vote_allowed": False},
        "cross_split_duplicate_policy": {"action": "ignore_original_split_for_v2_keep_one_hash_in_one_revised_split", "original_split_is_metadata_only": True},
        "representative_selection_rule": ["prefer readable valid records", "path asc", "record_id asc"],
        "original_split_treatment": "metadata_only_not_final_leakage_safe_split",
        "perceptual_near_duplicate_policy": {"diagnostic_only": True, "automatic_exclusion": False, "threshold": 6, "method": "Pillow average hash"},
        "class_balance_safeguards": {"strategy": "deterministic_stratified_by_canonical_emotion_label", "minimum_records_per_class_per_split": 1},
        "random_seed": 43107,
        "deterministic_tie_break_rule": "path asc",
        "minimum_records_per_class_per_split": 1,
        "notes": ["unit test policy"],
    }
    path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture()
def face_fixture(tmp_path):
    repo = paths.get_repository_root()
    root = repo / "generated" / "temporary" / "phase3g_unit"
    if root.exists():
        shutil.rmtree(root)
    image_root = root / "images"
    rows = []
    duplicate_groups = []
    image_index = 0

    def add(record_id: str, label: str, split: str, hash_value: str | None = None, value: int | None = None):
        nonlocal image_index
        image_index += 1
        rel = f"generated/temporary/phase3g_unit/images/{record_id}.png"
        image_path = repo / rel
        digest = _write_image(image_path, value if value is not None else (20 + image_index * 3) % 250)
        rows.append(
            {
                "canonical_emotion_label": label,
                "color_mode": "L",
                "file_format": "png",
                "file_size_bytes": str(image_path.stat().st_size),
                "height": "8",
                "image_hash": hash_value or digest,
                "image_relative_path": rel,
                "original_label": label,
                "readable": "True",
                "record_id": record_id,
                "safe_subject_key": "",
                "source_split": split,
                "validation_warnings": "",
                "width": "8",
            }
        )

    # Four unique records per class keep the stratified 70/15/15 split feasible.
    for label in LABELS:
        for idx in range(4):
            add(f"{label}-{idx}", label, "train" if idx < 3 else "test")

    same_hash = "b" * 64
    add("same-a", "happy", "test", same_hash, 80)
    add("same-b", "happy", "train", same_hash, 81)
    duplicate_groups.append(
        {"image_hash": same_hash, "record_ids": ["same-a", "same-b"], "source_splits": ["test", "train"], "labels": ["happy"], "cross_split": True, "cross_label": False}
    )

    multi_hash = "c" * 64
    add("multi-a", "sad", "train", multi_hash, 90)
    add("multi-b", "sad", "test", multi_hash, 91)
    add("multi-c", "sad", "train", multi_hash, 92)
    duplicate_groups.append(
        {"image_hash": multi_hash, "record_ids": ["multi-a", "multi-b", "multi-c"], "source_splits": ["test", "train"], "labels": ["sad"], "cross_split": True, "cross_label": False}
    )

    conflict_hash = "d" * 64
    add("conflict-a", "angry", "train", conflict_hash, 100)
    add("conflict-b", "fear", "test", conflict_hash, 101)
    add("conflict-c", "fear", "train", conflict_hash, 102)
    duplicate_groups.append(
        {
            "image_hash": conflict_hash,
            "record_ids": ["conflict-a", "conflict-b", "conflict-c"],
            "source_splits": ["test", "train"],
            "labels": ["angry", "fear"],
            "cross_split": True,
            "cross_label": True,
        }
    )

    canonical = root / "face_canonical_manifest.csv"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with canonical.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    duplicate_manifest = root / "face_duplicate_manifest.json"
    duplicate_manifest.write_text(
        json.dumps(
            {
                "duplicate_image_hash_groups": duplicate_groups,
                "duplicate_image_hash_group_count": len(duplicate_groups),
                "cross_split_duplicate_hash_groups": duplicate_groups,
                "cross_split_duplicate_hash_group_count": len(duplicate_groups),
                "cross_label_duplicate_hash_groups": [duplicate_groups[-1]],
                "cross_label_duplicate_hash_group_count": 1,
                "near_duplicate_candidates": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fingerprint = _write_source_fingerprint(root / "source_fingerprint.json")
    policy = _write_policy(root / "policy.json")
    yield {"root": root, "canonical": canonical, "duplicate": duplicate_manifest, "fingerprint": fingerprint, "policy": policy}
    if root.exists():
        shutil.rmtree(root)


def test_duplicate_classification_policy_and_group_edge_cases(face_fixture):
    groups = load_face_duplicate_groups(face_fixture["duplicate"], face_fixture["canonical"])
    summary = duplicate_group_summary(groups)
    assert summary["duplicate_group_count"] == 3
    assert summary["groups_with_more_than_two_records"] == 2
    assert summary["both_cross_split_and_cross_label_group_count"] == 1
    assert classify_duplicate_group(groups[-1]).cross_label is True

    result = build_face_remediation_decisions(face_fixture["canonical"], face_fixture["duplicate"], face_fixture["policy"])
    decisions = result["decisions_by_id"]
    records = result["records_by_id"]
    assert select_deterministic_representative(groups[0], records) == "same-a"
    assert decisions["same-a"].action == FaceRemediationAction.KEEP
    assert decisions["same-b"].action == FaceRemediationAction.EXCLUDE_DUPLICATE
    assert {decisions[rid].action for rid in ["conflict-a", "conflict-b", "conflict-c"]} == {
        FaceRemediationAction.QUARANTINE_CROSS_LABEL
    }
    assert records["same-a"].original_split == "test"


def test_canonical_view_reporting_and_no_image_copies(face_fixture):
    out = face_fixture["root"] / "remediation"
    result = build_face_deduplicated_view(
        canonical_manifest_path=face_fixture["canonical"],
        duplicate_manifest_path=face_fixture["duplicate"],
        source_fingerprint_path=face_fixture["fingerprint"],
        policy_config_path=face_fixture["policy"],
        output_dir=out,
        overwrite=True,
    )
    retained = _read_csv(out / "face_deduplicated_manifest.csv")
    retained_ids = {row["record_id"] for row in retained}
    assert "same-b" not in retained_ids
    assert not {"conflict-a", "conflict-b", "conflict-c"} & retained_ids
    assert (out / "face_remediation_decisions.csv").exists()
    assert result["report"].retained_record_count == len(retained)
    assert "image_pixels" not in (out / "face_remediation_report.json").read_text(encoding="utf-8")
    assert not list(out.rglob("*.png"))


def test_perceptual_diagnostics_candidate_only(face_fixture):
    out = face_fixture["root"] / "remediation"
    build_face_deduplicated_view(
        canonical_manifest_path=face_fixture["canonical"],
        duplicate_manifest_path=face_fixture["duplicate"],
        source_fingerprint_path=face_fixture["fingerprint"],
        policy_config_path=face_fixture["policy"],
        output_dir=out,
        overwrite=True,
    )
    payload = run_perceptual_duplicate_diagnostics(
        deduplicated_manifest_path=out / "face_deduplicated_manifest.csv",
        output_path=out / "face_perceptual_duplicate_candidates.json",
        limit=8,
        threshold=64,
        overwrite=True,
    )
    assert payload["candidate_count"] > 0
    assert all(candidate["automatic_exclusion"] is False for candidate in payload["candidates"])
    assert "recognition" not in payload["method"].lower()


def test_revised_split_validation_replay_and_changed_seed(face_fixture):
    rem = face_fixture["root"] / "remediation"
    split = paths.get_repository_root() / "generated" / "manifests" / "splits" / "face" / "phase3g_unit"
    if split.exists():
        shutil.rmtree(split)
    build_face_deduplicated_view(
        canonical_manifest_path=face_fixture["canonical"],
        duplicate_manifest_path=face_fixture["duplicate"],
        source_fingerprint_path=face_fixture["fingerprint"],
        policy_config_path=face_fixture["policy"],
        output_dir=rem,
        overwrite=True,
    )
    result = create_face_v2_split(
        deduplicated_manifest_path=rem / "face_deduplicated_manifest.csv",
        remediation_report_path=rem / "face_remediation_report.json",
        source_fingerprint_path=face_fixture["fingerprint"],
        policy_config_path=face_fixture["policy"],
        output_dir=split,
        seed=17,
        overwrite=True,
    )
    manifest = result["manifest"]
    assert set(manifest.label_distributions) == {"train", "validation", "test"}
    assert manifest.image_hash_overlap_count == 0
    assert manifest.duplicate_overlap_count == 0
    assert replay_face_v2_split(
        manifest_path=split / "face_split_manifest.json",
        deduplicated_manifest_path=rem / "face_deduplicated_manifest.csv",
        remediation_report_path=rem / "face_remediation_report.json",
        source_fingerprint_path=face_fixture["fingerprint"],
        policy_config_path=face_fixture["policy"],
    )
    changed = create_face_v2_split(
        deduplicated_manifest_path=rem / "face_deduplicated_manifest.csv",
        remediation_report_path=rem / "face_remediation_report.json",
        source_fingerprint_path=face_fixture["fingerprint"],
        policy_config_path=face_fixture["policy"],
        output_dir=split,
        seed=18,
        validate_only=True,
    )["manifest"]
    assert changed.train_ids != manifest.train_ids


def test_failures_readiness_and_cli(face_fixture):
    rem = face_fixture["root"] / "remediation"
    split = paths.get_repository_root() / "generated" / "manifests" / "splits" / "face" / "phase3g_unit_cli"
    readiness = paths.get_repository_root() / "generated" / "reports" / "phase3g_unit_face_readiness"
    for path in (split, readiness):
        if path.exists():
            shutil.rmtree(path)

    bad_fingerprint = _write_source_fingerprint(face_fixture["root"] / "bad_fingerprint.json")
    bad_fingerprint.write_text(json.dumps({"combined_sha256": "e" * 64}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        create_face_v2_split(
            deduplicated_manifest_path=face_fixture["canonical"],
            remediation_report_path=bad_fingerprint,
            source_fingerprint_path=face_fixture["fingerprint"],
            policy_config_path=face_fixture["policy"],
            output_dir=split,
            validate_only=True,
        )

    cmd = [
        sys.executable,
        "scripts/remediate_face_duplicates.py",
        "--canonical-manifest",
        str(face_fixture["canonical"]),
        "--duplicate-manifest",
        str(face_fixture["duplicate"]),
        "--source-fingerprint",
        str(face_fixture["fingerprint"]),
        "--policy-config",
        str(face_fixture["policy"]),
        "--output-dir",
        str(rem),
        "--run-perceptual-diagnostics",
        "--perceptual-limit",
        "8",
        "--overwrite",
    ]
    assert subprocess.run(cmd, cwd=paths.get_backend_root(), check=False).returncode == 0

    split_cmd = [
        sys.executable,
        "scripts/create_face_v2_splits.py",
        "--deduplicated-manifest",
        str(rem / "face_deduplicated_manifest.csv"),
        "--remediation-report",
        str(rem / "face_remediation_report.json"),
        "--source-fingerprint",
        str(face_fixture["fingerprint"]),
        "--policy-config",
        str(face_fixture["policy"]),
        "--output-dir",
        str(split),
        "--seed",
        "19",
        "--replay",
        "--overwrite",
    ]
    assert subprocess.run(split_cmd, cwd=paths.get_backend_root(), check=False).returncode == 0
    report = generate_face_readiness_report(
        canonical_manifest_path=face_fixture["canonical"],
        deduplicated_manifest_path=rem / "face_deduplicated_manifest.csv",
        remediation_report_path=rem / "face_remediation_report.json",
        source_fingerprint_path=face_fixture["fingerprint"],
        policy_config_path=face_fixture["policy"],
        split_manifest_path=split / "face_split_manifest.json",
        split_assignments_path=split / "face_split_assignments.csv",
        output_dir=readiness,
        overwrite=True,
    )
    assert report["readiness_classification"] == "ready_with_restrictions"
    assert report["exact_hash_overlap_count"] == 0
    assert "model" not in "".join(path.name for path in rem.rglob("*")).lower()

