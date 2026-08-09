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
from app.ml.review.face.canonical_update import create_reviewed_canonical_view
from app.ml.review.face.decisions import load_reviewer_decisions, save_validated_decisions
from app.ml.review.face.package import build_review_items, create_review_package
from app.ml.review.face.readiness import classify_review_readiness, validate_phase3h_review_readiness
from app.ml.review.face.reconciliation import create_reconciliation_manifest, reconcile_all_reviews, reconcile_review_item
from app.ml.review.face.reporting import write_review_audit_artifacts
from app.ml.review.face.splitting import create_face_reviewed_split

FINGERPRINT = "a" * 64
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_image(path: Path, value: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=value).save(path, format="PNG")
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture()
def review_fixture():
    repo = paths.get_repository_root()
    root = repo / "generated" / "review" / "face" / "phase3h_unit"
    extra_outputs = [
        repo / "generated" / "remediation" / "face" / "phase3h_unit_view",
        repo / "generated" / "manifests" / "splits" / "face" / "phase3h_unit",
        repo / "generated" / "reports" / "phase3h_unit_face_review",
    ]
    for path in [root, *extra_outputs]:
        if path.exists():
            shutil.rmtree(path)
    rows = []

    def add(record_id: str, label: str, split: str, image_hash: str | None = None, value: int = 50):
        rel = f"generated/review/face/phase3h_unit/images/{record_id}.png"
        digest = _write_image(repo / rel, value)
        rows.append(
            {
                "record_id": record_id,
                "source_split": split,
                "original_label": label,
                "canonical_emotion_label": label,
                "image_relative_path": rel,
                "image_hash": image_hash or digest,
                "readable": "True",
                "duplicate_group_id": "",
                "remediation_action": "keep",
                "remediation_policy_version": "1.0.0",
            }
        )

    for idx, label in enumerate(LABELS):
        add(f"{label}-base", label, "train", value=20 + idx)
    conflict_hash = "b" * 64
    add("conflict-a", "angry", "train", conflict_hash, 100)
    add("conflict-b", "fear", "test", conflict_hash, 101)
    add("near-a", "happy", "train", value=120)
    add("near-b", "happy", "test", value=121)

    canonical = root / "canonical.csv"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    with canonical.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    phase3g_manifest = root / "phase3g_manifest.csv"
    retained = [row for row in rows if row["record_id"] not in {"conflict-a", "conflict-b"}]
    with phase3g_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(retained)
    phase3g_decisions = root / "phase3g_decisions.csv"
    with phase3g_decisions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "action", "representative_id", "group_id", "reason", "policy_version"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "record_id": row["record_id"],
                    "action": "quarantine_cross_label" if row["record_id"].startswith("conflict") else "keep",
                    "representative_id": row["record_id"],
                    "group_id": "face-dup-conflict" if row["record_id"].startswith("conflict") else "",
                    "reason": "unit",
                    "policy_version": "1.0.0",
                }
            )

    quarantine = _write_json(
        root / "quarantine.json",
        {
            "count": 2,
            "quarantined_records": [
                {"record_id": "conflict-a", "group_id": "face-dup-conflict", "canonical_emotion_label": "angry", "reason": "cross"},
                {"record_id": "conflict-b", "group_id": "face-dup-conflict", "canonical_emotion_label": "fear", "reason": "cross"},
            ],
        },
    )
    perceptual = _write_json(
        root / "perceptual.json",
        {"candidate_count": 1, "candidates": [{"record_ids": ["near-a", "near-b"], "hamming_distance": 4, "automatic_exclusion": False}]},
    )
    policy = _write_json(
        root / "policy.json",
        {
            "policy_version": "1.0.0",
            "minimum_reviewer_count": 2,
            "double_review_required": True,
            "conflict_resolution_rule": "unanimous_consensus_required_no_majority_vote",
            "allowed_decisions": [
                "confirm_exact_duplicate",
                "confirm_near_duplicate",
                "not_duplicate",
                "label_conflict_unresolved",
                "source_label_likely_incorrect",
                "ambiguous",
                "corrupted",
                "retain_quarantine",
                "recommend_restore_same_label_representative",
                "requires_additional_review",
            ],
            "reason_codes": [
                "visually_identical",
                "near_identical",
                "not_duplicate",
                "label_conflict",
                "likely_source_label_error",
                "ambiguous_visual_evidence",
                "corrupted_or_unreadable",
                "privacy_or_policy_limit",
                "insufficient_consensus",
                "requires_additional_review",
                "synthetic_smoke_test",
            ],
            "restore_policy": {"allowed_for_cross_label_conflicts": True, "reviewer_consensus_required": True},
            "notes": ["unit"],
        },
    )
    fingerprint = _write_json(root / "fingerprint.json", {"combined_sha256": FINGERPRINT})
    yield {
        "root": root,
        "canonical": canonical,
        "phase3g_manifest": phase3g_manifest,
        "phase3g_decisions": phase3g_decisions,
        "quarantine": quarantine,
        "perceptual": perceptual,
        "policy": policy,
        "fingerprint": fingerprint,
    }
    for path in [root, *extra_outputs]:
        if path.exists():
            shutil.rmtree(path)


def _package(review_fixture, output_name="pkg") -> Path:
    out = review_fixture["root"] / output_name
    create_review_package(
        cross_label_quarantine_path=review_fixture["quarantine"],
        perceptual_candidates_path=review_fixture["perceptual"],
        canonical_manifest_path=review_fixture["canonical"],
        source_fingerprint=FINGERPRINT,
        policy_config_path=review_fixture["policy"],
        output_dir=out,
        include_html_index=True,
        overwrite=True,
    )
    return out


def _decision_file(path: Path, item_ids: list[str], *, decision: str = "retain_quarantine", reviewer: str = "reviewer_a") -> Path:
    rows = [
        {
            "review_item_id": item_id,
            "reviewer_alias": reviewer,
            "decision": decision,
            "reason_code": "synthetic_smoke_test",
            "confidence": "high",
            "notes": "synthetic",
            "reviewed_at": "2026-07-15T00:00:00+00:00",
        }
        for item_id in item_ids
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_package_creation_safe_references_and_deterministic_ids(review_fixture):
    items = build_review_items(
        cross_label_quarantine_path=review_fixture["quarantine"],
        perceptual_candidates_path=review_fixture["perceptual"],
        canonical_manifest_path=review_fixture["canonical"],
        policy_config_path=review_fixture["policy"],
    )
    replay = build_review_items(
        cross_label_quarantine_path=review_fixture["quarantine"],
        perceptual_candidates_path=review_fixture["perceptual"],
        canonical_manifest_path=review_fixture["canonical"],
        policy_config_path=review_fixture["policy"],
    )
    assert [item.review_item_id for item in items] == [item.review_item_id for item in replay]
    assert sum(item.item_type == "cross_label_conflict" for item in items) == 1
    assert sum(item.item_type == "perceptual_duplicate_candidate" for item in items) == 1
    assert all(not Path(ref).is_absolute() for item in items for ref in item.safe_image_references.values())
    pkg = _package(review_fixture)
    text = (pkg / "face_review_items.json").read_text(encoding="utf-8")
    assert "base64" not in text.lower()
    assert not list(pkg.rglob("*.png"))


def test_decision_validation_errors_and_audit_preservation(review_fixture):
    pkg = _package(review_fixture)
    item_ids = [item["review_item_id"] for item in json.loads((pkg / "face_review_items.json").read_text(encoding="utf-8"))["review_items"]]
    valid_file = _decision_file(review_fixture["root"] / "valid.csv", item_ids)
    decisions = load_reviewer_decisions(review_package=pkg, decision_file=valid_file, policy_config_path=review_fixture["policy"])
    assert len(decisions) == 2
    saved = save_validated_decisions(decisions=decisions, output_dir=review_fixture["root"] / "audit", decision_file=valid_file, overwrite=True)
    assert saved["outputs"]["validated_decisions"].exists()

    with pytest.raises(ValueError, match="unknown review item"):
        load_reviewer_decisions(review_package=pkg, decision_file=_decision_file(review_fixture["root"] / "unknown.csv", ["missing"]), policy_config_path=review_fixture["policy"])
    bad_reason = _decision_file(review_fixture["root"] / "bad_reason.csv", [item_ids[0]])
    rows = _read_csv(bad_reason)
    rows[0]["reason_code"] = "free_text"
    with bad_reason.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="invalid reason"):
        load_reviewer_decisions(review_package=pkg, decision_file=bad_reason, policy_config_path=review_fixture["policy"])
    email = _decision_file(review_fixture["root"] / "email.csv", [item_ids[0]], reviewer="person@example.com")
    with pytest.raises(ValueError, match="email"):
        load_reviewer_decisions(review_package=pkg, decision_file=email, policy_config_path=review_fixture["policy"])
    invalid_time = _decision_file(review_fixture["root"] / "bad_time.csv", [item_ids[0]])
    rows = _read_csv(invalid_time)
    rows[0]["reviewed_at"] = "2026-07-15T00:00:00"
    with invalid_time.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="timezone"):
        load_reviewer_decisions(review_package=pkg, decision_file=invalid_time, policy_config_path=review_fixture["policy"])


def test_reconciliation_consensus_disagreement_and_no_majority_relabel(review_fixture):
    pkg = _package(review_fixture)
    package_payload = json.loads((pkg / "face_review_items.json").read_text(encoding="utf-8"))
    cross = next(item for item in package_payload["review_items"] if item["item_type"] == "cross_label_conflict")
    policy = json.loads(review_fixture["policy"].read_text(encoding="utf-8"))
    one_decision = load_reviewer_decisions(
        review_package=pkg,
        decision_file=_decision_file(review_fixture["root"] / "one.csv", [cross["review_item_id"]]),
        policy_config_path=review_fixture["policy"],
    )
    assert reconcile_review_item(cross, one_decision, policy).final_action == "additional_review"
    a = _decision_file(review_fixture["root"] / "a.csv", [cross["review_item_id"]], decision="source_label_likely_incorrect", reviewer="reviewer_a")
    b = _decision_file(review_fixture["root"] / "b.csv", [cross["review_item_id"]], decision="retain_quarantine", reviewer="reviewer_b")
    decisions = load_reviewer_decisions(review_package=pkg, decision_file=a, policy_config_path=review_fixture["policy"]) + load_reviewer_decisions(
        review_package=pkg, decision_file=b, policy_config_path=review_fixture["policy"]
    )
    result = reconcile_review_item(cross, decisions, policy)
    assert result.final_status == "disagreement"
    assert result.final_action == "additional_review"


def test_reviewed_view_split_and_readiness(review_fixture):
    pkg = _package(review_fixture)
    item_ids = [item["review_item_id"] for item in json.loads((pkg / "face_review_items.json").read_text(encoding="utf-8"))["review_items"]]
    decision_file = _decision_file(review_fixture["root"] / "all.csv", item_ids, decision="retain_quarantine", reviewer="reviewer_a")
    decision_file_b = _decision_file(review_fixture["root"] / "all_b.csv", item_ids, decision="retain_quarantine", reviewer="reviewer_b")
    decisions = load_reviewer_decisions(review_package=pkg, decision_file=decision_file, policy_config_path=review_fixture["policy"]) + load_reviewer_decisions(
        review_package=pkg, decision_file=decision_file_b, policy_config_path=review_fixture["policy"]
    )
    audit = review_fixture["root"] / "audit"
    save_validated_decisions(decisions=decisions, output_dir=audit, decision_file=decision_file, overwrite=True)
    reconciled = reconcile_all_reviews(review_package=pkg, decisions_dir=audit, policy_config_path=review_fixture["policy"])
    manifest = create_reconciliation_manifest(reconciled_decisions=reconciled, output_dir=audit, source_fingerprint=FINGERPRINT, overwrite=True)
    write_review_audit_artifacts(output_dir=audit, source_fingerprint=FINGERPRINT, total_review_items=len(item_ids), reconciled_decisions=reconciled, overwrite=True)
    reviewed = create_reviewed_canonical_view(
        phase3g_manifest_path=review_fixture["phase3g_manifest"],
        phase3g_decisions_path=review_fixture["phase3g_decisions"],
        phase3g_quarantine_path=review_fixture["quarantine"],
        reconciliation_manifest_path=manifest["outputs"]["manifest"],
        canonical_manifest_path=review_fixture["canonical"],
        source_fingerprint=FINGERPRINT,
        output_dir=paths.get_repository_root() / "generated" / "remediation" / "face" / "phase3h_unit_view",
        overwrite=True,
    )
    assert reviewed["report"]["restored_record_count"] == 0
    assert reviewed["report"]["label_change_applied"] is False
    split = create_face_reviewed_split(
        reviewed_manifest_path=review_fixture["phase3g_manifest"],
        reviewed_view_report_path=reviewed["outputs"]["report_json"],
        source_fingerprint_path=review_fixture["fingerprint"],
        policy_config_path=review_fixture["policy"],
        output_dir=paths.get_repository_root() / "generated" / "manifests" / "splits" / "face" / "phase3h_unit",
        overwrite=True,
    )
    assert split["report"]["reviewed_split_generated"] is False
    readiness = validate_phase3h_review_readiness(
        review_package_path=pkg / "face_review_items.json",
        phase3g_quarantine_path=review_fixture["quarantine"],
        source_fingerprint_path=review_fixture["fingerprint"],
        output_dir=paths.get_repository_root() / "generated" / "reports" / "phase3h_unit_face_review",
        reconciliation_manifest_path=manifest["outputs"]["manifest"],
        reviewed_view_report_path=reviewed["outputs"]["report_json"],
        overwrite=True,
    )
    assert readiness["readiness_classification"] == "review_complete_with_unresolved_items"
    assert classify_review_readiness(total_items=2, reviewed_items=0, unresolved_items=2, integrity_failures=[]) == "review_not_started"
    assert classify_review_readiness(total_items=2, reviewed_items=1, unresolved_items=1, integrity_failures=[]) == "review_in_progress"
    assert classify_review_readiness(total_items=2, reviewed_items=2, unresolved_items=0, integrity_failures=[]) == "review_complete_ready_with_restrictions"
    assert classify_review_readiness(total_items=2, reviewed_items=2, unresolved_items=0, integrity_failures=["bad"]) == "review_failed_integrity"


def test_cli_validate_import_reconcile_and_overwrite_refusal(review_fixture):
    pkg = review_fixture["root"] / "cli_pkg"
    cmd = [
        sys.executable,
        "scripts/create_face_review_package.py",
        "--cross-label-quarantine",
        str(review_fixture["quarantine"]),
        "--perceptual-candidates",
        str(review_fixture["perceptual"]),
        "--canonical-manifest",
        str(review_fixture["canonical"]),
        "--source-fingerprint",
        FINGERPRINT,
        "--policy-config",
        str(review_fixture["policy"]),
        "--output-dir",
        str(pkg),
        "--include-html-index",
        "--overwrite",
    ]
    assert subprocess.run(cmd, cwd=paths.get_backend_root(), check=False).returncode == 0
    assert subprocess.run(cmd[:-1], cwd=paths.get_backend_root(), check=False).returncode != 0
    item_ids = [item["review_item_id"] for item in json.loads((pkg / "face_review_items.json").read_text(encoding="utf-8"))["review_items"]]
    decisions = _decision_file(review_fixture["root"] / "cli_decisions.csv", item_ids, reviewer="reviewer_a")
    import_cmd = [
        sys.executable,
        "scripts/import_face_review_decisions.py",
        "--review-package",
        str(pkg),
        "--decision-file",
        str(decisions),
        "--policy-config",
        str(review_fixture["policy"]),
        "--output-dir",
        str(review_fixture["root"] / "cli_audit"),
        "--overwrite",
    ]
    assert subprocess.run(import_cmd, cwd=paths.get_backend_root(), check=False).returncode == 0
    reconcile_cmd = [
        sys.executable,
        "scripts/reconcile_face_reviews.py",
        "--review-package",
        str(pkg),
        "--decisions-dir",
        str(review_fixture["root"] / "cli_audit"),
        "--policy-config",
        str(review_fixture["policy"]),
        "--output-dir",
        str(review_fixture["root"] / "cli_reconcile"),
        "--validate-only",
    ]
    assert subprocess.run(reconcile_cmd, cwd=paths.get_backend_root(), check=False).returncode == 0
