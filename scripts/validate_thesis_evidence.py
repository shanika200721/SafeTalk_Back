"""Validate thesis evidence package without changing application behavior."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "ml-research" / "thesis_evidence"
OUTPUT_DIR = REPO_ROOT / "generated" / "reports" / "thesis_evidence_validation"


EXPECTED_EVIDENCE_FILES = [
    "ml-research/thesis_evidence/final_implementation_progress_report.md",
    "ml-research/thesis_evidence/system_module_status.csv",
    "ml-research/thesis_evidence/frontend_implementation_report.md",
    "ml-research/thesis_evidence/frontend_feature_inventory.csv",
    "ml-research/thesis_evidence/backend_implementation_report.md",
    "ml-research/thesis_evidence/backend_endpoint_inventory.csv",
    "ml-research/thesis_evidence/database_implementation_report.md",
    "ml-research/thesis_evidence/database_table_inventory.csv",
    "ml-research/thesis_evidence/dataset_preprocessing_summary.md",
    "ml-research/thesis_evidence/dataset_summary_table.csv",
    "ml-research/thesis_evidence/model_development_summary.md",
    "ml-research/thesis_evidence/model_results_table.csv",
    "ml-research/thesis_evidence/testing_and_verification_report.md",
    "ml-research/thesis_evidence/test_inventory.csv",
    "ml-research/thesis_evidence/system_architecture_report.md",
    "ml-research/thesis_evidence/research_objective_achievement.md",
    "ml-research/thesis_evidence/thesis_claims_register.csv",
    "ml-research/thesis_evidence/thesis_chapter_evidence_map.md",
    "ml-research/thesis_evidence/post_thesis_production_roadmap.md",
    "backend/migration_report.json",
    "backend/test_results_postgres.txt",
    "generated/reports/model_governance/v1/unimodal_model_comparison.csv",
    "generated/reports/model_governance/v1/research_model_inventory.csv",
    "generated/reports/model_governance/v1/deployment_readiness_matrix.csv",
    "generated/reports/speech_domain_shift/v1/speech_loco_results.csv",
    "generated/reports/face_baseline/v1/face_metrics_test.json",
    "generated/preprocessing/mood/v1/mood_readiness_report.json",
    "generated/preprocessing/behavioral/v1/behavioral_readiness_report.json",
]


SELECTED_MANIFESTS = [
    "ml_models/profile/profile-depression-random-forest/1.0.0/profile-minimal_contextual-66e36ed73f40/artifact_manifest.json",
    "ml_models/text/text-classification-logistic-regression/1.0.0/text-e8d74030dfff/artifact_manifest.json",
    "ml_models/speech/speech-emotion-random-forest/1.0.0/speech-8b042739d5f6/artifact_manifest.json",
    "ml_models/face/face-emotion-random-forest/1.0.0/face-image_statistics-b9d5c76172fc/artifact_manifest.json",
]


PROHIBITED_CLAIM_PATTERNS = [
    r"\bsystem is clinically validated\b",
    r"\bclinically validated system\b",
    r"\bsystem is production deployed\b",
    r"\bproduction deployed system\b",
    r"\bfusion predicts suicide risk accurately\b",
    r"\bface model detects depression\b",
    r"\bspeech model detects suicide risk\b",
    r"\bmodels autonomously decide counselor alerts\b",
    r"\bmodels autonomously trigger counselor alerts\b",
]


ABSOLUTE_PATH_RE = re.compile(r"([A-Za-z]:\\|/home/|/Users/|\\\\)")
RAW_MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".wav", ".mp3", ".mp4", ".flac"}


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def read_json(relative_path: str) -> Any:
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (REPO_ROOT / relative_path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_files() -> list[Path]:
    if not EVIDENCE_DIR.exists():
        return []
    return sorted(path for path in EVIDENCE_DIR.rglob("*") if path.is_file())


def check_missing_evidence() -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for item in EXPECTED_EVIDENCE_FILES:
        if not (REPO_ROOT / item).exists():
            missing.append({"path": item, "reason": "expected evidence file is missing"})
    return missing


def check_metrics() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    table_path = "ml-research/thesis_evidence/model_results_table.csv"
    if not (REPO_ROOT / table_path).exists():
        return [{"status": "failed", "reason": "model_results_table.csv missing"}]

    rows = {row["modality"]: row for row in read_csv(table_path)}
    comparison = {row["modality"]: row for row in read_csv("generated/reports/model_governance/v1/unimodal_model_comparison.csv")}

    for modality in ("Profile", "Text", "Speech", "Face"):
        row = rows.get(modality)
        source = comparison.get(modality.lower())
        if not row or not source:
            findings.append({"status": "failed", "modality": modality, "reason": "missing row in result or source table"})
            continue
        for field in ("test_count", "test_macro_f1", "test_balanced_accuracy"):
            expected = str(source.get(field, "")).strip()
            actual = str(row.get(field, "")).strip()
            if expected != actual:
                findings.append(
                    {
                        "status": "failed",
                        "modality": modality,
                        "field": field,
                        "expected": expected,
                        "actual": actual,
                    }
                )
    return findings


def check_artifact_hashes() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    checked = 0
    for manifest_rel in SELECTED_MANIFESTS:
        manifest_path = REPO_ROOT / manifest_rel
        if not manifest_path.exists():
            findings.append({"status": "failed", "manifest": manifest_rel, "reason": "manifest missing"})
            continue
        manifest = read_json(manifest_rel)
        for file_rel, expected_hash in manifest.get("file_hashes", {}).items():
            artifact_path = REPO_ROOT / file_rel
            if not artifact_path.exists():
                findings.append({"status": "failed", "manifest": manifest_rel, "artifact": file_rel, "reason": "artifact missing"})
                continue
            actual_hash = sha256_file(artifact_path)
            checked += 1
            if actual_hash != expected_hash:
                findings.append(
                    {
                        "status": "failed",
                        "manifest": manifest_rel,
                        "artifact": file_rel,
                        "expected_hash": expected_hash,
                        "actual_hash": actual_hash,
                    }
                )
    if not findings:
        findings.append({"status": "passed", "checked_files": checked})
    return findings


def check_claims() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in evidence_files():
        if path.name == "thesis_claims_register.csv":
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for pattern in PROHIBITED_CLAIM_PATTERNS:
            for match in re.finditer(pattern, text):
                window = text[max(0, match.start() - 40): match.end() + 40]
                if re.search(r"\b(no|not|never|prohibit|prohibited|must not|do not|without)\b", window):
                    continue
                findings.append({"file": rel(path), "pattern": pattern, "context": " ".join(window.split())})
    return findings


def check_absolute_paths() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in evidence_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if ABSOLUTE_PATH_RE.search(line):
                findings.append({"file": rel(path), "line": str(line_no), "context": line[:240]})
    return findings


def check_raw_data_leakage() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in evidence_files():
        if path.suffix.lower() in RAW_MEDIA_SUFFIXES:
            findings.append({"file": rel(path), "reason": "raw media-like file included in evidence package"})
    return findings


def check_contradictions() -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    scannable_files = [path for path in evidence_files() if path.name != "thesis_claims_register.csv"]
    all_text = "\n".join(path.read_text(encoding="utf-8", errors="replace").lower() for path in scannable_files)
    if "frontend build verification was attempted" in all_text and "frontend build passed" in all_text:
        findings.append({"topic": "frontend build", "reason": "both timeout/inconclusive and passed wording found"})
    if "not production deployed" in all_text and "system is production deployed" in all_text:
        findings.append({"topic": "deployment", "reason": "both non-deployed and deployed wording found"})
    if "fusion was deferred" in all_text and "fusion predicts suicide risk accurately" in all_text:
        findings.append({"topic": "fusion", "reason": "both deferred fusion and accurate fusion prediction wording found"})
    return findings


def build_artifact_inventory() -> list[dict[str, Any]]:
    inventory = []
    for path in evidence_files():
        inventory.append(
            {
                "path": rel(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    missing = check_missing_evidence()
    metric_findings = check_metrics()
    hash_findings = check_artifact_hashes()
    unsupported_claims = check_claims()
    absolute_paths = check_absolute_paths()
    raw_leakage = check_raw_data_leakage()
    contradictions = check_contradictions()
    inventory = build_artifact_inventory()

    validation = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_file_count": len(inventory),
        "missing_evidence_count": len(missing),
        "metric_mismatch_count": len([item for item in metric_findings if item.get("status") == "failed"]),
        "artifact_hash_status": hash_findings,
        "unsupported_claim_count": len(unsupported_claims),
        "absolute_path_leakage_count": len(absolute_paths),
        "raw_data_leakage_count": len(raw_leakage),
        "contradiction_count": len(contradictions),
        "status": "passed"
        if not (missing or unsupported_claims or absolute_paths or raw_leakage or contradictions or any(item.get("status") == "failed" for item in metric_findings + hash_findings))
        else "needs_review",
    }

    write_json(OUTPUT_DIR / "thesis_evidence_validation.json", validation)
    write_json(OUTPUT_DIR / "unsupported_claims.json", unsupported_claims)
    write_json(OUTPUT_DIR / "contradictions.json", contradictions)
    write_json(OUTPUT_DIR / "missing_evidence.json", missing)
    write_json(OUTPUT_DIR / "evidence_artifact_inventory.json", inventory)

    report_lines = [
        "# Thesis Evidence Validation",
        "",
        f"Generated at: {validation['generated_at']}",
        "",
        f"Status: {validation['status']}",
        "",
        f"- Evidence files inventoried: {validation['evidence_file_count']}",
        f"- Missing evidence: {validation['missing_evidence_count']}",
        f"- Metric mismatches: {validation['metric_mismatch_count']}",
        f"- Unsupported claims: {validation['unsupported_claim_count']}",
        f"- Absolute path leakage: {validation['absolute_path_leakage_count']}",
        f"- Raw-data leakage: {validation['raw_data_leakage_count']}",
        f"- Contradictions: {validation['contradiction_count']}",
        "",
        "## Artifact Hash Verification",
        "",
        "```json",
        json.dumps(hash_findings, indent=2),
        "```",
    ]
    (OUTPUT_DIR / "thesis_evidence_validation.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return 0 if validation["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
