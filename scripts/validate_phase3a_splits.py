"""Aggregate Phase 3A split validation and readiness reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common.hashing import sha256_file
from app.ml.splitting.common import compute_split_artifact_hash
from app.ml.splitting.constants import SPLIT_DESIGN_VERSION, SPLIT_MANIFEST_VERSION
from app.ml.splitting.reporting import save_phase3a_reports


DEFAULT_SPLIT_ROOTS = {
    "profile": "../generated/manifests/splits/profile/v1",
    "text": "../generated/manifests/splits/text/v1",
    "speech": "../generated/manifests/splits/speech/v1",
}


def _resolve(value: str | None, default: str) -> Path:
    path = Path(value or default)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate aggregate Phase 3A split readiness.")
    parser.add_argument("--output-dir", default="../generated/reports/phase3a_splits")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_modality(modality: str, root: Path) -> tuple[dict, dict]:
    manifest_path = root / f"{modality}_split_manifest.json"
    assignments_path = root / f"{modality}_split_assignments.csv"
    report_path = root / f"{modality}_split_report.json"
    if not manifest_path.exists() or not assignments_path.exists() or not report_path.exists():
        return (
            {
                "modality": modality,
                "ready": False,
                "strategy": "missing",
                "train_count": 0,
                "validation_count": 0,
                "test_count": 0,
                "excluded_count": 0,
                "manifest_hash": "",
            },
            {"missing_artifacts": [str(path.name) for path in (manifest_path, assignments_path, report_path) if not path.exists()]},
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = pd.read_csv(assignments_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    overlap = 0
    for field in ["group_id", "duplicate_group_id"]:
        if field in assignments.columns:
            grouped = assignments.dropna(subset=[field]).groupby(field)["split"].nunique()
            overlap += int((grouped > 1).sum())
    blockers = []
    if overlap:
        blockers.append(f"group_or_duplicate_overlap_count={overlap}")
    manifest_hash = compute_split_artifact_hash(manifest)
    row = {
        "modality": modality,
        "ready": not blockers,
        "strategy": manifest.get("split_strategy", ""),
        "train_count": len(manifest.get("train_ids", [])),
        "validation_count": len(manifest.get("validation_ids", [])),
        "test_count": len(manifest.get("test_ids", [])),
        "excluded_count": len(manifest.get("excluded_ids", {})),
        "manifest_hash": manifest_hash,
        "label_distributions": {
            "train": manifest.get("validation_summary", {}).get("train_distribution", {}),
            "validation": manifest.get("validation_summary", {}).get("validation_distribution", {}),
            "test": manifest.get("validation_summary", {}).get("test_distribution", {}),
        },
        "preprocessing_artifact_hash": manifest.get("preprocessing_artifact_hash", ""),
        "source_fingerprint": manifest.get("source_fingerprint", ""),
    }
    details = {
        "blockers": blockers,
        "report_hash": sha256_file(report_path),
        "assignment_count": len(assignments),
        "report": report,
    }
    return row, details


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    modalities = []
    blockers = {}
    details = {}
    for modality, default_root in DEFAULT_SPLIT_ROOTS.items():
        row, modality_details = _load_modality(modality, _resolve(None, default_root))
        modalities.append(row)
        details[modality] = modality_details
        if modality_details.get("blockers") or modality_details.get("missing_artifacts"):
            blockers[modality] = modality_details.get("blockers") or modality_details.get("missing_artifacts")
    payload = {
        "split_design_version": SPLIT_DESIGN_VERSION,
        "split_manifest_version": SPLIT_MANIFEST_VERSION,
        "modalities": modalities,
        "details": details,
        "blockers": blockers,
        "blocked_modalities_not_split_ready": ["dass21", "mood", "face", "behavioral", "fusion"],
        "next_actions": [
            "Use profile, text, and speech manifests for common training-framework development only.",
            "Do not train blocked modalities until Phase 2 blockers are resolved.",
            "Consider leave-one-corpus-out speech evaluation during training design.",
        ],
        "completion_decision": "complete_with_restrictions" if not blockers else "blocked_pending_missing_or_invalid_split_artifacts",
    }
    save_phase3a_reports(payload, _resolve(args.output_dir, "../generated/reports/phase3a_splits"), overwrite=args.overwrite)
    print(f"Phase 3A split validation: {payload['completion_decision']} modalities={len(modalities)} blockers={len(blockers)}")
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
