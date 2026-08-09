from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.review.face.canonical_update import create_reviewed_canonical_view
from app.ml.review.face.reconciliation import create_reconciliation_manifest, reconcile_all_reviews
from app.ml.review.face.reporting import write_review_audit_artifacts
from app.ml.review.face.splitting import create_face_reviewed_split
from app.ml.common import paths


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_source_fingerprint(review_package: str | Path) -> str:
    package_path = _resolve_repo_path(review_package)
    if package_path.is_dir():
        package_path = package_path / "face_review_items.json"
    with package_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["source_fingerprint"]


def _total_items(review_package: str | Path) -> int:
    package_path = _resolve_repo_path(review_package)
    if package_path.is_dir():
        package_path = package_path / "face_review_items.json"
    with package_path.open("r", encoding="utf-8") as handle:
        return len(json.load(handle).get("review_items", []))


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile Phase 3H face reviews.")
    parser.add_argument("--review-package", required=True)
    parser.add_argument("--decisions-dir", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--apply-reviewed-view", action="store_true")
    parser.add_argument("--create-reviewed-split", action="store_true")
    parser.add_argument("--seed", type=int, default=43107)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path")
    parser.add_argument("--phase3g-manifest")
    parser.add_argument("--phase3g-decisions")
    parser.add_argument("--phase3g-quarantine")
    parser.add_argument("--canonical-manifest")
    parser.add_argument("--source-fingerprint-file")
    args = parser.parse_args()
    reconciled = reconcile_all_reviews(
        review_package=args.review_package,
        decisions_dir=args.decisions_dir,
        policy_config_path=args.policy_config,
    )
    if args.validate_only:
        print(f"reconciled review items: {len(reconciled)}")
        return 0
    source_fingerprint = _load_source_fingerprint(args.review_package)
    manifest = create_reconciliation_manifest(
        reconciled_decisions=reconciled,
        output_dir=args.output_dir,
        source_fingerprint=source_fingerprint,
        overwrite=args.overwrite,
    )
    write_review_audit_artifacts(
        output_dir=args.output_dir,
        source_fingerprint=source_fingerprint,
        total_review_items=_total_items(args.review_package),
        reconciled_decisions=reconciled,
        overwrite=args.overwrite,
    )
    reviewed_report = None
    if args.apply_reviewed_view:
        required = [args.phase3g_manifest, args.phase3g_decisions, args.phase3g_quarantine, args.canonical_manifest]
        if any(value is None for value in required):
            raise ValueError("reviewed view requires phase3g and canonical manifest arguments")
        reviewed = create_reviewed_canonical_view(
            phase3g_manifest_path=args.phase3g_manifest,
            phase3g_decisions_path=args.phase3g_decisions,
            phase3g_quarantine_path=args.phase3g_quarantine,
            reconciliation_manifest_path=Path(args.output_dir) / "face_reconciliation_manifest.json",
            canonical_manifest_path=args.canonical_manifest,
            source_fingerprint=source_fingerprint,
            output_dir="generated/remediation/face/v2_reviewed",
            overwrite=args.overwrite,
        )
        reviewed_report = reviewed["outputs"]["report_json"]
    if args.create_reviewed_split and reviewed_report:
        if args.source_fingerprint_file is None:
            raise ValueError("reviewed split requires --source-fingerprint-file")
        create_face_reviewed_split(
            reviewed_manifest_path="generated/remediation/face/v2_reviewed/face_reviewed_deduplicated_manifest.csv",
            reviewed_view_report_path=reviewed_report,
            source_fingerprint_path=args.source_fingerprint_file,
            policy_config_path=args.policy_config,
            output_dir="generated/manifests/splits/face/v3_reviewed",
            seed=args.seed,
            overwrite=args.overwrite,
        )
    if args.report_path:
        Path(args.report_path).write_text(json.dumps(manifest["payload"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"reconciled review items: {len(reconciled)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
