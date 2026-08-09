"""Run Phase 3G face duplicate remediation without training models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common import paths
from app.ml.remediation.face.canonical_view import (
    build_face_deduplicated_view,
    run_perceptual_duplicate_diagnostics,
)


def _resolve(value: str | None, default: str | None = None) -> Path | None:
    raw = value or default
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remediate exact duplicate face records into a manifest-only view.")
    parser.add_argument("--canonical-manifest", default="../generated/preprocessing/face/v1/face_canonical_manifest.csv")
    parser.add_argument("--duplicate-manifest", default="../generated/preprocessing/face/v1/face_duplicate_manifest.json")
    parser.add_argument("--cross-split-report", default="../generated/preprocessing/face/v1/face_cross_split_overlap.json")
    parser.add_argument("--cross-label-report", default="../generated/preprocessing/face/v1/face_cross_label_conflicts.json")
    parser.add_argument("--source-fingerprint", default="../generated/manifests/fingerprints/face/facial-emotion-v1.json")
    parser.add_argument("--policy-config", default="../ml-research/configs/face.duplicate_policy.v1.json")
    parser.add_argument("--output-dir", default="../generated/remediation/face/v1")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--run-perceptual-diagnostics", action="store_true")
    parser.add_argument("--perceptual-limit", type=int, default=1000)
    parser.add_argument("--perceptual-threshold", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = build_face_deduplicated_view(
            canonical_manifest_path=_resolve(args.canonical_manifest),
            duplicate_manifest_path=_resolve(args.duplicate_manifest),
            source_fingerprint_path=_resolve(args.source_fingerprint),
            policy_config_path=_resolve(args.policy_config),
            output_dir=_resolve(args.output_dir),
            validate_only=args.validate_only,
            overwrite=args.overwrite,
        )
        report = result["report"].to_safe_dict()
        if args.run_perceptual_diagnostics and not args.validate_only:
            output_dir = _resolve(args.output_dir)
            candidates = run_perceptual_duplicate_diagnostics(
                deduplicated_manifest_path=output_dir / "face_deduplicated_manifest.csv",
                output_path=output_dir / "face_perceptual_duplicate_candidates.json",
                limit=args.perceptual_limit,
                threshold=args.perceptual_threshold,
                overwrite=args.overwrite,
            )
            report["perceptual_candidate_count"] = candidates["candidate_count"]
        if args.report_path:
            report_path = _resolve(args.report_path)
            if report_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "face remediation: "
            f"source={report['source_record_count']} retained={report['retained_record_count']} "
            f"same_label_excluded={report['excluded_same_label_duplicate_count']} "
            f"cross_label_quarantined={report['quarantined_cross_label_record_count']} "
            f"validate_only={args.validate_only}"
        )
        return 0
    except Exception as exc:
        print(f"Face duplicate remediation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

