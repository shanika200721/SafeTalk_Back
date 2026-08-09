"""CLI for the Phase 3E Speech emotion acoustic research baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.training.speech.constants import (  # noqa: E402
    DEFAULT_CANONICAL_MANIFEST,
    DEFAULT_CORPUS_SUMMARY,
    DEFAULT_DUPLICATE_ISOLATION_REPORT,
    DEFAULT_DUPLICATE_MANIFEST,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_FEATURES,
    DEFAULT_FINGERPRINT_DIR,
    DEFAULT_MODEL_ROOT,
    DEFAULT_PREPROCESSING_REPORT,
    DEFAULT_REPORT_DIR,
    DEFAULT_SPEAKER_ISOLATION_REPORT,
    DEFAULT_SPLIT_ASSIGNMENTS,
    DEFAULT_SPLIT_MANIFEST,
)
from app.ml.training.speech.runner import run_speech_baseline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Speech acoustic emotion research baseline without activation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation-config", default="ml-research/configs/training.speech.ablation.v1.json")
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--canonical-manifest", default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--feature-schema", default=DEFAULT_FEATURE_SCHEMA)
    parser.add_argument("--preprocessing-report", default=DEFAULT_PREPROCESSING_REPORT)
    parser.add_argument("--fingerprint-dir", default=DEFAULT_FINGERPRINT_DIR)
    parser.add_argument("--duplicate-manifest", default=DEFAULT_DUPLICATE_MANIFEST)
    parser.add_argument("--corpus-summary", default=DEFAULT_CORPUS_SUMMARY)
    parser.add_argument("--split-assignments", default=DEFAULT_SPLIT_ASSIGNMENTS)
    parser.add_argument("--speaker-isolation-report", default=DEFAULT_SPEAKER_ISOLATION_REPORT)
    parser.add_argument("--duplicate-isolation-report", default=DEFAULT_DUPLICATE_ISOLATION_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--candidate", default="all", choices=["all", "logistic_regression", "random_forest", "svm", "linear_svm", "rbf_svm"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--extract-missing-features", action="store_true")
    parser.add_argument("--register-candidate", action="store_true")
    parser.add_argument("--test-database-url", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path", default=None, help="Optional JSON summary copy path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_speech_baseline(
            config_path=args.config,
            ablation_config_path=args.ablation_config,
            features_path=args.features,
            canonical_manifest_path=args.canonical_manifest,
            split_manifest_path=args.split_manifest,
            feature_schema_path=args.feature_schema,
            preprocessing_report_path=args.preprocessing_report,
            fingerprint_dir=args.fingerprint_dir,
            duplicate_manifest_path=args.duplicate_manifest,
            corpus_summary_path=args.corpus_summary,
            split_assignments_path=args.split_assignments,
            speaker_isolation_report_path=args.speaker_isolation_report,
            duplicate_isolation_report_path=args.duplicate_isolation_report,
            report_dir=args.output_dir,
            model_root=args.model_root,
            feature_set=args.feature_set,
            candidate=args.candidate,
            dry_run=args.dry_run,
            max_train_records=args.max_train_records,
            extract_missing_features=args.extract_missing_features,
            overwrite=args.overwrite,
            register_candidate=args.register_candidate,
            test_database_url=args.test_database_url,
        )
    except Exception as exc:
        print(f"Speech baseline failed: {exc}", file=sys.stderr)
        return 1

    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.report_path:
            Path(args.report_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    summary = result.metrics["summary"]
    test = summary.get("test_metrics") or {}
    payload = {
        "status": "completed",
        "run_id": result.run_id,
        "selected_candidate": summary.get("selected_candidate"),
        "validation_macro_f1": (summary.get("validation_metrics") or {}).get("macro_f1"),
        "validation_macro_recall": (summary.get("validation_metrics") or {}).get("macro_recall"),
        "test_macro_f1": test.get("macro_f1"),
        "test_weighted_f1": test.get("weighted_f1"),
        "test_balanced_accuracy": test.get("balanced_accuracy"),
        "feature_coverage": summary.get("feature_coverage"),
        "skipped_candidates": result.skipped_candidates,
        "report_dir": str(result.report_dir),
        "run_dir": str(result.run_dir),
        "registered": result.registered,
        "active": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.report_path:
        Path(args.report_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
