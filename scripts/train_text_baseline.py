"""CLI for the Phase 3D Text classification research baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.training.text.constants import (  # noqa: E402
    DEFAULT_CANONICAL_DATA,
    DEFAULT_CONFLICT_QUARANTINE,
    DEFAULT_DUPLICATE_MANIFEST,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_MODEL_ROOT,
    DEFAULT_REPORT_DIR,
    DEFAULT_SOURCE_FINGERPRINT,
    DEFAULT_SOURCE_OVERLAP_REPORT,
    DEFAULT_SPLIT_MANIFEST,
)
from app.ml.training.text.runner import run_text_baseline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Text classification research baseline without activation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation-config", default="ml-research/configs/training.text.ablation.v1.json")
    parser.add_argument("--canonical-data", default=DEFAULT_CANONICAL_DATA)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--feature-schema", default=DEFAULT_FEATURE_SCHEMA)
    parser.add_argument("--source-fingerprint", default=DEFAULT_SOURCE_FINGERPRINT)
    parser.add_argument("--duplicate-manifest", default=DEFAULT_DUPLICATE_MANIFEST)
    parser.add_argument("--conflict-quarantine", default=DEFAULT_CONFLICT_QUARANTINE)
    parser.add_argument("--source-overlap-report", default=DEFAULT_SOURCE_OVERLAP_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--candidate", default="all", choices=["all", "logistic_regression", "linear_svm"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--register-candidate", action="store_true")
    parser.add_argument("--test-database-url", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_text_baseline(
            config_path=args.config,
            canonical_data_path=args.canonical_data,
            split_manifest_path=args.split_manifest,
            feature_schema_path=args.feature_schema,
            source_fingerprint_path=args.source_fingerprint,
            duplicate_manifest_path=args.duplicate_manifest,
            conflict_quarantine_path=args.conflict_quarantine,
            source_overlap_report_path=args.source_overlap_report,
            report_dir=args.output_dir,
            model_root=args.model_root,
            feature_set=args.feature_set,
            candidate=args.candidate,
            dry_run=args.dry_run,
            max_train_records=args.max_train_records,
            overwrite=args.overwrite,
            register_candidate=args.register_candidate,
            test_database_url=args.test_database_url,
        )
    except Exception as exc:
        print(f"Text baseline failed: {exc}", file=sys.stderr)
        return 1

    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    summary = result.metrics["summary"]
    test = summary.get("test_metrics") or {}
    suicidal = test.get("suicidal_class") or {}
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": result.run_id,
                "selected_candidate": summary.get("selected_candidate"),
                "validation_macro_f1": (summary.get("validation_metrics") or {}).get("macro_f1"),
                "validation_suicidal_recall": ((summary.get("validation_metrics") or {}).get("suicidal_class") or {}).get("recall"),
                "test_macro_f1": test.get("macro_f1"),
                "test_weighted_f1": test.get("weighted_f1"),
                "test_balanced_accuracy": test.get("balanced_accuracy"),
                "suicidal_false_negatives": suicidal.get("false_negatives"),
                "report_dir": str(result.report_dir),
                "run_dir": str(result.run_dir),
                "registered": result.registered,
                "active": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

