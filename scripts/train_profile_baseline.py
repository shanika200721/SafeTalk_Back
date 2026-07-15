"""CLI for the Phase 3C Student Profile depression baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.training.profile.constants import (  # noqa: E402
    DEFAULT_CANONICAL_DATA,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_MODEL_ROOT,
    DEFAULT_REPORT_DIR,
    DEFAULT_SOURCE_FINGERPRINT,
    DEFAULT_SPLIT_MANIFEST,
)
from app.ml.training.profile.runner import run_profile_baseline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Profile depression research baseline without activation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation-config", default="ml-research/configs/training.profile.ablation.v1.json")
    parser.add_argument("--canonical-data", default=DEFAULT_CANONICAL_DATA)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--feature-schema", default=DEFAULT_FEATURE_SCHEMA)
    parser.add_argument("--source-fingerprint", default=DEFAULT_SOURCE_FINGERPRINT)
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--feature-set", default=None)
    parser.add_argument("--candidate", default="all", choices=["all", "logistic_regression", "random_forest"])
    parser.add_argument("--threshold-strategy", default=None, choices=["default", "max_f1", "recall_priority"])
    parser.add_argument("--register-candidate", action="store_true")
    parser.add_argument("--test-database-url", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--allow-sensitive-context", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_dir = args.report_path or args.output_dir
    try:
        result = run_profile_baseline(
            config_path=args.config,
            canonical_data_path=args.canonical_data,
            split_manifest_path=args.split_manifest,
            feature_schema_path=args.feature_schema,
            source_fingerprint_path=args.source_fingerprint,
            report_dir=report_dir,
            model_root=args.model_root,
            feature_set=args.feature_set,
            candidate=args.candidate,
            threshold_strategy=args.threshold_strategy,
            dry_run=args.dry_run,
            allow_sensitive_context=args.allow_sensitive_context,
            overwrite=args.overwrite,
            register_candidate=args.register_candidate,
            test_database_url=args.test_database_url,
        )
    except Exception as exc:
        print(f"Profile baseline failed: {exc}", file=sys.stderr)
        return 1

    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    summary = result.metrics["summary"]
    print(
        json.dumps(
            {
                "status": "completed",
                "run_id": result.run_id,
                "feature_set": summary["feature_set"],
                "selected_candidate": summary["selected_candidate"],
                "threshold_strategy": summary["threshold_strategy"],
                "selected_threshold": summary["selected_threshold"],
                "test_confusion_matrix": summary["test_metrics"].get("confusion_matrix"),
                "false_negatives": summary["test_metrics"].get("false_negatives"),
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
