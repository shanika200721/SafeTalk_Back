from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.training.face.constants import (  # noqa: E402
    DEFAULT_DEDUPLICATED_MANIFEST,
    DEFAULT_DUPLICATE_DECISIONS,
    DEFAULT_MODEL_ROOT,
    DEFAULT_QUARANTINE,
    DEFAULT_REPORT_DIR,
    DEFAULT_SOURCE_FINGERPRINT,
    DEFAULT_SPLIT_MANIFEST,
)
from app.ml.training.face.runner import run_face_baseline  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train restricted Face emotion research baseline without activation.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--ablation-config", default="ml-research/configs/training.face.ablation.v1.json")
    parser.add_argument("--deduplicated-manifest", default=DEFAULT_DEDUPLICATED_MANIFEST)
    parser.add_argument("--split-manifest", default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split-assignments", default="generated/manifests/splits/face/v2/face_split_assignments.csv")
    parser.add_argument("--quarantine", default=DEFAULT_QUARANTINE)
    parser.add_argument("--duplicate-decisions", default=DEFAULT_DUPLICATE_DECISIONS)
    parser.add_argument("--source-fingerprint", default=DEFAULT_SOURCE_FINGERPRINT)
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--candidate", default="all", choices=["all", "logistic_regression", "linear_svm", "random_forest"])
    parser.add_argument("--feature-set", default=None, choices=["flattened_pixels", "image_statistics"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--register-candidate", action="store_true")
    parser.add_argument("--test-database-url", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_face_baseline(
            config_path=args.config,
            ablation_config_path=args.ablation_config,
            deduplicated_manifest_path=args.deduplicated_manifest,
            split_manifest_path=args.split_manifest,
            split_assignments_path=args.split_assignments,
            quarantine_path=args.quarantine,
            duplicate_decisions_path=args.duplicate_decisions,
            source_fingerprint_path=args.source_fingerprint,
            report_dir=args.report_path or args.output_dir,
            model_root=args.model_root,
            candidate=args.candidate,
            feature_set=args.feature_set,
            dry_run=args.dry_run,
            max_train_records=args.max_train_records,
            overwrite=args.overwrite,
            register_candidate=args.register_candidate,
            test_database_url=args.test_database_url,
        )
    except Exception as exc:
        print(f"Face baseline failed: {exc}", file=sys.stderr)
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
                "selected_candidate": summary["selected_candidate"],
                "validation_macro_f1": summary["validation_metrics"].get("macro_f1"),
                "test_macro_f1": summary["test_metrics"].get("macro_f1"),
                "test_balanced_accuracy": summary["test_metrics"].get("balanced_accuracy"),
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

