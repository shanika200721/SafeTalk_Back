"""CLI for Phase 3F Speech leave-one-corpus-out domain-shift evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.evaluation.speech_domain.constants import (  # noqa: E402
    CORPORA,
    DEFAULT_CANONICAL_MANIFEST,
    DEFAULT_CORPUS_MAPPING,
    DEFAULT_FEATURE_SCHEMA,
    DEFAULT_FEATURES,
    DEFAULT_FINGERPRINT_DIR,
    DEFAULT_LABEL_POLICY,
    DEFAULT_MODEL_ROOT,
    DEFAULT_REPORT_DIR,
)
from app.ml.evaluation.speech_domain.runner import run_speech_domain_shift_evaluation  # noqa: E402


def _cli_path(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith("../") or normalized.startswith("./"):
        return str(Path(value).resolve(strict=False))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Speech domain shift with leave-one-corpus-out folds.")
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--canonical-manifest", default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--feature-schema", default=DEFAULT_FEATURE_SCHEMA)
    parser.add_argument("--corpus-mapping", default=DEFAULT_CORPUS_MAPPING)
    parser.add_argument("--label-policy", default=DEFAULT_LABEL_POLICY)
    parser.add_argument("--fingerprint-dir", default=DEFAULT_FINGERPRINT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--strategy", default="leave_one_corpus_out", choices=["leave_one_corpus_out", "corpus_transfer_matrix"])
    parser.add_argument("--held-out-corpus", default=None, choices=CORPORA)
    parser.add_argument("--candidate", default="all", choices=["all", "logistic_regression", "random_forest", "linear_svm", "svm"])
    parser.add_argument("--shared-labels-only", action="store_true", default=True)
    parser.add_argument("--full-labels", action="store_true", help="Use all policy-supported labels instead of the shared-label LOCO subset.")
    parser.add_argument("--run-transfer-matrix", action="store_true")
    parser.add_argument("--run-shortcut-diagnostics", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-records-per-corpus", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path", default=None)
    parser.add_argument("--no-artifacts", action="store_true", help="Do not save fold-specific joblib research artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_speech_domain_shift_evaluation(
            features_path=_cli_path(args.features),
            canonical_manifest_path=_cli_path(args.canonical_manifest),
            feature_schema_path=_cli_path(args.feature_schema),
            corpus_mapping_path=_cli_path(args.corpus_mapping),
            label_policy_path=_cli_path(args.label_policy),
            fingerprint_dir=_cli_path(args.fingerprint_dir),
            output_dir=_cli_path(args.output_dir),
            model_root=_cli_path(args.model_root),
            strategy=args.strategy,
            held_out_corpus=args.held_out_corpus,
            candidate=args.candidate,
            shared_labels_only=not args.full_labels,
            run_transfer=args.run_transfer_matrix,
            run_shortcut=args.run_shortcut_diagnostics,
            dry_run=args.dry_run,
            max_records_per_corpus=args.max_records_per_corpus,
            overwrite=args.overwrite,
            report_path=_cli_path(args.report_path),
            save_artifacts=not args.no_artifacts,
        )
    except Exception as exc:
        print(f"Speech domain-shift evaluation failed: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    fold_summary = [
        {
            "test_corpus": fold.get("test_corpus"),
            "selected_candidate": fold.get("candidate_model"),
            "validation_macro_f1": (fold.get("validation_metrics") or {}).get("macro_f1"),
            "test_macro_f1": (fold.get("test_metrics") or {}).get("macro_f1"),
            "test_balanced_accuracy": (fold.get("test_metrics") or {}).get("balanced_accuracy"),
            "worst_class_recall": (fold.get("test_metrics") or {}).get("worst_class_recall"),
        }
        for fold in result.get("folds", [])
    ]
    payload = {
        "status": "completed",
        "evaluation_version": result.get("evaluation_version"),
        "policy_version": result.get("policy_version"),
        "folds": fold_summary,
        "pooled_vs_loco": result.get("corpus_gap_summary"),
        "shortcut_accuracy": (result.get("shortcut_risk_findings") or {}).get("accuracy"),
        "report_paths": result.get("report_paths"),
        "registered": False,
        "active": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
