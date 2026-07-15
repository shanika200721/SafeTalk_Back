"""Validate and canonicalize the Student Profile dataset without training models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common.fingerprinting import load_dataset_fingerprint, verify_dataset_fingerprint
from app.ml.common.serialization import load_dataset_config, load_preprocessing_config
from app.ml.preprocessing.profile.mapping import load_profile_mapping_config
from app.ml.preprocessing.profile.preprocessor import preprocess_profile_dataset
from app.ml.preprocessing.profile.reporting import save_profile_report_markdown


def _resolve_cli_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Student Profile data into canonical research outputs.")
    parser.add_argument("--dataset-config", required=True, help="Path to DatasetConfig JSON.")
    parser.add_argument("--preprocessing-config", required=True, help="Path to PreprocessingConfig JSON.")
    parser.add_argument("--mapping-config", required=True, help="Path to profile field mapping JSON.")
    parser.add_argument("--fingerprint", required=True, help="Path to Student Profile fingerprint JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory under generated/ for canonical outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing preprocessing outputs.")
    parser.add_argument("--validate-only", action="store_true", help="Validate source/mapping without writing canonical data.")
    parser.add_argument("--include-sensitive-context", action="store_true", help="Include optional sensitive-context fields as features.")
    parser.add_argument("--exclude-treatment-seeking", action="store_true", default=True, help="Exclude treatment-seeking feature.")
    parser.add_argument("--include-treatment-seeking", dest="exclude_treatment_seeking", action="store_false", help="Explicitly include treatment-seeking despite leakage warning.")
    parser.add_argument("--report-path", help="Optional Markdown report path. In validate-only mode, this is the only write.")
    return parser


def _print_summary(result: dict, *, validate_only: bool) -> None:
    print(f"validation: {'passed' if result['valid'] else 'failed'}")
    print(f"validate only: {validate_only}")
    print(f"source rows: {result['source_rows']}")
    print(f"output rows: {result['output_rows']}")
    print(f"excluded rows: {result['excluded_rows']}")
    print(f"target distribution: {result['target_distribution']}")
    print(f"feature columns: {', '.join(result['feature_columns']) if result['feature_columns'] else 'none'}")
    print(f"excluded columns: {', '.join(result['excluded_columns']) if result['excluded_columns'] else 'none'}")
    if result["sensitive_context_columns"]:
        print(f"sensitive context included: {', '.join(result['sensitive_context_columns'])}")
    else:
        print("sensitive context included: none")
    if result["outputs"]:
        for label, path in result["outputs"].items():
            print(f"{label}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset_config = load_dataset_config(_resolve_cli_path(args.dataset_config))
        preprocessing_config = load_preprocessing_config(_resolve_cli_path(args.preprocessing_config))
        mapping_config = load_profile_mapping_config(_resolve_cli_path(args.mapping_config))
        fingerprint = load_dataset_fingerprint(_resolve_cli_path(args.fingerprint))

        if not verify_dataset_fingerprint(fingerprint, dataset_config):
            raise ValueError("Student Profile fingerprint mismatch: source changed")

        output_dir = _resolve_cli_path(args.output_dir)
        result = preprocess_profile_dataset(
            dataset_config,
            preprocessing_config,
            mapping_config,
            fingerprint,
            output_dir=output_dir,
            overwrite=args.overwrite,
            validate_only=args.validate_only,
            include_sensitive_context=args.include_sensitive_context,
            exclude_treatment_seeking=args.exclude_treatment_seeking,
        )

        if args.report_path:
            save_profile_report_markdown(
                result["report"],
                result["feature_schema"],
                _resolve_cli_path(args.report_path),
                overwrite=args.overwrite,
            )

        _print_summary(result, validate_only=args.validate_only)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
