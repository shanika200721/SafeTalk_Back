"""Validate and canonicalize Daily Mood data without training models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common.fingerprinting import load_dataset_fingerprint, verify_dataset_fingerprint
from app.ml.common.serialization import load_dataset_config, load_preprocessing_config
from app.ml.preprocessing.mood.mapping import load_mood_mapping_config
from app.ml.preprocessing.mood.preprocessor import (
    preprocess_mood_dataframe,
    preprocess_mood_dataset,
    synthetic_fingerprint,
    synthetic_mood_fixture,
)
from app.ml.preprocessing.mood.reporting import save_mood_report_markdown


def _resolve_cli_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Daily Mood data into canonical research outputs.")
    parser.add_argument("--dataset-config", help="Path to DatasetConfig JSON. Required for offline-file mode.")
    parser.add_argument("--preprocessing-config", help="Path to PreprocessingConfig JSON. Required for offline-file mode.")
    parser.add_argument("--mapping-config", required=True, help="Path to mood field mapping JSON.")
    parser.add_argument("--fingerprint", help="Path to Daily Mood fingerprint JSON. Required for offline-file mode.")
    parser.add_argument("--output-dir", required=True, help="Directory under generated/ for canonical outputs.")
    parser.add_argument("--validate-only", action="store_true", help="Validate source/mapping without writing canonical data.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing preprocessing outputs.")
    parser.add_argument("--source-mode", choices=["offline-file", "synthetic-test"], default="offline-file")
    parser.add_argument("--max-participants", type=int, help="Optional deterministic participant limit.")
    parser.add_argument("--max-records", type=int, help="Optional deterministic source-row limit.")
    parser.add_argument("--report-path", help="Optional Markdown report path. In validate-only mode, this is the only write.")
    return parser


def _print_summary(result: dict, *, source_mode: str) -> None:
    print(f"validation: {'passed' if result['valid'] else 'failed'}")
    print(f"source mode: {source_mode}")
    print(f"synthetic test data: {result['synthetic']}")
    print(f"validate only: {result['validate_only']}")
    print(f"source records: {result['source_rows']}")
    print(f"output records: {result['output_rows']}")
    print(f"participants: {result['participant_count']}")
    print(f"date range: {result['date_range']}")
    print(f"duplicate records: {result['duplicate_count']}")
    print(f"missing values: {result['missing_value_summary']}")
    print(f"feature columns: {', '.join(result['feature_columns'])}")
    if result["outputs"]:
        for label, path in result["outputs"].items():
            print(f"{label}: {path}")


def _require(value: str | None, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required for offline-file mode")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        mapping_config = load_mood_mapping_config(_resolve_cli_path(args.mapping_config))
        output_dir = _resolve_cli_path(args.output_dir)

        if args.source_mode == "synthetic-test":
            source_df = synthetic_mood_fixture()
            result = preprocess_mood_dataframe(
                source_df,
                None,
                mapping_config,
                source_fingerprint=synthetic_fingerprint(source_df),
                output_dir=output_dir,
                overwrite=args.overwrite,
                validate_only=args.validate_only,
                synthetic=True,
                max_participants=args.max_participants,
                max_records=args.max_records,
            )
        else:
            dataset_config = load_dataset_config(_resolve_cli_path(_require(args.dataset_config, "--dataset-config")))
            preprocessing_config = load_preprocessing_config(_resolve_cli_path(_require(args.preprocessing_config, "--preprocessing-config")))
            fingerprint = load_dataset_fingerprint(_resolve_cli_path(_require(args.fingerprint, "--fingerprint")))
            if not verify_dataset_fingerprint(fingerprint, dataset_config):
                raise ValueError("Daily Mood fingerprint mismatch: source changed")
            result = preprocess_mood_dataset(
                dataset_config,
                preprocessing_config,
                mapping_config,
                fingerprint,
                output_dir=output_dir,
                overwrite=args.overwrite,
                validate_only=args.validate_only,
                max_participants=args.max_participants,
                max_records=args.max_records,
            )

        if args.report_path:
            save_mood_report_markdown(
                result["report"],
                result["feature_schema"],
                _resolve_cli_path(args.report_path),
                overwrite=args.overwrite,
            )

        _print_summary(result, source_mode=args.source_mode)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
