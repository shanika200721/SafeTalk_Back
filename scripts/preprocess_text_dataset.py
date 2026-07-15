"""Validate and canonicalize text classification data without training models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common.fingerprinting import load_dataset_fingerprint, verify_dataset_fingerprint
from app.ml.common.serialization import load_dataset_config, load_preprocessing_config
from app.ml.preprocessing.text.mapping import load_text_label_mapping_config
from app.ml.preprocessing.text.preprocessor import preprocess_text_dataset
from app.ml.preprocessing.text.schemas import TextSourceSelectionConfig


def _resolve_cli_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def _load_source_selection(path: Path) -> TextSourceSelectionConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return TextSourceSelectionConfig.parse_obj(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess text data into canonical privacy-safe research outputs.")
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--preprocessing-config", required=True)
    parser.add_argument("--label-mapping-config", required=True)
    parser.add_argument("--source-selection-config", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--near-duplicate-limit", type=int, default=1000)
    parser.add_argument("--deduplicate-exact", action="store_true")
    parser.add_argument("--quarantine-conflicts", action="store_true", default=True)
    parser.add_argument("--include-reference-test", help="Optional reference/test CSV for overlap reporting.")
    parser.add_argument("--report-path", help="Optional Markdown report path. In validate-only mode, this is the only write.")
    return parser


def _print_summary(result: dict) -> None:
    print(f"validation: {'passed' if result['valid'] else 'failed'}")
    print(f"validate only: {result['validate_only']}")
    print(f"source records: {result['source_rows']}")
    print(f"output records: {result['output_rows']}")
    print(f"excluded records: {result['excluded_rows']}")
    print(f"exact duplicate groups: {result['exact_duplicate_groups']}")
    print(f"conflicting duplicate groups: {result['conflicting_duplicate_groups']}")
    print(f"near duplicate candidates: {result['near_duplicate_candidates']}")
    print(f"label distribution before: {result['label_distribution_before']}")
    print(f"label distribution after: {result['label_distribution_after']}")
    print(f"privacy replacements: {result['privacy_replacement_summary']}")
    if result["outputs"]:
        for label, path in result["outputs"].items():
            print(f"{label}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset_config = load_dataset_config(_resolve_cli_path(args.dataset_config))
        preprocessing_config = load_preprocessing_config(_resolve_cli_path(args.preprocessing_config))
        label_mapping_config = load_text_label_mapping_config(_resolve_cli_path(args.label_mapping_config))
        source_selection_config = _load_source_selection(_resolve_cli_path(args.source_selection_config))
        fingerprint = load_dataset_fingerprint(_resolve_cli_path(args.fingerprint))
        if not verify_dataset_fingerprint(fingerprint, dataset_config):
            raise ValueError("Text source fingerprint mismatch: source changed")
        reference_df = pd.read_csv(_resolve_cli_path(args.include_reference_test)) if args.include_reference_test else None
        result = preprocess_text_dataset(
            dataset_config,
            preprocessing_config,
            label_mapping_config,
            source_selection_config,
            fingerprint,
            output_dir=_resolve_cli_path(args.output_dir),
            overwrite=args.overwrite,
            validate_only=args.validate_only,
            max_records=args.max_records,
            near_duplicate_limit=args.near_duplicate_limit,
            deduplicate_exact=args.deduplicate_exact,
            quarantine_conflicts=args.quarantine_conflicts,
            reference_df=reference_df,
        )
        if args.report_path:
            from app.ml.preprocessing.text.reporting import save_text_report_markdown

            save_text_report_markdown(result["report"], _resolve_cli_path(args.report_path), overwrite=args.overwrite)
        _print_summary(result)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
