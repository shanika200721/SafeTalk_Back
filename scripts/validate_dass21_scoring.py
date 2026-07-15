"""Validate DASS-21 scoring and item mapping without producing participant data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common import paths
from app.ml.common.fingerprinting import load_dataset_fingerprint, verify_dataset_fingerprint
from app.ml.common.serialization import load_dataset_config
from app.ml.preprocessing.dass21.constants import DASS21_SCORING_VERSION
from app.ml.preprocessing.dass21.dataset_mapping import (
    inspect_dass_dataset_columns,
    load_mapping_config,
    map_dataset_row_to_dass21_responses,
    validate_dataset_item_mapping,
)
from app.ml.preprocessing.dass21.scoring import score_dass21


def _default_fingerprint_path(dataset_name: str) -> Path:
    return paths.get_generated_manifests_root() / "fingerprints" / f"{dataset_name}-v1.json"


def _resolve_project_path(value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def _read_header(source_path: Path) -> list[str]:
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError("DASS source dataset is empty") from exc


def _bounded_sample_validation(source_path: Path, mapping_config: dict, sample_size: int) -> tuple[int, int]:
    success_count = 0
    invalid_count = 0
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if index >= sample_size:
                break
            try:
                responses = map_dataset_row_to_dass21_responses(row, mapping_config)
                score_dass21(responses)
                success_count += 1
            except ValueError:
                invalid_count += 1
    return success_count, invalid_count


def _write_report(report: dict, report_path: Path, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(report_path)
    if report_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate DASS-21 scoring configuration without saving participant data.")
    parser.add_argument("--dataset-config", required=True, help="Path to the DASS21 DatasetConfig JSON.")
    parser.add_argument("--mapping-config", required=True, help="Path to DASS21 item mapping JSON.")
    parser.add_argument("--header-only", action="store_true", help="Validate only source headers and item mapping.")
    parser.add_argument("--sample-size", type=int, default=0, help="Small deterministic sample size to score.")
    parser.add_argument("--report-path", required=True, help="Safe JSON report path under generated/ or ml-research/.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        dataset_config = load_dataset_config(Path(args.dataset_config).resolve(strict=False))
        source_path = dataset_config.validate_source_exists()
        mapping_config = load_mapping_config(Path(args.mapping_config).resolve(strict=False))

        fingerprint_path = _default_fingerprint_path(dataset_config.dataset_name)
        fingerprint = load_dataset_fingerprint(fingerprint_path)
        fingerprint_unchanged = verify_dataset_fingerprint(fingerprint, dataset_config)
        if not fingerprint_unchanged:
            raise ValueError("DASS21 fingerprint verification failed: source changed")

        columns = _read_header(source_path)
        inspection = inspect_dass_dataset_columns(columns)
        mapping_result = validate_dataset_item_mapping(columns, mapping_config)

        sample_size = 0 if args.header_only else max(0, args.sample_size)
        sample_success_count = 0
        invalid_response_count = 0
        if sample_size:
            sample_success_count, invalid_response_count = _bounded_sample_validation(
                source_path,
                mapping_config,
                sample_size,
            )

        report = {
            "mapping_success": mapping_result["mapping_success"],
            "response_column_count": mapping_result["response_column_count"],
            "excluded_timing_column_count": mapping_result["excluded_timing_column_count"],
            "invalid_response_count": invalid_response_count,
            "sample_scoring_success_count": sample_success_count,
            "sample_size": sample_size,
            "scoring_version": DASS21_SCORING_VERSION,
            "mapping_version": mapping_result["mapping_version"],
            "fingerprint_unchanged": fingerprint_unchanged,
            "questionnaire_source": inspection["questionnaire_source"],
        }
        saved = _write_report(report, Path(args.report_path).resolve(strict=False), args.overwrite)
        print(f"mapping success: {report['mapping_success']}")
        print(f"response columns: {report['response_column_count']}")
        print(f"excluded timing columns: {report['excluded_timing_column_count']}")
        print(f"invalid responses: {report['invalid_response_count']}")
        print(f"sample scoring successes: {report['sample_scoring_success_count']}")
        print(f"report path: {saved}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
