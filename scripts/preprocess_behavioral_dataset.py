"""Validate and canonicalize behavioral telemetry without training models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common import paths
from app.ml.common.fingerprinting import load_dataset_fingerprint, verify_dataset_fingerprint
from app.ml.common.serialization import load_dataset_config, load_preprocessing_config
from app.ml.preprocessing.behavioral.constants import SOURCE_STATUS_NO_BEHAVIORAL_DATA, SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY
from app.ml.preprocessing.behavioral.mapping import load_behavioral_mapping_config
from app.ml.preprocessing.behavioral.preprocessor import (
    preprocess_behavioral_dataframe,
    preprocess_behavioral_dataset,
    write_schema_only_outputs,
)
from app.ml.preprocessing.behavioral.synthetic import generate_synthetic_behavioral_events, synthetic_behavioral_fingerprint


def _resolve_cli_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess behavioral telemetry into privacy-preserving research outputs.")
    parser.add_argument("--dataset-config", help="Path to DatasetConfig JSON. Required for offline-file mode.")
    parser.add_argument("--preprocessing-config", help="Path to PreprocessingConfig JSON. Required for offline-file mode.")
    parser.add_argument("--mapping-config", help="Path to behavioral field mapping JSON. Defaults to canonical v1 mapping.")
    parser.add_argument("--fingerprint", help="Path to dataset fingerprint JSON. Required for offline-file mode.")
    parser.add_argument("--output-dir", default=str(paths.get_generated_preprocessing_root() / "behavioral" / "v1"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source-mode", choices=["offline-file", "synthetic-engineering", "schema-only"], default="offline-file")
    parser.add_argument("--max-participants", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--minimum-baseline-sessions", type=int, default=3)
    parser.add_argument("--minimum-baseline-days", type=int, default=3)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--report-path")
    return parser


def _require(value: str | None, label: str) -> str:
    if not value:
        raise ValueError(f"{label} is required for offline-file mode")
    return value


def _print_summary(result: dict, *, source_mode: str) -> None:
    print(f"validation: {'passed' if result['valid'] else 'failed'}")
    print(f"source mode: {source_mode}")
    print(f"source type: {result.get('source_type')}")
    print(f"validate only: {result.get('validate_only')}")
    print(f"source records: {result.get('source_rows')}")
    print(f"output events: {result.get('output_events')}")
    print(f"output sessions: {result.get('output_sessions')}")
    print(f"participants: {result.get('participant_count')}")
    print(f"baseline eligible participants: {result.get('baseline_eligible_participant_count')}")
    print(f"date range: {result.get('date_range')}")
    print(f"feature columns: {', '.join(result.get('feature_columns', []))}")
    for label, path in result.get("outputs", {}).items():
        print(f"{label}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        mapping_config = load_behavioral_mapping_config(_resolve_cli_path(args.mapping_config) if args.mapping_config else None)
        output_dir = _resolve_cli_path(args.output_dir)
        if args.source_mode == "schema-only":
            outputs = write_schema_only_outputs(output_dir, overwrite=args.overwrite, source_status=SOURCE_STATUS_NO_BEHAVIORAL_DATA)
            print("validation: passed")
            print("source mode: schema-only")
            print("source type: no_behavioral_data")
            print("model training blocked: True")
            for label, path in outputs.items():
                print(f"{label}: {path}")
            return 0
        if args.source_mode == "synthetic-engineering":
            source_df = generate_synthetic_behavioral_events(seed=args.sample_seed)
            if "engineering_scenario" in source_df.columns:
                source_df = source_df.drop(columns=["engineering_scenario", "synthetic", "synthetic_schema_version"])
            result = preprocess_behavioral_dataframe(
                source_df,
                None,
                mapping_config,
                source_fingerprint=synthetic_behavioral_fingerprint(source_df),
                output_dir=output_dir / "generated-synthetic",
                overwrite=args.overwrite,
                validate_only=args.validate_only,
                source_type=SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY,
                max_participants=args.max_participants,
                max_records=args.max_records,
                minimum_baseline_sessions=args.minimum_baseline_sessions,
                minimum_baseline_days=args.minimum_baseline_days,
            )
        else:
            dataset_config = load_dataset_config(_resolve_cli_path(_require(args.dataset_config, "--dataset-config")))
            preprocessing_config = load_preprocessing_config(_resolve_cli_path(_require(args.preprocessing_config, "--preprocessing-config")))
            fingerprint = load_dataset_fingerprint(_resolve_cli_path(_require(args.fingerprint, "--fingerprint")))
            if not verify_dataset_fingerprint(fingerprint, dataset_config):
                raise ValueError("Behavioral fingerprint mismatch: source changed")
            result = preprocess_behavioral_dataset(
                dataset_config,
                preprocessing_config,
                mapping_config,
                fingerprint,
                output_dir=output_dir,
                overwrite=args.overwrite,
                validate_only=args.validate_only,
                max_participants=args.max_participants,
                max_records=args.max_records,
                minimum_baseline_sessions=args.minimum_baseline_sessions,
                minimum_baseline_days=args.minimum_baseline_days,
            )
        if args.report_path:
            from app.ml.preprocessing.behavioral.reporting import save_behavioral_report_markdown

            save_behavioral_report_markdown(result["report"], result["feature_schema"], _resolve_cli_path(args.report_path), overwrite=args.overwrite)
        _print_summary(result, source_mode=args.source_mode)
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

