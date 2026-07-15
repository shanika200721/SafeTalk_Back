"""Create or verify a Phase 2 dataset fingerprint report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common.fingerprinting import (
    fingerprint_dataset,
    load_dataset_fingerprint,
    save_dataset_fingerprint,
    verify_dataset_fingerprint,
)
from app.ml.common.serialization import load_dataset_config


def _parse_extensions(value: str | None) -> list[str] | None:
    if value is None:
        return None
    extensions = [item.strip() for item in value.split(",") if item.strip()]
    return extensions or None


def _print_summary(fingerprint, output_path: Path | None) -> None:
    print(f"dataset name: {fingerprint.dataset_name}")
    print(f"version: {fingerprint.dataset_version}")
    print(f"modality: {fingerprint.modality.value}")
    print(f"file count: {fingerprint.file_count}")
    print(f"total bytes: {fingerprint.total_bytes}")
    print(f"combined hash: {fingerprint.combined_sha256}")
    if output_path is not None:
        print(f"output path: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify a read-only dataset fingerprint report.")
    parser.add_argument("--config", required=True, help="Path to a DatasetConfig JSON file.")
    parser.add_argument("--output", help="Path for the fingerprint JSON report.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing fingerprint report.")
    parser.add_argument("--verify-existing", action="store_true", help="Verify --output against current source files.")
    parser.add_argument("--summary-only", action="store_true", help="Calculate and print a fingerprint without saving it.")
    parser.add_argument("--extensions", help="Comma-separated extension allow-list for folder datasets.")
    parser.add_argument("--include-modified-time", action="store_true", help="Include per-file UTC modified timestamps.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config_path = Path(args.config).resolve(strict=False)
        dataset_config = load_dataset_config(config_path)

        output_path = Path(args.output).resolve(strict=False) if args.output else None
        if args.verify_existing:
            if output_path is None:
                raise ValueError("--verify-existing requires --output")
            saved = load_dataset_fingerprint(output_path)
            ok = verify_dataset_fingerprint(saved, dataset_config)
            _print_summary(saved, output_path)
            print(f"verification: {'unchanged' if ok else 'changed'}")
            return 0 if ok else 1

        fingerprint = fingerprint_dataset(
            dataset_config,
            allowed_extensions=_parse_extensions(args.extensions),
            include_modified_time=args.include_modified_time,
        )

        saved_path = None
        if not args.summary_only:
            saved_path = save_dataset_fingerprint(fingerprint, output_path, overwrite=args.overwrite)

        _print_summary(fingerprint, saved_path or output_path)
        if args.summary_only:
            print("summary only: report not saved")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
