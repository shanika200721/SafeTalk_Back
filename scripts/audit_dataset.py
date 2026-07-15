"""Run a read-only Phase 2 dataset audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.audit import audit_dataset
from app.ml.audit.reporting import save_audit_json, save_audit_markdown
from app.ml.audit.schemas import AuditSeverity
from app.ml.common.fingerprinting import load_dataset_fingerprint
from app.ml.common.serialization import load_dataset_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reusable read-only dataset audit.")
    parser.add_argument("--config", required=True, help="Path to a DatasetConfig JSON file.")
    parser.add_argument("--fingerprint", help="Path to a dataset fingerprint JSON report.")
    parser.add_argument("--output-dir", help="Directory for audit.json and audit.md.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing audit reports.")
    parser.add_argument("--max-records", type=int, default=100_000, help="Maximum tabular/text records to audit.")
    parser.add_argument("--max-files", type=int, default=2_000, help="Maximum media files to audit.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument("--summary-only", action="store_true", help="Print summary without saving reports.")
    parser.add_argument("--fail-on-critical", action="store_true", help="Exit non-zero if critical issues exist.")
    parser.add_argument("--text-column", help="Explicit text column override.")
    parser.add_argument("--label-column", help="Explicit label column override.")
    return parser


def _issue_counts(report) -> dict[str, int]:
    counts = {severity.value: 0 for severity in AuditSeverity}
    for item in report.issues:
        counts[item.severity.value] += 1
    return counts


def _print_summary(report, json_path: Path | None, md_path: Path | None) -> None:
    counts = _issue_counts(report)
    print(f"dataset name: {report.dataset_name}")
    print(f"version: {report.dataset_version}")
    print(f"modality: {report.modality.value}")
    print(f"status: {report.summary_status}")
    if report.tabular_result is not None:
        print(f"rows: {report.tabular_result.row_count}")
        print(f"columns: {report.tabular_result.column_count}")
        print(f"duplicate rows: {report.tabular_result.duplicate_row_count}")
    elif report.text_result is not None:
        print(f"records: {report.text_result.record_count}")
        print(f"duplicate texts: {report.text_result.exact_duplicate_text_count}")
    elif report.audio_result is not None:
        print(f"files: {report.audio_result.file_count}")
        print(f"readable files: {report.audio_result.readable_file_count}")
    elif report.image_result is not None:
        print(f"files: {report.image_result.file_count}")
        print(f"readable files: {report.image_result.readable_file_count}")
    print(
        "issues: "
        f"critical={counts['critical']} error={counts['error']} "
        f"warning={counts['warning']} info={counts['info']}"
    )
    if json_path is not None:
        print(f"json report: {json_path}")
    if md_path is not None:
        print(f"markdown report: {md_path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_dataset_config(Path(args.config).resolve(strict=False))
        fingerprint = load_dataset_fingerprint(Path(args.fingerprint).resolve(strict=False)) if args.fingerprint else None
        report = audit_dataset(
            config,
            fingerprint=fingerprint,
            options={
                "max_records": args.max_records,
                "max_files": args.max_files,
                "sample_seed": args.sample_seed,
                "summary_only": args.summary_only,
                "text_column": args.text_column,
                "label_column": args.label_column,
            },
        )

        json_path = None
        md_path = None
        if not args.summary_only:
            output_dir = Path(args.output_dir).resolve(strict=False) if args.output_dir else None
            json_target = output_dir / "audit.json" if output_dir else None
            md_target = output_dir / "audit.md" if output_dir else None
            json_path = save_audit_json(report, json_target, overwrite=args.overwrite)
            md_path = save_audit_markdown(report, md_target, overwrite=args.overwrite)

        _print_summary(report, json_path, md_path)
        if args.summary_only:
            print("summary only: reports not saved")
        if report.summary_status == "critical":
            return 1
        if args.fail_on_critical and any(item.severity == AuditSeverity.CRITICAL for item in report.issues):
            return 1
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
