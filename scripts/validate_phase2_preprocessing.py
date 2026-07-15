"""Validate Phase 2 preprocessing outputs without training models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common import paths
from app.ml.validation.constants import DEFAULT_VALIDATION_OUTPUT_FILES, KNOWN_MODALITIES
from app.ml.validation.cross_modality import validate_phase2_cross_modality
from app.ml.validation.reporting import (
    create_phase2_markdown_summary,
    save_phase2_validation_json,
    save_phase2_validation_markdown,
    save_supporting_reports,
)
from app.ml.validation.schemas import ValidationSeverity, ValidationStatus


def _resolve_cli_path(value: str | None, *, default: Path) -> Path:
    if value is None:
        return default.resolve(strict=False)
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Phase 2 preprocessing artifacts and readiness.")
    parser.add_argument("--config-dir", default="../ml-research/configs")
    parser.add_argument("--generated-root", default="../generated")
    parser.add_argument("--output-dir", default="../generated/reports/phase2_validation")
    parser.add_argument("--modalities", nargs="*", choices=KNOWN_MODALITIES, default=list(KNOWN_MODALITIES))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--skip-source-reverification", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--report-path")
    return parser


def _print_summary(report) -> None:
    print("Phase 2 validation summary")
    print(f"validation_version={report.validation_version}")
    print(f"readiness_policy_version={report.readiness_policy_version}")
    print(f"checks passed={report.passed_checks} warnings={report.warning_checks} failed={report.failed_checks} blocked={report.blocked_checks}")
    for modality in report.modalities:
        print(
            f"{modality.modality}: {modality.readiness_classification} "
            f"(warnings={len(modality.warnings)}, blockers={len(modality.blockers)})"
        )
    print("fusion: blocked_pending_data (no shared participant key)")


def _has_source_mismatch(report) -> bool:
    return any(
        check.check_name == "fingerprint_matches_source" and str(check.status) == ValidationStatus.BLOCKED.value
        for modality in report.modalities
        for check in modality.checks
    )


def _has_critical_failure(report) -> bool:
    return any(
        str(check.status) == ValidationStatus.FAILED.value and str(check.severity) == ValidationSeverity.CRITICAL.value
        for modality in report.modalities
        for check in modality.checks
    )


def _write_inventory_only(inventory, output_dir: Path, *, overwrite: bool) -> Path:
    path = output_dir / DEFAULT_VALIDATION_OUTPUT_FILES["artifact_inventory"]
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing inventory: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.to_safe_dict() for item in inventory], indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_dir = _resolve_cli_path(args.config_dir, default=paths.get_ml_research_root() / "configs")
    generated_root = _resolve_cli_path(args.generated_root, default=paths.get_generated_root())
    output_dir = _resolve_cli_path(args.output_dir, default=paths.get_generated_reports_root() / "phase2_validation")

    try:
        report, inventory = validate_phase2_cross_modality(
            config_dir=config_dir,
            generated_root=generated_root,
            modalities=args.modalities,
            skip_source_reverification=args.skip_source_reverification,
        )
        _print_summary(report)

        if args.summary_only:
            return _exit_code(report, strict=args.strict, fail_on_warning=args.fail_on_warning)

        if args.inventory_only:
            _write_inventory_only(inventory, output_dir, overwrite=args.overwrite)
            return _exit_code(report, strict=args.strict, fail_on_warning=args.fail_on_warning)

        if args.report_path:
            report_path = _resolve_cli_path(args.report_path, default=output_dir / "phase2_validation_report.json")
            if report_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report.to_safe_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
            markdown_path = report_path.with_suffix(".md")
            if markdown_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing report: {markdown_path}")
            markdown_path.write_text(create_phase2_markdown_summary(report), encoding="utf-8")
        else:
            save_phase2_validation_json(report, output_dir, overwrite=args.overwrite)
            save_phase2_validation_markdown(report, output_dir, overwrite=args.overwrite)
        save_supporting_reports(report, inventory, output_dir, overwrite=args.overwrite)
        return _exit_code(report, strict=args.strict, fail_on_warning=args.fail_on_warning)
    except Exception as exc:
        print(f"Phase 2 validation failed: {exc}", file=sys.stderr)
        return 2


def _exit_code(report, *, strict: bool, fail_on_warning: bool) -> int:
    if _has_source_mismatch(report) or _has_critical_failure(report):
        return 1
    if fail_on_warning and report.warning_checks:
        return 1
    if strict and (report.blocked_checks or report.global_blockers or any(modality.blockers for modality in report.modalities)):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
