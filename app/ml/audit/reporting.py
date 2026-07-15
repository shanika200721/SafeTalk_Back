"""Privacy-safe audit report serialization and Markdown summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic.v1 import ValidationError

from app.ml.audit.base import severity_rank
from app.ml.audit.schemas import AuditIssue, AuditSeverity, DatasetAuditReport
from app.ml.common import paths


def _resolve_output_path(output_path: str | os.PathLike[str] | Path) -> Path:
    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _assert_audit_output_allowed(output_path: Path) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if not paths.is_path_inside(paths.get_generated_root(), output_path):
        raise ValueError("Audit reports must be saved under generated/")
    return output_path


def default_audit_output_dir(report: DatasetAuditReport) -> Path:
    return paths.get_generated_root() / "audits" / report.dataset_name / report.dataset_version


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> Path:
    resolved = _assert_audit_output_allowed(path)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing audit report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp_path = resolved.with_name(f".{resolved.name}.tmp")
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(resolved)
    return resolved


def save_audit_json(
    report: DatasetAuditReport,
    output_path: str | os.PathLike[str] | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    target = default_audit_output_dir(report) / "audit.json" if output_path is None else _resolve_output_path(output_path)
    text = json.dumps(report.to_safe_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    return _atomic_write_text(target, text, overwrite=overwrite)


def save_audit_markdown(
    report: DatasetAuditReport,
    output_path: str | os.PathLike[str] | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    target = default_audit_output_dir(report) / "audit.md" if output_path is None else _resolve_output_path(output_path)
    return _atomic_write_text(target, create_markdown_summary(report), overwrite=overwrite)


def load_audit_report(path: str | os.PathLike[str] | Path) -> DatasetAuditReport:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed audit JSON in {path}: {exc}") from exc
    try:
        return DatasetAuditReport.parse_obj(payload)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse audit report {path}: {exc}") from exc


def _issue_counts(issues: list[AuditIssue]) -> dict[str, int]:
    counts = {severity.value: 0 for severity in AuditSeverity}
    for item in issues:
        counts[item.severity.value] += 1
    return counts


def _line_items(items: list[str]) -> str:
    if not items:
        return "none"
    return ", ".join(f"`{item}`" for item in items[:25]) + (" ..." if len(items) > 25 else "")


def _class_distribution_md(distribution: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if not distribution:
        return ["- none configured or available"]
    for column, values in sorted(distribution.items()):
        rendered = ", ".join(f"{item.label}: {item.count} ({item.percentage:.2f}%)" for item in values)
        lines.append(f"- `{column}`: {rendered}")
    return lines


def _tabular_lines(report: DatasetAuditReport) -> list[str]:
    result = report.tabular_result
    if result is None:
        return []
    missing_columns = [column for column in result.columns if column.null_count > 0]
    missing_total = sum(column.null_count for column in result.columns)
    return [
        f"- Rows: {result.row_count}",
        f"- Columns: {result.column_count}",
        f"- Duplicate rows: {result.duplicate_row_count}",
        f"- Missing values: {missing_total} across {len(missing_columns)} columns",
        f"- Possible identifiers: {_line_items(result.possible_identifier_columns)}",
        f"- Possible sensitive fields: {_line_items(result.possible_sensitive_columns)}",
        f"- Leakage candidates: {_line_items(result.possible_leakage_columns)}",
    ]


def _text_lines(report: DatasetAuditReport) -> list[str]:
    result = report.text_result
    if result is None:
        return []
    return [
        f"- Records: {result.record_count}",
        f"- Missing or blank text: {result.missing_text_count}",
        f"- Exact duplicate normalized text: {result.exact_duplicate_text_count}",
        f"- Conflicting duplicate labels: {result.duplicate_text_conflicting_labels_count}",
        f"- Near-duplicate candidates: {result.near_duplicate_candidate_count}",
        f"- Privacy patterns: urls={result.url_occurrence_count}, emails={result.email_occurrence_count}, usernames={result.username_occurrence_count}, phones={result.phone_occurrence_count}",
    ]


def _audio_lines(report: DatasetAuditReport) -> list[str]:
    result = report.audio_result
    if result is None:
        return []
    return [
        f"- Files: {result.file_count}",
        f"- Readable: {result.readable_file_count}",
        f"- Unreadable: {result.unreadable_file_count}",
        f"- Total duration seconds: {result.total_duration_seconds:.3f}",
        f"- Duplicate hash groups: {result.duplicate_hash_group_count}",
    ]


def _image_lines(report: DatasetAuditReport) -> list[str]:
    result = report.image_result
    if result is None:
        return []
    return [
        f"- Files: {result.file_count}",
        f"- Readable: {result.readable_file_count}",
        f"- Unreadable: {result.unreadable_file_count}",
        f"- Duplicate hash groups: {result.duplicate_hash_group_count}",
    ]


def _distribution_for_report(report: DatasetAuditReport) -> dict[str, Any]:
    if report.tabular_result is not None:
        return report.tabular_result.class_distribution
    if report.text_result is not None:
        return report.text_result.label_distribution
    if report.audio_result is not None:
        return report.audio_result.label_distribution
    if report.image_result is not None:
        return report.image_result.label_distribution
    return {}


def create_markdown_summary(report: DatasetAuditReport) -> str:
    issue_counts = _issue_counts(report.issues)
    lines = [
        f"# Dataset Audit: {report.dataset_name}",
        "",
        f"- Dataset version: `{report.dataset_version}`",
        f"- Modality: `{report.modality.value}`",
        f"- Source path: `{report.source_relative_path}`",
        f"- Source fingerprint: `{report.source_fingerprint_hash}`",
        f"- Config hash: `{report.config_hash or 'none'}`",
        f"- Audit version: `{report.audit_version}`",
        f"- Summary status: `{report.summary_status}`",
        f"- Started: `{report.audit_started_at.isoformat()}`",
        f"- Completed: `{report.audit_completed_at.isoformat()}`",
        "",
        "## High-Level Counts",
        *(_tabular_lines(report) or _text_lines(report) or _audio_lines(report) or _image_lines(report) or ["- none"]),
        "",
        "## Class Distribution",
        *_class_distribution_md(_distribution_for_report(report)),
        "",
        "## Missing Data",
    ]
    if report.tabular_result is not None:
        missing = [f"`{column.column_name}`={column.null_count}" for column in report.tabular_result.columns if column.null_count]
        lines.append("- " + (", ".join(missing[:50]) if missing else "none detected"))
    elif report.text_result is not None:
        lines.append(f"- Missing or blank text: {report.text_result.missing_text_count}")
    else:
        lines.append("- not applicable")

    lines.extend(
        [
            "",
            "## Duplicate Summary",
        ]
    )
    if report.tabular_result is not None:
        lines.append(f"- Duplicate rows: {report.tabular_result.duplicate_row_count}")
    elif report.text_result is not None:
        lines.append(f"- Exact duplicate normalized text: {report.text_result.exact_duplicate_text_count}")
    elif report.audio_result is not None:
        lines.append(f"- Duplicate hash groups: {report.audio_result.duplicate_hash_group_count}")
    elif report.image_result is not None:
        lines.append(f"- Duplicate hash groups: {report.image_result.duplicate_hash_group_count}")
    else:
        lines.append("- none")

    privacy_issues = [item for item in report.issues if "sensitive" in item.code or "identifier" in item.code or "email" in item.code or "phone" in item.code]
    leakage_issues = [item for item in report.issues if "leakage" in item.code or "overlap" in item.code]
    critical = [item for item in report.issues if item.severity == AuditSeverity.CRITICAL]
    warnings = [item for item in report.issues if item.severity == AuditSeverity.WARNING]
    privacy_lines = [
        f"- `{item.code}`" + (f" on `{item.field_name}`" if item.field_name else "")
        for item in privacy_issues[:50]
    ] or ["- none detected"]
    leakage_lines = [
        f"- `{item.code}`" + (f" on `{item.field_name}`" if item.field_name else "")
        for item in leakage_issues[:50]
    ] or ["- none detected"]
    critical_lines = [f"- `{item.code}`: {item.message}" for item in critical] or ["- none"]
    warning_lines = [
        f"- `{item.code}`" + (f" on `{item.field_name}`" if item.field_name else "")
        for item in warnings[:100]
    ] or ["- none"]

    lines.extend(
        [
            "",
            "## Privacy Findings",
            *privacy_lines,
            "",
            "## Leakage Candidates",
            *leakage_lines,
            "",
            "## Critical Issues",
            *critical_lines,
            "",
            "## Warnings",
            *warning_lines,
            "",
            "## Issue Counts",
            f"- critical: {issue_counts['critical']}",
            f"- error: {issue_counts['error']}",
            f"- warning: {issue_counts['warning']}",
            f"- info: {issue_counts['info']}",
            "",
            "## Limitations",
            f"- {report.notes or 'No additional limitations recorded.'}",
            "- Audit reports intentionally omit raw text, image pixels, audio transcripts, direct identifiers, and absolute machine paths.",
            "",
        ]
    )
    return "\n".join(lines)


def sorted_issues(issues: list[AuditIssue]) -> list[AuditIssue]:
    return sorted(issues, key=lambda item: (severity_rank(item.severity), item.code, item.field_name or ""))
