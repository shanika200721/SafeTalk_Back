"""Read-only tabular dataset audit."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ml.audit.base import AuditContext, build_report, is_sensitive_field_name, issue, safe_label_value, safe_value_summary
from app.ml.audit.schemas import (
    AuditIssue,
    AuditSeverity,
    ClassDistributionItem,
    ColumnAudit,
    TabularAuditResult,
)
from app.ml.common.schemas import SupportedFileFormat


_IDENTIFIER_NAME_RE = re.compile(r"(^id$|_id$|student_id|participant|user|email|phone|name|address)", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_LEAKAGE_NAME_HINTS = {
    "target",
    "label",
    "class",
    "outcome",
    "prediction",
    "suicide_attempt",
    "depression_label",
    "source",
    "uniquenetworklocation",
}
_TIMING_HINTS = ("timestamp", "time", "elapsed", "elapse")


def load_tabular_source(source_path: Path, file_format: SupportedFileFormat) -> pd.DataFrame:
    try:
        if file_format == SupportedFileFormat.CSV:
            return pd.read_csv(source_path, encoding="utf-8")
        if file_format == SupportedFileFormat.TSV:
            return pd.read_csv(source_path, sep="\t", encoding="utf-8")
        if file_format == SupportedFileFormat.JSON:
            return pd.read_json(source_path, orient="records")
        if file_format == SupportedFileFormat.JSONL:
            return pd.read_json(source_path, lines=True)
        if file_format == SupportedFileFormat.XLSX:
            return pd.read_excel(source_path)
    except UnicodeDecodeError as exc:
        raise ValueError(f"Could not decode tabular source as UTF-8: {source_path.name}") from exc
    except Exception as exc:
        raise ValueError(f"Could not parse tabular source {source_path.name}: {exc}") from exc
    raise ValueError(f"Unsupported tabular file format: {file_format.value}")


def _finite(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _percentage(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round((count / total) * 100, 4)


def _label_columns(context: AuditContext, df: pd.DataFrame) -> list[str]:
    labels = list(context.dataset_config.label_columns)
    if context.options.label_column and context.options.label_column not in labels:
        labels.append(context.options.label_column)
    return [column for column in labels if column in df.columns]


def _distribution(series: pd.Series, *, field_name: str, max_items: int = 25) -> list[ClassDistributionItem]:
    total = int(series.notna().sum())
    counts = series.dropna().value_counts(dropna=True).head(max_items)
    return [
        ClassDistributionItem(
            label=safe_label_value(label, field_name=field_name),
            count=int(count),
            percentage=_percentage(int(count), total),
        )
        for label, count in counts.items()
    ]


def _most_common_values(series: pd.Series, column_name: str, unique_count: int, max_items: int = 5) -> list[dict[str, Any]]:
    if unique_count > max(100, len(series) * 0.5):
        return [{"value": "<suppressed:high-cardinality>", "count": int(series.notna().sum())}]
    force_hash = is_sensitive_field_name(column_name)
    values = []
    for value, count in series.dropna().value_counts(dropna=True).head(max_items).items():
        values.append(
            {
                "value": safe_label_value(value, field_name=column_name, force_hash=force_hash),
                "count": int(count),
            }
        )
    return values


def _sample_strings(series: pd.Series, limit: int = 100) -> list[str]:
    return [str(value) for value in series.dropna().astype(str).head(limit).tolist()]


def _phone_like(value: str) -> bool:
    return any(len(re.sub(r"\D", "", match.group(0))) >= 8 for match in _PHONE_RE.finditer(value))


def _possible_identifier(column_name: str, series: pd.Series, row_count: int, declared: set[str]) -> bool:
    if column_name in declared or _IDENTIFIER_NAME_RE.search(column_name):
        return True
    non_null = int(series.notna().sum())
    if row_count and non_null and series.nunique(dropna=True) / max(non_null, 1) >= 0.98 and non_null >= max(10, row_count * 0.5):
        return True
    sample = _sample_strings(series)
    return any(_EMAIL_RE.search(value) or _phone_like(value) or _UUID_RE.match(value) for value in sample)


def _possible_leakage(column_name: str, label_columns: set[str]) -> bool:
    lowered = column_name.lower().replace(" ", "_").replace("?", "")
    if column_name in label_columns:
        return False
    if any(label.lower().replace(" ", "_").replace("?", "") in lowered for label in label_columns):
        return True
    if any(hint in lowered for hint in _LEAKAGE_NAME_HINTS):
        return True
    return any(hint in lowered for hint in _TIMING_HINTS)


def _column_issues(column_name: str, null_count: int, unique_count: int, row_count: int, possible_identifier: bool) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    if null_count:
        issues.append(
            issue(
                "missing_values",
                AuditSeverity.WARNING,
                "Column contains missing values",
                field_name=column_name,
                count=null_count,
            )
        )
    if row_count and unique_count == 1:
        issues.append(
            issue("constant_column", AuditSeverity.WARNING, "Column has one non-null value", field_name=column_name)
        )
    elif row_count and unique_count <= 2 and row_count >= 20:
        issues.append(
            issue("near_constant_column", AuditSeverity.INFO, "Column has very low cardinality", field_name=column_name)
        )
    if possible_identifier:
        issues.append(
            issue(
                "identifier_candidate",
                AuditSeverity.WARNING,
                "Column may contain direct or indirect identifiers",
                field_name=column_name,
            )
        )
    return issues


def _numeric_summary(series: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "minimum": _finite(numeric.min()),
        "maximum": _finite(numeric.max()),
        "mean": _finite(numeric.mean()),
        "median": _finite(numeric.median()),
        "standard_deviation": _finite(numeric.std(ddof=0)),
    }


def _range_issues(context: AuditContext, df: pd.DataFrame) -> tuple[dict[str, int], list[AuditIssue]]:
    invalid_counts: dict[str, int] = {}
    issues: list[AuditIssue] = []
    for column, (minimum, maximum) in context.options.expected_numeric_ranges.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        mask = pd.Series(False, index=df.index)
        if minimum is not None:
            mask = mask | (numeric < minimum)
        if maximum is not None:
            mask = mask | (numeric > maximum)
        count = int(mask.fillna(False).sum())
        if count:
            invalid_counts[column] = count
            issues.append(
                issue(
                    "invalid_numeric_range",
                    AuditSeverity.WARNING,
                    "Column contains values outside configured numeric range",
                    field_name=column,
                    count=count,
                    details={"minimum": minimum, "maximum": maximum},
                )
            )
    return invalid_counts, issues


def audit_tabular_dataset(context: AuditContext):
    df = load_tabular_source(context.source_path, context.dataset_config.file_format)
    audit_issues: list[AuditIssue] = []
    original_count = len(df)
    sampled = False
    if context.options.max_records and len(df) > context.options.max_records:
        df = df.sample(n=context.options.max_records, random_state=context.options.sample_seed).sort_index()
        sampled = True
        audit_issues.append(
            issue(
                "sampling_used",
                AuditSeverity.INFO,
                "Audit used deterministic row sampling",
                count=len(df),
                details={"source_record_count": original_count, "sample_seed": context.options.sample_seed},
            )
        )

    row_count = int(len(df))
    column_count = int(len(df.columns))
    label_columns = set(_label_columns(context, df))
    declared_identifiers = set(context.dataset_config.identifier_columns)
    declared_sensitive = set(context.dataset_config.sensitive_columns)

    missing_declared = [column for column in context.dataset_config.label_columns if column not in df.columns]
    if context.options.label_column and context.options.label_column not in df.columns:
        missing_declared.append(context.options.label_column)
    for column in missing_declared:
        audit_issues.append(
            issue(
                "declared_label_missing",
                AuditSeverity.WARNING,
                "Configured label column is not present in source",
                field_name=column,
            )
        )

    class_distribution = {column: _distribution(df[column], field_name=column) for column in sorted(label_columns)}
    duplicate_row_count = int(df.duplicated().sum())
    if duplicate_row_count:
        audit_issues.append(
            issue(
                "duplicate_rows",
                AuditSeverity.WARNING,
                "Exact duplicate rows are present",
                count=duplicate_row_count,
            )
        )

    invalid_range_counts, range_issues = _range_issues(context, df)
    audit_issues.extend(range_issues)

    columns: list[ColumnAudit] = []
    possible_identifier_columns: list[str] = []
    possible_sensitive_columns: list[str] = []
    possible_leakage_columns: list[str] = []

    for column_name in [str(column) for column in df.columns]:
        series = df[column_name]
        null_count = int(series.isna().sum())
        non_null_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))
        duplicate_count = max(non_null_count - unique_count, 0)
        identifier_candidate = _possible_identifier(column_name, series, row_count, declared_identifiers)
        sensitive_candidate = column_name in declared_sensitive or is_sensitive_field_name(column_name)
        leakage_candidate = _possible_leakage(column_name, label_columns)
        if identifier_candidate:
            possible_identifier_columns.append(column_name)
        if sensitive_candidate:
            possible_sensitive_columns.append(column_name)
        if leakage_candidate:
            possible_leakage_columns.append(column_name)

        column_issues = _column_issues(column_name, null_count, unique_count, row_count, identifier_candidate)
        if sensitive_candidate:
            column_issues.append(
                issue(
                    "sensitive_field_candidate",
                    AuditSeverity.INFO,
                    "Column name suggests sensitive demographic or contact information",
                    field_name=column_name,
                )
            )
        if leakage_candidate:
            column_issues.append(
                issue(
                    "leakage_candidate",
                    AuditSeverity.WARNING,
                    "Column name suggests possible target leakage, timing, or metadata leakage",
                    field_name=column_name,
                )
            )

        payload = {
            "column_name": column_name,
            "inferred_dtype": str(series.dtype),
            "non_null_count": non_null_count,
            "null_count": null_count,
            "null_percentage": _percentage(null_count, row_count),
            "unique_count": unique_count,
            "duplicate_count": duplicate_count,
            "most_common_values": _most_common_values(series, column_name, unique_count),
            "possible_identifier": identifier_candidate,
            "possible_sensitive_field": sensitive_candidate,
            "possible_target_leakage": leakage_candidate,
            "issues": column_issues,
        }
        payload.update(_numeric_summary(series))
        columns.append(ColumnAudit(**payload))

    audit_issues.extend(item for column in columns for item in column.issues)
    result = TabularAuditResult(
        row_count=row_count,
        column_count=column_count,
        columns=columns,
        duplicate_row_count=duplicate_row_count,
        class_distribution=class_distribution,
        invalid_range_counts=invalid_range_counts,
        possible_identifier_columns=sorted(set(possible_identifier_columns)),
        possible_sensitive_columns=sorted(set(possible_sensitive_columns)),
        possible_leakage_columns=sorted(set(possible_leakage_columns)),
        issues=sorted(audit_issues, key=lambda item: (item.code, item.field_name or "")),
    )
    notes = "Deterministic sampling was used." if sampled else "Full tabular source was audited."
    return build_report(
        context,
        modality_result_name="tabular_result",
        modality_result=result,
        issues=sorted(audit_issues, key=lambda item: (item.code, item.field_name or "")),
        notes=notes,
    )
