"""Privacy-safe reporting for Student Profile preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.schemas import FeatureSchema
from app.ml.preprocessing.profile.schemas import ProfilePreprocessingReport


def create_profile_preprocessing_markdown(report: ProfilePreprocessingReport, feature_schema: FeatureSchema) -> str:
    lines = [
        "# Student Profile Preprocessing Report",
        "",
        f"- Preprocessing version: `{report.preprocessing_version}`",
        f"- Feature schema version: `{report.feature_schema_version}`",
        f"- Mapping version: `{report.mapping_version}`",
        f"- Source rows: {report.source_row_count}",
        f"- Output rows: {report.output_row_count}",
        f"- Excluded rows: {report.excluded_row_count}",
        f"- Source fingerprint: `{report.source_fingerprint}`",
        f"- Generated at: `{report.generated_at.isoformat()}`",
        "",
        "## Target Distribution",
    ]
    for label, count in report.target_distribution.items():
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Canonical Features"])
    for name in report.feature_columns:
        lines.append(f"- `{name}`")

    lines.extend(["", "## Excluded Columns"])
    for name in report.excluded_columns:
        lines.append(f"- `{name}`")

    lines.extend(["", "## Sensitive Context Columns In Current Output"])
    if report.sensitive_context_columns:
        for name in report.sensitive_context_columns:
            lines.append(f"- `{name}`")
    else:
        lines.append("- none")

    lines.extend(["", "## Leakage Checks"])
    lines.append(f"- Feature leakage detected: {report.leakage_checks['feature_leakage']['has_leakage']}")
    lines.append(f"- Treatment-seeking decision: {report.leakage_checks['treatment_seeking_decision']}")
    identifiers = report.leakage_checks["identifier_detection"]["identifier_candidates"]
    lines.append(f"- Identifier candidates: {', '.join(identifiers) if identifiers else 'none'}")

    lines.extend(["", "## Missing Values"])
    if report.missing_value_summary:
        for column, count in report.missing_value_summary.items():
            lines.append(f"- `{column}`: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Feature Schema Notes", feature_schema.notes or ""])
    lines.extend(["", "## Warnings And Limitations"])
    for warning in report.warnings:
        lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def save_profile_report_json(report: ProfilePreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report.to_safe_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def save_profile_report_markdown(
    report: ProfilePreprocessingReport,
    feature_schema: FeatureSchema,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(create_profile_preprocessing_markdown(report, feature_schema), encoding="utf-8")
    return resolved
