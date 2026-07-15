"""Privacy-safe reporting for Daily Mood preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.common import paths
from app.ml.common.schemas import FeatureSchema
from app.ml.preprocessing.mood.schemas import MoodPreprocessingReport


def create_mood_preprocessing_markdown(report: MoodPreprocessingReport, feature_schema: FeatureSchema) -> str:
    lines = [
        "# Daily Mood Preprocessing Report",
        "",
        f"- Preprocessing version: `{report.preprocessing_version}`",
        f"- Feature schema version: `{report.feature_schema_version}`",
        f"- Mapping version: `{report.mapping_version}`",
        f"- Source records: {report.source_record_count}",
        f"- Output records: {report.output_record_count}",
        f"- Participants: {report.participant_count}",
        f"- Date range: `{report.date_range.get('min')}` to `{report.date_range.get('max')}`",
        f"- Generated at: `{report.generated_at.isoformat()}`",
        "",
        "## Feature Columns",
    ]
    for name in report.feature_columns:
        lines.append(f"- `{name}`")

    lines.extend(["", "## Excluded Columns"])
    for name in report.excluded_columns:
        lines.append(f"- `{name}`")

    lines.extend(["", "## Missing Values"])
    if report.missing_value_summary:
        for column, count in report.missing_value_summary.items():
            lines.append(f"- `{column}`: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Duplicate And Temporal Checks"])
    lines.append(f"- Duplicate records reported: {report.duplicate_summary.get('duplicate_count', 0)}")
    lines.append(f"- Temporal order violations: {report.temporal_order_violations}")

    lines.extend(["", "## Experimental Trend Score"])
    lines.append(
        "- `mood_trend_score_0_100` is non-clinical and experimental: "
        "35% current low mood, 20% deterioration from previous mood, 20% recent low-mood ratio, "
        "10% missing-day ratio, 10% sudden deterioration flag, and 5% negative 7-observation slope."
    )

    lines.extend(["", "## Warnings And Limitations"])
    for warning in report.warnings:
        lines.append(f"- {warning}")

    lines.append("")
    return "\n".join(lines)


def save_mood_report_json(report: MoodPreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report.to_safe_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def save_mood_report_markdown(
    report: MoodPreprocessingReport,
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
    resolved.write_text(create_mood_preprocessing_markdown(report, feature_schema), encoding="utf-8")
    return resolved
