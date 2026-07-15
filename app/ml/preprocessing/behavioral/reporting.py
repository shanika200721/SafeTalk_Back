"""Privacy-safe report rendering for behavioral preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.preprocessing.behavioral.schemas import BehavioralPreprocessingReport


def create_behavioral_preprocessing_markdown(report: BehavioralPreprocessingReport, feature_schema: dict[str, Any]) -> str:
    lines = [
        "# Behavioral Preprocessing Report",
        "",
        f"- Preprocessing version: `{report.preprocessing_version}`",
        f"- Feature schema version: `{report.feature_schema_version}`",
        f"- Mapping version: `{report.mapping_version}`",
        f"- Source type: `{report.source_type}`",
        f"- Source records: {report.source_record_count}",
        f"- Output events: {report.output_event_count}",
        f"- Output sessions: {report.output_session_count}",
        f"- Participants: {report.participant_count}",
        f"- Date range: `{report.date_range}`",
        f"- Duplicate events: {report.duplicate_event_count}",
        f"- Baseline-eligible participants: {report.baseline_eligible_participant_count}",
        f"- Readiness: `{report.readiness_status}`",
        "",
        "## Features",
    ]
    for feature in feature_schema["features"]:
        lines.append(f"- `{feature['name']}`: {feature['units']}; {feature['non_clinical_status']}")
    lines.extend(["", "## Unavailable Features"])
    for feature in report.unavailable_features:
        lines.append(f"- `{feature}`")
    lines.extend(["", "## Privacy Warnings"])
    for warning in report.privacy_warnings:
        lines.append(f"- {warning}")
    lines.extend(
        [
            "",
            "## Limitations",
            "- Behavioral telemetry is indirect and is not a clinical biomarker by itself.",
            "- Slower typing, mouse hesitation, or missing sessions can have non-clinical explanations.",
            "- Device type, browser, disability, language, typing skill, and consent status can affect behavior.",
            "- Synthetic telemetry cannot validate predictive performance.",
            "- Behavioral features must not trigger autonomous crisis escalation.",
            "",
        ]
    )
    return "\n".join(lines)


def save_behavioral_report_json(report: BehavioralPreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report.to_safe_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def save_behavioral_report_markdown(report: BehavioralPreprocessingReport, feature_schema: dict[str, Any], output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(create_behavioral_preprocessing_markdown(report, feature_schema), encoding="utf-8")
    return resolved

