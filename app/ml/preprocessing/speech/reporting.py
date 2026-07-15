"""Privacy-safe report rendering for speech preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.common import paths
from app.ml.preprocessing.speech.schemas import SpeechPreprocessingReport


def create_speech_preprocessing_markdown(report: SpeechPreprocessingReport) -> str:
    lines = [
        "# Speech Preprocessing Report",
        "",
        f"- Preprocessing version: `{report.preprocessing_version}`",
        f"- Feature schema version: `{report.feature_schema_version}`",
        f"- Label mapping version: `{report.label_mapping_version}`",
        f"- Corpus mapping version: `{report.corpus_mapping_version}`",
        f"- Source files: {report.source_file_count}",
        f"- Readable files: {report.readable_file_count}",
        f"- Unreadable files: {report.unreadable_file_count}",
        f"- Output records: {report.output_record_count}",
        f"- Excluded records: {report.excluded_record_count}",
        f"- Safe speaker count: {report.speaker_count}",
        f"- Generated at: `{report.generated_at.isoformat()}`",
        "",
        "## Corpus Distribution",
    ]
    for label, count in report.corpus_distribution.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Label Distribution"])
    for label, count in report.label_distribution.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Sample Rates"])
    for label, count in report.sample_rate_distribution.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Channels"])
    for label, count in report.channel_distribution.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Duration Summary"])
    for label, value in report.duration_summary.items():
        lines.append(f"- `{label}`: {value}")
    lines.extend(["", "## Duplicate Summary"])
    for label, value in report.duplicate_summary.items():
        lines.append(f"- `{label}`: `{value}`")
    lines.extend(["", "## Feature Missing Summary"])
    for label, value in report.feature_missing_summary.items():
        lines.append(f"- `{label}`: {value}")
    lines.extend(["", "## Warnings And Limitations"])
    for warning in report.warnings:
        lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def save_speech_report_json(report: SpeechPreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report.to_safe_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def save_speech_report_markdown(report: SpeechPreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(create_speech_preprocessing_markdown(report), encoding="utf-8")
    return resolved

