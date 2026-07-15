"""Privacy-safe report rendering for facial emotion preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.common import paths
from app.ml.preprocessing.face.schemas import FacePreprocessingReport


def create_face_preprocessing_markdown(report: FacePreprocessingReport) -> str:
    lines = [
        "# Facial Emotion Preprocessing Report",
        "",
        f"- Preprocessing version: `{report.preprocessing_version}`",
        f"- Feature schema version: `{report.feature_schema_version}`",
        f"- Label mapping version: `{report.label_mapping_version}`",
        f"- Image policy version: `{report.image_policy_version}`",
        f"- Source fingerprint: `{report.source_fingerprint}`",
        f"- Source files: {report.source_file_count}",
        f"- Readable files: {report.readable_file_count}",
        f"- Unreadable files: {report.unreadable_file_count}",
        f"- Output records: {report.output_record_count}",
        f"- Excluded records: {report.excluded_record_count}",
        f"- Corrupt files: {report.corrupt_file_count}",
        f"- Zero-byte files: {report.zero_byte_file_count}",
        f"- Duplicate groups: {report.duplicate_group_count}",
        f"- Cross-split duplicate groups: {report.cross_split_duplicate_count}",
        f"- Cross-label duplicate groups: {report.cross_label_duplicate_count}",
        f"- Subject count: {report.subject_count if report.subject_count is not None else 'not available'}",
        f"- Generated at: `{report.generated_at.isoformat()}`",
        "",
        "## Split Distribution",
    ]
    for label, count in report.split_distribution.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Label Distribution"])
    for label, count in report.label_distribution.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Width Summary"])
    for label, value in report.width_summary.items():
        lines.append(f"- `{label}`: {value}")
    lines.extend(["", "## Height Summary"])
    for label, value in report.height_summary.items():
        lines.append(f"- `{label}`: {value}")
    lines.extend(["", "## Color Modes"])
    for label, count in report.color_mode_distribution.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Formats"])
    for label, count in report.format_distribution.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Warnings And Limitations"])
    for warning in report.warnings:
        lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def save_face_report_json(report: FacePreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report.to_safe_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def save_face_report_markdown(report: FacePreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(create_face_preprocessing_markdown(report), encoding="utf-8")
    return resolved

