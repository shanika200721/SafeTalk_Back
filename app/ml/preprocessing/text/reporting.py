"""Privacy-safe report rendering for text preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.common import paths
from app.ml.preprocessing.text.schemas import TextPreprocessingReport


def create_text_preprocessing_markdown(report: TextPreprocessingReport) -> str:
    lines = [
        "# Text Preprocessing Report",
        "",
        f"- Preprocessing version: `{report.preprocessing_version}`",
        f"- Feature schema version: `{report.feature_schema_version}`",
        f"- Label mapping version: `{report.label_mapping_version}`",
        f"- Privacy ruleset version: `{report.privacy_ruleset_version}`",
        f"- Source fingerprint: `{report.source_fingerprint}`",
        f"- Source records: {report.source_record_count}",
        f"- Output records: {report.output_record_count}",
        f"- Excluded records: {report.excluded_record_count}",
        f"- Exact duplicate groups: {report.exact_duplicate_group_count}",
        f"- Conflicting duplicate groups: {report.conflicting_duplicate_group_count}",
        f"- Near-duplicate candidates: {report.near_duplicate_candidate_count}",
        f"- Generated at: `{report.generated_at.isoformat()}`",
        "",
        "## Label Distribution Before",
    ]
    for label, count in report.label_distribution_before.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Label Distribution After"])
    for label, count in report.label_distribution_after.items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Privacy Replacements"])
    for name, count in report.privacy_replacement_summary.items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Language Summary"])
    for name, count in report.language_summary.items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Leakage Checks"])
    for name, value in report.leakage_checks.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(["", "## Warnings And Limitations"])
    for warning in report.warnings:
        lines.append(f"- {warning}")
    lines.append("")
    return "\n".join(lines)


def save_text_report_json(report: TextPreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(report.to_safe_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved


def save_text_report_markdown(report: TextPreprocessingReport, output_path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(output_path).resolve(strict=False)
    paths.assert_not_raw_dataset_path(resolved)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(create_text_preprocessing_markdown(report), encoding="utf-8")
    return resolved
