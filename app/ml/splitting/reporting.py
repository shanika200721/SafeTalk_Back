"""Privacy-safe reporting helpers for Phase 3A splits."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from app.ml.common import paths
from app.ml.splitting.constants import DEFAULT_OUTPUT_FILES, PHASE3A_REPORT_FILES, SPLIT_DESIGN_VERSION, SPLIT_MANIFEST_VERSION
from app.ml.splitting.schemas import ModalitySplitManifest, SplitDesignReport


def resolve_output_dir(output_dir: str | Path) -> Path:
    target = Path(output_dir)
    if not target.is_absolute():
        target = paths.get_repository_root() / target
    target = target.resolve(strict=False)
    paths.assert_not_raw_dataset_path(target)
    return target


def write_json(path: str | Path, payload, *, overwrite: bool = False) -> Path:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def create_split_markdown_report(manifest: ModalitySplitManifest, report: SplitDesignReport, manifest_hash: str) -> str:
    lines = [
        f"# Phase 3A {manifest.modality.title()} Split Report",
        "",
        f"- Split design version: `{SPLIT_DESIGN_VERSION}`",
        f"- Split manifest version: `{SPLIT_MANIFEST_VERSION}`",
        f"- Strategy: `{manifest.split_strategy}`",
        f"- Manifest hash: `{manifest_hash}`",
        f"- Source count: {report.source_count}",
        f"- Included count: {report.included_count}",
        f"- Excluded count: {report.excluded_count}",
        "",
        "## Split Counts",
        "",
        "| Split | Count | Distribution |",
        "|---|---:|---|",
    ]
    for split in ("train", "validation", "test"):
        distribution = report.label_distributions.get(split, {})
        lines.append(f"| {split} | {report.split_counts.get(split, 0)} | `{distribution}` |")
    lines.extend(["", "## Leakage Checks", ""])
    for key, value in sorted(report.leakage_checks.items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Grouping And Duplicates", ""])
    for key, value in sorted(report.grouping_summary.items()):
        lines.append(f"- {key}: `{value}`")
    for key, value in sorted(report.duplicate_handling.items()):
        lines.append(f"- duplicate_{key}: `{value}`")
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in report.warnings:
            lines.append(f"- {warning}")
    if report.limitations:
        lines.extend(["", "## Scientific Limitations", ""])
        for limitation in report.limitations:
            lines.append(f"- {limitation}")
    return "\n".join(lines).rstrip() + "\n"


def save_split_outputs(
    *,
    modality: str,
    output_dir: str | Path,
    manifest: ModalitySplitManifest,
    assignments_csv_writer,
    report: SplitDesignReport,
    exclusions: Mapping[str, str],
    manifest_hash: str,
    extra_json: Mapping[str, object] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    output = resolve_output_dir(output_dir)
    filenames = {key: value.format(modality=modality) for key, value in DEFAULT_OUTPUT_FILES.items()}
    paths_written: dict[str, Path] = {}
    manifest_path = output / filenames["manifest"]
    report_json_path = output / filenames["report_json"]
    report_md_path = output / filenames["report_markdown"]
    exclusions_path = output / filenames["exclusions"]
    assignments_path = output / filenames["assignments"]

    paths_written["manifest"] = write_json(manifest_path, manifest.to_safe_dict(), overwrite=overwrite)
    paths_written["assignments"] = assignments_csv_writer(assignments_path, overwrite=overwrite)
    paths_written["report_json"] = write_json(
        report_json_path,
        report.to_safe_dict() | {"manifest_hash": manifest_hash},
        overwrite=overwrite,
    )
    if report_md_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {report_md_path}")
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(create_split_markdown_report(manifest, report, manifest_hash), encoding="utf-8")
    paths_written["report_markdown"] = report_md_path
    paths_written["exclusions"] = write_json(exclusions_path, dict(sorted(exclusions.items())), overwrite=overwrite)
    for name, payload in (extra_json or {}).items():
        paths_written[name] = write_json(output / name, payload, overwrite=overwrite)
    return paths_written


def create_phase3a_markdown_summary(payload: Mapping[str, object]) -> str:
    lines = [
        "# Phase 3A Split Validation",
        "",
        f"- Split design version: `{SPLIT_DESIGN_VERSION}`",
        f"- Split manifest version: `{SPLIT_MANIFEST_VERSION}`",
        f"- Completion decision: **{payload.get('completion_decision', 'unknown')}**",
        "",
        "| Modality | Ready | Strategy | Train | Validation | Test | Excluded |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in payload.get("modalities", []):
        if isinstance(row, Mapping):
            lines.append(
                f"| {row.get('modality')} | {row.get('ready')} | {row.get('strategy')} | "
                f"{row.get('train_count')} | {row.get('validation_count')} | {row.get('test_count')} | {row.get('excluded_count')} |"
            )
    lines.extend(["", "## Remaining Blockers", ""])
    blockers = payload.get("blockers", {})
    if blockers:
        for modality, items in blockers.items():
            lines.append(f"- {modality}: {items}")
    else:
        lines.append("- No Phase 3A blockers for profile, text, or speech split manifests.")
    lines.extend(["", "## Next Actions", ""])
    for item in payload.get("next_actions", []):
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def save_phase3a_reports(
    payload: Mapping[str, object],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    output = resolve_output_dir(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths_written = {
        "validation_json": write_json(output / PHASE3A_REPORT_FILES["validation_json"], payload, overwrite=overwrite),
        "blockers": write_json(output / PHASE3A_REPORT_FILES["blockers"], payload.get("blockers", {}), overwrite=overwrite),
        "next_actions": write_json(
            output / PHASE3A_REPORT_FILES["next_actions"],
            {"next_actions": payload.get("next_actions", [])},
            overwrite=overwrite,
        ),
    }
    md_path = output / PHASE3A_REPORT_FILES["validation_markdown"]
    if md_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {md_path}")
    md_path.write_text(create_phase3a_markdown_summary(payload), encoding="utf-8")
    paths_written["validation_markdown"] = md_path

    rows = list(payload.get("modalities", []))
    matrix_path = output / PHASE3A_REPORT_FILES["matrix"]
    if matrix_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {matrix_path}")
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["modality", "ready", "strategy", "train_count", "validation_count", "test_count", "excluded_count", "manifest_hash"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    paths_written["matrix"] = matrix_path
    return paths_written
