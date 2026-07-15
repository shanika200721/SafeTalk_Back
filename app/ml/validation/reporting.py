"""Reporting helpers for Phase 2 validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from app.ml.common import paths
from app.ml.validation.constants import DEFAULT_VALIDATION_OUTPUT_FILES
from app.ml.validation.schemas import CrossModalityValidationReport, ModalityReadiness


def _resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    target = Path(output_dir) if output_dir is not None else paths.get_generated_reports_root() / "phase2_validation"
    if not target.is_absolute():
        target = paths.get_repository_root() / target
    target = target.resolve(strict=False)
    paths.assert_not_raw_dataset_path(target)
    return target


def _write_json(path: Path, payload, *, overwrite: bool) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_phase2_validation_json(
    report: CrossModalityValidationReport,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    path = _resolve_output_dir(output_dir) / DEFAULT_VALIDATION_OUTPUT_FILES["report_json"]
    return _write_json(path, report.to_safe_dict(), overwrite=overwrite)


def load_phase2_validation_report(report_path: str | Path) -> CrossModalityValidationReport:
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    return CrossModalityValidationReport.parse_obj(payload)


def create_readiness_matrix(report: CrossModalityValidationReport) -> list[dict[str, str]]:
    rows = []
    for modality in report.modalities:
        rows.append(
            {
                "modality": modality.modality,
                "dataset_name": modality.dataset_name,
                "readiness": str(modality.readiness_classification),
                "split_readiness": str(modality.split_readiness),
                "model_training_readiness": str(modality.model_training_readiness),
                "warnings": str(len(modality.warnings)),
                "blockers": str(len(modality.blockers)),
            }
        )
    rows.append(
        {
            "modality": "fusion",
            "dataset_name": "disconnected-offline-datasets",
            "readiness": "blocked_pending_data",
            "split_readiness": "blocked",
            "model_training_readiness": "blocked",
            "warnings": "0",
            "blockers": "1",
        }
    )
    return rows


def save_readiness_matrix(
    report: CrossModalityValidationReport,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    path = _resolve_output_dir(output_dir) / DEFAULT_VALIDATION_OUTPUT_FILES["readiness_matrix"]
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing readiness matrix: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = create_readiness_matrix(report)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def create_phase2_markdown_summary(report: CrossModalityValidationReport) -> str:
    lines = [
        "# Phase 2 Cross-Modality Validation Report",
        "",
        "## Executive Summary",
        "",
        f"- Validation version: `{report.validation_version}`",
        f"- Readiness policy version: `{report.readiness_policy_version}`",
        f"- Phase 2 completion decision: **{report.phase2_completion_status}**",
        f"- Next phase recommendation: {report.next_phase_recommendation}",
        "",
        "## Modality Readiness Matrix",
        "",
        "| Modality | Dataset | Readiness | Split readiness | Model-training readiness | Warnings | Blockers |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in create_readiness_matrix(report):
        lines.append(
            f"| {row['modality']} | {row['dataset_name']} | {row['readiness']} | "
            f"{row['split_readiness']} | {row['model_training_readiness']} | {row['warnings']} | {row['blockers']} |"
        )
    lines.extend(
        [
            "",
            "## Check Counts",
            "",
            f"- Passed checks: {report.passed_checks}",
            f"- Warning checks: {report.warning_checks}",
            f"- Failed checks: {report.failed_checks}",
            f"- Blocked checks: {report.blocked_checks}",
            "",
            "## Warnings And Blockers",
            "",
        ]
    )
    for modality in report.modalities:
        if modality.warnings or modality.blockers:
            lines.append(f"### {modality.modality}")
            for item in modality.warnings[:20]:
                lines.append(f"- Warning: {item}")
            for item in modality.blockers[:20]:
                lines.append(f"- Blocker: {item}")
            lines.append("")
    lines.extend(
        [
            "## Leakage Findings",
            "",
        ]
    )
    for modality in report.modalities:
        for item in modality.leakage_findings + modality.duplicate_findings:
            lines.append(f"- {modality.modality}: {item}")
    lines.extend(
        [
            "",
            "## Privacy Findings",
            "",
        ]
    )
    privacy_rows = [(m.modality, finding) for m in report.modalities for finding in m.privacy_findings]
    if privacy_rows:
        for modality, finding in privacy_rows:
            lines.append(f"- {modality}: {finding}")
    else:
        lines.append("- No raw participant data, raw text, audio, image payloads, identifiers, or telemetry payloads were intentionally included.")
    lines.extend(
        [
            "",
            "## Multimodal-Fusion Limitation",
            "",
            "- Current offline datasets are disconnected and have no common participant key.",
            "- Supervised multimodal fusion training is not valid from these datasets.",
            "- Later fusion must use production modality predictions, synthetic engineering data, or a future ethically collected aligned pilot dataset.",
            "",
            "## Recommended Next Actions",
            "",
        ]
    )
    for item in report.global_recommendations:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def save_phase2_validation_markdown(
    report: CrossModalityValidationReport,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    path = _resolve_output_dir(output_dir) / DEFAULT_VALIDATION_OUTPUT_FILES["report_markdown"]
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(create_phase2_markdown_summary(report), encoding="utf-8")
    return path


def save_supporting_reports(
    report: CrossModalityValidationReport,
    inventory: Iterable,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    output = _resolve_output_dir(output_dir)
    paths_written = {
        "readiness_matrix": save_readiness_matrix(report, output, overwrite=overwrite),
        "artifact_inventory": _write_json(
            output / DEFAULT_VALIDATION_OUTPUT_FILES["artifact_inventory"],
            [item.to_safe_dict() if hasattr(item, "to_safe_dict") else item for item in inventory],
            overwrite=overwrite,
        ),
        "blockers": _write_json(
            output / DEFAULT_VALIDATION_OUTPUT_FILES["blockers"],
            {
                modality.modality: modality.blockers
                for modality in report.modalities
                if modality.blockers
            },
            overwrite=overwrite,
        ),
        "next_actions": _write_json(
            output / DEFAULT_VALIDATION_OUTPUT_FILES["next_actions"],
            {"global_recommendations": report.global_recommendations, "next_phase_recommendation": report.next_phase_recommendation},
            overwrite=overwrite,
        ),
    }
    return paths_written
