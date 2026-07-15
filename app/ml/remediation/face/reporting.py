"""Reporting helpers for face remediation and revised splits."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from app.ml.common import paths
from app.ml.common.hashing import sha256_file


def write_json(path: str | Path, payload: Any, *, overwrite: bool = False) -> Path:
    target = Path(path)
    paths.assert_not_raw_dataset_path(target)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], *, fieldnames: list[str], overwrite: bool = False) -> Path:
    target = Path(path)
    paths.assert_not_raw_dataset_path(target)
    if target.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def artifact_inventory(paths_by_name: dict[str, Path]) -> list[dict[str, Any]]:
    inventory = []
    repo_root = paths.get_repository_root()
    for name, path in sorted(paths_by_name.items()):
        if not path.exists():
            continue
        try:
            relative = path.resolve(strict=False).relative_to(repo_root).as_posix()
        except ValueError:
            relative = path.name
        inventory.append(
            {
                "artifact_name": name,
                "relative_path": relative,
                "sha256": sha256_file(path, allow_outside_project=True),
                "size_bytes": path.stat().st_size,
            }
        )
    return inventory


def remediation_markdown(report: dict[str, Any]) -> str:
    warnings = "\n".join(f"- {item}" for item in report.get("warnings", []))
    return (
        "# Phase 3G Face Duplicate Remediation\n\n"
        f"- Remediation version: {report['remediation_version']}\n"
        f"- Deduplicated view version: {report['view_version']}\n"
        f"- Duplicate policy version: {report['policy_version']}\n"
        f"- Source records: {report['source_record_count']}\n"
        f"- Retained records: {report['retained_record_count']}\n"
        f"- Exact duplicate groups: {report['duplicate_group_count']}\n"
        f"- Same-label duplicate groups: {report['same_label_group_count']}\n"
        f"- Same-label duplicate records excluded: {report['excluded_same_label_duplicate_count']}\n"
        f"- Cross-split duplicate groups: {report['cross_split_group_count']}\n"
        f"- Cross-label conflict groups: {report['cross_label_group_count']}\n"
        f"- Cross-label records quarantined: {report['quarantined_cross_label_record_count']}\n\n"
        "## Research Status\n\n"
        "This artifact is research-only. It does not train, activate, or deploy a facial model.\n\n"
        "## Warnings\n\n"
        f"{warnings}\n"
    )


def split_markdown(report: dict[str, Any]) -> str:
    counts = report.get("split_counts", {})
    return (
        "# Phase 3G Face v2 Split Report\n\n"
        f"- Split version: {report.get('split_version')}\n"
        f"- Strategy: {report.get('strategy')}\n"
        f"- Train records: {counts.get('train', 0)}\n"
        f"- Validation records: {counts.get('validation', 0)}\n"
        f"- Test records: {counts.get('test', 0)}\n"
        f"- Record overlap count: {report.get('record_overlap_count', 0)}\n"
        f"- Exact image-hash overlap count: {report.get('image_hash_overlap_count', 0)}\n"
        f"- Duplicate-group overlap count: {report.get('duplicate_overlap_count', 0)}\n\n"
        "The original predefined train/test folders are retained only as metadata. Subject-independent "
        "splitting remains impossible because subject identifiers are unavailable.\n"
    )


def readiness_markdown(report: dict[str, Any]) -> str:
    blockers = "\n".join(f"- {item}" for item in report.get("remaining_blockers", [])) or "- None for exact duplicate remediation."
    next_actions = "\n".join(f"- {item}" for item in report.get("recommended_next_actions", []))
    return (
        "# Phase 3G Face Readiness Validation\n\n"
        f"- Readiness classification: {report.get('readiness_classification')}\n"
        f"- Source fingerprint verified: {report.get('source_fingerprint_verified')}\n"
        f"- Exact hash leakage count: {report.get('exact_hash_overlap_count')}\n"
        f"- Duplicate group leakage count: {report.get('duplicate_group_overlap_count')}\n"
        f"- Deterministic replay passed: {report.get('deterministic_replay_passed')}\n\n"
        "## Remaining Blockers\n\n"
        f"{blockers}\n\n"
        "## Next Actions\n\n"
        f"{next_actions}\n"
    )

