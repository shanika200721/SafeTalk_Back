"""Schema-only export helpers for synthetic Phase 4A pilot artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from app.ml.common import paths
from app.ml.pilot.consent import consent_matrix
from app.ml.pilot.schemas import to_plain_dict


def _resolve_output(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    root = paths.get_repository_root()
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise ValueError("pilot exports must use repository-relative output paths") from exc
    return paths.assert_not_raw_dataset_path(candidate)


def _write_json(path: Path, payload: Any, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing pilot export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain_dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing pilot export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: to_plain_dict(value) for key, value in row.items()} for row in rows])
    return path


def export_pilot_participants(participants: Sequence[Any], output_dir: str | Path, overwrite: bool = False) -> Path:
    return _write_json(_resolve_output(output_dir) / "pilot_participants.json", [item.to_dict() for item in participants], overwrite)


def export_pilot_sessions(sessions: Sequence[Any], output_dir: str | Path, overwrite: bool = False) -> Path:
    return _write_json(_resolve_output(output_dir) / "pilot_sessions.json", [item.to_dict() for item in sessions], overwrite)


def export_pilot_modality_manifest(records: Sequence[Any], output_dir: str | Path, overwrite: bool = False) -> Path:
    return _write_json(_resolve_output(output_dir) / "pilot_modality_records.json", [item.to_dict() for item in records], overwrite)


def export_pilot_outcomes(outcomes: Sequence[Any], output_dir: str | Path, overwrite: bool = False) -> Path:
    safe = []
    for outcome in outcomes:
        row = outcome.to_dict()
        row["notes"] = None
        safe.append(row)
    return _write_json(_resolve_output(output_dir) / "pilot_outcomes.json", safe, overwrite)


def export_pilot_consent_summary(consents: Sequence[Any], output_dir: str | Path, overwrite: bool = False) -> Path:
    rows: List[Dict[str, Any]] = []
    for consent in consents:
        rows.extend(consent_matrix(consent))
    return _write_csv(_resolve_output(output_dir) / "pilot_consent_matrix.csv", rows, overwrite)


def export_pilot_missingness_report(missingness: Dict[str, Any], output_dir: str | Path, overwrite: bool = False) -> Path:
    return _write_json(_resolve_output(output_dir) / "pilot_missingness_report.json", missingness, overwrite)


def export_pilot_alignment_report(alignment: Dict[str, Any], output_dir: str | Path, overwrite: bool = False) -> Path:
    return _write_json(_resolve_output(output_dir) / "pilot_alignment_report.json", alignment, overwrite)


def export_pilot_safety_summary(summary: Dict[str, Any], output_dir: str | Path, overwrite: bool = False) -> Path:
    return _write_json(_resolve_output(output_dir) / "pilot_safety_summary.json", summary, overwrite)
