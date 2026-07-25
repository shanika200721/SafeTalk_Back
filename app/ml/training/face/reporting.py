"""Reporting and artifact writers for Phase 3I Face baselines."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from app.ml.common import hashing, paths
from app.ml.training.artifacts import prevent_overwrite
from app.ml.training.face.constants import REQUIRED_MODEL_CARD_DISCLAIMER


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _write_text(path: Path, text: str, *, overwrite: bool) -> Path:
    prevent_overwrite(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def write_json(path: str | Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    return _write_text(_resolve(path), json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", overwrite=overwrite)


def write_markdown(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    return _write_text(_resolve(path), text.rstrip() + "\n", overwrite=overwrite)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> Path:
    output = _resolve(path)
    rows = list(rows)
    prevent_overwrite(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return output
    fieldnames = sorted({key for row in rows for key in row})
    tmp = output.with_name(f".{output.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    tmp.replace(output)
    return output


def save_joblib_artifact(obj: Any, path: str | Path, *, overwrite: bool = False) -> Path:
    output = _resolve(path)
    if output.suffix != ".joblib":
        raise ValueError("Face model artifacts must use .joblib")
    prevent_overwrite(output, overwrite=overwrite)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    joblib.dump(obj, tmp)
    tmp.replace(output)
    return output


def file_inventory(files: Iterable[str | Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        resolved = _resolve(path)
        if not resolved.exists() or not resolved.is_file():
            continue
        try:
            relative = resolved.relative_to(paths.get_repository_root()).as_posix()
        except ValueError:
            relative = resolved.as_posix()
        entries.append({"path": relative, "sha256": hashing.sha256_file(resolved), "size_bytes": resolved.stat().st_size})
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
        "file_hashes": {entry["path"]: entry["sha256"] for entry in entries},
    }


def build_face_model_card(summary: Mapping[str, Any]) -> str:
    selected = summary.get("selected_candidate") or {}
    test = summary.get("test_metrics") or {}
    return f"""# Face Emotion Research Baseline Model Card

Model name: {summary.get("model_name")}

Model version: {summary.get("model_version")}

Intended research use: seven-class facial emotion classification (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`).

Prohibited use: depression diagnosis, suicide-risk assessment, counselor alerting, autonomous intervention, identity recognition, demographic inference, production webcam scoring, treatment recommendation, or clinical decision support.

Dataset: Facial Emotion images, 48x48 grayscale.

Split: Phase 3G leakage-safe v2 split. Original source split is metadata only and is not a predictive feature.

Duplicate remediation: exact duplicate controls are applied; 155 cross-label records remain quarantined and excluded.

Review status: Phase 3H review complete with unresolved items. Reviewer independence is recorded as `reviewer_independence_unverified`.

Subject IDs: unavailable. Subject-independent evaluation is not established.

Demographic limitations: demographic composition is unavailable; demographic fairness is not established.

Class imbalance: disgust has much lower support than the other classes.

Selected candidate: {selected.get("candidate_id", "none")}

Validation macro F1: {(summary.get("validation_metrics") or {}).get("macro_f1")}

Test macro F1: {test.get("macro_f1")}

Test balanced accuracy: {test.get("balanced_accuracy")}

Biometric privacy: no face recognition, embeddings, identity matching, raw image export, thumbnails, or participant-level heatmaps are included.

Human oversight: required for any research interpretation.

{REQUIRED_MODEL_CARD_DISCLAIMER}
"""


def build_summary_markdown(summary: Mapping[str, Any]) -> str:
    selected = summary.get("selected_candidate") or {}
    val = summary.get("validation_metrics") or {}
    test = summary.get("test_metrics") or {}
    return f"""# Phase 3I Face Baseline Summary

Selected candidate: {selected.get("candidate_id", "none")}

Selection rationale: {summary.get("selection_rationale")}

Feature representation: {summary.get("feature_set")}

Validation macro F1: {val.get("macro_f1")}

Validation macro recall: {val.get("macro_recall")}

Validation minimum-class recall: {val.get("minimum_class_recall")}

Test macro F1: {test.get("macro_f1")}

Test weighted F1: {test.get("weighted_f1")}

Test balanced accuracy: {test.get("balanced_accuracy")}

Research-readiness decision: {summary.get("research_readiness_decision")}
"""


def limitations_markdown() -> str:
    return """# Face Baseline Dataset and Governance Limitations

- Facial emotion is not depression.
- Depression is not suicidal ideation.
- This model must not be used for autonomous crisis decisions.
- Subject identifiers are unavailable; subject-independent performance is not established.
- Reviewer independence was not independently verified.
- 55 unresolved cross-label groups remain quarantined.
- Perceptual near-duplicate risk may remain.
- Demographic information is unavailable.
- Acted-expression and dataset-domain bias may not generalize to webcam or real-world conditions.
- No face recognition, identity inference, embeddings, or clinical interpretation are supported.
"""


def artifact_manifest(
    *,
    run_id: str,
    model_name: str,
    model_version: str,
    files: Iterable[str | Path],
    split_manifest_hash: str,
    source_fingerprint: str,
    config_hash: str,
) -> dict[str, Any]:
    inventory = file_inventory(files)
    return {
        "manifest_version": "1.0.0",
        "run_id": run_id,
        "model_name": model_name,
        "model_version": model_version,
        "modality": "face",
        "files": [entry["path"] for entry in inventory["files"]],
        "file_hashes": inventory["file_hashes"],
        "split_manifest_hash": split_manifest_hash,
        "source_fingerprint": source_fingerprint,
        "config_hash": config_hash,
        "active": False,
        "registered": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

