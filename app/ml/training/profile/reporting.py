"""Report and artifact writers for Profile baseline runs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from app.ml.common import hashing, paths
from app.ml.training.artifacts import prevent_overwrite
from app.ml.training.profile.constants import REQUIRED_MODEL_CARD_DISCLAIMER


def _resolve_output_path(path: str | Path) -> Path:
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
    return _write_text(_resolve_output_path(path), json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", overwrite=overwrite)


def write_markdown(path: str | Path, text: str, *, overwrite: bool = False) -> Path:
    return _write_text(_resolve_output_path(path), text.rstrip() + "\n", overwrite=overwrite)


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = False) -> Path:
    output_path = _resolve_output_path(path)
    rows = list(rows)
    prevent_overwrite(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path
    fieldnames = sorted({key for row in rows for key in row.keys()})
    tmp = output_path.with_name(f".{output_path.name}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})
    tmp.replace(output_path)
    return output_path


def save_joblib_artifact(obj: Any, path: str | Path, *, overwrite: bool = False) -> Path:
    output_path = _resolve_output_path(path)
    if output_path.suffix != ".joblib":
        raise ValueError("Profile model artifacts must use .joblib")
    prevent_overwrite(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.tmp")
    joblib.dump(obj, tmp)
    tmp.replace(output_path)
    return output_path


def build_profile_model_card(
    *,
    model_name: str,
    model_version: str,
    feature_set: str,
    selected_candidate: str | None,
    threshold_strategy: str | None,
    threshold: float | None,
    metrics: Mapping[str, Any],
) -> str:
    test_metrics = metrics.get("test", {}) if metrics else {}
    return f"""# Profile Depression Baseline Model Card

Model name: {model_name}

Model version: {model_version}

Intended use: Research baseline for self-reported depression classification.

Prohibited use: Suicide-risk diagnosis, autonomous intervention, counselor alerting, clinical diagnosis, or treatment recommendation.

Dataset: 101 Student Profile records with self-reported labels.

Split: train 71, validation 15, test 15 using the locked Profile split manifest.

Target: Self-reported depression (`target_depression`), not suicidal ideation and not suicide-risk ground truth.

Feature set: {feature_set}

Selected candidate: {selected_candidate or "none"}

Threshold policy: {threshold_strategy or "none"} at threshold {threshold if threshold is not None else "none"}. Selected with validation data only and not clinically validated.

Test recall: {test_metrics.get("recall")}

Test F1: {test_metrics.get("f1")}

Test false negatives: {test_metrics.get("false_negatives")}

Limitations:
- Tiny dataset with unstable metrics.
- Validation and test splits each contain only 15 records.
- Self-reported label, likely single-context sample, and no suicidal-ideation label.
- Sensitive-feature concerns remain; sensitive context is not used by the primary model.
- Anxiety and panic features can conceptually overlap with depression and are ablation-only, not the primary minimal baseline.
- No causal, clinical, or generalizable claim is supported.

Human oversight requirement: Human review is required for any research interpretation; this model must not operate autonomously.

{REQUIRED_MODEL_CARD_DISCLAIMER}
"""


def build_limitations_markdown() -> str:
    return """# Profile Baseline Limitations

- This is a research baseline for self-reported depression classification only.
- It is not a suicide-risk model and does not predict suicidal ideation.
- The dataset has 101 records; validation and test each have 15 records.
- Metrics, thresholds, calibration, and feature interpretations are unstable.
- Self-reported anxiety and panic-attack features may create conceptual overlap with the target.
- Sensitive contextual attributes are not part of the primary model.
- No participant-level predictions or explanations are written.
- No clinical, causal, or population-generalization claim is supported.
"""


def build_summary_markdown(summary: Mapping[str, Any]) -> str:
    selected = summary.get("selected_candidate") or {}
    test = summary.get("test_metrics") or {}
    return f"""# Profile Baseline Summary

Feature set: {summary.get("feature_set")}

Selected candidate: {selected.get("candidate_id", "none")}

Candidate-selection rationale: {summary.get("selection_rationale")}

Threshold strategy: {summary.get("threshold_strategy")}

Selected threshold: {summary.get("selected_threshold")}

Test confusion matrix: {test.get("confusion_matrix")}

False positives: {test.get("false_positives")}

False negatives: {test.get("false_negatives")}

Research-readiness decision: {summary.get("research_readiness_decision")}
"""


def file_inventory(files: Iterable[str | Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        resolved = _resolve_output_path(path)
        if not resolved.exists() or not resolved.is_file():
            continue
        try:
            relative = resolved.relative_to(paths.get_repository_root()).as_posix()
        except ValueError:
            relative = resolved.as_posix()
        entries.append(
            {
                "path": relative,
                "sha256": hashing.sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
        "file_hashes": {entry["path"]: entry["sha256"] for entry in entries},
    }


def build_artifact_manifest(
    *,
    run_id: str,
    model_name: str,
    model_version: str,
    feature_set: str,
    files: Iterable[str | Path],
    split_manifest_hash: str,
    source_fingerprint: str,
    preprocessing_artifact_hash: str,
    config_hash: str,
) -> dict[str, Any]:
    inventory = file_inventory(files)
    return {
        "manifest_version": "1.0.0",
        "run_id": run_id,
        "model_name": model_name,
        "model_version": model_version,
        "modality": "profile",
        "feature_set": feature_set,
        "files": [entry["path"] for entry in inventory["files"]],
        "file_hashes": inventory["file_hashes"],
        "split_manifest_hash": split_manifest_hash,
        "source_fingerprint": source_fingerprint,
        "preprocessing_artifact_hash": preprocessing_artifact_hash,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": False,
    }
