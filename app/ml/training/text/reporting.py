"""Report and artifact writers for Text baseline runs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from app.ml.common import hashing, paths
from app.ml.training.artifacts import prevent_overwrite
from app.ml.training.text.constants import REQUIRED_MODEL_CARD_DISCLAIMER


def _resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _repo_relative(path: str | Path) -> str:
    resolved = _resolve_output_path(path)
    try:
        return resolved.relative_to(paths.get_repository_root()).as_posix()
    except ValueError:
        return resolved.as_posix()


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
    tmp = output_path.with_name(f".{output_path.name}.tmp")
    if not rows:
        tmp.write_text("", encoding="utf-8")
        tmp.replace(output_path)
        return output_path
    fieldnames = sorted({key for row in rows for key in row.keys()})
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
        raise ValueError("Text model artifacts must use .joblib")
    prevent_overwrite(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.tmp")
    joblib.dump(obj, tmp)
    tmp.replace(output_path)
    return output_path


def build_text_model_card(
    *,
    model_name: str,
    model_version: str,
    selected_candidate: str | None,
    vectorizer_name: str | None,
    metrics: Mapping[str, Any],
) -> str:
    test_metrics = metrics.get("test", {}) if metrics else {}
    suicidal = test_metrics.get("suicidal_class") or {}
    return f"""# Text Classification Baseline Model Card

Model name: {model_name}

Model version: {model_version}

Intended use: Research baseline for four-class mental-health text classification.

Prohibited use: Suicide-risk diagnosis, autonomous intervention, counselor alerting, clinical diagnosis, production chat scoring, or treatment recommendation.

Dataset origin: Authoritative Text Classification source `mental_heath_unbanlanced.csv`, canonicalized into the locked Phase 2 Text preprocessing artifact.

Labels: `anxiety`, `depression`, `normal`, and `suicidal` are source annotations, not clinical labels.

Duplicate handling: Exact duplicate and text-hash groups are isolated by the locked split. Conflicting duplicate records are quarantined and excluded.

Limitations: User grouping is incomplete because `Unique_ID` is missing for many records. Reference/test overlap is reported as an aggregate limitation. Social-media or dataset text may not match private production chat. Privacy normalization is imperfect and no raw text is included in reports.

Selected candidate: {selected_candidate or "none"}

Vectorizer: {vectorizer_name or "none"}

Test macro F1: {test_metrics.get("macro_f1")}

Test weighted F1: {test_metrics.get("weighted_f1")}

Test balanced accuracy: {test_metrics.get("balanced_accuracy")}

Test suicidal recall: {suicidal.get("recall")}

Test suicidal false negatives: {suicidal.get("false_negatives")}

Human oversight requirement: Human review is required for any research interpretation. This model must not operate autonomously.

{REQUIRED_MODEL_CARD_DISCLAIMER}
"""


def build_dataset_limitations_markdown() -> str:
    return """# Text Baseline Dataset Limitations

- Source labels are not clinical diagnoses or suicide-risk ground truth.
- `Unique_ID` is incomplete, so author-level leakage cannot be fully ruled out.
- Exact duplicate and text-hash groups are isolated, but near-duplicate semantic leakage may remain.
- Reference/test overlap is documented as an aggregate limitation.
- Domain mismatch may exist between source text and private production chats.
- Privacy normalization is imperfect; no raw or normalized text is included in reports.
- Suicidal-class false negatives are reported, but this is not a crisis detection system.
"""


def build_summary_markdown(summary: Mapping[str, Any]) -> str:
    selected = summary.get("selected_candidate") or {}
    test = summary.get("test_metrics") or {}
    suicidal = test.get("suicidal_class") or {}
    return f"""# Text Baseline Summary

Feature set: {summary.get("feature_set")}

Selected candidate: {selected.get("candidate_id", "none")}

Selection rationale: {summary.get("selection_rationale")}

Validation macro F1: {summary.get("validation_metrics", {}).get("macro_f1")}

Validation suicidal recall: {summary.get("validation_metrics", {}).get("suicidal_class", {}).get("recall")}

Test macro F1: {test.get("macro_f1")}

Test weighted F1: {test.get("weighted_f1")}

Test balanced accuracy: {test.get("balanced_accuracy")}

Test suicidal false negatives: {suicidal.get("false_negatives")}

Research-readiness decision: {summary.get("research_readiness_decision")}
"""


def file_inventory(files: Iterable[str | Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        resolved = _resolve_output_path(path)
        if not resolved.exists() or not resolved.is_file():
            continue
        entries.append(
            {
                "path": _repo_relative(resolved),
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
        "modality": "text",
        "files": [entry["path"] for entry in inventory["files"]],
        "file_hashes": inventory["file_hashes"],
        "split_manifest_hash": split_manifest_hash,
        "source_fingerprint": source_fingerprint,
        "preprocessing_artifact_hash": preprocessing_artifact_hash,
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": False,
    }

