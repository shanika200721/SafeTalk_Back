"""Report and artifact writers for Speech baseline runs."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib

from app.ml.common import hashing, paths
from app.ml.training.artifacts import prevent_overwrite
from app.ml.training.speech.constants import REQUIRED_MODEL_CARD_DISCLAIMER


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
        raise ValueError("Speech model artifacts must use .joblib")
    prevent_overwrite(output_path, overwrite=overwrite)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.tmp")
    joblib.dump(obj, tmp)
    tmp.replace(output_path)
    return output_path


def build_speech_model_card(*, model_name: str, model_version: str, selected_candidate: str | None, feature_set: str, metrics: Mapping[str, Any]) -> str:
    test_metrics = metrics.get("test", {}) if metrics else {}
    return f"""# Speech Emotion Acoustic Baseline Model Card

Model name: {model_name}

Model version: {model_version}

Intended use: Research eight-class acted speech-emotion classification using deterministic acoustic features only.

Prohibited use: Depression diagnosis, suicide-risk assessment, autonomous suicide-prevention, clinical decision-making, counselor alerting, treatment recommendation, or production voice inference.

Target labels: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

Feature set: {feature_set}

Selected candidate: {selected_candidate or "none"}

Split: Locked speaker-isolated Speech v1 train/validation/test split. Candidate selection used validation data only.

Test macro F1: {test_metrics.get("macro_f1")}

Test weighted F1: {test_metrics.get("weighted_f1")}

Test balanced accuracy: {test_metrics.get("balanced_accuracy")}

Limitations:
- Acted emotion corpora are not natural distress recordings.
- Emotion is not depression, and depression is not suicide risk.
- Corpus, device, accent, language, microphone, and sample-rate variation can create shortcut behavior.
- TESS and SAVEE have very low speaker counts, so corpus-specific split coverage is limited.
- Voice data is biometric and sensitive even after acoustic feature extraction.
- No transcription, pretrained speech embeddings, deep learning, or production recordings were used.
- Human oversight is required for research interpretation.

{REQUIRED_MODEL_CARD_DISCLAIMER}
"""


def build_dataset_limitations_markdown() -> str:
    return """# Speech Baseline Dataset Limitations

- The task is eight-class acted speech-emotion classification only.
- Emotion labels are not depression labels and are not suicide-risk labels.
- The corpora differ by language/accent context, recording device, microphone, and sample rate.
- Corpus distribution is imbalanced; TESS and SAVEE have too few speakers for stable three-way corpus-specific splitting.
- Corpus identity is retained only for stratified evaluation and domain-shift analysis, not as a primary predictive feature.
- No raw audio, raw speaker IDs, source paths, transcripts, alerts, or production records are included in reports.
"""


def build_summary_markdown(summary: Mapping[str, Any]) -> str:
    selected = summary.get("selected_candidate") or {}
    test = summary.get("test_metrics") or {}
    return f"""# Speech Baseline Summary

Feature set: {summary.get("feature_set")}

Selected candidate: {selected.get("candidate_id", "none")}

Selection rationale: {summary.get("selection_rationale")}

Validation macro F1: {(summary.get("validation_metrics") or {}).get("macro_f1")}

Validation macro recall: {(summary.get("validation_metrics") or {}).get("macro_recall")}

Test macro F1: {test.get("macro_f1")}

Test weighted F1: {test.get("weighted_f1")}

Test balanced accuracy: {test.get("balanced_accuracy")}

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
        entries.append({"path": relative, "sha256": hashing.sha256_file(resolved), "size_bytes": resolved.stat().st_size})
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
    feature_file_hash: str,
    corpus_fingerprints: Mapping[str, str],
    config_hash: str,
) -> dict[str, Any]:
    inventory = file_inventory(files)
    return {
        "manifest_version": "1.0.0",
        "run_id": run_id,
        "model_name": model_name,
        "model_version": model_version,
        "modality": "speech",
        "feature_set": feature_set,
        "files": [entry["path"] for entry in inventory["files"]],
        "file_hashes": inventory["file_hashes"],
        "split_manifest_hash": split_manifest_hash,
        "source_fingerprint": source_fingerprint,
        "preprocessing_artifact_hash": preprocessing_artifact_hash,
        "feature_file_hash": feature_file_hash,
        "corpus_fingerprints": dict(corpus_fingerprints),
        "config_hash": config_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "active": False,
    }

