"""Safe reports for Phase 3B local candidate training."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.ml.common import hashing, paths
from app.ml.training.model_card import model_card_to_markdown
from app.ml.training.schemas import ModelCard, TrainingRunResult


def _safe_payload(value: Any) -> Any:
    if hasattr(value, "to_safe_dict"):
        return value.to_safe_dict()
    if isinstance(value, Mapping):
        return {str(key): _safe_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_payload(item) for item in value]
    return value


def _write_text(path: Path, text: str, *, overwrite: bool = False) -> Path:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def training_summary_json(result: TrainingRunResult, output_path: str | Path, *, overwrite: bool = False) -> Path:
    payload = result.to_safe_dict()
    return _write_text(Path(output_path), json.dumps(payload, indent=2, sort_keys=True) + "\n", overwrite=overwrite)


def training_summary_markdown(result: TrainingRunResult, output_path: str | Path, *, overwrite: bool = False) -> Path:
    payload = result.to_safe_dict()
    lines = [
        f"# Training Summary: {result.run_id}",
        "",
        f"- Status: `{result.status.value}`",
        f"- Framework version: `{result.training_framework_version}`",
        f"- Config hash: `{result.config_hash}`",
        f"- Manifest hash: `{result.dataset_reference.manifest_hash}`",
        f"- Model artifact: `{result.model_artifact_path}`",
        f"- Metrics: `{result.metrics_path}`",
        f"- Model card: `{result.model_card_path}`",
        "",
        "## Metrics",
        "```json",
        json.dumps(
            {
                "train": payload.get("train_metrics"),
                "validation": payload.get("validation_metrics"),
                "test": payload.get("test_metrics"),
                "selected_thresholds": payload.get("selected_thresholds"),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    return _write_text(Path(output_path), "\n".join(lines), overwrite=overwrite)


def metric_comparison_csv(rows: Sequence[Mapping[str, Any]], output_path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def artifact_inventory(run_dir: str | Path, output_path: str | Path, *, overwrite: bool = False) -> Path:
    run_path = Path(run_dir)
    rows = []
    for file_path in sorted(path for path in run_path.rglob("*") if path.is_file()):
        relative = file_path.resolve(strict=False).relative_to(paths.get_repository_root()).as_posix()
        rows.append({"path": relative, "sha256": hashing.sha256_file(file_path), "size_bytes": file_path.stat().st_size})
    return _write_text(Path(output_path), json.dumps(rows, indent=2, sort_keys=True) + "\n", overwrite=overwrite)


def model_card_markdown(card: ModelCard, output_path: str | Path, *, overwrite: bool = False) -> Path:
    return _write_text(Path(output_path), model_card_to_markdown(card), overwrite=overwrite)
