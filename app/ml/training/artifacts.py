"""Local research artifact management for Phase 3B candidate models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import joblib

from app.ml.common import hashing, paths
from app.ml.training.constants import SAFE_ARTIFACT_EXTENSIONS
from app.ml.training.schemas import ArtifactManifest, TrainingConfig, utc_now


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(paths.get_repository_root()).as_posix()
    except ValueError:
        return path.resolve(strict=False).relative_to(paths.get_model_root()).as_posix()


def prevent_overwrite(path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = Path(path).resolve(strict=False)
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {resolved}")
    return resolved


def create_run_directory(
    *,
    modality: str,
    model_name: str,
    model_version: str,
    run_id: str,
    overwrite: bool = False,
) -> Path:
    relative = Path(modality) / model_name / model_version / run_id
    run_dir = paths.ensure_model_directory(relative)
    if any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Run directory already contains artifacts: {run_dir}")
    return run_dir


def _atomic_write_text(path: Path, text: str, *, overwrite: bool = False) -> Path:
    if path.suffix.lower() not in SAFE_ARTIFACT_EXTENSIONS:
        raise ValueError(f"Unsupported artifact extension: {path.suffix}")
    prevent_overwrite(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def _save_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    return _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", overwrite=overwrite)


def save_model_artifact(model: Any, run_dir: str | Path, *, filename: str = "model.joblib", overwrite: bool = False) -> Path:
    path = Path(run_dir) / filename
    if path.suffix.lower() != ".joblib":
        raise ValueError("scikit-learn model artifacts must use .joblib")
    prevent_overwrite(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    joblib.dump(model, tmp)
    tmp.replace(path)
    return path


def save_preprocessor_artifact(preprocessor: Any, run_dir: str | Path, *, filename: str = "preprocessor.joblib", overwrite: bool = False) -> Path:
    path = Path(run_dir) / filename
    if path.suffix.lower() != ".joblib":
        raise ValueError("preprocessor artifacts must use .joblib")
    prevent_overwrite(path, overwrite=overwrite)
    tmp = path.with_name(f".{path.name}.tmp")
    joblib.dump(preprocessor, tmp)
    tmp.replace(path)
    return path


def save_metrics_json(metrics: Mapping[str, Any], run_dir: str | Path, *, overwrite: bool = False) -> Path:
    return _save_json(Path(run_dir) / "metrics.json", metrics, overwrite=overwrite)


def save_training_config(config: TrainingConfig, run_dir: str | Path, *, overwrite: bool = False) -> Path:
    return _save_json(Path(run_dir) / "training_config.json", config.to_safe_dict(), overwrite=overwrite)


def save_artifact_manifest(manifest: ArtifactManifest, run_dir: str | Path, *, overwrite: bool = False) -> Path:
    return _save_json(Path(run_dir) / "artifact_manifest.json", manifest.to_safe_dict(), overwrite=overwrite)


def verify_artifact_hashes(manifest: ArtifactManifest | Mapping[str, Any]) -> bool:
    payload = manifest.to_safe_dict() if hasattr(manifest, "to_safe_dict") else dict(manifest)
    for relative_path, expected_hash in payload["file_hashes"].items():
        candidate = paths.get_repository_root() / relative_path
        if not candidate.exists():
            candidate = paths.get_model_root() / relative_path
        if hashing.sha256_file(candidate) != expected_hash:
            return False
    return True


def load_artifact_manifest(path: str | Path) -> ArtifactManifest:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    resolved = candidate.resolve(strict=False)
    if not paths.is_path_inside(paths.get_model_root(), resolved):
        raise ValueError("artifact manifests can only be loaded from trusted MODEL_ROOT paths")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ArtifactManifest.parse_obj(payload)


def create_candidate_bundle(
    *,
    run_dir: str | Path,
    config: TrainingConfig,
    run_id: str,
    split_manifest_hash: str,
    metrics_path: str | Path,
    model_path: str | Path,
    model_card_path: str | Path,
    extra_files: list[str | Path] | None = None,
    overwrite: bool = False,
) -> ArtifactManifest:
    files = [Path(model_path), Path(metrics_path), Path(model_card_path), Path(run_dir) / "training_config.json"]
    files.extend(Path(path) for path in (extra_files or []))
    file_paths = [_repo_relative(path) for path in files if path.exists()]
    manifest = ArtifactManifest(
        run_id=run_id,
        model_name=config.model_name,
        model_version=config.model_version,
        modality=config.modality,
        files=file_paths,
        file_hashes={relative: hashing.sha256_file(paths.get_repository_root() / relative) for relative in file_paths},
        dataset_version=config.dataset_version,
        preprocessing_version=config.preprocessing_version,
        feature_schema_version=config.feature_schema_version,
        split_manifest_hash=split_manifest_hash,
        config_hash=hashing.hash_json_data(config.to_safe_dict()),
        created_at=utc_now(),
    )
    save_artifact_manifest(manifest, run_dir, overwrite=overwrite)
    return manifest
