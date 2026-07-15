"""JSON serialization helpers for Phase 2 ML schemas."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Type, TypeVar

from pydantic.v1 import BaseModel, ValidationError

from app.ml.common import paths
from app.ml.common.schemas import DatasetConfig, DatasetFingerprint, FeatureSchema, PreprocessingConfig, SplitManifest


SchemaT = TypeVar("SchemaT", bound=BaseModel)


def _resolve_output_path(output_path: str | os.PathLike[str] | Path) -> Path:
    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _assert_json_output_allowed(output_path: Path) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    generated_root = paths.get_generated_root()
    ml_configs_root = paths.get_ml_research_root() / "configs"
    ml_manifests_root = paths.get_ml_research_root() / "manifests"

    allowed_roots = (generated_root, ml_configs_root, ml_manifests_root)
    if not any(paths.is_path_inside(root, output_path) for root in allowed_roots):
        raise ValueError(
            "JSON schema output must be under generated/, "
            "ml-research/configs/, or ml-research/manifests/"
        )
    return output_path


def _schema_payload(schema: BaseModel) -> dict:
    if hasattr(schema, "to_safe_dict"):
        return schema.to_safe_dict()
    return json.loads(schema.json())


def save_schema_json(schema: BaseModel, output_path: str | os.PathLike[str] | Path, *, overwrite: bool = False) -> Path:
    """Save a schema as deterministic JSON using an atomic replace."""
    resolved = _assert_json_output_allowed(_resolve_output_path(output_path))
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing schema JSON: {resolved}")

    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = _schema_payload(schema)
    json_text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    temp_path = resolved.with_name(f".{resolved.name}.tmp")
    temp_path.write_text(json_text, encoding="utf-8")
    temp_path.replace(resolved)
    return resolved


def _load_json(path: str | os.PathLike[str] | Path) -> dict:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _load_schema(path: str | os.PathLike[str] | Path, schema_type: Type[SchemaT]) -> SchemaT:
    payload = _load_json(path)
    try:
        return schema_type.parse_obj(payload)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse schema {path}: {exc}") from exc


def load_dataset_config(path: str | os.PathLike[str] | Path) -> DatasetConfig:
    return _load_schema(path, DatasetConfig)


def load_dataset_fingerprint(path: str | os.PathLike[str] | Path) -> DatasetFingerprint:
    return _load_schema(path, DatasetFingerprint)


def load_preprocessing_config(path: str | os.PathLike[str] | Path) -> PreprocessingConfig:
    return _load_schema(path, PreprocessingConfig)


def load_split_manifest(path: str | os.PathLike[str] | Path) -> SplitManifest:
    return _load_schema(path, SplitManifest)


def load_feature_schema(path: str | os.PathLike[str] | Path) -> FeatureSchema:
    return _load_schema(path, FeatureSchema)
