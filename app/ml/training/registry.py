"""Candidate-only integration with the existing model registry service."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from sqlalchemy.orm import Session

from app.models.database_models import ModelRegistry
from app.services import model_registry as registry_service
from app.ml.training.schemas import ArtifactManifest, TrainingConfig


def build_model_registry_payload(
    *,
    config: TrainingConfig,
    artifact_manifest: ArtifactManifest,
    metrics_json: Optional[Mapping[str, Any]] = None,
    thresholds_json: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    model_file = next((path for path in artifact_manifest.files if path.endswith(".joblib")), None)
    if model_file is None:
        raise ValueError("candidate bundle has no model artifact")
    return {
        "model_name": config.model_name,
        "modality": config.modality,
        "version": config.model_version,
        "framework": config.framework,
        "artifact_path": model_file,
        "preprocessing_path": None,
        "dataset_version": config.dataset_version,
        "feature_schema_version": config.feature_schema_version,
        "metrics_json": dict(metrics_json or {}),
        "thresholds_json": dict(thresholds_json or {}),
        "is_active": False,
    }


def validate_registry_compatibility(payload: Mapping[str, Any]) -> None:
    if payload.get("is_active") is True:
        raise ValueError("Phase 3B may register candidates only; activation is forbidden")
    for key in ("model_name", "modality", "version", "framework", "artifact_path"):
        if not payload.get(key):
            raise ValueError(f"registry payload missing required field: {key}")


def register_candidate_model(db: Session, payload: Mapping[str, Any]) -> ModelRegistry:
    validate_registry_compatibility(payload)
    return registry_service.register_model(db, **dict(payload, is_active=False))


def retrieve_registered_candidate(db: Session, *, model_name: str, modality: str, version: str) -> Optional[ModelRegistry]:
    return (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_name == model_name,
            ModelRegistry.modality == modality,
            ModelRegistry.version == version,
        )
        .one_or_none()
    )


def confirm_model_not_active(db: Session, *, model_name: str, modality: str, version: str) -> bool:
    model = retrieve_registered_candidate(db, model_name=model_name, modality=modality, version=version)
    return model is not None and model.is_active is False
