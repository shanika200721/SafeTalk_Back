from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models.database_models import ModelRegistry, ModalityPrediction


def register_model(
    db: Session,
    *,
    model_name: str,
    modality: str,
    version: str,
    framework: str,
    artifact_path: str,
    preprocessing_path: Optional[str] = None,
    dataset_version: Optional[str] = None,
    feature_schema_version: Optional[str] = None,
    metrics_json: Optional[dict] = None,
    thresholds_json: Optional[dict] = None,
    is_active: bool = False,
) -> ModelRegistry:
    model = ModelRegistry(
        model_name=model_name,
        modality=modality,
        version=version,
        framework=framework,
        artifact_path=artifact_path,
        preprocessing_path=preprocessing_path,
        dataset_version=dataset_version,
        feature_schema_version=feature_schema_version,
        metrics_json=metrics_json,
        thresholds_json=thresholds_json,
        is_active=False,
    )
    db.add(model)
    db.flush()

    if is_active:
        activate_model_version(db, model_name=model_name, modality=modality, version=version)
        db.refresh(model)

    return model


def activate_model_version(
    db: Session,
    *,
    model_name: str,
    modality: str,
    version: str,
) -> ModelRegistry:
    model = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_name == model_name,
            ModelRegistry.modality == modality,
            ModelRegistry.version == version,
        )
        .one_or_none()
    )
    if model is None:
        raise ValueError(f"Model version not found: {model_name}/{modality}/{version}")

    now = datetime.utcnow()
    (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_name == model_name,
            ModelRegistry.modality == modality,
            ModelRegistry.is_active == True,
        )
        .update({"is_active": False, "updated_at": now}, synchronize_session="fetch")
    )
    db.flush()

    model.is_active = True
    model.updated_at = now
    db.flush()
    return model


def deactivate_model_version(
    db: Session,
    *,
    model_name: str,
    modality: str,
    version: str,
) -> ModelRegistry:
    model = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_name == model_name,
            ModelRegistry.modality == modality,
            ModelRegistry.version == version,
        )
        .one_or_none()
    )
    if model is None:
        raise ValueError(f"Model version not found: {model_name}/{modality}/{version}")

    model.is_active = False
    model.updated_at = datetime.utcnow()
    db.flush()
    return model


def get_active_model(db: Session, *, modality: str, model_name: Optional[str] = None) -> Optional[ModelRegistry]:
    query = db.query(ModelRegistry).filter(
        ModelRegistry.modality == modality,
        ModelRegistry.is_active == True,
    )
    if model_name is not None:
        query = query.filter(ModelRegistry.model_name == model_name)
    return query.order_by(ModelRegistry.updated_at.desc()).first()


def delete_model_version(db: Session, model_registry_id: int) -> None:
    referenced = (
        db.query(ModalityPrediction.id)
        .filter(ModalityPrediction.model_registry_id == model_registry_id)
        .first()
    )
    if referenced:
        raise ValueError("Cannot delete a model version referenced by predictions")

    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_registry_id).one_or_none()
    if model is None:
        raise ValueError(f"Model version not found: {model_registry_id}")

    db.delete(model)
    db.flush()
