from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import ModelRegistry, User
from app.routes.auth import get_current_user
from app.schemas import ModelRegistryResponse, ModelVerificationResponse
from app.services.model_registry import (
    activate_model_version,
    apply_verification_result,
    deactivate_model_version,
    discover_runtime_candidates,
    get_model_by_id,
    verify_model_artifact,
)


router = APIRouter(prefix="/api/models", tags=["Model Governance"])


def _require_admin(user: User) -> None:
    if user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can manage runtime model governance",
        )


def _model_or_404(db: Session, model_id: int) -> ModelRegistry:
    model = get_model_by_id(db, model_id)
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model registry entry not found")
    return model


def _verification_response(model: ModelRegistry, result) -> ModelVerificationResponse:
    payload = result.to_dict()
    payload["model_id"] = model.id
    return ModelVerificationResponse(**payload)


@router.get("", response_model=list[ModelRegistryResponse])
def list_models(
    modality: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    query = db.query(ModelRegistry)
    if modality:
        query = query.filter(ModelRegistry.modality == modality)
    return query.order_by(ModelRegistry.modality, ModelRegistry.model_name, ModelRegistry.version).all()


@router.post("/discover", response_model=list[ModelRegistryResponse])
def discover_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    models = discover_runtime_candidates(db)
    db.commit()
    return models


@router.get("/{model_id}", response_model=ModelRegistryResponse)
def get_model(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    return _model_or_404(db, model_id)


@router.post("/{model_id}/verify", response_model=ModelVerificationResponse)
def verify_model(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    model = _model_or_404(db, model_id)
    result = verify_model_artifact(model)
    apply_verification_result(db, model, result)
    db.commit()
    db.refresh(model)
    return _verification_response(model, result)


@router.get("/{model_id}/verification", response_model=ModelVerificationResponse)
def get_model_verification(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    model = _model_or_404(db, model_id)
    payload = model.verification_json or {
        "passed": False,
        "failure_code": "NOT_VERIFIED",
        "failure_message_safe": "Model verification has not been run.",
        "metadata_complete": False,
        "smoke_test_status": "not_run",
        "activation_eligible": False,
        "details": {},
    }
    payload["model_id"] = model.id
    return ModelVerificationResponse(**payload)


@router.post("/{model_id}/activate", response_model=ModelRegistryResponse)
def activate_model(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    model = _model_or_404(db, model_id)
    try:
        activated = activate_model_version(
            db,
            model_name=model.model_name,
            modality=model.modality,
            version=model.version,
        )
        activated.approved_by = current_user.username
        db.commit()
        db.refresh(activated)
        return activated
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/{model_id}/deactivate", response_model=ModelRegistryResponse)
def deactivate_model(
    model_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(current_user)
    model = _model_or_404(db, model_id)
    deactivated = deactivate_model_version(
        db,
        model_name=model.model_name,
        modality=model.modality,
        version=model.version,
    )
    db.commit()
    db.refresh(deactivated)
    return deactivated
