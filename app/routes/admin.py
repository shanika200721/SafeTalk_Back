from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.database_models import (
    AdminAuditLog,
    AdminReport,
    Assessment,
    ChatMessage,
    CounselorAssignment,
    CounselorNote,
    CounselorProfile,
    CounselorProfileAudit,
    CounselorReview,
    DASS21Assessment,
    DailyCheckIn,
    ModalityPrediction,
    ModelRegistry,
    ProfileAssessment,
    Resource,
    RiskAssessment,
    SafeTalkBotMessage,
    SupportContactAction,
    SupportContact,
    SystemSetting,
    University,
    User,
    UserRole,
    WorkerJob,
)
from app.routes.auth import get_current_user
from app.security import hash_password
from app.services.model_registry import (
    _resolve_repo_path,
    activate_model_version as activate_registry_model,
    apply_verification_result,
    calculate_sha256,
    deactivate_model_version as deactivate_registry_model,
    discover_runtime_candidates,
    verify_model_artifact,
)
from app.services.support_contacts import (
    AVAILABILITY_STATUSES,
    audit_profile_change,
    has_unresolved_reviews,
    normalize_e164,
    preserve_university_assignment,
    public_id,
    require_admin,
    serialize_profile,
    serialize_support_contact,
    transfer_active_student_assignments,
    validate_time_range,
)


router = APIRouter(prefix="/api/admin", tags=["Admin Directory"])

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_TYPES = {
    "university_summary",
    "counselor_summary",
    "assessment_summary",
    "fusion_summary",
    "support_summary",
    "usage_summary",
}
RESOURCE_TYPES = {"article", "video", "meditation", "breathing", "animation", "audio", "exercise", "game"}
SETTING_SECTIONS = (
    "authentication",
    "consent",
    "fusion",
    "support_contacts",
    "safetalk",
    "resource_settings",
    "maintenance_mode",
    "storage",
    "feature_flags",
)


def _today_bounds() -> tuple[datetime, datetime]:
    start = datetime.combine(date.today(), datetime.min.time())
    return start, start + timedelta(days=1)


def _audit(
    db: Session,
    *,
    actor: User,
    action: str,
    entity_type: str,
    entity_id: str | int | None = None,
    old_value: dict | None = None,
    new_value: dict | None = None,
    request: Request | None = None,
    status_value: str = "success",
) -> AdminAuditLog:
    audit = AdminAuditLog(
        audit_id=public_id("aaudit"),
        user_id=actor.id if actor else None,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        old_value_json=_json_safe(old_value),
        new_value_json=_json_safe(new_value),
        ip_address=request.client.host if request and request.client else None,
        status=status_value,
        privacy_scope="administrative_summary",
    )
    db.add(audit)
    db.flush()
    return audit


def _json_safe(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UserRole):
        return value.value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _count(query) -> int:
    return int(query.count() or 0)


def _storage_usage() -> dict:
    roots = [REPO_ROOT / "suicideprevention.db", REPO_ROOT / "generated", REPO_ROOT / "ml_models"]
    total = 0
    entries = []
    for root in roots:
        size = 0
        if root.is_file():
            size = root.stat().st_size
        elif root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    size += path.stat().st_size
        total += size
        entries.append({"name": root.name, "bytes": size, "megabytes": round(size / 1024 / 1024, 2)})
    return {"bytes": total, "megabytes": round(total / 1024 / 1024, 2), "locations": entries}


def _system_status(db: Session) -> dict:
    active_models = db.query(ModelRegistry).filter(ModelRegistry.is_active.is_(True)).count()
    failed_jobs = db.query(WorkerJob).filter(WorkerJob.status == "failed").count()
    return {
        "status": "operational" if not failed_jobs else "attention_required",
        "database": "reachable",
        "active_models": active_models,
        "failed_jobs": int(failed_jobs or 0),
        "privacy_mode": "summary_only",
    }


def _active_assignment_for_student(db: Session, student_id: int) -> CounselorAssignment | None:
    return (
        db.query(CounselorAssignment)
        .options(joinedload(CounselorAssignment.counselor))
        .filter(CounselorAssignment.student_id == student_id, CounselorAssignment.active.is_(True))
        .order_by(CounselorAssignment.assigned_date.desc())
        .first()
    )


def _serialize_user_admin(db: Session, user: User) -> dict:
    assignment = _active_assignment_for_student(db, user.id) if user.role == UserRole.STUDENT else None
    status_label = "suspended" if user.suspended_at else ("active" if user.is_active else "inactive")
    return {
        "id": user.id,
        "name": user.full_name,
        "full_name": user.full_name,
        "username": user.username,
        "email": user.email,
        "university_id": user.university_id,
        "university": user.university.university_name if user.university else None,
        "role": user.role.value,
        "status": status_label,
        "is_active": user.is_active,
        "registration_date": user.created_at,
        "last_login": user.last_login_at,
        "assigned_counselor": assignment.counselor.full_name if assignment and assignment.counselor else None,
        "assigned_counselor_id": assignment.counselor_id if assignment else None,
    }


def _serialize_model(model: ModelRegistry) -> dict:
    metrics = model.metrics_json or {}
    accuracy = (
        metrics.get("accuracy")
        or metrics.get("test_accuracy")
        or metrics.get("weighted_accuracy")
        or metrics.get("macro_f1")
    )
    return {
        "id": model.id,
        "profile": model.modality,
        "name": model.model_name,
        "modality": model.modality,
        "version": model.version,
        "status": model.status,
        "hash": model.artifact_sha256,
        "framework": model.framework,
        "accuracy": accuracy,
        "created": model.created_at,
        "approved": model.approved_at,
        "approved_by": model.approved_by,
        "active": model.is_active,
        "verified": model.verification_status == "passed",
        "verification_status": model.verification_status,
        "verification_message": model.verification_message,
    }


def _runtime_status_item(db: Session, model: ModelRegistry | None, modality: str) -> dict:
    since = datetime.utcnow() - timedelta(hours=24)
    predictions_last_24h = (
        db.query(ModalityPrediction)
        .filter(ModalityPrediction.modality == modality, ModalityPrediction.created_at >= since)
        .count()
    )
    last_success = (
        db.query(ModalityPrediction)
        .filter(ModalityPrediction.modality == modality, ModalityPrediction.status == "succeeded")
        .order_by(ModalityPrediction.created_at.desc())
        .first()
    )
    last_failure = (
        db.query(ModalityPrediction)
        .filter(ModalityPrediction.modality == modality, ModalityPrediction.status.in_(["failed", "unavailable", "rejected"]))
        .order_by(ModalityPrediction.created_at.desc())
        .first()
    )
    if model is None:
        blocking_reason = "no_registered_model" if modality != "behavioral" else "no_verified_behavioral_model"
        return {
            "modality": modality,
            "model_id": None,
            "registry_status": "unregistered",
            "active": False,
            "artifact_available": False,
            "hash_valid": False,
            "loader_status": "not_available",
            "preprocessing_status": "not_available",
            "smoke_test_status": "not_run",
            "last_successful_inference": last_success.created_at if last_success else None,
            "last_failed_inference": last_failure.created_at if last_failure else None,
            "failure_reason": blocking_reason,
            "blocking_reason": blocking_reason,
            "predictions_last_24h": int(predictions_last_24h or 0),
            "average_inference_duration": None,
            "fusion_eligibility": modality in {"profile", "text"} and bool(last_success),
            "fusion_status": "eligible" if modality in {"profile", "text"} and bool(last_success) else "excluded_no_validated_model",
            "health_state": "unavailable",
            "technical_status": "unavailable",
            "research_reliability": "not_evaluated" if modality == "behavioral" else "experimental",
        }
    artifact_available = False
    hash_valid = False
    try:
        artifact_path = _resolve_repo_path(model.artifact_path)
        artifact_available = artifact_path.exists()
        hash_valid = bool(artifact_available and model.artifact_sha256 and calculate_sha256(artifact_path) == model.artifact_sha256)
    except Exception:
        artifact_available = False
        hash_valid = False
    verification = model.verification_json or {}
    activation_eligible = bool(verification.get("activation_eligible") or (model.verification_status == "passed" and model.modality in {"profile", "text"}))
    failure_reason = model.verification_failure_code or verification.get("failure_code")
    if model.modality == "face" and not activation_eligible:
        failure_reason = failure_reason or "insufficient_model_reliability"
    if model.modality == "speech" and not activation_eligible:
        failure_reason = failure_reason or "browser_audio_runtime_not_verified"
    if not artifact_available:
        health_state = "missing_artifact"
    elif failure_reason in {"PREPROCESSOR_MISSING", "runtime_preprocessor_not_approved"}:
        health_state = "incompatible_preprocessing"
    elif failure_reason == "LABEL_MAPPING_MISSING":
        health_state = "missing_label_mapping"
    elif failure_reason == "SMOKE_TEST_FAILED":
        health_state = "smoke_test_failed"
    elif model.is_active and activation_eligible:
        health_state = "verified_active"
    elif model.verification_status == "passed":
        health_state = "verified_inactive"
    else:
        health_state = "inactive"
    if health_state in {"verified_active", "verified_inactive"}:
        technical_status = "verified_active" if model.is_active and activation_eligible else "technically_verified"
    elif health_state == "incompatible_preprocessing":
        technical_status = "incompatible_preprocessing"
    elif artifact_available:
        technical_status = "artifact_present"
    else:
        technical_status = "unavailable"
    if modality in {"profile", "text"}:
        research_reliability = "validated" if model.verification_status == "passed" else "experimental"
    elif modality == "face":
        research_reliability = "low_reliability"
    elif modality == "behavioral":
        research_reliability = "not_evaluated"
    else:
        research_reliability = "experimental"
    if model.modality == "speech":
        fusion_eligible = False
        fusion_status = "excluded_no_approved_risk_mapping"
    elif model.modality == "face":
        fusion_eligible = False
        fusion_status = "excluded_low_reliability"
    else:
        fusion_eligible = model.modality in {"profile", "text"} and model.is_active and activation_eligible
        fusion_status = "eligible" if fusion_eligible else "excluded"
    return {
        "modality": modality,
        "model_id": model.id,
        "model_name": model.model_name,
        "model_version": model.version,
        "preprocessing_version": model.preprocessing_version,
        "registry_status": model.status,
        "verification_status": model.verification_status,
        "active": model.is_active,
        "artifact_available": artifact_available,
        "hash_valid": hash_valid,
        "loader_status": "available" if model.modality in {"profile", "text"} and model.is_active else ("available_with_ffmpeg" if model.modality == "speech" and activation_eligible else "not_approved"),
        "preprocessing_status": "available" if activation_eligible else "not_approved",
        "smoke_test_status": verification.get("smoke_test_status", "not_run"),
        "last_successful_inference": last_success.created_at if last_success else None,
        "last_failed_inference": last_failure.created_at if last_failure else None,
        "failure_reason": failure_reason,
        "blocking_reason": failure_reason,
        "predictions_last_24h": int(predictions_last_24h or 0),
        "average_inference_duration": None,
        "fusion_eligibility": fusion_eligible,
        "fusion_status": fusion_status,
        "health_state": health_state,
        "technical_status": technical_status,
        "research_reliability": research_reliability,
        "limitations": model.limitations_json or [],
    }


def _serialize_resource(resource: Resource) -> dict:
    return {
        "id": resource.id,
        "title": resource.title,
        "category": resource.category,
        "resource_type": resource.resource_type,
        "description": resource.description,
        "url": resource.url,
        "phone": resource.phone,
        "status": resource.status,
        "is_active": resource.is_active,
        "approved_by": resource.approved_by,
        "approved_at": resource.approved_at,
        "archived_at": resource.archived_at,
        "metadata": resource.metadata_json or {},
        "created_at": resource.created_at,
    }


class UniversityCreate(BaseModel):
    university_name: str
    university_code: str
    campus_name: Optional[str] = None
    address: Optional[str] = None
    district: Optional[str] = None
    province: Optional[str] = None
    general_phone: Optional[str] = None
    counseling_unit_phone: Optional[str] = None
    emergency_support_phone: Optional[str] = None
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    active: bool = True


class UniversityUpdate(BaseModel):
    university_name: Optional[str] = None
    campus_name: Optional[str] = None
    address: Optional[str] = None
    district: Optional[str] = None
    province: Optional[str] = None
    general_phone: Optional[str] = None
    counseling_unit_phone: Optional[str] = None
    emergency_support_phone: Optional[str] = None
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    active: Optional[bool] = None


class CounselorCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    university_id: Optional[int] = None
    professional_title: Optional[str] = None
    qualification: Optional[str] = None
    specialization: Optional[str] = None
    registration_number: Optional[str] = None
    office_name: Optional[str] = None
    office_location: Optional[str] = None
    telephone_number: Optional[str] = None
    whatsapp_number: Optional[str] = None
    available_days: Optional[str] = None
    available_from: Optional[str] = None
    available_until: Optional[str] = None
    languages_json: Optional[list[str]] = None
    approved: bool = True
    active: bool = True


class CounselorUpdate(BaseModel):
    full_name: Optional[str] = None
    professional_title: Optional[str] = None
    qualification: Optional[str] = None
    specialization: Optional[str] = None
    registration_number: Optional[str] = None
    office_name: Optional[str] = None
    office_location: Optional[str] = None
    telephone_number: Optional[str] = None
    whatsapp_number: Optional[str] = None
    email: Optional[EmailStr] = None
    available_days: Optional[str] = None
    available_from: Optional[str] = None
    available_until: Optional[str] = None
    accepts_voice_calls: Optional[bool] = None
    accepts_whatsapp_calls: Optional[bool] = None
    accepts_whatsapp_messages: Optional[bool] = None
    emergency_contact_enabled: Optional[bool] = None
    languages_json: Optional[list[str]] = None
    availability_status: Optional[str] = None
    approved: Optional[bool] = None
    student_visible: Optional[bool] = None
    active: Optional[bool] = None
    admin_notes: Optional[str] = None
    transfer_to_counselor_id: Optional[int] = None


class CounselorUniversityAssign(BaseModel):
    university_id: int
    assignment_reason: Optional[str] = None
    transfer_active_students_to_counselor_id: Optional[int] = None


class SupportContactCreate(BaseModel):
    university_id: Optional[int] = None
    counselor_profile_id: Optional[int] = None
    contact_type: str = "system_fallback"
    display_name: str
    telephone_number: Optional[str] = None
    whatsapp_number: Optional[str] = None
    email: Optional[EmailStr] = None
    available_days: Optional[str] = None
    available_from: Optional[str] = None
    available_until: Optional[str] = None
    telephone_enabled: bool = True
    whatsapp_enabled: bool = True
    student_visible: bool = True
    emergency_service: bool = False
    verified: bool = True
    active: bool = True
    priority: int = 100


class SupportContactUpdate(BaseModel):
    display_name: Optional[str] = None
    telephone_number: Optional[str] = None
    whatsapp_number: Optional[str] = None
    email: Optional[EmailStr] = None
    available_days: Optional[str] = None
    available_from: Optional[str] = None
    available_until: Optional[str] = None
    telephone_enabled: Optional[bool] = None
    whatsapp_enabled: Optional[bool] = None
    student_visible: Optional[bool] = None
    emergency_service: Optional[bool] = None
    verified: Optional[bool] = None
    active: Optional[bool] = None
    priority: Optional[int] = None


class UserCreateAdmin(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: UserRole = UserRole.STUDENT
    university_id: Optional[int] = None
    department: Optional[str] = None
    year_of_study: Optional[int] = None
    send_invitation: bool = False


class UserUpdateAdmin(BaseModel):
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    university_id: Optional[int] = None
    department: Optional[str] = None
    year_of_study: Optional[int] = None
    status: Optional[str] = None
    is_active: Optional[bool] = None
    assigned_counselor_id: Optional[int] = None


class PasswordResetPayload(BaseModel):
    temporary_password: Optional[str] = None


class AssignCounselorPayload(BaseModel):
    counselor_id: int
    assignment_reason: Optional[str] = None


class TransferStudentsPayload(BaseModel):
    from_counselor_id: Optional[int] = None
    to_counselor_id: int
    student_ids: Optional[list[int]] = None
    reason: Optional[str] = None


class AdminModelActionPayload(BaseModel):
    target_model_id: Optional[int] = None
    reason: Optional[str] = None


class ResourceCreate(BaseModel):
    title: str
    category: str
    description: Optional[str] = None
    resource_type: str = "article"
    url: Optional[str] = None
    phone: Optional[str] = None
    status: str = "draft"
    is_active: bool = True
    metadata: Optional[dict] = None


class ResourceUpdate(BaseModel):
    title: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    resource_type: Optional[str] = None
    url: Optional[str] = None
    phone: Optional[str] = None
    status: Optional[str] = None
    is_active: Optional[bool] = None
    metadata: Optional[dict] = None


class SettingUpdate(BaseModel):
    setting_value: Optional[dict | str | int | float | bool | list] = None
    description: Optional[str] = None
    read_only: Optional[bool] = None


class ReportCreate(BaseModel):
    report_type: str
    title: Optional[str] = None
    parameters: Optional[dict] = None
    export_formats: list[str] = ["pdf", "csv", "excel"]


def _university_or_404(db: Session, university_id: int) -> University:
    university = db.query(University).filter(University.id == university_id).first()
    if not university:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="University not found")
    return university


def _profile_or_404(db: Session, counselor_id: int) -> CounselorProfile:
    profile = db.query(CounselorProfile).filter(CounselorProfile.id == counselor_id).first()
    if not profile:
        profile = db.query(CounselorProfile).filter(CounselorProfile.user_id == counselor_id).first()
    if not profile:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Counselor profile not found")
    return profile


def _serialize_university(university: University, db: Session | None = None) -> dict:
    payload = {
        "id": university.id,
        "university_id": university.university_id,
        "university_name": university.university_name,
        "university": university.university_name,
        "university_code": university.university_code,
        "campus_name": university.campus_name,
        "campus": university.campus_name,
        "address": university.address,
        "district": university.district,
        "province": university.province,
        "general_phone": university.general_phone,
        "counseling_unit_phone": university.counseling_unit_phone,
        "emergency_support_phone": university.emergency_support_phone,
        "email": university.email,
        "website": university.website,
        "active": university.active,
        "status": "active" if university.active else "inactive",
    }
    if db:
        payload.update(
            {
                "students": db.query(User).filter(User.university_id == university.id, User.role == UserRole.STUDENT).count(),
                "counselors": db.query(CounselorProfile).filter(CounselorProfile.university_id == university.id).count(),
                "support_contacts": db.query(SupportContact).filter(SupportContact.university_id == university.id).count(),
            }
        )
    return payload


def _profile_admin_payload(db: Session, profile: CounselorProfile) -> dict:
    data = serialize_profile(profile, include_private=True)
    data["assignment_count"] = (
        db.query(CounselorAssignment)
        .filter(CounselorAssignment.counselor_id == profile.user_id, CounselorAssignment.active.is_(True))
        .count()
    )
    data["unresolved_review_count"] = int(has_unresolved_reviews(db, profile.user_id))
    return data


@router.post("/universities")
def create_university(payload: UniversityCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    university = University(
        university_id=public_id("uni"),
        university_name=payload.university_name,
        university_code=payload.university_code,
        campus_name=payload.campus_name,
        address=payload.address,
        district=payload.district,
        province=payload.province,
        general_phone=normalize_e164(payload.general_phone),
        counseling_unit_phone=normalize_e164(payload.counseling_unit_phone),
        emergency_support_phone=normalize_e164(payload.emergency_support_phone),
        email=str(payload.email) if payload.email else None,
        website=payload.website,
        active=payload.active,
    )
    db.add(university)
    db.commit()
    db.refresh(university)
    return _serialize_university(university)


@router.get("/universities")
def list_universities(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    return {"universities": [_serialize_university(item, db) for item in db.query(University).order_by(University.university_name.asc()).all()]}


@router.get("/universities/{university_id}")
def get_university(university_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    return _serialize_university(_university_or_404(db, university_id), db)


@router.patch("/universities/{university_id}")
def update_university(
    university_id: int,
    payload: UniversityUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    university = _university_or_404(db, university_id)
    updates = payload.dict(exclude_unset=True)
    for phone_field in ("general_phone", "counseling_unit_phone", "emergency_support_phone"):
        if phone_field in updates:
            updates[phone_field] = normalize_e164(updates[phone_field])
    if "email" in updates and updates["email"] is not None:
        updates["email"] = str(updates["email"])
    for field, value in updates.items():
        setattr(university, field, value)
    university.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(university)
    return _serialize_university(university)


@router.post("/counselors")
def create_counselor(payload: CounselorCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    if db.query(User).filter((User.username == payload.username) | (User.email == str(payload.email))).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Counselor username or email already exists")
    if payload.university_id:
        _university_or_404(db, payload.university_id)
    validate_time_range(payload.available_from, payload.available_until)
    user = User(
        email=str(payload.email),
        username=payload.username,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=UserRole.COUNSELOR,
        university_id=payload.university_id,
        is_active=payload.active,
    )
    db.add(user)
    db.flush()
    profile = CounselorProfile(
        counselor_profile_id=public_id("cprof"),
        user_id=user.id,
        university_id=payload.university_id,
        full_name=payload.full_name,
        professional_title=payload.professional_title,
        qualification=payload.qualification,
        specialization=payload.specialization,
        registration_number=payload.registration_number,
        office_name=payload.office_name,
        office_location=payload.office_location,
        telephone_number=normalize_e164(payload.telephone_number),
        whatsapp_number=normalize_e164(payload.whatsapp_number),
        email=str(payload.email),
        available_days=payload.available_days,
        available_from=payload.available_from,
        available_until=payload.available_until,
        languages_json=payload.languages_json or [],
        approved=payload.approved,
        active=payload.active,
    )
    db.add(profile)
    db.flush()
    if payload.university_id:
        preserve_university_assignment(db, profile=profile, university_id=payload.university_id, admin_user=current_user, reason="Initial profile assignment")
    audit_profile_change(db, profile=profile, changed_by=current_user, change_type="admin_create", after=serialize_profile(profile, include_private=True))
    db.commit()
    db.refresh(profile)
    return _profile_admin_payload(db, profile)


@router.get("/counselors")
def list_counselors(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    profiles = db.query(CounselorProfile).order_by(CounselorProfile.full_name.asc()).all()
    return {"counselors": [_profile_admin_payload(db, profile) for profile in profiles]}


@router.get("/counselors/{counselor_id}")
def get_counselor(counselor_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    return _profile_admin_payload(db, _profile_or_404(db, counselor_id))


@router.patch("/counselors/{counselor_id}")
def update_counselor(
    counselor_id: int,
    payload: CounselorUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    profile = _profile_or_404(db, counselor_id)
    updates = payload.dict(exclude_unset=True)
    transfer_target = updates.pop("transfer_to_counselor_id", None)
    if updates.get("availability_status") and updates["availability_status"] not in AVAILABILITY_STATUSES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid availability status")
    validate_time_range(updates.get("available_from", profile.available_from), updates.get("available_until", profile.available_until))
    if updates.get("active") is False and has_unresolved_reviews(db, profile.user_id) and not transfer_target:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Transfer unresolved work before deactivating counselor")
    if transfer_target:
        target_profile = _profile_or_404(db, transfer_target)
        transfer_active_student_assignments(db, from_counselor_id=profile.user_id, to_counselor_id=target_profile.user_id, admin_user=current_user)

    before = serialize_profile(profile, include_private=True)
    for field in ("telephone_number", "whatsapp_number"):
        if field in updates:
            updates[field] = normalize_e164(updates[field])
    if "email" in updates and updates["email"] is not None:
        updates["email"] = str(updates["email"])
    for field, value in updates.items():
        setattr(profile, field, value)
    if "full_name" in updates:
        profile.user.full_name = updates["full_name"]
    if "active" in updates:
        profile.user.is_active = updates["active"]
    profile.updated_at = datetime.utcnow()
    audit_profile_change(
        db,
        profile=profile,
        changed_by=current_user,
        change_type="admin_update",
        before=before,
        after=serialize_profile(profile, include_private=True),
    )
    db.commit()
    db.refresh(profile)
    return _profile_admin_payload(db, profile)


@router.post("/counselors/{counselor_id}/university")
def assign_counselor_university(
    counselor_id: int,
    payload: CounselorUniversityAssign,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    profile = _profile_or_404(db, counselor_id)
    _university_or_404(db, payload.university_id)
    if payload.transfer_active_students_to_counselor_id:
        target = _profile_or_404(db, payload.transfer_active_students_to_counselor_id)
        transfer_active_student_assignments(db, from_counselor_id=profile.user_id, to_counselor_id=target.user_id, admin_user=current_user)
    before = serialize_profile(profile, include_private=True)
    profile.university_id = payload.university_id
    profile.user.university_id = payload.university_id
    preserve_university_assignment(db, profile=profile, university_id=payload.university_id, admin_user=current_user, reason=payload.assignment_reason)
    audit_profile_change(
        db,
        profile=profile,
        changed_by=current_user,
        change_type="admin_university_assignment",
        before=before,
        after=serialize_profile(profile, include_private=True),
        notes=payload.assignment_reason,
    )
    db.commit()
    db.refresh(profile)
    return _profile_admin_payload(db, profile)


@router.post("/support-contacts")
def create_support_contact(
    payload: SupportContactCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    validate_time_range(payload.available_from, payload.available_until)
    if payload.university_id:
        _university_or_404(db, payload.university_id)
    contact = SupportContact(
        support_contact_id=public_id("support"),
        university_id=payload.university_id,
        counselor_profile_id=payload.counselor_profile_id,
        contact_type=payload.contact_type,
        display_name=payload.display_name,
        telephone_number=normalize_e164(payload.telephone_number),
        whatsapp_number=normalize_e164(payload.whatsapp_number),
        email=str(payload.email) if payload.email else None,
        available_days=payload.available_days,
        available_from=payload.available_from,
        available_until=payload.available_until,
        telephone_enabled=payload.telephone_enabled,
        whatsapp_enabled=payload.whatsapp_enabled,
        student_visible=payload.student_visible,
        emergency_service=payload.emergency_service,
        verified=payload.verified,
        verified_at=datetime.utcnow() if payload.verified else None,
        verified_by=current_user.id if payload.verified else None,
        active=payload.active,
        priority=payload.priority,
    )
    db.add(contact)
    db.commit()
    db.refresh(contact)
    return serialize_support_contact(contact)


@router.get("/support-contacts")
def list_support_contacts_admin(
    contact_type: Optional[str] = None,
    active: Optional[bool] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    query = db.query(SupportContact).order_by(SupportContact.priority.asc(), SupportContact.created_at.desc())
    if contact_type:
        query = query.filter(SupportContact.contact_type == contact_type)
    if active is not None:
        query = query.filter(SupportContact.active.is_(active))
    return {"support_contacts": [serialize_support_contact(item) for item in query.all()]}


@router.patch("/support-contacts/{contact_id}")
def update_support_contact(
    contact_id: int,
    payload: SupportContactUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    contact = db.query(SupportContact).filter(SupportContact.id == contact_id).first()
    if not contact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Support contact not found")
    updates = payload.dict(exclude_unset=True)
    validate_time_range(updates.get("available_from", contact.available_from), updates.get("available_until", contact.available_until))
    for field in ("telephone_number", "whatsapp_number"):
        if field in updates:
            updates[field] = normalize_e164(updates[field])
    if "email" in updates and updates["email"] is not None:
        updates["email"] = str(updates["email"])
    if updates.get("verified") is True and not contact.verified:
        contact.verified_at = datetime.utcnow()
        contact.verified_by = current_user.id
    for field, value in updates.items():
        setattr(contact, field, value)
    contact.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(contact)
    return serialize_support_contact(contact)


@router.delete("/support-contacts/{contact_id}")
def deactivate_support_contact(
    contact_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    contact = db.query(SupportContact).filter(SupportContact.id == contact_id).first()
    if not contact:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Support contact not found")
    before = serialize_support_contact(contact)
    contact.active = False
    contact.updated_at = datetime.utcnow()
    _audit(db, actor=current_user, action="deactivate_support_contact", entity_type="support_contact", entity_id=contact.id, old_value=before, new_value=serialize_support_contact(contact), request=request)
    db.commit()
    return serialize_support_contact(contact)


@router.get("/dashboard")
def get_admin_dashboard(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    today_start, today_end = _today_bounds()
    overview = {
        "total_universities": _count(db.query(University)),
        "total_students": _count(db.query(User).filter(User.role == UserRole.STUDENT)),
        "total_counselors": _count(db.query(User).filter(User.role == UserRole.COUNSELOR)),
        "total_administrators": _count(db.query(User).filter(User.role == UserRole.ADMIN)),
        "todays_logins": _count(db.query(User).filter(User.last_login_at >= today_start, User.last_login_at < today_end)),
        "todays_assessments": (
            _count(db.query(Assessment).filter(Assessment.created_at >= today_start, Assessment.created_at < today_end))
            + _count(db.query(DASS21Assessment).filter(DASS21Assessment.created_at >= today_start, DASS21Assessment.created_at < today_end))
            + _count(db.query(RiskAssessment).filter(RiskAssessment.created_at >= today_start, RiskAssessment.created_at < today_end))
        ),
        "fusion_assessments": _count(db.query(RiskAssessment)),
        "support_requests": _count(db.query(SupportContactAction)),
        "active_models": _count(db.query(ModelRegistry).filter(ModelRegistry.is_active.is_(True))),
        "system_status": _system_status(db),
        "storage_usage": _storage_usage(),
    }
    audit_items = [
        {
            "timestamp": item.created_at,
            "user": item.user.full_name if item.user else None,
            "action": item.action,
            "entity": item.entity_type,
            "status": item.status,
        }
        for item in db.query(AdminAuditLog).options(joinedload(AdminAuditLog.user)).order_by(AdminAuditLog.created_at.desc()).limit(10)
    ]
    support_items = [
        {
            "timestamp": item.created_at,
            "user": item.user.full_name if item.user else None,
            "action": item.action_type,
            "entity": "support_contact",
            "status": "success",
        }
        for item in db.query(SupportContactAction).options(joinedload(SupportContactAction.user)).order_by(SupportContactAction.created_at.desc()).limit(5)
    ]
    overview["recent_activity"] = sorted(audit_items + support_items, key=lambda item: item["timestamp"], reverse=True)[:10]
    return overview


@router.get("/statistics")
def get_admin_statistics(
    days: int = Query(default=30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    start_day = date.today() - timedelta(days=days - 1)
    labels = [start_day + timedelta(days=offset) for offset in range(days)]

    def daily_count(model, created_column):
        rows = (
            db.query(func.date(created_column).label("day"), func.count(model.id))
            .filter(created_column >= datetime.combine(start_day, datetime.min.time()))
            .group_by(func.date(created_column))
            .all()
        )
        counts = {str(day): count for day, count in rows}
        return [{"date": item.isoformat(), "value": int(counts.get(item.isoformat(), 0))} for item in labels]

    risk_rows = db.query(RiskAssessment.risk_level, func.count(RiskAssessment.id)).group_by(RiskAssessment.risk_level).all()
    mood_rows = (
        db.query(func.date(DailyCheckIn.created_at).label("day"), func.avg(DailyCheckIn.mood), func.avg(DailyCheckIn.stress_level))
        .filter(DailyCheckIn.created_at >= datetime.combine(start_day, datetime.min.time()))
        .group_by(func.date(DailyCheckIn.created_at))
        .all()
    )
    mood_map = {str(day): {"mood": round(float(mood or 0), 2), "stress": round(float(stress or 0), 2)} for day, mood, stress in mood_rows}
    workload = (
        db.query(User.id, User.full_name, func.count(CounselorAssignment.id))
        .outerjoin(CounselorAssignment, (CounselorAssignment.counselor_id == User.id) & (CounselorAssignment.active.is_(True)))
        .filter(User.role == UserRole.COUNSELOR)
        .group_by(User.id, User.full_name)
        .all()
    )
    university_comparison = [
        {
            "university": university.university_name,
            "students": db.query(User).filter(User.university_id == university.id, User.role == UserRole.STUDENT).count(),
            "counselors": db.query(CounselorProfile).filter(CounselorProfile.university_id == university.id).count(),
            "support_contacts": db.query(SupportContact).filter(SupportContact.university_id == university.id).count(),
        }
        for university in db.query(University).order_by(University.university_name.asc()).all()
    ]
    return {
        "daily_active_users": daily_count(User, User.last_login_at),
        "weekly_active_users": _count(db.query(User).filter(User.last_login_at >= datetime.utcnow() - timedelta(days=7))),
        "monthly_active_users": _count(db.query(User).filter(User.last_login_at >= datetime.utcnow() - timedelta(days=30))),
        "assessment_completion": {
            "profile": _count(db.query(ProfileAssessment)) if "ProfileAssessment" in globals() else 0,
            "dass21": _count(db.query(DASS21Assessment).filter(DASS21Assessment.is_complete.is_(True))),
            "daily_checkins": _count(db.query(DailyCheckIn)),
            "fusion": _count(db.query(RiskAssessment)),
        },
        "fusion_distribution": [{"risk_level": risk or "UNKNOWN", "value": int(count)} for risk, count in risk_rows],
        "mood_trends": [
            {"date": item.isoformat(), **mood_map.get(item.isoformat(), {"mood": 0, "stress": 0})}
            for item in labels
        ],
        "support_requests": daily_count(SupportContactAction, SupportContactAction.created_at),
        "counselor_workload": [{"counselor_id": cid, "counselor": name, "students_assigned": int(count)} for cid, name, count in workload],
        "university_comparison": university_comparison,
    }


@router.get("/users")
def list_admin_users(
    role: Optional[str] = None,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    university_id: Optional[int] = None,
    q: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    query = db.query(User).options(joinedload(User.university)).order_by(User.created_at.desc())
    if role:
        query = query.filter(User.role == UserRole(role))
    if university_id:
        query = query.filter(User.university_id == university_id)
    if status_filter == "active":
        query = query.filter(User.is_active.is_(True), User.suspended_at.is_(None))
    elif status_filter == "inactive":
        query = query.filter(User.is_active.is_(False))
    elif status_filter == "suspended":
        query = query.filter(User.suspended_at.isnot(None))
    if q:
        needle = f"%{q}%"
        query = query.filter(or_(User.full_name.ilike(needle), User.email.ilike(needle), User.username.ilike(needle)))
    users = query.all()
    return {"users": [_serialize_user_admin(db, item) for item in users]}


@router.post("/users")
def create_admin_user(
    payload: UserCreateAdmin,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    if payload.role == UserRole.PSYCHIATRIST:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Phase 4H manages student, counselor, and administrator roles only")
    if payload.university_id:
        _university_or_404(db, payload.university_id)
    if db.query(User).filter((User.username == payload.username) | (User.email == str(payload.email))).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already exists")
    user = User(
        username=payload.username,
        email=str(payload.email),
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        university_id=payload.university_id,
        department=payload.department,
        year_of_study=payload.year_of_study,
        invitation_sent_at=datetime.utcnow() if payload.send_invitation else None,
        is_active=True,
    )
    db.add(user)
    db.flush()
    if payload.role == UserRole.COUNSELOR:
        db.add(
            CounselorProfile(
                counselor_profile_id=public_id("cprof"),
                user_id=user.id,
                university_id=user.university_id,
                full_name=user.full_name,
                email=user.email,
                approved=True,
                active=True,
            )
        )
    _audit(db, actor=current_user, action="create_user", entity_type="user", entity_id=user.id, new_value=_serialize_user_admin(db, user), request=request)
    db.commit()
    db.refresh(user)
    return _serialize_user_admin(db, user)


@router.patch("/users/{user_id}")
def update_admin_user(
    user_id: int,
    payload: UserUpdateAdmin,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    before = _serialize_user_admin(db, user)
    updates = payload.dict(exclude_unset=True)
    assigned_counselor_id = updates.pop("assigned_counselor_id", None)
    if updates.get("university_id"):
        _university_or_404(db, updates["university_id"])
    if "email" in updates and updates["email"] is not None:
        updates["email"] = str(updates["email"])
    if "role" in updates and updates["role"] == UserRole.PSYCHIATRIST:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Phase 4H does not manage psychiatrist accounts")
    status_update = updates.pop("status", None)
    for field, value in updates.items():
        setattr(user, field, value)
    if status_update:
        if status_update == "active":
            user.is_active = True
            user.suspended_at = None
        elif status_update == "inactive":
            user.is_active = False
            user.suspended_at = None
        elif status_update == "suspended":
            user.is_active = False
            user.suspended_at = datetime.utcnow()
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user status")
    if assigned_counselor_id:
        _assign_student_to_counselor(db, student=user, counselor_id=assigned_counselor_id, actor=current_user, reason="Admin user update")
    user.updated_at = datetime.utcnow()
    _audit(db, actor=current_user, action="update_user", entity_type="user", entity_id=user.id, old_value=before, new_value=_serialize_user_admin(db, user), request=request)
    db.commit()
    db.refresh(user)
    return _serialize_user_admin(db, user)


@router.delete("/users/{user_id}")
def deactivate_admin_user(
    user_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    before = _serialize_user_admin(db, user)
    user.is_active = False
    user.suspended_at = None
    user.updated_at = datetime.utcnow()
    _audit(db, actor=current_user, action="deactivate_user", entity_type="user", entity_id=user.id, old_value=before, new_value=_serialize_user_admin(db, user), request=request)
    db.commit()
    return {"status": "inactive", "user_id": user.id}


@router.post("/users/{user_id}/reset-password")
def reset_admin_user_password(
    user_id: int,
    payload: PasswordResetPayload,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    temporary_password = payload.temporary_password or f"Temp{user.id}Pass123!"
    user.hashed_password = hash_password(temporary_password)
    user.updated_at = datetime.utcnow()
    _audit(db, actor=current_user, action="reset_password", entity_type="user", entity_id=user.id, request=request)
    db.commit()
    return {"user_id": user.id, "temporary_password_set": True}


@router.post("/users/{user_id}/resend-invitation")
def resend_admin_invitation(
    user_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user.invitation_sent_at = datetime.utcnow()
    _audit(db, actor=current_user, action="resend_invitation", entity_type="user", entity_id=user.id, request=request)
    db.commit()
    return {"user_id": user.id, "invitation_status": "queued_future_ready"}


def _assign_student_to_counselor(db: Session, *, student: User, counselor_id: int, actor: User, reason: str | None = None) -> CounselorAssignment:
    if student.role != UserRole.STUDENT:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Counselor assignment applies only to students")
    counselor = db.query(User).filter(User.id == counselor_id, User.role == UserRole.COUNSELOR).first()
    if not counselor:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Counselor not found")
    (
        db.query(CounselorAssignment)
        .filter(CounselorAssignment.student_id == student.id, CounselorAssignment.active.is_(True))
        .update({"active": False, "end_date": datetime.utcnow()}, synchronize_session="fetch")
    )
    assignment = CounselorAssignment(
        assignment_id=public_id("asg"),
        student_id=student.id,
        counselor_id=counselor.id,
        assigned_by=actor.id,
        assignment_reason=reason,
        active=True,
    )
    db.add(assignment)
    db.flush()
    return assignment


@router.post("/users/{user_id}/assign-counselor")
def assign_admin_student_counselor(
    user_id: int,
    payload: AssignCounselorPayload,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    student = db.query(User).filter(User.id == user_id).first()
    if not student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    assignment = _assign_student_to_counselor(db, student=student, counselor_id=payload.counselor_id, actor=current_user, reason=payload.assignment_reason)
    _audit(db, actor=current_user, action="assign_counselor", entity_type="counselor_assignment", entity_id=assignment.id, new_value={"student_id": user_id, "counselor_id": payload.counselor_id}, request=request)
    db.commit()
    return {"assignment_id": assignment.assignment_id, "student_id": user_id, "counselor_id": payload.counselor_id}


@router.delete("/universities/{university_id}")
def deactivate_university(
    university_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    university = _university_or_404(db, university_id)
    before = _serialize_university(university, db)
    university.active = False
    university.updated_at = datetime.utcnow()
    _audit(db, actor=current_user, action="deactivate_university", entity_type="university", entity_id=university.id, old_value=before, new_value=_serialize_university(university, db), request=request)
    db.commit()
    return _serialize_university(university, db)


@router.get("/universities/{university_id}/statistics")
def get_university_statistics(
    university_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    university = _university_or_404(db, university_id)
    student_ids = [row[0] for row in db.query(User.id).filter(User.university_id == university.id, User.role == UserRole.STUDENT).all()]
    return {
        "university": _serialize_university(university, db),
        "assessments": _count(db.query(Assessment).filter(Assessment.user_id.in_(student_ids))) if student_ids else 0,
        "dass21_completed": _count(db.query(DASS21Assessment).filter(DASS21Assessment.user_id.in_(student_ids), DASS21Assessment.is_complete.is_(True))) if student_ids else 0,
        "fusion_assessments": _count(db.query(RiskAssessment).filter(RiskAssessment.student_id.in_(student_ids))) if student_ids else 0,
        "active_assignments": _count(db.query(CounselorAssignment).filter(CounselorAssignment.student_id.in_(student_ids), CounselorAssignment.active.is_(True))) if student_ids else 0,
    }


@router.post("/universities/{university_id}/assign-counselors")
def assign_university_counselors(
    university_id: int,
    counselor_ids: list[int],
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    _university_or_404(db, university_id)
    updated = []
    for counselor_id in counselor_ids:
        profile = _profile_or_404(db, counselor_id)
        profile.university_id = university_id
        profile.user.university_id = university_id
        updated.append(profile.id)
    _audit(db, actor=current_user, action="assign_university_counselors", entity_type="university", entity_id=university_id, new_value={"counselor_profile_ids": updated}, request=request)
    db.commit()
    return {"university_id": university_id, "assigned_counselor_profiles": updated}


@router.post("/universities/{university_id}/transfer-students")
def transfer_university_students(
    university_id: int,
    payload: TransferStudentsPayload,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    _university_or_404(db, university_id)
    target = db.query(User).filter(User.id == payload.to_counselor_id, User.role == UserRole.COUNSELOR).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target counselor not found")
    query = db.query(User).filter(User.university_id == university_id, User.role == UserRole.STUDENT)
    if payload.student_ids:
        query = query.filter(User.id.in_(payload.student_ids))
    transferred = []
    for student in query.all():
        if payload.from_counselor_id:
            active = _active_assignment_for_student(db, student.id)
            if not active or active.counselor_id != payload.from_counselor_id:
                continue
        assignment = _assign_student_to_counselor(db, student=student, counselor_id=payload.to_counselor_id, actor=current_user, reason=payload.reason)
        transferred.append({"student_id": student.id, "assignment_id": assignment.assignment_id})
    _audit(db, actor=current_user, action="transfer_students", entity_type="university", entity_id=university_id, new_value={"count": len(transferred), "to_counselor_id": payload.to_counselor_id}, request=request)
    db.commit()
    return {"transferred": transferred, "count": len(transferred)}


@router.delete("/counselors/{counselor_id}")
def deactivate_counselor(
    counselor_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    profile = _profile_or_404(db, counselor_id)
    before = _profile_admin_payload(db, profile)
    profile.active = False
    profile.user.is_active = False
    profile.updated_at = datetime.utcnow()
    _audit(db, actor=current_user, action="deactivate_counselor", entity_type="counselor_profile", entity_id=profile.id, old_value=before, new_value=_profile_admin_payload(db, profile), request=request)
    db.commit()
    return _profile_admin_payload(db, profile)


@router.post("/counselors/{counselor_id}/transfer-students")
def transfer_counselor_students(
    counselor_id: int,
    payload: TransferStudentsPayload,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    source = _profile_or_404(db, counselor_id)
    target = _profile_or_404(db, payload.to_counselor_id)
    query = db.query(CounselorAssignment).filter(CounselorAssignment.counselor_id == source.user_id, CounselorAssignment.active.is_(True))
    if payload.student_ids:
        query = query.filter(CounselorAssignment.student_id.in_(payload.student_ids))
    transferred = []
    for existing in query.all():
        existing.active = False
        existing.end_date = datetime.utcnow()
        next_assignment = CounselorAssignment(
            assignment_id=public_id("asg"),
            student_id=existing.student_id,
            counselor_id=target.user_id,
            assigned_by=current_user.id,
            assignment_reason=payload.reason or "Admin counselor transfer",
            active=True,
        )
        db.add(next_assignment)
        transferred.append(existing.student_id)
    _audit(db, actor=current_user, action="transfer_counselor_students", entity_type="counselor_profile", entity_id=source.id, new_value={"student_ids": transferred, "to_counselor_profile_id": target.id}, request=request)
    db.commit()
    return {"count": len(transferred), "student_ids": transferred}


@router.get("/models")
def list_admin_models(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    discover_runtime_candidates(db)
    db.commit()
    models = db.query(ModelRegistry).order_by(ModelRegistry.modality.asc(), ModelRegistry.model_name.asc(), ModelRegistry.version.asc()).all()
    modalities = ["profile", "text", "speech", "face", "fusion"]
    return {
        "models": [_serialize_model(item) for item in models],
        "by_modality": {modality: [_serialize_model(item) for item in models if item.modality == modality] for modality in modalities},
    }


@router.get("/models/runtime-status")
def get_model_runtime_status(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    discover_runtime_candidates(db)
    db.commit()
    rows = db.query(ModelRegistry).order_by(ModelRegistry.modality.asc(), ModelRegistry.updated_at.desc()).all()
    latest_by_modality = {}
    for model in rows:
        latest_by_modality.setdefault(model.modality, model)
    modalities = ["profile", "text", "speech", "face", "behavioral"]
    status_rows = [_runtime_status_item(db, latest_by_modality.get(modality), modality) for modality in modalities]
    return {
        "runtime_status": status_rows,
        "fusion": {
            "status": "active",
            "behavioral": "unavailable",
            "speech_face_policy": "included only when verified active and fusion eligible",
        },
        "privacy": "No sensitive source content is returned by this endpoint.",
    }


@router.get("/models/{model_id}/smoke-test")
def smoke_test_admin_model(model_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    result = verify_model_artifact(model)
    return {"model": _serialize_model(model), "smoke_test": result.to_dict()}


@router.post("/models/{model_id}/verify-runtime")
def verify_runtime_admin_model(
    model_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    result = verify_model_artifact(model)
    apply_verification_result(db, model, result)
    _audit(db, actor=current_user, action="verify_runtime_model", entity_type="model_registry", entity_id=model.id, new_value=result.to_dict(), request=request)
    db.commit()
    db.refresh(model)
    return {"model": _serialize_model(model), "runtime_status": _runtime_status_item(db, model, model.modality), "verification": result.to_dict()}


@router.post("/models/{model_id}/verify")
def verify_admin_model(
    model_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    result = verify_model_artifact(model)
    apply_verification_result(db, model, result)
    _audit(db, actor=current_user, action="verify_model", entity_type="model_registry", entity_id=model.id, new_value=result.to_dict(), request=request)
    db.commit()
    db.refresh(model)
    return {"model": _serialize_model(model), "verification": result.to_dict()}


@router.post("/models/{model_id}/activate")
def activate_admin_model(
    model_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    try:
        activated = activate_registry_model(db, model_name=model.model_name, modality=model.modality, version=model.version)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    activated.approved_by = current_user.username
    _audit(db, actor=current_user, action="activate_model", entity_type="model_registry", entity_id=activated.id, new_value=_serialize_model(activated), request=request)
    db.commit()
    db.refresh(activated)
    return _serialize_model(activated)


@router.post("/models/{model_id}/deactivate")
def deactivate_admin_model(
    model_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    deactivated = deactivate_registry_model(db, model_name=model.model_name, modality=model.modality, version=model.version)
    _audit(db, actor=current_user, action="deactivate_model", entity_type="model_registry", entity_id=deactivated.id, new_value=_serialize_model(deactivated), request=request)
    db.commit()
    db.refresh(deactivated)
    return _serialize_model(deactivated)


@router.post("/models/{model_id}/rollback")
def rollback_admin_model(
    model_id: int,
    payload: AdminModelActionPayload,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    current_model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not current_model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    target = None
    if payload.target_model_id:
        target = db.query(ModelRegistry).filter(ModelRegistry.id == payload.target_model_id, ModelRegistry.modality == current_model.modality).first()
    else:
        target = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.modality == current_model.modality, ModelRegistry.id != current_model.id, ModelRegistry.verification_status == "passed")
            .order_by(ModelRegistry.updated_at.desc())
            .first()
        )
    if not target:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No verified rollback target available")
    activated = activate_registry_model(db, model_name=target.model_name, modality=target.modality, version=target.version)
    activated.approved_by = current_user.username
    _audit(db, actor=current_user, action="rollback_model", entity_type="model_registry", entity_id=activated.id, new_value={"from": model_id, "to": activated.id, "reason": payload.reason}, request=request)
    db.commit()
    db.refresh(activated)
    return _serialize_model(activated)


@router.get("/models/{model_id}/history")
def get_admin_model_history(model_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    siblings = db.query(ModelRegistry).filter(ModelRegistry.modality == model.modality, ModelRegistry.model_name == model.model_name).order_by(ModelRegistry.created_at.desc()).all()
    audits = db.query(AdminAuditLog).filter(AdminAuditLog.entity_type == "model_registry", AdminAuditLog.entity_id == str(model_id)).order_by(AdminAuditLog.created_at.desc()).all()
    return {
        "versions": [_serialize_model(item) for item in siblings],
        "audit": [
            {"timestamp": item.created_at, "action": item.action, "status": item.status, "old_value": item.old_value_json, "new_value": item.new_value_json}
            for item in audits
        ],
    }


@router.get("/resources")
def list_admin_resources(
    q: Optional[str] = None,
    category: Optional[str] = None,
    resource_type: Optional[str] = None,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    query = db.query(Resource).order_by(Resource.created_at.desc())
    if q:
        needle = f"%{q}%"
        query = query.filter(or_(Resource.title.ilike(needle), Resource.description.ilike(needle)))
    if category:
        query = query.filter(Resource.category == category)
    if resource_type:
        query = query.filter(Resource.resource_type == resource_type)
    if status_filter:
        query = query.filter(Resource.status == status_filter)
    return {
        "resources": [_serialize_resource(item) for item in query.all()],
        "categories": [row[0] for row in db.query(Resource.category).distinct().all() if row[0]],
        "resource_types": sorted(RESOURCE_TYPES),
    }


@router.post("/resources")
def create_admin_resource(
    payload: ResourceCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    if payload.resource_type not in RESOURCE_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid resource type")
    resource = Resource(
        title=payload.title,
        category=payload.category,
        description=payload.description,
        resource_type=payload.resource_type,
        url=payload.url,
        phone=normalize_e164(payload.phone) if payload.phone else None,
        status=payload.status,
        is_active=payload.is_active,
        metadata_json=payload.metadata or {},
    )
    db.add(resource)
    db.flush()
    _audit(db, actor=current_user, action="create_resource", entity_type="resource", entity_id=resource.id, new_value=_serialize_resource(resource), request=request)
    db.commit()
    db.refresh(resource)
    return _serialize_resource(resource)


@router.patch("/resources/{resource_id}")
def update_admin_resource(
    resource_id: int,
    payload: ResourceUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    resource = db.query(Resource).filter(Resource.id == resource_id).first()
    if not resource:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
    before = _serialize_resource(resource)
    updates = payload.dict(exclude_unset=True)
    if "resource_type" in updates and updates["resource_type"] not in RESOURCE_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid resource type")
    if "phone" in updates and updates["phone"]:
        updates["phone"] = normalize_e164(updates["phone"])
    if "metadata" in updates:
        updates["metadata_json"] = updates.pop("metadata")
    for field, value in updates.items():
        setattr(resource, field, value)
    if resource.status == "approved" and resource.approved_at is None:
        resource.approved_by = current_user.id
        resource.approved_at = datetime.utcnow()
    if resource.status == "archived" and resource.archived_at is None:
        resource.archived_at = datetime.utcnow()
        resource.is_active = False
    _audit(db, actor=current_user, action="update_resource", entity_type="resource", entity_id=resource.id, old_value=before, new_value=_serialize_resource(resource), request=request)
    db.commit()
    db.refresh(resource)
    return _serialize_resource(resource)


@router.post("/resources/{resource_id}/approve")
def approve_admin_resource(resource_id: int, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return update_admin_resource(resource_id, ResourceUpdate(status="approved", is_active=True), request, current_user, db)


@router.post("/resources/{resource_id}/archive")
def archive_admin_resource(resource_id: int, request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return update_admin_resource(resource_id, ResourceUpdate(status="archived", is_active=False), request, current_user, db)


@router.delete("/resources/{resource_id}")
def delete_admin_resource(
    resource_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    resource = db.query(Resource).filter(Resource.id == resource_id).first()
    if not resource:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
    before = _serialize_resource(resource)
    resource.is_active = False
    resource.status = "deleted"
    resource.archived_at = datetime.utcnow()
    _audit(db, actor=current_user, action="delete_resource", entity_type="resource", entity_id=resource.id, old_value=before, new_value=_serialize_resource(resource), request=request)
    db.commit()
    return {"resource_id": resource.id, "status": "deleted"}


@router.get("/settings")
def get_admin_settings(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    _seed_settings(db)
    db.commit()
    settings = db.query(SystemSetting).order_by(SystemSetting.section.asc(), SystemSetting.setting_key.asc()).all()
    grouped = {section: [] for section in SETTING_SECTIONS}
    for setting in settings:
        grouped.setdefault(setting.section, []).append(
            {
                "id": setting.id,
                "key": setting.setting_key,
                "value": setting.setting_value,
                "value_type": setting.value_type,
                "description": setting.description,
                "read_only": setting.read_only,
                "updated_at": setting.updated_at,
            }
        )
    return {"sections": grouped}


def _seed_settings(db: Session) -> None:
    defaults = [
        ("authentication", "student_self_registration", True, "Allow public student registration", False),
        ("authentication", "staff_registration_mode", "admin_provisioned", "Staff accounts require administrator provisioning", True),
        ("consent", "current_policy_version", "phase4b-v1", "Active consent policy version", True),
        ("fusion", "runtime_policy_version", "controlled-late-fusion-v2", "Controlled late-fusion policy", True),
        ("support_contacts", "fallback_required", True, "At least one active fallback contact must exist", True),
        ("safetalk", "privacy_mode", "summary_only_for_admin", "Admins cannot routinely inspect raw conversations", True),
        ("resource_settings", "approval_required", True, "Resources require admin approval before active publication", False),
        ("maintenance_mode", "enabled", False, "Platform maintenance toggle", False),
        ("storage", "retention_policy", "research_minimized", "Storage is governed by research minimization", True),
        ("feature_flags", "speech_runtime_enabled", False, "Speech runtime remains disabled in Phase 4H", True),
        ("feature_flags", "facial_runtime_enabled", False, "Facial runtime remains disabled in Phase 4H", True),
        ("feature_flags", "behavioral_runtime_enabled", False, "Behavioral runtime remains disabled in Phase 4H", True),
    ]
    for section, key, value, description, read_only in defaults:
        existing = db.query(SystemSetting).filter(SystemSetting.section == section, SystemSetting.setting_key == key).first()
        if not existing:
            db.add(
                SystemSetting(
                    section=section,
                    setting_key=key,
                    setting_value=value,
                    description=description,
                    read_only=read_only,
                )
            )
    db.flush()


@router.patch("/settings/{setting_id}")
def update_admin_setting(
    setting_id: int,
    payload: SettingUpdate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    setting = db.query(SystemSetting).filter(SystemSetting.id == setting_id).first()
    if not setting:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Setting not found")
    if setting.read_only:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This setting is read-only")
    before = {"value": setting.setting_value, "description": setting.description, "read_only": setting.read_only}
    updates = payload.dict(exclude_unset=True)
    for field, value in updates.items():
        setattr(setting, field, value)
    setting.updated_by = current_user.id
    setting.updated_at = datetime.utcnow()
    _audit(db, actor=current_user, action="update_setting", entity_type="system_setting", entity_id=setting.id, old_value=before, new_value={"value": setting.setting_value, "description": setting.description}, request=request)
    db.commit()
    return {"id": setting.id, "section": setting.section, "key": setting.setting_key, "value": setting.setting_value, "read_only": setting.read_only}


@router.get("/audit")
def list_admin_audit(
    q: Optional[str] = None,
    action: Optional[str] = None,
    entity: Optional[str] = None,
    status_filter: Optional[str] = Query(default=None, alias="status"),
    export: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    query = db.query(AdminAuditLog).options(joinedload(AdminAuditLog.user)).order_by(AdminAuditLog.created_at.desc())
    if action:
        query = query.filter(AdminAuditLog.action == action)
    if entity:
        query = query.filter(AdminAuditLog.entity_type == entity)
    if status_filter:
        query = query.filter(AdminAuditLog.status == status_filter)
    if q:
        needle = f"%{q}%"
        query = query.filter(or_(AdminAuditLog.action.ilike(needle), AdminAuditLog.entity_type.ilike(needle), AdminAuditLog.entity_id.ilike(needle)))
    logs = query.limit(500).all()
    rows = [
        {
            "timestamp": item.created_at,
            "user": item.user.full_name if item.user else None,
            "action": item.action,
            "entity": item.entity_type,
            "entity_id": item.entity_id,
            "old_value": item.old_value_json,
            "new_value": item.new_value_json,
            "ip": item.ip_address,
            "status": item.status,
            "privacy_scope": item.privacy_scope,
        }
        for item in logs
    ]
    if export == "csv":
        header = "timestamp,user,action,entity,entity_id,ip,status,privacy_scope\n"
        body = "\n".join(
            f"{row['timestamp']},{row['user'] or ''},{row['action']},{row['entity']},{row['entity_id'] or ''},{row['ip'] or ''},{row['status']},{row['privacy_scope']}"
            for row in rows
        )
        return Response(content=header + body, media_type="text/csv")
    return {"audit_logs": rows}


def _build_report_summary(db: Session, report_type: str, parameters: dict | None = None) -> dict:
    if report_type == "university_summary":
        return {"universities": [_serialize_university(item, db) for item in db.query(University).order_by(University.university_name.asc()).all()]}
    if report_type == "counselor_summary":
        return {"counselors": [_profile_admin_payload(db, item) for item in db.query(CounselorProfile).order_by(CounselorProfile.full_name.asc()).all()]}
    if report_type == "assessment_summary":
        return {
            "profile_assessments": _count(db.query(ProfileAssessment)),
            "dass21_assessments": _count(db.query(DASS21Assessment)),
            "daily_checkins": _count(db.query(DailyCheckIn)),
            "legacy_assessments": _count(db.query(Assessment)),
        }
    if report_type == "fusion_summary":
        rows = db.query(RiskAssessment.risk_level, func.count(RiskAssessment.id)).group_by(RiskAssessment.risk_level).all()
        return {"total": _count(db.query(RiskAssessment)), "distribution": [{"risk_level": risk or "UNKNOWN", "count": int(count)} for risk, count in rows]}
    if report_type == "support_summary":
        return {
            "contacts": _count(db.query(SupportContact)),
            "active_contacts": _count(db.query(SupportContact).filter(SupportContact.active.is_(True))),
            "support_actions": _count(db.query(SupportContactAction)),
        }
    if report_type == "usage_summary":
        return {
            "users": _count(db.query(User)),
            "active_users": _count(db.query(User).filter(User.is_active.is_(True))),
            "logins_30d": _count(db.query(User).filter(User.last_login_at >= datetime.utcnow() - timedelta(days=30))),
            "storage_usage": _storage_usage(),
        }
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported report type")


@router.get("/reports")
def list_admin_reports(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(current_user)
    reports = db.query(AdminReport).order_by(AdminReport.created_at.desc()).all()
    return {
        "available_report_types": sorted(REPORT_TYPES),
        "reports": [
            {
                "id": item.id,
                "report_id": item.report_id,
                "report_type": item.report_type,
                "title": item.title,
                "status": item.status,
                "export_formats": item.export_formats_json or [],
                "created_at": item.created_at,
            }
            for item in reports
        ],
    }


@router.post("/reports")
def generate_admin_report(
    payload: ReportCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    if payload.report_type not in REPORT_TYPES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported report type")
    summary = _json_safe(_build_report_summary(db, payload.report_type, payload.parameters))
    report = AdminReport(
        report_id=public_id("report"),
        report_type=payload.report_type,
        title=payload.title or payload.report_type.replace("_", " ").title(),
        parameters_json=payload.parameters or {},
        summary_json=summary,
        export_formats_json=payload.export_formats,
        requested_by=current_user.id,
    )
    db.add(report)
    db.flush()
    _audit(db, actor=current_user, action="generate_report", entity_type="admin_report", entity_id=report.id, new_value={"report_type": payload.report_type}, request=request)
    db.commit()
    db.refresh(report)
    return {
        "id": report.id,
        "report_id": report.report_id,
        "report_type": report.report_type,
        "title": report.title,
        "summary": report.summary_json,
        "export_formats": report.export_formats_json,
        "status": report.status,
    }


@router.get("/reports/{report_id}/export")
def export_admin_report(
    report_id: int,
    format: str = Query(default="csv"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    require_admin(current_user)
    report = db.query(AdminReport).filter(AdminReport.id == report_id).first()
    if not report:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    if format not in {"csv", "pdf", "excel"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported export format")
    if format == "csv":
        lines = ["metric,value"]
        for key, value in (report.summary_json or {}).items():
            lines.append(f"{key},{str(value).replace(',', ';')}")
        return Response(content="\n".join(lines), media_type="text/csv")
    return {
        "report_id": report.report_id,
        "format": format,
        "status": "generated_future_ready",
        "summary": report.summary_json,
    }
