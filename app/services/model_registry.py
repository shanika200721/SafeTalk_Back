from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib
import json

import joblib
import pandas as pd
import sklearn

from sqlalchemy.orm import Session

from app.models.database_models import ModelRegistry, ModalityPrediction

REPO_ROOT = Path(__file__).resolve().parents[3]
APPROVED_MODEL_ROOT = REPO_ROOT / "ml_models"
SUPPORTED_SERIALIZERS = {"joblib"}
RUNTIME_SUPPORTED_MODALITIES = {"profile", "text", "speech", "face"}
REGISTRY_STATUSES = {
    "discovered",
    "verified",
    "active",
    "inactive",
    "rejected",
    "corrupt",
    "incompatible",
}

APPROVED_RUNTIME_ARTIFACTS = [
    "ml_models/profile/profile-depression-random-forest/1.0.0/profile-minimal_contextual-66e36ed73f40/pipeline.joblib",
    "ml_models/text/text-classification-logistic-regression/1.0.0/text-e8d74030dfff/pipeline.joblib",
    "ml_models/speech/speech-emotion-random-forest/1.0.0/speech-99cdbe8dbb57/pipeline.joblib",
    "ml_models/face/face-emotion-random-forest/1.0.0/face-image_statistics-b9d5c76172fc/pipeline.joblib",
]


@dataclass
class VerificationResult:
    passed: bool
    failure_code: Optional[str] = None
    failure_message_safe: Optional[str] = None
    actual_hash: Optional[str] = None
    expected_hash: Optional[str] = None
    serializer: Optional[str] = None
    metadata_complete: bool = False
    smoke_test_status: str = "not_run"
    activation_eligible: bool = False
    details: Optional[dict] = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["details"] = payload.get("details") or {}
        return payload


def _safe_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _require_approved_artifact_path(path: Path) -> None:
    try:
        path.resolve().relative_to(APPROVED_MODEL_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Artifact path is outside approved model directories") from exc


def calculate_sha256(path: Path | str) -> str:
    resolved = _resolve_repo_path(str(path))
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_serializer(path: Path | str) -> Optional[str]:
    suffix = Path(path).suffix.lower()
    if suffix == ".joblib":
        return "joblib"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    if suffix in {".keras", ".h5"}:
        return "keras"
    if suffix in {".pt", ".pth"}:
        return "torch"
    if suffix == ".onnx":
        return "onnx"
    return None


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_for_artifact(artifact_path: Path) -> Optional[dict]:
    return _read_json(artifact_path.parent / "artifact_manifest.json")


def _expected_hash_from_manifest(artifact_path: Path, manifest: Optional[dict]) -> Optional[str]:
    if not manifest:
        return None
    relative = _safe_repo_relative(artifact_path)
    file_hashes = manifest.get("file_hashes") or {}
    return file_hashes.get(relative) or file_hashes.get(relative.replace("/", "\\"))


def _required_metadata_paths(artifact_path: Path) -> dict[str, Path]:
    parent = artifact_path.parent
    return {
        "manifest": parent / "artifact_manifest.json",
        "metrics": parent / "metrics.json",
        "training_config": parent / "training_config.json",
        "feature_schema": parent / "feature_schema.json",
        "model_card": parent / "model_card.md",
    }


def _metadata_for_artifact(artifact_path: Path) -> dict:
    parent = artifact_path.parent
    manifest = _read_json(parent / "artifact_manifest.json") or {}
    metrics = _read_json(parent / "metrics.json") or {}
    training_config = _read_json(parent / "training_config.json") or {}
    feature_schema = _read_json(parent / "feature_schema.json") or {}
    split_reference = _read_json(parent / "split_manifest_reference.json") or {}
    return {
        "manifest": manifest,
        "metrics": metrics,
        "training_config": training_config,
        "feature_schema": feature_schema,
        "split_reference": split_reference,
    }


def _label_mapping_for_model(model) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        for step in reversed(model.named_steps.values()):
            classes = getattr(step, "classes_", None)
            if classes is not None:
                break
    return [str(item) for item in classes] if classes is not None else []


def _smoke_input_for_modality(modality: str, feature_schema: dict):
    if modality == "profile":
        return pd.DataFrame([{"year_of_study": "year 1"}])
    if modality == "text":
        return ["I feel overwhelmed but safe today."]
    if modality == "speech":
        names = feature_schema.get("selected_features") or feature_schema.get("transformed_feature_names") or []
        row = {name: 0.0 for name in names}
        row["duration_seconds"] = 2.0
        return pd.DataFrame([row])
    if modality == "face":
        names = feature_schema.get("feature_names") or []
        defaults = {
            "mean_intensity": 128.0,
            "std_intensity": 24.0,
            "contrast": 48.0,
            "edge_density": 0.03,
            "entropy": 4.0,
        }
        return pd.DataFrame([{name: defaults.get(name, 0.0) for name in names}])
    return None


def _run_smoke_prediction(model, modality: str, feature_schema: dict) -> dict:
    smoke_input = _smoke_input_for_modality(modality, feature_schema)
    if smoke_input is None:
        raise ValueError("No smoke input defined for modality")
    prediction = model.predict(smoke_input)
    labels = _label_mapping_for_model(model)
    result = {"predicted_label": str(prediction[0]), "label_mapping": labels}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(smoke_input)
        result["probability_supported"] = True
        result["probability_count"] = int(probabilities.shape[1])
    else:
        result["probability_supported"] = False
    return result


def verify_model_artifact(model: ModelRegistry) -> VerificationResult:
    artifact_path = _resolve_repo_path(model.artifact_path)
    serializer = model.serializer or infer_serializer(artifact_path)
    if not artifact_path.exists():
        return VerificationResult(
            passed=False,
            failure_code="ARTIFACT_NOT_FOUND",
            failure_message_safe="Model artifact file was not found.",
            serializer=serializer,
            details={"artifact": _safe_repo_relative(artifact_path)},
        )
    try:
        _require_approved_artifact_path(artifact_path)
    except ValueError:
        return VerificationResult(
            passed=False,
            failure_code="ARTIFACT_NOT_FOUND",
            failure_message_safe="Model artifact is outside approved repository artifact directories.",
            serializer=serializer,
        )
    if serializer not in SUPPORTED_SERIALIZERS:
        return VerificationResult(
            passed=False,
            failure_code="UNSUPPORTED_SERIALIZER",
            failure_message_safe="Model serializer is not supported for Phase 4D runtime loading.",
            serializer=serializer,
        )

    actual_hash = calculate_sha256(artifact_path)
    manifest = _manifest_for_artifact(artifact_path)
    expected_hash = model.artifact_sha256 or _expected_hash_from_manifest(artifact_path, manifest)
    if not expected_hash:
        return VerificationResult(
            passed=False,
            failure_code="HASH_MISMATCH",
            failure_message_safe="Expected artifact hash is missing.",
            actual_hash=actual_hash,
            serializer=serializer,
        )
    if actual_hash != expected_hash:
        return VerificationResult(
            passed=False,
            failure_code="HASH_MISMATCH",
            failure_message_safe="Artifact hash does not match registry or manifest.",
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            serializer=serializer,
        )

    required_paths = _required_metadata_paths(artifact_path)
    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        return VerificationResult(
            passed=False,
            failure_code="METADATA_MISSING",
            failure_message_safe="Required model metadata is missing.",
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            serializer=serializer,
            details={"missing": missing},
        )

    feature_schema = _read_json(required_paths["feature_schema"])
    if not feature_schema:
        return VerificationResult(
            passed=False,
            failure_code="FEATURE_SCHEMA_MISSING",
            failure_message_safe="Feature schema is missing or unreadable.",
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            serializer=serializer,
        )

    try:
        loaded = joblib.load(artifact_path)
    except Exception:
        return VerificationResult(
            passed=False,
            failure_code="ARTIFACT_CORRUPT",
            failure_message_safe="Model artifact could not be deserialized.",
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            serializer=serializer,
            metadata_complete=True,
            smoke_test_status="not_run",
        )

    try:
        smoke = _run_smoke_prediction(loaded, model.modality, feature_schema)
    except Exception:
        return VerificationResult(
            passed=False,
            failure_code="SMOKE_TEST_FAILED",
            failure_message_safe="Runtime smoke prediction failed.",
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            serializer=serializer,
            metadata_complete=True,
            smoke_test_status="failed",
        )

    labels = smoke.get("label_mapping") or []
    if not labels:
        return VerificationResult(
            passed=False,
            failure_code="LABEL_MAPPING_MISSING",
            failure_message_safe="Classifier label mapping could not be verified.",
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            serializer=serializer,
            metadata_complete=True,
            smoke_test_status="passed",
        )

    activation_eligible = model.modality in RUNTIME_SUPPORTED_MODALITIES
    if not activation_eligible:
        return VerificationResult(
            passed=False,
            failure_code="PREPROCESSOR_MISSING",
            failure_message_safe="Artifact loaded, but this modality is not approved for Phase 4D runtime activation.",
            actual_hash=actual_hash,
            expected_hash=expected_hash,
            serializer=serializer,
            metadata_complete=True,
            smoke_test_status="passed",
            activation_eligible=False,
            details={
                "label_mapping": labels,
                "smoke": smoke,
                "runtime_supported_modalities": sorted(RUNTIME_SUPPORTED_MODALITIES),
            },
        )

    return VerificationResult(
        passed=True,
        actual_hash=actual_hash,
        expected_hash=expected_hash,
        serializer=serializer,
        metadata_complete=True,
        smoke_test_status="passed",
        activation_eligible=True,
        details={
            "label_mapping": labels,
            "smoke": smoke,
            "framework_version": sklearn.__version__,
            "feature_schema_path": _safe_repo_relative(required_paths["feature_schema"]),
        },
    )


def apply_verification_result(db: Session, model: ModelRegistry, result: VerificationResult) -> ModelRegistry:
    now = datetime.utcnow()
    model.artifact_sha256 = result.actual_hash or model.artifact_sha256
    model.serializer = result.serializer or model.serializer
    model.framework_version = model.framework_version or sklearn.__version__
    model.verification_checked_at = now
    model.verification_status = "passed" if result.passed else "failed"
    model.verification_failure_code = result.failure_code
    model.verification_message = result.failure_message_safe
    model.verification_json = result.to_dict()
    if result.passed:
        model.status = "verified"
    elif result.failure_code == "ARTIFACT_CORRUPT":
        model.status = "corrupt"
    elif result.failure_code in {"UNSUPPORTED_SERIALIZER", "DEPENDENCY_INCOMPATIBLE", "PREPROCESSOR_MISSING"}:
        model.status = "incompatible"
    else:
        model.status = "rejected"
    model.updated_at = now
    db.flush()
    return model


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
    artifact_sha256: Optional[str] = None,
    serializer: Optional[str] = None,
    framework_version: Optional[str] = None,
    preprocessing_version: Optional[str] = None,
    label_mapping_version: Optional[str] = None,
    training_dataset_identifier: Optional[str] = None,
    training_split_identifier: Optional[str] = None,
    evaluation_report_identifier: Optional[str] = None,
    model_card_path: Optional[str] = None,
    status: str = "discovered",
    notes: Optional[str] = None,
    limitations_json: Optional[list] = None,
    intended_use: Optional[str] = None,
    prohibited_use: Optional[str] = None,
    metadata_json: Optional[dict] = None,
    is_active: bool = False,
) -> ModelRegistry:
    if status not in REGISTRY_STATUSES:
        raise ValueError(f"Unsupported model registry status: {status}")
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
        artifact_sha256=artifact_sha256,
        serializer=serializer,
        framework_version=framework_version,
        preprocessing_version=preprocessing_version,
        label_mapping_version=label_mapping_version,
        training_dataset_identifier=training_dataset_identifier,
        training_split_identifier=training_split_identifier,
        evaluation_report_identifier=evaluation_report_identifier,
        model_card_path=model_card_path,
        status=status,
        notes=notes,
        limitations_json=limitations_json,
        intended_use=intended_use,
        prohibited_use=prohibited_use,
        metadata_json=metadata_json,
        is_active=False,
    )
    db.add(model)
    db.flush()

    if is_active:
        activate_model_version(db, model_name=model_name, modality=modality, version=version)
        db.refresh(model)

    return model


def register_or_update_candidate(db: Session, artifact_path: Path | str) -> ModelRegistry:
    resolved = _resolve_repo_path(str(artifact_path))
    _require_approved_artifact_path(resolved)
    metadata = _metadata_for_artifact(resolved)
    manifest = metadata["manifest"]
    training_config = metadata["training_config"]
    feature_schema = metadata["feature_schema"]
    expected_hash = _expected_hash_from_manifest(resolved, manifest)
    modality = manifest.get("modality") or resolved.parts[-6]
    model_name = manifest.get("model_name") or resolved.parts[-5]
    version = manifest.get("model_version") or resolved.parts[-4]
    run_id = manifest.get("run_id")

    model = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.model_name == model_name,
            ModelRegistry.modality == modality,
            ModelRegistry.version == version,
        )
        .one_or_none()
    )
    fields = {
        "framework": "sklearn",
        "artifact_path": _safe_repo_relative(resolved),
        "preprocessing_path": _safe_repo_relative(resolved.parent / "preprocessor.joblib")
        if (resolved.parent / "preprocessor.joblib").exists()
        else _safe_repo_relative(resolved),
        "dataset_version": manifest.get("source_fingerprint") or training_config.get("experiment_version"),
        "feature_schema_version": training_config.get("experiment_version") or manifest.get("manifest_version"),
        "metrics_json": metadata["metrics"],
        "artifact_sha256": expected_hash,
        "serializer": infer_serializer(resolved),
        "framework_version": sklearn.__version__,
        "preprocessing_version": str(manifest.get("preprocessing_artifact_hash") or training_config.get("feature_set") or "unknown"),
        "label_mapping_version": str(manifest.get("manifest_version") or "1.0.0"),
        "training_dataset_identifier": manifest.get("source_fingerprint"),
        "training_split_identifier": manifest.get("split_manifest_hash"),
        "evaluation_report_identifier": _safe_repo_relative(resolved.parent / "metrics.json"),
        "model_card_path": _safe_repo_relative(resolved.parent / "model_card.md"),
        "metadata_json": {
            "run_id": run_id,
            "training_config_path": _safe_repo_relative(resolved.parent / "training_config.json"),
            "feature_schema_path": _safe_repo_relative(resolved.parent / "feature_schema.json"),
            "artifact_manifest_path": _safe_repo_relative(resolved.parent / "artifact_manifest.json"),
            "feature_schema": feature_schema,
        },
    }
    if model is None:
        model = ModelRegistry(
            model_name=model_name,
            modality=modality,
            version=version,
            status="discovered",
            is_active=False,
            **fields,
        )
        db.add(model)
    else:
        for key, value in fields.items():
            setattr(model, key, value)
        if not model.status:
            model.status = "discovered"
        model.updated_at = datetime.utcnow()
    db.flush()
    return model


def discover_runtime_candidates(db: Session) -> list[ModelRegistry]:
    discovered = []
    for relative_path in APPROVED_RUNTIME_ARTIFACTS:
        artifact_path = _resolve_repo_path(relative_path)
        if artifact_path.exists():
            discovered.append(register_or_update_candidate(db, artifact_path))
    return discovered


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
    if model.verification_status != "passed" or model.status != "verified":
        raise ValueError("Model activation requires successful verification")

    now = datetime.utcnow()
    (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.modality == modality,
            ModelRegistry.is_active == True,
        )
        .update({"is_active": False, "status": "inactive", "updated_at": now}, synchronize_session="fetch")
    )
    db.flush()

    model.is_active = True
    model.status = "active"
    model.approved_at = model.approved_at or now
    model.approved_by = model.approved_by or "phase4d-governance"
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
    if model.status == "active":
        model.status = "inactive"
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


def get_model_by_id(db: Session, model_registry_id: int) -> Optional[ModelRegistry]:
    return db.query(ModelRegistry).filter(ModelRegistry.id == model_registry_id).one_or_none()


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
