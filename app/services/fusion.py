from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.ml.fusion.weighted_late_fusion import classify_score, weighted_score
from app.models.database_models import ModalityPrediction, RiskAssessment, RiskAssessmentInput


CONFIG_VERSION = "controlled-late-fusion-v2"
THRESHOLD_VERSION = "v2"
ASSESSMENT_TYPE = "screening_support"
LIMITATIONS = [
    "This result is a screening-support signal and not a clinical diagnosis.",
    "The fused score is model output only and is not a counselor decision.",
    "Speech, face and behavioral evidence are not used in Phase 4E.",
]

ARTIFACT_KEY_BY_MODALITY = {
    "profile": "profile_score",
    "dass21": "dass_score",
    "mood": "mood_score",
    "text": "text_score",
    "speech": "speech_score",
    "face": "face_score",
}
MODALITY_BY_ARTIFACT_KEY = {value: key for key, value in ARTIFACT_KEY_BY_MODALITY.items()}
CONFIGURED_MODALITIES = ["profile", "dass21", "mood", "text", "speech", "face"]
RUNTIME_DISABLED_MODALITIES = {"speech", "face"}
EXCLUDED_MODALITIES = {"behavioral"}


@dataclass(frozen=True)
class SelectedPrediction:
    prediction: ModalityPrediction
    mapped_score: float
    source_score: float
    source_timestamp: datetime
    prediction_age_seconds: float
    mapping_version: str


@dataclass(frozen=True)
class ExcludedPrediction:
    modality: str
    reason: str
    prediction: Optional[ModalityPrediction] = None
    source_score: Optional[float] = None
    mapped_score: Optional[float] = None
    source_timestamp: Optional[datetime] = None
    prediction_age_seconds: Optional[float] = None
    mapping_version: Optional[str] = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _artifact_dir() -> Path:
    return _repo_root() / "ml_models" / "fusion" / "controlled-late-fusion-v2" / "2.0.0"


def _mapping_config() -> dict[str, Any]:
    return _load_json(Path(__file__).resolve().parents[1] / "ml" / "fusion" / "runtime_mapping_v1.json")


def _policy_config() -> dict[str, Any]:
    return _load_json(Path(__file__).resolve().parents[1] / "ml" / "fusion" / "runtime_policy_v1.json")


def _weights() -> dict[str, float]:
    payload = _load_json(_artifact_dir() / "fusion_weights.json")["weights"]
    return {MODALITY_BY_ARTIFACT_KEY[key]: float(value) for key, value in payload.items()}


def _thresholds() -> dict[str, float]:
    return {key: float(value) for key, value in _load_json(_artifact_dir() / "fusion_thresholds.json")["thresholds"].items()}


def _canonical_config() -> dict[str, Any]:
    return {
        "config_version": CONFIG_VERSION,
        "threshold_version": THRESHOLD_VERSION,
        "weights": _weights(),
        "thresholds": _thresholds(),
        "mapping": _mapping_config(),
        "policy": _policy_config(),
    }


def configuration_hash() -> str:
    encoded = json.dumps(_canonical_config(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _storage_datetime(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    return _as_utc(value).replace(tzinfo=None)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return _as_utc(value).isoformat()
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def _now_utc(assessment_time: Optional[datetime] = None) -> datetime:
    return _as_utc(assessment_time) or datetime.now(timezone.utc)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in {float("inf"), float("-inf")}:
        return None
    return numeric


def _mapping_for(modality: str, output_type: str) -> dict[str, Any]:
    mapping = _mapping_config()["mappings"].get(modality, {})
    if modality == "profile":
        selected = dict(mapping.get(output_type, {}))
        selected["accepted_output_types"] = [
            item for item in mapping.get("accepted_output_types", []) if item == output_type
        ]
        return selected
    return mapping


def _mapping_versions() -> dict[str, str]:
    versions = {}
    for modality in CONFIGURED_MODALITIES:
        mapping = _mapping_config()["mappings"][modality]
        if modality == "profile":
            versions[modality] = ",".join(
                item["mapping_version"] for item in mapping.values() if isinstance(item, dict) and item.get("mapping_version")
            )
        else:
            versions[modality] = mapping.get("mapping_version", "")
    return versions


def _source_value(prediction: ModalityPrediction, field_name: str) -> Optional[float]:
    if field_name == "score_0_100":
        return _safe_float(prediction.score_0_100)
    if field_name == "probability":
        return _safe_float(prediction.probability)
    return None


def _normalize(prediction: ModalityPrediction) -> tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    mapping = _mapping_for(prediction.modality, prediction.output_type)
    accepted = mapping.get("accepted_output_types")
    if accepted is not None and prediction.output_type not in accepted:
        return None, None, f"output_type_not_allowed:{prediction.output_type}", None
    if not mapping:
        return None, None, "mapping_not_available", None

    source = _source_value(prediction, mapping.get("source_field"))
    if source is None:
        return None, None, "source_score_missing", mapping.get("mapping_version")
    lower, upper = mapping["input_range"]
    if source < float(lower) or source > float(upper):
        return None, source, "source_score_out_of_range", mapping.get("mapping_version")
    if mapping["transformation"] == "divide_by_100":
        mapped = source / 100.0
    elif mapping["transformation"] == "identity":
        mapped = source
    else:
        return None, source, "unsupported_transformation", mapping.get("mapping_version")
    if mapped < 0.0 or mapped > 1.0:
        return None, source, "mapped_score_out_of_range", mapping.get("mapping_version")
    return float(mapped), float(source), None, mapping.get("mapping_version")


def _failure(
    modality: str,
    reason: str,
    prediction: Optional[ModalityPrediction] = None,
    *,
    source_score: Optional[float] = None,
    mapped_score: Optional[float] = None,
    source_timestamp: Optional[datetime] = None,
    prediction_age_seconds: Optional[float] = None,
    mapping_version: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "modality": modality,
        "prediction_id": prediction.id if prediction else None,
        "reason": reason,
        "status": prediction.status if prediction else None,
        "is_available": prediction.is_available if prediction else None,
        "source_score": source_score,
        "mapped_score": mapped_score,
        "source_timestamp": source_timestamp,
        "prediction_age_seconds": prediction_age_seconds,
        "mapping_version": mapping_version,
    }


def _is_stale(prediction: ModalityPrediction, now: datetime) -> tuple[bool, str, Optional[datetime], Optional[float]]:
    source_time = _as_utc(prediction.source_timestamp or prediction.generated_at or prediction.created_at)
    generated_at = _as_utc(prediction.generated_at or prediction.created_at)
    if source_time is None or generated_at is None:
        return True, "timestamp_missing", source_time, None
    if prediction.valid_until and now > _as_utc(prediction.valid_until):
        age = (now - source_time).total_seconds()
        return True, "prediction_expired", source_time, age
    window_days = float(_policy_config()["staleness_windows_days"][prediction.modality])
    age = (now - source_time).total_seconds()
    if age < 0:
        return True, "source_timestamp_in_future", source_time, age
    if age > window_days * 86400:
        return True, "prediction_stale", source_time, age
    return False, "", source_time, age


def _check_prediction(
    prediction: ModalityPrediction,
    user_id: int,
    now: datetime,
) -> tuple[Optional[SelectedPrediction], Optional[ExcludedPrediction]]:
    if prediction.student_id != user_id:
        return None, ExcludedPrediction(prediction.modality, "wrong_user", prediction)
    if prediction.modality in EXCLUDED_MODALITIES:
        return None, ExcludedPrediction(prediction.modality, "modality_not_validated", prediction)
    if prediction.modality in RUNTIME_DISABLED_MODALITIES:
        return None, ExcludedPrediction(prediction.modality, "runtime_modality_unavailable", prediction)
    if prediction.status != "succeeded":
        return None, ExcludedPrediction(prediction.modality, f"status_{prediction.status}", prediction)
    if not prediction.is_available:
        return None, ExcludedPrediction(prediction.modality, "prediction_unavailable", prediction)
    if not prediction.evidence_available:
        return None, ExcludedPrediction(prediction.modality, "source_evidence_missing", prediction)
    if not prediction.consent_policy_version:
        return None, ExcludedPrediction(prediction.modality, "consent_policy_missing", prediction)
    if not prediction.model_version or not prediction.preprocessing_version:
        return None, ExcludedPrediction(prediction.modality, "model_or_preprocessing_version_missing", prediction)

    stale, reason, source_time, age = _is_stale(prediction, now)
    if stale:
        return None, ExcludedPrediction(
            prediction.modality,
            reason,
            prediction,
            source_timestamp=source_time,
            prediction_age_seconds=age,
        )
    mapped, source, reason, mapping_version = _normalize(prediction)
    if reason:
        return None, ExcludedPrediction(
            prediction.modality,
            reason,
            prediction,
            source_score=source,
            source_timestamp=source_time,
            prediction_age_seconds=age,
            mapping_version=mapping_version,
        )
    return SelectedPrediction(
        prediction=prediction,
        mapped_score=mapped,
        source_score=source,
        source_timestamp=source_time,
        prediction_age_seconds=age or 0.0,
        mapping_version=mapping_version or _mapping_config()["version"],
    ), None


def _select_predictions(db: Session, user_id: int, now: datetime) -> tuple[dict[str, SelectedPrediction], list[dict[str, Any]], list[str]]:
    selected: dict[str, SelectedPrediction] = {}
    excluded: list[dict[str, Any]] = []
    missing: list[str] = []

    for modality in CONFIGURED_MODALITIES:
        if modality in RUNTIME_DISABLED_MODALITIES:
            missing.append(modality)
            excluded.append(_failure(modality, "runtime_modality_unavailable"))
            continue
        candidates = (
            db.query(ModalityPrediction)
            .filter(ModalityPrediction.student_id == user_id, ModalityPrediction.modality == modality)
            .order_by(ModalityPrediction.created_at.desc(), ModalityPrediction.id.desc())
            .all()
        )
        if not candidates:
            missing.append(modality)
            excluded.append(_failure(modality, "missing_prediction"))
            continue
        for candidate in candidates:
            checked, failed = _check_prediction(candidate, user_id, now)
            if checked:
                selected[modality] = checked
                break
            excluded.append(
                _failure(
                    modality,
                    failed.reason if failed else "not_eligible",
                    candidate,
                    source_score=failed.source_score if failed else None,
                    mapped_score=failed.mapped_score if failed else None,
                    source_timestamp=failed.source_timestamp if failed else None,
                    prediction_age_seconds=failed.prediction_age_seconds if failed else None,
                    mapping_version=failed.mapping_version if failed else None,
                )
            )
        if modality not in selected and modality not in missing:
            missing.append(modality)

    excluded.append(_failure("behavioral", "modality_not_validated"))
    return selected, excluded, missing


def _coverage_category(base_coverage: float, modality_count: int) -> str:
    policy = _policy_config()["minimum_evidence_policy"]
    if modality_count < int(policy["minimum_modality_count"]) or base_coverage < float(policy["minimum_base_weight_coverage"]):
        return "insufficient"
    if base_coverage < 0.5:
        return "limited"
    if base_coverage < 0.9:
        return "moderate"
    return "strong"


def _build_response(
    *,
    assessment_id: Optional[int],
    user_id: int,
    status_value: str,
    score: Optional[float],
    risk_level: Optional[str],
    selected: dict[str, SelectedPrediction],
    excluded: list[dict[str, Any]],
    missing: list[str],
    effective_weights: dict[str, float],
    base_coverage: float,
    coverage_category: str,
) -> dict[str, Any]:
    source_times = [item.source_timestamp for item in selected.values()]
    latest_source = max(source_times) if source_times else None
    oldest_source = min(source_times) if source_times else None
    evidence_window = None
    if latest_source and oldest_source:
        evidence_window = {
            "oldest": oldest_source.isoformat(),
            "latest": latest_source.isoformat(),
            "span_seconds": (latest_source - oldest_source).total_seconds(),
        }

    inputs = [
        {
            "modality": modality,
            "prediction_id": item.prediction.id,
            "mapped_score": item.mapped_score,
            "base_weight": _weights()[modality],
            "effective_weight": effective_weights.get(modality),
            "included": True,
            "exclusion_reason": None,
            "source_timestamp": item.source_timestamp,
            "prediction_age_seconds": item.prediction_age_seconds,
        }
        for modality, item in sorted(selected.items())
    ]
    inputs.extend(
        {
            "modality": item["modality"],
            "prediction_id": item["prediction_id"],
            "mapped_score": item.get("mapped_score"),
            "base_weight": _weights().get(item["modality"]),
            "effective_weight": None,
            "included": False,
            "exclusion_reason": item["reason"],
            "source_timestamp": item.get("source_timestamp"),
            "prediction_age_seconds": item.get("prediction_age_seconds"),
        }
        for item in sorted(excluded, key=lambda value: (value["modality"], value["prediction_id"] or 0, value["reason"]))
        if item["prediction_id"] is not None
    )
    return {
        "assessment_id": assessment_id,
        "user_id": user_id,
        "status": status_value,
        "score": score,
        "risk_level": risk_level,
        "assessment_type": ASSESSMENT_TYPE,
        "fusion": {
            "config_version": CONFIG_VERSION,
            "config_hash": configuration_hash(),
            "threshold_version": THRESHOLD_VERSION,
            "mapping_version": _mapping_config()["version"],
            "coverage_policy_version": _policy_config()["coverage_policy_version"],
            "staleness_policy_version": _policy_config()["staleness_policy_version"],
        },
        "evidence": {
            "configured_modalities": CONFIGURED_MODALITIES,
            "available_modalities": sorted(selected),
            "used_modalities": sorted(selected),
            "missing_modalities": sorted(set(missing)),
            "excluded_modalities": excluded,
            "base_weight_coverage": base_coverage,
            "effective_weight_total": round(sum(effective_weights.values()), 12) if effective_weights else 0.0,
            "modality_count": len(selected),
            "latest_source_timestamp": latest_source,
            "oldest_source_timestamp": oldest_source,
            "evidence_window": evidence_window,
            "coverage_category": coverage_category,
            "minimum_evidence_met": coverage_category != "insufficient",
        },
        "inputs": inputs,
        "limitations": LIMITATIONS,
        "model_risk_level": risk_level,
        "model_score": score,
        "human_review_status": "not_requested",
        "counselor_decision": None,
        "counselor_override": None,
        "alert_created": False,
    }


def controlled_fusion_config() -> dict[str, Any]:
    policy = _policy_config()
    return {
        "config_version": CONFIG_VERSION,
        "config_hash": configuration_hash(),
        "modalities": CONFIGURED_MODALITIES,
        "base_weights": _weights(),
        "thresholds": _thresholds(),
        "threshold_version": THRESHOLD_VERSION,
        "mapping_versions": _mapping_versions(),
        "staleness_policy_version": policy["staleness_policy_version"],
        "staleness_windows_days": policy["staleness_windows_days"],
        "coverage_policy_version": policy["coverage_policy_version"],
        "coverage_categories": policy["coverage_categories"],
        "minimum_evidence_policy": policy["minimum_evidence_policy"],
        "limitations": LIMITATIONS,
    }


def run_controlled_fusion(
    db: Session,
    *,
    user_id: int,
    persist: bool = True,
    assessment_time: Optional[datetime] = None,
) -> dict[str, Any]:
    now = _now_utc(assessment_time)
    base_weights = _weights()
    selected, excluded, missing = _select_predictions(db, user_id, now)
    base_coverage = round(sum(base_weights[modality] for modality in selected), 12)
    coverage_category = _coverage_category(base_coverage, len(selected))

    score = None
    risk_level = None
    effective_weights: dict[str, float] = {}
    status_value = "insufficient_evidence" if coverage_category == "insufficient" else "completed"
    if status_value == "completed":
        effective_weights = {
            modality: base_weights[modality] / base_coverage
            for modality in sorted(selected)
        }
        score_inputs = {
            ARTIFACT_KEY_BY_MODALITY[modality]: selected[modality].mapped_score if modality in selected else None
            for modality in CONFIGURED_MODALITIES
        }
        score_weights = {
            ARTIFACT_KEY_BY_MODALITY[modality]: base_weights[modality]
            for modality in CONFIGURED_MODALITIES
        }
        score = round(weighted_score(score_inputs, score_weights, min_available_modalities=2), 12)
        risk_level = classify_score(score, _thresholds()).lower()

    assessment = None
    if persist:
        source_times = [item.source_timestamp for item in selected.values()]
        latest_source = max(source_times) if source_times else None
        oldest_source = min(source_times) if source_times else None
        response_preview = _build_response(
            assessment_id=None,
            user_id=user_id,
            status_value=status_value,
            score=score,
            risk_level=risk_level,
            selected=selected,
            excluded=excluded,
            missing=missing,
            effective_weights=effective_weights,
            base_coverage=base_coverage,
            coverage_category=coverage_category,
        )
        assessment = RiskAssessment(
            student_id=user_id,
            final_probability=score,
            final_score=round(score * 100, 6) if score is not None else None,
            risk_level=risk_level,
            confidence=None,
            status=status_value,
            assessment_type=ASSESSMENT_TYPE,
            model_score=score,
            model_risk_level=risk_level,
            fusion_config_version=CONFIG_VERSION,
            fusion_config_hash=configuration_hash(),
            threshold_version=THRESHOLD_VERSION,
            mapping_version=_mapping_config()["version"],
            staleness_policy_version=_policy_config()["staleness_policy_version"],
            coverage_policy_version=_policy_config()["coverage_policy_version"],
            configured_modalities=CONFIGURED_MODALITIES,
            available_modalities=sorted(selected),
            used_modalities=sorted(selected),
            missing_modalities=sorted(set(missing)),
            excluded_modalities=_json_safe(excluded),
            evidence_coverage=base_coverage,
            coverage_category=coverage_category,
            effective_weights=effective_weights,
            latest_source_timestamp=_storage_datetime(latest_source),
            oldest_source_timestamp=_storage_datetime(oldest_source),
            evidence_window_json=response_preview["evidence"]["evidence_window"],
            limitations_json=LIMITATIONS,
            screening_only=True,
            model_output_only=True,
            human_review_status="not_requested",
            counselor_decision=None,
            counselor_override=None,
            alert_created=False,
            data_completeness=_json_safe(response_preview["evidence"]),
            explanation_json={"fusion": response_preview["fusion"], "limitations": LIMITATIONS},
            created_at=_storage_datetime(now),
        )
        db.add(assessment)
        db.flush()
        for modality, item in sorted(selected.items()):
            db.add(
                RiskAssessmentInput(
                    risk_assessment_id=assessment.id,
                    modality_prediction_id=item.prediction.id,
                    modality=modality,
                    source_score=item.source_score,
                    mapped_score=item.mapped_score,
                    base_weight=base_weights[modality],
                    effective_weight=effective_weights.get(modality),
                    included=True,
                    source_timestamp=_storage_datetime(item.source_timestamp),
                    prediction_age_seconds=item.prediction_age_seconds,
                    mapping_version=item.mapping_version,
                    metadata_json={"source_output_type": item.prediction.output_type},
                )
            )
        for item in sorted(excluded, key=lambda value: (value["modality"], value["prediction_id"] or 0, value["reason"])):
            if item["prediction_id"] is None:
                continue
            modality = item["modality"]
            db.add(
                RiskAssessmentInput(
                    risk_assessment_id=assessment.id,
                    modality_prediction_id=item["prediction_id"],
                    modality=modality,
                    source_score=item.get("source_score"),
                    mapped_score=item.get("mapped_score"),
                    base_weight=base_weights.get(modality),
                    effective_weight=None,
                    included=False,
                    exclusion_reason=item["reason"],
                    source_timestamp=_storage_datetime(item.get("source_timestamp")),
                    prediction_age_seconds=item.get("prediction_age_seconds"),
                    mapping_version=item.get("mapping_version"),
                    metadata_json={"status": item.get("status"), "is_available": item.get("is_available")},
                )
            )
        db.commit()
        db.refresh(assessment)

    return _build_response(
        assessment_id=assessment.id if assessment else None,
        user_id=user_id,
        status_value=status_value,
        score=score,
        risk_level=risk_level,
        selected=selected,
        excluded=excluded,
        missing=missing,
        effective_weights=effective_weights,
        base_coverage=base_coverage,
        coverage_category=coverage_category,
    )


def serialize_assessment(assessment: RiskAssessment) -> dict[str, Any]:
    inputs = [
        {
            "modality": item.modality,
            "prediction_id": item.modality_prediction_id,
            "mapped_score": item.mapped_score,
            "base_weight": item.base_weight,
            "effective_weight": item.effective_weight,
            "included": item.included,
            "exclusion_reason": item.exclusion_reason,
            "source_timestamp": item.source_timestamp,
            "prediction_age_seconds": item.prediction_age_seconds,
        }
        for item in sorted(assessment.inputs, key=lambda value: (value.modality or "", value.id))
    ]
    return {
        "assessment_id": assessment.id,
        "user_id": assessment.student_id,
        "status": assessment.status,
        "score": assessment.model_score,
        "risk_level": assessment.model_risk_level,
        "assessment_type": assessment.assessment_type,
        "fusion": {
            "config_version": assessment.fusion_config_version,
            "config_hash": assessment.fusion_config_hash,
            "threshold_version": assessment.threshold_version,
            "mapping_version": assessment.mapping_version,
            "coverage_policy_version": assessment.coverage_policy_version,
            "staleness_policy_version": assessment.staleness_policy_version,
        },
        "evidence": {
            "configured_modalities": assessment.configured_modalities or [],
            "available_modalities": assessment.available_modalities or [],
            "used_modalities": assessment.used_modalities or [],
            "missing_modalities": assessment.missing_modalities or [],
            "excluded_modalities": assessment.excluded_modalities or [],
            "base_weight_coverage": assessment.evidence_coverage or 0.0,
            "effective_weight_total": round(sum((assessment.effective_weights or {}).values()), 12),
            "modality_count": len(assessment.used_modalities or []),
            "latest_source_timestamp": assessment.latest_source_timestamp,
            "oldest_source_timestamp": assessment.oldest_source_timestamp,
            "evidence_window": assessment.evidence_window_json,
            "coverage_category": assessment.coverage_category or "insufficient",
            "minimum_evidence_met": assessment.status != "insufficient_evidence",
        },
        "inputs": inputs,
        "limitations": assessment.limitations_json or LIMITATIONS,
        "model_risk_level": assessment.model_risk_level,
        "model_score": assessment.model_score,
        "human_review_status": assessment.human_review_status,
        "counselor_decision": assessment.counselor_decision,
        "counselor_override": assessment.counselor_override,
        "alert_created": assessment.alert_created,
    }
