"""Typed Phase 3J governance schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic.v1 import BaseModel, Field, validator


class ArtifactIntegrityStatus(str, Enum):
    verified = "verified"
    missing = "missing"
    hash_mismatch = "hash_mismatch"
    malformed = "malformed"
    not_applicable = "not_applicable"


class DeploymentReadiness(str, Enum):
    not_trained = "not_trained"
    scoring_only = "scoring_only"
    engineering_only = "engineering_only"
    research_baseline_only = "research_baseline_only"
    research_evaluated_not_deployable = "research_evaluated_not_deployable"
    blocked_pending_data = "blocked_pending_data"
    blocked_pending_governance = "blocked_pending_governance"
    prohibited_for_deployment = "prohibited_for_deployment"


class EvidenceStrength(str, Enum):
    none = "none"
    very_low = "very_low"
    low = "low"
    moderate = "moderate"
    high = "high"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class GovernanceBaseModel(BaseModel):
    class Config:
        use_enum_values = True
        validate_assignment = True

    def to_safe_dict(self) -> Dict[str, Any]:
        return self.dict()


class ModelGovernanceRecord(GovernanceBaseModel):
    modality: str
    model_name: str
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    trained: bool
    registered: bool
    active: bool
    training_scope: str
    dataset_summary: Dict[str, Any] = Field(default_factory=dict)
    split_summary: Dict[str, Any] = Field(default_factory=dict)
    primary_metric: Optional[str] = None
    test_metrics: Dict[str, Any] = Field(default_factory=dict)
    false_negative_summary: Dict[str, Any] = Field(default_factory=dict)
    domain_shift_summary: Dict[str, Any] = Field(default_factory=dict)
    fairness_summary: Dict[str, Any] = Field(default_factory=dict)
    privacy_summary: Dict[str, Any] = Field(default_factory=dict)
    governance_limitations: List[str] = Field(default_factory=list)
    artifact_integrity: ArtifactIntegrityStatus = ArtifactIntegrityStatus.not_applicable
    model_card_valid: bool = False
    clinical_disclaimer_present: bool = False
    deployment_readiness: DeploymentReadiness
    evidence_strength: EvidenceStrength
    blockers: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    @validator("active")
    def no_active_models(cls, value: bool) -> bool:
        if value:
            raise ValueError("Phase 3J governance records cannot mark models active")
        return value

    @validator("deployment_readiness")
    def no_deployable_status(cls, value: str) -> str:
        if value == "deployable":
            raise ValueError("Deployable status is prohibited")
        return value

    @validator("recommendations")
    def blockers_require_recommendations(cls, value: List[str], values: Dict[str, Any]) -> List[str]:
        if values.get("blockers") and not value:
            raise ValueError("Governance blockers require recommendations")
        return value


class UnimodalComparisonRecord(GovernanceBaseModel):
    modality: str
    task: str
    label_type: str
    dataset_size: Optional[int] = None
    train_count: Optional[int] = None
    validation_count: Optional[int] = None
    test_count: Optional[int] = None
    feature_type: str
    selected_estimator: str
    test_primary_metric: Optional[float] = None
    test_macro_f1: Optional[float] = None
    test_balanced_accuracy: Optional[float] = None
    safety_relevant_metric: str
    false_negative_count: Optional[int] = None
    domain_shift_evaluated: bool
    scope_limitation: str
    comparability_warning: str


class GovernanceValidationReport(GovernanceBaseModel):
    governance_version: str
    readiness_policy_version: str
    generated_at: datetime
    model_records: List[ModelGovernanceRecord]
    artifact_integrity_summary: Dict[str, Any]
    model_card_summary: Dict[str, Any]
    activation_summary: Dict[str, Any]
    registration_summary: Dict[str, Any]
    deployment_readiness_summary: Dict[str, Any]
    global_blockers: List[str]
    global_recommendations: List[str]
    final_research_status: str

    @validator("generated_at")
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("generated_at must be timezone-aware")
        return value
