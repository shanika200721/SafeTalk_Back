"""Deterministic Phase 3J evidence and readiness rules."""

from __future__ import annotations

from typing import Dict, List

from app.ml.governance.schemas import DeploymentReadiness, EvidenceStrength


READINESS_BY_MODALITY: Dict[str, DeploymentReadiness] = {
    "dass21": DeploymentReadiness.scoring_only,
    "profile": DeploymentReadiness.research_baseline_only,
    "text": DeploymentReadiness.research_evaluated_not_deployable,
    "speech": DeploymentReadiness.research_evaluated_not_deployable,
    "face": DeploymentReadiness.research_baseline_only,
    "mood": DeploymentReadiness.blocked_pending_data,
    "behavioral": DeploymentReadiness.engineering_only,
    "fusion": DeploymentReadiness.blocked_pending_data,
}

EVIDENCE_BY_MODALITY: Dict[str, EvidenceStrength] = {
    "dass21": EvidenceStrength.none,
    "profile": EvidenceStrength.very_low,
    "text": EvidenceStrength.low,
    "speech": EvidenceStrength.low,
    "face": EvidenceStrength.very_low,
    "mood": EvidenceStrength.none,
    "behavioral": EvidenceStrength.none,
    "fusion": EvidenceStrength.none,
}

BLOCKERS_BY_MODALITY: Dict[str, List[str]] = {
    "dass21": ["Rule-based scoring only; no ML prediction artifact and no clinical diagnosis."],
    "profile": [
        "Tiny sample size with unstable 15-record validation/test sets.",
        "Self-reported depression target is not an authoritative suicide-risk outcome.",
        "Weak all-positive decision behavior makes recall misleading.",
    ],
    "text": [
        "Incomplete author grouping.",
        "Weak non-clinical labels and social-media domain mismatch.",
        "Suicidal false-negative burden and shortcut terms detected.",
        "Privacy normalization is incomplete.",
    ],
    "speech": [
        "Acted speech corpus rather than natural clinical speech.",
        "Corpus shortcut diagnostic accuracy 1.0.",
        "Poor LOCO generalization and no natural Sri Lankan speech validation.",
    ],
    "face": [
        "Bounded image-statistics experiment only.",
        "Severe overfitting with full split not used for completed training.",
        "No subject IDs; reviewer independence unverified.",
        "Biometric privacy and perceptual near-duplicate uncertainty.",
    ],
    "mood": ["No reviewed real longitudinal source; project-generated or synthetic data only."],
    "behavioral": ["Synthetic engineering data only; covert-monitoring risk requires explicit consent before real use."],
    "fusion": ["No aligned participant records and no valid supervised fusion dataset."],
}

RECOMMENDATIONS_BY_MODALITY: Dict[str, List[str]] = {
    modality: [
        "Freeze current output as a research artifact.",
        "Do not deploy or activate this modality.",
        "Revisit only after ethically approved data, external validation, and governance approval.",
    ]
    for modality in READINESS_BY_MODALITY
}


def assess_deployment_readiness(modality: str) -> DeploymentReadiness:
    return READINESS_BY_MODALITY[modality.lower()]


def assess_evidence_strength(modality: str) -> EvidenceStrength:
    return EVIDENCE_BY_MODALITY[modality.lower()]


def modality_blockers(modality: str) -> List[str]:
    return list(BLOCKERS_BY_MODALITY.get(modality.lower(), []))


def modality_recommendations(modality: str) -> List[str]:
    return list(RECOMMENDATIONS_BY_MODALITY.get(modality.lower(), []))


def deployment_readiness_matrix() -> List[Dict[str, str]]:
    rows = []
    for modality, readiness in READINESS_BY_MODALITY.items():
        rows.append(
            {
                "modality": modality,
                "deployment_readiness": readiness.value,
                "evidence_strength": EVIDENCE_BY_MODALITY[modality].value,
                "deployable": "false",
                "primary_blocker": BLOCKERS_BY_MODALITY[modality][0],
            }
        )
    return rows
