"""Read-only model-card validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.ml.governance.artifacts import DiscoveredModelRun, repo_relative, resolve_repo_path
from app.ml.governance.constants import REQUIRED_CLINICAL_DISCLAIMER


REQUIRED_SECTION_TERMS = {
    "intended_use": ["intended"],
    "prohibited_use": ["prohibited"],
    "dataset_origin": ["dataset"],
    "split_strategy": ["split"],
    "target_label_meaning": ["target", "label"],
    "performance": ["performance", "metric"],
    "false_negative_concerns": ["false negative", "false-negative"],
    "privacy_risks": ["privacy"],
    "fairness_limitations": ["fairness"],
    "domain_shift_risks": ["domain"],
    "human_oversight": ["human", "oversight"],
    "activation_status": ["active", "activation"],
    "registration_status": ["registered", "registration"],
}

MODALITY_TERMS = {
    "profile": ["15-record", "all-positive"],
    "text": ["448", "shortcut"],
    "speech": ["loco", "domain"],
    "face": ["bounded", "reviewer independence", "no subject", "deployment"],
}


def validate_model_card(path: str | Path, modality: str) -> Dict[str, Any]:
    card_path = resolve_repo_path(path)
    if not card_path.exists():
        return {
            "path": repo_relative(card_path),
            "modality": modality,
            "valid": False,
            "clinical_disclaimer_present": False,
            "missing": ["model_card"],
            "recommendations": ["Create a model card before any future governance review."],
        }
    text = card_path.read_text(encoding="utf-8").lower()
    missing: List[str] = []
    for section, terms in REQUIRED_SECTION_TERMS.items():
        if not any(term in text for term in terms):
            missing.append(section)
    disclaimer_present = REQUIRED_CLINICAL_DISCLAIMER.lower() in text
    if not disclaimer_present:
        missing.append("clinical_disclaimer")
    for term in MODALITY_TERMS.get(modality.lower(), []):
        if term not in text:
            missing.append(f"modality_requirement:{term}")
    return {
        "path": repo_relative(card_path),
        "modality": modality,
        "valid": not missing,
        "clinical_disclaimer_present": disclaimer_present,
        "missing": missing,
        "recommendations": [f"Add or clarify: {item}" for item in missing],
    }


def validate_model_cards(runs: Iterable[DiscoveredModelRun]) -> Dict[str, Any]:
    records = []
    for run in runs:
        if run.synthetic or run.evaluation_only or not run.selected_candidate:
            continue
        records.append(validate_model_card(Path(run.run_path) / "model_card.md", run.modality))
    return {
        "validated_count": len(records),
        "valid_count": sum(1 for record in records if record["valid"]),
        "invalid_count": sum(1 for record in records if not record["valid"]),
        "records": records,
    }
