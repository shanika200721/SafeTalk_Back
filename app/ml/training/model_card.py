"""Model-card generation for local research candidate artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.ml.training.constants import CLINICAL_DISCLAIMER
from app.ml.training.schemas import ModelCard, TrainingConfig, utc_now


_MODALITY_LIMITATIONS = {
    "profile": [
        "Small dataset; validation and test metrics may be unstable.",
        "Target is self-reported depression, not suicide-risk ground truth.",
        "Profile features are not sufficient for autonomous clinical decisions.",
    ],
    "text": [
        "Social-media domain mismatch limits applicability to student support settings.",
        "Author grouping remains incomplete in current artifacts.",
        "Labels are weak, non-clinical annotations and duplicate restrictions remain documented.",
    ],
    "speech": [
        "Speech corpora are acted emotion datasets.",
        "Corpus, device, accent, language, and speaker bias may remain.",
        "Emotion labels are not depression or suicide-risk ground truth.",
    ],
}


def modality_specific_limitations(modality: str) -> list[str]:
    return list(_MODALITY_LIMITATIONS.get(modality.lower(), []))


def build_model_card(
    *,
    config: TrainingConfig,
    metrics: Mapping[str, Any],
    split_summary: str,
    dataset_summary: str,
    preprocessing_summary: str,
    extra_limitations: list[str] | None = None,
) -> ModelCard:
    limitations = modality_specific_limitations(config.modality) + list(extra_limitations or [])
    if not limitations:
        limitations = ["No modality-specific limitations were supplied; require human review before use."]
    return ModelCard(
        model_name=config.model_name,
        model_version=config.model_version,
        modality=config.modality,
        intended_use="Research baseline comparison on locked train/validation/test splits only.",
        prohibited_use="Clinical diagnosis, autonomous suicide-prevention decisions, alerts, treatment recommendations, or production student prediction.",
        dataset_summary=dataset_summary,
        preprocessing_summary=preprocessing_summary,
        split_summary=split_summary,
        model_description=f"{config.framework} {config.estimator_type} candidate trained through the common Phase 3B framework.",
        hyperparameters=config.hyperparameters,
        metrics=dict(metrics),
        threshold_policy=f"{config.threshold_strategy.value}; thresholds are not clinically validated.",
        fairness_considerations="Evaluate subgroup performance before any real-world use; sensitive attributes are documented but not used as predictive features by this framework.",
        privacy_considerations="Artifacts and reports must exclude raw text, audio, profile-sensitive values, production identifiers, secrets, and absolute local paths.",
        limitations=limitations,
        ethical_warnings=[
            "False negatives are safety-relevant and must remain visible in reports.",
            "Model output must be interpreted by qualified humans and never used as a standalone crisis decision.",
            "Known leakage restrictions from Phase 3A remain binding.",
        ],
        human_oversight_requirement="Human oversight is mandatory for any interpretation, research review, or future deployment decision.",
        clinical_disclaimer=CLINICAL_DISCLAIMER,
        created_at=utc_now(),
    )


def model_card_to_markdown(card: ModelCard) -> str:
    lines = [
        f"# Model Card: {card.model_name} {card.model_version}",
        "",
        f"- Version: `{card.model_card_version}`",
        f"- Modality: `{card.modality}`",
        f"- Created: `{card.created_at.isoformat()}`",
        "",
        "## Clinical Disclaimer",
        card.clinical_disclaimer,
        "",
        "## Intended Use",
        card.intended_use,
        "",
        "## Prohibited Use",
        card.prohibited_use,
        "",
        "## Dataset",
        card.dataset_summary,
        "",
        "## Preprocessing",
        card.preprocessing_summary,
        "",
        "## Split",
        card.split_summary,
        "",
        "## Performance",
        "```json",
        json.dumps(card.metrics, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Threshold Policy",
        card.threshold_policy,
        "",
        "## Fairness",
        card.fairness_considerations,
        "",
        "## Privacy",
        card.privacy_considerations,
        "",
        "## Limitations",
    ]
    lines.extend(f"- {item}" for item in card.limitations)
    lines.extend(["", "## Ethical Warnings"])
    lines.extend(f"- {item}" for item in card.ethical_warnings)
    lines.extend(["", "## Human Oversight", card.human_oversight_requirement, ""])
    return "\n".join(lines)


def save_model_card(card: ModelCard, run_dir: str | Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    run_path = Path(run_dir)
    json_path = run_path / "model_card.json"
    md_path = run_path / "model_card.md"
    if not overwrite and (json_path.exists() or md_path.exists()):
        raise FileExistsError("Refusing to overwrite model card artifacts")
    json_path.write_text(json.dumps(card.to_safe_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(model_card_to_markdown(card), encoding="utf-8")
    return json_path, md_path
