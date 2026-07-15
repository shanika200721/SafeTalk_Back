"""Constants for Phase 3J model governance reporting."""

MODEL_GOVERNANCE_VERSION = "1.0.0"
UNIMODAL_COMPARISON_VERSION = "1.0.0"
RESEARCH_READINESS_POLICY_VERSION = "1.0.0"

REQUIRED_CLINICAL_DISCLAIMER = (
    "This model is a research prototype and is not a clinical diagnostic "
    "or autonomous suicide-prevention system."
)

REAL_BASELINE_MODALITIES = ("profile", "text", "speech", "face")
ALL_GOVERNED_MODALITIES = ("dass21", "profile", "text", "speech", "face", "mood", "behavioral", "fusion")
SYNTHETIC_MODALITIES = {"synthetic"}
EVALUATION_ONLY_ROOT_MARKERS = ("speech-domain-evaluation",)

REPORT_FILENAMES = {
    "summary_json": "model_governance_summary.json",
    "summary_md": "model_governance_summary.md",
    "comparison_csv": "unimodal_model_comparison.csv",
    "artifact_integrity": "artifact_integrity_report.json",
    "model_cards": "model_card_validation.json",
    "readiness_csv": "deployment_readiness_matrix.csv",
    "safety": "false_negative_safety_summary.json",
    "limitations": "privacy_fairness_limitations.json",
    "evidence_csv": "research_evidence_matrix.csv",
    "blockers": "global_blockers.json",
    "actions": "recommended_next_actions.json",
    "inventory_json": "governance_artifact_inventory.json",
    "inventory_csv": "research_model_inventory.csv",
}
