"""Constants for Phase 3A split design."""

SPLIT_DESIGN_VERSION = "1.0.0"
SPLIT_MANIFEST_VERSION = "1.0.0"

ALLOWED_SPLIT_MODALITIES = ("profile", "text", "speech")
FORBIDDEN_SPLIT_MODALITIES = ("mood", "face", "behavioral", "dass21", "DASS21", "fusion")

SPLIT_NAMES = ("train", "validation", "test")

DEFAULT_OUTPUT_FILES = {
    "manifest": "{modality}_split_manifest.json",
    "assignments": "{modality}_split_assignments.csv",
    "report_json": "{modality}_split_report.json",
    "report_markdown": "{modality}_split_report.md",
    "exclusions": "{modality}_split_exclusions.json",
}

PHASE3A_REPORT_FILES = {
    "validation_json": "phase3a_split_validation.json",
    "validation_markdown": "phase3a_split_validation.md",
    "matrix": "phase3a_split_matrix.csv",
    "blockers": "phase3a_blockers.json",
    "next_actions": "phase3a_next_actions.json",
}
