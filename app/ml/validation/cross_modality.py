"""Cross-modality consistency validation for Phase 2 preprocessing outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from app.ml.common import paths
from app.ml.validation import checks as common_checks
from app.ml.validation.constants import KNOWN_MODALITIES, PHASE2_VALIDATION_VERSION, READINESS_POLICY_VERSION
from app.ml.validation.inventory import create_phase2_artifact_inventory
from app.ml.validation.modality import validate_modalities
from app.ml.validation.schemas import CrossModalityValidationReport, ValidationCheckResult, ValidationSeverity, ValidationStatus


def _repo_path(path: str | Path | None, default: Path) -> Path:
    candidate = Path(path) if path is not None else default
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _global_checks(generated_root: Path, config_dir: Path, modality_results) -> list[ValidationCheckResult]:
    checks: list[ValidationCheckResult] = [
        common_checks.check_model_artifacts_absent(generated_root, modality="global"),
        common_checks.check_split_manifest_absent(generated_root, modality="global"),
        common_checks.passed("no_machine_specific_absolute_paths", "global", "Per-report privacy checks enforce repository-relative paths."),
        common_checks.passed("no_path_traversal", "global", "Per-report privacy checks enforce no path traversal references."),
    ]

    names = [result.modality for result in modality_results]
    if len(names) == len(set(names)):
        checks.append(common_checks.passed("consistent_modality_names", "global", "Modality names are unique and consistent."))
    else:
        checks.append(
            common_checks.failed(
                "consistent_modality_names",
                "global",
                "Duplicate modality names were detected.",
                "Ensure each modality has one canonical validation result.",
            )
        )

    versions = [f"{result.modality}:{result.dataset_version}:{result.preprocessing_version}:{result.feature_schema_version}" for result in modality_results]
    if len(versions) == len(set(versions)):
        checks.append(common_checks.passed("unique_modality_versions", "global", "Dataset/preprocessing/schema version identities are unique by modality."))
    else:
        checks.append(
            common_checks.warning(
                "unique_modality_versions",
                "global",
                "Duplicate version identities were detected across modalities.",
                "Confirm that duplicate versions are intentional and documented.",
            )
        )

    label_names = {tuple(result.target_columns) for result in modality_results if result.target_columns}
    if len(label_names) > 1:
        checks.append(
            common_checks.warning(
                "consistent_canonical_label_naming",
                "global",
                "Canonical target label names differ across disconnected modality datasets.",
                "Do not assume labels are interchangeable; document modality-specific label semantics.",
            )
        )
    else:
        checks.append(common_checks.passed("consistent_canonical_label_naming", "global", "Canonical label naming is consistent or not applicable."))

    checks.append(
        common_checks.blocked(
            "no_row_level_multimodal_fusion",
            "fusion",
            "Current offline datasets are disconnected and cannot support row-level supervised multimodal fusion.",
            "Use production modality predictions, synthetic engineering data, or a future ethically collected aligned pilot dataset for fusion work.",
        )
    )
    return checks


def validate_phase2_cross_modality(
    *,
    config_dir: str | Path | None = None,
    generated_root: str | Path | None = None,
    modalities: Iterable[str] | None = None,
    skip_source_reverification: bool = False,
) -> tuple[CrossModalityValidationReport, list]:
    selected = [modality for modality in (modalities or KNOWN_MODALITIES)]
    unknown = sorted(set(selected) - set(KNOWN_MODALITIES))
    if unknown:
        raise ValueError(f"Unknown Phase 2 modalities: {unknown}")

    config_path = _repo_path(config_dir, paths.get_ml_research_root() / "configs")
    generated_path = _repo_path(generated_root, paths.get_generated_root())
    modality_results = validate_modalities(
        selected,
        config_dir=config_path,
        generated_root=generated_path,
        skip_source_reverification=skip_source_reverification,
    )
    global_checks = _global_checks(generated_path, config_path, modality_results)
    all_checks = [check for result in modality_results for check in result.checks] + global_checks

    passed_checks = sum(1 for check in all_checks if str(check.status) == ValidationStatus.PASSED.value)
    warning_checks = sum(1 for check in all_checks if str(check.severity) == ValidationSeverity.WARNING.value)
    failed_checks = sum(1 for check in all_checks if str(check.status) == ValidationStatus.FAILED.value)
    blocked_checks = sum(1 for check in all_checks if str(check.status) == ValidationStatus.BLOCKED.value)

    global_findings = [
        "Current offline datasets are disconnected.",
        "There is no common participant key across modalities.",
        "Supervised multimodal fusion training is not valid from these datasets.",
        "Later fusion must use production modality predictions, synthetic engineering data, or a future ethically collected pilot dataset.",
    ]
    global_blockers = [check.message for check in global_checks if str(check.status) == ValidationStatus.BLOCKED.value]
    blockers_present = global_blockers or any(result.blockers for result in modality_results)
    phase2_status = "preprocessing_foundation_complete_with_documented_blockers" if blockers_present else "complete_ready_for_next_phase_design"
    recommendations = [
        "Proceed to Phase 3A split design only for eligible restricted modalities: profile, text, and speech.",
        "Keep DASS21 as rule-based scoring, not ML training.",
        "Resolve face duplicate/conflict leakage before split design.",
        "Accept or replace mood real-data policy before any model-training design.",
        "Collect consented real behavioral data before behavioral model work.",
        "Do not create multimodal fusion training data until aligned participant-level records exist.",
    ]
    report = CrossModalityValidationReport(
        validation_version=PHASE2_VALIDATION_VERSION,
        readiness_policy_version=READINESS_POLICY_VERSION,
        modalities=modality_results,
        total_checks=len(all_checks),
        passed_checks=passed_checks,
        warning_checks=warning_checks,
        failed_checks=failed_checks,
        blocked_checks=blocked_checks,
        global_findings=global_findings,
        global_blockers=global_blockers,
        global_recommendations=recommendations,
        phase2_completion_status=phase2_status,
        next_phase_recommendation="Phase 3A split design for eligible modalities only; no fusion or model training yet.",
    )
    inventory = create_phase2_artifact_inventory(modalities=selected, generated_root=generated_path)
    return report, inventory
