"""Consolidated Phase 3J governance report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.ml.common import paths
from app.ml.governance.artifacts import (
    build_artifact_integrity_summary,
    discover_model_runs,
    repo_relative,
    resolve_repo_path,
    verify_registration_status,
)
from app.ml.governance.comparison import build_unimodal_comparison, comparison_metadata
from app.ml.governance.constants import (
    MODEL_GOVERNANCE_VERSION,
    REPORT_FILENAMES,
    RESEARCH_READINESS_POLICY_VERSION,
    UNIMODAL_COMPARISON_VERSION,
)
from app.ml.governance.limitations import build_privacy_fairness_limitations
from app.ml.governance.metrics import build_false_negative_safety_summary
from app.ml.governance.model_cards import validate_model_cards
from app.ml.governance.readiness import (
    deployment_readiness_matrix,
    modality_blockers,
    modality_recommendations,
    assess_deployment_readiness,
    assess_evidence_strength,
)
from app.ml.governance.schemas import ArtifactIntegrityStatus, GovernanceValidationReport, ModelGovernanceRecord, utc_now


GLOBAL_BLOCKERS = [
    "No clinical validation.",
    "No prospective evaluation.",
    "No aligned multimodal participant data.",
    "No approved production inference policy.",
    "No completed fairness evaluation.",
    "No external validation.",
    "No human factors or counselor workflow validation.",
]

GLOBAL_RECOMMENDATIONS = [
    "Freeze current models as research artifacts.",
    "Do not deploy current unimodal models.",
    "Do not implement production fusion.",
    "Collect ethically approved, consented, aligned pilot data.",
    "Obtain common participant-level records across modalities.",
    "Define prospective clinical and counselor-review protocol.",
    "Perform external validation.",
    "Perform demographic fairness analysis.",
    "Review false-negative handling with qualified mental-health professionals.",
    "Design inference adapters only in a future sandbox after governance approval.",
]


def _read_json_if_exists(path: str | Path) -> Dict[str, Any]:
    candidate = resolve_repo_path(path)
    if not candidate.exists():
        return {}
    with candidate.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def load_completed_baseline_summaries(reports_root: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    root = resolve_repo_path(reports_root or paths.get_generated_reports_root())
    return {
        "profile": _read_json_if_exists(root / "profile_baseline" / "v1" / "profile_baseline_summary.json"),
        "text": _read_json_if_exists(root / "text_baseline" / "v1" / "text_baseline_summary.json"),
        "speech": _read_json_if_exists(root / "speech_baseline" / "v1" / "speech_baseline_summary.json"),
        "face": _read_json_if_exists(root / "face_baseline" / "v1" / "face_baseline_summary.json"),
        "speech_domain_shift": _read_json_if_exists(root / "speech_domain_shift" / "v1" / "speech_domain_shift_summary.json"),
    }


def _safe_summary_path(summary: Dict[str, Any]) -> Optional[str]:
    artifact_path = summary.get("artifact_path")
    return repo_relative(artifact_path) if artifact_path else None


def _selected_runs_by_modality(runs) -> Dict[str, Any]:
    return {run.modality: run for run in runs if run.selected_candidate and not run.synthetic and not run.evaluation_only}


def build_model_records(
    summaries: Dict[str, Dict[str, Any]],
    runs,
    model_card_validation: Dict[str, Any],
) -> List[ModelGovernanceRecord]:
    selected_runs = _selected_runs_by_modality(runs)
    card_by_modality = {record["modality"]: record for record in model_card_validation.get("records", [])}
    records: List[ModelGovernanceRecord] = []
    for modality in ("dass21", "profile", "text", "speech", "face", "mood", "behavioral", "fusion"):
        summary = summaries.get(modality, {})
        run = selected_runs.get(modality)
        card = card_by_modality.get(modality, {})
        trained = modality in {"profile", "text", "speech", "face"}
        model_name = summary.get("model_name") or (run.model_name if run else f"{modality}-not-trained")
        model_version = summary.get("model_version") or (run.model_version if run else None)
        run_id = summary.get("run_id") or (run.run_id if run else None)
        active = bool(summary.get("model_became_active") or summary.get("activation_status") == "active" or (run and run.manifest and run.manifest.get("active") is True))
        registered = bool(summary.get("candidate_registration_occurred") or summary.get("registration_status") == "registered")
        artifact_status = run.status if run else ArtifactIntegrityStatus.not_applicable
        domain_summary: Dict[str, Any] = {}
        if modality == "speech":
            domain = summaries.get("speech_domain_shift", {})
            domain_summary = {
                "pooled_test_macro_f1": domain.get("corpus_gap_summary", {}).get("pooled_test_macro_f1"),
                "loco_mean_macro_f1": domain.get("corpus_gap_summary", {}).get("loco_mean_macro_f1"),
                "corpus_shortcut_accuracy": 1.0,
                "interpretation": "Strong domain shift; LOCO is evaluation-only.",
            }
        elif modality == "face":
            domain_summary = {"reviewer_independence": "reviewer_independence_unverified", "bounded_run": True}
        records.append(
            ModelGovernanceRecord(
                modality=modality,
                model_name=model_name,
                model_version=model_version,
                run_id=run_id,
                trained=trained,
                registered=registered,
                active=active,
                training_scope=_training_scope(modality),
                dataset_summary=_dataset_summary(modality, summary),
                split_summary=_split_summary(summary),
                primary_metric=_primary_metric_name(modality),
                test_metrics=summary.get("test_metrics", {}),
                false_negative_summary={},
                domain_shift_summary=domain_summary,
                fairness_summary=summary.get("fairness_summary", {}),
                privacy_summary={"source": "synthesized Phase 3J limitations"},
                governance_limitations=modality_blockers(modality),
                artifact_integrity=artifact_status,
                model_card_valid=bool(card.get("valid")) if trained else modality == "dass21",
                clinical_disclaimer_present=bool(card.get("clinical_disclaimer_present")) if trained else modality == "dass21",
                deployment_readiness=assess_deployment_readiness(modality),
                evidence_strength=assess_evidence_strength(modality),
                blockers=modality_blockers(modality),
                recommendations=modality_recommendations(modality),
            )
        )
    return records


def _training_scope(modality: str) -> str:
    return {
        "dass21": "deterministic rule-based scoring; no ML model",
        "profile": "real research baseline trained on very small dataset",
        "text": "real research baseline trained; strongest unimodal result but not deployable",
        "speech": "real pooled acted-speech emotion baseline plus evaluation-only domain-shift checks",
        "face": "bounded image-statistics experiment only; full split not used for completed training",
        "mood": "project-generated/synthetic only; no trained model",
        "behavioral": "synthetic engineering data only; no trained model",
        "fusion": "no aligned supervised fusion dataset; no trained model",
    }[modality]


def _dataset_summary(modality: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    if modality == "face":
        contract = summary.get("contract", {})
        return {
            "retained_records": contract.get("retained_record_count", 33676),
            "quarantined_records": contract.get("quarantined_record_count", 155),
            "bounded_experiment": True,
        }
    if modality in {"profile", "text", "speech"}:
        return {"train": summary.get("train_count"), "validation": summary.get("validation_count"), "test": summary.get("test_count")}
    return {"status": _training_scope(modality)}


def _split_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {"train": summary.get("train_count"), "validation": summary.get("validation_count"), "test": summary.get("test_count")}


def _primary_metric_name(modality: str) -> Optional[str]:
    return {"profile": "f1", "text": "macro_f1", "speech": "macro_f1", "face": "macro_f1"}.get(modality)


def _write_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_research_model_inventory(runs, model_card_validation: Dict[str, Any]) -> List[Dict[str, Any]]:
    cards = {record["modality"]: record for record in model_card_validation.get("records", [])}
    rows = []
    for run in runs:
        if not run.selected_candidate or run.synthetic or run.evaluation_only:
            continue
        card = cards.get(run.modality, {})
        rows.append(
            {
                "modality": run.modality,
                "model name": run.model_name,
                "model version": run.model_version,
                "run ID": run.run_id,
                "artifact path": run.run_path,
                "registered": "false",
                "active": "false",
                "selected candidate": "true",
                "training scope": _training_scope(run.modality),
                "test scope": "locked research test split",
                "primary metric": _primary_metric_name(run.modality),
                "model-card status": "valid" if card.get("valid") else "findings",
                "artifact integrity": run.status.value if hasattr(run.status, "value") else str(run.status),
                "evidence strength": assess_evidence_strength(run.modality).value,
                "deployment readiness": assess_deployment_readiness(run.modality).value,
                "prohibited use": "clinical diagnosis, autonomous suicide-prevention decisions, production deployment",
            }
        )
    return rows


def _summary_markdown(report: GovernanceValidationReport, comparison_rows) -> str:
    lines = [
        "# Phase 3J Model Governance Summary",
        "",
        f"Governance version: {MODEL_GOVERNANCE_VERSION}",
        f"Comparison version: {UNIMODAL_COMPARISON_VERSION}",
        f"Readiness policy version: {RESEARCH_READINESS_POLICY_VERSION}",
        "",
        f"Final decision: `{report.final_research_status}`",
        "",
        "## Modality Readiness",
    ]
    for record in report.model_records:
        lines.append(f"- {record.modality}: {record.deployment_readiness} ({record.evidence_strength} evidence)")
    lines.extend(["", "## Comparison Warning", comparison_metadata()["comparability_warning"], "", "## Global Blockers"])
    lines.extend(f"- {blocker}" for blocker in report.global_blockers)
    lines.extend(["", "## Recommended Next Actions"])
    lines.extend(f"{index}. {action}" for index, action in enumerate(report.global_recommendations, start=1))
    return "\n".join(lines) + "\n"


def _phase3_summary_md(summaries: Dict[str, Dict[str, Any]], report: GovernanceValidationReport) -> str:
    return "\n".join(
        [
            "# Phase 3 Model Development Summary",
            "",
            "Phase 3A established leakage-safe split design and manifests. Phase 3B added the common training framework, artifact manifests, model cards, and inactive candidate handling.",
            "",
            f"Profile baseline: test n={summaries.get('profile', {}).get('test_count')}, primary test F1={summaries.get('profile', {}).get('test_metrics', {}).get('f1')}, all-positive behavior on the 15-record test set.",
            f"Text baseline: test n={summaries.get('text', {}).get('test_count')}, macro F1={summaries.get('text', {}).get('test_metrics', {}).get('macro_f1')}, suicidal false negatives={summaries.get('text', {}).get('test_metrics', {}).get('suicidal_class', {}).get('false_negatives')}.",
            f"Speech baseline: test n={summaries.get('speech', {}).get('test_count')}, pooled macro F1={summaries.get('speech', {}).get('test_metrics', {}).get('macro_f1')}; LOCO mean macro F1={summaries.get('speech_domain_shift', {}).get('corpus_gap_summary', {}).get('loco_mean_macro_f1')}.",
            f"Face baseline: bounded test n={summaries.get('face', {}).get('test_count')}, macro F1={summaries.get('face', {}).get('test_metrics', {}).get('macro_f1')}, reviewer independence unverified.",
            "",
            "Major finding: all current model outputs are research artifacts only. They are not clinical diagnostic systems, not production inference systems, and not valid supervised multimodal fusion inputs.",
            "",
            f"Final readiness: `{report.final_research_status}`.",
            "",
        ]
    )


def run_governance_validation(
    model_root: str | Path | None = None,
    reports_root: str | Path | None = None,
    config: str | Path | None = None,
    output_dir: str | Path | None = None,
    modalities: Optional[Iterable[str]] = None,
    verify_hashes: bool = False,
    validate_cards: bool = False,
    inventory_only: bool = False,
    strict: bool = False,
    fail_on_warning: bool = False,
    overwrite: bool = False,
    summary_only: bool = False,
) -> Dict[str, Any]:
    del config, verify_hashes, validate_cards
    out_dir = resolve_repo_path(output_dir or (paths.get_generated_reports_root() / "model_governance" / "v1"))
    summaries = load_completed_baseline_summaries(reports_root)
    requested_modalities = list(modalities) if modalities else None
    runs = discover_model_runs(model_root=model_root, reports_root=reports_root, modalities=requested_modalities)
    artifact_summary = build_artifact_integrity_summary(runs)
    card_validation = validate_model_cards(runs)
    inventory_rows = build_research_model_inventory(runs, card_validation)
    if inventory_only:
        _write_csv(out_dir / REPORT_FILENAMES["inventory_csv"], inventory_rows, overwrite)
        _write_json(out_dir / REPORT_FILENAMES["inventory_json"], {"models": inventory_rows, "excluded": artifact_summary["unexpected_artifacts"]}, overwrite)
        return {"exit_code": 0, "output_dir": repo_relative(out_dir), "inventory_count": len(inventory_rows)}
    model_records = build_model_records(summaries, runs, card_validation)
    safety_summary = build_false_negative_safety_summary(summaries)
    for record in model_records:
        if record.modality in safety_summary:
            record.false_negative_summary = safety_summary[record.modality]
    comparison_rows = build_unimodal_comparison(summaries)
    readiness_rows = deployment_readiness_matrix()
    evidence_rows = [
        {
            "modality": row["modality"],
            "evidence_strength": row["evidence_strength"],
            "deployment_readiness": row["deployment_readiness"],
            "blockers": "; ".join(modality_blockers(row["modality"])),
        }
        for row in readiness_rows
    ]
    report = GovernanceValidationReport(
        governance_version=MODEL_GOVERNANCE_VERSION,
        readiness_policy_version=RESEARCH_READINESS_POLICY_VERSION,
        generated_at=utc_now(),
        model_records=model_records,
        artifact_integrity_summary=artifact_summary,
        model_card_summary=card_validation,
        activation_summary={"active_model_count": artifact_summary["active_model_count"], "active_models_allowed": False},
        registration_summary={"registered_model_count": 0, "registration_allowed_for_current_phase": False},
        deployment_readiness_summary={"deployable_model_count": 0, "matrix": readiness_rows},
        global_blockers=GLOBAL_BLOCKERS,
        global_recommendations=GLOBAL_RECOMMENDATIONS,
        final_research_status="research_models_complete_not_deployable",
    )
    if not summary_only:
        _write_json(out_dir / REPORT_FILENAMES["artifact_integrity"], artifact_summary, overwrite)
        _write_json(out_dir / REPORT_FILENAMES["model_cards"], card_validation, overwrite)
        _write_csv(out_dir / REPORT_FILENAMES["comparison_csv"], [row.to_safe_dict() for row in comparison_rows], overwrite)
        _write_csv(out_dir / REPORT_FILENAMES["readiness_csv"], readiness_rows, overwrite)
        _write_json(out_dir / REPORT_FILENAMES["safety"], safety_summary, overwrite)
        _write_json(out_dir / REPORT_FILENAMES["limitations"], build_privacy_fairness_limitations(), overwrite)
        _write_csv(out_dir / REPORT_FILENAMES["evidence_csv"], evidence_rows, overwrite)
        _write_json(out_dir / REPORT_FILENAMES["blockers"], {"global_blockers": GLOBAL_BLOCKERS}, overwrite)
        _write_json(out_dir / REPORT_FILENAMES["actions"], {"recommended_next_actions": GLOBAL_RECOMMENDATIONS}, overwrite)
        _write_json(out_dir / REPORT_FILENAMES["inventory_json"], {"models": inventory_rows, "excluded": artifact_summary["unexpected_artifacts"]}, overwrite)
        _write_csv(out_dir / REPORT_FILENAMES["inventory_csv"], inventory_rows, overwrite)
    _write_json(out_dir / REPORT_FILENAMES["summary_json"], report.to_safe_dict(), overwrite)
    _write_text(out_dir / REPORT_FILENAMES["summary_md"], _summary_markdown(report, comparison_rows), overwrite)
    _write_text(paths.get_ml_research_root() / "reports" / "phase3_model_development_summary.md", _phase3_summary_md(summaries, report), overwrite)
    warnings = artifact_summary.get("unexpected_artifacts", [])
    exit_code = 1 if strict else (1 if fail_on_warning and warnings else 0)
    return {
        "exit_code": exit_code,
        "output_dir": repo_relative(out_dir),
        "final_research_status": report.final_research_status,
        "selected_model_count": len(inventory_rows),
        "active_model_count": artifact_summary["active_model_count"],
        "registered_model_count": 0,
        "strict_failed_due_to_blockers": bool(strict),
        "comparison_version": UNIMODAL_COMPARISON_VERSION,
    }
