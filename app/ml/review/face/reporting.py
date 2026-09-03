"""Audit and summary reporting for Phase 3H face review."""

from __future__ import annotations

import json
from datetime import timezone
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.remediation.face.reporting import artifact_inventory, write_json
from app.ml.review.face.constants import FACE_REVIEW_DECISION_SCHEMA_VERSION, FACE_REVIEW_WORKFLOW_VERSION
from app.ml.review.face.schemas import FaceReconciledDecision, FaceReviewSummary, utc_now


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = _resolve(output_dir)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_root() / "review", output):
        raise ValueError("face review audit output must be under generated/review/")
    output.mkdir(parents=True, exist_ok=True)
    return output


def append_audit_log(path: str | Path, event: dict[str, Any]) -> Path:
    target = _resolve(path)
    paths.assert_not_raw_dataset_path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    return target


def build_review_summary(
    *,
    source_fingerprint: str,
    total_review_items: int,
    reconciled_decisions: list[FaceReconciledDecision],
) -> FaceReviewSummary:
    reviewed = sum(1 for item in reconciled_decisions if item.consensus_reached)
    pending = max(total_review_items - reviewed, 0)
    consensus = sum(1 for item in reconciled_decisions if item.consensus_reached)
    disagreement = sum(1 for item in reconciled_decisions if item.final_status == "disagreement")
    unresolved = sum(1 for item in reconciled_decisions if item.final_action in {"keep_quarantined", "unresolved", "additional_review"})
    restored = sum(len(item.retained_record_ids) for item in reconciled_decisions if item.final_action == "restore_record")
    retained_quarantine = sum(len(item.quarantined_record_ids) for item in reconciled_decisions)
    excluded = sum(len(item.excluded_record_ids) for item in reconciled_decisions)
    completion = 0.0 if total_review_items == 0 else round((reviewed / total_review_items) * 100, 2)
    return FaceReviewSummary(
        source_fingerprint=source_fingerprint,
        total_review_items=total_review_items,
        reviewed_items=reviewed,
        pending_items=pending,
        consensus_items=consensus,
        disagreement_items=disagreement,
        unresolved_items=unresolved,
        restored_record_count=restored,
        retained_quarantine_count=retained_quarantine,
        excluded_record_count=excluded,
        review_completion_percentage=completion,
    )


def summary_markdown(summary: dict[str, Any], readiness: str | None = None) -> str:
    return (
        "# Phase 3H Face Review Summary\n\n"
        f"- Workflow version: {summary.get('workflow_version', FACE_REVIEW_WORKFLOW_VERSION)}\n"
        f"- Decision schema version: {summary.get('decision_schema_version', FACE_REVIEW_DECISION_SCHEMA_VERSION)}\n"
        f"- Total review items: {summary.get('total_review_items', 0)}\n"
        f"- Reviewed items: {summary.get('reviewed_items', 0)}\n"
        f"- Pending items: {summary.get('pending_items', 0)}\n"
        f"- Consensus items: {summary.get('consensus_items', 0)}\n"
        f"- Disagreement items: {summary.get('disagreement_items', 0)}\n"
        f"- Unresolved items: {summary.get('unresolved_items', 0)}\n"
        f"- Restored records: {summary.get('restored_record_count', 0)}\n"
        f"- Retained quarantine records: {summary.get('retained_quarantine_count', 0)}\n"
        f"- Readiness: {readiness or 'not_evaluated'}\n\n"
        "No raw images are embedded in this report. Human review does not establish clinical truth.\n"
    )


def write_review_audit_artifacts(
    *,
    output_dir: str | Path,
    source_fingerprint: str,
    total_review_items: int,
    reconciled_decisions: list[FaceReconciledDecision],
    decisions_file: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = _ensure_output_dir(output_dir)
    summary = build_review_summary(
        source_fingerprint=source_fingerprint,
        total_review_items=total_review_items,
        reconciled_decisions=reconciled_decisions,
    )
    payload = summary.to_safe_dict()
    unresolved = [item.to_safe_dict() for item in reconciled_decisions if item.final_action in {"keep_quarantined", "unresolved", "additional_review"}]
    disagreement = [item.to_safe_dict() for item in reconciled_decisions if item.final_status == "disagreement"]
    label_reviews = [item.to_safe_dict() for item in reconciled_decisions if item.label_change_recommended]
    outputs: dict[str, Path] = {}
    outputs["summary_json"] = write_json(output / "face_review_summary.json", payload, overwrite=overwrite)
    summary_md = output / "face_review_summary.md"
    if summary_md.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {summary_md}")
    summary_md.write_text(summary_markdown(payload), encoding="utf-8")
    outputs["summary_md"] = summary_md
    outputs["unresolved"] = write_json(output / "face_unresolved_items.json", {"items": unresolved, "count": len(unresolved)}, overwrite=overwrite)
    outputs["disagreement"] = write_json(output / "face_disagreement_items.json", {"items": disagreement, "count": len(disagreement)}, overwrite=overwrite)
    outputs["label_reviews"] = write_json(output / "face_recommended_label_reviews.json", {"items": label_reviews, "count": len(label_reviews)}, overwrite=overwrite)
    audit_event = {
        "event_type": "phase3h_review_audit_snapshot",
        "workflow_version": FACE_REVIEW_WORKFLOW_VERSION,
        "source_fingerprint": source_fingerprint,
        "generated_at": utc_now().astimezone(timezone.utc).isoformat(),
        "decision_file_hash": sha256_file(decisions_file, allow_outside_project=True) if decisions_file else None,
    }
    outputs["audit_log"] = append_audit_log(output / "face_review_audit_log.jsonl", audit_event)
    outputs["inventory"] = write_json(output / "face_review_audit_artifact_inventory.json", {"artifacts": artifact_inventory(outputs)}, overwrite=overwrite)
    return {"summary": payload, "outputs": outputs}

