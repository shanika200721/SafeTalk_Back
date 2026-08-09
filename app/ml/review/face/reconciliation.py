"""Deterministic reconciliation for Phase 3H face review decisions."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import timezone
from pathlib import Path
from typing import Any

from app.ml.common import paths
from app.ml.remediation.face.reporting import artifact_inventory, write_json
from app.ml.review.face.constants import FACE_RECONCILIATION_POLICY_VERSION
from app.ml.review.face.policy import load_review_policy
from app.ml.review.face.schemas import FaceReconciledDecision, FaceReviewerDecision, utc_now


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _load_json(path: str | Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def load_review_items(review_package: str | Path) -> list[dict[str, Any]]:
    package_path = _resolve(review_package)
    if package_path.is_dir():
        package_path = package_path / "face_review_items.json"
    return list(_load_json(package_path).get("review_items", []))


def load_validated_decisions(decisions_dir: str | Path) -> list[FaceReviewerDecision]:
    source = _resolve(decisions_dir)
    if source.is_dir():
        source = source / "reviewer_decisions_validated.json"
    payload = _load_json(source)
    return [FaceReviewerDecision(**item) for item in payload.get("reviewer_decisions", [])]


def detect_reviewer_disagreement(decisions: list[FaceReviewerDecision]) -> bool:
    return len({decision.decision for decision in decisions}) > 1


def determine_consensus(decisions: list[FaceReviewerDecision], required_reviewers: int, policy: dict[str, Any]) -> tuple[bool, str | None]:
    aliases = {decision.reviewer_alias for decision in decisions}
    if len(aliases) < required_reviewers:
        return False, None
    decision_values = {decision.decision for decision in decisions}
    if policy.get("conflict_resolution_rule") == "unanimous_consensus_required_no_majority_vote" and len(decision_values) != 1:
        return False, None
    if len(decision_values) == 1:
        return True, next(iter(decision_values))
    return False, None


def determine_final_action(item: dict[str, Any], consensus_decision: str | None, policy: dict[str, Any]) -> tuple[str, str, bool]:
    item_type = item["item_type"]
    if consensus_decision is None:
        return "additional_review", "insufficient_consensus_or_reviewer_disagreement", False
    if item_type == "cross_label_conflict":
        if consensus_decision in {"label_conflict_unresolved", "retain_quarantine", "ambiguous", "corrupted"}:
            return "keep_quarantined", f"cross_label_{consensus_decision}", False
        if consensus_decision == "source_label_likely_incorrect":
            restore = policy.get("restore_policy", {})
            if restore.get("allowed_for_cross_label_conflicts") and restore.get("reviewer_consensus_required"):
                return "restore_record", "consensus_source_label_likely_incorrect_restore_requires_later_label_policy", True
        if consensus_decision == "recommend_restore_same_label_representative":
            return "restore_record", "consensus_restore_recommendation_without_label_change", False
        return "keep_quarantined", f"cross_label_default_quarantine_for_{consensus_decision}", consensus_decision == "source_label_likely_incorrect"
    if item_type == "perceptual_duplicate_candidate":
        if consensus_decision in {"confirm_exact_duplicate", "confirm_near_duplicate"}:
            return "retain_existing_representative", f"perceptual_{consensus_decision}_future_exclusion_candidate", False
        if consensus_decision == "not_duplicate":
            return "unresolved", "perceptual_not_duplicate_records_unchanged", False
        if consensus_decision in {"ambiguous", "requires_additional_review"}:
            return "additional_review", f"perceptual_{consensus_decision}", False
    return "unresolved", f"no_automatic_action_for_{consensus_decision}", False


def reconcile_review_item(item: dict[str, Any], decisions: list[FaceReviewerDecision], policy: dict[str, Any]) -> FaceReconciledDecision:
    required = int(item.get("required_reviewers") or policy.get("minimum_reviewer_count", 2))
    consensus, consensus_decision = determine_consensus(decisions, required, policy)
    final_action, reason, label_change_recommended = determine_final_action(item, consensus_decision, policy)
    record_ids = list(item.get("record_ids") or [])
    if final_action == "restore_record":
        retained = sorted(record_ids)
        quarantined: list[str] = []
    elif final_action in {"keep_quarantined", "additional_review", "unresolved"} and item["item_type"] == "cross_label_conflict":
        retained = []
        quarantined = sorted(record_ids)
    else:
        retained = sorted(record_ids)
        quarantined = []
    excluded: list[str] = []
    return FaceReconciledDecision(
        review_item_id=item["review_item_id"],
        final_status="consensus" if consensus else ("disagreement" if detect_reviewer_disagreement(decisions) else "unresolved"),
        final_action=final_action,
        retained_record_ids=retained,
        excluded_record_ids=excluded,
        quarantined_record_ids=quarantined,
        label_change_recommended=label_change_recommended,
        consensus_reached=consensus,
        reconciliation_reason=reason,
        policy_version=policy["policy_version"],
    )


def reconcile_all_reviews(
    *,
    review_package: str | Path,
    decisions_dir: str | Path,
    policy_config_path: str | Path | None = None,
) -> list[FaceReconciledDecision]:
    policy = load_review_policy(policy_config_path)
    items = load_review_items(review_package)
    decisions = load_validated_decisions(decisions_dir)
    by_item: dict[str, list[FaceReviewerDecision]] = defaultdict(list)
    for decision in decisions:
        by_item[decision.review_item_id].append(decision)
    return [
        reconcile_review_item(item, sorted(by_item.get(item["review_item_id"], []), key=lambda value: value.reviewer_alias), policy)
        for item in sorted(items, key=lambda value: value["review_item_id"])
    ]


def _ensure_output_dir(output_dir: str | Path) -> Path:
    output = _resolve(output_dir)
    paths.assert_not_raw_dataset_path(output)
    if not paths.is_path_inside(paths.get_generated_root() / "review", output):
        raise ValueError("face reconciliation output must be under generated/review/")
    output.mkdir(parents=True, exist_ok=True)
    return output


def create_reconciliation_manifest(
    *,
    reconciled_decisions: list[FaceReconciledDecision],
    output_dir: str | Path,
    source_fingerprint: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = _ensure_output_dir(output_dir)
    payload = {
        "reconciliation_policy_version": FACE_RECONCILIATION_POLICY_VERSION,
        "source_fingerprint": source_fingerprint,
        "reconciled_at": utc_now().astimezone(timezone.utc).isoformat(),
        "reconciled_decisions": [decision.to_safe_dict() for decision in reconciled_decisions],
    }
    outputs: dict[str, Path] = {}
    outputs["manifest"] = write_json(output / "face_reconciliation_manifest.json", payload, overwrite=overwrite)
    outputs["inventory"] = write_json(output / "face_reconciliation_artifact_inventory.json", {"artifacts": artifact_inventory(outputs)}, overwrite=overwrite)
    return {"payload": payload, "outputs": outputs}

