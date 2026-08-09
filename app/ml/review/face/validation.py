"""Validation helpers for Phase 3H review inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ml.review.face.schemas import FaceReviewItem, FaceReviewerDecision, validate_no_forbidden_fields


def validate_review_item_payload(payload: dict[str, Any]) -> FaceReviewItem:
    validate_no_forbidden_fields(payload)
    return FaceReviewItem(**payload)


def validate_decision_payload(payload: dict[str, Any]) -> FaceReviewerDecision:
    validate_no_forbidden_fields(payload)
    return FaceReviewerDecision(**payload)


def assert_no_raw_image_outputs(output_dir: str | Path) -> None:
    forbidden = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    found = [path for path in Path(output_dir).rglob("*") if path.is_file() and path.suffix.lower() in forbidden]
    if found:
        raise ValueError(f"raw image files must not be written to review outputs: {found[:5]}")

