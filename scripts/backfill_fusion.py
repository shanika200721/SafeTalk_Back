from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import SessionLocal
from app.models.database_models import ModalityPrediction, RiskAssessment, User, UserRole
from app.services.fusion import run_controlled_fusion


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def candidate_students(db, *, student_id: int | None, all_students: bool, from_date: datetime | None, to_date: datetime | None) -> list[int]:
    query = db.query(ModalityPrediction.student_id).distinct()
    if student_id is not None:
        query = query.filter(ModalityPrediction.student_id == student_id)
    elif not all_students:
        raise SystemExit("Use --student-id or --all")
    if from_date is not None:
        query = query.filter(ModalityPrediction.created_at >= from_date.replace(tzinfo=None))
    if to_date is not None:
        query = query.filter(ModalityPrediction.created_at <= to_date.replace(tzinfo=None))
    return [row[0] for row in query.order_by(ModalityPrediction.student_id.asc()).all()]


def latest_trigger_prediction(db, student_id: int, from_date: datetime | None, to_date: datetime | None) -> ModalityPrediction | None:
    query = db.query(ModalityPrediction).filter(ModalityPrediction.student_id == student_id)
    if from_date is not None:
        query = query.filter(ModalityPrediction.created_at >= from_date.replace(tzinfo=None))
    if to_date is not None:
        query = query.filter(ModalityPrediction.created_at <= to_date.replace(tzinfo=None))
    return query.order_by(ModalityPrediction.created_at.desc(), ModalityPrediction.id.desc()).first()


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill controlled late-fusion rows from existing modality predictions.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without creating risk_assessments.")
    parser.add_argument("--student-id", type=int, default=None)
    parser.add_argument("--all", action="store_true", dest="all_students")
    parser.add_argument("--from-date", default=None, help="Inclusive ISO datetime/date filter for prediction.created_at.")
    parser.add_argument("--to-date", default=None, help="Inclusive ISO datetime/date filter for prediction.created_at.")
    args = parser.parse_args()

    from_date = parse_dt(args.from_date)
    to_date = parse_dt(args.to_date)
    report: dict[str, Any] = {
        "dry_run": args.dry_run,
        "students_evaluated": 0,
        "predictions_considered": 0,
        "fusion_rows_created": 0,
        "rows_skipped": 0,
        "students": [],
    }

    db = SessionLocal()
    try:
        before_count = db.query(func.count(RiskAssessment.id)).scalar() or 0
        report["risk_assessments_before"] = int(before_count)
        for sid in candidate_students(db, student_id=args.student_id, all_students=args.all_students, from_date=from_date, to_date=to_date):
            user = db.query(User).filter(User.id == sid, User.role == UserRole.STUDENT).first()
            if not user:
                report["rows_skipped"] += 1
                report["students"].append({"student_id": sid, "skipped": True, "reason": "student_not_found_or_not_student"})
                continue
            trigger = latest_trigger_prediction(db, sid, from_date, to_date)
            if not trigger:
                report["rows_skipped"] += 1
                report["students"].append({"student_id": sid, "skipped": True, "reason": "no_predictions_in_range"})
                continue
            considered = db.query(func.count(ModalityPrediction.id)).filter(ModalityPrediction.student_id == sid).scalar() or 0
            existing = (
                db.query(RiskAssessment)
                .filter(RiskAssessment.trigger_prediction_id == trigger.id)
                .first()
            )
            if existing:
                report["rows_skipped"] += 1
                report["students"].append({
                    "student_id": sid,
                    "skipped": True,
                    "reason": "fusion_for_trigger_already_exists",
                    "trigger_prediction_id": trigger.id,
                    "assessment_id": existing.id,
                    "predictions_considered": int(considered),
                })
                continue
            preview = run_controlled_fusion(
                db,
                user_id=sid,
                persist=False,
                trigger_source="historical_backfill",
                trigger_prediction_id=trigger.id,
                trigger_metadata={"backfill": True, "from_date": args.from_date, "to_date": args.to_date},
            )
            if not preview.get("evidence", {}).get("used_modalities"):
                report["rows_skipped"] += 1
                report["students"].append({
                    "student_id": sid,
                    "skipped": True,
                    "reason": "no_eligible_predictions",
                    "trigger_prediction_id": trigger.id,
                    "predictions_considered": int(considered),
                    "excluded_reasons": [
                        {"modality": item.get("modality"), "reason": item.get("reason")}
                        for item in preview.get("evidence", {}).get("excluded_modalities", [])
                    ],
                })
                continue
            result = preview
            if not args.dry_run:
                result = run_controlled_fusion(
                    db,
                    user_id=sid,
                    persist=True,
                    trigger_source="historical_backfill",
                    trigger_prediction_id=trigger.id,
                    trigger_metadata={"backfill": True, "from_date": args.from_date, "to_date": args.to_date},
                )
            report["students_evaluated"] += 1
            report["predictions_considered"] += int(considered)
            if not args.dry_run and result.get("assessment_id"):
                report["fusion_rows_created"] += 1
            report["students"].append({
                "student_id": sid,
                "trigger_prediction_id": trigger.id,
                "assessment_id": result.get("assessment_id"),
                "status": result.get("status"),
                "risk_level": result.get("risk_level"),
                "score": result.get("score"),
                "used_modalities": result.get("evidence", {}).get("used_modalities", []),
                "excluded_reasons": [
                    {"modality": item.get("modality"), "reason": item.get("reason")}
                    for item in result.get("evidence", {}).get("excluded_modalities", [])
                ],
                "predictions_considered": int(considered),
            })
        after_count = db.query(func.count(RiskAssessment.id)).scalar() or 0
        report["risk_assessments_after"] = int(after_count)
    finally:
        db.close()

    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
