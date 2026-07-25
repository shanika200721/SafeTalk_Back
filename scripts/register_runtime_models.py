import argparse
import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database import SessionLocal
from app.models.database_models import ModelRegistry
from app.services.model_registry import (
    APPROVED_RUNTIME_ARTIFACTS,
    activate_model_version,
    apply_verification_result,
    register_or_update_candidate,
    verify_model_artifact,
    _resolve_repo_path,
)


def _model_summary(model: ModelRegistry) -> dict:
    return {
        "id": model.id,
        "model_name": model.model_name,
        "modality": model.modality,
        "version": model.version,
        "artifact_path": model.artifact_path,
        "status": model.status,
        "verification_status": model.verification_status,
        "verification_failure_code": model.verification_failure_code,
        "is_active": model.is_active,
    }


def _candidate_from_artifact(path: Path) -> ModelRegistry:
    return ModelRegistry(
        model_name=path.parents[2].name,
        modality=path.parents[3].name,
        version=path.parents[1].name,
        framework="sklearn",
        artifact_path=str(path),
        serializer="joblib",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Register and verify approved Phase 4D runtime model artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect approved artifacts without database writes.")
    parser.add_argument("--register", action="store_true", help="Create or update registry rows for approved artifacts.")
    parser.add_argument("--verify", action="store_true", help="Run governance verification for registered artifacts.")
    parser.add_argument(
        "--activate",
        nargs="*",
        choices=["profile", "text"],
        default=[],
        help="Explicitly activate verified models for the listed modalities.",
    )
    args = parser.parse_args()

    summaries = []
    if args.dry_run or not args.register:
        for relative_path in APPROVED_RUNTIME_ARTIFACTS:
            artifact_path = _resolve_repo_path(relative_path)
            candidate = _candidate_from_artifact(artifact_path)
            result = verify_model_artifact(candidate) if args.verify or args.dry_run else None
            summaries.append(
                {
                    "artifact_path": relative_path,
                    "exists": artifact_path.exists(),
                    "verification": result.to_dict() if result else None,
                    "database_write": False,
                }
            )
        print(json.dumps({"dry_run": True, "candidates": summaries}, indent=2))
        return 0

    db = SessionLocal()
    try:
        for relative_path in APPROVED_RUNTIME_ARTIFACTS:
            artifact_path = _resolve_repo_path(relative_path)
            if not artifact_path.exists():
                summaries.append({"artifact_path": relative_path, "exists": False, "database_write": False})
                continue

            model = register_or_update_candidate(db, artifact_path)
            verification = None
            if args.verify:
                result = verify_model_artifact(model)
                apply_verification_result(db, model, result)
                verification = result.to_dict()

            summaries.append(
                {
                    "artifact_path": relative_path,
                    "exists": True,
                    "database_write": True,
                    "model": _model_summary(model),
                    "verification": verification,
                }
            )

        activated = []
        for modality in args.activate:
            candidates = [
                item["model"]
                for item in summaries
                if item.get("model", {}).get("modality") == modality
                and item.get("model", {}).get("verification_status") == "passed"
            ]
            if not candidates:
                continue
            selected = candidates[0]
            model = activate_model_version(
                db,
                model_name=selected["model_name"],
                modality=selected["modality"],
                version=selected["version"],
            )
            activated.append(_model_summary(model))

        db.commit()
        print(json.dumps({"dry_run": False, "candidates": summaries, "activated": activated}, indent=2))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
