from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.training.config import load_training_config
from app.ml.training.data import build_training_dataset_bundle, load_split_manifest
from app.ml.training.runner import (
    create_synthetic_smoke_fixture,
    run_training_pipeline,
    synthetic_estimator_factory,
)
from app.ml.training.schemas import TrainingTask


REAL_MODALITIES = {"profile", "text", "speech"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Phase 3B training framework without real model training.")
    parser.add_argument("--config")
    parser.add_argument("--split-manifest")
    parser.add_argument("--canonical-data")
    parser.add_argument("--feature-schema")
    parser.add_argument("--output-dir", default="../generated/training-framework-smoke")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-multiclass", action="store_true")
    parser.add_argument("--register-candidate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-path")
    return parser.parse_args()


def _safe_print(message: str) -> None:
    print(message)


def _dry_run(args: argparse.Namespace) -> int:
    if args.split_manifest:
        manifest = load_split_manifest(args.split_manifest)
        _safe_print(
            f"split manifest ok: modality={manifest.modality}, "
            f"train={len(manifest.train_ids)}, validation={len(manifest.validation_ids)}, test={len(manifest.test_ids)}"
        )
    if args.config:
        config = load_training_config(args.config)
        _safe_print(f"config ok: model={config.model_name}, modality={config.modality}, task={config.task.value}")
        if config.modality in REAL_MODALITIES and not args.synthetic_smoke:
            _safe_print("real Profile/Text/Speech training is refused in Phase 3B; validation only")
        if args.canonical_data:
            build_training_dataset_bundle(
                config=config,
                split_manifest_path=args.split_manifest or config.split_manifest_path,
                canonical_data_path=args.canonical_data,
            )
            _safe_print("canonical data coverage/leakage checks ok")
    if args.feature_schema:
        path = Path(args.feature_schema)
        if not path.exists():
            _safe_print("feature schema missing")
            return 2
        _safe_print("feature schema path exists")
    return 0


def _synthetic_smoke(args: argparse.Namespace) -> int:
    if args.register_candidate and not os.getenv("SAFETALK_TEST_DATABASE_URL"):
        _safe_print("--register-candidate requires SAFETALK_TEST_DATABASE_URL for a temporary/test database")
        return 2
    output_root = Path(args.output_dir)
    task = TrainingTask.MULTICLASS_CLASSIFICATION if args.synthetic_multiclass else TrainingTask.BINARY_CLASSIFICATION
    fixture = create_synthetic_smoke_fixture(output_root, task=task, overwrite=args.overwrite)
    result = run_training_pipeline(
        config=fixture["config"],
        canonical_data_path=fixture["canonical_data"],
        estimator_factory=synthetic_estimator_factory,
        register_candidate=False,
        overwrite=args.overwrite,
    )
    if result.status.value != "completed":
        _safe_print(f"synthetic smoke failed: {result.failure_reason}")
        return 1
    _safe_print(f"synthetic smoke completed: run_id={result.run_id}")
    _safe_print(f"model_artifact={result.model_artifact_path}")
    _safe_print(f"metrics={result.metrics_path}")
    _safe_print(f"model_card={result.model_card_path}")
    _safe_print(f"artifact_manifest={result.artifact_manifest_path}")
    _safe_print("registered=false active=false")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.synthetic_smoke:
            return _synthetic_smoke(args)
        return _dry_run(args)
    except Exception as exc:
        _safe_print(f"validation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
