"""Create leakage-safe Phase 3A split manifests for eligible modalities only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.splitting.common import load_json
from app.ml.splitting.profile import create_profile_split
from app.ml.splitting.speech import create_speech_split
from app.ml.splitting.text import create_text_split
from app.ml.splitting.validation import validate_deterministic_replay, validate_no_forbidden_modalities


DEFAULTS = {
    "profile": {
        "config": "../ml-research/configs/profile.split.v1.json",
        "input_path": "../generated/preprocessing/profile/v1/canonical_profile.csv",
        "preprocessing_report": "../generated/preprocessing/profile/v1/profile_preprocessing_report.json",
        "feature_schema": "../generated/preprocessing/profile/v1/profile_feature_schema.json",
        "source_fingerprint": "../generated/manifests/fingerprints/student-profile-v1.json",
        "output_dir": "../generated/manifests/splits/profile/v1",
    },
    "text": {
        "config": "../ml-research/configs/text.split.v1.json",
        "input_path": "../generated/preprocessing/text/v1/canonical_text.csv",
        "preprocessing_report": "../generated/preprocessing/text/v1/text_preprocessing_report.json",
        "feature_schema": "../generated/preprocessing/text/v1/text_feature_schema.json",
        "duplicate_manifest": "../generated/preprocessing/text/v1/text_duplicate_manifest.json",
        "conflict_manifest": "../generated/preprocessing/text/v1/text_conflict_quarantine.csv",
        "source_fingerprint": "../generated/manifests/fingerprints/mental-health-text-v1.json",
        "output_dir": "../generated/manifests/splits/text/v1",
    },
    "speech": {
        "config": "../ml-research/configs/speech.split.v1.json",
        "input_path": "../generated/preprocessing/speech/v1/speech_canonical_manifest.csv",
        "preprocessing_report": "../generated/preprocessing/speech/v1/speech_preprocessing_report.json",
        "feature_schema": "../generated/preprocessing/speech/v1/speech_feature_schema.json",
        "duplicate_manifest": "../generated/preprocessing/speech/v1/speech_duplicate_manifest.json",
        "output_dir": "../generated/manifests/splits/speech/v1",
    },
}


def _resolve(value: str | None, default: str | None = None) -> Path | None:
    raw = value or default
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create Phase 3A split manifests.")
    parser.add_argument("--modality", required=True)
    parser.add_argument("--config")
    parser.add_argument("--input-path")
    parser.add_argument("--preprocessing-report")
    parser.add_argument("--feature-schema")
    parser.add_argument("--source-fingerprint")
    parser.add_argument("--duplicate-manifest")
    parser.add_argument("--conflict-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-path")
    return parser


def _verify_source_fingerprint(source_fingerprint: Path | None, preprocessing_report: Path, *, strict: bool) -> None:
    if source_fingerprint is None or not source_fingerprint.exists():
        if strict:
            raise ValueError("source fingerprint path is required and must exist in strict mode")
        return
    fingerprint = load_json(source_fingerprint)
    report = load_json(preprocessing_report)
    expected = report.get("source_fingerprint")
    if expected and fingerprint.get("combined_sha256") != expected:
        raise ValueError("source fingerprint verification failed")


def _verify_expected_preprocessing_hash(config_path: Path, input_path: Path) -> None:
    config = load_json(config_path)
    expected = config.get("expected_preprocessing_artifact_hash")
    if expected and sha256_file(input_path, allow_outside_project=True) != expected:
        raise ValueError("preprocessing hash verification failed")


def _generate(args) -> dict:
    modality = args.modality.lower()
    defaults = DEFAULTS[modality]
    config = _resolve(args.config, defaults.get("config"))
    input_path = _resolve(args.input_path, defaults.get("input_path"))
    preprocessing_report = _resolve(args.preprocessing_report, defaults.get("preprocessing_report"))
    feature_schema = _resolve(args.feature_schema, defaults.get("feature_schema"))
    source_fingerprint = _resolve(args.source_fingerprint, defaults.get("source_fingerprint"))
    output_dir = _resolve(args.output_dir, defaults.get("output_dir"))
    duplicate_manifest = _resolve(args.duplicate_manifest, defaults.get("duplicate_manifest"))
    conflict_manifest = _resolve(args.conflict_manifest, defaults.get("conflict_manifest"))

    _verify_source_fingerprint(source_fingerprint, preprocessing_report, strict=args.strict)
    _verify_expected_preprocessing_hash(config, input_path)

    common = {
        "input_path": input_path,
        "config_path": config,
        "preprocessing_report_path": preprocessing_report,
        "feature_schema_path": feature_schema,
        "output_dir": output_dir,
        "seed": args.seed,
        "overwrite": args.overwrite,
        "validate_only": args.validate_only,
    }
    if modality == "profile":
        return create_profile_split(**common)
    if modality == "text":
        return create_text_split(
            **common,
            duplicate_manifest_path=duplicate_manifest,
            conflict_manifest_path=conflict_manifest,
            source_overlap_report_path=_resolve(None, "../generated/preprocessing/text/v1/text_source_overlap_report.json"),
        )
    if modality == "speech":
        return create_speech_split(**common, duplicate_manifest_path=duplicate_manifest)
    raise ValueError(f"Unsupported modality: {modality}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_no_forbidden_modalities(args.modality)
        result = _generate(args)
        if args.replay:
            replay_args = argparse.Namespace(**vars(args))
            replay_args.validate_only = True
            replay = _generate(replay_args)
            validate_deterministic_replay(result["manifest"], replay["manifest"])
        if args.report_path:
            report_path = _resolve(args.report_path)
            if report_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result["report"].to_safe_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = result["manifest"].validation_summary
        print(
            f"{args.modality.lower()} split: train={summary.train_count} "
            f"validation={summary.validation_count} test={summary.test_count} "
            f"excluded={len(result['exclusions'])} replay={'passed' if args.replay else 'not_requested'} "
            f"manifest_hash={result['manifest_hash']}"
        )
        return 0
    except Exception as exc:
        print(f"Phase 3A split generation failed: {exc}", file=sys.stderr)
        return 1 if args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
