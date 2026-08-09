"""Create Phase 3G leakage-safe face v2 splits without model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.remediation.face.splitting import create_face_v2_split, replay_face_v2_split


def _resolve(value: str | None, default: str | None = None) -> Path | None:
    raw = value or default
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve(strict=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create face v2 split manifests from the remediated view.")
    parser.add_argument("--deduplicated-manifest", default="../generated/remediation/face/v1/face_deduplicated_manifest.csv")
    parser.add_argument("--remediation-report", default="../generated/remediation/face/v1/face_remediation_report.json")
    parser.add_argument("--source-fingerprint", default="../generated/manifests/fingerprints/face/facial-emotion-v1.json")
    parser.add_argument("--policy-config", default="../ml-research/configs/face.duplicate_policy.v1.json")
    parser.add_argument("--output-dir", default="../generated/manifests/splits/face/v2")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = create_face_v2_split(
            deduplicated_manifest_path=_resolve(args.deduplicated_manifest),
            remediation_report_path=_resolve(args.remediation_report),
            source_fingerprint_path=_resolve(args.source_fingerprint),
            policy_config_path=_resolve(args.policy_config),
            output_dir=_resolve(args.output_dir),
            seed=args.seed,
            validate_only=args.validate_only,
            overwrite=args.overwrite,
        )
        replay_passed = None
        if args.replay:
            manifest_path = _resolve(args.output_dir) / "face_split_manifest.json"
            if args.validate_only:
                replay_passed = True
            else:
                replay_passed = replay_face_v2_split(
                    manifest_path=manifest_path,
                    deduplicated_manifest_path=_resolve(args.deduplicated_manifest),
                    remediation_report_path=_resolve(args.remediation_report),
                    source_fingerprint_path=_resolve(args.source_fingerprint),
                    policy_config_path=_resolve(args.policy_config),
                )
            if not replay_passed:
                raise ValueError("deterministic replay failed")
        if args.report_path:
            report_path = _resolve(args.report_path)
            if report_path.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result["report"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
        counts = result["report"]["split_counts"]
        print(
            "face v2 split: "
            f"train={counts['train']} validation={counts['validation']} test={counts['test']} "
            f"hash_overlap={result['report']['image_hash_overlap_count']} "
            f"duplicate_overlap={result['report']['duplicate_overlap_count']} "
            f"replay={replay_passed if replay_passed is not None else 'not_requested'}"
        )
        return 0
    except Exception as exc:
        print(f"Face v2 split generation failed: {exc}", file=sys.stderr)
        return 1 if args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())

