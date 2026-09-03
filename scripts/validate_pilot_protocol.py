"""CLI for Phase 4A pilot protocol validation and synthetic smoke generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.common import paths
from app.ml.pilot.export import (
    export_pilot_alignment_report,
    export_pilot_consent_summary,
    export_pilot_missingness_report,
    export_pilot_modality_manifest,
    export_pilot_outcomes,
    export_pilot_participants,
    export_pilot_safety_summary,
    export_pilot_sessions,
)
from app.ml.pilot.privacy import validate_privacy
from app.ml.pilot.reporting import build_readiness_report
from app.ml.pilot.retention import validate_retention_policy
from app.ml.pilot.safety import safety_summary
from app.ml.pilot.sessions import align_modality_records
from app.ml.pilot.synthetic import generate_synthetic_pilot_dataset
from app.ml.pilot.validation import validate_pilot_dataset, validate_protocol_configs


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _default_config(name: str) -> Path:
    return paths.get_ml_research_root() / "configs" / name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 4A pilot protocol artifacts without real data collection.")
    parser.add_argument("--protocol-config", default=str(_default_config("pilot.protocol.v1.json")))
    parser.add_argument("--modality-scope", default=str(_default_config("pilot.modality_scope.v1.json")))
    parser.add_argument("--alignment-policy", default=str(_default_config("pilot.alignment_policy.v1.json")))
    parser.add_argument("--retention-policy", default=str(_default_config("pilot.retention_policy.v1.json")))
    parser.add_argument("--output-dir", default="generated/pilot-protocol-smoke/v1")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--participants", type=int, default=12)
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report-path", default="generated/reports/pilot_readiness/v1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol_config = _load_json(args.protocol_config)
    modality_scope = _load_json(args.modality_scope)
    alignment_policy = _load_json(args.alignment_policy)
    retention_policy = _load_json(args.retention_policy)
    config_validation = validate_protocol_configs(protocol_config, modality_scope, alignment_policy, retention_policy, strict=args.strict)
    if args.schema_only and not args.synthetic_smoke:
        out = paths.get_repository_root() / args.report_path
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing pilot readiness directory: {out}")
        out.mkdir(parents=True, exist_ok=True)
        (out / "pilot_protocol_validation.json").write_text(json.dumps(config_validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Phase 4A pilot schema validation: valid={config_validation['valid']} output_dir={out}")
        return 0 if config_validation["valid"] else 1

    if not args.synthetic_smoke:
        print(f"Phase 4A pilot config validation: valid={config_validation['valid']}")
        return 0 if config_validation["valid"] else 1

    output_dir = paths.get_repository_root() / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing synthetic smoke directory: {output_dir}")
    dataset = generate_synthetic_pilot_dataset(participants=args.participants, weeks=args.weeks, seed=args.seed)
    validation = validate_pilot_dataset(
        dataset.participants,
        dataset.consents,
        dataset.sessions,
        dataset.modality_records,
        dataset.outcomes,
        dataset.safety_events,
        dataset.withdrawals,
        modality_scope,
        alignment_policy,
        retention_policy,
        strict=args.strict,
    )
    alignment = align_modality_records(dataset.modality_records, dataset.sessions, alignment_policy.get("timestamp_tolerance_minutes", 120))
    privacy = validate_privacy(dataset.modality_records, [participant.pilot_participant_id for participant in dataset.participants])
    export_pilot_participants(dataset.participants, output_dir, overwrite=args.overwrite)
    export_pilot_sessions(dataset.sessions, output_dir, overwrite=args.overwrite)
    export_pilot_modality_manifest(dataset.modality_records, output_dir, overwrite=args.overwrite)
    export_pilot_outcomes(dataset.outcomes, output_dir, overwrite=args.overwrite)
    export_pilot_consent_summary(dataset.consents, output_dir, overwrite=args.overwrite)
    export_pilot_missingness_report(dataset.manifest.missingness_summary, output_dir, overwrite=args.overwrite)
    export_pilot_alignment_report(alignment, output_dir, overwrite=args.overwrite)
    export_pilot_safety_summary(safety_summary(dataset.safety_events), output_dir, overwrite=args.overwrite)
    (output_dir / "pilot_dataset_manifest.json").write_text(json.dumps(dataset.manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "synthetic_metadata.json").write_text(json.dumps(dataset.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readiness = build_readiness_report(args.report_path, dataset, validation, modality_scope, alignment, retention_policy, privacy, overwrite=args.overwrite)
    retention = validate_retention_policy(retention_policy)
    print(
        "Phase 4A pilot synthetic smoke: "
        f"valid={validation['valid']} participants={len(dataset.participants)} weeks={args.weeks} "
        f"sessions={len(dataset.sessions)} modality_records={len(dataset.modality_records)} "
        f"withdrawals={len(dataset.withdrawals)} safety_events={len(dataset.safety_events)} "
        f"alignment_valid={alignment['valid']} privacy_valid={privacy['valid']} "
        f"retention_valid={retention['valid']} readiness_dir={readiness['output_dir']}"
    )
    return 0 if validation["valid"] and config_validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
