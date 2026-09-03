"""CLI for Phase 3J read-only model governance validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.governance.reporting import run_governance_validation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Phase 3J model governance artifacts.")
    parser.add_argument("--model-root", default=None)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--validate-model-cards", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_governance_validation(
        model_root=args.model_root,
        reports_root=args.reports_root,
        config=args.config,
        output_dir=args.output_dir,
        modalities=args.modalities,
        verify_hashes=args.verify_hashes,
        validate_cards=args.validate_model_cards,
        inventory_only=args.inventory_only,
        strict=args.strict,
        fail_on_warning=args.fail_on_warning,
        overwrite=args.overwrite,
        summary_only=args.summary_only,
    )
    print(
        "Phase 3J governance validation: "
        f"status={result.get('final_research_status', 'inventory_only')} "
        f"selected_models={result.get('selected_model_count', result.get('inventory_count', 0))} "
        f"active_models={result.get('active_model_count', 0)} "
        f"registered_models={result.get('registered_model_count', 0)} "
        f"output_dir={result.get('output_dir')}"
    )
    if args.strict:
        print("Strict mode failed as expected because deployment blockers remain.")
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
