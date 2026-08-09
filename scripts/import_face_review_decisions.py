from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.review.face.decisions import load_reviewer_decisions, save_validated_decisions


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Phase 3H face reviewer decisions.")
    parser.add_argument("--review-package", required=True)
    parser.add_argument("--decision-file", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reviewer-alias")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    decisions = load_reviewer_decisions(
        review_package=args.review_package,
        decision_file=args.decision_file,
        policy_config_path=args.policy_config,
        reviewer_alias=args.reviewer_alias,
    )
    if not args.validate_only:
        save_validated_decisions(
            decisions=decisions,
            output_dir=args.output_dir,
            decision_file=args.decision_file,
            overwrite=args.overwrite,
        )
    print(f"validated reviewer decisions: {len(decisions)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

