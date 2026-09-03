from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.review.face.package import create_review_package


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Phase 3H face human-review package.")
    parser.add_argument("--cross-label-quarantine", required=True)
    parser.add_argument("--perceptual-candidates", required=True)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-html-index", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = create_review_package(
        cross_label_quarantine_path=args.cross_label_quarantine,
        perceptual_candidates_path=args.perceptual_candidates,
        canonical_manifest_path=args.canonical_manifest,
        source_fingerprint=args.source_fingerprint,
        policy_config_path=args.policy_config,
        output_dir=args.output_dir,
        include_html_index=args.include_html_index,
        validate_only=args.validate_only,
        overwrite=args.overwrite,
    )
    print(
        "face review package:",
        {
            key: value
            for key, value in result.items()
            if key in {"valid", "review_item_count", "cross_label_review_item_count", "perceptual_review_item_count", "review_status"}
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

