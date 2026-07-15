"""Validate and preprocess facial emotion images without training models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.common import paths
from app.ml.common.fingerprinting import dataset_config_hash, load_dataset_fingerprint
from app.ml.common.serialization import load_dataset_config
from app.ml.preprocessing.face.mapping import load_face_label_mapping, load_face_source_structure
from app.ml.preprocessing.face.preprocessor import preprocess_face_dataset
from app.ml.preprocessing.face.reporting import save_face_report_markdown


def _resolve_cli_path(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve(strict=False)
    return candidate


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _fast_verify_fingerprint(fingerprint, dataset_config) -> bool:
    if fingerprint.config_hash and fingerprint.config_hash != dataset_config_hash(dataset_config):
        return False
    source_path = dataset_config.validate_source_exists()
    report_root = paths.get_repository_root() if paths.is_path_inside(paths.get_repository_root(), source_path) else source_path
    files = sorted(path for path in source_path.rglob("*") if path.is_file()) if source_path.is_dir() else [source_path]
    current = {}
    for path in files:
        try:
            relative = path.resolve(strict=False).relative_to(report_root).as_posix()
        except ValueError:
            relative = path.name
        current[relative] = path.stat().st_size
    saved = {item.relative_path: item.size_bytes for item in fingerprint.files}
    return (
        fingerprint.file_count == len(files)
        and fingerprint.total_bytes == sum(current.values())
        and saved == current
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess facial emotion dataset into canonical research outputs.")
    parser.add_argument("--dataset-config", required=True, help="Face dataset overview JSON.")
    parser.add_argument("--preprocessing-config", required=True, help="Face preprocessing policy JSON.")
    parser.add_argument("--label-mapping-config", required=True)
    parser.add_argument("--source-structure-config", required=True)
    parser.add_argument("--fingerprint", required=True, help="Dataset fingerprint JSON report.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--source-split", choices=["train", "test"])
    parser.add_argument("--compute-image-statistics", action="store_true")
    parser.add_argument("--write-normalized-images", action="store_true")
    parser.add_argument("--target-width", type=int, default=48)
    parser.add_argument("--target-height", type=int, default=48)
    parser.add_argument("--color-mode", default="L", choices=["L", "RGB", "grayscale", "gray"])
    parser.add_argument("--near-duplicate-limit", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--report-path", help="Optional Markdown report path. In validate-only mode, this is the only write.")
    return parser


def _print_summary(result: dict) -> None:
    print(f"validation: {'passed' if result['valid'] else 'failed'}")
    print(f"validate only: {result['validate_only']}")
    print(f"image statistics: {result['compute_image_statistics']}")
    print(f"normalized images written: {result['write_normalized_images']}")
    print(f"source files: {result['source_files']}")
    print(f"output records: {result['output_records']}")
    print(f"excluded records: {result['excluded_records']}")
    print(f"readable files: {result['readable_files']}")
    print(f"unreadable files: {result['unreadable_files']}")
    print(f"split distribution: {result['split_distribution']}")
    print(f"label distribution: {result['label_distribution']}")
    print(f"formats: {result['format_distribution']}")
    print(f"color modes: {result['color_mode_distribution']}")
    print(f"width summary: {result['width_summary']}")
    print(f"height summary: {result['height_summary']}")
    print(f"duplicates: {result['duplicate_summary']}")
    if result["outputs"]:
        for label, path in result["outputs"].items():
            print(f"{label}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dataset_config = load_dataset_config(_resolve_cli_path(args.dataset_config))
        preprocessing_payload = _load_json(_resolve_cli_path(args.preprocessing_config))
        if preprocessing_payload.get("preprocessing_version") not in {None, "1.0.0"}:
            raise ValueError("Unsupported face preprocessing version")
        label_mapping = load_face_label_mapping(_resolve_cli_path(args.label_mapping_config))
        source_structure = load_face_source_structure(_resolve_cli_path(args.source_structure_config))
        fingerprint = load_dataset_fingerprint(_resolve_cli_path(args.fingerprint))
        if not _fast_verify_fingerprint(fingerprint, dataset_config):
            raise ValueError("Face source fingerprint mismatch")
        result = preprocess_face_dataset(
            source_structure,
            label_mapping,
            source_fingerprint=fingerprint,
            output_dir=_resolve_cli_path(args.output_dir),
            overwrite=args.overwrite,
            validate_only=args.validate_only,
            max_files=args.max_files,
            source_split=args.source_split,
            compute_image_statistics=args.compute_image_statistics,
            write_normalized_images=args.write_normalized_images,
            target_width=args.target_width,
            target_height=args.target_height,
            color_mode=args.color_mode,
            near_duplicate_limit=args.near_duplicate_limit,
        )
        if args.report_path:
            save_face_report_markdown(result["report"], _resolve_cli_path(args.report_path), overwrite=args.overwrite)
        _print_summary(result)
        return 1 if result["critical_conflicts"] else 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
